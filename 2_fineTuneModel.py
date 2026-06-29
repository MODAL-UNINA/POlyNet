#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune a pretrained DeepNMR model using pseudo-synthetic spectra.

This script intentionally preserves the original fine-tuning procedure:
  - selection of experimental monomaterial prototypes;
  - pseudo-synthetic single-spectrum augmentation;
  - pseudo-synthetic mixture generation;
  - optional masking augmentation;
  - chain-end artifact augmentation;
  - ppm shift augmentation;
  - area normalization;
  - fine-tuning from pretrained weights.

The generated fine-tuning arrays are saved under:
  ft_sets/<model_name>/

The fine-tuned weights, options, and history are saved under:
  models/<model_name>/
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import importlib

import numpy as np
import pandas as pd
from tqdm import tqdm
from fastcore.all import dict2obj, obj2dict
from scipy.special import voigt_profile
from scipy.integrate import simpson as simps


# %% SUFFIX UTILITIES


def normalize_suffix(suffix: str | None) -> str:
    """Normalize suffix so that non-empty suffixes start with '_'."""
    if suffix is None:
        return ""

    suffix = str(suffix).strip().strip("_")

    if suffix == "":
        return ""

    return f"_{suffix}"


# %% ARGUMENTS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a DeepNMR model with pseudo-synthetic spectra.",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3,4,5",
        help='Comma-separated list of GPUs to use, e.g. "0,1".',
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=25000,
        help="Number of fine-tuning examples.",
    )
    parser.add_argument(
        "--n_max_components",
        type=int,
        default=2,
        help="Maximum number of components for pseudo-synthetic mixtures.",
    )
    parser.add_argument(
        "--masking",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to apply random masking to spectra: 1=True, 0=False.",
    )
    parser.add_argument(
        "--proportion_weights",
        type=str,
        default="0.4, 0.6",
        help=(
            "Sampling proportions for single and mixture examples. "
            "The first value is the single-spectrum fraction; the remaining "
            "values correspond to 2-, 3-, ..., n_max_components-component mixtures."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Per-GPU batch size used for fine-tuning.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2500,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model_weights_kl_mse_loss_composition_neg2_mse_hybrid",
        help="Name of the pretrained model folder under models/ and val_sets/.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        required=True,
        help=(
            "Suffix used for fine-tuned weights, options, history, and FT sets. "
            "Pass it without leading underscore, e.g. 'ft'. "
            "Input with leading underscore is also accepted and normalized."
        ),
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="DATASET/test_data.pkl",
        help="Experimental dataset used to generate pseudo-synthetic FT samples.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help=(
            "Optional random seed. Default is None, preserving the stochastic "
            "behavior of the original script."
        ),
    )

    args, _ = parser.parse_known_args()
    return args


# %% MAIN


def main() -> None:
    """Generate pseudo-synthetic FT data and fine-tune the pretrained model."""
    args = parse_args()

    CVD = args.gpus
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = CVD

    import tensorflow as tf
    import utils

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        tf.random.set_seed(args.random_seed)

    # %% OVERALL PARAMETERS

    opts = dict2obj(
        dict(
            labels=["LDPE", "PE", "PP", "EH", "EO", "EB", "RACO", "EPR"],
            n_ft_samples=args.dataset_size,
            proportions=[float(k.strip()) for k in args.proportion_weights.split(",")],
            epochs=args.epochs,
            batch_size=args.batch_size * len(CVD.split(",")),
            validation_split=0.3,
            learning_rate=5e-6,
            reg_l2=1e-9,
            dropout_rate=0.1,
            run_name=args.model_name,
            masking=True if args.masking == 1 else False,
        )
    )

    suffix = normalize_suffix(args.suffix)
    print(f"[INFO] Fine-tuning suffix: {suffix}")

    if suffix == "":
        raise ValueError(
            "--suffix cannot be empty for fine-tuning, to avoid overwriting base model artifacts."
        )

    if args.n_max_components < 2:
        raise ValueError("--n_max_components must be at least 2.")

    if len(opts.proportions) != args.n_max_components:
        raise ValueError(
            f"Number of proportions ({len(opts.proportions)}) must match "
            f"the number of mixture options ({args.n_max_components})."
        )

    if not np.isclose(sum(opts.proportions), 1.0):
        raise ValueError(f"Sum of proportions ({sum(opts.proportions)}) must be 1.0.")

    single_percent = opts.proportions[0]
    proportions = opts.proportions[1:] if len(opts.proportions) > 1 else None

    # %% TEST DATA IMPORT

    print("[INFO] Loading experimental data for fine-tuning generation.")
    test_data = pd.read_pickle(args.test_dataset)

    # Store spectral columns before adding the fine-tuning flag.
    # This preserves the original logic of using columns 4:-1 after adding the flag.
    spectral_columns = test_data.columns[4:]

    # %% GENERATION OF FINE-TUNING DATASET

    test_data["fine-tuning"] = False

    single_copolymers = test_data[
        (test_data["copo_tuple"].apply(lambda x: len(x) == 1))
        & (test_data["copolymer"] != "TEST")
    ]

    def add_chain_end_artifacts(
        spectrum,
        ppm_domain,
        intensity_range=(1e-5, 5e-4),
        probability_per_peak=0.5,
        sigma_range=(0.8, 1.2),
        gamma_range=(0.8, 1.2),
    ):
        """Add weak chain-end-like Voigt features to an input spectrum."""
        peaks_ppm = [13.995, 22.69, 29.15, 29.25, 29.35, 29.80, 30.8, 32.01, 33.8]
        positions = [np.argmin(np.abs(ppm_domain - p)) for p in peaks_ppm]
        domain = np.arange(len(spectrum))

        out = spectrum.copy()
        artifact = np.zeros_like(out)

        base_intensity = np.random.uniform(*intensity_range)

        for pos in positions:
            if np.random.rand() < probability_per_peak:
                artifact += (
                    np.random.uniform(0.8, 1.2)
                    * base_intensity
                    * voigt_profile(
                        domain - pos,
                        np.random.uniform(*sigma_range),
                        np.random.uniform(*gamma_range),
                    )
                )

        out += artifact
        return out

    def stratified_sample_by_composition(
        class_df,
        n_samples,
        n_bins=5,
        random_state=42,
    ):
        """Select monomaterial prototypes by stratifying samples over composition."""
        df = class_df.copy()

        df["comp_scalar"] = df["c"].apply(
            lambda x: (
                float(x[0])
                if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0
                else np.nan
            )
        )
        df = df.dropna(subset=["comp_scalar"])

        if len(df) == 0:
            return []

        if len(df) <= n_samples:
            return df.sample(
                n=n_samples,
                replace=True,
                random_state=random_state,
            ).index.to_list()

        try:
            df["comp_bin"] = pd.qcut(df["comp_scalar"], q=n_bins, duplicates="drop")
        except ValueError:
            df["comp_bin"] = pd.cut(df["comp_scalar"], bins=n_bins)

        bin_counts = df["comp_bin"].value_counts()
        df["weight"] = df["comp_bin"].apply(lambda b: 1.0 / bin_counts[b])

        sampled = df.sample(
            n=n_samples,
            replace=False,
            weights="weight",
            random_state=random_state,
        )
        return sampled.index.to_list()

    def safe_area_normalize(spectrum, eps=1e-12, clip_min=1e-16):
        """Clip and area-normalize a spectrum."""
        spectrum = np.asarray(spectrum, dtype=float)
        spectrum = np.maximum(spectrum, clip_min)

        area = simps(spectrum)
        if not np.isfinite(area) or np.abs(area) < eps:
            raise ValueError(f"Invalid spectrum area during normalization: {area}")

        return spectrum / area

    # %% PROTOTYPE POOL GENERATION

    n_random_samples_dict = {
        "LDPE": 5,
        "PE": 14,
        "PP": 25,
        "EH": 30,
        "EO": 20,
        "EB": 20,
        "RACO": 25,
        "EPR": 20,
    }
    copolymers_with_composition = {"EH", "EO", "EB", "RACO", "EPR"}

    single_pool_indices = []
    index_weights = []

    unique_classes = single_copolymers["copolymer"].unique()
    num_classes = len(unique_classes)

    for copolymer in unique_classes:
        class_df = single_copolymers[single_copolymers["copolymer"] == copolymer]
        n_req = n_random_samples_dict.get(copolymer, 20)

        if copolymer in copolymers_with_composition:
            selected_idx = stratified_sample_by_composition(class_df, n_req)
        else:
            replace_flag = len(class_df) < n_req
            selected_idx = class_df.sample(
                n=n_req,
                replace=replace_flag,
                random_state=42,
            ).index.to_list()

        single_pool_indices.extend(selected_idx)

        class_weight = 1.0 / (len(selected_idx) * num_classes)
        index_weights.extend([class_weight] * len(selected_idx))

    test_data.loc[single_pool_indices, "fine-tuning"] = True
    fine_tune_data = test_data[test_data["fine-tuning"] == True]

    # %% SINGLE-SAMPLE INDEX GENERATION

    num_single_samples = int(opts.n_ft_samples * single_percent)

    index_weights = np.array(index_weights)
    index_weights /= index_weights.sum()

    copolymers_list = np.random.choice(
        single_pool_indices,
        size=num_single_samples,
        p=index_weights,
        replace=True,
    )

    X_ft_single = np.empty((num_single_samples, len(spectral_columns)))
    y_ft_single_w = np.zeros((num_single_samples, len(opts.labels)))
    y_ft_single_c = np.zeros_like(y_ft_single_w)

    print(f"\n[INFO] Number of final single FT samples: {num_single_samples}")

    # %% SINGLE-SAMPLE AUGMENTATION

    ppm_domain_general = np.linspace(55, 5, len(spectral_columns))
    eta = np.abs(ppm_domain_general[1] - ppm_domain_general[0])
    max_shift_samples = int(round(0.015 / eta))

    single_samples_data = {}

    for idx in tqdm(single_pool_indices, desc="Caching single samples"):
        single_samples_data[idx] = {
            "copolymer": fine_tune_data.loc[idx, "copolymer"],
            "composition": fine_tune_data.loc[idx, "c"][0],
            "spectrum": fine_tune_data.loc[idx, spectral_columns].values.astype(float),
        }

    for i, idx in tqdm(
        enumerate(copolymers_list),
        total=num_single_samples,
        desc="Computing single copolymers",
    ):
        copolymer = single_samples_data[idx]["copolymer"]
        composition = single_samples_data[idx]["composition"]
        spectrum = single_samples_data[idx]["spectrum"].copy()

        assert np.isclose(
            simps(spectrum), 1
        ), f"Spectrum for index {idx} not area-normalized before augmentation."

        if np.random.rand() < 0.30:
            empirical_std = np.std(np.concatenate([spectrum[:1500], spectrum[-1500:]]))
            std_noise = empirical_std * np.random.uniform(0.8, 1.2)
            spectrum += np.random.normal(loc=0.0, scale=std_noise, size=spectrum.shape)

        if np.random.rand() < 0.1 and getattr(opts, "masking", False):
            win_to_erase_ppm = np.random.uniform(0.5, 1.5)
            win_to_erase = int(win_to_erase_ppm / eta)
            pos_to_erase = np.random.randint(0, len(spectrum) - win_to_erase)

            spectrum[pos_to_erase : pos_to_erase + win_to_erase] = np.random.normal(
                np.mean(spectrum),
                np.max(spectrum),
                win_to_erase,
            )

            mask_region = spectrum[pos_to_erase : pos_to_erase + win_to_erase]
            mask_region[mask_region < np.median(spectrum)] = 1e-6
            spectrum[pos_to_erase : pos_to_erase + win_to_erase] = mask_region

        if copolymer in ["EH", "EO", "EB"] and np.random.rand() < 0.2:
            spectrum = add_chain_end_artifacts(
                spectrum,
                ppm_domain_general,
                intensity_range=(1e-5, 5e-4),
                probability_per_peak=0.5,
            )

        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        spectrum = np.roll(spectrum, shift)
        spectrum = safe_area_normalize(spectrum, clip_min=1e-16)

        assert np.isclose(
            simps(spectrum), 1
        ), f"Spectrum for index {idx} not area-normalized after augmentation."

        X_ft_single[i] = spectrum
        y_ft_single_w[i, opts.labels.index(copolymer)] = 1.0
        y_ft_single_c[i, opts.labels.index(copolymer)] = composition

    # %% MULTICOMPONENT MIXTURE GENERATION

    num_mixed_samples = opts.n_ft_samples - num_single_samples

    X_ft_mix = np.empty((num_mixed_samples, len(spectral_columns)))
    y_ft_mix_w = np.zeros((num_mixed_samples, len(opts.labels)))
    y_ft_mix_c = np.zeros_like(y_ft_mix_w)

    available_spectra = {}

    for label in opts.labels:
        class_data = fine_tune_data[fine_tune_data["copolymer"] == label]
        available_spectra[label] = {
            "spectra": class_data.loc[:, spectral_columns].values.astype(float),
            "compositions": class_data["c"].apply(lambda x: x[0]).values.astype(float),
        }

    possible_n_values = np.arange(2, args.n_max_components + 1)

    for i in tqdm(
        range(num_mixed_samples),
        total=num_mixed_samples,
        desc="Computing and augmenting n-component mixtures",
    ):
        if proportions is not None:
            n_components = np.random.choice(possible_n_values, p=proportions)
        else:
            n_components = np.random.randint(2, args.n_max_components + 1)

        valid_weights = False

        while not valid_weights:
            chosen_labels = np.random.choice(opts.labels, n_components, replace=False)

            raw_weights = np.random.dirichlet(np.ones(n_components), size=1)[0]
            weights = np.round(raw_weights, 2)
            weights[-1] = np.round(1.0 - np.sum(weights[:-1]), 2)

            if np.all(weights >= 0.05):
                valid_weights = True

        mixed_spectrum = np.zeros_like(X_ft_mix[0], dtype=float)

        for label, weight in zip(chosen_labels, weights):
            random_idx = np.random.randint(len(available_spectra[label]["spectra"]))
            spectrum_to_add = available_spectra[label]["spectra"][random_idx].copy()
            composition_to_add = available_spectra[label]["compositions"][random_idx]

            assert np.isclose(
                simps(spectrum_to_add), 1
            ), f"Spectrum for {label} not area-normalized before mixing."

            mixed_spectrum += weight * spectrum_to_add

            label_idx = opts.labels.index(label)
            y_ft_mix_w[i, label_idx] = weight
            y_ft_mix_c[i, label_idx] = composition_to_add

        assert np.isclose(
            simps(mixed_spectrum), 1
        ), f"Mixed spectrum not area-normalized before augmentation."

        if np.random.rand() < 0.30:
            empirical_std = np.std(
                np.concatenate([mixed_spectrum[:1500], mixed_spectrum[-1500:]])
            )
            std_noise = empirical_std * np.random.uniform(0.8, 1.2)
            mixed_spectrum += np.random.normal(
                loc=0.0,
                scale=std_noise,
                size=mixed_spectrum.shape,
            )

        if np.random.rand() < 0.1 and getattr(opts, "masking", False):
            win_to_erase_ppm = np.random.uniform(0.5, 1.5)
            win_to_erase = int(win_to_erase_ppm / eta)
            pos_to_erase = np.random.randint(0, len(mixed_spectrum) - win_to_erase)

            mixed_spectrum[pos_to_erase : pos_to_erase + win_to_erase] = (
                np.random.normal(
                    np.mean(mixed_spectrum),
                    np.max(mixed_spectrum),
                    win_to_erase,
                )
            )

            mask_region = mixed_spectrum[pos_to_erase : pos_to_erase + win_to_erase]
            mask_region[mask_region < np.median(mixed_spectrum)] = 1e-6
            mixed_spectrum[pos_to_erase : pos_to_erase + win_to_erase] = mask_region

        present_labels = set(chosen_labels)
        has_lldpe = bool(present_labels.intersection({"EH", "EO", "EB"}))
        has_pe_like = bool(present_labels.intersection({"PE", "LDPE"}))

        if has_lldpe and np.random.rand() < 0.10:
            mixed_spectrum = add_chain_end_artifacts(
                mixed_spectrum,
                ppm_domain_general,
                intensity_range=(1e-5, 5e-4),
                probability_per_peak=0.4,
            )

        elif (not has_lldpe) and has_pe_like and np.random.rand() < 0.05:
            mixed_spectrum = add_chain_end_artifacts(
                mixed_spectrum,
                ppm_domain_general,
                intensity_range=(5e-6, 2e-4),
                probability_per_peak=0.3,
            )

        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        mixed_spectrum = np.roll(mixed_spectrum, shift)
        mixed_spectrum = safe_area_normalize(mixed_spectrum, clip_min=1e-16)

        assert np.isclose(
            simps(mixed_spectrum), 1
        ), f"Mixed spectrum not area-normalized after augmentation."

        X_ft_mix[i] = mixed_spectrum

    # %% MERGE FINE-TUNING DATASETS

    X_ft = np.concatenate([X_ft_single, X_ft_mix], axis=0)
    y_ft_w = np.concatenate([y_ft_single_w, y_ft_mix_w], axis=0)
    y_ft_c = np.concatenate([y_ft_single_c, y_ft_mix_c], axis=0)

    areas_ft = np.array([simps(s) for s in X_ft])
    print("FT areas:", areas_ft.min(), np.median(areas_ft), areas_ft.max())
    print("FT areas std:", areas_ft.std())

    print(
        "Weight sums:",
        y_ft_w.sum(axis=1).min(),
        np.median(y_ft_w.sum(axis=1)),
        y_ft_w.sum(axis=1).max(),
    )
    print(
        "Active components:",
        np.unique((y_ft_w > 0).sum(axis=1), return_counts=True),
    )

    for j, label in enumerate(opts.labels):
        vals = y_ft_w[:, j]
        print(
            label,
            "presence:",
            np.mean(vals > 0),
            "mean when present:",
            vals[vals > 0].mean() if np.any(vals > 0) else np.nan,
        )

    # %% NORMALIZATION AND FT SET SAVING

    scaler = pd.read_pickle(f"val_sets/{opts.run_name}/scaler.pkl")

    y_ft_c_norm = np.concatenate(
        [y_ft_c[:, 0:3], scaler.transform(y_ft_c[:, 3:])],
        axis=1,
    )
    y_ft_c_norm[y_ft_w == 0.0] = -1.0

    if "norm" in opts.run_name:
        scaler_spectra = pd.read_pickle(f"val_sets/{opts.run_name}/scaler_spectra.pkl")
        X_ft = scaler_spectra.transform(X_ft.flatten().reshape(-1, 1)).reshape(
            X_ft.shape
        )

    ft_sets_folder = f"ft_sets/{opts.run_name}"
    os.makedirs(ft_sets_folder, exist_ok=True)

    np.save(f"{ft_sets_folder}/X{suffix}", X_ft)
    np.save(f"{ft_sets_folder}/y_w{suffix}", y_ft_w)
    np.save(f"{ft_sets_folder}/y_c{suffix}", y_ft_c)
    np.save(f"{ft_sets_folder}/y_c_norm{suffix}", y_ft_c_norm)

    X_ft = X_ft[..., np.newaxis].astype(np.float32)

    # %% TEST DATASET PREPARATION

    test_data = test_data[
        test_data["w"].apply(lambda x: not any(np.isnan(k) for k in x))
    ]
    test_data = test_data[
        (~test_data["fine-tuning"]) & (test_data["copolymer"] != "TEST")
    ]

    areas_test = np.array(
        [simps(s) for s in test_data.loc[:, spectral_columns].values.astype(float)]
    )
    print("Test areas:", np.min(areas_test), np.median(areas_test), np.max(areas_test))
    print("Test areas std:", np.std(areas_test))

    X_ft_test = test_data.loc[:, spectral_columns].values

    if "norm" in opts.run_name:
        X_ft_test = scaler_spectra.transform(
            X_ft_test.flatten().reshape(-1, 1)
        ).reshape(X_ft_test.shape)

    X_ft_test = X_ft_test[..., np.newaxis].astype(np.float32)

    y_ft_w_test = np.vstack(
        test_data.apply(
            lambda row: np.array(
                [
                    (
                        row["w"][row["copo_tuple"].index(copo)]
                        if copo in row["copo_tuple"]
                        else 0
                    )
                    for copo in opts.labels
                ]
            ),
            axis=1,
        ).to_numpy()
    )

    y_ft_c_test = np.vstack(
        test_data.apply(
            lambda row: np.array(
                [
                    (
                        row["c"][row["copo_tuple"].index(copo)]
                        if copo in row["copo_tuple"]
                        else 0
                    )
                    for copo in opts.labels
                ]
            ),
            axis=1,
        ).to_numpy()
    )

    y_ft_c_norm_test = np.concatenate(
        [y_ft_c_test[:, 0:3], scaler.transform(y_ft_c_test[:, 3:])],
        axis=1,
    )
    y_ft_c_norm_test[y_ft_w_test == 0] = -1

    # %% MODEL COMPILE

    with open(f"models/{opts.run_name}/opts.json", "r") as json_file:
        opts_model = dict2obj(json.load(json_file))

    parent_directory = os.path.abspath(os.path.join("models", os.pardir))
    if parent_directory not in sys.path:
        sys.path.insert(0, parent_directory)

    module_name = f"models.{opts.run_name}.model"
    module = importlib.import_module(module_name)

    CustomModel = getattr(module, "CustomModel")

    gpus = tf.config.list_physical_devices("GPU")

    if len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.MirroredStrategy()

    print(f"Number of GPUs used: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        loss = {}
        metrics = {}

        if "weight" in opts_model.task:
            loss["weight_output"] = [
                getattr(utils.losses, m, None) or m for m in opts_model.loss_weights
            ]
            metrics["weight_output"] = [
                getattr(tf.metrics, m)(**v)
                for m, v in opts_model.metrics_weights.items()
            ]

            if "composition" in opts_model.task:
                loss["composition_output"] = [
                    getattr(utils.losses, m, None) or m
                    for m in opts_model.loss_composition
                ]
                metrics["composition_output"] = [
                    getattr(tf.metrics, m)(**v)
                    for m, v in opts_model.metrics_composition.items()
                ]

        model = CustomModel(
            reg_l2=opts.reg_l2,
            dropout_rate=opts.dropout_rate,
            n_outputs=y_ft_c_norm.shape[1],
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=opts.learning_rate,
                clipnorm=1.0,
            ),
            loss=loss,
            metrics=metrics,
        )

        model(X_ft[:1])
        model.summary()
        model.load_weights(f"models/{opts.run_name}/model.weights.h5")

    # %% DATASET CREATION

    with tf.device("/cpu:0"):
        train_size = int(len(X_ft) * (1 - opts.validation_split))
        val_size = len(X_ft) - train_size

        indices = np.arange(len(X_ft))
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        X_ft_train, X_ft_val = X_ft[train_indices], X_ft[val_indices]
        y_ft_w_train, y_ft_w_val = y_ft_w[train_indices], y_ft_w[val_indices]
        y_ft_c_norm_train, y_ft_c_norm_val = (
            y_ft_c_norm[train_indices],
            y_ft_c_norm[val_indices],
        )

        if "weight" in opts_model.task and "composition" not in opts_model.task:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (X_ft_train, {"weight_output": y_ft_w_train})
            )
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (X_ft_val, {"weight_output": y_ft_w_val})
            )
            test_dataset = tf.data.Dataset.from_tensor_slices(
                (X_ft_test, {"weight_output": y_ft_w_test})
            )

        elif "composition" in opts_model.task and "weight" in opts_model.task:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    X_ft_train,
                    {
                        "weight_output": y_ft_w_train,
                        "composition_output": y_ft_c_norm_train,
                    },
                )
            )
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    X_ft_val,
                    {
                        "weight_output": y_ft_w_val,
                        "composition_output": y_ft_c_norm_val,
                    },
                )
            )
            test_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    X_ft_test,
                    {
                        "weight_output": y_ft_w_test,
                        "composition_output": y_ft_c_norm_test,
                    },
                )
            )

        train_dataset = (
            train_dataset.shuffle(buffer_size=25000)
            .batch(opts.batch_size, drop_remainder=True)
            .repeat(opts.epochs)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        val_dataset = (
            val_dataset.batch(opts.batch_size, drop_remainder=True)
            .repeat(opts.epochs)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        test_dataset = (
            test_dataset.shuffle(buffer_size=len(X_ft_test))
            .batch(len(X_ft_test), drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    steps_per_epoch = np.floor(train_size / opts.batch_size).astype(int)
    validation_steps = np.floor(val_size / opts.batch_size).astype(int)
    test_steps = 1

    # %% MODEL TRAINING

    with open(f"models/{opts.run_name}/opts{suffix}.json", "w") as f:
        json.dump(obj2dict(opts), f)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=opts.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=30,
                min_delta=1e-5,
                restore_best_weights=False,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f"models/{opts.run_name}/model{suffix}.weights.h5",
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            utils.CosineDecayAfterPlateau(
                fixed_lr=opts.learning_rate,
                final_lr=1e-7,
                plateau_epochs=15,
                decay_epochs=50,
            ),
            utils.TestSetEvaluationCallback(
                test_dataset,
                test_steps,
                opts.labels,
                strategy,
            ),
            utils.SaveHistoryCallback(
                f"models/{opts.run_name}/training_history{suffix}.json",
                save_interval=10,
            ),
        ],
        verbose=1,
    )

    print("DONE!")


if __name__ == "__main__":
    main()