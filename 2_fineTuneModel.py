#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:25:08 2024

@author: MODAL
"""

# %% IMPORT SECTION
import os
import sys
import json
import argparse
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from fastcore.all import dict2obj, obj2dict
from scipy.integrate import simpson as simps

import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NMR training script with overrides")

    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3,4,5",
        help='Comma‐separated list of GPUs to use, e.g. "0,1"',
    )

    parser.add_argument(
        "--dataset-size",
        type=int,
        default=50000,
        help="Number of fine-tuning examples.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size to use for training."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="model_weights_kl_mse_loss_composition_neg_mse_model_norm",
        help="Which model should be fine-tuned.",
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        default="DATASET/test_data.pkl",
        help="Which dataset should be used to test the model.",
    )

    args, _ = parser.parse_known_args()

    # Now override the defaults from the command line
    CVD = args.gpus
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = CVD

    # %% OVERALL PARAMETERS

    opts = dict2obj(
        dict(
            labels=["LDPE", "PE", "PP", "EH", "EO", "EB", "RACO", "EPR"],
            n_ft_samples=args.dataset_size,
            single_percent=0.4,
            epochs=175,
            batch_size=args.batch_size * len(CVD.split(",")),
            validation_split=0.3,
            learning_rate=3e-6,  # 3e-6,
            reg_l2=1e-12,
            dropout_rate=0.1,
            run_name=args.model,  # "model_weights_kl_mse_loss_composition_neg_mse_ppraco_bis7_norm_v3",  # model_weights_composition_ZH_NEW",
        )
    )

    # %% TEST DATA IMPORT AND MANIPULATION

    test_data = pd.read_pickle(args.test_dataset)

    # Define a function to zero out values based on the condition
    def update_c(row):
        updated_c = tuple(
            0.0 if material == "PP" else value
            for material, value in zip(row["copo_tuple"], row["c"])
        )
        return updated_c

    y_labels = test_data.values[:, 0]

    if "pp_raco" in opts.run_name or "ppraco" in opts.run_name:
        # substitute PP and RACO with PP_RACO in y_labels
        opts.labels.remove("PP")
        opts.labels.remove("RACO")
        opts.labels.append("PP_RACO")

        y_labels = np.where(y_labels == "PP", "PP_RACO", y_labels)
        y_labels = np.where(y_labels == "RACO", "PP_RACO", y_labels)
        y_labels = np.where(y_labels == "PP+EPR", "PP_RACO+EPR", y_labels)

        test_data["c"] = test_data.apply(update_c, axis=1)

        test_data["copolymer"] = y_labels
        test_data["copo_tuple"] = test_data["copo_tuple"].apply(
            lambda x: tuple([i if i != "PP" and i != "RACO" else "PP_RACO" for i in x])
        )

    # %% TEST ON COMPOSITE MIXTURES
    
    # pp_epr = test_data[test_data["copolymer"] == "PP_RACO+EPR"]
    # pp = test_data[(test_data["copolymer"] == "PP_RACO") & (test_data["c"] == (0.0,))]
    # pp = pp[pp.index.str.contains("Prodigy")]
    # epr = test_data[test_data["copolymer"] == "EPR"]
    # epr = epr[epr.index.str.contains("Envelope")]

    # for c, w in zip(pp_epr["c"], pp_epr["w"]):
    #     if np.all([not pd.isna(k) for k in c]):
    #         epr_comp = c[1]
    #         suitable_epr = epr[epr["c"] == (epr_comp,)]
    #         if len(suitable_epr) > 0:
    #             epr_vals = suitable_epr.values[0][4:-1]
    #             epr_vals -= np.min(epr_vals)
    #             epr_vals /= simps(epr_vals)

    #             pp_vals = pp.sample(1).values[0][4:-1]
    #             pp_vals -= np.min(pp_vals)
    #             pp_vals /= simps(pp_vals)

    #             pp_epr_vals = pp_epr.values[0][4:-1]
    #             pp_epr_vals -= np.min(pp_epr_vals)
    #             pp_epr_vals /= simps(pp_epr_vals)

    #             print(f"PP_RACO+EPR: {w[0]} PP_RACO + {w[1]} EPR")
    #             pp_epr_synth = w[0] * pp_vals + w[1] * epr_vals
    #             pp_epr_synth -= np.min(pp_epr_synth)
    #             pp_epr_synth /= simps(pp_epr_synth)
    #             plt.figure(figsize=(15, 5))
    #             # plt.plot(pp_vals, label="EPR")
    #             plt.plot(pp_epr_vals, label="PP_RACO_EPR")
    #             plt.plot(pp_epr_synth, label="PP_RACO_EPR Synth", alpha=0.5)
    #             plt.title(f"PP_RACO+EPR: {w[0]} PP_RACO + {w[1]} EPR {c[1]}")
    #             plt.yscale("log")
    #             plt.legend()
    #             plt.show()

    # %% GENERATION OF FINE-TUNING DATASET

    # Set 'fine-tuning' flag to False
    test_data["fine-tuning"] = False

    # Randomly select 30% of single copolymers for fine-tuning
    single_copolymers = test_data[
        (test_data["copo_tuple"].apply(lambda x: len(x) == 1))
        & (test_data["copolymer"] != "TEST")
    ]

    n_random_samples_dict = {
        "LDPE": 5,
        "PE": 14,
        "PP": 25,
        "EH": 30,
        "EO": 20,
        "EB": 20,
        "RACO": 20,
        "EPR": 20,
    }
    single_indices = []
    probabilities = []
    for copolymer in single_copolymers["copolymer"].unique():
        n_random_samples = n_random_samples_dict.get(copolymer, 20)
        replace_flag = (
            True
            if len(single_copolymers[single_copolymers["copolymer"] == copolymer])
            < n_random_samples
            else False
        )
        single_indices.extend(
            list(
                single_copolymers[single_copolymers["copolymer"] == copolymer]
                .sample(n_random_samples, replace=replace_flag)
                .index
            )
        )
        probabilities.extend(
            [
                1.0 / (n_random_samples * len(n_random_samples_dict))
                for _ in range(n_random_samples)
            ]
        )

    test_data.loc[single_indices, "fine-tuning"] = True

    # Preprocess data
    # Filter 'test_data' once for 'fine-tuning' samples
    fine_tune_data = test_data[test_data["fine-tuning"] == True]

    # Extract single copolymers for fine tuning
    num_single_samples = int(opts.n_ft_samples * opts.single_percent)
    X_ft_single = np.empty((num_single_samples, test_data.shape[1] - 5))
    y_ft_single_w = np.zeros((num_single_samples, len(opts.labels)))
    y_ft_single_c = np.zeros_like(y_ft_single_w)

    # Generate a list of copolymers to process
    copolymers_list = np.random.choice(
        single_indices, size=num_single_samples, p=probabilities
    )

    for i, idx in tqdm(
        enumerate(copolymers_list),
        total=num_single_samples,
        desc="Computing single copolymers",
    ):
        copolymer = fine_tune_data.loc[idx, "copolymer"]
        composition = fine_tune_data.loc[idx, "c"][0]
        spectrum = fine_tune_data.loc[idx].values[4:-1]

        flag = True
        while flag:
            try:
                mean_noise = np.random.uniform(1e-5, 5e-5)
                std_noise = np.random.uniform(0.75e-5, 1.75e-5)
                spectrum = spectrum + np.random.normal(
                    mean_noise, std_noise, spectrum.shape
                )
                flag = False
            except:
                continue
        spectrum = np.roll(spectrum, int(np.random.uniform(-15, 15)))
        spectrum -= np.min(spectrum)
        spectrum /= simps(spectrum)
        X_ft_single[i] = spectrum
        y_ft_single_w[i, opts.labels.index(copolymer)] = 1.0
        y_ft_single_c[i, opts.labels.index(copolymer)] = composition

    copolymer_pairs = [
        tuple(np.random.choice(opts.labels, 2, replace=False))
        for _ in range(opts.n_ft_samples - num_single_samples)
    ]

    index_tuples = [
        tuple(
            fine_tune_data[fine_tune_data["copolymer"] == copolymer].sample(1).index[0]
            for copolymer in pair
        )
        for pair in copolymer_pairs
    ]

    weights_tuples = [
        (weight, 1 - weight)
        for weight in [
            np.round(np.random.uniform(0.1, 0.9), 2)
            for _ in range(opts.n_ft_samples - num_single_samples)
        ]
    ]

    X_ft_mix = np.empty(
        (opts.n_ft_samples - num_single_samples, test_data.shape[1] - 5)
    )
    y_ft_mix_w = np.zeros((opts.n_ft_samples - num_single_samples, len(opts.labels)))
    y_ft_mix_c = np.zeros_like(y_ft_mix_w)

    for i, (idx1, idx2) in tqdm(
        enumerate(index_tuples),
        total=len(index_tuples),
        desc="Computing mixed copolymers",
    ):
        spectrum1 = fine_tune_data.loc[idx1].values[4:-1]
        spectrum2 = fine_tune_data.loc[idx2].values[4:-1]
        weight1, weight2 = weights_tuples[i]

        spectrum = weight1 * spectrum1 + weight2 * spectrum2

        flag = True
        while flag:
            try:
                mean_noise = np.random.uniform(1e-5, 5e-5)
                std_noise = np.random.uniform(0.75e-5, 1.75e-5)
                spectrum = spectrum + np.random.normal(
                    mean_noise, std_noise, spectrum.shape
                )
                flag = False
            except:
                continue
        spectrum = np.roll(spectrum, int(np.random.uniform(-15, 15)))
        spectrum -= np.min(spectrum)
        spectrum /= simps(spectrum)

        X_ft_mix[i] = spectrum
        y_ft_mix_w[i, opts.labels.index(fine_tune_data.loc[idx1, "copolymer"])] = (
            weight1
        )
        y_ft_mix_w[i, opts.labels.index(fine_tune_data.loc[idx2, "copolymer"])] = (
            weight2
        )
        y_ft_mix_c[i, opts.labels.index(fine_tune_data.loc[idx1, "copolymer"])] = (
            fine_tune_data.loc[idx1, "c"][0]
        )
        y_ft_mix_c[i, opts.labels.index(fine_tune_data.loc[idx2, "copolymer"])] = (
            fine_tune_data.loc[idx2, "c"][0]
        )

    X_ft = np.concatenate([X_ft_single, X_ft_mix], axis=0)
    y_ft_w = np.concatenate([y_ft_single_w, y_ft_mix_w], axis=0)
    y_ft_c = np.concatenate([y_ft_single_c, y_ft_mix_c], axis=0)

    scaler = pd.read_pickle(f"val_sets/{opts.run_name}/scaler.pkl")

    if "pp_raco" in opts.run_name or "ppraco" in opts.run_name:
        y_ft_c_norm = scaler.transform(y_ft_c)
    else:
        if "LDPE" in opts.labels and "PE" in opts.labels and "PP" in opts.labels:
            y_ft_c_norm = np.concatenate(
                [y_ft_c[:, 0:3], scaler.transform(y_ft_c[:, 3:])], axis=1
            )
        else:
            y_ft_c_norm = np.concatenate(
                [y_ft_c[:, 0:1], scaler.transform(y_ft_c[:, 1:])], axis=1
            )

    y_ft_c_norm[y_ft_w == 0.0] = -1.0

    if "norm" in opts.run_name:
        scaler_spectra = pd.read_pickle(f"val_sets/{opts.run_name}/scaler_spectra.pkl")
        X_ft = scaler_spectra.transform(X_ft.flatten().reshape(-1, 1)).reshape(
            X_ft.shape
        )
    X_ft = X_ft[..., np.newaxis].astype(np.float32)

    # %% SET TEST DATA

    # Exclude samples with unknown copolymers or compositions
    test_data = test_data[
        test_data["w"].apply(lambda x: not any(np.isnan(k) for k in x))
    ]
    test_data = test_data[
        (~test_data["fine-tuning"]) & (test_data["copolymer"] != "TEST")
    ]

    X_ft_test = test_data.values[:, 4:-1]
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

    if "pp_raco" in opts.run_name or "ppraco" in opts.run_name:
        print(f"Aggregated PP_RACO")
        y_ft_c_norm_test = scaler.transform(y_ft_c_test)
    else:
        print(f"Separate PP and RACO")
        if "PE" in opts.labels and "PP" in opts.labels and "LDPE" in opts.labels:
            y_ft_c_norm_test = np.concatenate(
                [y_ft_c_test[:, 0:3], scaler.transform(y_ft_c_test[:, 3:])], axis=1
            )
        else:
            y_ft_c_norm_test = np.concatenate(
                [y_ft_c_test[:, 0:1], scaler.transform(y_ft_c_test[:, 1:])], axis=1
            )

    y_ft_c_norm_test[y_ft_w_test == 0] = -1

    # %% MODEL COMPILE

    with open(f"models/{opts.run_name}/opts.json", "r") as json_file:
        opts_model = dict2obj(json.load(json_file))

    # Ensure that 'models' directory is in the parent directory
    parent_directory = os.path.abspath(os.path.join("models", os.pardir))
    if parent_directory not in sys.path:
        sys.path.insert(0, parent_directory)

    # Construct the module name dynamically
    module_name = f"models.{opts.run_name}.model"

    # Import the module dynamically
    module = importlib.import_module(module_name)

    CustomModel = getattr(module, "CustomModel")

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.MirroredStrategy()

    # Creazione della strategia per distribuire il calcolo su più GPU
    print("Numero di GPU utilizzate: {}".format(strategy.num_replicas_in_sync))

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
                # clipnorm=1.0,
            ),
            loss=loss,
            metrics=metrics,
        )

        model(X_ft[:1])
        model.summary()
        model.load_weights(f"models/{opts.run_name}/model.weights.h5")

    # %% DATASET CREATION

    with tf.device("/cpu:0"):
        # Convert numpy arrays to tf.data.Dataset
        train_size = int(len(X_ft) * (1 - opts.validation_split))
        val_size = len(X_ft) - train_size

        # Shuffle indices
        indices = np.arange(len(X_ft))
        np.random.shuffle(indices)  # Inplace operation

        # Split indices into training and validation
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Split the data
        X_ft_train, X_ft_val = X_ft[train_indices], X_ft[val_indices]
        y_ft_w_train, y_ft_w_val = y_ft_w[train_indices], y_ft_w[val_indices]
        y_ft_c_train, y_ft_c_val = y_ft_c[train_indices], y_ft_c[val_indices]
        y_ft_c_norm_train, y_ft_c_norm_val = (
            y_ft_c_norm[train_indices],
            y_ft_c_norm[val_indices],
        )

        # Save the val data
        os.makedirs(f"val_sets/{opts.run_name}", exist_ok=True)

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
            test_dataset.shuffle(buffer_size=len(X_ft_test)).batch(
                len(X_ft_test), drop_remainder=True
            )
            # .repeat(opts.epochs)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    # Set steps_per_epoch and validation_steps based on the dataset sizes
    steps_per_epoch = np.floor(train_size / opts.batch_size).astype(int)
    validation_steps = np.floor(val_size / opts.batch_size).astype(int)
    test_steps = np.floor(len(X_ft_test) / len(X_ft_test)).astype(
        int
    )  # Ensure correct batch count


    # %% MODEL TRAINING
    # save opts file
    with open(f"models/{opts.run_name}/opts_ft.json", "w") as f:
        json.dump(obj2dict(opts), f)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=opts.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=150, restore_best_weights=False
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f"models/{opts.run_name}/model_ft.weights.h5",
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            utils.CosineDecayAfterPlateau(
                fixed_lr=opts.learning_rate,
                final_lr=1e-7,
                plateau_epochs=100,
                decay_epochs=1000,  # opts.epochs * (len(CVD.split(","))) * 5,
            ),
            utils.TestSetEvaluationCallback(test_dataset, test_steps, opts.labels, strategy),
            utils.SaveHistoryCallback(
                f"models/{opts.run_name}/training_history_ft.json", save_interval=10
            ),
        ],
        verbose=1,
    )
# %%
