#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train and select the simple baselines used in the POlyNet comparison.

Baselines:
- PLS: classical chemometric baseline;
- MLP: fully connected neural baseline;
- CNN: conventional 1D-CNN baseline;
- ResCNN: residual 1D-CNN baseline.

Common rules:
- same POlyNet spectral/composition scalers;
- same deterministic synthetic and fine-tuning train/validation splits;
- same supervised targets and losses for the neural models;
- save frozen baseline artifacts for the final comparison in
  ``9_baselineAnalysis.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression


LABELS = ["LDPE", "PE", "PP", "EH", "EO", "EB", "RACO", "EPR"]
N_OUTPUTS = len(LABELS)
TEST_THRESHOLD = 2.5e-2

DEFAULT_REFERENCE_RUN = "model_weights_kl_mse_loss_composition_neg2_mse_hybrid_norm"

# Simple, fixed neural training protocol.
NEURAL_CONFIG = {
    "mlp": {"pretrain_lr": 5e-4, "finetune_lr": 5e-6},
    "cnn": {"pretrain_lr": 5e-4, "finetune_lr": 5e-6},
    "rescnn": {"pretrain_lr": 5e-4, "finetune_lr": 5e-6},
}
PRETRAIN_EPOCHS = 1000
FINETUNE_EPOCHS = 5000
PRETRAIN_PATIENCE = 100
FINETUNE_PATIENCE = 150
# MONITOR = "val_weight_output_mae"
MONITOR = "val_loss"

PLS_COMPONENT_GRID = (5, 10, 20, 50, 100, 200, 500, 1000)


# -----------------------------------------------------------------------------
# Arguments and small utilities
# -----------------------------------------------------------------------------


def normalize_suffix(value: str) -> str:
    value = str(value).strip().strip("_")
    return f"_{value}" if value else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="POlyNet baseline training and selection.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--reference_run",
        default=DEFAULT_REFERENCE_RUN,
        help="POlyNet run whose scalers and options are reused.",
    )
    parser.add_argument(
        "--ft_suffix",
        default="ft",
        help="Suffix of the saved fine-tuning arrays.",
    )
    parser.add_argument(
        "--ft_dir",
        default=None,
        help="Defaults to ft_sets/<reference_run>.",
    )
    parser.add_argument(
        "--synthetic_dataset",
        default="DATASET/synthetic_dataset.pkl",
    )
    parser.add_argument(
        "--models",
        default="pls,mlp,cnn,rescnn",
        help="Comma-separated subset of pls,mlp,cnn,rescnn.",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument(
        "--reference_gpu_count",
        type=int,
        default=6,
        help="GPU count used by the stored POlyNet global batch sizes.",
    )
    parser.add_argument(
        "--conv_batch_divisor",
        type=int,
        default=2,
        help=(
            "CNN/ResCNN batch-size divisor relative to POlyNet. "
            "Default 2 gives 64 spectra/GPU in pretraining and 32/GPU "
            "in fine-tuning on six GPUs. Use 4 if needed for memory."
        ),
    )
    parser.add_argument(
        "--pls_synthetic_samples",
        type=int,
        default=10000,
        help=(
            "Number of synthetic training spectra added to the PLS training set. "
            "All fine-tuning training spectra are also used."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="OUTPUT/baseline_comparison",
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Ignore existing baseline artifacts and retrain requested baselines.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def check_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def get_or_create_split(
    path: Path,
    n_samples: int,
    validation_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one deterministic split and reuse it for every baseline."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        data = np.load(path)
        if int(data["n_samples"]) != n_samples:
            raise ValueError(f"{path}: dataset size changed.")
        if int(data["seed"]) != seed:
            raise ValueError(f"{path}: seed changed.")
        if not np.isclose(float(data["validation_split"]), validation_split):
            raise ValueError(f"{path}: validation split changed.")
        return data["train_indices"], data["val_indices"]

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_train = int(n_samples * (1.0 - validation_split))

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    np.savez(
        path,
        train_indices=train_indices,
        val_indices=val_indices,
        n_samples=n_samples,
        seed=seed,
        validation_split=validation_split,
    )
    return train_indices, val_indices


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------


def normalize_compositions(y_c: np.ndarray, scaler) -> np.ndarray:
    return np.concatenate(
        [y_c[:, :3], scaler.transform(y_c[:, 3:])],
        axis=1,
    )


def denormalize_compositions(y_c_norm: np.ndarray, scaler) -> np.ndarray:
    return np.concatenate(
        [y_c_norm[:, :3], scaler.inverse_transform(y_c_norm[:, 3:])],
        axis=1,
    )


def normalize_spectra(X: np.ndarray, scaler) -> np.ndarray:
    """Apply the exact one-feature MinMaxScaler used by POlyNet."""
    if len(scaler.scale_) != 1:
        raise ValueError("Expected one-feature spectral MinMaxScaler.")
    X = np.asarray(X, dtype=np.float32)
    return X * np.float32(scaler.scale_[0]) + np.float32(scaler.min_[0])


def load_synthetic(path: Path, composition_scaler, spectral_scaler):
    print(f"[INFO] Loading synthetic dataset: {path}")
    data = pd.read_pickle(path)

    X = normalize_spectra(data[0], spectral_scaler)
    y_w = np.asarray(data[3], dtype=np.float32)
    y_c = np.asarray(data[4], dtype=np.float32)

    y_c_norm = normalize_compositions(y_c, composition_scaler).astype(np.float32)
    y_c_norm[y_w == 0] = -1.0

    if y_w.ndim != 2 or y_w.shape[1] != N_OUTPUTS:
        raise ValueError(f"Expected {N_OUTPUTS} weight outputs, got {y_w.shape}.")

    print(f"[INFO] Synthetic: X={X.shape}, y={y_w.shape}")
    return X, y_w, y_c, y_c_norm


def load_finetuning(ft_dir: Path, suffix: str):
    """Load the exact arrays already saved by 2_fineTuneModel.py."""
    paths = {
        "X": check_file(ft_dir / f"X{suffix}.npy"),
        "y_w": check_file(ft_dir / f"y_w{suffix}.npy"),
        "y_c": check_file(ft_dir / f"y_c{suffix}.npy"),
        "y_c_norm": check_file(ft_dir / f"y_c_norm{suffix}.npy"),
    }

    X = np.load(paths["X"], mmap_mode="r")
    y_w = np.load(paths["y_w"], mmap_mode="r")
    y_c = np.load(paths["y_c"], mmap_mode="r")
    y_c_norm = np.load(paths["y_c_norm"], mmap_mode="r")

    if len(X) != len(y_w) or y_w.shape != y_c.shape or y_w.shape != y_c_norm.shape:
        raise ValueError("Inconsistent fine-tuning array shapes.")
    if X.ndim != 2:
        raise ValueError(f"Expected rank-2 FT spectra, got X.shape={X.shape}.")
    if y_w.ndim != 2 or y_w.shape[1] != N_OUTPUTS:
        raise ValueError(f"Expected {N_OUTPUTS} FT outputs, got {y_w.shape}.")

    print(f"[INFO] Fine-tuning arrays: X={X.shape}, y={y_w.shape}")
    return X, y_w, y_c, y_c_norm


# -----------------------------------------------------------------------------
# Neural baselines
# -----------------------------------------------------------------------------


def make_dataset(
    tf,
    X,
    y_w,
    y_c_norm,
    indices,
    batch_size: int,
    training: bool,
    seed: int,
):
    X_part = np.asarray(X[indices], dtype=np.float32)[..., np.newaxis]
    y_w_part = np.asarray(y_w[indices], dtype=np.float32)
    y_c_part = np.asarray(y_c_norm[indices], dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices(
        (
            X_part,
            {
                "weight_output": y_w_part,
                "composition_output": y_c_part,
            },
        )
    )

    if training:
        ds = ds.shuffle(
            buffer_size=min(25000, len(indices)),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    return (
        ds.batch(batch_size, drop_remainder=True)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )


def build_loss_and_metrics(tf, utils, opts: dict):
    losses = {
        "weight_output": [
            getattr(utils.losses, name, None) or name
            for name in opts["loss_weights"]
        ],
        "composition_output": [
            getattr(utils.losses, name, None) or name
            for name in opts["loss_composition"]
        ],
    }

    metrics = {
        "weight_output": [
            getattr(tf.metrics, name)(**kwargs)
            for name, kwargs in opts["metrics_weights"].items()
        ],
        "composition_output": [
            getattr(tf.metrics, name)(**kwargs)
            for name, kwargs in opts["metrics_composition"].items()
        ],
    }
    return losses, metrics


def compile_model(model, tf, utils, opts: dict, learning_rate: float):
    losses, metrics = build_loss_and_metrics(tf, utils, opts)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
        ),
        loss=losses,
        metrics=metrics,
    )


def save_history(path: Path, history) -> None:
    save_json(
        path,
        {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    )


def train_neural_stage(
    model,
    tf,
    utils,
    reference_opts,
    train_ds,
    val_ds,
    learning_rate: float,
    epochs: int,
    patience: int,
    steps_per_epoch: int,
    validation_steps: int,
    weights_path: Path,
    history_path: Path,
):
    """Train one neural stage with a simple validation-driven schedule."""
    compile_model(model, tf, utils, reference_opts, learning_rate)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                weights_path,
                monitor=MONITOR,
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=MONITOR,
                mode="min",
                factor=0.5,
                patience=35,
                min_delta=1e-5,
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor=MONITOR,
                mode="min",
                patience=patience,
                min_delta=1e-5,
                restore_best_weights=False,
                verbose=1,
            ),
        ],
        verbose=2,
    )

    save_history(history_path, history)
    model.load_weights(weights_path)


# -----------------------------------------------------------------------------
# PLS baseline
# -----------------------------------------------------------------------------


def postprocess_weight_predictions(y_pred: np.ndarray) -> np.ndarray:
    """Enforce non-negative weight fractions summing to one."""
    raw = np.asarray(y_pred, dtype=np.float64)
    clipped = np.clip(raw, 0.0, None)
    sums = clipped.sum(axis=1, keepdims=True)

    zero_rows = sums[:, 0] <= 1e-12
    if np.any(zero_rows):
        clipped[zero_rows] = 0.0
        best = np.argmax(raw[zero_rows], axis=1)
        clipped[np.where(zero_rows)[0], best] = 1.0
        sums = clipped.sum(axis=1, keepdims=True)

    return (clipped / sums).astype(np.float32)


def fit_pls_baseline(
    X_synth, y_w_synth, y_c_synth, synth_train_idx,
    X_ft, y_w_ft, y_c_ft, ft_train_idx, ft_val_idx,
    n_synthetic_samples: int, seed: int,
):
    """Fit one simple PLS baseline; choose n_components on FT validation."""
    rng = np.random.default_rng(seed)
    n_synthetic_samples = min(n_synthetic_samples, len(synth_train_idx))
    synth_subset_idx = rng.choice(
        synth_train_idx, size=n_synthetic_samples, replace=False
    )

    X_train = np.concatenate([
        np.asarray(X_synth[synth_subset_idx], dtype=np.float32),
        np.asarray(X_ft[ft_train_idx], dtype=np.float32),
    ])
    y_w_train = np.concatenate([
        np.asarray(y_w_synth[synth_subset_idx], dtype=np.float32),
        np.asarray(y_w_ft[ft_train_idx], dtype=np.float32),
    ])
    y_c_train = np.concatenate([
        np.asarray(y_c_synth[synth_subset_idx], dtype=np.float32),
        np.asarray(y_c_ft[ft_train_idx], dtype=np.float32),
    ])

    X_val = np.asarray(X_ft[ft_val_idx], dtype=np.float32)
    y_w_val = np.asarray(y_w_ft[ft_val_idx], dtype=np.float32)

    print(
        f"[INFO] PLS training set: {len(X_train)} spectra "
        f"({n_synthetic_samples} synthetic + {len(ft_train_idx)} FT)"
    )

    best_model = None
    best_k = None
    best_score = np.inf
    validation_scores = {}

    for k in PLS_COMPONENT_GRID:
        print(f"[INFO] PLS: trying n_components={k}")
        model = PLSRegression(n_components=k, scale=False, max_iter=500)
        model.fit(X_train, y_w_train)
        pred = postprocess_weight_predictions(model.predict(X_val))
        present = y_w_val > 0
        score = float(np.mean(np.abs(y_w_val - pred)[present]))
        validation_scores[str(k)] = score
        print(f"[INFO] PLS validation weight MAE={score:.6f}")
        if score < best_score:
            best_model, best_k, best_score = model, k, score

    print(f"[INFO] PLS selected n_components={best_k}")

    # Use the same selected latent dimensionality for the five composition
    # regressions. Each is trained only where that copolymer is present.
    composition_models = {}
    composition_ranges = {}
    for j, label in enumerate(LABELS[3:], start=3):
        present = y_w_train[:, j] > 0
        if np.sum(present) <= best_k:
            raise ValueError(f"Not enough PLS composition samples for {label}.")
        model = PLSRegression(n_components=best_k, scale=False, max_iter=500)
        model.fit(X_train[present], y_c_train[present, j].reshape(-1, 1))
        composition_models[label] = model
        composition_ranges[label] = (
            float(np.min(y_c_train[present, j])),
            float(np.max(y_c_train[present, j])),
        )

    bundle = {
        "weight_model": best_model,
        "n_components": int(best_k),
        "composition_models": composition_models,
        "composition_ranges": composition_ranges,
    }
    selection = {
        "n_components": int(best_k),
        "validation_weight_mae": validation_scores,
        "synthetic_samples": int(n_synthetic_samples),
    }
    return bundle, selection


def predict_pls(bundle: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_w_pred = postprocess_weight_predictions(bundle["weight_model"].predict(X))
    y_c_pred = np.zeros((len(X), N_OUTPUTS), dtype=np.float32)

    for j, label in enumerate(LABELS[3:], start=3):
        model = bundle["composition_models"][label]
        pred = model.predict(X).reshape(-1)
        lo, hi = bundle["composition_ranges"][label]
        y_c_pred[:, j] = np.clip(pred, lo, hi)

    return y_w_pred, y_c_pred


def load_reference_polynet(model_dir: Path, reference_opts: dict, suffix: str):
    """Load the original fine-tuned POlyNet without retraining it."""
    model_py = check_file(model_dir / "model.py")
    weights_path = check_file(model_dir / f"model{suffix}.weights.h5")

    spec = importlib.util.spec_from_file_location("polynet_reference_model", model_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import reference model from {model_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    CustomModel = getattr(module, "CustomModel")

    model = CustomModel(
        reg_l2=float(reference_opts["reg_l2"]),
        dropout_rate=float(reference_opts["dropout_rate"]),
        n_outputs=N_OUTPUTS,
    )
    return model, weights_path


# -----------------------------------------------------------------------------
# Main baseline training and selection
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    import tensorflow as tf
    import utils
    from utils.baseline_models import build_baseline_model

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    requested_models = tuple(
        name.strip().lower()
        for name in args.models.split(",")
        if name.strip()
    )
    valid_models = {"pls", "mlp", "cnn", "rescnn"}
    invalid = set(requested_models) - valid_models
    if invalid:
        raise ValueError(f"Unknown baseline(s): {sorted(invalid)}")

    neural_models = [m for m in requested_models if m in NEURAL_CONFIG]

    reference_run = args.reference_run
    suffix = normalize_suffix(args.ft_suffix)

    model_dir = Path("models") / reference_run
    val_dir = Path("val_sets") / reference_run
    ft_dir = Path(args.ft_dir) if args.ft_dir else Path("ft_sets") / reference_run

    reference_opts = load_json(check_file(model_dir / "opts.json"))
    ft_opts = load_json(check_file(model_dir / f"opts{suffix}.json"))

    composition_scaler = pd.read_pickle(check_file(val_dir / "scaler.pkl"))
    spectral_scaler = pd.read_pickle(check_file(val_dir / "scaler_spectra.pkl"))

    physical_gpus = tf.config.list_physical_devices("GPU")
    if len(physical_gpus) == 0:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
    elif len(physical_gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
    else:
        strategy = tf.distribute.MirroredStrategy()

    print(f"[INFO] Strategy replicas: {strategy.num_replicas_in_sync}")

    if args.reference_gpu_count <= 0:
        raise ValueError("--reference_gpu_count must be positive.")
    if args.conv_batch_divisor <= 0:
        raise ValueError("--conv_batch_divisor must be positive.")
    if args.pls_synthetic_samples <= 0:
        raise ValueError("--pls_synthetic_samples must be positive.")

    stored_pretrain_batch = int(reference_opts["batch_size"])
    stored_ft_batch = int(ft_opts["batch_size"])

    if stored_pretrain_batch % args.reference_gpu_count != 0:
        raise ValueError("Cannot infer reference per-GPU pretraining batch size.")
    if stored_ft_batch % args.reference_gpu_count != 0:
        raise ValueError("Cannot infer reference per-GPU fine-tuning batch size.")

    reference_pretrain_per_replica = stored_pretrain_batch // args.reference_gpu_count
    reference_ft_per_replica = stored_ft_batch // args.reference_gpu_count

    X_synth, y_w_synth, y_c_synth, y_c_norm_synth = load_synthetic(
        check_file(Path(args.synthetic_dataset)),
        composition_scaler,
        spectral_scaler,
    )
    X_ft, y_w_ft, y_c_ft, y_c_norm_ft = load_finetuning(ft_dir, suffix)

    expected_ft_samples = int(ft_opts["n_ft_samples"])
    if len(X_ft) != expected_ft_samples:
        raise ValueError(
            f"Fine-tuning artifact mismatch: expected {expected_ft_samples}, "
            f"found {len(X_ft)} in {ft_dir}."
        )

    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    split_dir = output_dir / "splits"
    checkpoint_dir = output_dir / "checkpoints"
    history_dir = output_dir / "history"
    for directory in (split_dir, checkpoint_dir, history_dir):
        directory.mkdir(parents=True, exist_ok=True)

    synth_train_idx, synth_val_idx = get_or_create_split(
        split_dir / "synthetic_split.npz",
        len(X_synth),
        float(reference_opts["validation_split"]),
        args.seed,
    )
    ft_train_idx, ft_val_idx = get_or_create_split(
        split_dir / "finetuning_split.npz",
        len(X_ft),
        float(ft_opts["validation_split"]),
        args.seed,
    )

    print(
        f"[INFO] Synthetic split: {len(synth_train_idx)} train / "
        f"{len(synth_val_idx)} val"
    )
    print(
        f"[INFO] FT split: {len(ft_train_idx)} train / "
        f"{len(ft_val_idx)} val"
    )

    # ------------------------------------------------------------------
    # Phase 1A: PLS baseline
    # ------------------------------------------------------------------
    pls_path = checkpoint_dir / "pls" / "pls_bundle.joblib"
    pls_selection_path = history_dir / "pls_selection.json"

    if "pls" in requested_models:
        print("\n" + "=" * 72)
        print("[INFO] TRAINING PLS")
        print("=" * 72)

        if args.force_retrain or not (pls_path.exists() and pls_selection_path.exists()):
            pls_path.parent.mkdir(parents=True, exist_ok=True)
            bundle, selection = fit_pls_baseline(
                X_synth,
                y_w_synth,
                y_c_synth,
                synth_train_idx,
                X_ft,
                y_w_ft,
                y_c_ft,
                ft_train_idx,
                ft_val_idx,
                args.pls_synthetic_samples,
                args.seed,
            )
            joblib.dump(bundle, pls_path, compress=3)
            save_json(pls_selection_path, selection)
        else:
            print(f"[INFO] Reusing completed PLS baseline: {pls_path}")

    # ------------------------------------------------------------------
    # Phase 1B: neural baselines
    # ------------------------------------------------------------------
    batch_protocol = {}

    for model_name in neural_models:
        print("\n" + "=" * 72)
        print(f"[INFO] TRAINING {model_name.upper()}")
        print("=" * 72)

        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.keras.utils.set_random_seed(args.seed)

        batch_divisor = 1 if model_name == "mlp" else args.conv_batch_divisor

        if reference_pretrain_per_replica % batch_divisor != 0:
            raise ValueError(
                f"{model_name}: pretraining per-GPU batch is not divisible "
                f"by batch_divisor={batch_divisor}."
            )
        if reference_ft_per_replica % batch_divisor != 0:
            raise ValueError(
                f"{model_name}: fine-tuning per-GPU batch is not divisible "
                f"by batch_divisor={batch_divisor}."
            )

        pretrain_per_replica = reference_pretrain_per_replica // batch_divisor
        ft_per_replica = reference_ft_per_replica // batch_divisor

        pretrain_batch = pretrain_per_replica * strategy.num_replicas_in_sync
        ft_batch = ft_per_replica * strategy.num_replicas_in_sync

        synth_steps = len(synth_train_idx) // pretrain_batch
        synth_val_steps = len(synth_val_idx) // pretrain_batch
        ft_steps = len(ft_train_idx) // ft_batch
        ft_val_steps = len(ft_val_idx) // ft_batch

        if min(synth_steps, synth_val_steps, ft_steps, ft_val_steps) <= 0:
            raise ValueError(f"{model_name}: a split is smaller than its batch size.")

        config = NEURAL_CONFIG[model_name]
        batch_protocol[model_name] = {
            "pretrain_lr": config["pretrain_lr"],
            "finetune_lr": config["finetune_lr"],
            "pretrain_batch_global": pretrain_batch,
            "pretrain_batch_per_replica": pretrain_per_replica,
            "finetune_batch_global": ft_batch,
            "finetune_batch_per_replica": ft_per_replica,
        }

        print(
            f"[INFO] LR: pretrain={config['pretrain_lr']:.1e}, "
            f"fine-tune={config['finetune_lr']:.1e}"
        )
        print(
            f"[INFO] Batch: pretrain={pretrain_batch} global "
            f"({pretrain_per_replica}/GPU), fine-tune={ft_batch} global "
            f"({ft_per_replica}/GPU)"
        )

        baseline_dir = checkpoint_dir / model_name
        baseline_dir.mkdir(parents=True, exist_ok=True)

        pretrained_weights = baseline_dir / "model.weights.h5"
        finetuned_weights = baseline_dir / f"model{suffix}.weights.h5"
        pretrain_history = history_dir / f"{model_name}_pretrain.json"
        finetune_history = history_dir / f"{model_name}_finetune.json"

        with strategy.scope():
            model = build_baseline_model(
                model_name,
                n_outputs=N_OUTPUTS,
                reg_l2=float(reference_opts["reg_l2"]),
                dropout_rate=float(reference_opts["dropout_rate"]),
            )
            _ = model(
                np.zeros((1, X_synth.shape[1], 1), dtype=np.float32),
                training=False,
            )

        print(f"[INFO] Parameters: {model.count_params():,}")

        pretrain_complete = pretrained_weights.exists() and pretrain_history.exists()
        pretrain_retrained = False

        if args.force_retrain or not pretrain_complete:
            pretrained_weights.unlink(missing_ok=True)
            pretrain_history.unlink(missing_ok=True)

            train_ds = make_dataset(
                tf,
                X_synth,
                y_w_synth,
                y_c_norm_synth,
                synth_train_idx,
                pretrain_batch,
                True,
                args.seed,
            )
            val_ds = make_dataset(
                tf,
                X_synth,
                y_w_synth,
                y_c_norm_synth,
                synth_val_idx,
                pretrain_batch,
                False,
                args.seed,
            )

            with strategy.scope():
                train_neural_stage(
                    model,
                    tf,
                    utils,
                    reference_opts,
                    train_ds,
                    val_ds,
                    config["pretrain_lr"],
                    PRETRAIN_EPOCHS,
                    PRETRAIN_PATIENCE,
                    synth_steps,
                    synth_val_steps,
                    pretrained_weights,
                    pretrain_history,
                )

            del train_ds, val_ds
            pretrain_retrained = True
        else:
            print(f"[INFO] Reusing completed pretraining: {pretrained_weights}")
            model.load_weights(pretrained_weights)

        finetune_complete = finetuned_weights.exists() and finetune_history.exists()
        must_finetune = args.force_retrain or pretrain_retrained or not finetune_complete

        if must_finetune:
            finetuned_weights.unlink(missing_ok=True)
            finetune_history.unlink(missing_ok=True)

            train_ds = make_dataset(
                tf,
                X_ft,
                y_w_ft,
                y_c_norm_ft,
                ft_train_idx,
                ft_batch,
                True,
                args.seed,
            )
            val_ds = make_dataset(
                tf,
                X_ft,
                y_w_ft,
                y_c_norm_ft,
                ft_val_idx,
                ft_batch,
                False,
                args.seed,
            )

            with strategy.scope():
                train_neural_stage(
                    model,
                    tf,
                    utils,
                    reference_opts,
                    train_ds,
                    val_ds,
                    config["finetune_lr"],
                    FINETUNE_EPOCHS,
                    FINETUNE_PATIENCE,
                    ft_steps,
                    ft_val_steps,
                    finetuned_weights,
                    finetune_history,
                )

            del train_ds, val_ds
        else:
            print(f"[INFO] Reusing completed fine-tuning: {finetuned_weights}")
            model.load_weights(finetuned_weights)

        del model
        tf.keras.backend.clear_session()

    protocol = {
        "reference_run": reference_run,
        "ft_suffix": suffix,
        "ft_dir": str(ft_dir),
        "seed": args.seed,
        "requested_baselines": list(requested_models),
        "synthetic_validation_split": reference_opts["validation_split"],
        "ft_validation_split": ft_opts["validation_split"],
        "neural_monitor": MONITOR,
        "pretrain_epochs_max": PRETRAIN_EPOCHS,
        "finetune_epochs_max": FINETUNE_EPOCHS,
        "pretrain_patience": PRETRAIN_PATIENCE,
        "finetune_patience": FINETUNE_PATIENCE,
        "neural_config": NEURAL_CONFIG,
        "batch_protocol": batch_protocol,
        "pls_component_grid": list(PLS_COMPONENT_GRID),
        "pls_synthetic_samples": args.pls_synthetic_samples,
    }
    save_json(output_dir / "protocol.json", protocol)

    print("\n[INFO] Baseline training and selection complete.")
    print(f"[INFO] Frozen artifacts: {checkpoint_dir}")
    print(f"[INFO] Protocol metadata: {output_dir / 'protocol.json'}")


if __name__ == "__main__":
    main()
