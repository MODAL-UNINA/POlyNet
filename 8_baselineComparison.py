#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the simpler neural baselines used in the POlyNet comparison.

Design choices:
- reuse the original POlyNet synthetic dataset;
- reuse POlyNet composition/spectral scalers;
- reuse the exact pseudo-synthetic fine-tuning arrays already generated;
- create one deterministic synthetic split and one deterministic FT split,
  shared by MLP, CNN and ResCNN;
- never use the experimental test set during training/model selection;
- evaluate the three baselines only after fine-tuning.

PLS is intentionally left out for now because it follows a different
chemometric training route.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


LABELS = ["LDPE", "PE", "PP", "EH", "EO", "EB", "RACO", "EPR"]
N_OUTPUTS = len(LABELS)
TEST_THRESHOLD = 2.5e-2

DEFAULT_REFERENCE_RUN = "model_weights_kl_mse_loss_composition_neg2_mse_hybrid_norm"


def normalize_suffix(value: str) -> str:
    value = str(value).strip().strip("_")
    return f"_{value}" if value else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled POlyNet neural-baseline comparison.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--reference_run",
        default=DEFAULT_REFERENCE_RUN,
        help="POlyNet run whose scalers/options are reused.",
    )
    parser.add_argument(
        "--ft_suffix",
        default="ft",
        help="Suffix of the saved POlyNet fine-tuning arrays.",
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
        "--test_dataset",
        default="DATASET/test_data.pkl",
    )
    parser.add_argument(
        "--models",
        default="mlp,cnn,rescnn",
        help="Comma-separated subset of mlp,cnn,rescnn.",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument(
        "--reference_gpu_count",
        type=int,
        default=6,
        help="GPU count used by the stored POlyNet global batch sizes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="OUTPUT/baseline_comparison",
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Ignore existing baseline checkpoints.",
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
    """Create one split and reuse it identically for every baseline."""
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
    """Apply the exact one-feature MinMaxScaler coefficients used by POlyNet."""
    if len(scaler.scale_) != 1:
        raise ValueError("Expected one-feature spectral MinMaxScaler.")
    X = np.asarray(X, dtype=np.float32)
    return X * np.float32(scaler.scale_[0]) + np.float32(scaler.min_[0])


def load_synthetic(
    path: Path,
    composition_scaler,
    spectral_scaler,
):
    print(f"[INFO] Loading synthetic dataset: {path}")
    data = pd.read_pickle(path)

    X = normalize_spectra(data[0], spectral_scaler)
    y_w = np.asarray(data[3], dtype=np.float32)
    y_c = np.asarray(data[4], dtype=np.float32)

    y_c_norm = normalize_compositions(y_c, composition_scaler).astype(np.float32)
    y_c_norm[y_w == 0] = -1.0

    if y_w.shape[1] != N_OUTPUTS:
        raise ValueError(f"Expected {N_OUTPUTS} outputs, got {y_w.shape[1]}.")

    print(f"[INFO] Synthetic: X={X.shape}, y={y_w.shape}")
    return X, y_w, y_c_norm


def load_finetuning(ft_dir: Path, suffix: str):
    """Load the exact arrays saved by 2_fineTuneModel.py."""
    paths = {
        "X": check_file(ft_dir / f"X{suffix}.npy"),
        "y_w": check_file(ft_dir / f"y_w{suffix}.npy"),
        "y_c": check_file(ft_dir / f"y_c{suffix}.npy"),
        "y_c_norm": check_file(ft_dir / f"y_c_norm{suffix}.npy"),
    }

    # mmap avoids an unnecessary second full copy on load.
    X = np.load(paths["X"], mmap_mode="r")
    y_w = np.load(paths["y_w"], mmap_mode="r")
    y_c_norm = np.load(paths["y_c_norm"], mmap_mode="r")

    if len(X) != len(y_w) or y_w.shape != y_c_norm.shape:
        raise ValueError("Inconsistent fine-tuning array shapes.")

    print(f"[INFO] Fine-tuning arrays: X={X.shape}, y={y_w.shape}")
    return X, y_w, y_c_norm


def build_targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    label_to_index = {label: i for i, label in enumerate(LABELS)}
    y_w = np.zeros((len(df), N_OUTPUTS), dtype=np.float32)
    y_c = np.zeros((len(df), N_OUTPUTS), dtype=np.float32)

    for row_i, (_, row) in enumerate(df.iterrows()):
        for label, weight, composition in zip(
            row["copo_tuple"], row["w"], row["c"]
        ):
            if label not in label_to_index:
                raise ValueError(f"Unknown test label: {label}")
            j = label_to_index[label]
            y_w[row_i, j] = weight
            y_c[row_i, j] = composition

    return y_w, y_c


def load_test(path: Path, composition_scaler, spectral_scaler):
    """Prepare the experimental set, used only after model selection."""
    print(f"[INFO] Loading experimental test set: {path}")
    df = pd.read_pickle(path)
    df = df[
        df["w"].apply(lambda values: not any(np.isnan(v) for v in values))
    ].copy()

    X = np.asarray(df.iloc[:, 4:].values, dtype=np.float32)
    X = normalize_spectra(X, spectral_scaler)[..., np.newaxis]

    y_w, y_c = build_targets(df)
    y_c_norm = normalize_compositions(y_c, composition_scaler).astype(np.float32)
    y_c_norm[y_w == 0] = -1.0

    print(f"[INFO] Test: X={X.shape}, y={y_w.shape}")
    return X, y_w, y_c_norm


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
    # This mirrors the existing POlyNet code: materialize the selected split,
    # then construct a standard tf.data pipeline.
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


def pretrain(
    model,
    tf,
    utils,
    opts,
    train_ds,
    val_ds,
    steps_per_epoch: int,
    validation_steps: int,
    weights_path: Path,
    history_path: Path,
):
    """Same synthetic-training schedule as 1_trainModel.py, without test callback."""
    lr = float(opts["learning_rate"])
    compile_model(model, tf, utils, opts, lr)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(opts["epochs"]),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=25,
                min_delta=1e-5,
                restore_best_weights=False,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                weights_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            utils.CosineDecayAfterPlateau(
                fixed_lr=lr,
                final_lr=1e-6,
                plateau_epochs=15,
                decay_epochs=35,
            ),
        ],
        verbose=1,
    )

    save_history(history_path, history)
    model.load_weights(weights_path)


def finetune(
    model,
    tf,
    utils,
    reference_opts,
    ft_opts,
    train_ds,
    val_ds,
    steps_per_epoch: int,
    validation_steps: int,
    weights_path: Path,
    history_path: Path,
):
    """Same FT schedule as 2_fineTuneModel.py, reusing already-generated arrays."""
    lr = float(ft_opts["learning_rate"])
    compile_model(model, tf, utils, reference_opts, lr)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(ft_opts["epochs"]),
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
                weights_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            utils.CosineDecayAfterPlateau(
                fixed_lr=lr,
                final_lr=1e-7,
                plateau_epochs=15,
                decay_epochs=50,
            ),
        ],
        verbose=1,
    )

    save_history(history_path, history)
    model.load_weights(weights_path)


def evaluate(model, X_test, y_w_true, y_c_true_norm, composition_scaler):
    pred = model.predict(X_test, batch_size=64, verbose=1)
    y_w_pred = np.asarray(pred["weight_output"])
    y_c_pred_norm = np.asarray(pred["composition_output"])

    true_presence = y_w_true > 0
    pred_presence = y_w_pred >= TEST_THRESHOLD

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_presence.astype(int),
        pred_presence.astype(int),
        average="macro",
        zero_division=0,
    )

    present = true_presence
    weight_mae = float(
        np.mean(np.abs(y_w_true - y_w_pred)[present])
    )

    exact_match = float(
        np.mean(np.all(true_presence == pred_presence, axis=1))
    )

    y_c_true = denormalize_compositions(y_c_true_norm, composition_scaler)
    y_c_pred = denormalize_compositions(y_c_pred_norm, composition_scaler)

    # Only the five actual copolymer composition channels are meaningful.
    copolymer_present = true_presence[:, 3:]
    composition_mae = float(
        np.mean(
            np.abs(y_c_true[:, 3:] - y_c_pred[:, 3:])[copolymer_present]
        )
    )

    fp_mass = float(
        np.mean(
            np.sum(
                np.where(~true_presence, y_w_pred, 0.0),
                axis=1,
            )
        )
    )

    return {
        "detection_macro_precision": float(precision),
        "detection_macro_recall": float(recall),
        "detection_macro_f1": float(f1),
        "exact_component_set_accuracy": exact_match,
        "weight_mae_present": weight_mae,
        "composition_mae_copolymers": composition_mae,
        "false_positive_weight_mass": fp_mass,
    }


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
    invalid = set(requested_models) - {"mlp", "cnn", "rescnn"}
    if invalid:
        raise ValueError(f"Unknown baseline(s): {sorted(invalid)}")

    reference_run = args.reference_run
    suffix = normalize_suffix(args.ft_suffix)

    model_dir = Path("models") / reference_run
    val_dir = Path("val_sets") / reference_run
    ft_dir = (
        Path(args.ft_dir)
        if args.ft_dir
        else Path("ft_sets") / reference_run
    )

    reference_opts = load_json(check_file(model_dir / "opts.json"))
    ft_opts = load_json(check_file(model_dir / f"opts{suffix}.json"))

    composition_scaler = pd.read_pickle(
        check_file(val_dir / "scaler.pkl")
    )
    spectral_scaler = pd.read_pickle(
        check_file(val_dir / "scaler_spectra.pkl")
    )

    physical_gpus = tf.config.list_physical_devices("GPU")
    if len(physical_gpus) == 0:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
    elif len(physical_gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
    else:
        strategy = tf.distribute.MirroredStrategy()

    print(f"[INFO] Strategy replicas: {strategy.num_replicas_in_sync}")

    # The opts files store global batch sizes. Preserve the original per-GPU
    # batch size if the comparison is run on a different number of replicas.
    if args.reference_gpu_count <= 0:
        raise ValueError("--reference_gpu_count must be positive.")

    stored_pretrain_batch = int(reference_opts["batch_size"])
    stored_ft_batch = int(ft_opts["batch_size"])

    if stored_pretrain_batch % args.reference_gpu_count != 0:
        raise ValueError("Cannot infer reference per-GPU pretraining batch size.")
    if stored_ft_batch % args.reference_gpu_count != 0:
        raise ValueError("Cannot infer reference per-GPU fine-tuning batch size.")

    pretrain_batch = (
        stored_pretrain_batch // args.reference_gpu_count
    ) * strategy.num_replicas_in_sync
    ft_batch = (
        stored_ft_batch // args.reference_gpu_count
    ) * strategy.num_replicas_in_sync

    X_synth, y_w_synth, y_c_norm_synth = load_synthetic(
        check_file(Path(args.synthetic_dataset)),
        composition_scaler,
        spectral_scaler,
    )
    X_ft, y_w_ft, y_c_norm_ft = load_finetuning(ft_dir, suffix)

    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    split_dir = output_dir / "splits"
    checkpoint_dir = output_dir / "checkpoints"
    history_dir = output_dir / "history"
    result_dir = output_dir / "results"

    for directory in (
        split_dir,
        checkpoint_dir,
        history_dir,
        result_dir,
    ):
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

    synth_steps_per_epoch = len(synth_train_idx) // pretrain_batch
    synth_validation_steps = len(synth_val_idx) // pretrain_batch
    ft_steps_per_epoch = len(ft_train_idx) // ft_batch
    ft_validation_steps = len(ft_val_idx) // ft_batch

    if min(
        synth_steps_per_epoch,
        synth_validation_steps,
        ft_steps_per_epoch,
        ft_validation_steps,
    ) <= 0:
        raise ValueError(
            "At least one split is smaller than its batch size; "
            "cannot use drop_remainder=True with explicit epoch steps."
        )

    print(
        f"[INFO] Pretraining steps: {synth_steps_per_epoch}/epoch, "
        f"{synth_validation_steps} validation"
    )
    print(
        f"[INFO] Fine-tuning steps: {ft_steps_per_epoch}/epoch, "
        f"{ft_validation_steps} validation"
    )

    # ------------------------------------------------------------------
    # Phase 1: train/select every neural baseline without touching the
    # experimental test set.
    # ------------------------------------------------------------------

    for model_name in requested_models:
        print("\n" + "=" * 72)
        print(f"[INFO] TRAINING {model_name.upper()}")
        print("=" * 72)

        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.keras.utils.set_random_seed(args.seed)

        baseline_dir = checkpoint_dir / model_name
        baseline_dir.mkdir(parents=True, exist_ok=True)

        pretrained_weights = baseline_dir / "model.weights.h5"
        finetuned_weights = baseline_dir / f"model{suffix}.weights.h5"

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

        if args.force_retrain or not pretrained_weights.exists():
            train_ds = make_dataset(
                tf, X_synth, y_w_synth, y_c_norm_synth,
                synth_train_idx, pretrain_batch, True, args.seed
            )
            val_ds = make_dataset(
                tf, X_synth, y_w_synth, y_c_norm_synth,
                synth_val_idx, pretrain_batch, False, args.seed
            )

            with strategy.scope():
                pretrain(
                    model, tf, utils, reference_opts,
                    train_ds, val_ds,
                    synth_steps_per_epoch,
                    synth_validation_steps,
                    pretrained_weights,
                    history_dir / f"{model_name}_pretrain.json",
                )
            del train_ds, val_ds
        else:
            print(f"[INFO] Reusing {pretrained_weights}")
            model.load_weights(pretrained_weights)

        if args.force_retrain or not finetuned_weights.exists():
            train_ds = make_dataset(
                tf, X_ft, y_w_ft, y_c_norm_ft,
                ft_train_idx, ft_batch, True, args.seed
            )
            val_ds = make_dataset(
                tf, X_ft, y_w_ft, y_c_norm_ft,
                ft_val_idx, ft_batch, False, args.seed
            )

            with strategy.scope():
                finetune(
                    model, tf, utils, reference_opts, ft_opts,
                    train_ds, val_ds,
                    ft_steps_per_epoch,
                    ft_validation_steps,
                    finetuned_weights,
                    history_dir / f"{model_name}_finetune.json",
                )
            del train_ds, val_ds
        else:
            print(f"[INFO] Reusing {finetuned_weights}")
            model.load_weights(finetuned_weights)

        del model
        tf.keras.backend.clear_session()

    # ------------------------------------------------------------------
    # Phase 2: only after every model has been trained/selected, access
    # the experimental set once and evaluate the frozen checkpoints.
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("[INFO] FINAL EXPERIMENTAL EVALUATION")
    print("=" * 72)

    X_test, y_w_test, y_c_norm_test = load_test(
        check_file(Path(args.test_dataset)),
        composition_scaler,
        spectral_scaler,
    )

    results = []

    for model_name in requested_models:
        print(f"\n[INFO] Evaluating {model_name.upper()}")

        finetuned_weights = (
            checkpoint_dir / model_name / f"model{suffix}.weights.h5"
        )
        check_file(finetuned_weights)

        with strategy.scope():
            model = build_baseline_model(
                model_name,
                n_outputs=N_OUTPUTS,
                reg_l2=float(reference_opts["reg_l2"]),
                dropout_rate=float(reference_opts["dropout_rate"]),
            )
            _ = model(X_test[:1], training=False)
            model.load_weights(finetuned_weights)

        metrics = evaluate(
            model,
            X_test,
            y_w_test,
            y_c_norm_test,
            composition_scaler,
        )
        metrics["model"] = model_name
        metrics["parameters"] = int(model.count_params())
        metrics["weights_file_mb"] = float(
            finetuned_weights.stat().st_size / 1024**2
        )

        save_json(result_dir / f"{model_name}.json", metrics)
        results.append(metrics)

        print(
            f"[RESULT] F1={metrics['detection_macro_f1']:.4f} | "
            f"W-MAE={metrics['weight_mae_present']:.4f} | "
            f"C-MAE={metrics['composition_mae_copolymers']:.4f}"
        )

        del model
        tf.keras.backend.clear_session()

    results_df = pd.DataFrame(results)
    csv_path = output_dir / "neural_baseline_comparison.csv"
    results_df.to_csv(csv_path, index=False)

    save_json(
        output_dir / "protocol.json",
        {
            "reference_run": reference_run,
            "ft_suffix": suffix,
            "ft_dir": str(ft_dir),
            "seed": args.seed,
            "models": list(requested_models),
            "reference_gpu_count": args.reference_gpu_count,
            "current_replica_count": strategy.num_replicas_in_sync,
            "synthetic_validation_split": reference_opts["validation_split"],
            "ft_validation_split": ft_opts["validation_split"],
            "pretrain_batch_size": pretrain_batch,
            "ft_batch_size": ft_batch,
            "weight_loss": reference_opts["loss_weights"],
            "composition_loss": reference_opts["loss_composition"],
            "test_presence_threshold": TEST_THRESHOLD,
            "experimental_test_used_during_training": False,
        },
    )

    print("\n[INFO] Neural baseline comparison complete.")
    print(results_df.to_string(index=False))
    print(f"[INFO] Results: {csv_path}")


if __name__ == "__main__":
    main()
