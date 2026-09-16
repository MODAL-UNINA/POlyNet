#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compact post-hoc comparison of the frozen baseline models.

Run this only after ``8_baselineComparison.py`` has finished.
Nothing is trained or selected here. The script uses the same experimental
evaluation set as the previous POlyNet analyses, loads frozen PLS, MLP, CNN,
ResCNN and POlyNet models, and produces:

- overall experimental-test metrics;
- pure-material vs mixture metrics;
- class-resolved detection, weight and composition metrics;
- compact comparative versions of the existing POlyNet class-wise analyses;
- the binary-mixture series already discussed in the manuscript/SI.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, r2_score


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODEL_ORDER = ["PLS", "MLP", "CNN", "ResCNN", "POlyNet"]
DISPLAY_NAMES = {
    "LDPE": "LDPE",
    "PE": "HDPE",
    "PP": "i-PP",
    "EH": "LLDPE-H",
    "EO": "LLDPE-O",
    "EB": "LLDPE-B",
    "RACO": "RaCo-PP",
    "EPR": "EPR",
}
HOMOPOLYMER_LIKE = {"LDPE", "PE", "PP"}
COPOLYMER_LABELS = ("EH", "EO", "EB", "RACO", "EPR")
EXPECTED_LABELS = ("LDPE", "PE", "PP", *COPOLYMER_LABELS)

DEFAULT_REFERENCE_RUN = "model_weights_kl_mse_loss_composition_neg2_mse_hybrid_norm"
DEFAULT_FT_SUFFIX = "ft"

# Pre-specified from analyses already present in the manuscript/SI; they are
# not chosen after seeing the new benchmark results.
CONTROLLED_MIXTURES = [
    ("PP", "EPR"),       # main-text i-PP + EPR example
    ("EH", "RACO"),     # SI LLDPE-H + RaCo-PP
    ("EO", "RACO"),     # SI LLDPE-O + RaCo-PP
    ("EB", "RACO"),     # SI LLDPE-B + RaCo-PP
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-hoc analysis of frozen POlyNet baselines.",
        allow_abbrev=False,
    )
    p.add_argument("--reference_run", default=DEFAULT_REFERENCE_RUN)
    p.add_argument("--ft_suffix", default=DEFAULT_FT_SUFFIX)
    p.add_argument("--test_dataset", default="DATASET/test_data.pkl")
    p.add_argument(
        "--benchmark_dir",
        default="OUTPUT/baseline_comparison/seed_42",
        help="Directory produced by 8_baselineComparison.py.",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Defaults to <benchmark_dir>/analysis.",
    )
    return p.parse_args()


def load_benchmark_module():
    """Import 8_baselineComparison.py without executing its main()."""
    path = Path("8_baselineComparison.py")
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("baseline_benchmark", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def sample_f1(y_w_true, y_w_pred, threshold) -> float:
    true_presence = (y_w_true > 0).astype(int)
    pred_presence = (y_w_pred >= threshold).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(
        true_presence,
        pred_presence,
        average="samples",
        zero_division=0,
    )
    return float(f1)


def canonical_metrics(bench, y_w_true, y_c_true_norm, y_w_pred, y_c_pred_norm) -> dict:
    """Metrics on the normalized composition representation."""
    for name, values in {
        "y_w_true": y_w_true,
        "y_c_true_norm": y_c_true_norm,
        "y_w_pred": y_w_pred,
        "y_c_pred_norm": y_c_pred_norm,
    }.items():
        if np.asarray(values).shape != y_w_true.shape:
            raise AssertionError(f"{name} shape {np.asarray(values).shape} != {y_w_true.shape}")

    true_presence = y_w_true > 0
    pred_presence = y_w_pred >= bench.TEST_THRESHOLD
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_presence.astype(int), pred_presence.astype(int), average="macro", zero_division=0
    )
    composition_mask = (
        (y_c_true_norm >= 0)
        & np.isin(np.arange(y_w_true.shape[1])[None, :], [3, 4, 5, 6, 7])
    )
    if not np.any(composition_mask):
        raise AssertionError("No present copolymer composition entries available for metrics.")
    composition_error = np.abs(y_c_true_norm - y_c_pred_norm)[composition_mask]
    composition_true = y_c_true_norm[composition_mask]
    composition_pred = y_c_pred_norm[composition_mask]

    out = {
        "detection_macro_precision": float(precision),
        "detection_macro_recall": float(recall),
        "detection_macro_f1": float(f1),
        "exact_component_set_accuracy": float(np.mean(np.all(true_presence == pred_presence, axis=1))),
        "weight_mae_present": float(np.mean(np.abs(y_w_true - y_w_pred)[true_presence])),
        "composition_mae_copolymers": float(np.mean(composition_error)),
        "composition_rmse_copolymers": float(np.sqrt(np.mean((composition_true - composition_pred) ** 2))),
        "composition_r2_copolymers": float(r2_score(composition_true, composition_pred)),
        "false_positive_weight_mass": float(np.mean(np.sum(np.where(~true_presence, y_w_pred, 0.0), axis=1))),
    }
    out["detection_sample_f1"] = sample_f1(
        y_w_true, y_w_pred, bench.TEST_THRESHOLD
    )
    return out


def build_targets(df: pd.DataFrame, labels: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Build dense weight and composition target matrices."""
    y_w = np.zeros((len(df), len(labels)), dtype=np.float32)
    y_c = np.zeros((len(df), len(labels)), dtype=np.float32)
    label_to_index = {label: j for j, label in enumerate(labels)}
    for i, idx in enumerate(df.index):
        copo_tuple, w_tuple, c_tuple = df.loc[idx, ["copo_tuple", "w", "c"]]
        if not (len(copo_tuple) == len(w_tuple) == len(c_tuple)):
            raise ValueError(f"Row {idx!r} has inconsistent copo_tuple/w/c lengths.")
        for label, weight in zip(copo_tuple, w_tuple):
            if label not in label_to_index:
                raise ValueError(f"Row {idx!r} has unknown label {label!r}.")
            y_w[i, label_to_index[label]] = weight
        for label, composition in zip(copo_tuple, c_tuple):
            y_c[i, label_to_index[label]] = composition
    return y_w, y_c


def validate_composition_scaler(scaler) -> None:
    """Validate the five-channel copolymer composition scaler."""
    if scaler.__class__.__name__ != "MinMaxScaler":
        raise AssertionError("The composition scaler must be a MinMaxScaler.")
    n_features = getattr(scaler, "n_features_in_", len(getattr(scaler, "scale_", [])))
    if n_features != len(COPOLYMER_LABELS) or len(scaler.scale_) != len(COPOLYMER_LABELS):
        raise AssertionError("Expected a five-channel copolymer composition scaler.")


def assert_reference_composition_scaler(scaler, reference_run: str) -> None:
    """Verify the loaded scaler matches the reference fitted scaler."""
    reference_path = Path("val_sets") / reference_run / "scaler.pkl"
    reference_scaler = pd.read_pickle(reference_path)
    validate_composition_scaler(reference_scaler)
    for attribute in ("scale_", "min_", "data_min_", "data_max_"):
        np.testing.assert_allclose(
            getattr(scaler, attribute), getattr(reference_scaler, attribute),
            atol=0.0, rtol=0.0,
            err_msg=f"Composition scaler differs from reference scaler ({attribute}).",
        )


def load_experimental_test(bench, path: Path, spectral_scaler, composition_scaler):
    """Load the established experimental evaluation set, preserving row order."""
    df = pd.read_pickle(path)
    required = {"copolymer", "copo_tuple", "w", "c"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required experimental columns: {sorted(missing)}")
    if tuple(bench.LABELS) != EXPECTED_LABELS:
        raise AssertionError(f"Unexpected label order: {bench.LABELS}")
    if df.empty:
        raise AssertionError("Experimental dataframe is empty.")

    rows_with_missing_w = int(df["w"].apply(lambda x: any(np.isnan(v) for v in x)).sum())
    if rows_with_missing_w:
        raise ValueError("Experimental test dataframe has missing weights.")
    X_raw = np.asarray(df.iloc[:, 4:].values, dtype=np.float32)
    y_w_true, y_c_true_physical = build_targets(df, EXPECTED_LABELS)
    if not np.all(np.isfinite(y_w_true)) or not np.all(np.isfinite(y_c_true_physical)):
        raise ValueError("Experimental targets contain non-finite values.")
    if not (len(df) == len(X_raw) == len(y_w_true) == len(y_c_true_physical)):
        raise AssertionError("Dataframe, spectra, and targets are not row-aligned.")
    X_test = bench.normalize_spectra(X_raw, spectral_scaler)
    y_c_true_norm = bench.normalize_compositions(y_c_true_physical, composition_scaler).astype(np.float32)
    y_c_true_norm = np.where(y_w_true != 0, y_c_true_norm, -1.0).astype(np.float32)
    expected_norm = bench.normalize_compositions(y_c_true_physical, composition_scaler).astype(np.float32)
    np.testing.assert_allclose(y_c_true_norm[y_w_true != 0], expected_norm[y_w_true != 0], atol=1e-7)
    if not np.all(y_c_true_norm[y_w_true == 0] == -1.0):
        raise AssertionError("Absent composition targets must use the -1 sentinel.")
    return df, X_test, y_w_true, y_c_true_physical, y_c_true_norm


def predict_neural(bench, model, X, composition_scaler):
    pred = model.predict(X[..., np.newaxis], batch_size=64, verbose=0)
    y_w = np.asarray(pred["weight_output"], dtype=np.float32)
    y_c_norm = np.asarray(pred["composition_output"], dtype=np.float32)
    y_c_physical = bench.denormalize_compositions(y_c_norm, composition_scaler)
    # Inverting then reapplying the scaler must recover the raw network output.
    np.testing.assert_allclose(
        bench.normalize_compositions(y_c_physical, composition_scaler)[:, 3:],
        y_c_norm[:, 3:], atol=2e-6, rtol=2e-6,
    )
    return y_w, y_c_norm, y_c_physical


def class_metrics(bench, model_name, y_w_true, y_c_true_norm, y_w_pred, y_c_pred_norm):
    rows = []
    for j, label in enumerate(bench.LABELS):
        present = y_w_true[:, j] > 0
        absent = ~present
        pred_present = y_w_pred[:, j] >= bench.TEST_THRESHOLD

        precision, recall, f1, _ = precision_recall_fscore_support(
            present.astype(int),
            pred_present.astype(int),
            average="binary",
            zero_division=0,
        )

        # Normalized targets >= 0 are valid; absent targets are exactly -1.
        present_composition = y_c_true_norm[:, j] >= 0
        comp_mae = np.nan
        if label in COPOLYMER_LABELS and np.any(present_composition):
            comp_mae = float(np.mean(np.abs(
                y_c_true_norm[present_composition, j] - y_c_pred_norm[present_composition, j]
            )))

        rows.append(
            {
                "model": model_name,
                "label": label,
                "display_label": DISPLAY_NAMES[label],
                "support_present": int(np.sum(present)),
                "detection_precision": float(precision),
                "detection_recall": float(recall),
                "detection_f1": float(f1),
                "weight_mae_present": float(
                    np.mean(np.abs(y_w_true[present, j] - y_w_pred[present, j]))
                ) if np.any(present) else np.nan,
                "weight_mae_absent": float(
                    np.mean(np.abs(y_w_pred[absent, j]))
                ) if np.any(absent) else np.nan,
                "false_positive_rate_absent": float(
                    np.mean(pred_present[absent])
                ) if np.any(absent) else np.nan,
                "false_negative_rate_present": float(
                    np.mean(~pred_present[present])
                ) if np.any(present) else np.nan,
                "composition_mae_present": comp_mae,
            }
        )
    return rows


def plot_pure_vs_mixture(df: pd.DataFrame, path: Path) -> None:
    specs = [
        ("detection_sample_f1", "Detection sample-F1"),
        ("exact_component_set_accuracy", "Exact component-set accuracy"),
        ("weight_mae_present", "Weight MAE"),
        ("composition_mae_copolymers", "Normalized composition MAE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    for ax, (metric, title) in zip(axes.ravel(), specs):
        pure = [df[(df.model == m) & (df.sample_type == "Pure")][metric].iloc[0]
                for m in MODEL_ORDER]
        mix = [df[(df.model == m) & (df.sample_type == "Mixture")][metric].iloc[0]
               for m in MODEL_ORDER]
        ax.bar(x - width / 2, pure, width, label="Pure")
        ax.bar(x + width / 2, mix, width, label="Mixture")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes.ravel()[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_class_heatmaps(bench, df: pd.DataFrame, path: Path) -> None:
    specs = [
        ("detection_f1", "Detection F1"),
        ("weight_mae_present", "Weight MAE"),
        ("composition_mae_present", "Normalized composition MAE"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (metric, title) in zip(axes, specs):
        matrix = np.full((len(MODEL_ORDER), len(bench.LABELS)), np.nan)
        for i, model in enumerate(MODEL_ORDER):
            for j, label in enumerate(bench.LABELS):
                row = df[(df.model == model) & (df.label == label)]
                if len(row):
                    matrix[i, j] = row.iloc[0][metric]
        im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto")
        ax.set_xticks(range(len(bench.LABELS)))
        ax.set_xticklabels([DISPLAY_NAMES[x] for x in bench.LABELS], rotation=45, ha="right")
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_yticklabels(MODEL_ORDER)
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(j, i, "–" if not np.isfinite(value) else f"{value:.3f}",
                        ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_class_weight_profiles(bench, predictions, y_w_true, path: Path) -> None:
    """Comparative version of the existing class-wise POlyNet weight plots."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    handles = labels = None
    for j, label in enumerate(bench.LABELS):
        ax = axes.ravel()[j]
        mask = y_w_true[:, j] > 0
        true = y_w_true[mask, j]
        order = np.argsort(true)
        true = true[order]
        x = np.arange(len(true))
        ax.plot(x, true, linewidth=2, label="True")
        for model in MODEL_ORDER:
            pred = predictions[model]["y_w"][mask, j][order]
            ax.plot(x, pred, marker="o", linestyle="none", alpha=0.55, label=model)
        ax.set_title(DISPLAY_NAMES[label])
        ax.set_xlabel("Samples sorted by true weight")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[0, 0].set_ylabel("Weight fraction")
    axes[1, 0].set_ylabel("Weight fraction")
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_class_composition_parity(bench, predictions, y_w_true, y_c_true_norm, path: Path) -> None:
    """Comparative version of the existing POlyNet composition parity plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    handles = labels = None
    for p, label in enumerate(bench.LABELS[3:]):
        j = bench.LABELS.index(label)
        ax = axes.ravel()[p]
        mask = y_c_true_norm[:, j] >= 0
        true = y_c_true_norm[mask, j]
        lo, hi = float(np.min(true)), float(np.max(true))
        pad = max((hi - lo) * 0.05, 0.05)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linewidth=1.5, label="Ideal")
        for model in MODEL_ORDER:
            ax.scatter(true, predictions[model]["y_c_pred_norm"][mask, j], s=22, alpha=0.55, label=model)
        ax.set_title(DISPLAY_NAMES[label])
        ax.set_xlabel("True normalized composition")
        ax.set_ylabel("Predicted normalized composition")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.2)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes.ravel()[-1].set_axis_off()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def mixture_mask(bench, y_w_true, family) -> np.ndarray:
    presence = y_w_true > 0
    j1, j2 = [bench.LABELS.index(x) for x in family]
    return presence[:, j1] & presence[:, j2] & (presence.sum(axis=1) == 2)


def mixture_name(family) -> str:
    return f"{DISPLAY_NAMES[family[0]]} + {DISPLAY_NAMES[family[1]]}"


def plot_mixture_series(bench, family, predictions, y_w_true, y_c_true_norm, path: Path) -> int:
    """Direct model-to-model comparison on one pre-specified binary series."""
    mask = mixture_mask(bench, y_w_true, family)
    n = int(mask.sum())
    if n == 0:
        return 0

    js = [bench.LABELS.index(x) for x in family]
    order = np.argsort(y_w_true[mask, js[0]])
    x = np.arange(n)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
    handles = labels = None

    for col, j in enumerate(js):
        label = bench.LABELS[j]
        axes[0, col].plot(x, y_w_true[mask, j][order], linewidth=2, label="True")
        for model in MODEL_ORDER:
            axes[0, col].plot(
                x, predictions[model]["y_w"][mask, j][order],
                marker="o", linestyle="none", alpha=0.6, label=model,
            )
        axes[0, col].set_title(f"Weight — {DISPLAY_NAMES[label]}")
        axes[0, col].set_ylabel("Weight fraction")
        axes[0, col].set_ylim(-0.05, 1.05)
        axes[0, col].grid(alpha=0.2)
        if handles is None:
            handles, labels = axes[0, col].get_legend_handles_labels()

        if label in HOMOPOLYMER_LIKE:
            axes[1, col].text(0.5, 0.5, "Composition not applicable",
                              ha="center", va="center", transform=axes[1, col].transAxes)
            axes[1, col].set_axis_off()
        else:
            axes[1, col].plot(x, y_c_true_norm[mask, j][order], linewidth=2, label="True")
            for model in MODEL_ORDER:
                axes[1, col].plot(
                    x, predictions[model]["y_c_pred_norm"][mask, j][order],
                    marker="o", linestyle="none", alpha=0.6, label=model,
                )
            axes[1, col].set_title(f"Composition — {DISPLAY_NAMES[label]}")
            axes[1, col].set_xlabel("Mixture samples")
            axes[1, col].set_ylabel("Normalized composition")
            axes[1, col].grid(alpha=0.2)

    fig.suptitle(mixture_name(family), fontsize=15)
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return n


def main() -> None:
    args = parse_args()
    bench = load_benchmark_module()

    # Imported here so --help and syntax checks do not require TensorFlow.
    import tensorflow as tf
    from utils.baseline_models import build_baseline_model

    suffix = bench.normalize_suffix(args.ft_suffix)
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir) if args.output_dir else benchmark_dir / "analysis"
    figure_dir = output_dir / "figures"
    prediction_dir = output_dir / "predictions"
    mixture_dir = figure_dir / "mixture_series"
    for d in (output_dir, figure_dir, prediction_dir, mixture_dir):
        d.mkdir(parents=True, exist_ok=True)

    model_dir = Path("models") / args.reference_run
    val_dir = Path("val_sets") / args.reference_run
    checkpoint_dir = benchmark_dir / "checkpoints"

    reference_opts = bench.load_json(bench.check_file(model_dir / "opts.json"))
    composition_scaler = pd.read_pickle(bench.check_file(val_dir / "scaler.pkl"))
    spectral_scaler = pd.read_pickle(bench.check_file(val_dir / "scaler_spectra.pkl"))
    test_path = bench.check_file(Path(args.test_dataset))
    validate_composition_scaler(composition_scaler)
    assert_reference_composition_scaler(composition_scaler, args.reference_run)

    test_df, X_test, y_w_true, y_c_true_physical, y_c_true_norm = (
        load_experimental_test(bench, test_path, spectral_scaler, composition_scaler)
    )
    if not (len(test_df) == len(X_test) == len(y_w_true) == len(y_c_true_norm)):
        raise AssertionError("Dataframe, spectra, and normalized targets must have identical row counts.")
    cardinality = (y_w_true > 0).sum(axis=1)
    print(
        "[INFO] Using the established POlyNet experimental evaluation set: "
        f"{len(X_test)} spectra ({int((cardinality == 1).sum())} pure, "
        f"{int((cardinality > 1).sum())} mixtures)."
    )

    metadata = test_df[["copolymer", "copo_tuple", "w", "c"]].copy()
    metadata.insert(0, "original_index", test_df.index)
    metadata.to_csv(output_dir / "test_metadata.csv", index=False)

    predictions = {}

    # PLS --------------------------------------------------------------------
    print("[INFO] Loading PLS")
    pls_path = bench.check_file(checkpoint_dir / "pls" / "pls_bundle.joblib")
    bundle = joblib.load(pls_path)
    y_w, y_c_pred_physical = bench.predict_pls(bundle, X_test)
    y_c_pred_norm = bench.normalize_compositions(
        y_c_pred_physical, composition_scaler
    ).astype(np.float32)
    np.testing.assert_allclose(
        bench.denormalize_compositions(y_c_pred_norm, composition_scaler)[:, 3:],
        y_c_pred_physical[:, 3:], atol=2e-6, rtol=2e-6,
    )
    predictions["PLS"] = {
        "y_w": y_w,
        "y_c_pred_norm": y_c_pred_norm,
        "y_c_pred_physical": y_c_pred_physical,
    }
    np.savez_compressed(
        prediction_dir / "pls.npz", y_w=y_w, y_c_pred_norm=y_c_pred_norm,
        y_c_pred_physical=y_c_pred_physical,
    )
    del bundle

    # MLP / CNN / ResCNN ------------------------------------------------------
    for key, name in [("mlp", "MLP"), ("cnn", "CNN"), ("rescnn", "ResCNN")]:
        print(f"[INFO] Loading {name}")
        weights = bench.check_file(checkpoint_dir / key / f"model{suffix}.weights.h5")
        model = build_baseline_model(
            key,
            n_outputs=len(bench.LABELS),
            reg_l2=float(reference_opts["reg_l2"]),
            dropout_rate=float(reference_opts["dropout_rate"]),
        )
        _ = model(X_test[:1, ..., None], training=False)
        model.load_weights(weights)
        y_w, y_c_pred_norm, y_c_pred_physical = predict_neural(
            bench, model, X_test, composition_scaler
        )
        predictions[name] = {
            "y_w": y_w,
            "y_c_pred_norm": y_c_pred_norm,
            "y_c_pred_physical": y_c_pred_physical,
        }
        np.savez_compressed(
            prediction_dir / f"{key}.npz", y_w=y_w,
            y_c_pred_norm=y_c_pred_norm, y_c_pred_physical=y_c_pred_physical,
        )
        del model
        tf.keras.backend.clear_session()

    # POlyNet -----------------------------------------------------------------
    print("[INFO] Loading POlyNet")
    model, weights = bench.load_reference_polynet(model_dir, reference_opts, suffix)
    _ = model(X_test[:1, ..., None], training=False)
    model.load_weights(weights)
    y_w, y_c_pred_norm, y_c_pred_physical = predict_neural(
        bench, model, X_test, composition_scaler
    )
    predictions["POlyNet"] = {
        "y_w": y_w,
        "y_c_pred_norm": y_c_pred_norm,
        "y_c_pred_physical": y_c_pred_physical,
    }
    np.savez_compressed(
        prediction_dir / "polynet.npz", y_w=y_w,
        y_c_pred_norm=y_c_pred_norm, y_c_pred_physical=y_c_pred_physical,
    )
    del model
    tf.keras.backend.clear_session()

    for name in MODEL_ORDER:
        if len(predictions[name]["y_w"]) != len(X_test):
            raise AssertionError(f"{name} did not predict every canonical test spectrum.")
        if predictions[name]["y_c_pred_norm"].shape != y_c_true_norm.shape:
            raise AssertionError(f"{name} normalized composition prediction shape is inconsistent.")

    # Overall -----------------------------------------------------------------
    rows = []
    for name in MODEL_ORDER:
        rows.append({
            "model": name,
            **canonical_metrics(
                bench, y_w_true, y_c_true_norm,
                predictions[name]["y_w"], predictions[name]["y_c_pred_norm"],
            ),
        })
    overall_df = pd.DataFrame(rows)
    overall_df.to_csv(output_dir / "overall_metrics.csv", index=False)

    # Pure vs mixtures --------------------------------------------------------
    type_masks = {"Pure": cardinality == 1, "Mixture": cardinality > 1}
    rows = []
    for name in MODEL_ORDER:
        for sample_type, mask in type_masks.items():
            rows.append({
                "model": name,
                "sample_type": sample_type,
                "n_samples": int(mask.sum()),
                **canonical_metrics(
                    bench, y_w_true[mask], y_c_true_norm[mask],
                    predictions[name]["y_w"][mask], predictions[name]["y_c_pred_norm"][mask],
                ),
            })
    pure_mix_df = pd.DataFrame(rows)
    pure_mix_df.to_csv(output_dir / "pure_vs_mixture_metrics.csv", index=False)
    plot_pure_vs_mixture(pure_mix_df, figure_dir / "pure_vs_mixture.png")

    # Class-resolved ----------------------------------------------------------
    rows = []
    for name in MODEL_ORDER:
        rows.extend(class_metrics(
            bench, name, y_w_true, y_c_true_norm,
            predictions[name]["y_w"], predictions[name]["y_c_pred_norm"],
        ))
    class_df = pd.DataFrame(rows)
    class_df.to_csv(output_dir / "class_resolved_metrics.csv", index=False)
    plot_class_heatmaps(bench, class_df, figure_dir / "class_resolved_heatmaps.png")
    plot_class_weight_profiles(
        bench, predictions, y_w_true, figure_dir / "class_weight_profiles.png"
    )
    plot_class_composition_parity(
        bench, predictions, y_w_true, y_c_true_norm,
        figure_dir / "class_composition_parity.png",
    )

    # Pre-specified mixture series -------------------------------------------
    rows = []
    mixture_counts = {}
    for family in CONTROLLED_MIXTURES:
        mask = mixture_mask(bench, y_w_true, family)
        n = int(mask.sum())
        title = mixture_name(family)
        mixture_counts[title] = n
        if n == 0:
            print(f"[WARNING] No held-out samples for {title}; skipping.")
            continue
        plot_mixture_series(
            bench, family, predictions, y_w_true, y_c_true_norm,
            mixture_dir / f"{'_'.join(family)}.png",
        )
        for name in MODEL_ORDER:
            rows.append({
                "mixture": title,
                "model": name,
                "n_samples": n,
                **canonical_metrics(
                    bench, y_w_true[mask], y_c_true_norm[mask],
                    predictions[name]["y_w"][mask], predictions[name]["y_c_pred_norm"][mask],
                ),
            })
    pd.DataFrame(rows).to_csv(output_dir / "mixture_series_metrics.csv", index=False)

    save_json(output_dir / "analysis_protocol.json", {
        "test_dataset": str(test_path),
        "models": MODEL_ORDER,
        "labels": list(EXPECTED_LABELS),
        "test_presence_threshold": float(bench.TEST_THRESHOLD),
        "n_test_samples": int(len(X_test)),
        "n_pure_samples": int((cardinality == 1).sum()),
        "n_mixture_samples": int((cardinality > 1).sum()),
        "composition_metric_scale": "normalized composition target scale",
        "controlled_mixture_sample_counts": mixture_counts,
    })

    print("\n[INFO] Analysis complete")
    print(overall_df.to_string(index=False))
    print(f"[INFO] Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
