#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Attention-based interpretability analysis for DeepNMR models.

The script evaluates whether native attention maps are chemically structured by:

  1. building soft spectral ROIs from experimental monomaterial spectra;
  2. measuring global attention enrichment over ROI-supported spectral regions;
  3. quantifying class-specific attention reactivity through Delta Occupancy;
  4. assessing head redundancy using Jensen-Shannon distance;
  5. evaluating class-wise attention consistency with a leave-one-out contrastive metric;
  6. comparing native attention with Integrated Gradients.

Generated outputs are saved under:

  OUTPUT/attention_analysis/<model_name>/plots<suffix>/

Suffix convention
-----------------
Use --suffix without leading underscore, e.g.:

  --suffix ft

Internally, the suffix is normalized to '_ft', producing paths such as:

  models/<model_name>/model_ft.weights.h5
  OUTPUT/attention_analysis/<model_name>/plots_ft/
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import ast
import sys
import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import jensenshannon

from utils import MappingNames


# %% FIXED CONFIGURATION

LABELS = ["LDPE", "PE", "PP", "EH", "EO", "EB", "RACO", "EPR"]
DISPLAY_NAMES = MappingNames()

VAL_FOLDER = "val_sets"
MODEL_FOLDER = "models"
OUTPUT_ROOT = "OUTPUT/attention_analysis"

EPS = 1e-12
MIN_LOG_INTENSITY = 1e-6

ROI_Q_MASS = 1.0 - 5e-3
TOP_Q = 0.95
TOPK_PERMUTATIONS = 500
TOPK_NULL_MODE = "circular_shift"

DELTA_SMOOTH_SIGMA = 1.0
DELTA_MIN_PRESENT_FRACTION = 0.01
DELTA_MAX_PRESENT_FRACTION = 0.99
DELTA_MIN_SAMPLES = 5

GT_PRESENCE_THRESHOLD = 1e-2
RANDOM_SEED = 42

IG_STEPS = 50
IG_BATCH_SIZE = 16

sns.set_context("talk")
sns.set_style("whitegrid")


# %% ARGUMENTS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Attention-based interpretability analysis for DeepNMR models.",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=(
            "model_weights_kl_mse_loss_composition_neg2_mse_hybrid"
        ),
        help="Name of the model folder under models/ and val_sets/.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="ft",
        help=(
            "Suffix used for the weights file and output folder, without leading "
            "underscore, e.g. 'ftc'. Input with leading underscore is also "
            "accepted. Use '' for base weights."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Optional output directory. If omitted, uses "
            "OUTPUT/attention_analysis/<model_name>/plots<suffix>."
        ),
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="DATASET/test_data.pkl",
        help="Experimental test dataset used for attention analysis.",
    )
    parser.add_argument(
        "--synth_ppm_domain",
        type=str,
        default="DATASET/ppm_domain_general.pkl",
        help="PPM-domain file used for plotting spectra.",
    )
    parser.add_argument(
        "--n_samples_analysis",
        type=int,
        default=1000,
        help="Maximum number of experimental spectra used for attention analysis.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for prediction and attention extraction.",
    )
    parser.add_argument(
        "--snr_k",
        type=float,
        default=6.0,
        help="Noise multiplier used to build soft ROIs from mean spectra.",
    )
    parser.add_argument(
        "--roi_sigma_points",
        type=float,
        default=8.0,
        help="Gaussian smoothing width, in spectral points, used for soft ROI maps.",
    )
    parser.add_argument(
        "--presence_thr",
        type=float,
        default=1e-2,
        help="Predicted-weight threshold used only when ground-truth presence is unavailable.",
    )
    parser.add_argument(
        "--ig_samples",
        type=int,
        default=-1,
        help=(
            "Number of samples used for Integrated Gradients. "
            "Use -1 to use all samples in the analyzed subset."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively in addition to saving them.",
    )

    args, _ = parser.parse_known_args()
    return args


# %% GENERIC UTILITIES


def normalize_suffix(suffix: str | None) -> str:
    """Normalize suffix so that non-empty suffixes start with '_'."""
    if suffix is None:
        return ""

    suffix = str(suffix).strip().strip("_")

    if suffix == "":
        return ""

    return f"_{suffix}"


def mapped_label(label: str) -> str:
    """Map internal class labels to display labels."""
    return DISPLAY_NAMES[label]


def mapped_labels(labels: list[str]) -> list[str]:
    """Map a list of internal labels to display labels."""
    return [mapped_label(label) for label in labels]


def get_plot_dir(
    model_name: str,
    suffix: str | None,
    output_dir: str | None = None,
) -> Path:
    """Return and create the output directory for plots."""
    if output_dir is not None:
        plot_dir = Path(output_dir)
    else:
        suffix = normalize_suffix(suffix)
        plot_dir = Path(OUTPUT_ROOT) / model_name / f"plots{suffix}"

    (plot_dir / "ROI_Check").mkdir(parents=True, exist_ok=True)
    return plot_dir


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    *,
    show: bool = False,
    dpi: int = 300,
    tight: bool = True,
) -> None:
    """Save a matplotlib figure and either show or close it."""
    path = Path(path)

    if tight:
        fig.tight_layout()

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[INFO] Saved: {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def set_nmr_xaxis(ax: plt.Axes) -> None:
    """Invert the x-axis if needed, following NMR ppm convention."""
    xmin, xmax = ax.get_xlim()

    if xmin < xmax:
        ax.invert_xaxis()


def scale_vector(vec: np.ndarray, new_size: int, kind: str = "linear") -> np.ndarray:
    """Interpolate a 1D vector to a new size."""
    vec = np.asarray(vec, dtype=float)

    f = interp1d(
        np.linspace(0, 1, len(vec)),
        vec,
        kind=kind,
        bounds_error=False,
        fill_value=(vec[0], vec[-1]),
    )

    return f(np.linspace(0, 1, new_size))


def normalize_rows(A: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Normalize a matrix row-wise to sum to one."""
    A = np.maximum(np.asarray(A, dtype=float), 0.0)
    return A / (A.sum(axis=1, keepdims=True) + eps)


def hard_mask_from_soft_roi(
    roi: np.ndarray,
    *,
    q_mass: float = ROI_Q_MASS,
) -> np.ndarray:
    """Convert a soft ROI vector to a hard support mask preserving q_mass."""
    roi = np.maximum(np.asarray(roi, dtype=float), 0.0)
    roi = roi / (roi.sum() + EPS)

    sorted_idx = np.argsort(roi)[::-1]
    cumulative = np.cumsum(roi[sorted_idx])
    n_support = int(np.searchsorted(cumulative, q_mass)) + 1
    support_idx = sorted_idx[:n_support]

    mask = np.zeros_like(roi, dtype=int)
    mask[support_idx] = 1

    return mask


def hard_mask_from_full_roi_at_attention_length(
    roi_full: np.ndarray,
    attention_length: int,
    *,
    q_mass: float = ROI_Q_MASS,
) -> np.ndarray:
    """Resize a full-resolution ROI and convert it to a hard attention-length mask."""
    roi_resized = scale_vector(roi_full, attention_length, kind="linear")
    return hard_mask_from_soft_roi(roi_resized, q_mass=q_mass)


def parse_tuple_like(value: Any) -> tuple:
    """Parse tuple-like entries that may already be tuples or may be strings."""
    if isinstance(value, tuple):
        return value

    if isinstance(value, list):
        return tuple(value)

    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, tuple):
            return parsed
        if isinstance(parsed, list):
            return tuple(parsed)
        return (parsed,)

    return (value,)


def load_ppm_domain(path: str | Path) -> np.ndarray:
    """Load ppm domain as a NumPy array."""
    ppm_domain = pd.read_pickle(path)

    if hasattr(ppm_domain, "values"):
        return ppm_domain.values

    return np.asarray(ppm_domain)


# %% DATA AND MODEL LOADING


def load_test_data(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    """Load and optionally normalize the experimental test spectra."""
    test_data = pd.read_pickle(args.test_dataset)
    X_test = test_data.iloc[:, 4:].values.astype(np.float32)

    if "norm" in args.model_name:
        scaler_path = Path(VAL_FOLDER) / args.model_name / "scaler_spectra.pkl"

        if not scaler_path.exists():
            raise FileNotFoundError(f"Spectral scaler not found: {scaler_path}")

        scaler_spectra = pd.read_pickle(scaler_path)
        X_test = scaler_spectra.transform(X_test.flatten().reshape(-1, 1)).reshape(
            X_test.shape
        )

    return test_data, X_test.astype(np.float32)


def load_model(args: argparse.Namespace, X_test: np.ndarray):
    """Load the trained model and weights."""
    project_root = os.path.abspath(os.path.join(MODEL_FOLDER, os.pardir))

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    module = importlib.import_module(f"{MODEL_FOLDER}.{args.model_name}.model")

    model_class_name = (
        "CustomModelSimplified" if "simply" in args.model_name else "CustomModel"
    )
    CustomModel = getattr(module, model_class_name)

    y_c_norm_path = Path(VAL_FOLDER) / args.model_name / "y_c_norm_val.npy"

    if not y_c_norm_path.exists():
        raise FileNotFoundError(f"Validation composition file not found: {y_c_norm_path}")

    dummy_y = np.load(y_c_norm_path)
    model = CustomModel(n_outputs=dummy_y.shape[1])
    model(X_test[:1, :, np.newaxis])

    suffix = normalize_suffix(args.suffix)
    weights_path = Path(MODEL_FOLDER) / args.model_name / f"model{suffix}.weights.h5"

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model.load_weights(weights_path)
    print(f"[INFO] Weights loaded from: {weights_path}")

    return model


# %% ROI CONSTRUCTION


def pure_sample_mask(test_data: pd.DataFrame, copolymer: str) -> np.ndarray:
    """Return a boolean mask for pure samples of a given copolymer."""
    if "copo_tuple" in test_data.columns:
        return test_data["copo_tuple"].apply(
            lambda value: parse_tuple_like(value) == (copolymer,)
        ).values

    if "copolymer" in test_data.columns:
        return (test_data["copolymer"] == copolymer).values

    return np.zeros(len(test_data), dtype=bool)


def build_soft_roi_from_mean_spectrum(
    mean_spectrum: np.ndarray,
    *,
    snr_k: float = 6.0,
    sigma_points: float = 8.0,
    noise_head_tail: int = 2500,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build a soft ROI vector from a mean experimental spectrum.

    The ROI is obtained by retaining signal above an adaptive noise threshold,
    smoothing the resulting positive signal with a Gaussian kernel, and
    normalizing the final soft mask to unit mass.
    """
    mean_spectrum = np.asarray(mean_spectrum, dtype=float)
    n_points = len(mean_spectrum)
    n_noise = min(noise_head_tail, n_points)

    noise_region = np.concatenate(
        [mean_spectrum[:n_noise], mean_spectrum[-n_noise:]]
    )

    noise_mean = float(np.mean(noise_region))
    noise_std = float(np.std(noise_region) + EPS)
    threshold = noise_mean + snr_k * noise_std

    signal_above_threshold = np.maximum(mean_spectrum - threshold, 0.0)
    roi = gaussian_filter1d(signal_above_threshold, sigma=sigma_points)

    if roi.sum() < EPS:
        roi[:] = 1.0

    roi = np.maximum(roi, 0.0)
    roi = roi / (roi.sum() + EPS)

    info = {
        "noise_mean": noise_mean,
        "noise_std": noise_std,
        "threshold": threshold,
        "n_active_points": int(np.sum(signal_above_threshold > 0)),
    }

    return roi, info


def build_class_rois(
    test_data: pd.DataFrame,
    X_test: np.ndarray,
    labels: list[str],
    *,
    snr_k: float,
    roi_sigma_points: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Build one soft ROI map for each copolymer subclass using pure samples."""
    print("\n[INFO] Building soft spectral ROIs from pure experimental spectra.")

    roi_soft_raw: dict[str, np.ndarray] = {}
    roi_info: dict[str, dict[str, float]] = {}

    for copolymer in labels:
        mask = pure_sample_mask(test_data, copolymer)

        if not np.any(mask):
            print(f"[WARNING] ROI {copolymer}: skipped because no pure samples were found.")
            continue

        mean_spectrum = np.mean(X_test[mask], axis=0)

        roi_soft, info = build_soft_roi_from_mean_spectrum(
            mean_spectrum,
            snr_k=snr_k,
            sigma_points=roi_sigma_points,
        )

        roi_soft_raw[copolymer] = roi_soft
        roi_info[copolymer] = info

        print(
            f"[INFO] ROI {copolymer}: pure_samples={int(mask.sum())}, "
            f"active_points={info['n_active_points']}, "
            f"threshold={info['threshold']:.4e}"
        )

    if len(roi_soft_raw) == 0:
        raise RuntimeError(
            "No ROIs were built. Check that the test dataset contains pure samples."
        )

    return roi_soft_raw, roi_info


def build_global_roi(roi_soft_raw: dict[str, np.ndarray]) -> np.ndarray:
    """Combine class-specific ROIs into a single global soft ROI."""
    first_roi = next(iter(roi_soft_raw.values()))
    global_roi = np.zeros_like(first_roi, dtype=float)

    for roi in roi_soft_raw.values():
        global_roi += roi

    global_roi = np.maximum(global_roi, 0.0)
    global_roi = global_roi / (global_roi.sum() + EPS)

    return global_roi


# %% ATTENTION EXTRACTION


def attention_weights_to_position_profile(attention_weights: np.ndarray) -> np.ndarray:
    """Convert raw attention weights to per-head position profiles.

    Accepted shapes:
      - (B, H, Q, K): averaged over query dimension Q -> (B, H, K)
      - (B, H, L): already a per-position attention profile.
    """
    if attention_weights.ndim == 4:
        return attention_weights.mean(axis=2)

    if attention_weights.ndim == 3:
        return attention_weights

    raise ValueError(
        f"Unexpected attention_weights ndim={attention_weights.ndim}, "
        f"shape={attention_weights.shape}"
    )


def extract_attention_weights_from_batch(model, X_batch: np.ndarray) -> np.ndarray:
    """Extract attention weights for one batch using native API or layer fallback."""
    x = X_batch[..., np.newaxis]
    attention_weights = None

    try:
        output = model(x, training=False, return_attention_weights=True)

        if isinstance(output, dict) and "attention_weights" in output:
            attention_weights = output["attention_weights"].numpy()
        else:
            raise TypeError("The model output does not contain 'attention_weights'.")

    except (TypeError, ValueError, KeyError):
        x_tmp = x

        for layer in model.layers:
            layer_name = layer.name.lower()

            if any(
                keyword in layer_name
                for keyword in [
                    "pool",
                    "dense",
                    "global",
                    "output",
                    "composition",
                    "weight",
                ]
            ):
                continue

            layer_output = layer(x_tmp, training=False)

            if isinstance(layer_output, tuple) and len(layer_output) == 2:
                x_tmp, attn_w = layer_output
                attention_weights = attn_w.numpy()
                break

            x_tmp = layer_output

    if attention_weights is None:
        raise RuntimeError(
            "Attention weights could not be extracted. "
            "Check whether the model exposes attention weights or contains an attention block."
        )

    return attention_weights_to_position_profile(attention_weights)


def extract_attention_tensor(
    model,
    X_test: np.ndarray,
    *,
    n_samples: int,
    batch_size: int,
) -> np.ndarray:
    """Extract attention position profiles for an analyzed subset."""
    print("\n[INFO] Extracting attention weights on experimental test spectra.")

    n_samples = min(n_samples, len(X_test))
    all_attention_profiles = []
    printed_shape = False

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        X_batch = X_test[start_idx:end_idx]

        batch_attention = extract_attention_weights_from_batch(model, X_batch)

        if not printed_shape:
            print(f"[INFO] Attention profile shape per batch: {batch_attention.shape}")
            printed_shape = True

        all_attention_profiles.append(batch_attention)

        if end_idx % 256 == 0 or end_idx == n_samples:
            print(f"[INFO] Processed {end_idx}/{n_samples} samples.")

    attn_vec = np.concatenate(all_attention_profiles, axis=0)

    print(f"[INFO] Final attention tensor: {attn_vec.shape}")
    return attn_vec


# %% ATTENTION OCCUPANCY AND ENRICHMENT


def occupancy_topk(
    attn_head_NL: np.ndarray,
    *,
    top_q: float = TOP_Q,
    smooth_sigma: float | None = None,
) -> np.ndarray:
    """Compute top-k occupancy over sequence positions for one attention head."""
    A = normalize_rows(attn_head_NL)
    n_samples, sequence_length = A.shape

    k = int(np.ceil((1.0 - top_q) * sequence_length))
    k = max(1, min(k, sequence_length))

    top_indices = np.argpartition(A, sequence_length - k, axis=1)[:, sequence_length - k :]

    active = np.zeros((n_samples, sequence_length), dtype=float)
    rows = np.arange(n_samples)[:, None]
    active[rows, top_indices] = 1.0

    occupancy = active.mean(axis=0)

    if smooth_sigma is not None and smooth_sigma > 0:
        occupancy = gaussian_filter1d(occupancy, sigma=smooth_sigma)

    return occupancy


def delta_occupancy(
    attn_head_NL: np.ndarray,
    present_mask_N: np.ndarray,
    *,
    top_q: float = TOP_Q,
    smooth_sigma: float | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Compute occupancy difference between samples where a class is present/absent."""
    present_idx = np.where(present_mask_N == 1)[0]
    absent_idx = np.where(present_mask_N == 0)[0]

    if len(present_idx) < DELTA_MIN_SAMPLES or len(absent_idx) < DELTA_MIN_SAMPLES:
        return None, None, None

    occupancy_present = occupancy_topk(
        attn_head_NL[present_idx],
        top_q=top_q,
        smooth_sigma=smooth_sigma,
    )
    occupancy_absent = occupancy_topk(
        attn_head_NL[absent_idx],
        top_q=top_q,
        smooth_sigma=smooth_sigma,
    )

    return occupancy_present - occupancy_absent, occupancy_present, occupancy_absent


def topk_hit_rate_per_sample(
    attn_NL: np.ndarray,
    mask_L: np.ndarray,
    *,
    top_q: float = TOP_Q,
) -> np.ndarray:
    """Compute per-sample fraction of top-k attention positions falling inside a mask."""
    A = normalize_rows(attn_NL)
    mask = (mask_L > 0).astype(int)

    n_samples, sequence_length = A.shape
    k = int(np.ceil((1.0 - top_q) * sequence_length))
    k = max(1, min(k, sequence_length))

    top_indices = np.argpartition(A, sequence_length - k, axis=1)[:, sequence_length - k :]
    hit_rate = mask[top_indices].sum(axis=1) / float(k)

    return hit_rate


def permute_mask(
    mask_L: np.ndarray,
    rng: np.random.Generator,
    *,
    mode: str = TOPK_NULL_MODE,
) -> np.ndarray:
    """Generate a null mask by circular shift or full shuffle."""
    if mode == "circular_shift":
        return np.roll(mask_L, rng.integers(0, len(mask_L)))

    if mode == "shuffle":
        return rng.permutation(mask_L)

    raise ValueError(f"Invalid null mode: {mode}")


# %% PLOTS AND ANALYSES


def plot_chemical_alignment_check(
    X_test: np.ndarray,
    *,
    n_analysis: int,
    global_roi_raw: np.ndarray,
    q_mass: float,
    plot_dir: Path,
    ppm_domain: np.ndarray,
    show: bool,
) -> np.ndarray:
    """Plot mean spectra, soft ROI, and hard ROI support."""
    fig, ax = plt.subplots(figsize=(25, 6))

    x_full = ppm_domain if ppm_domain is not None else np.arange(X_test.shape[1])
    mean_spectrum = np.mean(X_test[:n_analysis], axis=0)
    mean_plot = np.clip(mean_spectrum, a_min=MIN_LOG_INTENSITY, a_max=None)

    for spectrum in X_test[:n_analysis]:
        ax.plot(
            x_full,
            np.clip(spectrum, a_min=MIN_LOG_INTENSITY, a_max=None),
            color="0.85",
            alpha=0.01,
            linewidth=0.5,
        )

    ax.plot(
        x_full,
        mean_plot,
        label="Mean Spectrum",
        color="tab:blue",
        alpha=0.6,
        linewidth=1.2,
    )

    roi_scaled = (global_roi_raw / (global_roi_raw.max() + EPS)) * mean_plot.max()
    ax.plot(
        x_full,
        roi_scaled,
        label="Soft ROI Map",
        color="orange",
        linewidth=1.5,
        alpha=0.7,
    )

    hard_mask_full = hard_mask_from_soft_roi(global_roi_raw, q_mass=q_mass)

    ax.fill_between(
        x_full,
        MIN_LOG_INTENSITY,
        mean_plot.max(),
        where=(hard_mask_full == 1),
        color="tab:green",
        alpha=0.2,
        label="Hard ROI Mask",
    )

    ax.set_yscale("log")
    set_nmr_xaxis(ax)
    ax.set_ylim(bottom=MIN_LOG_INTENSITY, top=mean_plot.max() * 1.2)
    ax.set_title("Global Chemical ROI", fontsize=22, pad=10)
    ax.set_xlabel("Chemical Shift (ppm)")
    ax.set_ylabel("Intensity")
    ax.legend(loc="upper right", fontsize="small")

    save_figure(
        fig,
        plot_dir / "ROI_Check" / "chemical_alignment.png",
        show=show,
    )

    return hard_mask_full


def plot_attention_grid_enrichment(
    attn_vec: np.ndarray,
    X_test: np.ndarray,
    *,
    n_analysis: int,
    global_roi_raw: np.ndarray,
    global_mask_L: np.ndarray,
    plot_dir: Path,
    ppm_domain: np.ndarray,
    show: bool,
) -> None:
    """Plot per-head top-k occupancy and enrichment over the global ROI mask."""
    n_heads = attn_vec.shape[1]
    x_full = ppm_domain if ppm_domain is not None else np.arange(X_test.shape[1])

    mean_spectrum = np.mean(X_test[:n_analysis], axis=0)
    mean_plot = np.clip(mean_spectrum, a_min=MIN_LOG_INTENSITY, a_max=None)

    rng = np.random.default_rng(RANDOM_SEED)
    n_shade = min(150, n_analysis)
    indices_shade = rng.choice(n_analysis, n_shade, replace=False)

    roi_scaled = (global_roi_raw / (global_roi_raw.max() + EPS)) * mean_plot.max()

    ncols = 2
    nrows = int(np.ceil(n_heads / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(24, 4 * nrows),
        sharex=True,
        sharey=True,
    )

    axes_flat = np.ravel(axes)

    for h in range(n_heads):
        ax = axes_flat[h]

        for i in indices_shade:
            ax.plot(
                x_full,
                np.clip(X_test[i], a_min=MIN_LOG_INTENSITY, a_max=None),
                color="0.85",
                alpha=0.01,
                linewidth=0.5,
            )

        ax.plot(
            x_full,
            mean_plot,
            label="Mean Spectrum",
            color="tab:blue",
            alpha=0.6,
            linewidth=1.2,
        )
        ax.plot(
            x_full,
            roi_scaled,
            label="Global ROI Map",
            color="orange",
            linewidth=1.5,
            alpha=0.7,
        )

        occupancy = occupancy_topk(attn_vec[:, h, :], top_q=TOP_Q)
        occupancy_threshold = 0.10
        occupancy_filtered = np.where(occupancy >= occupancy_threshold, occupancy, 0.0)

        occupancy_full = scale_vector(occupancy_filtered, X_test.shape[1], kind="linear")
        occupancy_full = np.maximum(occupancy_full, 0.0)

        baseline_prob = float(global_mask_L.mean())
        weighted_occ = np.clip(occupancy_filtered, 0.0, None)
        weighted_occ = weighted_occ / (weighted_occ.sum() + EPS)

        hit_rate = float(weighted_occ[global_mask_L == 1].sum())
        enrichment = hit_rate / (baseline_prob + EPS)

        if occupancy_filtered.max() > 0:
            occupancy_scaled = (
                occupancy_full / (occupancy_filtered.max() + EPS)
            ) * mean_plot.max()
        else:
            occupancy_scaled = occupancy_full

        ax.fill_between(
            x_full,
            MIN_LOG_INTENSITY,
            occupancy_scaled,
            where=(occupancy_full > 0),
            color="red",
            alpha=0.2,
            # label=f"Head {h} Occupancy",
            label="Head Occupancy",
        )

        ax.set_yscale("log")
        set_nmr_xaxis(ax)
        ax.set_ylim(MIN_LOG_INTENSITY, mean_plot.max() * 2)

        if h % ncols == 0:
            ax.set_ylabel("Intensity", fontsize=16)
        if h >= (nrows - 1) * ncols:
            ax.set_xlabel("Chemical Shift (ppm)", fontsize=16)

        ax.text(
            0.02,
            0.85,
            f"Head {h}",
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

        badge_color = "lightgreen" if enrichment >= 3.9 else "lightgray"
        ax.text(
            0.02,
            0.72,
            f"Hit: {hit_rate:.1%} (Enrichment: {enrichment:.1f}x)",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            bbox=dict(facecolor=badge_color, alpha=0.9, edgecolor="gray"),
        )

        if h == 0:
            ax.legend(loc="upper right", fontsize="small")

    for h in range(n_heads, nrows * ncols):
        fig.delaxes(axes_flat[h])

    fig.suptitle(
        "Attention Density, Peak Hit Rate, and Enrichment Factor",
        fontsize=22,
        y=0.97,
    )
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.03, hspace=0.05)

    save_figure(
        fig,
        plot_dir / "ROI_Check" / "all_heads_grid_enrichment.png",
        show=show,
        tight=False,
    )


def run_topk_permutation_analysis(
    attn_vec_NHL: np.ndarray,
    mask_L: np.ndarray,
    *,
    mask_name: str,
    plot_dir: Path,
    top_q: float = TOP_Q,
    n_permutations: int = TOPK_PERMUTATIONS,
    seed: int = 123,
    null_mode: str = TOPK_NULL_MODE,
    show: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run top-k enrichment permutation analysis for each attention head."""
    rng = np.random.default_rng(seed)
    mask = (mask_L > 0).astype(int)

    baseline_prob = float(mask.mean())
    n_samples, n_heads, sequence_length = attn_vec_NHL.shape

    sample_rows = []
    head_rows = []

    for h in range(n_heads):
        hit_obs = topk_hit_rate_per_sample(attn_vec_NHL[:, h, :], mask, top_q=top_q)
        enrichment_obs = hit_obs / (baseline_prob + EPS)
        observed_mean = float(np.mean(enrichment_obs))

        null = np.empty(n_permutations, dtype=float)

        for b in range(n_permutations):
            permuted_mask = permute_mask(mask, rng, mode=null_mode)
            hit_null = topk_hit_rate_per_sample(
                attn_vec_NHL[:, h, :],
                permuted_mask,
                top_q=top_q,
            )
            null[b] = float(np.mean(hit_null / (permuted_mask.mean() + EPS)))

        null_mean = float(np.mean(null))
        null_std = float(np.std(null) + EPS)
        z_score = (observed_mean - null_mean) / null_std
        p_value = (np.sum(null >= observed_mean) + 1.0) / (n_permutations + 1.0)

        for value in enrichment_obs:
            sample_rows.append(
                {
                    "Head": f"Head {h}",
                    "Enrichment": float(value),
                }
            )

        head_rows.append(
            {
                "Head": f"Head {h}",
                "baseline_prob": baseline_prob,
                "k": int(np.ceil((1.0 - top_q) * sequence_length)),
                "obs_mean_enrichment": observed_mean,
                "null_mean": null_mean,
                "z": z_score,
                "p_one_sided": p_value,
                "neglog10p": -np.log10(p_value + 1e-300),
            }
        )

    df_samples = pd.DataFrame(sample_rows)
    df_head = pd.DataFrame(head_rows).sort_values("p_one_sided").reset_index(drop=True)

    print(f"\n[INFO] Per-head TOP-K summary for {mask_name}:")
    print(df_head)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(
        x="Head",
        y="Enrichment",
        data=df_samples,
        inner="box",
        cut=0,
        ax=ax,
    )
    ax.axhline(1.0, linestyle="--", alpha=0.7)
    ax.set_title(
        f"TOP-K enrichment by head (mask={mask_name}, top_q={top_q}, null={null_mode})",
        pad=10,
    )
    ax.set_ylabel("Enrichment Factor")
    ax.set_xlabel("Attention head")

    save_figure(
        fig,
        plot_dir / f"topk_{mask_name}_enrichment_violin_{null_mode}.png",
        show=show,
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(x="Head", y="neglog10p", data=df_head, ax=ax)
    ax.axhline(-np.log10(0.05), linestyle="--", alpha=0.7)

    ymax = df_head["neglog10p"].max()
    ax.set_ylim(0, ymax * 1.2 if ymax > 0 else 1.0)
    ax.set_title("TOP-K significance per head", pad=10)
    ax.set_ylabel("-log10(p)")
    ax.set_xlabel("Attention head")

    save_figure(
        fig,
        plot_dir / f"topk_{mask_name}_neglog10p_{null_mode}.png",
        show=show,
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        df_head["Head"],
        df_head["obs_mean_enrichment"],
        marker="o",
        label="Observed mean enrichment",
    )
    ax.plot(
        df_head["Head"],
        df_head["null_mean"],
        marker="o",
        label="Null mean",
    )
    ax.axhline(1.0, linestyle="--", alpha=0.7)

    for i, row in df_head.iterrows():
        ax.text(
            i,
            row["obs_mean_enrichment"] + 0.1,
            f"z={row['z']:.1f}",
            fontsize=16,
            ha="center",
            va="bottom",
        )

    ymax = max(df_head["obs_mean_enrichment"].max(), df_head["null_mean"].max())
    ax.set_ylim(0, ymax * 1.2 if ymax > 0 else 1.0)
    ax.set_title("TOP-K observed vs null mean enrichment", pad=10)
    ax.set_ylabel("Mean enrichment")
    ax.set_xlabel("Attention head")
    ax.legend()

    save_figure(
        fig,
        plot_dir / f"topk_{mask_name}_obs_vs_null_{null_mode}.png",
        show=show,
    )

    return df_samples, df_head


def run_delta_occupancy_analysis(
    attn_vec: np.ndarray,
    X_test: np.ndarray,
    *,
    n_analysis: int,
    labels: list[str],
    presence_mat: np.ndarray,
    roi_soft_raw: dict[str, np.ndarray],
    plot_dir: Path,
    ppm_domain: np.ndarray,
    show: bool,
) -> pd.DataFrame:
    """Run class-specific Delta Occupancy and Delta Enrichment analysis."""
    print("\n[INFO] Running Delta Occupancy analysis.")

    output_dir = plot_dir / "ROI_Check" / "DeltaOccupancy"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_heads = attn_vec.shape[1]
    sequence_length = attn_vec.shape[2]

    x_full = ppm_domain if ppm_domain is not None else np.arange(X_test.shape[1])
    mean_plot_global = np.clip(
        np.mean(X_test[:n_analysis], axis=0),
        a_min=MIN_LOG_INTENSITY,
        a_max=None,
    )

    rng = np.random.default_rng(RANDOM_SEED)
    delta_enrichment_mat = np.full((n_heads, len(labels)), np.nan, dtype=float)

    ncols = 2
    nrows = int(np.ceil(n_heads / ncols))

    for c_idx, copolymer in enumerate(labels):
        present_mask = presence_mat[:n_analysis, c_idx].astype(int)
        present_fraction = float(present_mask.mean())

        if (
            present_fraction < DELTA_MIN_PRESENT_FRACTION
            or present_fraction > DELTA_MAX_PRESENT_FRACTION
            or copolymer not in roi_soft_raw
        ):
            continue

        present_indices = np.where(present_mask == 1)[0]
        n_shade = min(150, len(present_indices))
        shade_indices = rng.choice(present_indices, n_shade, replace=False)

        mean_plot_specific = np.clip(
            np.mean(X_test[present_indices], axis=0),
            a_min=MIN_LOG_INTENSITY,
            a_max=None,
        )

        copolymer_mask_L = hard_mask_from_full_roi_at_attention_length(
            roi_soft_raw[copolymer],
            attention_length=sequence_length,
            q_mass=ROI_Q_MASS,
        )

        baseline_prob = float(copolymer_mask_L.mean())

        if baseline_prob < EPS:
            continue

        roi_c_norm = roi_soft_raw[copolymer] / (roi_soft_raw[copolymer].sum() + EPS)
        roi_c_full = np.clip(
            (roi_c_norm / (roi_c_norm.max() + EPS)) * mean_plot_global.max(),
            a_min=MIN_LOG_INTENSITY,
            a_max=None,
        )

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(24, 4 * nrows),
            sharex=True,
            sharey=True,
        )

        axes_flat = np.ravel(axes)

        for h in range(n_heads):
            ax = axes_flat[h]

            delta, _, _ = delta_occupancy(
                attn_vec[:n_analysis, h, :],
                present_mask,
                top_q=TOP_Q,
                smooth_sigma=DELTA_SMOOTH_SIGMA,
            )

            if delta is None:
                continue

            delta_positive = np.maximum(delta, 0.0)
            positive_mass = float(delta_positive.sum())

            if positive_mass < EPS:
                enrichment = np.nan
            else:
                enrichment = (
                    float(
                        (delta_positive / (positive_mass + EPS))[
                            copolymer_mask_L == 1
                        ].sum()
                    )
                    / (baseline_prob + EPS)
                )

            delta_enrichment_mat[h, c_idx] = enrichment

            if np.any(delta_positive > 0):
                threshold = np.quantile(delta_positive[delta_positive > 0], 0.75)
                delta_filtered = np.where(delta_positive >= threshold, delta_positive, 0.0)
            else:
                delta_filtered = np.zeros_like(delta_positive)

            delta_full = scale_vector(delta_filtered, X_test.shape[1], kind="linear")
            delta_full = np.maximum(delta_full, 0.0)

            if delta_filtered.max() > 0:
                delta_scaled = (delta_full / (delta_filtered.max() + EPS)) * mean_plot_global.max()
            else:
                delta_scaled = delta_full

            for i in shade_indices:
                ax.plot(
                    x_full,
                    np.clip(X_test[i], a_min=MIN_LOG_INTENSITY, a_max=None),
                    color="0.85",
                    alpha=0.01,
                    linewidth=0.5,
                )

            ax.plot(
                x_full,
                mean_plot_specific,
                color="tab:blue",
                alpha=0.6,
                linewidth=1.2,
                label=f"Mean Spectrum (with {mapped_label(copolymer)})",
            )
            ax.plot(
                x_full,
                roi_c_full,
                color="goldenrod",
                linewidth=1.5,
                alpha=0.8,
                label=f"Soft ROI ({mapped_label(copolymer)})",
            )
            ax.fill_between(
                x_full,
                MIN_LOG_INTENSITY,
                np.clip(delta_scaled, a_min=MIN_LOG_INTENSITY, a_max=None),
                where=(delta_full > 0),
                color="red",
                alpha=0.20,
                label="Delta Occupancy",
            )

            ax.set_yscale("log")
            set_nmr_xaxis(ax)
            ax.set_ylim(MIN_LOG_INTENSITY, mean_plot_global.max() * 2)

            if h % ncols == 0:
                ax.set_ylabel("Intensity (log)", fontsize=16)
            if h >= (nrows - 1) * ncols:
                ax.set_xlabel("Chemical Shift (ppm)", fontsize=16)

            ax.text(
                0.02,
                0.85,
                f"Head {h}",
                transform=ax.transAxes,
                fontsize=16,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )
            ax.text(
                0.02,
                0.72,
                (
                    f"Delta Enrichment: {enrichment:.1f}x"
                    if np.isfinite(enrichment)
                    else "Delta Enrichment: NA"
                ),
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                bbox=dict(
                    facecolor=(
                        "lightgreen"
                        if np.isfinite(enrichment) and enrichment >= 3.0
                        else "lightgray"
                    ),
                    alpha=0.9,
                    edgecolor="gray",
                ),
            )

            if h == 0:
                ax.legend(loc="upper right", fontsize="small")

        for h in range(n_heads, nrows * ncols):
            fig.delaxes(axes_flat[h])

        fig.suptitle(
            f"Attention Reactivity to {mapped_label(copolymer)}",
            fontsize=22,
            y=0.97,
        )
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.03, hspace=0.05)

        save_figure(
            fig,
            output_dir / f"delta_occ_{copolymer}_mosaic.png",
            show=show,
            tight=False,
        )

    heat_df = pd.DataFrame(
        delta_enrichment_mat,
        index=[f"Head {h}" for h in range(n_heads)],
        columns=mapped_labels(labels),
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="mako", vmin=1.0, ax=ax)
    ax.set_title("Delta Occupancy Enrichment Factor", pad=10)
    ax.set_xlabel("Polyolefin subclass")
    ax.set_ylabel("Head")
    ax.tick_params(axis="x", rotation=45)

    save_figure(
        fig,
        output_dir / "delta_occ_heatmap_enrichment.png",
        show=show,
    )

    return heat_df


def run_head_redundancy_analysis(
    attn_vec_NHL: np.ndarray,
    *,
    plot_dir: Path,
    show: bool,
) -> np.ndarray:
    """Compute Jensen-Shannon distance between attention heads."""
    print("\n[INFO] Running head redundancy analysis with Jensen-Shannon distance.")

    _, n_heads, _ = attn_vec_NHL.shape

    attn_prob = np.maximum(attn_vec_NHL, 0.0)
    attn_prob = attn_prob / (attn_prob.sum(axis=2, keepdims=True) + EPS)

    jsd_matrix = np.zeros((n_heads, n_heads), dtype=float)

    for i in range(n_heads):
        for j in range(i, n_heads):
            if i == j:
                jsd_matrix[i, j] = 0.0
                continue

            dist_ij = jensenshannon(attn_prob[:, i, :], attn_prob[:, j, :], axis=1)
            mean_dist = float(np.nanmean(dist_ij))

            jsd_matrix[i, j] = mean_dist
            jsd_matrix[j, i] = mean_dist

    head_labels = [f"Head {h}" for h in range(n_heads)]
    heat_df = pd.DataFrame(jsd_matrix, index=head_labels, columns=head_labels)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="Blues", vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title("Head Redundancy (Jensen-Shannon Distance)", pad=10)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    save_figure(
        fig,
        plot_dir / "head_redundancy_JSD.png",
        show=show,
    )

    upper_triangle = np.triu_indices_from(jsd_matrix, k=1)
    mean_jsd = float(np.mean(jsd_matrix[upper_triangle]))

    print(f"[INFO] Mean JSD between different heads: {mean_jsd:.4f}")

    return jsd_matrix


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with constant-vector protection."""
    if np.std(x) < EPS or np.std(y) < EPS:
        return np.nan

    r, _ = pearsonr(x, y)
    return float(r)


def loo_template_correlations(maps: np.ndarray) -> np.ndarray:
    """Compute leave-one-out correlations with the class template."""
    maps = normalize_rows(maps)
    n_maps = maps.shape[0]

    if n_maps < 2:
        return np.array([], dtype=float)

    sum_map = np.sum(maps, axis=0)
    correlations = []

    for i in range(n_maps):
        sample_map = maps[i]
        template = (sum_map - sample_map) / max(n_maps - 1, 1)
        template = np.maximum(template, 0.0)
        template = template / (template.sum() + EPS)

        correlations.append(safe_pearson(sample_map, template))

    return np.asarray(correlations, dtype=float)


def run_class_wise_consistency_contrastive(
    attn_vec_NHL: np.ndarray,
    presence_mat: np.ndarray,
    labels: list[str],
    *,
    plot_dir: Path,
    min_samples: int = 9,
    show: bool,
) -> dict[str, pd.DataFrame]:
    """Compute leave-one-out class-wise consistency for present and absent samples."""
    print("\n[INFO] Running class-wise consistency analysis.")

    n_samples, n_heads, _ = attn_vec_NHL.shape

    present_mean = np.full((n_heads, len(labels)), np.nan, dtype=float)
    absent_mean = np.full((n_heads, len(labels)), np.nan, dtype=float)
    delta_mean = np.full((n_heads, len(labels)), np.nan, dtype=float)

    present_std = np.full((n_heads, len(labels)), np.nan, dtype=float)
    absent_std = np.full((n_heads, len(labels)), np.nan, dtype=float)

    n_present_arr = np.zeros(len(labels), dtype=int)
    n_absent_arr = np.zeros(len(labels), dtype=int)

    summary_rows = []

    for c_idx, copolymer in enumerate(labels):
        idx_present = np.where(presence_mat[:, c_idx] == 1)[0]
        idx_absent = np.where(presence_mat[:, c_idx] == 0)[0]

        n_present_arr[c_idx] = len(idx_present)
        n_absent_arr[c_idx] = len(idx_absent)

        if len(idx_present) < min_samples:
            print(
                f"[WARNING] {copolymer}: too few present samples "
                f"({len(idx_present)}); present consistency skipped."
            )

        if len(idx_absent) < min_samples:
            print(
                f"[WARNING] {copolymer}: too few absent samples "
                f"({len(idx_absent)}); absent consistency skipped."
            )

        for h in range(n_heads):
            correlations_present = np.array([], dtype=float)

            if len(idx_present) >= min_samples:
                maps_present = attn_vec_NHL[idx_present, h, :]
                correlations_present = loo_template_correlations(maps_present)

                if len(correlations_present) > 0:
                    present_mean[h, c_idx] = np.nanmean(correlations_present)
                    present_std[h, c_idx] = np.nanstd(correlations_present)

            correlations_absent = np.array([], dtype=float)

            if len(idx_absent) >= min_samples:
                maps_absent = attn_vec_NHL[idx_absent, h, :]
                correlations_absent = loo_template_correlations(maps_absent)

                if len(correlations_absent) > 0:
                    absent_mean[h, c_idx] = np.nanmean(correlations_absent)
                    absent_std[h, c_idx] = np.nanstd(correlations_absent)

            if np.isfinite(present_mean[h, c_idx]) and np.isfinite(absent_mean[h, c_idx]):
                delta_mean[h, c_idx] = present_mean[h, c_idx] - absent_mean[h, c_idx]

            summary_rows.append(
                {
                    "Copolymer": mapped_label(copolymer),
                    "Head": f"Head {h}",
                    "n_present": int(len(idx_present)),
                    "n_absent": int(len(idx_absent)),
                    "consistency_present_mean": present_mean[h, c_idx],
                    "consistency_present_std": present_std[h, c_idx],
                    "consistency_absent_mean": absent_mean[h, c_idx],
                    "consistency_absent_std": absent_std[h, c_idx],
                    "delta_consistency": delta_mean[h, c_idx],
                }
            )

    head_index = [f"Head {h}" for h in range(n_heads)]

    df_present = pd.DataFrame(
        present_mean,
        index=head_index,
        columns=mapped_labels(labels),
    )
    df_absent = pd.DataFrame(
        absent_mean,
        index=head_index,
        columns=mapped_labels(labels),
    )
    df_delta = pd.DataFrame(
        delta_mean,
        index=head_index,
        columns=mapped_labels(labels),
    )
    df_counts = pd.DataFrame(
        {
            "Copolymer": mapped_labels(labels),
            "n_present": n_present_arr,
            "n_absent": n_absent_arr,
        }
    )
    df_summary = pd.DataFrame(summary_rows)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(df_present, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
    ax.set_title("Consistency (Present)", pad=10)
    # ax.set_xlabel("Polyolefin (sub-)class")
    # ax.set_ylabel("Attention Head")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    save_figure(
        fig,
        plot_dir / "head_consistency_per_class_contrastive_present.png",
        show=show,
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(df_absent, annot=True, fmt=".3f", cmap="YlOrBr", vmin=0, vmax=1, ax=ax)
    ax.set_title("Consistency (Absent)", pad=10)
    # ax.set_xlabel("Polyolefin (sub-)class")
    # ax.set_ylabel("Attention Head")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    save_figure(
        fig,
        plot_dir / "head_consistency_per_class_contrastive_absent.png",
        show=show,
    )

    vmax_abs = np.nanmax(np.abs(df_delta.values))

    if not np.isfinite(vmax_abs) or vmax_abs < 1e-6:
        vmax_abs = 1.0

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        df_delta,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0.0,
        vmin=-1.2 * vmax_abs,
        vmax=1.2 * vmax_abs,
        ax=ax,
    )
    ax.set_title("Consistency Gain", pad=10)
    # ax.set_xlabel("Polyolefin (sub-)class")
    # ax.set_ylabel("Attention Head")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    save_figure(
        fig,
        plot_dir / "head_consistency_per_class_contrastive_delta.png",
        show=show,
    )

    print("\n[INFO] Top positive Delta-consistency entries:")
    print(
        df_summary.sort_values("delta_consistency", ascending=False)[
            [
                "Copolymer",
                "Head",
                "n_present",
                "n_absent",
                "consistency_present_mean",
                "consistency_absent_mean",
                "delta_consistency",
            ]
        ].head(15)
    )

    return {
        "present": df_present,
        "absent": df_absent,
        "delta": df_delta,
        "counts": df_counts,
        "summary_long": df_summary,
    }


# %% INTEGRATED GRADIENTS


def get_integrated_gradients(
    model,
    inputs: tf.Tensor,
    target_class_idx: int,
    *,
    m_steps: int = IG_STEPS,
    batch_size: int = IG_BATCH_SIZE,
    desc: str = "Integrated Gradients",
) -> np.ndarray:
    """Compute Integrated Gradients in mini-batches."""
    total_samples = inputs.shape[0]
    integrated_gradients_all = []

    for start_idx in tqdm(
        range(0, total_samples, batch_size),
        desc=desc,
        leave=False,
    ):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_inputs = inputs[start_idx:end_idx]

        baseline = tf.zeros_like(batch_inputs)
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]

        baseline_x = tf.expand_dims(baseline, 0)
        inputs_x = tf.expand_dims(batch_inputs, 0)

        interpolated_path = baseline_x + alphas_x * (inputs_x - baseline_x)

        gradients_list = []

        for step_idx in range(m_steps + 1):
            step_inputs = interpolated_path[step_idx]

            with tf.GradientTape() as tape:
                tape.watch(step_inputs)
                predictions = model(step_inputs, training=False)

                if isinstance(predictions, dict) and "weight_output" in predictions:
                    target_prediction = predictions["weight_output"][:, target_class_idx]
                else:
                    target_prediction = predictions[:, target_class_idx]

            gradients = tape.gradient(target_prediction, step_inputs)
            gradients_list.append(gradients)

        avg_gradients = tf.reduce_mean(tf.stack(gradients_list), axis=0)
        integrated_gradients = (batch_inputs - baseline) * avg_gradients
        integrated_gradients_all.append(integrated_gradients.numpy().squeeze(-1))

    return np.concatenate(integrated_gradients_all, axis=0)


def run_integrated_gradients_benchmark(
    model,
    X_test: np.ndarray,
    attn_vec_NHL: np.ndarray,
    presence_mat: np.ndarray,
    labels: list[str],
    *,
    plot_dir: Path,
    n_samples: int = -1,
    show: bool,
) -> pd.DataFrame:
    """Compare native attention with Integrated Gradients using Pearson correlation."""
    print("\n[INFO] Running Integrated Gradients benchmark.")

    if n_samples == -1 or n_samples > len(X_test):
        n_samples = len(X_test)

    X_sub = X_test[:n_samples]
    attn_sub = attn_vec_NHL[:n_samples]

    _, n_heads, _ = attn_sub.shape
    full_length = X_sub.shape[1]

    rows = []

    for target_idx, target_name in enumerate(labels):
        valid_indices = np.where(presence_mat[:n_samples, target_idx] == 1)[0]

        if len(valid_indices) == 0:
            print(f"[WARNING] {target_name}: no valid samples for IG benchmark.")
            continue

        print(
            f"[INFO] Computing IG for {target_name} "
            f"on {len(valid_indices)} valid samples."
        )

        X_valid = X_sub[valid_indices]

        ig_attributions = get_integrated_gradients(
            model,
            tf.convert_to_tensor(X_valid[..., np.newaxis], dtype=tf.float32),
            target_class_idx=target_idx,
            m_steps=IG_STEPS,
            batch_size=IG_BATCH_SIZE,
            desc=f"IG {target_name}",
        )

        ig_abs = np.abs(ig_attributions)

        for local_idx, original_idx in enumerate(valid_indices):
            ig_signal = ig_abs[local_idx]

            if np.max(ig_signal) == 0.0 or np.std(ig_signal) < EPS:
                continue

            ig_signal = ig_signal / (ig_signal.sum() + EPS)

            for h in range(n_heads):
                attn_signal = attn_sub[original_idx, h, :]
                attn_full = scale_vector(attn_signal, full_length, kind="linear")
                attn_full = np.maximum(attn_full, 0.0)

                if np.max(attn_full) == 0.0 or np.std(attn_full) < EPS:
                    continue

                attn_full = attn_full / (attn_full.sum() + EPS)

                r = safe_pearson(ig_signal, attn_full)

                rows.append(
                    {
                        "Sample": int(original_idx),
                        "Target": mapped_label(target_name),
                        "Head": f"Head {h}",
                        "Pearson_R": r,
                    }
                )

    df_results = pd.DataFrame(rows)

    if df_results.empty:
        print("[WARNING] No valid IG benchmark results were generated.")
        return df_results

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_results, x="Target", y="Pearson_R", hue="Head", ax=ax)
    ax.axhline(0.0, linestyle="--", color="gray")
    ax.set_title("Spatial Correlation: Native Attention vs Integrated Gradients")
    ax.set_ylabel("Pearson Correlation Coefficient ($r$)")
    ax.set_xlabel("Polyolefin subclass")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    save_figure(
        fig,
        plot_dir / "XAI_benchmark_IG_vs_Attention.png",
        show=show,
        dpi=300,
    )

    print("\n[INFO] Mean IG-attention correlation by head:")
    print(df_results.groupby("Head")["Pearson_R"].mean())

    return df_results


# %% PRESENCE MATRICES


def predict_presence_matrix(
    model,
    X_test: np.ndarray,
    *,
    n_analysis: int,
    batch_size: int,
    presence_threshold: float,
) -> np.ndarray:
    """Predict class presence from model weight outputs."""
    print("\n[INFO] Predicting class presence from model weights.")

    predictions = model.predict(
        X_test[:n_analysis, ..., np.newaxis],
        batch_size=batch_size,
    )

    y_w_pred = predictions["weight_output"]
    return (y_w_pred > presence_threshold).astype(int)


def ground_truth_presence_matrix(
    test_data: pd.DataFrame,
    labels: list[str],
    *,
    n_analysis: int,
) -> np.ndarray | None:
    """Build ground-truth presence matrix from copo_tuple and w columns."""
    if "copo_tuple" not in test_data.columns or "w" not in test_data.columns:
        return None

    gt_weights = np.zeros((n_analysis, len(labels)), dtype=float)

    for i, idx in enumerate(test_data.index[:n_analysis]):
        copolymers = parse_tuple_like(test_data.loc[idx, "copo_tuple"])
        weights = parse_tuple_like(test_data.loc[idx, "w"])

        for copolymer, weight in zip(copolymers, weights):
            if copolymer in labels:
                gt_weights[i, labels.index(copolymer)] = float(weight)

    return (gt_weights > GT_PRESENCE_THRESHOLD).astype(int)


def get_presence_matrix(
    model,
    test_data: pd.DataFrame,
    X_test: np.ndarray,
    labels: list[str],
    *,
    n_analysis: int,
    batch_size: int,
    predicted_presence_threshold: float,
) -> tuple[np.ndarray, str]:
    """Use GT presence if available, otherwise predicted presence."""
    gt_presence = ground_truth_presence_matrix(
        test_data,
        labels,
        n_analysis=n_analysis,
    )

    if gt_presence is not None:
        print("[INFO] Presence source: ground truth tuples.")
        return gt_presence, "GT"

    pred_presence = predict_presence_matrix(
        model,
        X_test,
        n_analysis=n_analysis,
        batch_size=batch_size,
        presence_threshold=predicted_presence_threshold,
    )

    print("[INFO] Presence source: model predictions.")
    return pred_presence, "PRED"


# %% MAIN


def main() -> None:
    args = parse_args()
    print(args)

    plot_dir = get_plot_dir(
        model_name=args.model_name,
        suffix=args.suffix,
        output_dir=args.output_dir,
    )

    ppm_domain = load_ppm_domain(args.synth_ppm_domain)

    test_data, X_test = load_test_data(args)
    model = load_model(args, X_test)

    n_analysis = min(len(X_test), args.n_samples_analysis)

    roi_soft_raw, roi_info = build_class_rois(
        test_data,
        X_test,
        LABELS,
        snr_k=args.snr_k,
        roi_sigma_points=args.roi_sigma_points,
    )

    global_roi_raw = build_global_roi(roi_soft_raw)

    attn_vec = extract_attention_tensor(
        model,
        X_test,
        n_samples=n_analysis,
        batch_size=args.batch_size,
    )

    n_heads = attn_vec.shape[1]
    sequence_length = attn_vec.shape[2]

    global_roi_L = scale_vector(global_roi_raw, sequence_length, kind="linear")
    global_roi_L = np.maximum(global_roi_L, 0.0)
    global_roi_L = global_roi_L / (global_roi_L.sum() + EPS)

    global_mask_L = hard_mask_from_soft_roi(global_roi_L, q_mass=ROI_Q_MASS)
    roi_indices = np.where(global_mask_L == 1)[0]

    print(
        f"[INFO] Peak-support size: {len(roi_indices)} / {len(global_mask_L)} "
        f"({len(roi_indices) / len(global_mask_L):.2%})"
    )

    plot_chemical_alignment_check(
        X_test,
        n_analysis=n_analysis,
        global_roi_raw=global_roi_raw,
        q_mass=ROI_Q_MASS,
        plot_dir=plot_dir,
        ppm_domain=ppm_domain,
        show=args.show,
    )

    for h in range(n_heads):
        if len(roi_indices) > 2:
            mean_std = np.mean(
                [np.std(attn_vec[i, h, roi_indices]) for i in range(n_analysis)]
            )
        else:
            mean_std = 0.0

        print(f"[INFO] Head {h} mean std in ROI: {mean_std:.6e}")

    plot_attention_grid_enrichment(
        attn_vec,
        X_test,
        n_analysis=n_analysis,
        global_roi_raw=global_roi_raw,
        global_mask_L=global_mask_L,
        plot_dir=plot_dir,
        ppm_domain=ppm_domain,
        show=args.show,
    )

    presence_mat, presence_source = get_presence_matrix(
        model,
        test_data,
        X_test,
        LABELS,
        n_analysis=n_analysis,
        batch_size=args.batch_size,
        predicted_presence_threshold=args.presence_thr,
    )

    print(f"[INFO] Presence matrix source used downstream: {presence_source}")

    run_delta_occupancy_analysis(
        attn_vec,
        X_test,
        n_analysis=n_analysis,
        labels=LABELS,
        presence_mat=presence_mat,
        roi_soft_raw=roi_soft_raw,
        plot_dir=plot_dir,
        ppm_domain=ppm_domain,
        show=args.show,
    )

    run_topk_permutation_analysis(
        attn_vec_NHL=attn_vec[:n_analysis],
        mask_L=global_mask_L.astype(int),
        mask_name="global_peaks",
        plot_dir=plot_dir,
        top_q=TOP_Q,
        n_permutations=TOPK_PERMUTATIONS,
        seed=123,
        null_mode=TOPK_NULL_MODE,
        show=args.show,
    )

    run_class_wise_consistency_contrastive(
        attn_vec_NHL=attn_vec[:n_analysis],
        presence_mat=presence_mat,
        labels=LABELS,
        plot_dir=plot_dir,
        show=args.show,
    )

    run_head_redundancy_analysis(
        attn_vec[:n_analysis],
        plot_dir=plot_dir,
        show=args.show,
    )

    run_integrated_gradients_benchmark(
        model,
        X_test[:n_analysis],
        attn_vec[:n_analysis],
        presence_mat,
        LABELS,
        plot_dir=plot_dir,
        n_samples=args.ig_samples,
        show=args.show,
    )

    print(f"[INFO] Attention analysis completed. Output directory: {plot_dir}")


if __name__ == "__main__":
    main()