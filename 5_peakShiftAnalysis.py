#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Experimental-reference peak-shift analysis for DeepNMR spectra.

This script does not use externally supplied peak anchors such as
legit_peaks.json. Reference peaks are derived directly from experimental
monomaterial spectra by:

  1. detecting peaks in pure experimental spectra;
  2. clustering recurrent peak positions within each polymer class;
  3. using the median ppm position of each recurrent cluster as experimental reference.

The resulting experimental reference peaks are then used to quantify relative
peak shifts in synthetic, fine-tuning, and experimental spectra.

Generated plots:
  - 1_Global_Peak_Shift_Violin.png
  - 2_Global_Peak_Shift_KDE.png
  - 3_Pure_Polymers_Peak_Shift_Boxplot.png
  - 4_Pure_Polymers_Peak_Shift_KDE_FacetGrid.png
  - 5_Peak_by_Peak_Shift_<polymer>.png
  - 5B_Bubble_Plot_<polymer>.png

A JSON file containing the experimentally derived reference peaks is also saved:
  - experimental_reference_peaks.json

Suffix convention
-----------------
Use --suffix without leading underscore, e.g.:

  --suffix ftc_o2_chem

Internally, the suffix is normalized to '_ftc_o2_chem', producing artifact keys
such as:

  X_ftc_o2_chem
  y_w_ftc_o2_chem

and output folders such as:

  OUTPUT/peak_shift/<synth_tag>/plots_ftc_o2_chem/
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.signal import find_peaks
from adjustText import adjust_text

from utils import MappingNames

# %% FIXED CONFIGURATION

LABELS = ["LDPE", "PE", "PP", "EH", "EO", "EB", "RACO", "EPR"]
DISPLAY_NAMES = MappingNames()

FT_FOLDER = "ft_sets"

TOL_PPM = 0.05
REFERENCE_CLUSTER_TOL_PPM = 0.025
REFERENCE_MIN_FRACTION = 0.75
REFERENCE_MIN_COUNT = 2

PRESENCE_THRESHOLD = 1e-2
NOISE_WINDOW_POINTS = 2500
NOISE_SIGMA_MULTIPLIER = 6.0
SUBSAMPLING_SEED = 42

DATASET_ORDER = ["Synthetic", "Fine-Tuning", "Experimental"]


# %% ARGUMENTS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Peak-shift analysis using experimentally derived reference peaks.",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--synth_data",
        type=str,
        default="DATASET/synthetic_dataset.pkl",
        help="Synthetic dataset to be analyzed.",
    )
    parser.add_argument(
        "--synth_ppm_domain",
        type=str,
        default="DATASET/ppm_domain_general.pkl",
        help="PPM-domain file associated with the synthetic spectra.",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="DATASET/test_data.pkl",
        help="Experimental test dataset to be analyzed.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=(
            "model_weights_kl_mse_loss_composition_neg2_mse_hybrid"
        ),
        help="Model name used to locate the fine-tuning dataset under ft_sets/.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="ft",
        help=(
            "Suffix identifying the fine-tuning arrays, without leading underscore, "
            "e.g. 'ft'. Input with leading underscore is also accepted."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Optional output directory. If omitted, uses "
            "OUTPUT/peak_shift/<synthetic-dataset-tag>/plots<suffix>."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Maximum number of samples per dataset used for shift extraction.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively in addition to saving them.",
    )

    args, _ = parser.parse_known_args()
    return args


# %% PATH AND SUFFIX UTILITIES


def normalize_suffix(suffix: str | None) -> str:
    """Normalize suffix so that non-empty suffixes start with '_'."""
    if suffix is None:
        return ""

    suffix = str(suffix).strip().strip("_")

    if suffix == "":
        return ""

    return f"_{suffix}"


def artifact_key(prefix: str, suffix: str | None) -> str:
    """Build artifact keys using the common suffix convention."""
    suffix = normalize_suffix(suffix)
    return f"{prefix}{suffix}"


def get_plot_dir(
    synth_data: str,
    suffix: str | None,
    output_dir: str | None = None,
) -> str:
    """Return plot directory using the common suffix convention."""
    if output_dir is not None:
        return output_dir

    synth_tag = Path(synth_data).stem.split("_")[-1]
    suffix = normalize_suffix(suffix)

    return f"OUTPUT/peak_shift/{synth_tag}/plots{suffix}"


# %% LOADING UTILITIES


def load_artifact_folder(folder: str | Path) -> dict[str, Any]:
    """Load all .npy and .pkl artifacts from a folder into a dictionary."""
    folder = Path(folder)
    artifacts: dict[str, Any] = {}

    if not folder.exists():
        raise FileNotFoundError(f"Fine-tuning artifact folder not found: {folder}")

    for path in folder.iterdir():
        if path.suffix == ".npy":
            artifacts[path.stem] = np.load(path)
        elif path.suffix == ".pkl":
            artifacts[path.stem] = pd.read_pickle(path)

    return artifacts


def require_artifact(artifacts: dict[str, Any], key: str) -> Any:
    """Return an artifact or raise a helpful error."""
    if key not in artifacts:
        available = ", ".join(sorted(artifacts.keys()))
        raise KeyError(
            f"Required artifact '{key}' was not found. "
            f"Available artifacts are: {available}"
        )

    return artifacts[key]


def load_ppm_domain(path: str | Path) -> np.ndarray:
    """Load ppm domain as a NumPy array."""
    ppm_domain = pd.read_pickle(path)

    if isinstance(ppm_domain, pd.Series):
        return ppm_domain.values

    return np.asarray(ppm_domain)


# %% PEAK DETECTION AND REFERENCES


def get_active_polymers(
    annotation: Any,
    labels: list[str],
    presence_threshold: float = PRESENCE_THRESHOLD,
) -> tuple[str, ...]:
    """Extract active polymer names from either tuple labels or weight vectors."""
    if isinstance(annotation, tuple):
        return tuple(str(value) for value in annotation)

    if isinstance(annotation, list) and all(
        isinstance(value, str) for value in annotation
    ):
        return tuple(annotation)

    weights = np.asarray(annotation, dtype=float)
    active_indices = np.where((weights > presence_threshold) & ~pd.isna(weights))[0]

    return tuple(labels[idx] for idx in active_indices)


def detect_peak_positions(
    spectrum: np.ndarray,
    ppm_domain: np.ndarray,
) -> np.ndarray:
    """Detect peak positions in ppm using an adaptive noise-based threshold."""
    spectrum = np.asarray(spectrum, dtype=float)

    noise_window = min(NOISE_WINDOW_POINTS, len(spectrum))
    noise_region = spectrum[:noise_window]

    noise_mean = np.mean(noise_region)
    noise_std = np.std(noise_region)

    height_threshold = noise_mean + NOISE_SIGMA_MULTIPLIER * noise_std
    prominence_threshold = NOISE_SIGMA_MULTIPLIER * noise_std

    peak_indices, _ = find_peaks(
        spectrum,
        height=height_threshold,
        prominence=prominence_threshold,
    )

    return ppm_domain[peak_indices]


def cluster_peak_positions(
    peak_records: list[tuple[float, int]],
    tolerance_ppm: float,
    min_count: int,
) -> list[dict[str, Any]]:
    """Cluster detected peak positions using a simple tolerance-based rule.

    Parameters
    ----------
    peak_records:
        List of (ppm_position, spectrum_id) pairs.
    tolerance_ppm:
        Maximum distance from the current cluster median for assignment.
    min_count:
        Minimum number of spectra contributing to a cluster.
    """
    if len(peak_records) == 0:
        return []

    peak_records = sorted(peak_records, key=lambda item: item[0])

    clusters: list[list[tuple[float, int]]] = []

    for ppm_value, spectrum_id in peak_records:
        if len(clusters) == 0:
            clusters.append([(ppm_value, spectrum_id)])
            continue

        current_values = np.array([record[0] for record in clusters[-1]], dtype=float)
        current_median = float(np.median(current_values))

        if abs(ppm_value - current_median) <= tolerance_ppm:
            clusters[-1].append((ppm_value, spectrum_id))
        else:
            clusters.append([(ppm_value, spectrum_id)])

    reference_clusters = []

    for cluster in clusters:
        values = np.array([record[0] for record in cluster], dtype=float)
        spectrum_ids = {record[1] for record in cluster}

        if len(spectrum_ids) < min_count:
            continue

        reference_clusters.append(
            {
                "ppm": float(np.median(values)),
                "mean_ppm": float(np.mean(values)),
                "std_ppm": float(np.std(values)),
                "n_detections": int(len(values)),
                "n_spectra": int(len(spectrum_ids)),
            }
        )

    reference_clusters = sorted(
        reference_clusters,
        key=lambda cluster: cluster["ppm"],
        reverse=True,
    )

    return reference_clusters


def build_experimental_reference_peaks(
    spectra: np.ndarray,
    labels_matrix: np.ndarray,
    ppm_domain: np.ndarray,
    labels: list[str],
) -> dict[str, dict[str, Any]]:
    """Build experimental reference peaks directly from pure experimental spectra."""
    print("\n[INFO] Building experimental reference peaks from monomaterial spectra.")

    peak_records_by_polymer: dict[str, list[tuple[float, int]]] = {
        label: [] for label in labels
    }
    pure_counts_by_polymer: dict[str, int] = {label: 0 for label in labels}

    for i in tqdm(range(len(spectra)), desc="Detecting experimental reference peaks"):
        active_polymers = get_active_polymers(labels_matrix[i], labels)

        if len(active_polymers) != 1:
            continue

        polymer = active_polymers[0]

        if polymer not in peak_records_by_polymer:
            continue

        pure_counts_by_polymer[polymer] += 1

        detected_ppm = detect_peak_positions(
            spectrum=spectra[i],
            ppm_domain=ppm_domain,
        )

        for ppm_value in detected_ppm:
            peak_records_by_polymer[polymer].append((float(ppm_value), i))

    experimental_references: dict[str, dict[str, Any]] = {}

    for polymer in labels:
        n_pure = pure_counts_by_polymer[polymer]

        if n_pure == 0:
            experimental_references[polymer] = {
                "reference_peaks": [],
                "clusters": [],
                "n_pure_spectra": 0,
                "min_spectra_required": 0,
            }
            continue

        min_required = int(np.ceil(REFERENCE_MIN_FRACTION * n_pure))
        min_required = max(1, min(min_required, n_pure))

        if n_pure >= REFERENCE_MIN_COUNT:
            min_required = max(min_required, REFERENCE_MIN_COUNT)

        clusters = cluster_peak_positions(
            peak_records=peak_records_by_polymer[polymer],
            tolerance_ppm=REFERENCE_CLUSTER_TOL_PPM,
            min_count=min_required,
        )

        reference_peaks = [cluster["ppm"] for cluster in clusters]

        experimental_references[polymer] = {
            "reference_peaks": reference_peaks,
            "clusters": clusters,
            "n_pure_spectra": int(n_pure),
            "min_spectra_required": int(min_required),
        }

        print(
            f"[INFO] {polymer}: {len(reference_peaks)} reference peaks "
            f"from {n_pure} pure spectra "
            f"(min spectra per peak: {min_required})."
        )

    return experimental_references


def get_reference_peak_array(
    polymer: str,
    experimental_references: dict[str, dict[str, Any]],
) -> np.ndarray:
    """Return experimentally derived reference peak positions for one polymer."""
    if polymer not in experimental_references:
        return np.array([], dtype=float)

    return np.asarray(
        experimental_references[polymer].get("reference_peaks", []),
        dtype=float,
    )


def mutual_nearest_matches(
    reference_ppm: np.ndarray,
    detected_ppm: np.ndarray,
    tolerance_ppm: float,
) -> list[tuple[float, float]]:
    """Match reference and detected peaks using a mutual-nearest-neighbor rule."""
    matches: list[tuple[float, float]] = []

    if len(reference_ppm) == 0 or len(detected_ppm) == 0:
        return matches

    reference_ppm = np.asarray(reference_ppm, dtype=float)
    detected_ppm = np.asarray(detected_ppm, dtype=float)

    for ref_value in reference_ppm:
        detected_idx = np.argmin(np.abs(detected_ppm - ref_value))
        best_detected = detected_ppm[detected_idx]

        reciprocal_idx = np.argmin(np.abs(reference_ppm - best_detected))
        reciprocal_ref = reference_ppm[reciprocal_idx]

        is_mutual = np.isclose(ref_value, reciprocal_ref, rtol=0.0, atol=1e-12)
        is_within_tolerance = abs(best_detected - ref_value) <= tolerance_ppm

        if is_mutual and is_within_tolerance:
            matches.append((float(ref_value), float(best_detected)))

    return matches


def extract_peak_shifts(
    spectra: np.ndarray,
    labels_matrix: np.ndarray,
    ppm_domain: np.ndarray,
    experimental_references: dict[str, dict[str, Any]],
    labels: list[str],
    dataset_name: str,
    max_samples: int,
) -> pd.DataFrame:
    """Extract relative peak shifts for monomaterial spectra."""
    shift_records: list[dict[str, Any]] = []

    rng = np.random.default_rng(SUBSAMPLING_SEED)
    n_samples = min(len(spectra), max_samples)
    sample_indices = rng.choice(len(spectra), size=n_samples, replace=False)

    for i in tqdm(sample_indices, desc=f"Analyzing {dataset_name} shifts"):
        spectrum = spectra[i]
        active_polymers = get_active_polymers(labels_matrix[i], labels)

        if len(active_polymers) != 1:
            continue

        polymer = active_polymers[0]
        reference_peaks = get_reference_peak_array(polymer, experimental_references)

        if len(reference_peaks) == 0:
            continue

        detected_ppm = detect_peak_positions(spectrum, ppm_domain)

        if len(detected_ppm) == 0:
            continue

        matches = mutual_nearest_matches(
            reference_ppm=reference_peaks,
            detected_ppm=detected_ppm,
            tolerance_ppm=TOL_PPM,
        )

        for expected_ppm, detected_value in matches:
            shift = detected_value - expected_ppm

            if abs(shift) <= TOL_PPM:
                shift_records.append(
                    {
                        "Dataset": dataset_name,
                        "Expected_ppm": expected_ppm,
                        "Detected_ppm": detected_value,
                        "Shift_ppm": shift,
                        "Mixture": DISPLAY_NAMES[polymer],
                    }
                )

    return pd.DataFrame(shift_records)


# %% PLOTTING UTILITIES


def save_current_figure(path: str | Path, show: bool) -> None:
    """Save current matplotlib figure and optionally show it."""
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved: {path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_global_shift_violin(
    df_all_shifts: pd.DataFrame,
    plot_dir: str,
    show: bool,
) -> None:
    """Plot global peak-shift violin plot."""
    plt.figure(figsize=(10, 6))

    sns.violinplot(
        data=df_all_shifts,
        x="Dataset",
        y="Shift_ppm",
        order=DATASET_ORDER,
        palette="muted",
        inner="quartile",
        bw_adjust=2.5,
    )

    plt.title(
        "Peak Shifts Relative to Experimental References",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("Relative Shift ($\\Delta$ ppm)", fontsize=12)
    plt.xlabel("Dataset Origin", fontsize=12)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    save_current_figure(
        Path(plot_dir) / "1_Global_Peak_Shift_Violin.png",
        show=show,
    )


def plot_global_shift_kde(
    df_all_shifts: pd.DataFrame,
    plot_dir: str,
    show: bool,
) -> None:
    """Plot global peak-shift KDE."""
    plt.figure(figsize=(10, 6))

    sns.kdeplot(
        data=df_all_shifts,
        x="Shift_ppm",
        hue="Dataset",
        hue_order=DATASET_ORDER,
        fill=True,
        common_norm=False,
        alpha=0.5,
        bw_adjust=2.5,
    )

    plt.title("KDE of Peak Shifts", fontsize=14, fontweight="bold")
    plt.xlabel("Relative Shift ($\\Delta$ ppm)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlim(-TOL_PPM, TOL_PPM)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)

    save_current_figure(
        Path(plot_dir) / "2_Global_Peak_Shift_KDE.png",
        show=show,
    )


def polymer_display_order(df_pure_shifts: pd.DataFrame) -> list[str]:
    """Return polymer display order following LABELS."""
    available = set(df_pure_shifts["Mixture"].unique())
    ordered = [
        DISPLAY_NAMES[label] for label in LABELS if DISPLAY_NAMES[label] in available
    ]

    remaining = sorted(available.difference(ordered))
    return ordered + remaining


def plot_pure_polymer_boxplot(
    df_pure_shifts: pd.DataFrame,
    plot_dir: str,
    show: bool,
) -> None:
    """Plot stratified peak-shift boxplot for pure polymers."""
    order = polymer_display_order(df_pure_shifts)

    plt.figure(figsize=(14, 6))

    sns.boxplot(
        data=df_pure_shifts,
        x="Mixture",
        y="Shift_ppm",
        hue="Dataset",
        hue_order=DATASET_ORDER,
        order=order,
        palette="muted",
        showfliers=True,
    )

    plt.title(
        "Relative Peak Shift Distribution (Pure Polymers Only)",
        fontsize=16,
        fontweight="bold",
    )
    plt.ylabel("Relative Shift ($\\Delta$ ppm)", fontsize=14)
    plt.xlabel("Polymer Configuration", fontsize=14)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5)
    plt.legend(title="Dataset", fontsize=12, title_fontsize=12)

    save_current_figure(
        Path(plot_dir) / "3_Pure_Polymers_Peak_Shift_Boxplot.png",
        show=show,
    )


def plot_pure_polymer_kde_facetgrid(
    df_pure_shifts: pd.DataFrame,
    plot_dir: str,
    show: bool,
) -> None:
    """Plot KDE facet grid for pure-polymer peak shifts."""
    order = polymer_display_order(df_pure_shifts)

    g = sns.FacetGrid(
        df_pure_shifts,
        col="Mixture",
        col_order=order,
        hue="Dataset",
        hue_order=DATASET_ORDER,
        col_wrap=4,
        height=3,
        aspect=1.2,
        sharey=False,
    )

    g.map(sns.kdeplot, "Shift_ppm", fill=True, alpha=0.5, bw_adjust=2.5)
    g.map(plt.axvline, x=0, color="black", linestyle="--", linewidth=1)
    g.set_axis_labels("Relative Shift ($\\Delta$ ppm)", "Density")
    g.set_titles(col_template="{col_name}", fontweight="bold")
    g.add_legend(title="Dataset")

    save_current_figure(
        Path(plot_dir) / "4_Pure_Polymers_Peak_Shift_KDE_FacetGrid.png",
        show=show,
    )


def plot_peak_by_peak_boxplots(
    df_pure_shifts: pd.DataFrame,
    plot_dir: str,
    show: bool,
) -> None:
    """Plot peak-by-peak shift distributions for each pure polymer."""
    print("[INFO] Generating peak-by-peak boxplots.")

    for polymer in polymer_display_order(df_pure_shifts):
        df_poly = df_pure_shifts[df_pure_shifts["Mixture"] == polymer].copy()

        if df_poly.empty:
            continue

        df_poly["Peak_Label"] = df_poly["Expected_ppm"].apply(lambda x: f"{x:.2f}")

        peaks_order_float = sorted(df_poly["Expected_ppm"].unique(), reverse=True)
        peaks_order_str = [f"{x:.2f}" for x in peaks_order_float]

        fig_width = max(8, len(peaks_order_str) * 1.5)

        plt.figure(figsize=(fig_width, 6))

        sns.boxplot(
            data=df_poly,
            x="Peak_Label",
            y="Shift_ppm",
            hue="Dataset",
            hue_order=DATASET_ORDER,
            order=peaks_order_str,
            palette="muted",
            showfliers=True,
        )

        plt.title(
            f"Peak-by-Peak Relative Shift Analysis - {polymer}",
            fontsize=16,
            fontweight="bold",
        )
        plt.ylabel("Relative Shift ($\\Delta$ ppm)", fontsize=14)
        plt.xlabel("Experimental Reference Peak Position (ppm)", fontsize=14)
        plt.axhline(0, color="black", linestyle="--", linewidth=1.5)
        plt.xticks(rotation=45, ha="right")
        plt.ylim(-TOL_PPM, TOL_PPM)
        plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")

        safe_polymer = polymer.replace("/", "_").replace(" ", "_")
        save_current_figure(
            Path(plot_dir) / f"5_Peak_by_Peak_Shift_{safe_polymer}.png",
            show=show,
        )


def plot_bubble_shift_analysis(
    df_pure_shifts: pd.DataFrame,
    plot_dir: str,
    show: bool,
) -> None:
    """Plot bubble scatter charts of median absolute shift per peak."""
    print("[INFO] Generating bubble plots.")

    df_pure_shifts = df_pure_shifts.copy()
    df_pure_shifts["Abs_Shift"] = df_pure_shifts["Shift_ppm"].abs()

    all_peak_stats = (
        df_pure_shifts.groupby(["Mixture", "Dataset", "Expected_ppm"])
        .agg(
            Std_Shift=("Shift_ppm", "std"),
            N_Found=("Shift_ppm", "count"),
        )
        .reset_index()
    )

    all_peak_stats = all_peak_stats[all_peak_stats["N_Found"] > 2]

    universal_max_std = all_peak_stats["Std_Shift"].max()

    if pd.isna(universal_max_std) or universal_max_std == 0:
        universal_max_std = TOL_PPM

    print(f"[INFO] Global bubble-plot color scale: {universal_max_std:.4f} ppm.")

    for polymer in polymer_display_order(df_pure_shifts):
        df_poly = df_pure_shifts[df_pure_shifts["Mixture"] == polymer].copy()

        if df_poly.empty:
            continue

        df_poly["Abs_Shift"] = df_poly["Shift_ppm"].abs()

        peak_stats = (
            df_poly.groupby(["Dataset", "Expected_ppm"])
            .agg(
                Median_AE=("Abs_Shift", "median"),
                Std_Shift=("Shift_ppm", "std"),
                N_Found=("Shift_ppm", "count"),
            )
            .reset_index()
        )

        peak_stats = peak_stats[peak_stats["N_Found"] > 2].copy()

        if peak_stats.empty:
            continue

        fig, axes = plt.subplots(
            len(DATASET_ORDER),
            1,
            figsize=(22, 3 * len(DATASET_ORDER)),
        )

        axes = np.atleast_1d(axes)
        fig.subplots_adjust(hspace=0.30)

        cmap_std = plt.get_cmap("viridis")
        scatter_for_colorbar = None

        polymer_peaks = df_poly["Expected_ppm"].unique()

        for idx, dataset_name in enumerate(DATASET_ORDER):
            ax = axes[idx]
            ds_stats = peak_stats[peak_stats["Dataset"] == dataset_name]

            ax.set_xlim(55, 5)
            ax.set_ylim(-0.002, 0.032)

            if ds_stats.empty:
                ax.set_title(
                    f"{polymer} — {dataset_name} (no valid data)",
                    fontsize=13,
                    fontweight="bold",
                )
                continue

            ppms = ds_stats["Expected_ppm"].values
            median_abs_errors = ds_stats["Median_AE"].values
            stds = ds_stats["Std_Shift"].values

            sizes = np.clip(
                80 + (stds / (universal_max_std + 1e-9)) * 520,
                80,
                600,
            )

            for peak_ppm in polymer_peaks:
                ax.axvline(
                    peak_ppm,
                    color="gray",
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.2,
                    zorder=1,
                )

            scatter = ax.scatter(
                ppms,
                median_abs_errors,
                s=sizes,
                c=stds,
                cmap=cmap_std,
                vmin=0.0,
                vmax=universal_max_std,
                alpha=0.85,
                zorder=3,
                edgecolors="white",
                linewidths=1.2,
            )

            scatter_for_colorbar = scatter

            texts = []

            for x_value, y_value in zip(ppms, median_abs_errors):
                text = ax.text(
                    x_value,
                    y_value,
                    f"{y_value:.3f}",
                    fontsize=11,
                    ha="center",
                    color="black",
                    zorder=4,
                )
                texts.append(text)

            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(
                    arrowstyle="-",
                    color="dimgray",
                    lw=0.8,
                ),
            )

            ax.set_title(f"{polymer} — {dataset_name}", fontsize=14, fontweight="bold")
            ax.axhline(0, color="black", linewidth=1, linestyle="-", zorder=2)
            ax.grid(axis="both", linestyle="--", alpha=0.3)

        axes[1].set_ylabel(
            "Median Absolute Shift ($\\Delta$ ppm)",
            fontsize=12,
            fontweight="bold",
        )
        axes[-1].set_xlabel(
            "Peak Position (ppm)",
            fontsize=12,
            fontweight="bold",
        )

        if scatter_for_colorbar is not None:
            colorbar = fig.colorbar(
                scatter_for_colorbar,
                ax=axes,
                location="left",
                pad=0.08,
                fraction=0.04,
            )
            colorbar.set_label(
                "Std of Shift (ppm)",
                rotation=90,
                labelpad=15,
                fontweight="bold",
            )

        safe_polymer = polymer.replace("/", "_").replace(" ", "_")
        out_path = Path(plot_dir) / f"5B_Bubble_Plot_{safe_polymer}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved: {out_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)


def save_reference_peaks(
    experimental_references: dict[str, dict[str, Any]],
    plot_dir: str,
) -> None:
    """Save experimentally derived reference peaks to JSON."""
    out_path = Path(plot_dir) / "experimental_reference_peaks.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(experimental_references, f, indent=2)

    print(f"[INFO] Experimental reference peaks saved to: {out_path}")


# %% DATA LOADING


def load_inputs(args: argparse.Namespace):
    """Load synthetic, fine-tuning, and experimental datasets."""
    plot_dir = get_plot_dir(
        synth_data=args.synth_data,
        suffix=args.suffix,
        output_dir=args.output_dir,
    )
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    ppm_domain = load_ppm_domain(args.synth_ppm_domain)

    print("[INFO] Loading synthetic dataset.")
    synth_data = pd.read_pickle(args.synth_data)
    X_train = synth_data[0]
    y_w_train = synth_data[3]

    print("[INFO] Loading fine-tuning dataset.")
    ft_dir = Path(FT_FOLDER) / args.model_name
    ft_artifacts = load_artifact_folder(ft_dir)

    X_ft = require_artifact(ft_artifacts, artifact_key("X", args.suffix))
    y_w_ft = require_artifact(ft_artifacts, artifact_key("y_w", args.suffix))

    print("[INFO] Loading experimental test dataset.")
    test_data = pd.read_pickle(args.test_data)

    test_spectra = test_data.iloc[:, 4:].values
    test_labels = test_data["copo_tuple"].values

    return (
        ppm_domain,
        X_train,
        y_w_train,
        X_ft,
        y_w_ft,
        test_spectra,
        test_labels,
        plot_dir,
    )


# %% MAIN


def main() -> None:
    args = parse_args()
    print(args)

    (
        ppm_domain,
        X_train,
        y_w_train,
        X_ft,
        y_w_ft,
        test_spectra,
        test_labels,
        plot_dir,
    ) = load_inputs(args)

    experimental_references = build_experimental_reference_peaks(
        spectra=test_spectra,
        labels_matrix=test_labels,
        ppm_domain=ppm_domain,
        labels=LABELS,
    )

    save_reference_peaks(
        experimental_references=experimental_references,
        plot_dir=plot_dir,
    )

    print("\n[INFO] Extracting relative peak shifts.")

    df_synth_shifts = extract_peak_shifts(
        spectra=X_train,
        labels_matrix=y_w_train,
        ppm_domain=ppm_domain,
        experimental_references=experimental_references,
        labels=LABELS,
        dataset_name="Synthetic",
        max_samples=args.max_samples,
    )

    df_ft_shifts = extract_peak_shifts(
        spectra=X_ft,
        labels_matrix=y_w_ft,
        ppm_domain=ppm_domain,
        experimental_references=experimental_references,
        labels=LABELS,
        dataset_name="Fine-Tuning",
        max_samples=args.max_samples,
    )

    df_test_shifts = extract_peak_shifts(
        spectra=test_spectra,
        labels_matrix=test_labels,
        ppm_domain=ppm_domain,
        experimental_references=experimental_references,
        labels=LABELS,
        dataset_name="Experimental",
        max_samples=args.max_samples,
    )

    df_all_shifts = pd.concat(
        [df_synth_shifts, df_ft_shifts, df_test_shifts],
        ignore_index=True,
    )

    if df_all_shifts.empty:
        raise ValueError(
            "No peak-shift records were extracted. "
            "Check peak detection thresholds and experimental reference peaks."
        )

    print(f"\n[INFO] Generating peak-shift plots in: {plot_dir}")

    plot_global_shift_violin(
        df_all_shifts=df_all_shifts,
        plot_dir=plot_dir,
        show=args.show,
    )

    plot_global_shift_kde(
        df_all_shifts=df_all_shifts,
        plot_dir=plot_dir,
        show=args.show,
    )

    print("\n[INFO] Descriptive peak-shift statistics.")
    stats = df_all_shifts.groupby("Dataset")["Shift_ppm"].describe()
    print(stats)

    df_pure_shifts = df_all_shifts[
        ~df_all_shifts["Mixture"].str.contains(r"\+", regex=True)
    ].copy()

    if df_pure_shifts.empty:
        raise ValueError("No monomaterial peak-shift records were extracted.")

    plot_pure_polymer_boxplot(
        df_pure_shifts=df_pure_shifts,
        plot_dir=plot_dir,
        show=args.show,
    )

    plot_pure_polymer_kde_facetgrid(
        df_pure_shifts=df_pure_shifts,
        plot_dir=plot_dir,
        show=args.show,
    )

    print("\n[INFO] Stratified descriptive statistics for monomaterials.")
    stratified_stats = df_pure_shifts.groupby(["Mixture", "Dataset"])[
        "Shift_ppm"
    ].describe()
    print(stratified_stats)

    plot_peak_by_peak_boxplots(
        df_pure_shifts=df_pure_shifts,
        plot_dir=plot_dir,
        show=args.show,
    )

    plot_bubble_shift_analysis(
        df_pure_shifts=df_pure_shifts,
        plot_dir=plot_dir,
        show=args.show,
    )

    print("[INFO] Peak-shift analysis completed.")


if __name__ == "__main__":
    main()
