#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UMAP-based domain-gap analysis for DeepNMR datasets.

The script compares the topology of synthetic pretraining spectra,
pseudo-synthetic fine-tuning spectra, and experimental test spectra using a
shared UMAP embedding.

Generated plots:
  - 1_General_Domain_Gap.png
  - 2_Detailed_Categories_Gap.png
  - 3_Detailed_Mixtures_Gap.png
  - 4_Grouped_Matrix_Gap.png
  - 5_Matched_Coverage_Gap.png

Suffix convention
-----------------
Use --suffix without leading underscore, e.g.:

  --suffix ftc_o2

Internally, the suffix is normalized to '_ftc_o2', producing artifact keys such as:

  X_ftc_o2
  y_w_ftc_o2

and output folders such as:

  OUTPUT/domain_gap/<train_tag>/plots_ftc_o2/
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import MappingNames


# %% FIXED CONFIGURATION

LABELS = ["LDPE", "PE", "PP", "EH", "EO", "EB", "RACO", "EPR"]
DISPLAY_NAMES = MappingNames()

VAL_FOLDER = "val_sets"
FT_FOLDER = "ft_sets"

PRESENCE_THRESHOLD = 1e-2

UMAP_N_NEIGHBORS = 50
UMAP_MIN_DIST = 0.3
UMAP_METRIC = "euclidean"
UMAP_N_JOBS = 64
UMAP_RANDOM_STATE = None

SUBSAMPLING_SEED = 42

INCLUDE_TRAIN = True
INCLUDE_FT = True
INCLUDE_TEST = True


# %% ARGUMENTS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="UMAP-based domain-gap analysis for DeepNMR datasets.",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--train_set",
        type=str,
        default="DATASET/synthetic_dataset.pkl",
        help="Synthetic pretraining dataset to be analyzed.",
    )
    parser.add_argument(
        "--test_dataset",
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
        help="Name of the model folder under val_sets/ and ft_sets/.",
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
            "OUTPUT/domain_gap/<train-set-tag>/plots<suffix>."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Maximum number of samples per domain used in the UMAP embedding.",
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


def fine_tuning_key(prefix: str, suffix: str | None) -> str:
    """Build fine-tuning array keys from a prefix and normalized suffix."""
    suffix = normalize_suffix(suffix)
    return f"{prefix}{suffix}"


def get_plot_dir(
    train_set: str,
    suffix: str | None,
    output_dir: str | None = None,
) -> str:
    """Return plot directory using the common suffix convention."""
    if output_dir is not None:
        return output_dir

    train_tag = Path(train_set).stem.split("_")[-1]
    suffix = normalize_suffix(suffix)

    return f"OUTPUT/domain_gap2/{train_tag}/plots{suffix}"


# %% ARTIFACT UTILITIES


def load_artifact_folder(folder: str | Path) -> dict[str, Any]:
    """Load all .npy and .pkl artifacts from a folder into a dictionary."""
    folder = Path(folder)
    artifacts: dict[str, Any] = {}

    if not folder.exists():
        print(f"[WARNING] Artifact folder not found: {folder}")
        return artifacts

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


# %% UMAP UTILITIES


def patch_umap_check_array():
    """Patch UMAP's check_array call for newer scikit-learn versions."""
    import umap
    import sklearn.utils.validation

    if getattr(umap, "_is_patched_for_sklearn", False):
        print("[INFO] UMAP / scikit-learn compatibility patch already present.")
        return umap

    print("[INFO] Applying UMAP / scikit-learn compatibility patch.")

    real_check_array = sklearn.utils.validation.check_array

    def patched_check_array(*args, **kwargs):
        if "ensure_all_finite" in kwargs:
            kwargs["force_all_finite"] = kwargs.pop("ensure_all_finite")
        return real_check_array(*args, **kwargs)

    if hasattr(umap, "check_array"):
        umap.check_array = patched_check_array

    if hasattr(umap, "umap_"):
        umap.umap_.check_array = patched_check_array

    umap._is_patched_for_sklearn = True
    return umap


def compute_umap_embedding(X_global: np.ndarray) -> np.ndarray:
    """Compute a shared UMAP embedding for all active domains."""
    umap = patch_umap_check_array()

    print(f"[INFO] Computing UMAP on {len(X_global)} samples.")

    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        n_jobs=UMAP_N_JOBS,
        random_state=UMAP_RANDOM_STATE,
    )

    return reducer.fit_transform(X_global)


# %% DATA UTILITIES


def subsample_aligned(
    X: np.ndarray,
    annotations: np.ndarray | None = None,
    max_samples: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Subsample X and optional annotations using the same random indices."""
    if len(X) <= max_samples:
        return X, annotations

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), max_samples, replace=False)

    X_sub = X[indices]

    if annotations is None:
        return X_sub, None

    return X_sub, annotations[indices]


def get_axis_limits(values: np.ndarray, margin: float = 0.05) -> tuple[float, float]:
    """Return axis limits with a relative margin."""
    v_min, v_max = values.min(), values.max()
    pad = (v_max - v_min) * margin
    return float(v_min - pad), float(v_max + pad)


def normalize_tuple(value: Any) -> tuple[str, ...]:
    """Convert a tuple/list/scalar copolymer annotation to a tuple of strings."""
    if isinstance(value, tuple):
        return tuple(str(v) for v in value)

    if isinstance(value, list):
        return tuple(str(v) for v in value)

    return (str(value),)


def components_from_annotation(
    annotation: Any,
    *,
    is_test: bool,
    labels: list[str],
    presence_threshold: float,
) -> tuple[str, ...]:
    """Extract active component labels from either a test tuple or a weight vector."""
    if is_test:
        return normalize_tuple(annotation)

    annotation = np.asarray(annotation, dtype=float)
    active_indices = np.where(annotation > presence_threshold)[0]

    return tuple(labels[i] for i in active_indices)


def mapped_components(components: tuple[str, ...]) -> list[str]:
    """Map internal component names to display names."""
    return [DISPLAY_NAMES[c] for c in components]


# %% LABELING AND PALETTES


def component_label(
    *,
    components: tuple[str, ...],
    focus_domain: str,
    level: int,
    test_tuples: set[tuple[str, ...]],
) -> str:
    """Build a plot label for a component tuple at a given visualization level."""
    if len(components) == 0:
        return f"{focus_domain} No component"

    components_sorted = tuple(sorted(components))
    display_components = mapped_components(components)
    is_pure = len(components) == 1

    if level == 2:
        if is_pure:
            return f"{focus_domain} Pure: {display_components[0]}"
        return f"{focus_domain} Mixture"

    if level == 3:
        if is_pure:
            return f"{focus_domain} Pure: {display_components[0]}"
        return f"{focus_domain} Mix: {'+'.join(display_components)}"

    if level == 4:
        if is_pure:
            return f"{focus_domain} Pure: {display_components[0]}"

        has_pp = "PP" in components
        has_ldpe = "LDPE" in components
        has_pe = "PE" in components

        if has_pp and has_ldpe and has_pe:
            return f"{focus_domain} Mix: PP + LDPE + PE"
        if has_pp and has_ldpe:
            return f"{focus_domain} Mix: PP + LDPE"
        if has_pp and has_pe:
            return f"{focus_domain} Mix: PP + PE"
        if has_ldpe and has_pe:
            return f"{focus_domain} Mix: LDPE + PE"
        if has_pp:
            return f"{focus_domain} Mix: PP + Others"
        if has_ldpe:
            return f"{focus_domain} Mix: LDPE + Others"
        if has_pe:
            return f"{focus_domain} Mix: PE + Others"

        return f"{focus_domain} Mix: Other Mixtures"

    if level == 5:
        if components_sorted in test_tuples:
            if is_pure:
                return f"Matched Pure: {display_components[0]}"
            return f"Matched Mix: {'+'.join(display_components)}"

        return f"Z_Background (Unmatched {focus_domain})"

    raise ValueError(f"Unsupported plot level: {level}")


def build_palette(
    labels_for_plot: list[str],
    level: int,
) -> tuple[dict[str, Any], list[str]]:
    """Build a deterministic palette and hue order for the requested labels."""
    unique_labels = set(labels_for_plot)
    palette: dict[str, Any] = {}

    for label in unique_labels:
        if "Z_Background" in label:
            if "Unmatched" in label:
                palette[label] = (0.85, 0.85, 0.85, 0.01)
            else:
                palette[label] = (0.90, 0.90, 0.90, 0.005)

    active_labels = [label for label in unique_labels if not label.startswith("Z_")]
    pure_labels = sorted([label for label in active_labels if "Pure" in label])
    mix_labels = sorted(
        [
            label
            for label in active_labels
            if "Mix" in label or "Mixture" in label
        ]
    )

    if level == 2:
        colors = sns.color_palette("tab10", len(active_labels))
        for i, label in enumerate(sorted(active_labels)):
            palette[label] = colors[i]

    elif level in {3, 5}:
        pure_colors = sns.color_palette("Pastel1", len(pure_labels))
        mix_colors = sns.color_palette("husl", len(mix_labels))

        for i, label in enumerate(pure_labels):
            palette[label] = pure_colors[i]

        for i, label in enumerate(mix_labels):
            palette[label] = mix_colors[i]

    elif level == 4:
        pure_colors = sns.color_palette("Pastel1", len(pure_labels))

        for i, label in enumerate(pure_labels):
            palette[label] = pure_colors[i]

        for label in mix_labels:
            if "PP + Others" in label:
                palette[label] = "tab:orange"
            elif "LDPE + Others" in label:
                palette[label] = "tab:blue"
            elif "PE + Others" in label:
                palette[label] = "tab:green"
            elif "PP + LDPE" in label and "PE" not in label:
                palette[label] = "tab:purple"
            elif "PP + PE" in label and "LDPE" not in label:
                palette[label] = "tab:red"
            elif "LDPE + PE" in label and "PP" not in label:
                palette[label] = "tab:cyan"
            elif "PP + LDPE + PE" in label:
                palette[label] = "tab:brown"
            else:
                palette[label] = "tab:olive"

    background_labels = sorted(
        [label for label in unique_labels if label.startswith("Z_")]
    )
    hue_order = background_labels + sorted(pure_labels) + sorted(mix_labels)

    return palette, hue_order


def clean_legend_labels(labels: list[str]) -> list[str]:
    """Clean background labels for compact legends."""
    return [
        label.replace("Z_Background (", "Bg: ").replace(")", "")
        for label in labels
    ]


# %% PLOTTING


def plot_umap_level(
    *,
    level: int,
    title_prefix: str,
    filename_prefix: str,
    X_umap: np.ndarray,
    domain_blocks: list[dict[str, Any]],
    test_tuples: set[tuple[str, ...]],
    labels: list[str],
    presence_threshold: float,
    plot_dir: str,
    show: bool,
) -> None:
    """Create and save one UMAP visualization level."""
    umap_xlim = get_axis_limits(X_umap[:, 0])
    umap_ylim = get_axis_limits(X_umap[:, 1])

    if level == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        all_labels = []
        for block in domain_blocks:
            all_labels.extend([f"Domain: {block['name']}"] * len(block["X"]))

        sns.scatterplot(
            x=X_umap[:, 0],
            y=X_umap[:, 1],
            hue=all_labels,
            palette="colorblind",
            alpha=0.6,
            s=20,
            edgecolor=None,
            ax=ax,
        )

        ax.set_title("UMAP Embedding", fontweight="bold")
        ax.set_xlim(umap_xlim)
        ax.set_ylim(umap_ylim)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        sns.despine()
        fig.tight_layout()

        out_path = Path(plot_dir) / f"{filename_prefix}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved: {out_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return

    n_cols = len(domain_blocks)
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 8))
    axes = np.atleast_1d(axes)

    for idx, focus_block in enumerate(domain_blocks):
        focus_name = focus_block["name"]
        all_labels = []

        for block in domain_blocks:
            block_name = block["name"]

            if block_name != focus_name:
                all_labels.extend([f"Z_Background ({block_name})"] * len(block["X"]))
                continue

            for annotation in block["annotations"]:
                components = components_from_annotation(
                    annotation,
                    is_test=block["is_test"],
                    labels=labels,
                    presence_threshold=presence_threshold,
                )

                label = component_label(
                    components=components,
                    focus_domain=focus_name,
                    level=level,
                    test_tuples=test_tuples,
                )
                all_labels.append(label)

        palette, hue_order = build_palette(all_labels, level)

        sns.scatterplot(
            x=X_umap[:, 0],
            y=X_umap[:, 1],
            hue=all_labels,
            hue_order=hue_order,
            palette=palette,
            alpha=0.6,
            s=25,
            edgecolor=None,
            ax=axes[idx],
        )

        axes[idx].set_title(f"{title_prefix} - {focus_name}", fontweight="bold")
        axes[idx].set_xlim(umap_xlim)
        axes[idx].set_ylim(umap_ylim)
        axes[idx].set_xlabel("UMAP 1")

        if idx == 0:
            axes[idx].set_ylabel("UMAP 2")
        else:
            axes[idx].set_ylabel("")

        handles, legend_labels = axes[idx].get_legend_handles_labels()
        axes[idx].legend(
            handles,
            clean_legend_labels(legend_labels),
            title="Domains",
            loc="best",
            fontsize=8,
        )

    sns.despine()
    fig.tight_layout()

    out_path = Path(plot_dir) / f"{filename_prefix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# %% DOMAIN PREPARATION


def prepare_domain_blocks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    """Load, normalize, subsample, and assemble the requested domain blocks."""
    plot_dir = get_plot_dir(
        train_set=args.train_set,
        suffix=args.suffix,
        output_dir=args.output_dir,
    )
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading synthetic training dataset.")
    train_data = pd.read_pickle(args.train_set)
    X_train = train_data[0]
    y_w_train = train_data[3]

    val_dir = Path(VAL_FOLDER) / args.model_name
    ft_dir = Path(FT_FOLDER) / args.model_name

    artifacts: dict[str, Any] = {}
    artifacts.update(load_artifact_folder(val_dir))
    artifacts.update(load_artifact_folder(ft_dir))

    X_ft = require_artifact(artifacts, fine_tuning_key("X", args.suffix))
    y_w_ft = require_artifact(artifacts, fine_tuning_key("y_w", args.suffix))

    scaler_spectra = artifacts.get("scaler_spectra", None)

    print("[INFO] Loading experimental test dataset.")
    test_data = pd.read_pickle(args.test_dataset)
    X_test = test_data.iloc[:, 4:].values.astype(np.float32)
    test_annotations = test_data["copo_tuple"].values

    if scaler_spectra is not None:
        print("[INFO] Applying spectral scaler to training and test domains.")
        X_train = scaler_spectra.transform(X_train.flatten().reshape(-1, 1)).reshape(
            X_train.shape
        )
        X_test = scaler_spectra.transform(X_test.flatten().reshape(-1, 1)).reshape(
            X_test.shape
        )
    else:
        print("[WARNING] No spectral scaler found. Train/test spectra are not normalized.")

    print(
        f"[INFO] Applying subsampling with max_samples={args.max_samples} "
        f"to reduce overplotting."
    )

    X_train_sub, y_w_train_sub = subsample_aligned(
        X_train.reshape(X_train.shape[0], -1),
        y_w_train,
        max_samples=args.max_samples,
        seed=SUBSAMPLING_SEED,
    )

    X_ft_sub, y_w_ft_sub = subsample_aligned(
        X_ft.reshape(X_ft.shape[0], -1),
        y_w_ft,
        max_samples=args.max_samples,
        seed=SUBSAMPLING_SEED,
    )

    X_test_sub, test_annotations_sub = subsample_aligned(
        X_test.reshape(X_test.shape[0], -1),
        test_annotations,
        max_samples=args.max_samples,
        seed=SUBSAMPLING_SEED,
    )

    domain_blocks: list[dict[str, Any]] = []

    if INCLUDE_TRAIN:
        domain_blocks.append(
            {
                "name": "Train",
                "X": X_train_sub,
                "annotations": y_w_train_sub,
                "is_test": False,
            }
        )

    if INCLUDE_FT:
        domain_blocks.append(
            {
                "name": "FT",
                "X": X_ft_sub,
                "annotations": y_w_ft_sub,
                "is_test": False,
            }
        )

    if INCLUDE_TEST:
        domain_blocks.append(
            {
                "name": "Test",
                "X": X_test_sub,
                "annotations": test_annotations_sub,
                "is_test": True,
            }
        )

    if len(domain_blocks) == 0:
        raise ValueError("At least one domain must be included in the analysis.")

    return domain_blocks, plot_dir


# %% MAIN


def main() -> None:
    """Run the domain-gap analysis."""
    args = parse_args()
    print(args)

    domain_blocks, plot_dir = prepare_domain_blocks(args)

    X_global = np.vstack([block["X"] for block in domain_blocks])

    test_block = next(
        (block for block in domain_blocks if block["name"] == "Test"),
        None,
    )

    if test_block is not None:
        test_tuples = {
            tuple(sorted(normalize_tuple(annotation)))
            for annotation in test_block["annotations"]
        }
    else:
        test_tuples = set()

    X_umap = compute_umap_embedding(X_global)

    plots = [
        (1, "Global Topology", "1_General_Domain_Gap"),
        (2, "Category-aware UMAP embedding", "2_Detailed_Categories_Gap"),
        (3, "Detailed Mixtures Expansion", "3_Detailed_Mixtures_Gap"),
        (4, "Topology by Polymer Matrix", "4_Grouped_Matrix_Gap"),
        (5, "Validation Cross-Section", "5_Matched_Coverage_Gap"),
    ]

    for level, title_prefix, filename_prefix in plots:
        print(f"[INFO] Generating plot {level}: {filename_prefix}")

        plot_umap_level(
            level=level,
            title_prefix=title_prefix,
            filename_prefix=filename_prefix,
            X_umap=X_umap,
            domain_blocks=domain_blocks,
            test_tuples=test_tuples,
            labels=LABELS,
            presence_threshold=PRESENCE_THRESHOLD,
            plot_dir=plot_dir,
            show=args.show,
        )

    print(f"[INFO] Domain-gap analysis completed. Output folder: {plot_dir}")


if __name__ == "__main__":
    main()