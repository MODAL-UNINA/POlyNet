#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluation script for DeepNMR models."""

# %% IMPORT SECTION
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import json
import math
import argparse
import importlib
import itertools

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from fastcore.all import dict2obj
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    r2_score,
)

from utils import MappingNames

LABELS = ["LDPE", "PE", "PP", "EH", "EO", "EB", "RACO", "EPR"]
HOMOPOLYMER_LIKE = {"LDPE", "PE", "PP"}

VAL_PRESENCE_THRESHOLD = 1e-2
TEST_PRESENCE_THRESHOLD = 2.5e-2
PLOT_COMPONENT_THRESHOLD = 1e-2
REPORT_PRIMARY_THRESHOLD = 2.5e-2
REPORT_SECONDARY_THRESHOLD = 1e-2

DISPLAY_NAMES = MappingNames()


def normalize_suffix(suffix):
    """Normalize suffix so that non-empty suffixes start with '_'."""
    if suffix is None:
        return ""

    suffix = str(suffix).strip().strip("_")

    if suffix == "":
        return ""

    return f"_{suffix}"


def mapped_label(label):
    """Map internal class names to display names."""
    return DISPLAY_NAMES[label]


def normalize_compositions(y_c, scaler):
    """Normalize composition channels using the training composition scaler.

    LDPE, PE, and PP are composition-invariant and are kept unchanged.
    EH, EO, EB, RACO, and EPR are transformed with the fitted MinMaxScaler.
    """
    return np.concatenate(
        [y_c[:, 0:3], scaler.transform(y_c[:, 3:])],
        axis=1,
    )


def denormalize_compositions(y_c_norm, scaler):
    """Map normalized composition predictions back to the original composition scale."""
    return np.concatenate(
        [y_c_norm[:, 0:3], scaler.inverse_transform(y_c_norm[:, 3:])],
        axis=1,
    )


def build_targets_from_dataframe(df, labels):
    """Build dense weight and composition matrices from tuple-based labels."""
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    y_w = np.zeros((len(df), len(labels)))
    y_c = np.zeros((len(df), len(labels)))

    for i, idx in enumerate(df.index):
        copo_tuple = df.loc[idx, "copo_tuple"]
        w_tuple = df.loc[idx, "w"]
        c_tuple = df.loc[idx, "c"]

        for copo, w in zip(copo_tuple, w_tuple):
            y_w[i, label_to_index[copo]] = w

        for copo, c in zip(copo_tuple, c_tuple):
            y_c[i, label_to_index[copo]] = c

    return y_w, y_c


def load_history(run_name, suffix):
    """Load training history for either the base or the fine-tuned model."""
    suffix = normalize_suffix(suffix)
    history_path = f"models/{run_name}/training_history{suffix}.json"

    with open(history_path, "r") as f:
        return json.load(f)


def get_plot_dir(run_name, suffix):
    """Return plot directory for the evaluated model."""
    suffix = normalize_suffix(suffix)

    plot_dir = f"OUTPUT/eval_models/{run_name}/plots{suffix}"
    os.makedirs(plot_dir, exist_ok=True)

    return plot_dir


def get_weights_path(run_name, suffix):
    """Return model-weights path using the common suffix convention."""
    suffix = normalize_suffix(suffix)
    return f"models/{run_name}/model{suffix}.weights.h5"


def plot_history_with_lr(history, y_key, val_key, ylabel, title, filename, plot_dir):
    """Plot a training/validation curve together with the learning-rate schedule."""
    if y_key not in history or val_key not in history:
        print(f"[WARNING] Missing history keys: {y_key}, {val_key}")
        return

    fig, ax1 = plt.subplots(figsize=(15, 5))

    ax1.plot(history[y_key], label="Training Loss", color="blue")
    ax1.plot(history[val_key], label="Validation Loss", color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(ylabel, color="blue")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.legend(loc="upper left")

    if "learning_rate" in history:
        ax2 = ax1.twinx()
        ax2.plot(
            history["learning_rate"],
            label="Learning Rate",
            color="red",
            linestyle="--",
            alpha=0.7,
        )
        ax2.set_ylabel("Learning Rate", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.set_yscale("log")
        ax2.legend(loc="upper right")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{filename}")
    plt.show()


def plot_metric_history_with_lr(history, key_filter, ylabel, title, filename, plot_dir):
    """Plot all history series matching a filter together with the learning rate."""
    selected_keys = [key for key in history.keys() if key_filter(key)]

    if len(selected_keys) == 0:
        print(f"[WARNING] No history keys found for: {title}")
        return

    fig, ax1 = plt.subplots(figsize=(15, 5))

    for key in selected_keys:
        label = key

        if "composition" in key and key.split("_")[-1] in LABELS:
            label = "_".join(key.split("_")[:-1] + [mapped_label(key.split("_")[-1])])

        ax1.plot(history[key], label=label, linestyle="-")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(ylabel, color="blue")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.legend(loc="upper left")

    if "learning_rate" in history:
        ax2 = ax1.twinx()
        ax2.plot(
            history["learning_rate"],
            label="Learning Rate",
            color="red",
            linestyle="--",
            alpha=0.7,
        )
        ax2.set_ylabel("Learning Rate", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.set_yscale("log")
        ax2.legend(loc="upper right")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{filename}")
    plt.show()


def active_combination_from_presence(vector, labels, sort_labels=False):
    """Convert a binary presence vector into a tuple of active labels."""
    combination = tuple(labels[i] for i, value in enumerate(vector) if value == 1)

    if sort_labels:
        return tuple(sorted(combination))

    return combination


def display_combination(combination, sort_labels=False):
    """Create a display label for a tuple of labels."""
    if len(combination) == 0:
        return ""

    if sort_labels:
        combination = tuple(sorted(combination))

    return " + ".join(mapped_label(label) for label in combination)


def ordered_combinations_by_cardinality(labels, max_components):
    """Generate combinations ordered by cardinality and label order."""
    ordered_combinations = []

    for n_components in range(1, max_components + 1):
        ordered_combinations.extend(list(itertools.combinations(labels, n_components)))

    return ordered_combinations


def confusion_from_combinations(
    true_combinations,
    pred_combinations,
    ordered_combinations,
    sort_display_labels=False,
):
    """Build a confusion matrix from true/predicted label combinations."""
    combination_to_index = {
        combination: idx for idx, combination in enumerate(ordered_combinations)
    }

    y_true_indices = []
    y_pred_indices = []

    for true_combination, pred_combination in zip(true_combinations, pred_combinations):
        true_idx = combination_to_index.get(true_combination, None)
        pred_idx = combination_to_index.get(pred_combination, None)

        if true_idx is not None and pred_idx is not None:
            y_true_indices.append(true_idx)
            y_pred_indices.append(pred_idx)

    cm = confusion_matrix(
        y_true_indices,
        y_pred_indices,
        labels=range(len(ordered_combinations)),
    )

    class_labels = [
        display_combination(combination, sort_labels=sort_display_labels)
        for combination in ordered_combinations
    ]

    return cm, class_labels, y_true_indices, y_pred_indices


def validation_confusion_from_presence(
    true_presence,
    pred_presence,
    labels,
    max_components_to_plot=None,
):
    """Build validation confusion matrix.

    If max_components_to_plot is an integer, only samples whose true and predicted
    component cardinalities are <= max_components_to_plot are included in the plot.

    If max_components_to_plot is None, all observed true/predicted combinations
    are included, ordered by cardinality and label order.
    """
    true_combinations = [
        active_combination_from_presence(vector, labels, sort_labels=False)
        for vector in true_presence
    ]
    pred_combinations = [
        active_combination_from_presence(vector, labels, sort_labels=False)
        for vector in pred_presence
    ]

    label_order = {label: idx for idx, label in enumerate(labels)}

    if max_components_to_plot is not None:
        filtered_true_combinations = []
        filtered_pred_combinations = []

        excluded_true_high_order = 0
        excluded_pred_high_order = 0
        excluded_total = 0

        for true_combination, pred_combination in zip(
            true_combinations, pred_combinations
        ):
            true_high_order = len(true_combination) > max_components_to_plot
            pred_high_order = len(pred_combination) > max_components_to_plot

            if true_high_order:
                excluded_true_high_order += 1

            if pred_high_order:
                excluded_pred_high_order += 1

            if true_high_order or pred_high_order:
                excluded_total += 1
                continue

            filtered_true_combinations.append(true_combination)
            filtered_pred_combinations.append(pred_combination)

        ordered_combinations = ordered_combinations_by_cardinality(
            labels,
            max_components=max_components_to_plot,
        )

        print(
            f"[INFO] Validation confusion matrix restricted to combinations "
            f"with at most {max_components_to_plot} component(s)."
        )
        print(
            f"[INFO] Samples shown: {len(filtered_true_combinations)} / "
            f"{len(true_combinations)}"
        )
        print(
            f"[INFO] Samples excluded because true cardinality > "
            f"{max_components_to_plot}: {excluded_true_high_order}"
        )
        print(
            f"[INFO] Samples excluded because predicted cardinality > "
            f"{max_components_to_plot}: {excluded_pred_high_order}"
        )
        print(f"[INFO] Samples excluded in total: {excluded_total}")

        return confusion_from_combinations(
            filtered_true_combinations,
            filtered_pred_combinations,
            ordered_combinations,
            sort_display_labels=False,
        )

    observed_combinations = sorted(
        set(true_combinations).union(set(pred_combinations)),
        key=lambda combination: (
            len(combination),
            [label_order[label] for label in combination],
        ),
    )

    return confusion_from_combinations(
        true_combinations,
        pred_combinations,
        observed_combinations,
        sort_display_labels=False,
    )


def test_confusion_from_presence(y_labels, pred_presence, labels):
    """Build test confusion matrix using the same combination ordering as the old script."""
    true_combinations = [tuple(sorted(str(label).split("+"))) for label in y_labels]

    pred_combinations = [
        active_combination_from_presence(vector, labels, sort_labels=True)
        for vector in pred_presence
    ]

    ordered_combinations = sorted(set(true_combinations).union(set(pred_combinations)))

    return confusion_from_combinations(
        true_combinations,
        pred_combinations,
        ordered_combinations,
        sort_display_labels=True,
    )


def plot_confusion_matrix(cm, class_labels, filename, plot_dir):
    """Plot and save a confusion matrix."""
    plt.figure(figsize=(15, 15))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cmap="Blues",
        annot_kws={"size": 14},
        cbar=False,
    )
    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("Actual", fontsize=18)
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{filename}")
    plt.show()


def plot_channel_series_grid(
    y_true,
    y_pred,
    labels,
    ylabel,
    filename,
    plot_dir,
    valid_mask_mode,
    point_alpha=0.5,
    row_height=3.5,
    suptitle=None,
    title_metric_name="AvgMAE",
    use_display_label=True,
):
    """Plot per-channel ordered true/predicted series in a grid."""
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    n_labels = len(labels)
    n_cols = math.ceil(np.sqrt(n_labels))
    n_rows = math.ceil(n_labels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * row_height))
    axes = axes.flatten()

    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=18, y=1)

    for idx, copo in enumerate(labels):
        pos = label_to_index[copo]
        y_true_copo = y_true[:, pos]
        y_pred_copo = y_pred[:, pos]

        if valid_mask_mode == "weight":
            valid_idxs = np.where(y_true_copo > 0)[0]
        elif valid_mask_mode == "composition":
            valid_idxs = np.where(y_true_copo >= 0)[0]
        else:
            raise ValueError(f"Unknown valid_mask_mode: {valid_mask_mode}")

        y_true_copo = y_true_copo[valid_idxs]
        y_pred_copo = y_pred_copo[valid_idxs]

        order = np.argsort(y_true_copo)
        y_true_copo = y_true_copo[order]
        y_pred_copo = y_pred_copo[order]

        ax = axes[idx]
        ax.plot(
            y_pred_copo,
            ".",
            alpha=point_alpha,
            label="Predicted",
            color="tab:orange",
        )
        ax.plot(y_true_copo, label="True", color="tab:blue")

        avg_error = (
            np.mean(np.abs(y_true_copo - y_pred_copo)) if len(y_true_copo) else np.nan
        )

        title_label = mapped_label(copo) if use_display_label else copo
        ax.set_title(
            f"{title_label} - {title_metric_name}: {avg_error:.2f}",
            fontsize=16,
        )
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=12)
        ax.set_xlabel("Experiment", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)

    for idx in range(len(labels), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{filename}")
    plt.show()


def plot_metric_radar(
    df,
    metrics,
    plot_dir,
    multi_plot=True,
    save_single=True,
    show_single=False,
    dpi=300,
):
    """Plot radar charts for classification metrics."""
    labels = df["Label"].tolist()

    for i, label in enumerate(labels):
        if label == "LLDPE-H":
            labels[i] = label + "\n\n"
        elif label == "LLDPE-B + RaCo-PP":
            labels[i] = "     " + label
        elif label == "LLDPE-H + LLDPE-O":
            labels[i] = label + "     "

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    def safe_metric_name(metric):
        return metric.lower().replace(" ", "_").replace("-", "_").replace("/", "_")

    def plot_single_metric(metric, save=True, show=True):
        values = df[metric].tolist()
        values += values[:1]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

        ax.plot(angles, values, linewidth=2, linestyle="solid", label=metric)
        ax.fill(angles, values, alpha=0.25)

        ax.set_thetagrids(
            np.degrees(angles[:-1]),
            labels,
            fontsize=18,
            rotation=90,
        )
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "", "1.0"], fontsize=16)
        ax.set_ylim(0, 1.0)

        plt.tight_layout()

        if save:
            fig.savefig(
                f"{plot_dir}/{safe_metric_name(metric)}_radar.png",
                dpi=dpi,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close(fig)

    if not multi_plot:
        for metric in metrics:
            plot_single_metric(metric, save=True, show=True)
        return

    fig, axs = plt.subplots(2, 2, figsize=(14, 14), subplot_kw=dict(polar=True))
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        values = df[metric].tolist()
        values += values[:1]

        ax = axs[i]
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=metric)
        ax.fill(angles, values, alpha=0.25)

        ax.set_thetagrids(
            np.degrees(angles[:-1]),
            labels,
            fontsize=12,
            rotation=90,
        )
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "", "1.0"], fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_title(metric, fontsize=16, pad=20)

    for j in range(len(metrics), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    fig.savefig(f"{plot_dir}/combined_radar.png", dpi=dpi, bbox_inches="tight")
    plt.show()

    if save_single:
        for metric in metrics:
            plot_single_metric(metric, save=True, show=show_single)


def safe_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def safe_rmse(y_true_arr, y_pred_arr):
    err = np.asarray(y_true_arr, dtype=float) - np.asarray(y_pred_arr, dtype=float)
    err = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(err**2))) if err.size else np.nan


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if np.sum(mask) < 2:
        return np.nan

    if np.allclose(y_true[mask], y_true[mask][0]):
        return np.nan

    return float(r2_score(y_true[mask], y_pred[mask]))


def fmt(x, ndigits=4):
    if x is None:
        return "NA"

    try:
        if not np.isfinite(float(x)):
            return "NA"
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def sample_composition_label(row):
    try:
        return "+".join(row["copo_tuple"])
    except Exception:
        return str(row.get("copolymer", "NA"))


def binary_presence_matrix_from_weights(y_w, threshold):
    return (np.asarray(y_w, dtype=float) >= threshold).astype(int)


def write_presence_metrics_at_threshold(
    f_out,
    y_w_true_arr,
    y_w_pred_arr,
    threshold,
    prefix="",
):
    """Write multilabel presence/detection metrics at a given weight threshold."""
    true_presence = (y_w_true_arr > 0).astype(int)
    pred_presence = binary_presence_matrix_from_weights(y_w_pred_arr, threshold)

    exact_match = np.all(true_presence == pred_presence, axis=1)
    component_hamming_error = np.mean(true_presence != pred_presence)

    sample_fp = np.sum((true_presence == 0) & (pred_presence == 1), axis=1)
    sample_fn = np.sum((true_presence == 1) & (pred_presence == 0), axis=1)

    absent_entries = true_presence == 0
    present_entries = true_presence == 1

    fp_rate = (
        np.mean(pred_presence[absent_entries] == 1)
        if np.any(absent_entries)
        else np.nan
    )
    fn_rate = (
        np.mean(pred_presence[present_entries] == 0)
        if np.any(present_entries)
        else np.nan
    )

    f_out.write(f"{prefix}Presence metrics at weight threshold {threshold:g}\n")
    f_out.write(
        f"{prefix}  Exact component-set match rate: {fmt(np.mean(exact_match))}\n"
    )
    f_out.write(
        f"{prefix}  Component-level Hamming error: {fmt(component_hamming_error)}\n"
    )
    f_out.write(f"{prefix}  False-positive rate on absent entries: {fmt(fp_rate)}\n")
    f_out.write(f"{prefix}  False-negative rate on present entries: {fmt(fn_rate)}\n")
    f_out.write(
        f"{prefix}  Mean false-positive components per sample: {fmt(np.mean(sample_fp))}\n"
    )
    f_out.write(
        f"{prefix}  Mean false-negative components per sample: {fmt(np.mean(sample_fn))}\n"
    )
    f_out.write(
        f"{prefix}  Mean predicted components per sample: {fmt(np.mean(np.sum(pred_presence, axis=1)))}\n"
    )
    f_out.write(
        f"{prefix}  Mean true components per sample: {fmt(np.mean(np.sum(true_presence, axis=1)))}\n"
    )

    return true_presence, pred_presence, exact_match


def write_supervised_split_report(
    f_out,
    split_name,
    labels,
    y_w_true_arr,
    y_w_pred_arr,
    y_c_true_norm_arr,
    y_c_pred_norm_arr,
    y_c_true_denorm_arr=None,
    y_c_pred_denorm_arr=None,
    sample_names=None,
    sample_labels=None,
    weight_threshold=5e-2,
    secondary_threshold=1e-2,
    detail_top_k=25,
):
    """Write aggregate, per-copolymer, and per-sample metrics for a supervised split."""
    f_out.write(f"\n{'=' * 90}\n")
    f_out.write(f"{split_name}\n")
    f_out.write(f"{'=' * 90}\n")

    if y_w_true_arr is None or y_w_pred_arr is None:
        f_out.write("Split not available.\n")
        return

    y_w_true_arr = np.asarray(y_w_true_arr, dtype=float)
    y_w_pred_arr = np.asarray(y_w_pred_arr, dtype=float)
    y_c_true_norm_arr = np.asarray(y_c_true_norm_arr, dtype=float)
    y_c_pred_norm_arr = np.asarray(y_c_pred_norm_arr, dtype=float)

    present_w = y_w_true_arr > 0
    absent_w = ~present_w
    present_c = y_c_true_norm_arr >= 0

    w_abs_err = np.abs(y_w_true_arr - y_w_pred_arr)
    c_abs_err_norm = np.abs(y_c_true_norm_arr - y_c_pred_norm_arr)

    label_to_index = {label: j for j, label in enumerate(labels)}
    mono_idx = [label_to_index[x] for x in ["LDPE", "PE", "PP"] if x in label_to_index]
    copo_idx = [
        j for j, label in enumerate(labels) if label not in ["LDPE", "PE", "PP"]
    ]

    f_out.write(f"Samples: {y_w_true_arr.shape[0]}\n")
    f_out.write(f"Copolymer channels: {', '.join(labels)}\n")
    f_out.write(
        f"Primary presence threshold for predicted weights: {weight_threshold:g}\n"
    )
    f_out.write(
        f"Secondary presence threshold for diagnostics: {secondary_threshold:g}\n\n"
    )

    f_out.write("Aggregate supervised metrics\n")
    f_out.write("----------------------------\n")
    f_out.write(
        f"Weight MAE, all entries [diagnostic, absence-dominated]: "
        f"{fmt(safe_mean(w_abs_err))}\n"
    )
    f_out.write(
        f"Weight RMSE, all entries [diagnostic, absence-dominated]: "
        f"{fmt(safe_rmse(y_w_true_arr, y_w_pred_arr))}\n"
    )
    f_out.write(
        f"Weight MAE, true-present entries only: "
        f"{fmt(safe_mean(w_abs_err[present_w]))}\n"
    )
    f_out.write(
        f"Weight MAE, true-absent entries only: "
        f"{fmt(safe_mean(w_abs_err[absent_w]))}\n"
    )
    f_out.write(
        f"Composition MAE, normalized present entries: "
        f"{fmt(safe_mean(c_abs_err_norm[present_c]))}\n"
    )
    f_out.write(
        f"Composition RMSE, normalized present entries: "
        f"{fmt(safe_rmse(y_c_true_norm_arr[present_c], y_c_pred_norm_arr[present_c]))}\n"
    )

    if mono_idx:
        present_c_mono = present_c[:, mono_idx]
        f_out.write(
            f"Composition MAE, normalized mono channels present "
            f"({', '.join([labels[j] for j in mono_idx])}): "
            f"{fmt(safe_mean(c_abs_err_norm[:, mono_idx][present_c_mono]))}\n"
        )

    if copo_idx:
        present_c_copo = present_c[:, copo_idx]
        f_out.write(
            f"Composition MAE, normalized copolymer channels present "
            f"({', '.join([labels[j] for j in copo_idx])}): "
            f"{fmt(safe_mean(c_abs_err_norm[:, copo_idx][present_c_copo]))}\n"
        )

    pred_present_primary = y_w_pred_arr >= weight_threshold
    negative_norm_when_pred_present = pred_present_primary & (y_c_pred_norm_arr < 0)

    f_out.write(
        f"Predicted-present entries with negative NORMALIZED composition "
        f"at threshold {weight_threshold:g}: "
        f"{int(np.sum(negative_norm_when_pred_present))} / "
        f"{int(np.sum(pred_present_primary))}\n"
    )

    if y_c_true_denorm_arr is not None and y_c_pred_denorm_arr is not None:
        y_c_true_denorm_arr = np.asarray(y_c_true_denorm_arr, dtype=float)
        y_c_pred_denorm_arr = np.asarray(y_c_pred_denorm_arr, dtype=float)
        c_abs_err_denorm = np.abs(y_c_true_denorm_arr - y_c_pred_denorm_arr)

        f_out.write(
            f"Composition MAE, denormalized present entries: "
            f"{fmt(safe_mean(c_abs_err_denorm[present_c]))}\n"
        )
        f_out.write(
            f"Composition RMSE, denormalized present entries: "
            f"{fmt(safe_rmse(y_c_true_denorm_arr[present_c], y_c_pred_denorm_arr[present_c]))}\n"
        )

        if copo_idx:
            present_c_copo = present_c[:, copo_idx]
            f_out.write(
                f"Composition MAE, denormalized copolymer channels present "
                f"({', '.join([labels[j] for j in copo_idx])}): "
                f"{fmt(safe_mean(c_abs_err_denorm[:, copo_idx][present_c_copo]))}\n"
            )

    f_out.write("\nPresence / detection metrics\n")
    f_out.write("----------------------------\n")

    true_presence, pred_presence, exact_match = write_presence_metrics_at_threshold(
        f_out,
        y_w_true_arr,
        y_w_pred_arr,
        weight_threshold,
        prefix="",
    )

    if secondary_threshold is not None and secondary_threshold != weight_threshold:
        f_out.write("\n")
        write_presence_metrics_at_threshold(
            f_out,
            y_w_true_arr,
            y_w_pred_arr,
            secondary_threshold,
            prefix="",
        )

    f_out.write("\nPer-copolymer metrics\n")
    f_out.write("---------------------\n")
    f_out.write(
        "label\t"
        "support_present\t"
        "w_mae_present\t"
        "w_mae_absent\t"
        "w_fp_rate_absent\t"
        "w_fn_rate_present\t"
        "c_mae_norm_present\t"
        "c_mae_denorm_present\t"
        "pred_present_negative_c_norm\n"
    )

    for j, label in enumerate(labels):
        w_true_j = y_w_true_arr[:, j]
        w_pred_j = y_w_pred_arr[:, j]
        c_true_j = y_c_true_norm_arr[:, j]
        c_pred_j = y_c_pred_norm_arr[:, j]

        present_w_j = w_true_j > 0
        absent_w_j = ~present_w_j
        present_c_j = c_true_j >= 0
        pred_present_j = w_pred_j >= weight_threshold

        c_mae_denorm = np.nan
        if y_c_true_denorm_arr is not None and y_c_pred_denorm_arr is not None:
            c_mae_denorm = safe_mean(
                np.abs(y_c_true_denorm_arr[:, j] - y_c_pred_denorm_arr[:, j])[
                    present_c_j
                ]
            )

        f_out.write(
            f"{label}\t"
            f"{int(np.sum(present_w_j))}\t"
            f"{fmt(safe_mean(np.abs(w_true_j - w_pred_j)[present_w_j]))}\t"
            f"{fmt(safe_mean(np.abs(w_true_j - w_pred_j)[absent_w_j]))}\t"
            f"{fmt(np.mean(w_pred_j[absent_w_j] >= weight_threshold) if np.any(absent_w_j) else np.nan)}\t"
            f"{fmt(np.mean(w_pred_j[present_w_j] < weight_threshold) if np.any(present_w_j) else np.nan)}\t"
            f"{fmt(safe_mean(np.abs(c_true_j - c_pred_j)[present_c_j]))}\t"
            f"{fmt(c_mae_denorm)}\t"
            f"{int(np.sum(pred_present_j & (c_pred_j < 0)))}\n"
        )

    sample_w_mae = np.mean(w_abs_err, axis=1)

    c_err_present_only = np.where(present_c, c_abs_err_norm, 0.0)
    n_present_c_per_sample = np.sum(present_c, axis=1)

    sample_c_mae = np.divide(
        np.sum(c_err_present_only, axis=1),
        n_present_c_per_sample,
        out=np.zeros(y_w_true_arr.shape[0], dtype=float),
        where=n_present_c_per_sample > 0,
    )

    combined_score = sample_w_mae + sample_c_mae
    order = np.argsort(combined_score)[::-1][: min(detail_top_k, len(combined_score))]

    f_out.write(
        f"\nWorst {len(order)} samples by weight_MAE + normalized composition_MAE\n"
    )
    f_out.write("--------------------------------------------------------------\n")
    f_out.write(
        "sample\ttrue_label\texact_match\tweight_mae\tcomposition_mae_norm\t"
        "true_components\tpred_components\n"
    )

    for i in order:
        true_components = [labels[j] for j in np.where(true_presence[i] == 1)[0]]
        pred_components = [labels[j] for j in np.where(pred_presence[i] == 1)[0]]

        sample_name = sample_names[i] if sample_names is not None else str(i)
        sample_label = (
            sample_labels[i] if sample_labels is not None else "+".join(true_components)
        )

        f_out.write(
            f"{sample_name}\t"
            f"{sample_label}\t"
            f"{bool(exact_match[i])}\t"
            f"{fmt(sample_w_mae[i])}\t"
            f"{fmt(sample_c_mae[i])}\t"
            f"{'+'.join(true_components) if true_components else 'NONE'}\t"
            f"{'+'.join(pred_components) if pred_components else 'NONE'}\n"
        )


def write_unknown_split_report(
    f_out,
    labels,
    y_w_pred_arr,
    y_c_pred_norm_arr,
    y_c_pred_denorm_arr,
    unknown_df,
    weight_thresholds=(1e-2, 5e-2),
):
    """Write prediction-only diagnostics for the unknown/industrial split."""
    f_out.write(f"\n{'=' * 90}\n")
    f_out.write("UNKNOWN / INDUSTRIAL TEST SET\n")
    f_out.write(f"{'=' * 90}\n")

    if unknown_df is None or y_w_pred_arr is None:
        f_out.write("Unknown test set not available.\n")
        return

    y_w_pred_arr = np.asarray(y_w_pred_arr, dtype=float)
    y_c_pred_norm_arr = np.asarray(y_c_pred_norm_arr, dtype=float)
    y_c_pred_denorm_arr = np.asarray(y_c_pred_denorm_arr, dtype=float)

    f_out.write(f"Samples: {len(unknown_df)}\n")
    f_out.write(f"Copolymer channels: {', '.join(labels)}\n\n")

    for threshold in weight_thresholds:
        pred_presence = y_w_pred_arr >= threshold
        n_components = np.sum(pred_presence, axis=1)

        f_out.write(
            f"Prediction sparsity/internal consistency at weight threshold {threshold:g}\n"
        )
        f_out.write(f"  Mean predicted components: {fmt(np.mean(n_components))}\n")
        f_out.write(f"  Median predicted components: {fmt(np.median(n_components))}\n")
        f_out.write(
            f"  Min/max predicted components: "
            f"{int(np.min(n_components))} / {int(np.max(n_components))}\n"
        )
        f_out.write(
            f"  Samples with no predicted component: "
            f"{int(np.sum(n_components == 0))}\n"
        )
        f_out.write(
            f"  Entries with predicted-present negative NORMALIZED composition: "
            f"{int(np.sum(pred_presence & (y_c_pred_norm_arr < 0)))}\n"
        )
        f_out.write(
            f"  Entries with predicted-present negative DENORMALIZED composition: "
            f"{int(np.sum(pred_presence & (y_c_pred_denorm_arr < 0)))}\n\n"
        )

    f_out.write("Mean predicted weights by copolymer\n")
    f_out.write("----------------------------------\n")

    for j, label in enumerate(labels):
        f_out.write(
            f"{label}: "
            f"mean_w={fmt(np.mean(y_w_pred_arr[:, j]))}, "
            f"std_w={fmt(np.std(y_w_pred_arr[:, j]))}, "
            f"max_w={fmt(np.max(y_w_pred_arr[:, j]))}\n"
        )

    f_out.write("\nDetailed unknown predictions, threshold 1e-2\n")
    f_out.write("------------------------------------------\n")

    for i, idx in enumerate(unknown_df.index):
        true_tuple = (
            unknown_df.loc[idx, "copo_tuple"]
            if "copo_tuple" in unknown_df.columns
            else "NA"
        )
        true_w = unknown_df.loc[idx, "w"] if "w" in unknown_df.columns else "NA"
        true_c = unknown_df.loc[idx, "c"] if "c" in unknown_df.columns else "NA"

        f_out.write(f"{idx} {true_tuple} {true_w} {true_c}\n")

        for j, label in enumerate(labels):
            if y_w_pred_arr[i, j] > PLOT_COMPONENT_THRESHOLD:
                flag = ""

                if y_c_pred_norm_arr[i, j] < 0:
                    flag += "  [NEGATIVE_NORM_COMPOSITION]"

                if y_c_pred_denorm_arr[i, j] < 0:
                    flag += "  [NEGATIVE_DENORM_COMPOSITION]"

                f_out.write(
                    f"{label} -> "
                    f"w: {y_w_pred_arr[i, j]:.4f} "
                    f"c_norm: {y_c_pred_norm_arr[i, j]:.4f} "
                    f"c_denorm: {y_c_pred_denorm_arr[i, j]:.4f}"
                    f"{flag}\n"
                )

        f_out.write("---------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NMR model evaluation script with overrides"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="model_weights_kl_mse_loss_composition_neg2_mse_hybrid",
        help="Name of the model to be evaluated.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help=(
            "Suffix used to load fine-tuned weights, without leading underscore, "
            "e.g. 'ft'. Use empty string for the base model."
        ),
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="DATASET/test_data.pkl",
        help="Dataset used to test the model.",
    )
    parser.add_argument(
        "--test_dataset_unknown",
        type=str,
        default="DATASET/test_data_industrial.pkl",
        help="Optional dataset with unknown/industrial samples.",
    )

    args, _ = parser.parse_known_args()
    print(args)

    # %% OVERALL PARAMETERS

    opts = dict2obj(
        dict(
            copolymer_list=LABELS,
            run_name=args.model_name,
            suffix=normalize_suffix(args.suffix),
            n_val_samples=10000,
        )
    )

    labels = opts.copolymer_list
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    plot_dir = get_plot_dir(opts.run_name, opts.suffix)

    # %% IMPORT DATA

    scaler = pd.read_pickle(f"val_sets/{opts.run_name}/scaler.pkl")

    try:
        X_val = np.load(f"val_sets/{opts.run_name}/X_val.npy")
        y_w_val = np.load(f"val_sets/{opts.run_name}/y_w_val.npy")
        y_c_val = np.load(f"val_sets/{opts.run_name}/y_c_val.npy")
        y_c_norm_val = np.load(f"val_sets/{opts.run_name}/y_c_norm_val.npy")
    except FileNotFoundError:
        X_val = None
        y_w_val = None
        y_c_val = None
        y_c_norm_val = None

    test_data = pd.read_pickle(args.test_dataset)

    if args.test_dataset_unknown:
        test_data_unknown = pd.read_pickle(args.test_dataset_unknown)
    else:
        test_data_unknown = None

    # %% IMPORT MODEL

    parent_directory = os.path.abspath(os.path.join("models", os.pardir))
    if parent_directory not in sys.path:
        sys.path.insert(0, parent_directory)

    module_name = f"models.{opts.run_name}.model"
    module = importlib.import_module(module_name)

    if "simply" in opts.run_name:
        CustomModel = getattr(module, "CustomModelSimplified")
    else:
        CustomModel = getattr(module, "CustomModel")

    n_outputs = y_c_norm_val.shape[1] if y_c_norm_val is not None else len(labels)

    loaded_model = CustomModel(n_outputs=n_outputs)

    if X_val is not None:
        loaded_model(X_val[:1])
    else:
        X_dummy = np.array(test_data.values[:1, 4:]).astype(np.float32)
        loaded_model(X_dummy[..., np.newaxis])

    loaded_model.summary()

    weights_path = get_weights_path(opts.run_name, opts.suffix)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    print(f"Loading weights from: {weights_path}")
    loaded_model.load_weights(weights_path)

    # %% ANALYSIS ON CONVERGENCE

    history = None

    try:
        history = load_history(opts.run_name, opts.suffix)

        plot_history_with_lr(
            history,
            "loss",
            "val_loss",
            "Loss",
            "Loss Convergence vs Learning Rate",
            "loss.png",
            plot_dir,
        )

        plot_history_with_lr(
            history,
            "weight_output_loss",
            "val_weight_output_loss",
            "Weight Loss",
            "Weight Loss Convergence vs Learning Rate",
            "weight_loss.png",
            plot_dir,
        )

        plot_history_with_lr(
            history,
            "composition_output_loss",
            "val_composition_output_loss",
            "Composition Loss",
            "Composition Loss Convergence vs Learning Rate",
            "composition_loss.png",
            plot_dir,
        )

        plot_metric_history_with_lr(
            history,
            lambda key: "test" in key and "weight" in key and "mae" in key,
            "MAE (Test Weights)",
            "MAE on Test Weights vs Learning Rate",
            "test_mae_weights.png",
            plot_dir,
        )

        plot_metric_history_with_lr(
            history,
            lambda key: "test" in key and "composition" in key and "mae" in key,
            "MAE (Test Compositions)",
            "MAE on Test Compositions vs Learning Rate",
            "test_mae_compositions.png",
            plot_dir,
        )

    except FileNotFoundError:
        print("No training history found.")
    except Exception as e:
        print(f"Could not plot training history: {e}")

    # %% ANALYSIS ON VALIDATION SET

    if X_val is not None:
        y_pred_val = loaded_model.predict(X_val[: opts.n_val_samples, :, :])

        y_w_pred_val = y_pred_val["weight_output"]
        y_c_pred_val = y_pred_val["composition_output"]

        y_w_true_val = y_w_val[: opts.n_val_samples]
        y_c_true_val_norm = y_c_norm_val[: opts.n_val_samples]
        y_c_true_val_denorm = y_c_val[: opts.n_val_samples]

        y_p_true_val = (y_w_true_val >= VAL_PRESENCE_THRESHOLD).astype(int)
        y_p_pred_val = (y_w_pred_val >= VAL_PRESENCE_THRESHOLD).astype(int)

        cm_val, class_labels_val, _, _ = validation_confusion_from_presence(
            y_p_true_val,
            y_p_pred_val,
            labels,
            max_components_to_plot=2,
        )

        plot_confusion_matrix(
            cm_val,
            class_labels_val,
            "confusion_matrix_val.png",
            plot_dir,
        )

        plot_channel_series_grid(
            y_w_true_val,
            y_w_pred_val,
            labels,
            "Normalized Weight",
            "weights_val.png",
            plot_dir,
            valid_mask_mode="weight",
            point_alpha=0.5,
            row_height=3.5,
            suptitle=None,
            title_metric_name="AvgMAE",
            use_display_label=True,
        )

        plot_channel_series_grid(
            y_c_true_val_norm,
            y_c_pred_val,
            labels,
            "Normalized Composition",
            "composition_val.png",
            plot_dir,
            valid_mask_mode="composition",
            point_alpha=0.5,
            row_height=3.5,
            suptitle=None,
            title_metric_name="AvgMAE",
            use_display_label=True,
        )

    else:
        print("No validation data available for analysis.")
        y_w_pred_val = None
        y_c_pred_val = None
        y_w_true_val = None
        y_c_true_val_norm = None
        y_c_true_val_denorm = None

    # %% ANALYSIS ON TEST SET

    X_test = np.array(test_data.values[:, 4:]).astype(np.float32)

    if X_val is not None:
        assert X_test.shape[1] == X_val.shape[1]

    if "norm" in opts.run_name:
        scaler_spectra = pd.read_pickle(f"val_sets/{opts.run_name}/scaler_spectra.pkl")
        X_test = scaler_spectra.transform(X_test.flatten().reshape(-1, 1)).reshape(
            X_test.shape
        )

    y_pred_test = loaded_model.predict(X_test[..., np.newaxis])

    y_w_pred_test = y_pred_test["weight_output"]
    y_c_pred_test = y_pred_test["composition_output"]

    y_w_true_test, y_c_true_test = build_targets_from_dataframe(test_data, labels)

    y_c_true_test_norm = normalize_compositions(y_c_true_test, scaler)
    y_c_true_test_norm = np.where(y_w_true_test != 0, y_c_true_test_norm, -1)

    y_c_pred_test_denorm = denormalize_compositions(y_c_pred_test, scaler)

    y_labels = test_data.values[:, 0]
    y_p_pred_test = (y_w_pred_test >= TEST_PRESENCE_THRESHOLD).astype(int)

    cm_test, class_labels_test, y_true_indices, y_pred_indices = (
        test_confusion_from_presence(
            y_labels,
            y_p_pred_test,
            labels,
        )
    )

    plot_confusion_matrix(
        cm_test,
        class_labels_test,
        "confusion_matrix_test.png",
        plot_dir,
    )

    # %% CLASSIFICATION REPORT

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_indices,
        y_pred_indices,
        labels=range(len(class_labels_test)),
        zero_division=0,
    )

    cm_diag = np.diag(cm_test)

    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy = np.divide(cm_diag, cm_test.sum(axis=1))
        accuracy[np.isnan(accuracy)] = 0.0

    df_report = pd.DataFrame(
        {
            "Label": class_labels_test,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Accuracy": accuracy,
            "Support": support,
        }
    )

    df_report_filtered = df_report[df_report["Support"] > 0].reset_index(drop=True)

    print(df_report_filtered)

    metrics = ["Precision", "Recall", "F1-score", "Accuracy"]
    plot_metric_radar(df_report_filtered, metrics, plot_dir, multi_plot=True)

    # %% TEST WEIGHT AND COMPOSITION ANALYSIS

    plot_channel_series_grid(
        y_w_true_test,
        y_w_pred_test,
        labels,
        "Normalized Weight",
        "weights_test.png",
        plot_dir,
        valid_mask_mode="weight",
        point_alpha=0.75,
        row_height=4,
        suptitle="Weight Analysis on Test Set",
        title_metric_name="AvgError",
        use_display_label=False,
    )

    plot_channel_series_grid(
        y_c_true_test_norm,
        y_c_pred_test,
        labels,
        "Normalized Composition",
        "composition_test.png",
        plot_dir,
        valid_mask_mode="composition",
        point_alpha=0.75,
        row_height=4,
        suptitle="Composition Analysis on Test Set",
        title_metric_name="AvgError",
        use_display_label=False,
    )

    # %% INDIVIDUAL WEIGHT PLOTS

    for copo in labels:
        pos = label_to_index[copo]

        y_w_pred_test_copo = y_w_pred_test[:, pos]
        y_w_true_test_copo = y_w_true_test[:, pos]

        idxs = np.where(y_w_true_test_copo > 0)[0]
        y_w_true_test_copo = y_w_true_test_copo[idxs]
        y_w_pred_test_copo = y_w_pred_test_copo[idxs]

        order = np.argsort(y_w_true_test_copo)
        y_w_true_test_copo = y_w_true_test_copo[order]
        y_w_pred_test_copo = y_w_pred_test_copo[order]

        if len(y_w_true_test_copo) == 0:
            continue

        mask_mix = y_w_true_test_copo < 1.0
        mask_single = ~mask_mix

        fig, ax = plt.subplots(figsize=(6, 4))

        y5_lower = y_w_true_test_copo * 0.95
        y5_upper = y_w_true_test_copo * 1.05
        y10_lower = y_w_true_test_copo * 0.90
        y10_upper = y_w_true_test_copo * 1.10

        x_vals = np.arange(len(y_w_true_test_copo))

        ax.fill_between(
            x_vals,
            y10_lower,
            y10_upper,
            color="blue",
            alpha=0.05,
            label="±10% Band",
        )
        ax.fill_between(
            x_vals,
            y5_lower,
            y5_upper,
            color="blue",
            alpha=0.075,
            label="±5% Band",
        )

        ax.plot(
            x_vals[: mask_mix.sum()],
            y_w_pred_test_copo[mask_mix],
            "^",
            alpha=0.75,
            label="Predicted (mix)",
            color="tab:orange",
            markersize=5,
        )
        ax.plot(
            x_vals[mask_mix.sum() :],
            y_w_pred_test_copo[mask_single],
            ".",
            alpha=0.75,
            label="Predicted (mono)",
            color="tab:orange",
        )

        ax.plot(y_w_true_test_copo, label="True", color="tab:blue")

        avg_error = np.mean(np.abs(y_w_true_test_copo - y_w_pred_test_copo))
        ax.set_title(f"{mapped_label(copo)} - AvgMAE: {avg_error:.3f}", fontsize=14)
        ax.set_xlabel("Experiment", fontsize=14)
        ax.set_ylabel("Normalized Weight", fontsize=14)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=12)

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/weights_{copo}.png", dpi=150)
        plt.show()

    # %% COMPOSITION REGRESSION ANALYSIS

    n_labels = len(labels)
    n_cols = math.ceil(np.sqrt(n_labels))
    n_rows = math.ceil(n_labels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    plt.suptitle("Composition Analysis on Test Set - Regression", fontsize=18, y=1)
    axes = axes.flatten()

    plot_idx = 0

    for copo in labels:
        if copo in HOMOPOLYMER_LIKE:
            continue

        pos = label_to_index[copo]
        y_c_pred_test_copo = y_c_pred_test[:, pos]
        y_c_true_test_copo = y_c_true_test_norm[:, pos]

        idxs = np.where(y_c_true_test_copo > 0)[0]
        y_c_true_test_copo = y_c_true_test_copo[idxs]
        y_c_pred_test_copo = y_c_pred_test_copo[idxs]

        if len(y_c_true_test_copo) == 0:
            continue

        ax = axes[plot_idx]
        plot_idx += 1

        ax.scatter(
            y_c_true_test_copo,
            y_c_pred_test_copo,
            alpha=0.75,
            color="tab:orange",
        )

        y5_lower = y_c_true_test_copo * 0.95
        y5_upper = y_c_true_test_copo * 1.05
        y10_lower = y_c_true_test_copo * 0.90
        y10_upper = y_c_true_test_copo * 1.10
        y15_lower = y_c_true_test_copo * 0.85
        y15_upper = y_c_true_test_copo * 1.15
        y20_lower = y_c_true_test_copo * 0.80
        y20_upper = y_c_true_test_copo * 1.20

        sort_idx = np.argsort(y_c_true_test_copo)
        y_sorted = y_c_true_test_copo[sort_idx]

        ax.fill_between(
            y_sorted,
            y5_lower[sort_idx],
            y5_upper[sort_idx],
            color="blue",
            alpha=0.1,
            label="±5% Error",
        )
        ax.fill_between(
            y_sorted,
            y10_lower[sort_idx],
            y10_upper[sort_idx],
            color="blue",
            alpha=0.05,
            label="±10% Error",
        )
        ax.fill_between(
            y_sorted,
            y15_lower[sort_idx],
            y15_upper[sort_idx],
            color="blue",
            alpha=0.03,
            label="±15% Error",
        )
        ax.fill_between(
            y_sorted,
            y20_lower[sort_idx],
            y20_upper[sort_idx],
            color="blue",
            alpha=0.01,
            label="±20% Error",
        )

        if len(y_c_true_test_copo) >= 2:
            slope, intercept = np.polyfit(y_c_true_test_copo, y_c_pred_test_copo, 1)
            regression_line = slope * y_c_true_test_copo + intercept
            ax.plot(
                y_c_true_test_copo,
                regression_line,
                color="tab:orange",
                label="Regression Line",
            )

        ax.plot(
            y_c_true_test_copo,
            y_c_true_test_copo,
            color="tab:blue",
            label="Ideal Line",
        )

        avg_error = np.mean(np.abs(y_c_true_test_copo - y_c_pred_test_copo))
        ax.set_title(f"{copo} - AvgError: {avg_error:.2f}", fontsize=16)
        ax.set_xlabel("True", fontsize=14)
        ax.set_ylabel("Predicted", fontsize=14)
        ax.legend(fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=12)

    for idx in range(plot_idx, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/comps_regression_test.png")
    plt.show()

    # %% PARITY JOINTPLOTS

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)

    def parity_jointplot(y_true, mask_mix, y_pred, copo_name):
        g = sns.JointGrid(x=y_true, y=y_pred, height=6)

        pure_mask = ~mask_mix

        g.ax_joint.scatter(
            y_true[pure_mask],
            y_pred[pure_mask],
            color="tab:orange",
            alpha=0.6,
            label="Predicted (mono)",
            marker="o",
            s=25,
        )
        g.ax_joint.scatter(
            y_true[mask_mix],
            y_pred[mask_mix],
            color="tab:orange",
            alpha=0.6,
            label="Predicted (mix)",
            marker="^",
            s=30,
        )

        g.ax_joint.plot(y_true, y_true, color="tab:blue", label="Ideal Line")

        y_sorted = np.sort(np.concatenate([y_true, y_pred]))

        for pct, alpha in zip([10, 20], [0.1, 0.05]):
            lower = y_sorted * (1 - pct / 100)
            upper = y_sorted * (1 + pct / 100)
            g.ax_joint.fill_between(
                y_sorted,
                lower,
                upper,
                color="blue",
                alpha=alpha,
                label=f"±{pct}% Error Band",
            )

        sns.histplot(
            y_true,
            bins=20,
            kde=True,
            ax=g.ax_marg_x,
            color="tab:blue",
            alpha=0.6,
            stat="density",
            fill=True,
        )
        sns.histplot(
            y=y_pred,
            bins=20,
            kde=True,
            ax=g.ax_marg_y,
            color="tab:orange",
            alpha=0.6,
            stat="density",
            fill=True,
            orientation="horizontal",
        )

        g.ax_joint.set_xlim(-0.1, 1.1)
        g.ax_joint.set_ylim(-0.1, 1.1)
        g.ax_joint.set_xlabel("True Composition", fontsize=14)
        g.ax_joint.set_ylabel("Predicted Composition", fontsize=14)
        g.ax_joint.legend(loc="upper left", fontsize=10)
        g.figure.suptitle(
            f"{mapped_label(copo_name)} - Parity Plot",
            fontsize=16,
            y=0.95,
        )

        sns.despine()
        plt.tight_layout()
        g.figure.savefig(
            f"{plot_dir}/{copo_name}_parity_plot.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

    for copo in labels:
        if copo in HOMOPOLYMER_LIKE:
            continue

        pos = label_to_index[copo]
        y_true = y_c_true_test_norm[:, pos]
        y_pred = y_c_pred_test[:, pos]

        mask_mix = y_w_true_test[:, pos] < 1.0

        idxs = np.where(y_true > 0)[0]
        y_true = y_true[idxs]
        y_pred = y_pred[idxs]
        mask_mix = mask_mix[idxs]

        if len(y_true) > 0:
            parity_jointplot(y_true, mask_mix, y_pred, copo)

    # %% ERROR CORRELATION

    error_matrix = {}

    for copo in labels:
        pos = label_to_index[copo]
        y_true = y_c_true_test_norm[:, pos]
        y_pred = y_c_pred_test[:, pos]
        error_matrix[copo] = y_pred - y_true

    error_df = pd.DataFrame(error_matrix)
    corr_matrix = error_df.corr()
    corr_matrix.index = [mapped_label(copo) for copo in corr_matrix.index]
    corr_matrix.columns = [mapped_label(copo) for copo in corr_matrix.columns]

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Error Correlation Matrix (Prediction Error Correlation)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/error_correlation_matrix.png", dpi=150)
    plt.show()

    # %% SINGLE-COPOLYMER TEST PLOTS

    cmap = matplotlib.colormaps.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(labels))]
    label_to_color = dict(zip(labels, colors))

    for copo in labels:
        y_w_true_test_copo = y_w_true_test[:, label_to_index[copo]]
        y_c_true_test_copo = y_c_true_test[:, label_to_index[copo]]

        y_w_pred_test_copo = np.round(y_w_pred_test[:, label_to_index[copo]], 2)
        y_c_pred_test_copo = y_c_pred_test_denorm[:, label_to_index[copo]]

        idxs_copo_single = np.where(y_w_true_test_copo == 1.0)[0]

        if len(idxs_copo_single) == 0:
            continue

        test_data_copo = test_data.iloc[idxs_copo_single].copy()

        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(max(0.3 * len(test_data_copo.index), 12), 12),
        )

        axes[0].vlines(
            test_data_copo.index,
            0,
            1,
            color="black",
            alpha=0.5,
            linewidth=0.5,
        )
        axes[0].plot(
            test_data_copo.index,
            y_w_true_test_copo[idxs_copo_single],
            label=f"True {mapped_label(copo)} weight",
            color=label_to_color[copo],
            linewidth=2,
        )

        for k in range(y_w_pred_test.shape[1]):
            mask = y_w_pred_test[idxs_copo_single, k] > PLOT_COMPONENT_THRESHOLD

            if np.any(mask):
                axes[0].scatter(
                    test_data_copo.index[mask],
                    y_w_pred_test[idxs_copo_single, k][mask],
                    label=f"Predicted {mapped_label(labels[k])} weight",
                    color=label_to_color[labels[k]],
                )

        if copo not in HOMOPOLYMER_LIKE:
            axes[0].tick_params(
                axis="x",
                which="both",
                bottom=False,
                labelbottom=False,
                labelsize=12,
            )
        else:
            axes[0].set_xticks(test_data_copo.index)
            axes[0].set_xticklabels(
                [f"Exp{i + 1}" for i, _ in enumerate(test_data_copo.index)]
            )
            axes[0].tick_params(axis="x", rotation=90, labelsize=12)

        axes[0].tick_params(axis="y", which="major", labelsize=12)
        axes[0].set_title("Weight Plot", fontsize=16)
        axes[0].legend(fontsize=12)
        axes[0].set_ylabel("Normalized Weight", fontsize=14)

        axes[1].vlines(
            test_data_copo.index,
            0,
            max(y_c_true_test_copo),
            color="black",
            alpha=0.5,
            linewidth=0.5,
        )
        axes[1].plot(
            test_data_copo.index,
            y_c_true_test_copo[idxs_copo_single],
            label=f"True {mapped_label(copo)} composition",
            color=label_to_color[copo],
            linewidth=2,
        )
        axes[1].scatter(
            test_data_copo.index,
            y_c_pred_test_copo[idxs_copo_single],
            color=label_to_color[copo],
            label=f"Predicted {mapped_label(copo)} composition",
        )

        axes[1].set_xticks(test_data_copo.index)
        axes[1].set_xticklabels(
            [f"Exp{i + 1}" for i, _ in enumerate(test_data_copo.index)]
        )
        axes[1].tick_params(axis="x", rotation=90, labelsize=12)
        axes[1].tick_params(axis="y", which="major", labelsize=12)
        axes[1].set_title("Composition Plot", fontsize=16)
        axes[1].set_ylabel("Composition", fontsize=14)
        axes[1].legend(fontsize=12)

        plt.tight_layout()
        os.makedirs(f"{plot_dir}/SingleCopos", exist_ok=True)
        plt.savefig(f"{plot_dir}/SingleCopos/single_{copo}_test.png", dpi=150)
        plt.show()

    # %% MIXTURE TEST PLOTS

    possible_mixes = [mix for mix in test_data["copo_tuple"].unique() if len(mix) > 1]

    for mix_tuple in possible_mixes:
        check_mix = np.ones(y_w_true_test.shape[0], dtype=bool)

        for copo in mix_tuple:
            check_mix &= y_w_true_test[:, label_to_index[copo]] > 0.0

        mix = "+".join(mix_tuple)
        test_data_mix = test_data[test_data["copolymer"] == mix]
        idxs_mix = np.where(check_mix)[0]

        if len(idxs_mix) == 0:
            continue

        y_w_true_test_mix = y_w_true_test[idxs_mix]
        y_c_true_test_mix = y_c_true_test[idxs_mix]

        y_w_pred_test_mix = y_w_pred_test[idxs_mix]
        y_c_pred_test_mix = y_c_pred_test_denorm[idxs_mix]

        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(max(0.3 * len(test_data_mix.index), 15), 12),
        )

        axes[0].vlines(
            test_data_mix.index,
            0,
            1,
            color="black",
            alpha=0.5,
            linewidth=0.5,
        )

        for k in range(y_w_pred_test_mix.shape[1]):
            true_mask = y_w_true_test_mix[:, k] > 0.0
            pred_mask = y_w_pred_test_mix[:, k] > PLOT_COMPONENT_THRESHOLD

            if np.any(true_mask):
                axes[0].plot(
                    test_data_mix.index[true_mask],
                    y_w_true_test_mix[:, k][true_mask],
                    label=f"True {mapped_label(labels[k])} weight",
                    color=label_to_color[labels[k]],
                    linewidth=1.5,
                )

            if np.any(pred_mask):
                axes[0].scatter(
                    test_data_mix.index[pred_mask],
                    y_w_pred_test_mix[:, k][pred_mask],
                    label=f"Predicted {mapped_label(labels[k])} weight",
                    color=label_to_color[labels[k]],
                )

        axes[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        axes[0].tick_params(axis="y", labelsize=12)
        axes[0].set_title("Weight Plot", fontsize=16)
        axes[0].set_ylabel("Normalized Weight", fontsize=14)
        axes[0].legend(fontsize=12)

        valid_true_vals = y_c_true_test_mix[y_w_true_test_mix > 0.0]

        ymin = valid_true_vals.min() if len(valid_true_vals) > 0 else 0
        ymax = valid_true_vals.max() if len(valid_true_vals) > 0 else 1

        axes[1].vlines(
            test_data_mix.index,
            ymin,
            ymax,
            color="black",
            alpha=0.5,
            linewidth=0.5,
        )

        for k in range(y_w_pred_test_mix.shape[1]):
            true_mask = y_w_true_test_mix[:, k] > 0.0
            pred_mask = y_w_pred_test_mix[:, k] > PLOT_COMPONENT_THRESHOLD

            if np.any(true_mask):
                axes[1].plot(
                    test_data_mix.index[true_mask],
                    y_c_true_test_mix[:, k][true_mask],
                    label=f"True {mapped_label(labels[k])} composition",
                    color=label_to_color[labels[k]],
                    linewidth=1.5,
                )

            if np.any(pred_mask):
                axes[1].scatter(
                    test_data_mix.index[pred_mask],
                    y_c_pred_test_mix[:, k][pred_mask],
                    color=label_to_color[labels[k]],
                    label=f"Predicted {mapped_label(labels[k])} composition",
                )

        mix_copolymers = mix.split("+")
        if any(copo in ["EH", "EO", "EB", "RACO"] for copo in mix_copolymers):
            axes[1].set_ylim(-0.5, 10.5)

        axes[1].tick_params(axis="x", rotation=90, labelsize=12)
        axes[1].set_xticks(test_data_mix.index)
        axes[1].set_xticklabels(
            [f"Exp{i + 1}" for i, _ in enumerate(test_data_mix.index)]
        )
        axes[1].tick_params(axis="y", labelsize=12)
        axes[1].set_title("Composition Plot", fontsize=16)
        axes[1].set_ylabel("Composition", fontsize=14)
        axes[1].legend(fontsize=12)

        plt.tight_layout()
        os.makedirs(f"{plot_dir}/Mixes", exist_ok=True)
        plt.savefig(f"{plot_dir}/Mixes/{mix}_test.png", dpi=150)
        plt.show()

    # %% UNKNOWN / INDUSTRIAL TEST

    y_w_pred_unknown = None
    y_c_pred_unknown = None
    y_c_pred_unknown_denorm = None

    if test_data_unknown is not None:
        X_unknown = np.array(test_data_unknown.values[:, 4:]).astype(np.float32)

        if "norm" in opts.run_name:
            scaler_spectra = pd.read_pickle(
                f"val_sets/{opts.run_name}/scaler_spectra.pkl"
            )
            X_unknown = scaler_spectra.transform(
                X_unknown.flatten().reshape(-1, 1)
            ).reshape(X_unknown.shape)

        y_pred_unknown = loaded_model.predict(X_unknown[..., np.newaxis])

        y_w_pred_unknown = y_pred_unknown["weight_output"]
        y_c_pred_unknown = y_pred_unknown["composition_output"]
        y_c_pred_unknown_denorm = denormalize_compositions(y_c_pred_unknown, scaler)

        output_txt_path = f"{plot_dir}/unknown_test_predictions.txt"

        with open(output_txt_path, "w", encoding="utf-8") as f_out:
            for i, idx in enumerate(test_data_unknown.index):
                true_tuple = (
                    test_data_unknown.loc[idx, "copo_tuple"]
                    if "copo_tuple" in test_data_unknown.columns
                    else "NA"
                )
                true_w = (
                    test_data_unknown.loc[idx, "w"]
                    if "w" in test_data_unknown.columns
                    else "NA"
                )
                true_c = (
                    test_data_unknown.loc[idx, "c"]
                    if "c" in test_data_unknown.columns
                    else "NA"
                )

                f_out.write(f"{idx} {true_tuple} {true_w} {true_c}\n")

                for j in range(len(labels)):
                    if y_w_pred_unknown[i, j] > PLOT_COMPONENT_THRESHOLD:
                        f_out.write(
                            f"{labels[j]} -> "
                            f"w: {y_w_pred_unknown[i, j]:.2f} "
                            f"c: {y_c_pred_unknown_denorm[i, j]:.2f}\n"
                        )

                f_out.write("---------------------------------------------\n")

        print(f"Unknown test predictions saved to: {output_txt_path}")

    else:
        print("No unknown test data available.")

    # %% GLOBAL TEXT REPORT

    global_report_path = f"{plot_dir}/global_evaluation_report{opts.suffix}.txt"

    with open(global_report_path, "w", encoding="utf-8") as f_report:
        f_report.write("GLOBAL MODEL EVALUATION REPORT\n")
        f_report.write("==============================\n")
        f_report.write(f"Model name: {opts.run_name}\n")
        f_report.write(f"Model suffix: {opts.suffix if opts.suffix else '<base>'}\n")
        f_report.write(f"Weights path attempted: {weights_path}\n")
        f_report.write(f"Test dataset: {args.test_dataset}\n")
        f_report.write(f"Unknown dataset: {args.test_dataset_unknown}\n")
        f_report.write(f"Plot/report directory: {plot_dir}\n")

        if history is not None:
            try:
                f_report.write("\nTraining history summary\n")
                f_report.write("------------------------\n")

                for key in [
                    "loss",
                    "val_loss",
                    "weight_output_loss",
                    "val_weight_output_loss",
                    "composition_output_loss",
                    "val_composition_output_loss",
                ]:
                    if key in history:
                        values = np.asarray(history[key], dtype=float)
                        finite_values = values[np.isfinite(values)]

                        if finite_values.size:
                            best_idx = int(np.nanargmin(values))
                            f_report.write(
                                f"{key}: "
                                f"final={fmt(values[-1])}, "
                                f"best={fmt(values[best_idx])} "
                                f"at epoch={best_idx + 1}\n"
                            )

                if "learning_rate" in history:
                    f_report.write(
                        f"final_learning_rate: {fmt(history['learning_rate'][-1], 8)}\n"
                    )

            except Exception as e:
                f_report.write(f"\nTraining history summary unavailable: {e}\n")

        if X_val is not None:
            y_c_pred_val_denorm = denormalize_compositions(y_c_pred_val, scaler)

            write_supervised_split_report(
                f_report,
                "VALIDATION SET",
                labels,
                y_w_true_val,
                y_w_pred_val,
                y_c_true_val_norm,
                y_c_pred_val,
                y_c_true_denorm_arr=y_c_true_val_denorm,
                y_c_pred_denorm_arr=y_c_pred_val_denorm,
                sample_names=[str(i) for i in range(len(y_w_true_val))],
                sample_labels=None,
                weight_threshold=REPORT_PRIMARY_THRESHOLD,
                secondary_threshold=REPORT_SECONDARY_THRESHOLD,
            )
        else:
            write_supervised_split_report(
                f_report,
                "VALIDATION SET",
                labels,
                None,
                None,
                None,
                None,
            )

        write_supervised_split_report(
            f_report,
            "TEST SET",
            labels,
            y_w_true_test,
            y_w_pred_test,
            y_c_true_test_norm,
            y_c_pred_test,
            y_c_true_denorm_arr=y_c_true_test,
            y_c_pred_denorm_arr=y_c_pred_test_denorm,
            sample_names=[str(idx) for idx in test_data.index],
            sample_labels=[
                sample_composition_label(test_data.loc[idx]) for idx in test_data.index
            ],
            weight_threshold=REPORT_PRIMARY_THRESHOLD,
            secondary_threshold=REPORT_SECONDARY_THRESHOLD,
        )

        try:
            f_report.write(
                "\nTest-set classification report used for the confusion matrix\n"
            )
            f_report.write(
                "---------------------------------------------------------\n"
            )
            f_report.write(df_report_filtered.to_string(index=False))
            f_report.write("\n")
        except Exception as e:
            f_report.write(f"\nClassification report unavailable: {e}\n")

        write_unknown_split_report(
            f_report,
            labels,
            y_w_pred_unknown,
            y_c_pred_unknown,
            y_c_pred_unknown_denorm,
            test_data_unknown,
            weight_thresholds=(REPORT_SECONDARY_THRESHOLD, REPORT_PRIMARY_THRESHOLD),
        )

    print(f"Global evaluation report saved to: {global_report_path}")

    # %% SUPPLEMENTARY TABLE - TEST-SET PER-CLASS METRICS

    presence_threshold = REPORT_PRIMARY_THRESHOLD
    supplementary_rows = []

    for label in labels:
        j = label_to_index[label]

        w_true_j = y_w_true_test[:, j]
        w_pred_j = y_w_pred_test[:, j]

        present_w = w_true_j > 0
        absent_w = ~present_w
        pred_present = w_pred_j >= presence_threshold

        support_present = int(np.sum(present_w))
        support_absent = int(np.sum(absent_w))

        w_mae_present = safe_mean(np.abs(w_true_j[present_w] - w_pred_j[present_w]))
        w_rmse_present = safe_rmse(w_true_j[present_w], w_pred_j[present_w])

        w_mae_absent = safe_mean(np.abs(w_true_j[absent_w] - w_pred_j[absent_w]))
        w_rmse_absent = safe_rmse(w_true_j[absent_w], w_pred_j[absent_w])

        w_fp_rate_absent = (
            float(np.mean(w_pred_j[absent_w] >= presence_threshold))
            if np.any(absent_w)
            else np.nan
        )

        w_fn_rate_present = (
            float(np.mean(w_pred_j[present_w] < presence_threshold))
            if np.any(present_w)
            else np.nan
        )

        c_true_norm_j = y_c_true_test_norm[:, j]
        c_pred_norm_j = y_c_pred_test[:, j]

        c_true_denorm_j = y_c_true_test[:, j]
        c_pred_denorm_j = y_c_pred_test_denorm[:, j]

        present_c = c_true_norm_j >= 0

        if label in HOMOPOLYMER_LIKE:
            c_mae_norm_present = np.nan
            c_rmse_norm_present = np.nan
            c_r2_norm_present = np.nan
            c_mae_denorm_present = np.nan
            c_rmse_denorm_present = np.nan
            c_r2_denorm_present = np.nan
        else:
            c_mae_norm_present = safe_mean(
                np.abs(c_true_norm_j[present_c] - c_pred_norm_j[present_c])
            )
            c_rmse_norm_present = safe_rmse(
                c_true_norm_j[present_c],
                c_pred_norm_j[present_c],
            )
            c_r2_norm_present = safe_r2(
                c_true_norm_j[present_c],
                c_pred_norm_j[present_c],
            )

            c_mae_denorm_present = safe_mean(
                np.abs(c_true_denorm_j[present_c] - c_pred_denorm_j[present_c])
            )
            c_rmse_denorm_present = safe_rmse(
                c_true_denorm_j[present_c],
                c_pred_denorm_j[present_c],
            )
            c_r2_denorm_present = safe_r2(
                c_true_denorm_j[present_c],
                c_pred_denorm_j[present_c],
            )

        pred_present_negative_c_norm = int(np.sum(pred_present & (c_pred_norm_j < 0)))
        pred_present_negative_c_denorm = int(
            np.sum(pred_present & (c_pred_denorm_j < 0))
        )

        supplementary_rows.append(
            {
                "label": label,
                "display_label": mapped_label(label),
                "support_present": support_present,
                "support_absent": support_absent,
                "w_mae_present": w_mae_present,
                "w_rmse_present": w_rmse_present,
                "w_mae_absent": w_mae_absent,
                "w_rmse_absent": w_rmse_absent,
                "w_fp_rate_absent": w_fp_rate_absent,
                "w_fn_rate_present": w_fn_rate_present,
                "c_mae_norm_present": c_mae_norm_present,
                "c_rmse_norm_present": c_rmse_norm_present,
                "c_r2_norm_present": c_r2_norm_present,
                "c_mae_denorm_present": c_mae_denorm_present,
                "c_rmse_denorm_present": c_rmse_denorm_present,
                "c_r2_denorm_present": c_r2_denorm_present,
                "pred_present_negative_c_norm": pred_present_negative_c_norm,
                "pred_present_negative_c_denorm": pred_present_negative_c_denorm,
            }
        )

    df_supp_metrics = pd.DataFrame(supplementary_rows)

    supp_cols = [
        "display_label",
        "support_present",
        "support_absent",
        "w_mae_present",
        "w_rmse_present",
        "w_mae_absent",
        "w_rmse_absent",
        "w_fp_rate_absent",
        "w_fn_rate_present",
        "c_mae_norm_present",
        "c_rmse_norm_present",
        "c_r2_norm_present",
        "c_mae_denorm_present",
        "c_rmse_denorm_present",
        "c_r2_denorm_present",
        "pred_present_negative_c_norm",
        "pred_present_negative_c_denorm",
    ]

    print("\n" + "=" * 120)
    print("SUPPLEMENTARY TABLE - COMPLETE TEST-SET PER-CLASS METRICS")
    print("=" * 120)
    print(f"Presence threshold used for FP/FN diagnostics: {presence_threshold:g}")
    print("-" * 120)

    print(
        df_supp_metrics[supp_cols].to_string(
            index=False,
            formatters={
                "w_mae_present": fmt,
                "w_rmse_present": fmt,
                "w_mae_absent": fmt,
                "w_rmse_absent": fmt,
                "w_fp_rate_absent": fmt,
                "w_fn_rate_present": fmt,
                "c_mae_norm_present": fmt,
                "c_rmse_norm_present": fmt,
                "c_r2_norm_present": fmt,
                "c_mae_denorm_present": fmt,
                "c_rmse_denorm_present": fmt,
                "c_r2_denorm_present": fmt,
            },
        )
    )
    print("=" * 120 + "\n")
