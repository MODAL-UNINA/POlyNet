#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:10:53 2024

@author: MODAL
"""

# %%
import os
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
    confusion_matrix
)
from scipy.interpolate import interp1d

from utils import MappingNames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NMR training script with overrides")

    parser.add_argument(
        "--model_name",
        type=str,
        default="model_weights_kl_mse_loss_composition_neg_mse_model_norm",
        help="Name of the model to be evaluated."
    )

    parser.add_argument(
        "--fine-tuned",
        action="store_true",
        help="Whether to use a fine-tuned model."
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        default="DATASET/test_data.pkl",
        help="Which dataset should be used to test the model.",
    )

    parser.add_argument(
        "--test_dataset_unknown",
        type=str,
        default=None,
        help="Path to a dataset with unknown copolymers for testing.",
    )

    args, _ = parser.parse_known_args()
    
    # %% OVERALL PARAMETERS
    opts = dict2obj(
        dict(
            copolymer_list=["LDPE", "PE", "PP", "EH", "EO", "EB", "RACO", "EPR"],
            run_name=args.model_name,
            fine_tuned=args.fine_tuned,
            n_val_samples=10000,  # Number of validation samples to use for analysis
        )
    )

    PLOT_DIR = f"OUTPUT/eval_models/{opts.run_name}/plots"
    if opts.fine_tuned:
        PLOT_DIR += "_ft"

    os.makedirs(PLOT_DIR, exist_ok=True)
    TEST_DATA = "test_data"

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

    test_data = pd.read_pickle(f"DATASET/{TEST_DATA}.pkl")

    if args.test_dataset_unknown:
        test_data_unknown = pd.read_pickle(args.test_dataset_unknown)
    else:
        test_data_unknown = None

    # %% IMPORT MODEL

    # Ensure that 'models' directory is in the parent directory
    parent_directory = os.path.abspath(os.path.join("models", os.pardir))
    if parent_directory not in sys.path:
        sys.path.insert(0, parent_directory)

    # Construct the module name dynamically
    module_name = f"models.{opts.run_name}.model"

    # Import the module dynamically
    module = importlib.import_module(module_name)

    # Access the CustomModel class
    if "simply" in opts.run_name:
        CustomModel = getattr(module, "CustomModelSimplified")
    else:
        CustomModel = getattr(module, "CustomModel")

    # Now you can use CustomModel
    loaded_model = CustomModel(n_outputs=y_c_norm_val.shape[1])

    # loaded_model = custom_objects["CustomModel"]()
    loaded_model(X_val[:1])
    loaded_model.summary()

    if opts.fine_tuned:
        loaded_model.load_weights(f"models/{opts.run_name}/model_ft.weights.h5")
    else:
        loaded_model.load_weights(f"models/{opts.run_name}/model.weights.h5")

    # %% ANALYSIS ON CONVERGENCE

    try:
        if opts.fine_tuned:
            with open(f"models/{opts.run_name}/training_history_ft.json", "r") as f:
                history = json.load(f)
        else:
            with open(f"models/{opts.run_name}/training_history.json", "r") as f:
                history = json.load(f)

        # General Loss Convergence
        fig, ax1 = plt.subplots(figsize=(15, 5))
        ax1.plot(history["loss"], label="Training Loss", color="blue")
        ax1.plot(history["val_loss"], label="Validation Loss", color="orange")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color="blue")
        ax1.set_yscale("log")  # Log scale for loss
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.legend(loc="upper left")

        # Secondary y-axis for Learning Rate
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
        ax2.set_yscale("log")  # Log scale for Learning Rate
        ax2.legend(loc="upper right")

        plt.title("Loss Convergence vs Learning Rate")
        plt.savefig(f"{PLOT_DIR}/loss.png")
        plt.show()

        # Weight Loss Convergence
        fig, ax1 = plt.subplots(figsize=(15, 5))
        ax1.plot(history["weight_output_loss"], label="Training Loss", color="blue")
        ax1.plot(history["val_weight_output_loss"], label="Validation Loss", color="orange")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Weight Loss", color="blue")
        ax1.set_yscale("log")  # Log scale for loss
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.legend(loc="upper left")

        # Secondary y-axis for Learning Rate
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
        ax2.set_yscale("log")  # Log scale for Learning Rate
        ax2.legend(loc="upper right")

        plt.title("Weight Loss Convergence vs Learning Rate")
        plt.savefig(f"{PLOT_DIR}/weight_loss.png")
        plt.show()

        # Composition Loss Convergence
        fig, ax1 = plt.subplots(figsize=(15, 5))
        ax1.plot(history["composition_output_loss"], label="Training Loss", color="blue")
        ax1.plot(
            history["val_composition_output_loss"], label="Validation Loss", color="orange"
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Composition Loss", color="blue")
        ax1.set_yscale("log")  # Log scale for loss
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.legend(loc="upper left")

        # Secondary y-axis for Learning Rate
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
        ax2.set_yscale("log")  # Log scale for Learning Rate
        ax2.legend(loc="upper right")

        plt.title("Composition Loss Convergence vs Learning Rate")
        plt.savefig(f"{PLOT_DIR}/composition_loss.png")
        plt.show()

        plt.figure(figsize=(15, 5))
        # Create primary y-axis for MAE (on the left)
        fig, ax1 = plt.subplots(figsize=(15, 5))
        # Plot MAE on test weights
        for err in [
            k for k in history.keys() if "test" in k and "weight" in k and "mae" in k
        ]:
            ax1.plot(history[err], label=f"{err}", linestyle="-")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MAE (Test Weights)", color="blue")
        ax1.set_yscale("log")  # Log scale for MAE
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.legend(loc="upper left")
        # Create secondary y-axis for Learning Rate (on the right)
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
        ax2.set_yscale("log")  # Log scale for Learning Rate
        ax2.legend(loc="upper right")
        # Title and save
        plt.title("MAE on Test Weights vs Learning Rate")
        plt.savefig(f"{PLOT_DIR}/test_mae_weights.png")
        plt.show()

        plt.figure(figsize=(15, 5))
        # Create primary y-axis for MAE (on the left)
        fig, ax1 = plt.subplots(figsize=(15, 5))
        # Plot MAE on test weights
        for err in [
            k for k in history.keys() if "test" in k and "composition" in k and "mae" in k
        ]:
            err_name = ("_").join(
                err.split("_")[:-1] + [MappingNames()[err.split("_")[-1]]]
            )  # Extract the last part of the key
            ax1.plot(history[err], label=f"{err_name}", linestyle="-")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MAE (Test Compositions)", color="blue")
        ax1.set_yscale("log")  # Log scale for MAE
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.legend(loc="upper left")
        # Create secondary y-axis for Learning Rate (on the right)
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
        ax2.set_yscale("log")  # Log scale for Learning Rate
        ax2.legend(loc="upper right")
        # Title and save
        plt.title("MAE on Test Compositions vs Learning Rate")
        plt.savefig(f"{PLOT_DIR}/test_mae_compositions.png")
        plt.show()

    except FileNotFoundError:
        print("No training history found")

    except Exception as e:
        print(e)

    # %% ANALYSIS ON VAL SET

    if X_val is not None:
        y_pred = loaded_model.predict(X_val[: opts.n_val_samples, :, :])

        y_w_pred = y_pred["weight_output"]
        y_c_pred = y_pred["composition_output"]
        y_p_pred = np.where(y_w_pred < 1e-2, 0, 1)

        y_w_true = y_w_val[: opts.n_val_samples]
        y_c_true = y_c_norm_val[: opts.n_val_samples]
        y_p_true = np.where(y_w_true < 1e-2, 0, 1)


        labels = opts.copolymer_list
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        index_to_label = {idx: label for idx, label in enumerate(labels)}

        # Step 2: Generate all combinations
        single_label_combinations = [(label,) for label in labels]
        two_label_combinations = list(itertools.combinations(labels, 2))
        all_combinations = single_label_combinations + two_label_combinations

        # Step 3: Map combinations to vectors and vice versa
        combination_to_vector = {}
        vector_to_combination = {}
        for combination in all_combinations:
            vector = [0] * len(labels)
            for label in combination:
                idx = label_to_index[label]
                vector[idx] = 1
            vector_tuple = tuple(vector)
            combination_to_vector[combination] = vector_tuple
            vector_to_combination[vector_tuple] = combination

        # Step 4: Create mappings to indices
        combination_to_index = {
            combination: idx for idx, combination in enumerate(all_combinations)
        }
        index_to_combination = {
            idx: combination for idx, combination in enumerate(all_combinations)
        }
        vector_to_index = {
            combination_to_vector[combination]: idx
            for idx, combination in enumerate(all_combinations)
        }
        index_to_vector = {
            idx: combination_to_vector[combination]
            for idx, combination in enumerate(all_combinations)
        }

        # Convert binary vectors to class indices
        y_true_indices = []
        y_pred_indices = []
        for true_vector, pred_vector in zip(y_p_true, y_p_pred):
            true_vector_tuple = tuple(true_vector)
            pred_vector_tuple = tuple(pred_vector)
            true_idx = vector_to_index.get(true_vector_tuple, None)
            pred_idx = vector_to_index.get(pred_vector_tuple, None)
            if true_idx is not None and pred_idx is not None:
                y_true_indices.append(true_idx)
                y_pred_indices.append(pred_idx)
            else:
                # Handle unknown combinations or skip
                pass

        # Compute confusion matrix
        cm = confusion_matrix(
            y_true_indices, y_pred_indices, labels=range(len(all_combinations))
        )

        # Prepare class labels for plotting
        class_labels = [
            " + ".join(MappingNames()[co] for co in sorted(comb))
            for comb in sorted(all_combinations)
        ]

        # Plot the confusion matrix
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
        # plt.title("Confusion Matrix for Multi-Label Classification")
        plt.xticks(rotation=90, fontsize=14)
        plt.yticks(rotation=0, fontsize=14)

        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/confusion_matrix_val.png")
        plt.show()
    else:
        print("No validation data available for analysis.")

    # %% WEIGHTS ANALYSIS
    if X_val is not None:
        # Calculate the number of labels
        n_labels = len(labels)

        # Determine the grid size for subplots (e.g., 2x2, 3x3)
        n_cols = math.ceil(np.sqrt(n_labels))
        n_rows = math.ceil(n_labels / n_cols)

        # Create a figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # WEIGHTS ANALYSIS
        for idx, copo in enumerate(labels):
            pos = label_to_index[copo]
            y_w_true_copo = y_w_true[:, pos]
            y_w_pred_copo = y_w_pred[:, pos]

            idxs = np.where(y_w_true_copo > 0)[0]
            y_w_true_copo = y_w_true_copo[idxs]
            y_w_pred_copo = y_w_pred_copo[idxs]

            # Order y_w_true_copo
            order = np.argsort(y_w_true_copo)
            y_w_true_copo = y_w_true_copo[order]
            y_w_pred_copo = y_w_pred_copo[order]

            # Select the appropriate subplot
            ax = axes[idx]
            ax.plot(y_w_pred_copo, ".", alpha=0.5, label="Predicted", color="tab:orange")
            ax.plot(y_w_true_copo, label="True", color="tab:blue")
            avg_error = np.mean(np.abs(y_w_true_copo - y_w_pred_copo))
            ax.set_title(f"{MappingNames()[copo]} - AvgMAE: {avg_error:.2f}", fontsize=16)
            ax.set_ylim(-0.1, 1.1)
            ax.legend(fontsize=12)

            ax.set_xlabel("Experiment", fontsize=14)
            ax.set_ylabel("Normalized Weight", fontsize=14)
        # Hide any unused subplots if the grid is larger than the number of labels
        for idx in range(len(labels), len(axes)):
            fig.delaxes(axes[idx])

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Display the figure with all subplots
        plt.savefig(f"{PLOT_DIR}/weights_val.png")
        plt.show()
    else:
        print("No validation data available for weights analysis.")

    # %% COMPOSITION ANALYSIS
    if X_val is not None:
        # Create a figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # WEIGHTS ANALYSIS
        for idx, copo in enumerate(labels):
            pos = label_to_index[copo]
            y_c_true_copo = y_c_true[:, pos]
            y_c_pred_copo = y_c_pred[:, pos]

            idxs = np.where(y_c_true_copo > -1)[0]
            y_c_true_copo = y_c_true_copo[idxs]
            y_c_pred_copo = y_c_pred_copo[idxs]

            # Order y_w_true_copo
            order = np.argsort(y_c_true_copo)
            y_c_true_copo = y_c_true_copo[order]
            y_c_pred_copo = y_c_pred_copo[order]

            # Select the appropriate subplot
            ax = axes[idx]
            ax.plot(y_c_pred_copo, ".", alpha=0.5, label="Predicted", color="tab:orange")
            ax.plot(y_c_true_copo, label="True", color="tab:blue")
            avg_error = np.mean(np.abs(y_c_true_copo - y_c_pred_copo))
            ax.set_title(f"{MappingNames()[copo]} - AvgMAE: {avg_error:.2f}", fontsize=16)
            ax.set_ylim(-0.1, 1.1)
            ax.legend(fontsize=12)

            ax.set_xlabel("Experiment", fontsize=14)
            ax.set_ylabel("Normalized Composition", fontsize=14)

        # Hide any unused subplots if the grid is larger than the number of labels
        for idx in range(len(labels), len(axes)):
            fig.delaxes(axes[idx])

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Display the figure with all subplots
        plt.savefig(f"{PLOT_DIR}/composition_val.png")
        plt.show()
    else:
        print("No validation data available for composition analysis.")

    # %% ANALYSIS ON TEST SET - ONLY SINGLE COPOLYMERS

    # Exclude samples with unknown copolymers or compositions

    # test_data = test_data_orig[test_data_orig["w"].apply(lambda x: not any(np.isnan(k) for k in x))]

    X_test = np.array(test_data.values[:, 4:]).astype(np.float32)

    assert X_test.shape[1] == X_val.shape[1]

    if "norm" in opts.run_name:
        with open(f"val_sets/{opts.run_name}/scaler_spectra.pkl", "rb") as f:
            scaler_spectra = pd.read_pickle(f)
        X_test = scaler_spectra.transform(X_test.flatten().reshape(-1, 1)).reshape(
            X_test.shape
        )


    y_pred_test = loaded_model.predict(X_test[..., np.newaxis])


    y_w_pred_test = y_pred_test["weight_output"]
    y_c_pred_test = y_pred_test["composition_output"]
    y_p_pred_test = np.where(y_w_pred_test < 5e-2, 0, 1)

    y_labels = test_data.values[:, 0]


    # Define a function to zero out values based on the condition
    def update_c(row):
        updated_c = tuple(
            0.0 if material == "PP" else value
            for material, value in zip(row["copo_tuple"], row["c"])
        )
        return updated_c



    # Step 3: Build combinations from your true labels
    combinations_in_true = set()
    for label in y_labels:
        labels_in_sample = tuple(sorted(label.split("+")))
        combinations_in_true.add(labels_in_sample)


    # Step 5: Build combinations from your predicted labels
    combinations_in_pred = set()
    for pred_vector in y_p_pred_test:
        labels_in_pred = []
        for idx, val in enumerate(pred_vector):
            if val == 1:
                labels_in_pred.append(labels[idx])  # Use 'labels' for correct order
        pred_combination = tuple(sorted(labels_in_pred))
        combinations_in_pred.add(pred_combination)

    # Step 6: Combine combinations from true labels and predictions
    all_combinations = combinations_in_true.union(combinations_in_pred)

    # Map combinations to unique indices
    combination_to_index = {
        combination: idx for idx, combination in enumerate(sorted(all_combinations))
    }
    index_to_combination = {
        idx: combination for combination, idx in combination_to_index.items()
    }

    # Map combinations to vectors
    combination_to_vector = {}
    for combination in all_combinations:
        vector = [0] * len(labels)
        for lbl in combination:
            idx = label_to_index[lbl]
            vector[idx] = 1
        combination_to_vector[combination] = tuple(vector)

    vector_to_combination = {v: k for k, v in combination_to_vector.items()}

    # Step 7: Convert true and predicted labels to indices
    y_true_indices = []
    y_pred_indices = []

    for i in range(len(y_labels)):
        # True labels
        labels_in_sample = tuple(sorted(y_labels[i].split("+")))
        true_combination = labels_in_sample
        true_idx = combination_to_index[true_combination]

        # Predicted labels
        pred_vector = y_p_pred_test[i]
        labels_in_pred = []
        for idx, val in enumerate(pred_vector):
            if val == 1:
                labels_in_pred.append(labels[idx])
        pred_combination = tuple(sorted(labels_in_pred))

        # Handle unknown combinations (if any)
        if pred_combination in combination_to_index:
            pred_idx = combination_to_index[pred_combination]
        else:
            # Optionally, you can assign to an 'Other' category or skip
            # For now, let's skip samples with unknown combinations
            continue

        y_true_indices.append(true_idx)
        y_pred_indices.append(pred_idx)

    # Step 8: Compute the confusion matrix
    cm = confusion_matrix(
        y_true_indices, y_pred_indices, labels=range(len(all_combinations))
    )

    class_labels = [
        " + ".join(MappingNames()[co] for co in sorted(comb))
        for comb in sorted(all_combinations)
    ]

    # Step 9: Plot the confusion matrix
    plt.figure(figsize=(15, 15))
    ax = sns.heatmap(
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
    plt.savefig(f"{PLOT_DIR}/confusion_matrix_test.png")
    plt.show()

    # %% CLASSIFICATION REPORT

    # Compute metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_indices, y_pred_indices, labels=range(len(all_combinations)), zero_division=0
    )

    # Compute per-class accuracy: TP / support
    cm_diag = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy = np.divide(cm_diag, cm.sum(axis=1))
        accuracy[np.isnan(accuracy)] = 0.0  # Replace NaN due to 0 support with 0


    # Build dataframe
    df_report = pd.DataFrame(
        {
            "Label": class_labels,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Accuracy": accuracy,
            "Support": support,
        }
    )

    # Filter out zero-support classes
    df_report_filtered = df_report[df_report["Support"] > 0].reset_index(drop=True)

    # Optional: print or save
    print(df_report_filtered)

    # %% NEW RADAR PLOTS

    def plot_metric_radar(df, metrics, title_suffix="", multi_plot=False):
        labels = df["Label"].tolist()
        for i, l in enumerate(labels):
            if l == "LLDPE-H":
                labels[i] = l + "\n\n"
            elif l == "LLDPE-B + RaCo-PP":
                labels[i] = "     " + l
            elif l == "LLDPE-H + LLDPE-O":
                labels[i] = l + "     "

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        if not multi_plot:
            for metric in metrics:
                values = df[metric].tolist()
                values += values[:1]

                fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
                ax.plot(angles, values, linewidth=2, linestyle="solid", label=metric)
                ax.fill(angles, values, alpha=0.25)

                ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=18, rotation=90)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_yticklabels(["0.2", "0.4", "0.6", "", "1.0"], fontsize=16)
                ax.set_ylim(0, 1.0)

                ax.set_title(f"{metric} per Class {title_suffix}", fontsize=16, pad=20)
                plt.tight_layout()
                plt.savefig(f"{PLOT_DIR}/{metric.lower()}_radar.png", bbox_inches="tight")
                plt.show()

        else:
            fig, axs = plt.subplots(2, 2, figsize=(14, 14), subplot_kw=dict(polar=True))
            axs = axs.flatten()

            for i, metric in enumerate(metrics):
                values = df[metric].tolist()
                values += values[:1]

                ax = axs[i]
                ax.plot(angles, values, linewidth=2, linestyle="solid", label=metric)
                ax.fill(angles, values, alpha=0.25)

                ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12, rotation=90)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_yticklabels(["0.2", "0.4", "0.6", "", "1.0"], fontsize=10)
                ax.set_ylim(0, 1.0)

                ax.set_title(metric, fontsize=16, pad=20)

            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/combined_radar.png", bbox_inches="tight")
            plt.show()


    # Example usage:
    metrics = ["Precision", "Recall", "F1-score", "Accuracy"]

    # Single plots
    # plot_metric_radar(df_report_filtered, metrics, multi_plot=False)

    # Combined 2x2 plot
    plot_metric_radar(df_report_filtered, metrics, multi_plot=True)

    # %%
    ############################## WEIGHTS ANALYSIS ##############################

    y_w_true_test = np.zeros((len(y_labels), len(labels)))
    y_c_true_test = np.zeros((len(y_labels), len(labels)))
    for i, idx in enumerate(test_data.index):
        copo_tuple = test_data.loc[idx, "copo_tuple"]
        w_tuple = test_data.loc[idx, "w"]
        c_tuple = test_data.loc[idx, "c"]
        for copo, w in zip(copo_tuple, w_tuple):
            pos = label_to_index[copo]
            y_w_true_test[i, pos] = w
        for copo, c in zip(copo_tuple, c_tuple):
            pos = label_to_index[copo]
            y_c_true_test[i, pos] = c

    if "pp_raco" in opts.run_name or "ppraco" in opts.run_name:
        y_c_true_test_norm = scaler.transform(y_c_true_test)
        y_c_true_test_norm = np.where(y_w_true_test != 0, y_c_true_test_norm, -1)

    else:
        if "PE" in labels and "PP" in labels and "LDPE" in labels:
            y_c_true_test_norm = np.concatenate(
                [y_c_true_test[:, 0:3], scaler.transform(y_c_true_test[:, 3:])], axis=1
            )
        else:
            y_c_true_test_norm = np.concatenate(
                [y_c_true_test[:, 0:1], scaler.transform(y_c_true_test[:, 1:])], axis=1
            )
        y_c_true_test_norm = np.where(y_w_true_test != 0, y_c_true_test_norm, -1)


    n_labels = len(labels)

    # Determine the grid size for subplots (e.g., 2x2, 3x3)
    n_cols = math.ceil(np.sqrt(n_labels))
    n_rows = math.ceil(n_labels / n_cols)

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    plt.suptitle(f"Weight Analysis on Test Set", fontsize=18, y=1)

    axes = axes.flatten()

    # WEIGHTS ANALYSIS
    for idx, copo in enumerate(labels):
        pos = label_to_index[copo]
        y_w_pred_test_copo = y_w_pred_test[:, pos]
        y_w_true_test_copo = y_w_true_test[:, pos]

        idxs = np.where(y_w_true_test_copo > 0)[0]
        y_w_true_test_copo = y_w_true_test_copo[idxs]
        y_w_pred_test_copo = y_w_pred_test_copo[idxs]

        # Order y_w_true_copo
        order = np.argsort(y_w_true_test_copo)
        y_w_true_test_copo = y_w_true_test_copo[order]
        y_w_pred_test_copo = y_w_pred_test_copo[order]

        # Select the appropriate subplot
        ax = axes[idx]
        ax.plot(y_w_pred_test_copo, ".", alpha=0.75, label="Predicted", color="tab:orange")
        ax.plot(y_w_true_test_copo, label="True", color="tab:blue")
        ax.set_xlabel("Experiment", fontsize=14)
        ax.set_ylabel("Normalized Weight", fontsize=14)
        avg_error = np.mean(np.abs(y_w_true_test_copo - y_w_pred_test_copo))
        ax.set_title(f"{copo} - AvgError: {avg_error:.2f}", fontsize=16)
        ax.legend(fontsize=12)
        ax.set_ylim(-0.1, 1.1)
        ax.tick_params(axis="both", which="major", labelsize=12)

    # Hide any unused subplots if the grid is larger than the number of labels
    for idx in range(len(labels), len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the figure with all subplots
    plt.savefig(f"{PLOT_DIR}/weights_test.png")
    plt.show()


    for copo in labels:
        pos = label_to_index[copo]
        y_w_pred_test_copo = y_w_pred_test[:, pos]
        y_w_true_test_copo = y_w_true_test[:, pos]

        # Filter non-zero weights
        idxs = np.where(y_w_true_test_copo > 0)[0]
        y_w_true_test_copo = y_w_true_test_copo[idxs]
        y_w_pred_test_copo = y_w_pred_test_copo[idxs]

        # Sort by true weights
        order = np.argsort(y_w_true_test_copo)
        y_w_true_test_copo = y_w_true_test_copo[order]
        y_w_pred_test_copo = y_w_pred_test_copo[order]

        mask_mix = y_w_true_test_copo < 1.0
        mask_single = ~mask_mix

        # Create a new figure for this copolymer
        fig, ax = plt.subplots(figsize=(6, 4))

        # Plotting
        # Compute ±5% and ±10% bands around the true weight line
        y5_lower = y_w_true_test_copo * 0.95
        y5_upper = y_w_true_test_copo * 1.05
        y10_lower = y_w_true_test_copo * 0.90
        y10_upper = y_w_true_test_copo * 1.10

        x_vals = np.arange(len(y_w_true_test_copo))

        # Plot shaded error regions
        ax.fill_between(
            x_vals, y10_lower, y10_upper, color="blue", alpha=0.05, label="±10% Band"
        )
        ax.fill_between(
            x_vals, y5_lower, y5_upper, color="blue", alpha=0.075, label="±5% Band"
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
        ax.set_xlabel("Experiment", fontsize=14)
        ax.set_ylabel("Normalized Weight", fontsize=14)
        avg_error = np.mean(np.abs(y_w_true_test_copo - y_w_pred_test_copo))
        ax.set_title(f"{MappingNames()[copo]} - AvgMAE: {avg_error:.3f}", fontsize=14)
        ax.legend(fontsize=10)
        ax.set_ylim(-0.1, 1.1)
        ax.tick_params(axis="both", which="major", labelsize=12)

        # Save and close each figure
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/weights_{copo}.png", dpi=150)
        plt.show()

    # %%

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    plt.suptitle(f"Composition Analysis on Test Set", fontsize=18, y=1)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # COMPOSITION ANALYSIS
    for idx, copo in enumerate(labels):
        pos = label_to_index[copo]
        y_c_pred_test_copo = y_c_pred_test[:, pos]
        y_c_true_test_copo = y_c_true_test_norm[:, pos]

        idxs = np.where(y_c_true_test_copo >= 0)[0]
        y_c_true_test_copo = y_c_true_test_copo[idxs]
        y_c_pred_test_copo = y_c_pred_test_copo[idxs]

        # Order y_w_true_copo
        order = np.argsort(y_c_true_test_copo)
        y_c_true_test_copo = y_c_true_test_copo[order]
        y_c_pred_test_copo = y_c_pred_test_copo[order]

        # Select the appropriate subplot
        ax = axes[idx]
        ax.plot(y_c_pred_test_copo, ".", alpha=0.75, label="Predicted", color="tab:orange")
        ax.plot(y_c_true_test_copo, label="True", color="tab:blue")
        ax.set_xlabel("Experiment", fontsize=14)
        ax.set_ylabel("Normalized Composition", fontsize=14)
        avg_error = np.mean(np.abs(y_c_true_test_copo - y_c_pred_test_copo))
        ax.set_title(f"{copo} - AvgError: {avg_error:.2f}", fontsize=16)
        ax.legend(fontsize=12)
        ax.set_ylim(-0.1, 1.1)
        ax.tick_params(axis="both", which="major", labelsize=12)


    # Hide any unused subplots if the grid is larger than the number of labels
    for idx in range(len(labels), len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the figure with all subplots
    plt.savefig(f"{PLOT_DIR}/composition_test.png")
    plt.show()

    # %%
    # REGRESSION ANALYSIS on COMPOSITIONS

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    plt.suptitle(f"Composition Analysis on Test Set - Regression", fontsize=18, y=1)
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    plot_idx = 0
    for copo in labels:
        if copo in ["LDPE", "PE", "PP"]:
            continue
        pos = label_to_index[copo]
        y_c_pred_test_copo = y_c_pred_test[:, pos]
        y_c_true_test_copo = y_c_true_test_norm[:, pos]

        idxs = np.where(y_c_true_test_copo > 0)[0]
        y_c_true_test_copo = y_c_true_test_copo[idxs]
        y_c_pred_test_copo = y_c_pred_test_copo[idxs]

        # Select the appropriate subplot
        ax = axes[plot_idx]
        plot_idx += 1
        ax.scatter(y_c_true_test_copo, y_c_pred_test_copo, alpha=0.75, color="tab:orange")

        # Compute and plot the regression line
        # Define the 5% and 10% error bands around the ideal line
        y5_lower = y_c_true_test_copo * 0.95
        y5_upper = y_c_true_test_copo * 1.05
        y10_lower = y_c_true_test_copo * 0.90
        y10_upper = y_c_true_test_copo * 1.10
        y15_lower = y_c_true_test_copo * 0.85
        y15_upper = y_c_true_test_copo * 1.15
        y20_lower = y_c_true_test_copo * 0.80
        y20_upper = y_c_true_test_copo * 1.20

        # Sort for proper fill plotting
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

        slope, intercept = np.polyfit(y_c_true_test_copo, y_c_pred_test_copo, 1)
        regression_line = slope * y_c_true_test_copo + intercept
        ax.plot(
            y_c_true_test_copo, regression_line, color="tab:orange", label="Regression Line"
        )

        ax.plot(
            y_c_true_test_copo, y_c_true_test_copo, color="tab:blue", label="Ideal Line"
        )

        ax.set_xlabel("True", fontsize=14)
        ax.set_ylabel("Predicted", fontsize=14)
        avg_error = np.mean(np.abs(y_c_true_test_copo - y_c_pred_test_copo))
        ax.set_title(f"{copo} - AvgError: {avg_error:.2f}", fontsize=16)
        ax.legend(fontsize=12)
        # ax.set_ylim(-1.1, 1.1)
        ax.tick_params(axis="both", which="major", labelsize=12)


    # Hide any unused subplots if the grid is larger than the number of labels
    for idx in range(plot_idx, len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the figure with all subplots
    plt.savefig(f"{PLOT_DIR}/comps_regression_test.png")
    plt.show()

    # %% PARTITY SEABORN

    # Set global seaborn style
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)


    def parity_jointplot(y_true, mask_mix, y_pred, copo_name, save_path=None):
        # Create the joint plot (scatter + histograms)
        g = sns.JointGrid(x=y_true, y=y_pred, height=6)

        # Scatter plot in the center
        pure_mask = ~mask_mix  # true for pure points
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

        # Ideal line
        g.ax_joint.plot(y_true, y_true, color="tab:blue", label="Ideal Line")

        # Error bands: ±5%, ±10%, ±15%, ±20%
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
                label=f"±{pct}% Error Band" if pct <= 20 else None,  # avoid legend clutter
            )

        # Histograms with KDE
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

        # Styling
        g.ax_joint.set_xlim(-0.1, 1.1)
        g.ax_joint.set_ylim(-0.1, 1.1)
        g.ax_joint.set_xlabel("True Composition", fontsize=14)
        g.ax_joint.set_ylabel("Predicted Composition", fontsize=14)
        g.ax_joint.legend(loc="upper left", fontsize=10)
        g.figure.suptitle(f"{MappingNames()[copo_name]} - Parity Plot", fontsize=16, y=0.95)
        sns.despine()

        plt.tight_layout()

        # Save if desired
        g.figure.savefig(
            f"{PLOT_DIR}/{copo_name}_parity_plot.png", dpi=150, bbox_inches="tight"
        )
        plt.show()


    skip_labels = ["LDPE", "PE", "PP"]

    for copo in labels:
        if copo in skip_labels:
            continue

        pos = label_to_index[copo]
        y_true = y_c_true_test_norm[:, pos]
        y_pred = y_c_pred_test[:, pos]

        mask_mix = y_w_true_test[:, pos] < 1.0

        # Filter only non-zero samples
        idxs = np.where(y_true > 0)[0]
        y_true = y_true[idxs]
        y_pred = y_pred[idxs]

        mask_mix = mask_mix[idxs]

        parity_jointplot(y_true, mask_mix, y_pred, copo)

    # %% ERROR CORELATION
    error_matrix = {}

    for copo in labels:

        pos = label_to_index[copo]
        y_true = y_c_true_test_norm[:, pos]
        y_pred = y_c_pred_test[:, pos]
        error = y_pred - y_true

        error_matrix[copo] = error

    # Create a DataFrame and compute the correlation matrix
    error_df = pd.DataFrame(error_matrix)
    corr_matrix = error_df.corr()
    corr_matrix.index = [MappingNames()[copo] for copo in corr_matrix.index]
    corr_matrix.columns = [MappingNames()[copo] for copo in corr_matrix.columns]

    # Plot with seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Error Correlation Matrix (Prediction Error Correlation)", fontsize=16)
    plt.tight_layout()
    plt.show()

    # %% TEST SINGLE EXPERIMENTS

    if "PE" in labels and "PP" in labels and "LDPE" in labels:
        y_c_pred_test_denorm = np.concatenate(
            [y_c_pred_test[:, 0:3], scaler.inverse_transform(y_c_pred_test[:, 3:])],
            axis=1,
        )
    else:
        y_c_pred_test_denorm = np.concatenate(
            [y_c_pred_test[:, 0:1], scaler.inverse_transform(y_c_pred_test[:, 1:])],
            axis=1,
        )

    cmap = matplotlib.colormaps.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(labels))]
    label_to_color = dict(zip(labels, colors))

    for copo in labels:
        y_w_true_test_copo = y_w_true_test[:, label_to_index[copo]]
        y_c_true_test_copo = y_c_true_test[:, label_to_index[copo]]

        y_w_pred_test_copo = y_w_pred_test[:, label_to_index[copo]]
        y_c_pred_test_copo = y_c_pred_test_denorm[:, label_to_index[copo]]

        idxs_copo_single = np.where(y_w_true_test_copo == 1.0)[0]

        test_data_copo = test_data.iloc[idxs_copo_single].copy()

        fig, axes = plt.subplots(
            nrows=2, ncols=1, figsize=(max(0.3 * len(test_data_copo.index), 12), 12)
        )

        # --- First subplot ---
        axes[0].vlines(test_data_copo.index, 0, 1, color="black", alpha=0.5, linewidth=0.5)
        axes[0].plot(
            test_data_copo.index,
            y_w_true_test_copo[idxs_copo_single],
            label=f"True {MappingNames()[copo]} weight",
            color=label_to_color[copo],
            linewidth=2,
        )

        for k in range(y_w_pred_test.shape[1]):
            mask = y_w_pred_test[idxs_copo_single, k] > 1e-2
            if sum(mask) > 0:
                # Now plot only those points that pass the mask
                axes[0].scatter(
                    test_data_copo.index[mask],
                    y_w_pred_test[idxs_copo_single, k][mask],
                    label=f"Predicted {MappingNames()[labels[k]]} weight",
                    color=label_to_color[labels[k]],
                )

        # Remove x-tick labels from the first subplot
        if copo not in ["LDPE", "PE", "PP"]:
            axes[0].tick_params(
                axis="x", which="both", bottom=False, labelbottom=False, labelsize=12
            )
        else:
            axes[0].set_xticks(test_data_copo.index)
            axes[0].set_xticklabels(
                [f"Exp{i+1}" for i, _ in enumerate(test_data_copo.index)]
            )
            axes[0].tick_params(axis="x", rotation=90, labelsize=12)

        axes[0].tick_params(axis="y", which="major", labelsize=12)
        axes[0].set_title("Weight Plot", fontsize=16)
        axes[0].legend(fontsize=12)
        # axes[0].tick_params(axis='both', which='major', labelsize=12)
        axes[0].set_ylabel("Normalized Weight", fontsize=14)

        # --- Second subplot ---
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
            label=f"True {MappingNames()[copo]} composition",
            color=label_to_color[copo],
            linewidth=2,
        )
        axes[1].scatter(
            test_data_copo.index,
            y_c_pred_test_copo[idxs_copo_single],
            color=label_to_color[copo],
            label=f"Predicted {MappingNames()[copo]} composition",
        )

        # Allow for x-ticks with rotation on the second subplot
        axes[1].set_xticks(test_data_copo.index)
        axes[1].set_xticklabels([f"Exp{i+1}" for i, _ in enumerate(test_data_copo.index)])
        axes[1].tick_params(axis="x", rotation=90, labelsize=12)
        axes[1].tick_params(axis="y", which="major", labelsize=12)
        axes[1].set_title("Composition Plot", fontsize=16)
        axes[1].set_ylabel("Composition", fontsize=14)

        axes[1].legend(fontsize=12)
        # axes[1].tick_params(axis='both', which='major', labelsize=12)

        # plt.tight_layout()
        plt.tight_layout()
        os.makedirs(f"{PLOT_DIR}/SingleCopos", exist_ok=True)
        plt.savefig(f"{PLOT_DIR}/SingleCopos/single_{copo}_test.png", dpi=150)
        plt.show()

    # %% TEST PP-EPR MIXTURES
    possible_mixes = [m for m in test_data["copo_tuple"].unique() if len(m) > 1]

    for mix in possible_mixes:  # [:1]:

        check_mix = np.ones(y_w_true_test.shape[0], dtype=bool)
        # For each copolymer in the tuple, refine the mask
        for co in mix:
            check_mix &= y_w_true_test[:, label_to_index[co]] > 0.0

        mix = "+".join(mix)
        test_data_mix = test_data[test_data["copolymer"] == mix]
        idxs_mix = np.where(check_mix)[0]

        y_w_true_test_mix = y_w_true_test[idxs_mix]
        y_c_true_test_mix = y_c_true_test[idxs_mix]

        y_w_pred_test_mix = y_w_pred_test[idxs_mix]
        y_c_pred_test_mix = y_c_pred_test_denorm[idxs_mix]

        fig, axes = plt.subplots(
            nrows=2, ncols=1, figsize=(max(0.3 * len(test_data_mix.index), 15), 12)
        )

        # --- First subplot ---
        axes[0].vlines(test_data_mix.index, 0, 1, color="black", alpha=0.5, linewidth=0.5)
        for k in range(y_w_pred_test_mix.shape[1]):
            mask = y_w_pred_test_mix[:, k] > 1e-3
            if sum(mask) > 0:
                axes[0].plot(
                    test_data_mix.index[mask],
                    y_w_true_test_mix[:, k][mask],
                    label=f"True {MappingNames()[labels[k]]} weight",
                    color=label_to_color[labels[k]],
                    linewidth=1.5,
                )
                axes[0].scatter(
                    test_data_mix.index[mask],
                    y_w_pred_test_mix[:, k][mask],
                    label=f"Predicted {MappingNames()[labels[k]]} weight",
                    color=label_to_color[labels[k]],
                )
        axes[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        axes[0].tick_params(axis="y", labelsize=12)
        axes[0].set_title("Weight Plot", fontsize=16)
        axes[0].set_ylabel("Normalized Weight", fontsize=14)
        axes[0].legend(fontsize=12)

        axes[1].vlines(
            test_data_mix.index,
            min(
                min(y_c_true_test_mix[y_w_pred_test_mix > 1e-3].flatten()),
                min(y_c_pred_test_mix[y_w_pred_test_mix > 1e-3].flatten()),
            ),
            max(
                max(y_c_true_test_mix[y_w_pred_test_mix > 1e-3].flatten()),
                max(y_c_pred_test_mix[y_w_pred_test_mix > 1e-3].flatten()),
            ),
            color="black",
            alpha=0.5,
            linewidth=0.5,
        )
        for k in range(y_w_pred_test_mix.shape[1]):
            mask = y_w_pred_test_mix[:, k] > 1e-3
            if sum(mask) > 0:
                axes[1].plot(
                    test_data_mix.index[mask],
                    y_c_true_test_mix[:, k][mask],
                    label=f"True {MappingNames()[labels[k]]} composition",
                    color=label_to_color[labels[k]],
                    linewidth=1.5,
                )
                axes[1].scatter(
                    test_data_mix.index[mask],
                    y_c_pred_test_mix[:, k][mask],
                    color=label_to_color[labels[k]],
                    label=f"True {MappingNames()[labels[k]]} composition",
                )
        mix_copolymers = mix.split("+")
        if any(co in ["EH", "EO", "EB", "RACO"] for co in mix_copolymers):
            axes[1].set_ylim(-0.5, 10.5)
        # Allow for x-ticks with rotation on the second subplot
        axes[1].tick_params(axis="x", rotation=90, labelsize=12)
        axes[1].set_xticks(test_data_mix.index)
        axes[1].set_xticklabels([f"Exp{i+1}" for i, _ in enumerate(test_data_mix.index)])
        axes[1].tick_params(axis="y", labelsize=12)
        axes[1].set_title("Composition Plot", fontsize=16)
        axes[1].set_ylabel("Composition", fontsize=14)
        axes[1].legend(fontsize=12)

        plt.tight_layout()
        os.makedirs(f"{PLOT_DIR}/Mixes", exist_ok=True)
        plt.savefig(f"{PLOT_DIR}/Mixes/{mix}_test.png", dpi=150)
        plt.show()

    # %% ANALYSIS ON ATTENTION WEIGHTS

    for i in np.arange(len(test_data))[425:426]:
        X_test_sample = X_test[i : i + 1]

        

        for l in loaded_model.layers[:8]:
            if l == loaded_model.layers[0]:
                x = loaded_model.layers[0](X_test_sample[..., np.newaxis], training=False)
            else:
                x = l(x, training=False)

        attention_weights = x[1].numpy()

        attention_weights_sample = attention_weights[
            0
        ]  # Shape: (num_heads, seq_length, seq_length)

        # Compute mean attention over all heads and time steps
        mean_attention = np.mean(attention_weights_sample, axis=1)
        mean_attention = np.mean(mean_attention, axis=0)[
            np.newaxis, ...
        ]  # Shape: (1, seq_length)

        # Normalize each head's attention weights
        for j in range(mean_attention.shape[0]):
            mean_attention[j] = (mean_attention[j] - np.min(mean_attention[j])) / (
                np.max(mean_attention[j]) - np.min(mean_attention[j])
            )

        mean_attention_interp = []
        for j in range(mean_attention.shape[0]):
            mean_attention_interp.append(
                interp1d(np.linspace(0, 1, len(mean_attention[j])), mean_attention[j])
            )

        # Prepare data
        spectrum = np.clip(X_test_sample[0], a_min=1e-6, a_max=None)
        x = np.arange(len(spectrum))

        # Set up figure and colormap
        fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
        axes = axes.flatten()
        colors = plt.get_cmap("tab10").colors  # 10 distinct colors

        for h in range(4):
            ax = axes[h]

            # Diagonal attention and interpolation
            attn_diag = np.diag(attention_weights_sample[h])
            attn_interp = interp1d(np.linspace(0, 1, len(attn_diag)), attn_diag)
            attn_values = attn_interp(np.linspace(0, 1, len(spectrum)))

            # Normalize attention weights for alpha
            alpha_values = (attn_values - attn_values.min()) / (
                attn_values.max() - attn_values.min() + 1e-9
            )

            # Plot spectrum
            ax.plot(x, spectrum, color="tab:blue", linewidth=1.2)
            ax.set_yscale("log")
            ax.set_title(f"Head {h}", fontsize=14)
            ax.tick_params(axis="both", labelsize=10)

            # Fill between with per-point alpha
            for j in range(len(x) - 1):
                alpha_val = (alpha_values[j] + alpha_values[j + 1]) / 2
                ax.fill_between(
                    x[j : j + 2],
                    0,
                    spectrum[j : j + 2],
                    color=colors[h + 1],
                    alpha=(
                        alpha_val / 2 if alpha_val / 2 > 0.125 else 0.0
                    ), 
                    linewidth=0,
                )

            ax2 = ax.twinx()
            ax2.plot(x, alpha_values, color=colors[h + 1], linewidth=1.2)
            if h in [1, 3]:
                ax2.set_ylabel("Normalized Attention Weight", fontsize=12)
                ax2.tick_params(axis="y", labelsize=10)
            if h in [0, 2]:
                ax2.set_yticks([])  # Hide y-ticks for the first and third plots
                ax2.tick_params(axis="y", which="both", right=False, labelright=False)
            ax2.set_ylim(-0.1, 1.1)

        # Shared labels
        axes[0].set_ylabel("Log(Intensity)", fontsize=12)
        axes[2].set_ylabel("Log(Intensity)", fontsize=12)
        axes[2].set_xlabel("Spectral Index", fontsize=12)
        axes[3].set_xlabel("Spectral Index", fontsize=12)

        fig.suptitle(
            f"Spectrum {test_data.iloc[i].name} with Self-Attention Behaviour", fontsize=16
        )
        plt.tight_layout()

        # Save or show
        os.makedirs(f"{PLOT_DIR}/Attention", exist_ok=True)
        plt.savefig(
            f"{PLOT_DIR}/Attention/{test_data.iloc[i].name}_alpha_fill_heads.png", dpi=150
        )
        plt.show()


    # %% UNKNOWN TEST

    if test_data_unknown is not None:

        X_unknown = np.array(test_data_unknown.values[:, 4:]).astype(np.float32)

        if "norm" in opts.run_name:
            with open(f"val_sets/{opts.run_name}/scaler_spectra.pkl", "rb") as f:
                scaler_spectra = pd.read_pickle(f)

            X_unknown = scaler_spectra.transform(X_unknown.flatten().reshape(-1, 1)).reshape(
                X_unknown.shape
            )

        y_pred_unknown = loaded_model.predict(X_unknown[..., np.newaxis])

        y_w_pred_unknown = y_pred_unknown["weight_output"]
        y_c_pred_unknown = y_pred_unknown["composition_output"]
        if "PE" in labels and "PP" in labels and "LDPE" in labels:
            y_c_pred_unknown_denorm = np.concatenate(
                [y_c_pred_unknown[:, :3], scaler.inverse_transform(y_c_pred_unknown[:, 3:])],
                axis=1,
            )
        else:
            y_c_pred_unknown_denorm = np.concatenate(
                [y_c_pred_unknown[:, :1], scaler.inverse_transform(y_c_pred_unknown[:, 1:])],
                axis=1,
            )

        for i, idx in enumerate(test_data_unknown.index):
            print(
                idx,
                test_data_unknown.loc[idx, "copo_tuple"],
                test_data_unknown.loc[idx, "w"],
                test_data_unknown.loc[idx, "c"],
            )

            for j in range(len(labels)):
                if y_w_pred_unknown[i][j] > 1e-2:
                    print(
                        f"{labels[j]} -> w: {y_w_pred_unknown[i][j]:.2f} c: {y_c_pred_unknown_denorm[i][j]:.2f}"
                    )

            print("---------------------------------------------")
    else:
        print("No unknown test data available.")

#%%