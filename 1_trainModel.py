#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a DeepNMR model on synthetic 13C-NMR spectra.

This script trains the base multi-task model for:
  - polymer weight-fraction estimation;
  - comonomer composition estimation.

The experimental test set is used only for diagnostic monitoring through
``utils.TestSetEvaluationCallback``. Model selection is performed using the
synthetic validation loss.
"""

from __future__ import annotations

import os
import json
import pickle
import shutil
import argparse

import numpy as np
import pandas as pd
from fastcore.all import dict2obj, obj2dict
from sklearn.preprocessing import MinMaxScaler


# %% FIXED CONFIGURATION

COPOLYMER_LIST = ["LDPE", "PE", "PP", "EH", "EO", "EB", "RACO", "EPR"]

VAL_FOLDER = "val_sets"
MODEL_FOLDER = "models"


# %% ARGUMENTS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a DeepNMR model on synthetic NMR spectra.",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Per-GPU batch size to use for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3,4,5",
        help='Comma-separated list of GPUs to use, e.g. "0,1".',
    )
    parser.add_argument(
        "--loss_weights",
        type=str,
        default="kl_mse_loss",
        help="Loss function to use for the weight output.",
    )
    parser.add_argument(
        "--loss_composition",
        type=str,
        default="neg2_mse_hybrid",
        help="Loss function to use for the composition output.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="DATASET/synthetic_dataset.pkl",
        help="Synthetic dataset used to train the model.",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="DATASET/test_data.pkl",
        help="Experimental test dataset used for diagnostic monitoring.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help=(
            "Optional explicit run name. If omitted, the run name is generated "
            "with the same convention as the original script."
        ),
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
    """Train the DeepNMR base model."""
    args = parse_args()

    # Configure GPU visibility before importing TensorFlow and utils.
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
            n_tot_samples="all",
            epochs=args.epochs,
            batch_size=args.batch_size * len(CVD.split(",")),
            validation_split=0.25,
            learning_rate=5e-4,
            reg_l2=1e-9,
            dropout_rate=0.1,
            normalize_spectra=True,
            task=["weight", "composition"],
            loss_classification=None,
            metrics_classification=None,
            loss_weights=[args.loss_weights],
            metrics_weights={
                "MeanAbsoluteError": {"name": "mae"},
                "RootMeanSquaredError": {"name": "rmse"},
                "KLDivergence": {"name": "kld"},
            },
            loss_composition=[args.loss_composition],
            metrics_composition={
                "MeanAbsoluteError": {"name": "mae"},
                "RootMeanSquaredError": {"name": "rmse"},
            },
        )
    )

    if args.run_name is not None and args.run_name.strip():
        opts.run_name = args.run_name.strip()
    else:
        opts.run_name = (
            f"model_weights_{opts.loss_weights[0]}"
            f"_composition_{opts.loss_composition[0]}"
        )

        if opts.normalize_spectra:
            opts.run_name += "_norm"

        opts.dataset = args.dataset.split("/")[-1].split(".")[0]
        opts.run_name += f"_{args.dataset.split('_')[-1].split('.')[0]}"

    # %% DATA IMPORT AND MANIPULATION

    print("Loading data...")
    data = pd.read_pickle(args.dataset)
    print("Data loaded!")

    X, y_w, y_c = data[0], data[3], data[4]

    y_p = np.where(y_w != 0, 1, 0)
    np.unique(np.sum(y_p, axis=1), return_counts=True)

    if opts.n_tot_samples != "all":
        assert (
            opts.n_tot_samples <= X.shape[0]
        ), "The number of samples is greater than the dataset size"

        rand_idxs = np.random.choice(
            np.arange(X.shape[0]),
            opts.n_tot_samples,
            replace=False,
        )

        X = X[rand_idxs]
        y_w = y_w[rand_idxs]
        y_c = y_c[rand_idxs]

    X = X[..., np.newaxis]
    y_p = np.where(y_w != 0, 1, 0)

    if opts.normalize_spectra:
        scaler_spectra = MinMaxScaler()
        X = scaler_spectra.fit_transform(X.flatten().reshape(-1, 1)).reshape(X.shape)

    scaler = MinMaxScaler()

    y_c_norm = np.concatenate(
        [y_c[:, 0:3], scaler.fit_transform(y_c[:, 3:])],
        axis=1,
    )

    y_c_norm[y_p == 0] = -1

    val_dir = f"{VAL_FOLDER}/{opts.run_name}"
    model_dir = f"{MODEL_FOLDER}/{opts.run_name}"

    os.makedirs(val_dir, exist_ok=True)

    with open(f"{val_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    if opts.normalize_spectra:
        with open(f"{val_dir}/scaler_spectra.pkl", "wb") as f:
            pickle.dump(scaler_spectra, f)

    # %% IMPORT TEST DATA

    test_data = pd.read_pickle(args.test_dataset)

    test_data = test_data[
        test_data["w"].apply(lambda x: not any(np.isnan(k) for k in x))
    ]

    X_test = test_data.values[:, 4:]

    if opts.normalize_spectra:
        X_test = scaler_spectra.transform(X_test.flatten().reshape(-1, 1)).reshape(
            X_test.shape
        )

    X_test = X_test[..., np.newaxis].astype(np.float32)

    y_w_test = np.vstack(
        test_data.apply(
            lambda row: np.array(
                [
                    (
                        row["w"][row["copo_tuple"].index(copo)]
                        if copo in row["copo_tuple"]
                        else 0
                    )
                    for copo in COPOLYMER_LIST
                ]
            ),
            axis=1,
        ).to_numpy()
    )

    y_c_test = np.vstack(
        test_data.apply(
            lambda row: np.array(
                [
                    (
                        row["c"][row["copo_tuple"].index(copo)]
                        if copo in row["copo_tuple"]
                        else 0
                    )
                    for copo in COPOLYMER_LIST
                ]
            ),
            axis=1,
        ).to_numpy()
    )

    y_c_norm_test = np.concatenate(
        [y_c_test[:, 0:3], scaler.transform(y_c_test[:, 3:])],
        axis=1,
    )

    y_c_norm_test[y_w_test == 0] = -1

    # %% MODEL COMPILING

    gpus = tf.config.list_physical_devices("GPU")

    if len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.MirroredStrategy()

    print(f"Number of GPUs used: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        loss = {}
        metrics = {}

        if "weight" in opts.task:
            loss["weight_output"] = [
                getattr(utils.losses, m, None) or m for m in opts.loss_weights
            ]
            metrics["weight_output"] = [
                getattr(tf.metrics, m)(**v) for m, v in opts.metrics_weights.items()
            ]

            if "composition" in opts.task:
                loss["composition_output"] = [
                    getattr(utils.losses, m, None) or m
                    for m in opts.loss_composition
                ]
                metrics["composition_output"] = [
                    getattr(tf.metrics, m)(**v)
                    for m, v in opts.metrics_composition.items()
                ]
                model = utils.model.CustomModel(
                    reg_l2=opts.reg_l2,
                    dropout_rate=opts.dropout_rate,
                    n_outputs=data[3].shape[1],
                )

            else:
                model = utils.model.CustomModelWeights(
                    reg_l2=opts.reg_l2,
                    dropout_rate=opts.dropout_rate,
                )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=opts.learning_rate,
                clipnorm=1.0,
            ),
            loss=loss,
            metrics=metrics,
        )

        model(X[:1])
        model.summary()

    # %% MODEL TRAINING DATASETS

    with tf.device("/cpu:0"):
        train_size = int(len(X) * (1 - opts.validation_split))
        val_size = len(X) - train_size

        indices = np.arange(len(X))
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        X_train, X_val = X[train_indices], X[val_indices]
        y_w_train, y_w_val = y_w[train_indices], y_w[val_indices]
        y_c_train, y_c_val = y_c[train_indices], y_c[val_indices]
        y_c_norm_train, y_c_norm_val = (
            y_c_norm[train_indices],
            y_c_norm[val_indices],
        )

        os.makedirs(val_dir, exist_ok=True)
        np.save(f"{val_dir}/X_val", X_val)
        np.save(f"{val_dir}/y_w_val", y_w_val)
        np.save(f"{val_dir}/y_c_val", y_c_val)
        np.save(f"{val_dir}/y_c_norm_val", y_c_norm_val)

        if "weight" in opts.task and "composition" not in opts.task:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (X_train, {"weight_output": y_w_train})
            )
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (X_val, {"weight_output": y_w_val})
            )
            test_dataset = tf.data.Dataset.from_tensor_slices(
                (X_test, {"weight_output": y_w_test})
            )

        elif "composition" in opts.task and "weight" in opts.task:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    X_train,
                    {
                        "weight_output": y_w_train,
                        "composition_output": y_c_norm_train,
                    },
                )
            )
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    X_val,
                    {
                        "weight_output": y_w_val,
                        "composition_output": y_c_norm_val,
                    },
                )
            )
            test_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    X_test,
                    {
                        "weight_output": y_w_test,
                        "composition_output": y_c_norm_test,
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
            test_dataset.shuffle(buffer_size=len(X_test))
            .batch(len(X_test), drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    steps_per_epoch = np.floor(train_size / opts.batch_size).astype(int)
    validation_steps = np.floor(val_size / opts.batch_size).astype(int)
    test_steps = 1

    # %% MODEL OUTPUT DIRECTORY

    os.makedirs(model_dir, exist_ok=True)
    shutil.copy(utils.model.__file__, f"{model_dir}/model.py")

    with open(f"{model_dir}/opts.json", "w") as f:
        json.dump(obj2dict(opts), f)

    # %% MODEL TRAINING

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=opts.epochs,
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
                f"{model_dir}/model.weights.h5",
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            utils.CosineDecayAfterPlateau(
                fixed_lr=opts.learning_rate,
                final_lr=1e-6,
                plateau_epochs=15,
                decay_epochs=35,
            ),
            utils.TestSetEvaluationCallback(
                test_dataset,
                test_steps,
                COPOLYMER_LIST,
                strategy,
            ),
            utils.SaveHistoryCallback(
                f"{model_dir}/training_history.json",
                save_interval=10,
            ),
        ],
        verbose=1,
    )

    print("DONE!")


if __name__ == "__main__":
    main()