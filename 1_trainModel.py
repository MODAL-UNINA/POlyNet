#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 06 16:22:08 2024

@author: MODAL
"""

# %% IMPORT SECTION
import os
import json
import utils
import pickle
import shutil
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from fastcore.all import dict2obj, obj2dict
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NMR training script with overrides")

    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size to use for training."
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3,4,5",
        help='Comma‐separated list of GPUs to use, e.g. "0,1"',
    )
    parser.add_argument(
        "--loss_weights",
        type=str,
        default="kl_mse_loss",
        help="Which loss functions to use for weight output.",
    )
    parser.add_argument(
        "--loss_composition",
        type=str,
        default="neg_mse",
        help="Which loss functions to use for composition output.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="DATASET/synthetic_dataset.pkl",
        help="Which dataset should be used to train the model.",
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
            n_tot_samples="all",
            epochs=10000,
            batch_size=args.batch_size * len(CVD.split(",")),
            validation_split=0.25,
            learning_rate=7e-4,
            reg_l2=3e-12,
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

    opts.run_name = f"model_weights_{opts.loss_weights[0]}_composition_{opts.loss_composition[0]}_model"
    if opts.normalize_spectra:
        opts.run_name += "_norm"

    # opts.dataset = args.dataset.split("/")[-1].split(".")[0]
    # opts.run_name += f"_{args.dataset.split("_")[-1].split('.')[0]}"

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
            np.arange(X.shape[0]), opts.n_tot_samples, replace=False
        )
        X = X[rand_idxs]
        y_w = y_w[rand_idxs]
        y_c = y_c[rand_idxs]

    X = X[..., np.newaxis]
    y_p = np.where(y_w != 0, 1, 0)

    if opts.normalize_spectra:
        scaler_spectra = MinMaxScaler()
        # X = scaler_spectra.fit_transform(X.T).T
        X = scaler_spectra.fit_transform(X.flatten().reshape(-1, 1)).reshape(X.shape)

    scaler = MinMaxScaler()

    if "ppraco" in opts.run_name or "pp_raco" in opts.run_name:
        print(f"Aggregated PP_RACO")
        y_c_norm = scaler.fit_transform(y_c)
    else:
        print(f"Separate PP and RACO")
        y_c_norm = np.concatenate(
            [y_c[:, 0:3], scaler.fit_transform(y_c[:, 3:])], axis=1
        )

    y_c_norm[y_p == 0] = -1

    os.makedirs(f"val_sets/{opts.run_name}", exist_ok=True)

    with open(f"val_sets/{opts.run_name}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    if opts.normalize_spectra:
        with open(f"val_sets/{opts.run_name}/scaler_spectra.pkl", "wb") as f:
            pickle.dump(scaler_spectra, f)

    # %% IMPORT TEST DATA
    test_data = pd.read_pickle(args.test_dataset)

    # Exclude samples with unknown copolymers or compositions
    test_data = test_data[
        test_data["w"].apply(lambda x: not any(np.isnan(k) for k in x))
    ]

    copolymer_list = ["LDPE", "PE", "PP", "EH", "EO", "EB", "RACO", "EPR"]

    if "ppraco" in opts.run_name or "pp_raco" in opts.run_name:
        copolymer_list.append("PP_RACO")
        copolymer_list.remove("PP")
        copolymer_list.remove("RACO")

        for idx in test_data.index:
            if (
                "PP" in test_data.loc[idx, "copo_tuple"]
                or "RACO" in test_data.loc[idx, "copo_tuple"]
            ):
                test_data.at[idx, "c"] = tuple(
                    [
                        0.0 if k == "PP" else test_data.loc[idx, "c"][i]
                        for i, k in enumerate(test_data.loc[idx, "copo_tuple"])
                    ]
                )

                test_data.at[idx, "copo_tuple"] = tuple(
                    [
                        "PP_RACO" if k in ["PP", "RACO"] else k
                        for k in test_data.loc[idx, "copo_tuple"]
                    ]
                )

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
                    for copo in copolymer_list
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
                    for copo in copolymer_list
                ]
            ),
            axis=1,
        ).to_numpy()
    )

    if "ppraco" in opts.run_name or "pp_raco" in opts.run_name:
        print(f"Aggregated PP_RACO")
        y_c_norm_test = scaler.transform(y_c_test)

    else:
        print(f"Separate PP and RACO")
        y_c_norm_test = np.concatenate(
            [y_c_test[:, 0:3], scaler.transform(y_c_test[:, 3:])], axis=1
        )

    y_c_norm_test[y_w_test == 0] = -1

    # %% MODEL COMPILING

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

        if "weight" in opts.task:
            loss["weight_output"] = [
                getattr(utils.losses, m, None) or m for m in opts.loss_weights
            ]
            metrics["weight_output"] = [
                getattr(tf.metrics, m)(**v) for m, v in opts.metrics_weights.items()
            ]

            if "composition" in opts.task:
                loss["composition_output"] = [
                    getattr(utils.losses, m, None) or m for m in opts.loss_composition
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
                    reg_l2=opts.reg_l2, dropout_rate=opts.dropout_rate
                )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=opts.learning_rate,
            ),
            loss=loss,
            metrics=metrics,
        )

        model(X[:1])
        model.summary()

    # %% MODEL TRAINING

    with tf.device("/cpu:0"):
        # Convert numpy arrays to tf.data.Dataset
        train_size = int(len(X) * (1 - opts.validation_split))
        val_size = len(X) - train_size

        # Shuffle indices
        indices = np.arange(len(X))
        np.random.shuffle(indices)  # Inplace operation

        # Split indices into training and validation
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Split the data
        X_train, X_val = X[train_indices], X[val_indices]
        # y_p_train, y_p_val = y_p[train_indices], y_p[val_indices]
        y_w_train, y_w_val = y_w[train_indices], y_w[val_indices]
        y_c_train, y_c_val = y_c[train_indices], y_c[val_indices]
        y_c_norm_train, y_c_norm_val = y_c_norm[train_indices], y_c_norm[val_indices]

        # Save the val data
        os.makedirs(f"val_sets/{opts.run_name}", exist_ok=True)
        np.save(f"val_sets/{opts.run_name}/X_val", X_val)
        # np.save(f"val_sets/{opts.run_name}/y_p_val", y_p_val)
        np.save(f"val_sets/{opts.run_name}/y_w_val", y_w_val)
        np.save(f"val_sets/{opts.run_name}/y_c_val", y_c_val)
        np.save(f"val_sets/{opts.run_name}/y_c_norm_val", y_c_norm_val)

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
                    {"weight_output": y_w_train, "composition_output": y_c_norm_train},
                )
            )
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (X_val, {"weight_output": y_w_val, "composition_output": y_c_norm_val})
            )
            test_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    X_test,
                    {"weight_output": y_w_test, "composition_output": y_c_norm_test},
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
            test_dataset.shuffle(buffer_size=len(X_test)).batch(
                len(X_test), drop_remainder=True
            )
            # .repeat(opts.epochs)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    # Set steps_per_epoch and validation_steps based on the dataset sizes
    steps_per_epoch = np.floor(train_size / opts.batch_size).astype(int)
    validation_steps = np.floor(val_size / opts.batch_size).astype(int)
    test_steps = np.floor(len(X_test) / len(X_test)).astype(
        int
    )  # Ensure correct batch count

    os.makedirs(f"models/{opts.run_name}", exist_ok=True)
    shutil.copy(utils.model.__file__, f"models/{opts.run_name}/model.py")

    # %% MODEL TRAINING

    # save opts file
    with open(f"models/{opts.run_name}/opts.json", "w") as f:
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
                monitor="val_loss", patience=100, restore_best_weights=False
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f"models/{opts.run_name}/model.weights.h5",
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            utils.CosineDecayAfterPlateau(
                fixed_lr=opts.learning_rate,
                final_lr=1e-6,
                plateau_epochs=30,
                decay_epochs=150,  # opts.epochs * (len(CVD.split(","))) * 5,
            ),
            utils.TestSetEvaluationCallback(
                test_dataset, test_steps, copolymer_list, strategy
            ),
            utils.SaveHistoryCallback(
                f"models/{opts.run_name}/training_history.json", save_interval=10
            ),
        ],
        verbose=1,
    )

    print("DONE!")

# %%
