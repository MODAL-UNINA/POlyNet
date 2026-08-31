#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Neural baselines for the POlyNet architecture comparison.

The baselines preserve the analytical task used by POlyNet (joint prediction of
polymer weight fractions and copolymer compositions) but deliberately exclude
POlyNet-specific components such as the multi-scale stem, self-attention,
attention pooling, cross-task coupling, and consistency regularization.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="POlyNetBaselines")
class BaseMultiTaskBaseline(tf.keras.Model):
    """Common simple multi-task decoder shared by all neural baselines."""

    def __init__(
        self,
        n_outputs: int,
        shared_units: int = 256,
        head_units: int = 64,
        dropout_rate: float = 0.1,
        reg_l2: float = 1e-9,
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if n_outputs is None or n_outputs <= 0:
            raise ValueError("n_outputs must be a positive integer")

        self.n_outputs = int(n_outputs)
        self.shared_units = int(shared_units)
        self.head_units = int(head_units)
        self.dropout_rate = float(dropout_rate)
        self.reg_l2 = float(reg_l2)
        self.activation = activation

        regularizer = tf.keras.regularizers.L2(reg_l2) if reg_l2 > 0 else None
        self.regularizer = regularizer

        self.shared_dense = tf.keras.layers.Dense(
            shared_units,
            activation=activation,
            kernel_regularizer=regularizer,
            name="shared_dense",
        )
        self.shared_dropout = tf.keras.layers.Dropout(
            dropout_rate,
            name="shared_dropout",
        )

        self.weight_hidden = tf.keras.layers.Dense(
            head_units,
            activation=activation,
            kernel_regularizer=regularizer,
            name="weight_hidden",
        )
        self.weight_output = tf.keras.layers.Dense(
            n_outputs,
            activation="softmax",
            name="weight_output",
        )

        self.composition_hidden = tf.keras.layers.Dense(
            head_units,
            activation=activation,
            kernel_regularizer=regularizer,
            name="composition_hidden",
        )
        self.composition_output = tf.keras.layers.Dense(
            n_outputs,
            activation="elu",
            name="composition_output",
        )

    def encode(self, inputs, training=False):
        """Return a rank-2 feature tensor of shape ``(batch, features)``."""
        raise NotImplementedError

    def call(self, inputs, training=False):
        x = self.encode(inputs, training=training)
        x = self.shared_dense(x)
        x = self.shared_dropout(x, training=training)

        weight_output = self.weight_output(self.weight_hidden(x))
        composition_output = self.composition_output(self.composition_hidden(x))

        return {
            "weight_output": weight_output,
            "composition_output": composition_output,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_outputs": self.n_outputs,
                "shared_units": self.shared_units,
                "head_units": self.head_units,
                "dropout_rate": self.dropout_rate,
                "reg_l2": self.reg_l2,
                "activation": self.activation,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="POlyNetBaselines")
class MLPBaseline(BaseMultiTaskBaseline):
    """Simple fully connected baseline with no explicit spectral locality."""

    def __init__(self, n_outputs: int, encoder_units: int = 64, **kwargs):
        super().__init__(n_outputs=n_outputs, **kwargs)
        self.encoder_units = int(encoder_units)

        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.encoder_dense = tf.keras.layers.Dense(
            encoder_units,
            activation=self.activation,
            kernel_regularizer=self.regularizer,
            name="encoder_dense",
        )
        self.encoder_dropout = tf.keras.layers.Dropout(
            self.dropout_rate,
            name="encoder_dropout",
        )

    def encode(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.encoder_dense(x)
        return self.encoder_dropout(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"encoder_units": self.encoder_units})
        return config


@tf.keras.utils.register_keras_serializable(package="POlyNetBaselines")
class CNNBaseline(BaseMultiTaskBaseline):
    """Conventional three-stage 1D-CNN spectral baseline."""

    def __init__(
        self,
        n_outputs: int,
        filters: Sequence[int] = (32, 64, 96),
        kernel_sizes: Sequence[int] = (9, 7, 5),
        **kwargs,
    ):
        super().__init__(n_outputs=n_outputs, **kwargs)

        if len(filters) != len(kernel_sizes):
            raise ValueError("filters and kernel_sizes must have the same length")

        self.filters = tuple(int(v) for v in filters)
        self.kernel_sizes = tuple(int(v) for v in kernel_sizes)

        self.conv_blocks = []
        for i, (n_filters, kernel_size) in enumerate(
            zip(self.filters, self.kernel_sizes)
        ):
            self.conv_blocks.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Conv1D(
                            n_filters,
                            kernel_size=kernel_size,
                            padding="same",
                            kernel_regularizer=self.regularizer,
                        ),
                        tf.keras.layers.Activation(self.activation),
                        tf.keras.layers.MaxPooling1D(
                            pool_size=2,
                            strides=2,
                            padding="same",
                        ),
                    ],
                    name=f"conv_block_{i}",
                )
            )

        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D(
            name="global_avg_pool"
        )
        self.global_max_pool = tf.keras.layers.GlobalMaxPooling1D(
            name="global_max_pool"
        )
        self.pool_concat = tf.keras.layers.Concatenate(name="pool_concat")

    def encode(self, inputs, training=False):
        x = inputs
        for block in self.conv_blocks:
            x = block(x, training=training)

        return self.pool_concat(
            [self.global_avg_pool(x), self.global_max_pool(x)]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": list(self.filters),
                "kernel_sizes": list(self.kernel_sizes),
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="POlyNetBaselines")
class ResidualBlock1D(tf.keras.layers.Layer):
    """Standard residual Conv1D block with factor-2 downsampling."""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        activation: str = "gelu",
        reg_l2: float = 1e-9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.activation = activation
        self.reg_l2 = float(reg_l2)

        regularizer = tf.keras.regularizers.L2(reg_l2) if reg_l2 > 0 else None

        self.conv1 = tf.keras.layers.Conv1D(
            filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=regularizer,
            name="conv1",
        )
        self.act1 = tf.keras.layers.Activation(activation, name="act1")
        self.conv2 = tf.keras.layers.Conv1D(
            filters,
            kernel_size=3,
            padding="same",
            kernel_regularizer=regularizer,
            name="conv2",
        )
        self.add = tf.keras.layers.Add(name="residual_add")
        self.out_act = tf.keras.layers.Activation(activation, name="out_act")
        self.pool = tf.keras.layers.MaxPooling1D(
            pool_size=2,
            strides=2,
            padding="same",
            name="max_pool",
        )

        self.shortcut_projection = None
        self._regularizer = regularizer

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.shortcut_projection = tf.keras.layers.Conv1D(
                self.filters,
                kernel_size=1,
                padding="same",
                kernel_regularizer=self._regularizer,
                name="shortcut_projection",
            )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.act1(x)
        x = self.conv2(x)

        shortcut = inputs
        if self.shortcut_projection is not None:
            shortcut = self.shortcut_projection(shortcut)

        x = self.add([x, shortcut])
        x = self.out_act(x)
        return self.pool(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "activation": self.activation,
                "reg_l2": self.reg_l2,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="POlyNetBaselines")
class ResCNNBaseline(BaseMultiTaskBaseline):
    """Simple residual CNN without the POlyNet multi-scale stem or attention."""

    def __init__(
        self,
        n_outputs: int,
        entry_filters: int = 24,
        entry_kernel_size: int = 9,
        block_filters: Sequence[int] = (32, 48, 64, 96),
        block_kernel_sizes: Sequence[int] = (9, 7, 5, 3),
        **kwargs,
    ):
        super().__init__(n_outputs=n_outputs, **kwargs)

        if len(block_filters) != len(block_kernel_sizes):
            raise ValueError(
                "block_filters and block_kernel_sizes must have the same length"
            )

        self.entry_filters = int(entry_filters)
        self.entry_kernel_size = int(entry_kernel_size)
        self.block_filters = tuple(int(v) for v in block_filters)
        self.block_kernel_sizes = tuple(int(v) for v in block_kernel_sizes)

        self.entry_conv = tf.keras.layers.Conv1D(
            entry_filters,
            kernel_size=entry_kernel_size,
            padding="same",
            kernel_regularizer=self.regularizer,
            name="entry_conv",
        )
        self.entry_act = tf.keras.layers.Activation(
            self.activation,
            name="entry_act",
        )
        self.entry_pool = tf.keras.layers.MaxPooling1D(
            pool_size=2,
            strides=2,
            padding="same",
            name="entry_pool",
        )

        self.residual_blocks = [
            ResidualBlock1D(
                filters=n_filters,
                kernel_size=kernel_size,
                activation=self.activation,
                reg_l2=self.reg_l2,
                name=f"residual_block_{i}",
            )
            for i, (n_filters, kernel_size) in enumerate(
                zip(self.block_filters, self.block_kernel_sizes)
            )
        ]

        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D(
            name="global_avg_pool"
        )
        self.global_max_pool = tf.keras.layers.GlobalMaxPooling1D(
            name="global_max_pool"
        )
        self.pool_concat = tf.keras.layers.Concatenate(name="pool_concat")

    def encode(self, inputs, training=False):
        x = self.entry_conv(inputs)
        x = self.entry_act(x)
        x = self.entry_pool(x)

        for block in self.residual_blocks:
            x = block(x, training=training)

        return self.pool_concat(
            [self.global_avg_pool(x), self.global_max_pool(x)]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "entry_filters": self.entry_filters,
                "entry_kernel_size": self.entry_kernel_size,
                "block_filters": list(self.block_filters),
                "block_kernel_sizes": list(self.block_kernel_sizes),
            }
        )
        return config


BASELINE_MODELS = {
    "mlp": MLPBaseline,
    "cnn": CNNBaseline,
    "rescnn": ResCNNBaseline,
}


def build_baseline_model(name: str, n_outputs: int, **kwargs):
    """Instantiate a neural baseline by name."""
    key = name.strip().lower()
    if key not in BASELINE_MODELS:
        valid = ", ".join(sorted(BASELINE_MODELS))
        raise ValueError(f"Unknown baseline '{name}'. Expected one of: {valid}")
    return BASELINE_MODELS[key](n_outputs=n_outputs, **kwargs)


__all__ = [
    "BaseMultiTaskBaseline",
    "MLPBaseline",
    "CNNBaseline",
    "ResidualBlock1D",
    "ResCNNBaseline",
    "BASELINE_MODELS",
    "build_baseline_model",
]