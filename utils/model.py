#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DeepNMR model definition.

This module defines the neural architecture used by the DeepNMR pipeline:

  - a multi-scale convolutional stem;
  - a stack of residual 1D convolutional blocks with progressive downsampling;
  - a pre-normalized multi-head self-attention block;
  - multi-strategy pooling;
  - a shared dense trunk;
  - two task-specific branches for mixture weights and copolymer compositions.

The file also contains training callbacks used by the training/fine-tuning
scripts.

Important
---------
This file is weight-compatibility sensitive. Do not change layer names, layer
creation order, active architectural defaults, or the forward pass unless the
model is intentionally retrained from scratch.
"""

from __future__ import annotations

import json

import numpy as np
import tensorflow as tf

# %% CONVOLUTIONAL RESIDUAL BLOCK


class ConvResidualBlock(tf.keras.layers.Layer):
    """Residual 1D convolutional block with optional pooling downsampling.

    Architecture:
        Conv1D -> optional LayerNorm -> activation -> optional Dropout
        -> Conv1D -> optional LayerNorm
        -> residual Add with projection/identity workaround
        -> activation
        -> optional pooling
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        downsample=True,
        reg_l2=1e-6,
        dropout_rate=0.1,
        use_layer_norm_conv=False,
        activation="relu",
        pool_type="max",
        **kwargs,
    ):
        super(ConvResidualBlock, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.downsample = downsample
        self.reg_l2 = reg_l2
        self.dropout_rate = dropout_rate
        self.use_layer_norm_conv = use_layer_norm_conv
        self.activation = activation
        self.pool_type = pool_type

        self.regularizer = tf.keras.regularizers.L2(reg_l2) if reg_l2 > 0 else None

        self.conv1 = tf.keras.layers.Conv1D(
            filters,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            name="conv1",
        )

        if use_layer_norm_conv:
            self.ln1 = tf.keras.layers.LayerNormalization(name="ln1")

        if activation == "relu":
            self.act1 = tf.keras.layers.ReLU(name="act1")
        elif activation == "gelu":
            self.act1 = tf.keras.layers.Activation("gelu", name="act1")
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if dropout_rate > 0:
            self.dropout = tf.keras.layers.Dropout(dropout_rate, name="dropout")

        self.conv2 = tf.keras.layers.Conv1D(
            filters,
            kernel_size=3,
            strides=1,
            padding="same",
            name="conv2",
        )

        if use_layer_norm_conv:
            self.ln2 = tf.keras.layers.LayerNormalization(name="ln2")

        self.projection = None
        self.add_op = tf.keras.layers.Add(name="residual_add")

        if activation == "relu":
            self.final_activation = tf.keras.layers.ReLU(name="final_act")
        elif activation == "gelu":
            self.final_activation = tf.keras.layers.Activation("gelu", name="final_act")
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.pool = None

        if pool_type == "max" and (downsample or stride > 1):
            self.pool = tf.keras.layers.MaxPooling1D(
                pool_size=kernel_size,
                strides=stride,
                padding="valid",
                name="max_pool",
            )
        elif pool_type == "avg" and (downsample or stride > 1):
            self.pool = tf.keras.layers.AveragePooling1D(
                pool_size=kernel_size,
                strides=stride,
                padding="valid",
                name="avg_pool",
            )

    def build(self, input_shape):
        input_channels = input_shape[-1]

        self.conv1.build(input_shape)
        out_shape = self.conv1.compute_output_shape(input_shape)

        if self.use_layer_norm_conv:
            self.ln1.build(out_shape)

        if self.dropout_rate > 0:
            self.dropout.build(out_shape)

        self.conv2.build(out_shape)
        out_shape = self.conv2.compute_output_shape(out_shape)

        if self.use_layer_norm_conv:
            self.ln2.build(out_shape)

        need_projection = input_channels != self.filters

        if need_projection:
            self.projection = tf.keras.layers.Conv1D(
                self.filters,
                kernel_size=1,
                strides=1,
                padding="same",
                kernel_regularizer=self.regularizer,
                kernel_initializer="he_normal",
                name="projection",
            )
        else:
            self.projection = tf.keras.layers.Lambda(
                lambda x: tf.concat([x] * (self.filters // x.shape[-1]), axis=-1),
                name="workaround_projection",
            )

        self.projection.build(input_shape)

        if self.pool is not None:
            pool_input_shape = (*input_shape[:-1], self.filters)
            self.pool.build(pool_input_shape)

        super(ConvResidualBlock, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_shape = (*input_shape[:-1], self.filters)

        if self.pool is not None:
            output_shape = self.pool.compute_output_shape(output_shape)

        return output_shape

    def call(self, inputs, training=False):
        x = self.conv1(inputs)

        if self.use_layer_norm_conv:
            x = self.ln1(x, training=training)

        x = self.act1(x)

        if self.dropout_rate > 0:
            x = self.dropout(x, training=training)

        x = self.conv2(x)

        if self.use_layer_norm_conv:
            x = self.ln2(x, training=training)

        skip = self.projection(inputs)

        out = self.add_op([x, skip])
        out = self.final_activation(out)

        if self.pool is not None:
            out = self.pool(out)

        return out

    def get_config(self):
        config = super(ConvResidualBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "downsample": self.downsample,
                "reg_l2": self.reg_l2,
                "dropout_rate": self.dropout_rate,
                "use_layer_norm_conv": self.use_layer_norm_conv,
                "activation": self.activation,
                "pool_type": self.pool_type,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% MULTI-HEAD ATTENTION BLOCK


class MultiHeadAttentionBlock(tf.keras.layers.Layer):
    """Pre-LN multi-head self-attention block with optional feedforward network.

    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> Dropout -> Add
          -> LayerNorm -> FeedForward -> Dropout -> Add
    """

    def __init__(
        self,
        filters,
        num_heads=4,
        key_dim=None,
        ff_dim=None,
        dropout_rate=0.1,
        attention_dropout_rate=0.05,
        head_dropout_rate=0.0,
        use_layer_norm_attn=True,
        use_feedforward=True,
        feedforward_activation="gelu",
        norm_first=True,
        causal=False,
        **kwargs,
    ):
        super(MultiHeadAttentionBlock, self).__init__(**kwargs)

        self.filters = filters
        self.num_heads = num_heads
        self.key_dim = key_dim or (filters // num_heads)
        self.ff_dim = ff_dim or (4 * filters)
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.head_dropout_rate = head_dropout_rate
        self.use_layer_norm_attn = use_layer_norm_attn
        self.use_feedforward = use_feedforward
        self.feedforward_activation = feedforward_activation
        self.norm_first = norm_first
        self.causal = causal

        if filters % num_heads != 0:
            raise ValueError(
                f"filters={filters} must be divisible by num_heads={num_heads}"
            )

        self.multihead_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.key_dim,
            dropout=self.attention_dropout_rate,
            name="mha",
        )

        if use_layer_norm_attn:
            self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln1")

            if use_feedforward:
                self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln2")

        if dropout_rate > 0:
            self.dropout1 = tf.keras.layers.Dropout(dropout_rate, name="dropout1")

            if use_feedforward:
                self.dropout2 = tf.keras.layers.Dropout(dropout_rate, name="dropout2")

        if use_feedforward:
            self.ff_dense1 = tf.keras.layers.Dense(
                self.ff_dim,
                activation=feedforward_activation,
                name="ff_dense1",
            )
            self.ff_dense2 = tf.keras.layers.Dense(
                filters,
                name="ff_dense2",
            )

        self.add1 = tf.keras.layers.Add(name="add1")

        if use_feedforward:
            self.add2 = tf.keras.layers.Add(name="add2")

        self.input_projection = None

    def build(self, input_shape):
        input_dim = input_shape[-1]

        if input_dim != self.filters:
            self.input_projection = tf.keras.layers.Dense(
                self.filters,
                kernel_initializer="glorot_uniform",
                name="input_projection",
            )
            self.input_projection.build(input_shape)
            input_shape = (*input_shape[:-1], self.filters)

        self.multihead_attention.build(input_shape, input_shape, input_shape)

        if self.use_layer_norm_attn:
            self.ln1.build(input_shape)

            if self.use_feedforward:
                self.ln2.build(input_shape)

        if self.use_feedforward:
            self.ff_dense1.build(input_shape)
            ff_output_shape = self.ff_dense1.compute_output_shape(input_shape)
            self.ff_dense2.build(ff_output_shape)

        super(MultiHeadAttentionBlock, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.filters)

    def _apply_head_dropout(self, attention_output, training):
        """Apply dropout at the level of whole attention heads."""
        if not training or self.head_dropout_rate <= 0.0:
            return attention_output

        batch_size = tf.shape(attention_output)[0]
        seq_len = tf.shape(attention_output)[1]
        head_dim = self.filters // self.num_heads

        reshaped = tf.reshape(
            attention_output,
            [batch_size, seq_len, self.num_heads, head_dim],
        )

        keep_prob = 1.0 - self.head_dropout_rate
        random_mask = tf.random.uniform([1, 1, self.num_heads, 1])
        head_mask = tf.cast(random_mask < keep_prob, attention_output.dtype)

        masked = reshaped * head_mask
        masked = tf.reshape(masked, [batch_size, seq_len, self.filters])

        return tf.reshape(masked, [batch_size, seq_len, self.filters])

    def call(self, inputs, training=False):
        if self.input_projection is not None:
            x = self.input_projection(inputs)
        else:
            x = inputs

        if self.norm_first and self.use_layer_norm_attn:
            attn_input = self.ln1(x, training=training)
        else:
            attn_input = x

        attn_output, attn_weights = self.multihead_attention(
            query=attn_input,
            key=attn_input,
            value=attn_input,
            return_attention_scores=True,
            training=training,
        )

        attn_output = self._apply_head_dropout(attn_output, training)

        if self.dropout_rate > 0:
            attn_output = self.dropout1(attn_output, training=training)

        x = self.add1([x, attn_output])

        if not self.norm_first and self.use_layer_norm_attn:
            x = self.ln1(x, training=training)

        if self.use_feedforward:
            if self.norm_first and self.use_layer_norm_attn:
                ff_input = self.ln2(x, training=training)
            else:
                ff_input = x

            ff_output = self.ff_dense1(ff_input)
            ff_output = self.ff_dense2(ff_output)

            if self.dropout_rate > 0:
                ff_output = self.dropout2(ff_output, training=training)

            x = self.add2([x, ff_output])

            if not self.norm_first and self.use_layer_norm_attn:
                x = self.ln2(x, training=training)

        return x, attn_weights

    def get_config(self):
        config = super(MultiHeadAttentionBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
                "attention_dropout_rate": self.attention_dropout_rate,
                "head_dropout_rate": self.head_dropout_rate,
                "use_layer_norm_attn": self.use_layer_norm_attn,
                "use_feedforward": self.use_feedforward,
                "feedforward_activation": self.feedforward_activation,
                "norm_first": self.norm_first,
                "causal": self.causal,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% LEGACY / OPTIONAL COMPONENTS - NOT USED BY CURRENT ACTIVE FORWARD PASS


class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    """Optional sinusoidal positional encoding with learnable scale.

    This layer is kept for backward compatibility and experimentation. The
    current active ``CustomModel`` does not call ``_build_positional_encoding()``
    and does not apply this layer in the forward pass.
    """

    def __init__(self, max_wavelength=10000.0, initial_scale=0.05, **kwargs):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength
        self.initial_scale = initial_scale

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="positional_scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_scale),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        channels = tf.shape(x)[2]

        position = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        channel_idx = tf.cast(tf.range(channels)[tf.newaxis, :], tf.float32)

        angle_rates = 1.0 / tf.pow(
            self.max_wavelength,
            (2.0 * tf.floor(channel_idx / 2.0)) / tf.cast(channels, tf.float32),
        )

        angles = position * angle_rates
        channel_int = tf.range(channels)[tf.newaxis, :]

        pe = tf.where(
            tf.equal(channel_int % 2, 0),
            tf.sin(angles),
            tf.cos(angles),
        )

        pe = pe[tf.newaxis, :, :]
        pe = tf.cast(pe, x.dtype)

        return x + tf.cast(self.alpha, x.dtype) * pe

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_wavelength": self.max_wavelength,
                "initial_scale": self.initial_scale,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="Custom", name="scaled_sigmoid")
def scaled_sigmoid(x):
    """Legacy activation kept for compatibility with older gated outputs."""
    return 1.001 * tf.sigmoid(x)


# %% CUSTOM DEEPNMR MODEL


class CustomModel(tf.keras.Model):
    """DeepNMR multi-task model for weights and copolymer compositions."""

    def __init__(
        self,
        reg_l2=1e-9,
        dropout_rate=0.1,
        n_outputs=None,
        conv_dropout_rate=0.1,
        attention_dropout_rate=0.05,
        head_dropout_rate=0.0,
        num_attention_heads=6,
        use_feedforward=True,
        use_layer_norm_attn=True,
        use_layer_norm_conv=False,
        activation="gelu",
        pooling_strategy="multi",
        dense_units=[1024, 512],
        branch_units=[256, 64, 16],
        consistency_lambda=0.025,
        consistency_weight_threshold=1e-2,
        consistency_comp_threshold=0.0,
        gate_consistency_lambda=0.1,
        **kwargs,
    ):
        super(CustomModel, self).__init__(**kwargs)

        self.reg_l2 = reg_l2
        self.dropout_rate = dropout_rate
        self.conv_dropout_rate = conv_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.head_dropout_rate = head_dropout_rate
        self.n_outputs = n_outputs
        self.num_attention_heads = num_attention_heads
        self.use_feedforward = use_feedforward
        self.use_layer_norm_attn = use_layer_norm_attn
        self.use_layer_norm_conv = use_layer_norm_conv
        self.activation = activation
        self.pooling_strategy = pooling_strategy
        self.dense_units = dense_units
        self.branch_units = branch_units
        self.consistency_lambda = consistency_lambda
        self.consistency_weight_threshold = consistency_weight_threshold
        self.consistency_comp_threshold = consistency_comp_threshold
        self.gate_consistency_lambda = gate_consistency_lambda

        if n_outputs is None:
            raise ValueError("n_outputs must be specified")

        self.regularizer = tf.keras.regularizers.L2(reg_l2) if reg_l2 > 0 else None

        self._build_stem_layers()
        self._build_conv_layers()
        self._build_attention_layer()
        self._build_pooling_layers()
        self._build_dense_layers()
        self._build_output_branches()

        self.weight_output = tf.keras.layers.Dense(
            n_outputs,
            activation="softmax",
            name="weight_output",
        )

        self.composition_output = tf.keras.layers.Dense(
            n_outputs,
            activation="elu",
            name="composition_output",
        )

    def _build_stem_layers(self):
        """Build the initial multi-scale convolutional stem."""
        stem_filters = 16
        stem_out_filters = 24

        self.stem_conv_k3 = tf.keras.layers.Conv1D(
            filters=stem_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=None,
            name="stem_conv_k3",
        )

        self.stem_conv_k7 = tf.keras.layers.Conv1D(
            filters=stem_filters,
            kernel_size=7,
            strides=1,
            padding="same",
            activation=None,
            name="stem_conv_k7",
        )

        self.stem_conv_k13 = tf.keras.layers.Conv1D(
            filters=stem_filters,
            kernel_size=13,
            strides=1,
            padding="same",
            activation=None,
            name="stem_conv_k13",
        )

        self.stem_concat = tf.keras.layers.Concatenate(
            axis=-1,
            name="stem_concat",
        )

        self.stem_ln1 = tf.keras.layers.LayerNormalization(name="stem_ln1")

        if self.activation == "relu":
            self.stem_act1 = tf.keras.layers.ReLU(name="stem_act1")
            self.stem_act2 = tf.keras.layers.ReLU(name="stem_act2")
        elif self.activation == "gelu":
            self.stem_act1 = tf.keras.layers.Activation("gelu", name="stem_act1")
            self.stem_act2 = tf.keras.layers.Activation("gelu", name="stem_act2")
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        self.stem_fuse = tf.keras.layers.Conv1D(
            filters=stem_out_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            activation=None,
            name="stem_fuse",
        )

        self.stem_ln2 = tf.keras.layers.LayerNormalization(name="stem_ln2")

    def _apply_stem(self, x, training=False):
        b1 = self.stem_conv_k3(x)
        b2 = self.stem_conv_k7(x)
        b3 = self.stem_conv_k13(x)

        x = self.stem_concat([b1, b2, b3])
        x = self.stem_ln1(x, training=training)
        x = self.stem_act1(x)

        x = self.stem_fuse(x)

        return x

    def _build_conv_layers(self):
        """Build the active residual convolutional backbone."""
        conv_configs = [
            {"filters": 24, "kernel_size": 9, "downsample": True, "stride": 2},
            {"filters": 32, "kernel_size": 9, "downsample": True, "stride": 2},
            {"filters": 48, "kernel_size": 7, "downsample": True, "stride": 2},
            {"filters": 64, "kernel_size": 5, "downsample": True, "stride": 2},
            {"filters": 72, "kernel_size": 5, "downsample": True, "stride": 2},
            {"filters": 96, "kernel_size": 3, "downsample": True, "stride": 2},
            {"filters": 96, "kernel_size": 3, "downsample": True, "stride": 1},
        ]

        self.conv_blocks = []

        for i, config in enumerate(conv_configs):
            block = ConvResidualBlock(
                filters=config["filters"],
                kernel_size=config["kernel_size"],
                stride=config["stride"],
                downsample=config["downsample"],
                reg_l2=self.reg_l2,
                dropout_rate=self.conv_dropout_rate,
                use_layer_norm_conv=self.use_layer_norm_conv,
                activation=self.activation,
                pool_type="max" if config["downsample"] else None,
                name=f"conv_block_{i}",
            )
            self.conv_blocks.append(block)

    def _build_positional_encoding(self):
        """Build optional positional encoding layer.

        Legacy/optional: this method is intentionally not called by the active
        constructor.
        """
        self.pos_encoding = SinusoidalPositionalEncoding(
            initial_scale=0.05,
            name="positional_encoding",
        )

    def _build_attention_layer(self):
        """Build the multi-head attention block."""
        self.multi_head_attention = MultiHeadAttentionBlock(
            filters=96,
            num_heads=self.num_attention_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            head_dropout_rate=self.head_dropout_rate,
            use_feedforward=self.use_feedforward,
            use_layer_norm_attn=self.use_layer_norm_attn,
            norm_first=True,
            name="attention_block",
        )

    def _build_pooling_layers(self):
        """Build pooling layers for the active pooling strategy."""
        if self.pooling_strategy in ["multi", "global"]:
            self.global_avg_pooling_x_1 = tf.keras.layers.GlobalAveragePooling1D(
                name="global_avg_pool_x_1"
            )
            self.global_max_pooling_x_1 = tf.keras.layers.GlobalMaxPooling1D(
                name="global_max_pool_x_1"
            )
            self.global_avg_pooling_x_2 = tf.keras.layers.GlobalAveragePooling1D(
                data_format="channels_first",
                name="global_avg_pool_x_2",
            )
            self.global_max_pooling_x_2 = tf.keras.layers.GlobalMaxPooling1D(
                data_format="channels_first",
                name="global_max_pool_x_2",
            )

    def _build_dense_layers(self):
        """Build shared dense trunk."""
        self.dense_layers = []
        self.dropout_layers = []

        for i, units in enumerate(self.dense_units):
            dense = tf.keras.layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=self.regularizer,
                name=f"shared_dense_{i}",
            )
            dropout = tf.keras.layers.Dropout(
                self.dropout_rate,
                name=f"shared_dropout_{i}",
            )

            self.dense_layers.append(dense)
            self.dropout_layers.append(dropout)

    def _build_output_branches(self):
        """Build weight and composition branches."""
        self.weight_branch = []

        for i, units in enumerate(self.branch_units):
            layer = tf.keras.layers.Dense(
                units,
                activation=(
                    self.activation if i < len(self.branch_units) - 1 else "tanh"
                ),
                kernel_regularizer=self.regularizer,
                name=f"weight_dense_{i}",
            )
            self.weight_branch.append(layer)

        self.composition_branch = []

        for i, units in enumerate(self.branch_units):
            layer = tf.keras.layers.Dense(
                units,
                activation=(
                    self.activation if i < len(self.branch_units) - 1 else "tanh"
                ),
                kernel_regularizer=self.regularizer,
                name=f"comp_dense_{i}",
            )
            self.composition_branch.append(layer)

    def _compute_attention_pooling(self, features, attention_weights):
        """Compute attention-weighted pooling from attention weights."""
        attn_map = tf.reduce_mean(attention_weights, axis=1)
        token_importances = tf.reduce_mean(attn_map, axis=1)

        token_importances = tf.expand_dims(token_importances, axis=-1)
        token_importances = token_importances / (
            tf.reduce_sum(token_importances, axis=1, keepdims=True) + 1e-9
        )

        attention_pooled = tf.reduce_sum(features * token_importances, axis=1)

        return attention_pooled

    def _apply_pooling(self, features, attention_weights=None):
        """Apply the selected pooling strategy."""
        pooled_features = []

        if self.pooling_strategy == "attention":
            if attention_weights is not None:
                attention_pooled = self._compute_attention_pooling(
                    features,
                    attention_weights,
                )
                pooled_features.append(attention_pooled)
            else:
                pooled_features.extend(
                    [
                        self.global_avg_pooling(features),
                        self.global_max_pooling(features),
                    ]
                )

        elif self.pooling_strategy == "global":
            pooled_features.extend(
                [
                    self.global_avg_pooling(features),
                    self.global_max_pooling(features),
                ]
            )

        elif self.pooling_strategy == "multi":
            pooled_features.extend(
                [
                    self.global_avg_pooling_x_1(features),
                    self.global_max_pooling_x_1(features),
                    self.global_avg_pooling_x_2(features),
                    self.global_max_pooling_x_2(features),
                ]
            )

            if attention_weights is not None:
                attention_pooled = self._compute_attention_pooling(
                    features,
                    attention_weights,
                )
                pooled_features.append(attention_pooled)

        return (
            tf.concat(pooled_features, axis=1)
            if len(pooled_features) > 1
            else pooled_features[0]
        )

    def build(self, input_shape):
        """Build the model with a given input shape."""
        print(f"Building model with input shape: {input_shape}")
        super().build(input_shape)

    def call(self, inputs, training=False, return_attention_weights=False):
        x = inputs

        x = self._apply_stem(x, training=training)

        for conv_block in self.conv_blocks:
            x = conv_block(x, training=training)

        x, attention_weights = self.multi_head_attention(
            x,
            training=training,
        )

        x = self._apply_pooling(x, attention_weights)

        for dense_layer, dropout_layer in zip(self.dense_layers, self.dropout_layers):
            x = dense_layer(x)
            x = dropout_layer(x, training=training)

        xa = x

        for layer in self.weight_branch:
            xa = layer(xa)

        weight_output = self.weight_output(xa)

        xb = x

        for layer in self.composition_branch[:-1]:
            xb = layer(xb)

        xb = self.composition_branch[-1](tf.concat([xb, weight_output], axis=1))

        composition_output = self.composition_output(xb)

        presence_strength = tf.nn.relu(
            weight_output - self.consistency_weight_threshold
        )
        negative_comp = tf.nn.relu(self.consistency_comp_threshold - composition_output)
        consistency_penalty = tf.reduce_mean(presence_strength * negative_comp)

        self.add_loss(self.consistency_lambda * consistency_penalty)

        outputs = {
            "weight_output": weight_output,
            "composition_output": composition_output,
        }

        if return_attention_weights:
            outputs["attention_weights"] = attention_weights

        return outputs

    def get_config(self):
        config = super(CustomModel, self).get_config()
        config.update(
            {
                "reg_l2": self.reg_l2,
                "dropout_rate": self.dropout_rate,
                "conv_dropout_rate": self.conv_dropout_rate,
                "attention_dropout_rate": self.attention_dropout_rate,
                "head_dropout_rate": self.head_dropout_rate,
                "n_outputs": self.n_outputs,
                "num_attention_heads": self.num_attention_heads,
                "use_layer_norm_attn": self.use_layer_norm_attn,
                "use_layer_norm_conv": self.use_layer_norm_conv,
                "use_feedforward": self.use_feedforward,
                "activation": self.activation,
                "pooling_strategy": self.pooling_strategy,
                "dense_units": self.dense_units,
                "branch_units": self.branch_units,
                "consistency_lambda": self.consistency_lambda,
                "consistency_weight_threshold": self.consistency_weight_threshold,
                "consistency_comp_threshold": self.consistency_comp_threshold,
                "gate_consistency_lambda": self.gate_consistency_lambda,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% TRAINING CALLBACKS


class CosineDecayAfterPlateau(tf.keras.callbacks.Callback):
    """Maintain a fixed LR until validation plateau, then apply cosine decay."""

    def __init__(self, fixed_lr, final_lr, plateau_epochs=30, decay_epochs=50):
        super().__init__()

        self.fixed_lr = fixed_lr
        self.final_lr = final_lr
        self.plateau_epochs = plateau_epochs
        self.decay_epochs = decay_epochs
        self.wait = 0
        self.best = tf.constant(np.inf)
        self.plateau_start = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get("val_loss")

        if current_loss is None:
            return

        if not isinstance(self.model.optimizer.learning_rate, tf.Variable):
            current_lr_val = tf.keras.backend.get_value(
                self.model.optimizer.learning_rate
            )
            self.model.optimizer.learning_rate = tf.Variable(
                current_lr_val,
                trainable=False,
            )

        if current_loss < self.best:
            self.best = current_loss
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.plateau_epochs and self.plateau_start is None:
                self.plateau_start = epoch

        if self.plateau_start is not None:
            t = epoch - self.plateau_start
            t = min(t, self.decay_epochs)

            new_lr = self.final_lr + 0.5 * (self.fixed_lr - self.final_lr) * (
                1 + tf.cos(np.pi * t / self.decay_epochs)
            )

            self.model.optimizer.learning_rate.assign(new_lr)
            print(f"Epoch {epoch + 1}: setting learning rate to {new_lr:.2e}")
        else:
            self.model.optimizer.learning_rate.assign(self.fixed_lr)


class SaveHistoryCallback(tf.keras.callbacks.Callback):
    """Save training history periodically as JSON."""

    def __init__(self, save_path, save_interval=10):
        super().__init__()

        self.save_path = save_path
        self.save_interval = save_interval
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        history_dict = self.model.history.history

        history_serializable = {
            k: [float(v) for v in values] for k, values in history_dict.items()
        }

        if "learning_rate" in history_dict:
            self.learning_rates.extend(
                [float(lr) for lr in history_dict["learning_rate"]]
            )
        else:
            optimizer = self.model.optimizer
            lr = float(tf.keras.backend.get_value(optimizer.learning_rate))
            self.learning_rates.append(lr)

        history_serializable["learning_rate"] = self.learning_rates

        if (epoch + 1) % self.save_interval == 0:
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(history_serializable, f)

            print(f"[INFO] Saved training history at epoch {epoch + 1}")


class TestSetEvaluationCallback(tf.keras.callbacks.Callback):
    """Evaluate a supervised test dataset after each epoch.

    The callback computes per-copolymer MAE diagnostics for weight and
    composition outputs and injects them into the Keras logs dictionary.
    """

    def __init__(self, test_dataset, test_steps, copolymer_list, strategy):
        super().__init__()

        self.test_dataset = test_dataset
        self.test_steps = test_steps
        self.copolymer_list = copolymer_list
        self.strategy = strategy

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\n[INFO] Evaluating on test set at epoch {epoch + 1}...")

        test_iterator = iter(self.test_dataset)

        y_true_w_list = []
        y_true_c_list = []
        y_pred_w_list = []
        y_pred_c_list = []

        for _ in range(self.test_steps):
            try:
                batch = next(test_iterator)
                X_batch, y_batch = batch
            except StopIteration:
                print("[WARNING] Test dataset exhausted early.")
                break

            y_pred = self.model(X_batch, training=False)

            y_pred_w_list.append(y_pred["weight_output"])
            y_pred_c_list.append(y_pred["composition_output"])

            y_true_w_list.append(y_batch["weight_output"])
            y_true_c_list.append(y_batch["composition_output"])

        y_true_w = tf.concat(y_true_w_list, axis=0)
        y_true_c = tf.concat(y_true_c_list, axis=0)

        y_pred_w = tf.concat(y_pred_w_list, axis=0)
        y_pred_c = tf.concat(y_pred_c_list, axis=0)

        y_true_w = tf.cast(y_true_w, tf.float32)
        y_true_c = tf.cast(y_true_c, tf.float32)
        y_pred_w = tf.cast(y_pred_w, tf.float32)
        y_pred_c = tf.cast(y_pred_c, tf.float32)

        samples_w = tf.reduce_sum(
            tf.where(
                tf.logical_and(y_true_w == 0.0, y_pred_w < 1e-3),
                0.0,
                1.0,
            ),
            axis=0,
        )

        samples_c = tf.reduce_sum(
            tf.where(
                tf.logical_and(y_true_c == -1.0, y_pred_c < 0.0),
                0.0,
                1.0,
            ),
            axis=0,
        )

        delta_w = tf.where(
            tf.logical_and(y_true_w == 0.0, y_pred_w < 1e-3),
            0.0,
            tf.abs(y_true_w - y_pred_w),
        )

        y_true_c_compute_errors = tf.where(y_true_c == -1.0, 0.0, y_true_c)

        delta_c = tf.where(
            tf.logical_and(y_true_c == -1.0, y_pred_c < 0.0),
            0.0,
            tf.abs(y_true_c_compute_errors - y_pred_c),
        )

        mae_weight = tf.reduce_sum(delta_w, axis=0) / samples_w
        mae_composition = tf.reduce_sum(delta_c, axis=0) / samples_c

        for i, copolymer in enumerate(self.copolymer_list):
            logs[f"test_mae_weight_{copolymer}"] = float(mae_weight[i].numpy())

        for i, copolymer in enumerate(self.copolymer_list):
            logs[f"test_mae_composition_{copolymer}"] = float(
                mae_composition[i].numpy()
            )

        if (epoch + 1) % 10 == 0:
            print("  ---- Per-Copolymer Weight MAE ----")

            for i, copolymer in enumerate(self.copolymer_list):
                print(f"  {copolymer} Weight MAE: {mae_weight[i].numpy():.5f}")

            print("\n  ---- Per-Copolymer Composition MAE ----")

            for i, copolymer in enumerate(self.copolymer_list):
                print(
                    f"  {copolymer} Composition MAE: "
                    f"{mae_composition[i].numpy():.5f}"
                )

            print("\n")
