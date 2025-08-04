# %%
import tensorflow as tf


class ConvResidualBlock(tf.keras.layers.Layer):
    """
    A ResNet-like residual block for 1D data that optionally performs downsampling
    if stride > 1 or if input channels != filters.

    Architecture:
        Main path: Conv1D -> LayerNorm -> ReLU -> Dropout -> Conv1D -> LayerNorm
        Skip path: Projection conv (if needed) to match dimensions
        Output: ReLU(main + skip) -> Optional pooling
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        downsample=True,
        reg_l2=1e-6,
        dropout_rate=0.1,
        use_layer_norm=False,
        activation="relu",
        pool_type="max",  # 'max', 'avg', or None
        **kwargs,
    ):
        super(ConvResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.downsample = downsample
        self.reg_l2 = reg_l2
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.activation = activation
        self.pool_type = pool_type

        # Regularizer
        self.regularizer = tf.keras.regularizers.L2(reg_l2) if reg_l2 > 0 else None

        # Main branch
        self.conv1 = tf.keras.layers.Conv1D(
            filters,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            # kernel_regularizer=self.regularizer,
            # kernel_initializer="he_normal",
            name="conv1",
        )

        if use_layer_norm:
            self.ln1 = tf.keras.layers.LayerNormalization(name="ln1")

        # self.act1 = tf.keras.layers.Activation(activation, name="act1")
        if activation == "relu":
            self.act1 = tf.keras.layers.ReLU(name="act1")
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if dropout_rate > 0:
            self.dropout = tf.keras.layers.Dropout(dropout_rate, name="dropout")

        self.conv2 = tf.keras.layers.Conv1D(
            filters,
            kernel_size=3,  # kernel_size,
            strides=1,
            padding="same",
            # kernel_regularizer=self.regularizer,
            # kernel_initializer="he_normal",
            name="conv2",
        )

        if use_layer_norm:
            self.ln2 = tf.keras.layers.LayerNormalization(name="ln2")

        # Skip connection projection (will be created in build if needed)
        self.projection = None

        # Combine operation
        self.add_op = tf.keras.layers.Add(name="residual_add")
        # self.final_activation = tf.keras.layers.Activation(activation, name="final_act")
        if activation == "relu":
            self.final_activation = tf.keras.layers.ReLU(name="final_act")
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Pooling layer (if specified)
        self.pool = None
        if pool_type == "max" and (downsample or stride > 1):
            self.pool = tf.keras.layers.MaxPooling1D(
                pool_size=kernel_size, strides=stride, padding="valid", name="max_pool"
            )
        elif pool_type == "avg" and (downsample or stride > 1):
            self.pool = tf.keras.layers.AveragePooling1D(
                pool_size=kernel_size, strides=stride, padding="valid", name="avg_pool"
            )

    def build(self, input_shape):
        input_channels = input_shape[-1]

        # Build main path
        self.conv1.build(input_shape)
        out_shape = self.conv1.compute_output_shape(input_shape)

        if self.use_layer_norm:
            self.ln1.build(out_shape)

        if self.dropout_rate > 0:
            self.dropout.build(out_shape)

        self.conv2.build(out_shape)
        out_shape = self.conv2.compute_output_shape(out_shape)

        if self.use_layer_norm:
            self.ln2.build(out_shape)

        # Determine if we need a projection for the skip connection
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
            # self.projection.build(input_shape)
        else:
            self.projection = tf.keras.layers.Lambda(
                lambda x: tf.concat([x] * (self.filters // x.shape[-1]), axis=-1),
                name="workaround_projection",
            )
        self.projection.build(input_shape)

        # Build pooling if it exists
        if self.pool is not None:
            # Pool operates on the output after residual connection
            pool_input_shape = (*input_shape[:-1], self.filters)
            self.pool.build(pool_input_shape)

        super(ConvResidualBlock, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # After residual connection, we have (batch, length, filters)
        output_shape = (*input_shape[:-1], self.filters)

        # Apply pooling if present
        if self.pool is not None:
            output_shape = self.pool.compute_output_shape(output_shape)

        return output_shape

    def call(self, inputs, training=False):
        # Main path
        x = self.conv1(inputs)

        if self.use_layer_norm:
            x = self.ln1(x, training=training)

        x = self.act1(x)

        if self.dropout_rate > 0:
            x = self.dropout(x, training=training)

        x = self.conv2(x)

        if self.use_layer_norm:
            x = self.ln2(x, training=training)

        # Skip connection
        # if self.projection is not None:
        skip = self.projection(inputs)
        # else:
        # If no projection needed, input channels should match filters
        # skip = inputs

        # Residual connection
        out = self.add_op([x, skip])
        out = self.final_activation(out)

        # Optional pooling
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
                "use_layer_norm": self.use_layer_norm,
                "activation": self.activation,
                "pool_type": self.pool_type,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiHeadAttentionBlock(tf.keras.layers.Layer):
    """
    An improved multi-head self-attention block with proper normalization,
    feedforward network, and flexible dropout strategies.

    Architecture (Pre-LN Transformer style):
        x -> LayerNorm -> MultiHeadAttention -> Dropout -> Add(x, .) ->
        LayerNorm -> FeedForward -> Dropout -> Add(., previous)
    """

    def __init__(
        self,
        filters,
        num_heads=4,
        key_dim=None,
        ff_dim=None,
        dropout_rate=0.1,  # 0.1,
        attention_dropout_rate=0.1,  # 0.1,
        head_dropout_rate=0.25,
        use_layer_norm=True,
        use_feedforward=True,
        feedforward_activation="relu",
        norm_first=True,  # Pre-LN vs Post-LN
        causal=False,  # For causal/masked attention
        **kwargs,
    ):
        super(MultiHeadAttentionBlock, self).__init__(**kwargs)

        self.filters = filters
        self.num_heads = num_heads
        self.key_dim = key_dim or (filters // num_heads)
        self.ff_dim = ff_dim or (4 * filters)  # Standard transformer ratio
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.head_dropout_rate = head_dropout_rate
        self.use_layer_norm = use_layer_norm
        self.use_feedforward = use_feedforward
        self.feedforward_activation = feedforward_activation
        self.norm_first = norm_first
        self.causal = causal

        # Validate parameters
        if filters % num_heads != 0:
            raise ValueError(
                f"filters={filters} must be divisible by num_heads={num_heads}"
            )

        # Multi-head attention
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.key_dim,
            # dropout=attention_dropout_rate,
            # use_bias=True,
            # kernel_initializer="glorot_uniform",
            name="mha",
        )

        # Layer normalization layers
        if use_layer_norm:
            self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln1")
            if use_feedforward:
                self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln2")

        # Dropout layers
        if dropout_rate > 0:
            self.dropout1 = tf.keras.layers.Dropout(dropout_rate, name="dropout1")
            if use_feedforward:
                self.dropout2 = tf.keras.layers.Dropout(dropout_rate, name="dropout2")

        # Feedforward network (optional)
        if use_feedforward:
            self.ff_dense1 = tf.keras.layers.Dense(
                self.ff_dim,
                # activation=feedforward_activation,
                # kernel_initializer="he_normal",
                name="ff_dense1",
            )
            self.ff_dense2 = tf.keras.layers.Dense(
                filters,
                # kernel_initializer="glorot_uniform",
                name="ff_dense2",
            )

        # Add operations for residual connections
        self.add1 = tf.keras.layers.Add(name="add1")
        if use_feedforward:
            self.add2 = tf.keras.layers.Add(name="add2")

        # Input projection (if needed)
        self.input_projection = None

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Create input projection if dimensions don't match
        if input_dim != self.filters:
            self.input_projection = tf.keras.layers.Dense(
                self.filters,
                kernel_initializer="glorot_uniform",
                name="input_projection",
            )
            self.input_projection.build(input_shape)
            # Update input shape for subsequent layers
            input_shape = (*input_shape[:-1], self.filters)

        # Build attention layer
        self.multihead_attention.build(input_shape, input_shape, input_shape)

        # Build normalization layers
        if self.use_layer_norm:
            self.ln1.build(input_shape)
            if self.use_feedforward:
                self.ln2.build(input_shape)

        # Build feedforward layers
        if self.use_feedforward:
            self.ff_dense1.build(input_shape)
            ff_output_shape = self.ff_dense1.compute_output_shape(input_shape)
            self.ff_dense2.build(ff_output_shape)

        super(MultiHeadAttentionBlock, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # Output shape has the same sequence length but potentially different feature dim
        return (*input_shape[:-1], self.filters)

    def _apply_head_dropout(self, attention_output, training):
        """Apply head dropout - zero out entire attention heads randomly"""
        if not training or self.head_dropout_rate <= 0.0:
            return attention_output

        batch_size = tf.shape(attention_output)[0]
        seq_len = tf.shape(attention_output)[1]

        # Reshape to separate heads: (batch, seq_len, num_heads, head_dim)
        head_dim = self.filters // self.num_heads
        reshaped = tf.reshape(
            attention_output, [batch_size, seq_len, self.num_heads, head_dim]
        )

        # Create head dropout mask: (1, 1, num_heads, 1)
        keep_prob = 1.0 - self.head_dropout_rate
        random_mask = tf.random.uniform([1, 1, self.num_heads, 1])
        head_mask = tf.cast(random_mask < keep_prob, attention_output.dtype)

        # Scale by keep_prob to maintain expected value (like standard dropout)
        # head_mask = head_mask / keep_prob

        # Apply mask and reshape back
        masked = reshaped * head_mask
        masked = tf.reshape(masked, [batch_size, seq_len, self.filters])

        # random_mask = tf.random.uniform([self.num_heads], 0, 1)
        # keep_mask = tf.cast(
        #     random_mask > self.head_dropout_rate, reshaped.dtype
        # )

        # # Reshape keep_mask for broadcasting (1, 1, num_heads, 1)
        # keep_mask = tf.reshape(keep_mask, [1, 1, self.num_heads, 1])

        # # Apply mask
        # masked = reshaped * keep_mask

        return tf.reshape(masked, [batch_size, seq_len, self.filters])

    def call(self, inputs, training=False):  # , return_attention_scores=True
        # Input projection if needed
        if self.input_projection is not None:
            x = self.input_projection(inputs)
        else:
            x = inputs

        # --- Attention Sub-layer ---
        if self.norm_first and self.use_layer_norm:
            # Pre-LN: Normalize before attention
            attn_input = self.ln1(x, training=training)
        else:
            attn_input = x

        # # Create causal mask if needed
        # if self.causal:
        #     seq_len = tf.shape(attn_input)[1]
        #     causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        #     if attention_mask is not None:
        #         attention_mask = attention_mask * causal_mask
        #     else:
        #         attention_mask = causal_mask

        # Apply multi-head attention
        # if return_attention_scores:
        attn_output, attn_weights = self.multihead_attention(
            query=attn_input,
            key=attn_input,
            value=attn_input,
            # attention_mask=attention_mask,
            return_attention_scores=True,
            training=training,
        )
        # else:
        #     attn_output = self.multihead_attention(
        #         query=attn_input,
        #         key=attn_input,
        #         value=attn_input,
        #         # attention_mask=attention_mask,
        #         training=training,
        #     )
        #     attn_weights = None

        # Apply head dropout
        attn_output = self._apply_head_dropout(attn_output, training)

        # Apply dropout
        if self.dropout_rate > 0:
            attn_output = self.dropout1(attn_output, training=training)

        # First residual connection
        x = self.add1([x, attn_output])

        if not self.norm_first and self.use_layer_norm:
            # Post-LN: Normalize after residual connection
            x = self.ln1(x, training=training)

        # --- Feedforward Sub-layer (optional) ---
        if self.use_feedforward:
            if self.norm_first and self.use_layer_norm:
                # Pre-LN: Normalize before feedforward
                ff_input = self.ln2(x, training=training)
            else:
                ff_input = x

            # Apply feedforward network
            ff_output = self.ff_dense1(ff_input)
            ff_output = self.ff_dense2(ff_output)

            # Apply dropout
            if self.dropout_rate > 0:
                ff_output = self.dropout2(ff_output, training=training)

            # Second residual connection
            x = self.add2([x, ff_output])

            if not self.norm_first and self.use_layer_norm:
                # Post-LN: Normalize after residual connection
                x = self.ln2(x, training=training)

        # if return_attention_scores:
        return x, attn_weights
        # else:
        #     return x

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
                "use_layer_norm": self.use_layer_norm,
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


class CustomModel(tf.keras.Model):
    """
    Improved custom model with proper architecture design, efficient attention pooling,
    and better regularization strategies.
    """

    def __init__(
        self,
        reg_l2=1e-9,
        dropout_rate=0.1,
        n_outputs=None,
        conv_dropout_rate=0.1,
        attention_dropout_rate=0,
        head_dropout_rate=0.25,
        num_attention_heads=4,
        use_feedforward=True,
        use_layer_norm=False,
        activation="gelu",
        pooling_strategy="multi",  # 'multi', 'attention', 'global'
        dense_units=[1024, 512],
        branch_units=[256, 64, 32, 16, 8],
        **kwargs,
    ):
        super(CustomModel, self).__init__(**kwargs)

        # Store hyperparameters
        self.reg_l2 = reg_l2
        self.dropout_rate = dropout_rate
        self.conv_dropout_rate = conv_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.head_dropout_rate = head_dropout_rate
        self.n_outputs = n_outputs
        self.num_attention_heads = num_attention_heads
        self.use_feedforward = use_feedforward
        self.use_layer_norm = use_layer_norm
        self.activation = activation
        self.pooling_strategy = pooling_strategy
        self.dense_units = dense_units
        self.branch_units = branch_units

        if n_outputs is None:
            raise ValueError("n_outputs must be specified")

        # Regularizer
        self.regularizer = tf.keras.regularizers.L2(reg_l2) if reg_l2 > 0 else None

        # Build convolutional feature extractor
        self._build_conv_layers()

        # Build attention layer
        self._build_attention_layer()

        # Build pooling layers
        self._build_pooling_layers()

        # Build dense layers
        self._build_dense_layers()

        # Build output branches
        self._build_output_branches()

        # Output layers

        self.weight_output = tf.keras.layers.Dense(
            n_outputs, 
            activation="softmax", 
            name="weight_output"
        )

        self.composition_output = tf.keras.layers.Dense(
            n_outputs,
            activation="elu",
            name="composition_output",
        )

    def _build_conv_layers(self):
        """Build the convolutional residual blocks with progressive downsampling"""
        conv_configs = [
            {"filters": 32, "kernel_size": 11, "downsample": True, "stride": 2},
            {"filters": 64, "kernel_size": 9, "downsample": True, "stride": 2},
            {"filters": 128, "kernel_size": 7, "downsample": True, "stride": 2},
            {"filters": 128, "kernel_size": 5, "downsample": True, "stride": 2},
            {"filters": 128, "kernel_size": 3, "downsample": True, "stride": 2},
            {"filters": 128, "kernel_size": 3, "downsample": True, "stride": 2},
            {"filters": 128, "kernel_size": 3, "downsample": True, "stride": 1},
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
                use_layer_norm=self.use_layer_norm,
                activation="relu",#self.activation,
                pool_type="max" if config["downsample"] else None,
                name=f"conv_block_{i}",
            )
            self.conv_blocks.append(block)

    def _build_attention_layer(self):
        """Build the multi-head attention block"""
        self.multi_head_attention = MultiHeadAttentionBlock(
            filters=128,
            num_heads=self.num_attention_heads,
            dropout_rate=self.attention_dropout_rate,
            head_dropout_rate=self.head_dropout_rate,
            use_feedforward=self.use_feedforward,
            use_layer_norm=self.use_layer_norm,
            norm_first=True,
            name="attention_block",
        )

    def _build_pooling_layers(self):
        """Build pooling layers based on strategy"""
        if self.pooling_strategy in ["multi", "global"]:
            self.global_avg_pooling_x_1 = tf.keras.layers.GlobalAveragePooling1D(
                name="global_avg_pool_x_1"
            )
            self.global_max_pooling_x_1 = tf.keras.layers.GlobalMaxPooling1D(
                name="global_max_pool_x_1"
            )
            self.global_avg_pooling_x_2 = tf.keras.layers.GlobalAveragePooling1D(
                data_format="channels_first", name="global_avg_pool_x_2"
            )
            self.global_max_pooling_x_2 = tf.keras.layers.GlobalMaxPooling1D(
                data_format="channels_first", name="global_max_pool_x_2"
            )

        # if self.pooling_strategy in ["multi", "attention"]:
        #     # Attention pooling will be computed dynamically
        #     pass

    def _build_dense_layers(self):
        """Build shared dense layers"""
        self.dense_layers = []
        self.dropout_layers = []

        for i, units in enumerate(self.dense_units):
            dense = tf.keras.layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=self.regularizer,
                # kernel_initializer='he_normal',
                name=f"shared_dense_{i}",
            )
            dropout = tf.keras.layers.Dropout(
                self.dropout_rate, name=f"shared_dropout_{i}"
            )
            self.dense_layers.append(dense)
            self.dropout_layers.append(dropout)

    def _build_output_branches(self):
        """Build the two output branches with cross-connection"""
        # Weight branch
        self.weight_branch = []
        for i, units in enumerate(self.branch_units):
            layer = tf.keras.layers.Dense(
                units,
                activation=(
                    self.activation if i < len(self.branch_units) - 2 else "tanh"
                ),
                kernel_regularizer=self.regularizer,
                # kernel_initializer='he_normal',
                name=f"weight_dense_{i}",
            )
            self.weight_branch.append(layer)

        # Composition branch
        self.composition_branch = []
        for i, units in enumerate(self.branch_units):
            layer = tf.keras.layers.Dense(
                units,
                activation=(
                    self.activation if i < len(self.branch_units) - 2 else "tanh"
                ),
                kernel_regularizer=self.regularizer,
                # kernel_initializer='he_normal',
                name=f"comp_dense_{i}",
            )
            self.composition_branch.append(layer)

        # Final output layers
        # self.weight_output = tf.keras.layers.Dense(
        #     self.n_outputs,
        #     activation="softmax",
        #     kernel_initializer='glorot_uniform',
        #     name="weight_output"
        # )

        # self.composition_output = tf.keras.layers.Dense(
        #     self.n_outputs,
        #     activation="elu",
        #     kernel_initializer='glorot_uniform',
        #     name="composition_output"
        # )

    def _compute_attention_pooling(self, features, attention_weights):
        """Compute attention-weighted pooling from attention weights"""
        # Average attention weights across heads: [B, T, T] -> [B, T]
        attn_map = tf.reduce_mean(attention_weights, axis=1)

        # Compute per-token importance (mean attention received)
        token_importances = tf.reduce_mean(attn_map, axis=-1)  # [B, T]

        # Normalize to create valid probability distribution
        # token_importances = tf.nn.softmax(token_importances, axis=1)  # [B, T]
        # Apply attention weighting: [B, T, D] * [B, T, 1] -> [B, D]
        # token_importances = tf.expand_dims(token_importances, axis=-1)

        ####### OLD STRATEGY #######
        token_importances = tf.expand_dims(token_importances, axis=-1)
        token_importances = token_importances / (
            tf.reduce_sum(token_importances, axis=1, keepdims=True) + 1e-9
        )
        ##################
        
        attention_pooled = tf.reduce_sum(features * token_importances, axis=1)

        return attention_pooled

    def _apply_pooling(self, features, attention_weights=None):
        """Apply the specified pooling strategy"""
        pooled_features = []

        if self.pooling_strategy == "attention":
            if attention_weights is not None:
                attention_pooled = self._compute_attention_pooling(
                    features, attention_weights
                )
                pooled_features.append(attention_pooled)
            else:
                # Fallback to global pooling if no attention weights
                pooled_features.extend(
                    [
                        self.global_avg_pooling(features),
                        self.global_max_pooling(features),
                    ]
                )

        elif self.pooling_strategy == "global":
            pooled_features.extend(
                [self.global_avg_pooling(features), self.global_max_pooling(features)]
            )

        elif self.pooling_strategy == "multi":
            # Combine all pooling strategies
            pooled_features.extend(
                [
                    self.global_avg_pooling_x_1(features),
                    self.global_max_pooling_x_1(features),
                    self.global_avg_pooling_x_2(features),
                    self.global_max_pooling_x_2(features)
                ]
            )
            if attention_weights is not None:
                attention_pooled = self._compute_attention_pooling(
                    features, attention_weights
                )
                pooled_features.append(attention_pooled)

        return (
            tf.concat(pooled_features, axis=1)
            if len(pooled_features) > 1
            else pooled_features[0]
        )

    def build(self, input_shape):
        """Build the model with the given input shape."""
        # You can print the input shape for debugging if needed
        print(f"Building model with input shape: {input_shape}")
        super().build(input_shape)  # Call the parent class's build() method

    def call(self, inputs, training=False, return_attention_weights=False):
        # First branch (conv + residual + attention)
        x = inputs
        for conv_block in self.conv_blocks:
            x = conv_block(x, training=training)

        # Apply attention
        x, attention_weights = self.multi_head_attention(
            x, training=training  # , return_attention_scores=True
        )

        # Apply pooling strategy
        x = self._apply_pooling(x, attention_weights)

        # Shared dense layers
        for dense_layer, dropout_layer in zip(self.dense_layers, self.dropout_layers):
            x = dense_layer(x)
            x = dropout_layer(x, training=training)

        # Weight branch
        xa = x
        for layer in self.weight_branch:
            xa = layer(xa)

        weight_output = self.weight_output(xa)

        # Composition branch (with cross-connection)
        xb = x
        for layer in self.composition_branch[:-1]:
            xb = layer(xb)
        xb = self.composition_branch[-1](tf.concat([xb, weight_output], axis=1))

        composition_output = self.composition_output(xb)

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
                "use_layer_norm": self.use_layer_norm,
                "activation": self.activation,
                "pooling_strategy": self.pooling_strategy,
                "dense_units": self.dense_units,
                "branch_units": self.branch_units,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# %% CALLBACK ADDITION
import json
import numpy as np


class CosineDecayAfterPlateau(tf.keras.callbacks.Callback):
    def __init__(self, fixed_lr, final_lr, plateau_epochs=30, decay_epochs=50):
        """
        fixed_lr: The learning rate maintained until plateau is detected (e.g. 1e-5)
        final_lr: The final learning rate after cosine decay (non-zero)
        plateau_epochs: Number of epochs with no improvement to wait before starting decay
        decay_epochs: Number of epochs over which to perform cosine decay
        """
        super().__init__()
        self.fixed_lr = fixed_lr
        self.final_lr = final_lr
        self.plateau_epochs = plateau_epochs
        self.decay_epochs = decay_epochs
        self.wait = 0
        # self.best = tf.constant(np.Inf)
        self.best = tf.constant(np.inf)
        self.plateau_start = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get("val_loss")
        if current_loss is None:
            return  # Nothing to monitor

        if not isinstance(self.model.optimizer.learning_rate, tf.Variable):
            current_lr_val = tf.keras.backend.get_value(
                self.model.optimizer.learning_rate
            )
            self.model.optimizer.learning_rate = tf.Variable(
                current_lr_val, trainable=False
            )

        # Check if current validation loss is an improvement
        if current_loss < self.best:
            self.best = current_loss
            self.wait = 0
            # self.plateau_start = None  # reset plateau detection
        else:
            self.wait += 1
            if self.wait >= self.plateau_epochs and self.plateau_start is None:
                self.plateau_start = epoch  # mark the epoch when plateau started

        # If plateau has started, update learning rate using cosine decay
        if self.plateau_start is not None:
            # Calculate how many epochs have passed since plateau began
            t = epoch - self.plateau_start
            # Limit t to decay_epochs so that after decay_epochs the lr stays at final_lr
            t = min(t, self.decay_epochs)
            # Cosine decay formula:
            #   lr = final_lr + 0.5 * (fixed_lr - final_lr) * (1 + cos(pi * t / decay_epochs))
            new_lr = self.final_lr + 0.5 * (self.fixed_lr - self.final_lr) * (
                1 + tf.cos(np.pi * t / self.decay_epochs)
            )
            self.model.optimizer.learning_rate.assign(new_lr)
            print(f"Epoch {epoch+1}: setting learning rate to {new_lr:.2e}")
        else:
            # If plateau has not started, ensure the learning rate stays at fixed_lr.
            # tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.fixed_lr)
            self.model.optimizer.learning_rate.assign(self.fixed_lr)


class SaveHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, save_interval=10):
        """
        Custom callback to save training history every `save_interval` epochs.

        Parameters:
        - save_path (str): Path to save the history JSON file.
        - save_interval (int): Number of epochs between each save.
        """
        super().__init__()
        self.save_path = save_path
        self.save_interval = save_interval
        self.learning_rates = []  # Store learning rate per epoch

    def on_epoch_end(self, epoch, logs=None):
        """Save history every `save_interval` epochs, including learning rate tracking."""
        logs = logs or {}

        # Get current training history
        history_dict = self.model.history.history

        # Convert TensorFlow tensors to Python-native types
        history_serializable = {
            k: [float(v) for v in values] for k, values in history_dict.items()
        }

        # Ensure learning rate tracking
        if "learning_rate" in history_dict:
            # Append existing learning rates (Ensure float conversion)
            self.learning_rates.extend(
                [float(lr) for lr in history_dict["learning_rate"]]
            )
        else:
            # If learning rate is missing, get it manually from optimizer
            optimizer = self.model.optimizer
            lr = float(
                tf.keras.backend.get_value(optimizer.learning_rate)
            )  # Convert tensor to float
            self.learning_rates.append(lr)

        # Ensure learning rate is stored in the history
        history_serializable["learning_rate"] = self.learning_rates

        # Save history every `save_interval` epochs
        if (epoch + 1) % self.save_interval == 0:
            with open(self.save_path, "w") as f:
                json.dump(history_serializable, f)

            print(f"[INFO] Saved training history at epoch {epoch+1}")


class TestSetEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset, test_steps, copolymer_list, strategy):
        """
        Multi-GPU compatible callback to evaluate test dataset after each epoch.

        Parameters:
            test_dataset (tf.data.Dataset): The test dataset.
            test_steps (int): Number of test batches.
            copolymer_list (list): List of copolymer names.
            strategy (tf.distribute.Strategy): Multi-GPU strategy.
        """
        super().__init__()
        self.test_dataset = test_dataset
        self.test_steps = test_steps
        self.copolymer_list = copolymer_list
        self.strategy = strategy  # Needed for multi-GPU execution

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}
        print(f"\n[INFO] Evaluating on test set at epoch {epoch+1}...")

        test_iterator = iter(self.test_dataset)  # Create a new iterator

        y_true_w_list = []
        y_true_c_list = []
        y_pred_w_list = []
        y_pred_c_list = []

        for _ in range(self.test_steps):
            try:
                batch = next(test_iterator)  # Fetch the next batch
                X_batch, y_batch = batch
            except StopIteration:
                print("[WARNING] Test dataset exhausted early.")
                break  # Prevent OUT_OF_RANGE error

            y_pred = self.model(
                X_batch, training=False
            )  # Get predictions without affecting batch norm, dropout

            y_pred_w_list.append(y_pred["weight_output"])
            y_pred_c_list.append(y_pred["composition_output"])

            y_true_w_list.append(y_batch["weight_output"])
            y_true_c_list.append(y_batch["composition_output"])

        y_true_w = tf.concat(y_true_w_list, axis=0)
        y_true_c = tf.concat(y_true_c_list, axis=0)

        y_pred_w = tf.concat(y_pred_w_list, axis=0)
        y_pred_c = tf.concat(y_pred_c_list, axis=0)

        # Convert all tensors to float32 to avoid dtype mismatches
        y_true_w = tf.cast(y_true_w, tf.float32)
        y_true_c = tf.cast(y_true_c, tf.float32)
        y_pred_w = tf.cast(y_pred_w, tf.float32)
        y_pred_c = tf.cast(y_pred_c, tf.float32)

        # # Compute Mean Absolute Error (MAE)
        # mae_weight = tf.reduce_mean(tf.abs(y_true_w - y_pred_w), axis=0)
        # mae_composition = tf.reduce_mean(tf.abs(y_true_c - y_pred_c), axis=0)

        samples_w = tf.reduce_sum(
            tf.where(tf.logical_and(y_true_w == 0.0, y_pred_w < 1e-3), 0.0, 1.0), axis=0
        )
        samples_c = tf.reduce_sum(
            tf.where(tf.logical_and(y_true_c == -1.0, y_pred_c < 0.0), 0.0, 1.0), axis=0
        )

        delta_w = tf.where(
            tf.logical_and(y_true_w == 0.0, y_pred_w < 1e-3),
            0.0,
            tf.abs(y_true_w - y_pred_w),
        )
        delta_c = tf.where(
            tf.logical_and(y_true_c == -1.0, y_pred_c < 0.0),
            0.0,
            tf.abs(y_true_c - y_pred_c),
        )

        mae_weight = tf.reduce_sum(delta_w, axis=0) / samples_w
        # mae_weight = tf.reduce_sum(tf.abs(y_true_w - y_pred_w), axis=0) / samples_w
        mae_composition = tf.reduce_sum(delta_c, axis=0) / samples_c
        # mae_composition = tf.reduce_sum(tf.abs(y_true_c - y_pred_c), axis=0) / samples_c

        # print("  ---- Per-Copolymer Weight MAE ----")
        for i, copolymer in enumerate(self.copolymer_list):
            logs[f"test_mae_weight_{copolymer}"] = float(mae_weight[i].numpy())
            # print(f"  {copolymer} Weight MAE: {mae_weight[i].numpy():.5f}")

        # print("\n  ---- Per-Copolymer Composition MAE ----")
        for i, copolymer in enumerate(self.copolymer_list):
            logs[f"test_mae_composition_{copolymer}"] = float(
                mae_composition[i].numpy()
            )
            # print(f"  {copolymer} Composition MAE: {mae_composition[i].numpy():.5f}")

        if epoch + 1 % 50 == 0:
            print("  ---- Per-Copolymer Weight MAE ----")
            for i, copolymer in enumerate(self.copolymer_list):
                print(f"  {copolymer} Weight MAE: {mae_weight[i].numpy():.5f}")

            print("\n  ---- Per-Copolymer Composition MAE ----")
            for i, copolymer in enumerate(self.copolymer_list):
                print(
                    f"  {copolymer} Composition MAE: {mae_composition[i].numpy():.5f}"
                )

            print("\n")
