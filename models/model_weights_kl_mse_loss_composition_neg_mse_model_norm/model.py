# %%
import tensorflow as tf


class ConvResidualBlock(tf.keras.layers.Layer):
    """
    A ResNet-like residual block for 1D data that optionally performs downsampling
    if stride > 1 or if input channels != filters.

    main path:
        Conv1D(filters, kernel_size, stride=stride) -> BN -> ReLU ->
        Conv1D(filters, kernel_size, stride=1)      -> BN -> ...
    skip path:
        If stride != 1 or input_channels != filters, use a 1x1 Conv to align shape
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        reg_l2=1e-6,
        dropout_rate=0.1,
        use_projection=True,
        **kwargs,
    ):
        super(ConvResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.reg_l2 = reg_l2
        self.dropout_rate = dropout_rate
        self.use_projection = use_projection

        # # Projection branch
        # self.convP = tf.keras.layers.Conv1D(
        #     filters,
        #     kernel_size=1,  # 3,
        #     strides=1,
        #     padding="same",
        # )

        # Main branch
        self.conv1 = tf.keras.layers.Conv1D(
            filters,
            kernel_size=kernel_size,  # 3,
            strides=1,
            padding="same",
        )
        # self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv1D(
            filters,
            kernel_size=3,
            strides=1,
            padding="same",
        )
        # self.bn2 = tf.keras.layers.BatchNormalization()

        self.add_op = tf.keras.layers.Add()

        # Skip / projection branch (only created if needed)
        # self.shortcut_conv = None
        # if self.use_projection:
        #     # We'll define a 1x1 Conv to match the shape if downsampling
        #     self.shortcut_conv = tf.keras.layers.Conv1D(
        #         filters,
        #         kernel_size=1,
        #         strides=1,
        #         # kernel_regularizer=tf.keras.regularizers.L2(reg_l2),
        #         padding="same",
        #     )
            # self.shortcut_bn = tf.keras.layers.BatchNormalization()

        # self.avg_pool = tf.keras.layers.AveragePooling1D(
        #     pool_size=kernel_size, strides=stride, padding="valid"
        # )
        self.max_pool = tf.keras.layers.MaxPooling1D(
            pool_size=kernel_size, strides=stride, padding="valid"
        )

    def build(self, input_shape):
        # Build main path
        #
        # self.convP.build(input_shape)
        # out_shape = self.convP.compute_output_shape(input_shape)

        self.conv1.build(input_shape)
        out_shape = self.conv1.compute_output_shape(input_shape)
        # self.bn1.build(out_shape)

        self.conv2.build(out_shape)
        out_shape = self.conv2.compute_output_shape(out_shape)
        # self.bn2.build(out_shape)

        # Potentially build projection path
        # if self.shortcut_conv is not None:
        #     self.shortcut_conv.build(input_shape)
        #     out_shape = self.shortcut_conv.compute_output_shape(input_shape)
            # self.shortcut_bn.build(proj_out_shape)

        super(ConvResidualBlock, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # output_shape = self.shortcut_conv.compute_output_shape(input_shape)
        # output_shape = self.avg_pool.compute_output_shape(output_shape)
        # output_shape = self.avg_pool.compute_output_shape((*input_shape[:-1], self.filters))
        # output_shape = self.max_pool.compute_output_shape((*input_shape[:-1], self.filters))
        output_shape = self.max_pool.compute_output_shape((*input_shape[:-1], self.filters))
        return output_shape

    def call(self, inputs, training=False):

        # skip = self.convP(inputs)
        # ----- main path -----
        x = self.conv1(inputs)
        # x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        # x = self.bn2(x, training=training)

        # ----- skip/projection path -----
        # if self.shortcut_conv is not None:
        #     skip = self.shortcut_conv(inputs)
        #     # skip = self.shortcut_bn(skip, training=training)
        # else:
        #     # skip = inputs  # no downsampling or channel change
        #     skip = tf.concat([inputs] * (self.filters//inputs.shape[-1]), axis=-1)

        # Combine
        skip = tf.concat([inputs] * (self.filters//inputs.shape[-1]), axis=-1)
        out = self.add_op([x, skip])
        out = self.relu(out)

        # out = self.avg_pool(out)
        out = self.max_pool(out)

        return out

    def get_config(self):
        config = super(ConvResidualBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "dropout_rate": self.dropout_rate,
                "reg_l2": self.reg_l2,
                "use_projection": self.use_projection,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiHeadAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self, filters, num_heads=4, dropout_rate=0.1, head_dropout_rate=0.3, **kwargs
    ):
        super(MultiHeadAttentionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.num_heads = num_heads
        self.head_dropout_rate = head_dropout_rate

        if filters % num_heads != 0:
            raise ValueError(
                f"filters={filters} must be divisible by num_heads={num_heads}"
            )

        # Multi-head attention layer from Keras
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=filters // num_heads
        )
        # Optionally, use a Conv1D layer to project the input to match the output dimensions if needed
        # self.projection = tf.keras.layers.Conv1D(filters, 1)

        # Skip connection addition
        self.add_op = tf.keras.layers.Add()

        # self.layer_norm = tf.keras.layers.LayerNormalization()

        # self.attention_dropout = tf.keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        # Build the multi-head attention and projection layers
        self.multihead_attention.build(input_shape, input_shape, input_shape)

        super(MultiHeadAttentionBlock, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # Output shape is the same as input shape (due to residual connection)
        output_shape = self.multihead_attention.compute_output_shape(
            input_shape, input_shape, input_shape
        )
        return output_shape

    def call(self, inputs, training=False):
        # Query, key, and value are the same in self-attention

        # Apply PRE layer normalization for better convergence and stability
        # x = self.layer_norm(inputs)

        attention_output, attention_weights = self.multihead_attention(
            query=inputs, key=inputs, value=inputs, return_attention_scores=True
        )

        # attention_output = self.attention_dropout(attention_output, training=training)
        # If dimensions do not match, apply projection to the inputs to match the attention output
        # inputs_projected = self.projection(inputs)

        # -------------------------------------------------------
        #  HEAD DROPOUT: Zero out entire heads at random
        # -------------------------------------------------------
        if training and self.head_dropout_rate > 0.0:
            b = tf.shape(attention_output)[0]  # Batch size
            seq_len = tf.shape(attention_output)[1]  # Sequence length
            d_model = tf.shape(attention_output)[
                2
            ]  # Feature dimension (same as filters)

            head_dim = d_model // self.num_heads
            # if d_model % self.num_heads != 0:
            #     raise ValueError(f"filters={d_model} must be divisible by num_heads={self.num_heads}")

            # Reshape to (batch, seq_len, num_heads, head_dim)
            attention_output = tf.reshape(
                attention_output, [b, seq_len, self.num_heads, head_dim]
            )

            # Draw a Bernoulli mask for each head: (num_heads,)
            random_mask = tf.random.uniform([self.num_heads], 0, 1)
            keep_mask = tf.cast(
                random_mask > self.head_dropout_rate, attention_output.dtype
            )

            # Reshape keep_mask for broadcasting (1, 1, num_heads, 1)
            keep_mask = tf.reshape(keep_mask, [1, 1, self.num_heads, 1])

            # Apply mask
            attention_output = attention_output * keep_mask

            # Reshape back to (batch, seq_len, d_model)
            attention_output = tf.reshape(attention_output, [b, seq_len, d_model])

        # # Apply layer normalization for better convergence and stability
        # attention_output = self.layer_norm(attention_output)

        # Skip connection (Add residual connection: inputs + attention output)
        output = self.add_op([inputs, attention_output])

        # Apply POST layer normalization for better convergence and stability
        # output = self.layer_norm(output, training=training)

        return output, attention_weights

    def get_config(self):
        config = super(MultiHeadAttentionBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "num_heads": self.num_heads,
                "head_dropout_rate": self.head_dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomModel(tf.keras.Model):
    def __init__(self, reg_l2=1e-6, dropout_rate=0.1, n_outputs=None, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.reg_l2 = reg_l2
        self.dropout_rate = dropout_rate
        self.n_outputs = n_outputs

        self.convres0 = ConvResidualBlock(
            filters=32,
            kernel_size=11,
            stride=2,
            reg_l2=reg_l2,
            dropout_rate=dropout_rate,
            use_projection=True,
        )
        self.convres1 = ConvResidualBlock(
            filters=64,
            kernel_size=9,
            stride=2,
            reg_l2=reg_l2,
            dropout_rate=dropout_rate,
            use_projection=True,
        )
        self.convres2 = ConvResidualBlock(
            filters=128,
            kernel_size=7,
            stride=2,
            reg_l2=reg_l2,
            dropout_rate=dropout_rate,
            use_projection=True,
        )
        self.convres3 = ConvResidualBlock(
            filters=128,
            kernel_size=5,
            stride=2,
            reg_l2=reg_l2,
            dropout_rate=dropout_rate,
            use_projection=True,
        )
        self.convres4 = ConvResidualBlock(
            filters=128,
            kernel_size=3,
            stride=2,
            dropout_rate=dropout_rate,
            reg_l2=reg_l2,
            use_projection=True,
        )
        self.convres5 = ConvResidualBlock(
            filters=128,
            kernel_size=3,
            stride=2,
            dropout_rate=dropout_rate,
            reg_l2=reg_l2,
            use_projection=True,
        )
        self.convres6 = ConvResidualBlock(
            filters=128,
            kernel_size=3,
            stride=1,
            dropout_rate=dropout_rate,
            reg_l2=reg_l2,
            use_projection=True,
        )

        # Multi-head attention
        self.multi_head_attention = MultiHeadAttentionBlock(
            128, num_heads=4, dropout_rate=dropout_rate, head_dropout_rate=0.25
        )

        # self.conv_attn_1 = tf.keras.layers.Conv2D(
        #     filters=32,  # Number of output feature maps
        #     kernel_size=(5, 5),  # Small receptive field
        #     strides=(3, 3),
        #     padding="valid",
        #     activation="relu",
        # )
        # self.conv_attn_2 = tf.keras.layers.Conv2D(
        #     filters=64,  # Further reduce channels
        #     kernel_size=(5, 5),
        #     strides=(3, 3),
        #     padding="valid",
        #     activation="relu",
        # )
        # self.pooled_attn = tf.keras.layers.GlobalAveragePooling2D()

        # Pooling
        self.global_avg_pooling_x_1 = tf.keras.layers.GlobalAveragePooling1D(
            data_format="channels_first"
        )
        self.global_avg_pooling_x_2 = tf.keras.layers.GlobalAveragePooling1D(
            # data_format="channels_first"
        )
        self.global_max_pooling_x_1 = tf.keras.layers.GlobalMaxPooling1D(
            data_format="channels_first"
        )
        self.global_max_pooling_x_2 = tf.keras.layers.GlobalMaxPooling1D(
            # data_format="channels_first"
        )

        # Dense layers for combined output
        act = "gelu"
        self.dense_1024 = tf.keras.layers.Dense(
            1024, activation=act, kernel_regularizer=tf.keras.regularizers.L2(reg_l2)
        )
        self.dropout_1024 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_512 = tf.keras.layers.Dense(
            512, activation=act, kernel_regularizer=tf.keras.regularizers.L2(reg_l2)
        )
        self.dropout_512 = tf.keras.layers.Dropout(dropout_rate)

        # self.dense_1024a = tf.keras.layers.Dense(
        #     1024, activation=act, kernel_regularizer=tf.keras.regularizers.L2(reg_l2)
        # )
        # # self.dropout_4 = tf.keras.layers.Dropout(dropout_rate)
        # self.dense_512a = tf.keras.layers.Dense(
        #     512, activation=act, kernel_regularizer=tf.keras.regularizers.L2(reg_l2)
        # )
        # self.dropout_5 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_256a = tf.keras.layers.Dense(
            256, activation=act, kernel_regularizer=tf.keras.regularizers.L2(reg_l2)
        )
        # self.dropout_6 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_64a = tf.keras.layers.Dense(
            64, activation=act, kernel_regularizer=tf.keras.regularizers.L2(reg_l2)
        )
        # self.dropout_7a = tf.keras.layers.Dropout(dropout_rate)
        self.dense_32a = tf.keras.layers.Dense(
            32, activation=act, kernel_regularizer=tf.keras.regularizers.L2(reg_l2)
        )
        # self.dropout_8a = tf.keras.layers.Dropout(dropout_rate)
        self.dense_16a = tf.keras.layers.Dense(
            16, activation="tanh", kernel_regularizer=tf.keras.regularizers.L2(reg_l2)
        )
        # self.dropout_9a = tf.keras.layers.Dropout(dropout_rate)
        self.dense_8a = tf.keras.layers.Dense(
            8, activation="tanh", kernel_regularizer=tf.keras.regularizers.L2(reg_l2)
        )

        # self.dense_1024b = tf.keras.layers.Dense(
        #     1024, activation=act, kernel_regularizer=tf.keras.regularizers.L2(reg_l2)
        # )
        # # self.dropout_4 = tf.keras.layers.Dropout(dropout_rate)
        # self.dense_512b = tf.keras.layers.Dense(
        #     512, activation=act, kernel_regularizer=tf.keras.regularizers.L2(reg_l2)
        # )
        # self.dropout_5 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_256b = tf.keras.layers.Dense(
            256, activation=act, kernel_regularizer=tf.keras.regularizers.L2(reg_l2)
        )
        # self.dropout_6 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_64b = tf.keras.layers.Dense(
            64, activation=act, kernel_regularizer=tf.keras.regularizers.L2(reg_l2)
        )
        # self.dropout_7b = tf.keras.layers.Dropout(dropout_rate)
        self.dense_32b = tf.keras.layers.Dense(
            32, activation=act, kernel_regularizer=tf.keras.regularizers.L2(reg_l2)
        )
        # self.dropout_8b = tf.keras.layers.Dropout(dropout_rate)
        self.dense_16b = tf.keras.layers.Dense(
            16, activation="tanh", kernel_regularizer=tf.keras.regularizers.L2(reg_l2)
        )
        # self.dropout_9b = tf.keras.layers.Dropout(dropout_rate)
        self.dense_8b = tf.keras.layers.Dense(
            8, activation="tanh", kernel_regularizer=tf.keras.regularizers.L2(reg_l2)
        )

        # Output layers

        self.weight_output = tf.keras.layers.Dense(
            n_outputs, activation="softmax", name="weight_output"
        )

        self.composition_output = tf.keras.layers.Dense(
            # 6, activation="relu", name="composition_output"
            n_outputs,
            activation="elu",
            name="composition_output",
        )

    def build(self, input_shape):
        """Build the model with the given input shape."""
        # You can print the input shape for debugging if needed
        print(f"Building model with input shape: {input_shape}")
        super().build(input_shape)  # Call the parent class's build() method

    def call(self, inputs, training=False):
        # First branch (conv + residual + attention)
        x = self.convres0(inputs, training=training)
        x = self.convres1(x, training=training)
        x = self.convres2(x, training=training)
        x = self.convres3(x, training=training)
        x = self.convres4(x, training=training)
        x = self.convres5(x, training=training)
        x = self.convres6(x, training=training)

        x, attn_weights = self.multi_head_attention(x, training=training)
        # xb, attn_weights_b = self.multi_head_attention2(x, training=training)

        #######################################################################
        attn_map = tf.reduce_mean(attn_weights, axis=1)   

        # Compute per-token importance, [B, T]
        token_importances = tf.reduce_mean(attn_map, axis=-1)  

        # Expand to [B, T, 1] for broadcasting with x ([B, T, features])
        token_importances = tf.expand_dims(token_importances, axis=-1)

        # Optionally, normalize token importances to sum to 1 over T
        token_importance_normalized = token_importances / (tf.reduce_sum(token_importances, axis=1, keepdims=True) + 1e-9)

        # Compute the attention-pooled vector by summing over the sequence dimension (axis=1)
        attention_pooled = tf.reduce_sum(x * token_importance_normalized, axis=1)
        #######################################################################

        x_sig_max = self.global_max_pooling_x_1(x)
        x_sig_avg = self.global_avg_pooling_x_1(x)
        x_fil_avg = self.global_max_pooling_x_2(x)
        x_fil_max = self.global_avg_pooling_x_2(x)

        x = tf.concat([x_sig_max, x_sig_avg, x_fil_max, x_fil_avg, attention_pooled], axis=1)
        # x = tf.concat([x_sig_max, x_sig_avg, x_fil_max, x_fil_avg], axis=1)
        # x = tf.concat([x_sig_max, x_sig_avg], axis=1)
        x = self.dense_1024(x)
        x = self.dropout_1024(x, training=training)
        x = self.dense_512(x)
        x = self.dropout_512(x, training=training)

        # Fully connected layers - Weights branch
        # xa = self.dense_1024a(x)
        # x = self.dropout_4(x, training=training)
        # xa = self.dense_512a(xa)
        # x = self.dropout_5(x, training=training)
        xa = self.dense_256a(x)
        # x = self.dropout_6(x, training=training)
        xa = self.dense_64a(xa)
        # xa = self.dropout_7a(x, training=training)
        xa = self.dense_32a(xa)
        # xa = self.dropout_8a(xa, training=training)
        xa = self.dense_16a(xa)
        # xa = self.dropout_9a(xa, training=training)
        # xa_f = self.dense_8a(tf.concat([xa, composition_output], axis=1))
        xa_f = self.dense_8a(xa)
        # Output layer for the weights
        weight_output = self.weight_output(xa_f)

        # Fully connected layers - Comps branch
        # xb = self.dense_1024b(x)
        # x = self.dropout_4(x, training=training)
        # xb = self.dense_512b(xb)
        # x = self.dropout_5(x, training=training)
        xb = self.dense_256b(x)
        # x = self.dropout_6(x, training=training)
        xb = self.dense_64b(xb)
        # xb = self.dropout_7b(x, training=training)
        xb = self.dense_32b(xb)
        # xb = self.dropout_8b(xb, training=training)
        xb = self.dense_16b(xb)
        # xb = self.dropout_9b(xb, training=training)
        xb_f = self.dense_8b(tf.concat([xb, weight_output], axis=1))
        # xb_f = self.dense_8b(xb)
        # Output layer for the compositions
        composition_output = self.composition_output(xb_f)

        # composition_output = tf.where(
        #     tf.less(weight_output, 5e-3), -1.0, self.composition_output(xb)
        # )
        # composition_output = tf.where(
        #     tf.less(composition_output, -1e-1), -1.0, composition_output
        # )

        return {
            "weight_output": weight_output,
            "composition_output": composition_output,
        }

    def get_config(self):
        config = super(CustomModel, self).get_config()
        config.update(
            {
                "reg_l2": self.reg_l2,
                "dropout_rate": self.dropout_rate,
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
