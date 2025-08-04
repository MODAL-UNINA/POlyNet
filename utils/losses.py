#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:47:46 2024

@author: MODAL
"""

# %%
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom", name="neg_mse")
def neg_mse(y_true, y_pred):
    is_absent = tf.equal(y_true, -1.)
    is_present = tf.logical_not(is_absent)

    present_loss = tf.square(y_true - y_pred)
    absent_loss = tf.nn.relu(y_pred+0.05)

    loss_per_entry = tf.where(is_present, present_loss, absent_loss)
    return tf.reduce_mean(loss_per_entry)

@tf.keras.utils.register_keras_serializable(package="Custom", name="mae_mse")
def mae_mse(y_true, y_pred):
    mae_loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    return .5*mae_loss + .5*mse_loss

@tf.keras.utils.register_keras_serializable(package="Custom", name="kl_mae_loss")
def kl_mae_loss(y_true, y_pred):
    epsilon = 1e-7  # Small constant to avoid division by zero and log(0)
    # Clip y_pred to prevent log(0) and division by zero errors
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
    kld_loss = tf.keras.losses.KLDivergence()(y_true, y_pred)
    mae_loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return .5*kld_loss + .5*mae_loss

@tf.keras.utils.register_keras_serializable(package="Custom", name="kl_mse_loss")
def kl_mse_loss(y_true, y_pred):
    epsilon = 1e-7  # Small constant to avoid division by zero and log(0)
    # Clip y_pred to prevent log(0) and division by zero errors
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
    kld_loss = tf.keras.losses.KLDivergence()(y_true, y_pred)
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    return .5*kld_loss + .5*mse_loss

@tf.keras.utils.register_keras_serializable(package="Custom", name="ce_mse_loss")
def ce_mse_loss(y_true, y_pred):
    epsilon = 1e-7  # Small constant to avoid division by zero and log(0)
    # Clip y_pred to prevent log(0) and division by zero errors
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
    ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true, y_pred)
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    return .5*ce_loss + .5*mse_loss

@tf.keras.utils.register_keras_serializable(package="Custom", name="ce_mae_loss")
def ce_mae_loss(y_true, y_pred):
    epsilon = 1e-7  # Small constant to avoid division by zero and log(0)
    # Clip y_pred to prevent log(0) and division by zero errors
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
    ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true, y_pred)
    mse_loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return .5*ce_loss + .5*mse_loss

@tf.keras.utils.register_keras_serializable(package="Custom", name="ce_mae_pun_loss")
def ce_mae_pun_loss(y_true, y_pred):
    epsilon = 1e-7  # Small constant to avoid division by zero and log(0)
    # Clip y_pred to prevent log(0) and division by zero errors
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
    ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true, y_pred)

    # Mask where y_true >= 0 (copolymers that are present)
    y_true_masked = tf.where(y_true > 1e-3, y_true, -1)
    y_pred_masked = tf.where(y_pred > 1e-3, y_pred, -1)
    mae_loss = tf.keras.losses.MeanAbsoluteError()(y_true_masked, y_pred_masked)

    return .5*ce_loss + .5*mae_loss

@tf.keras.utils.register_keras_serializable(package="Custom", name="ce_mse_pun_loss")
def ce_mse_pun_loss(y_true, y_pred):
    epsilon = 1e-7  # Small constant to avoid division by zero and log(0)
    # Clip y_pred to prevent log(0) and division by zero errors
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
    ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true, y_pred)

    # Mask where y_true >= 0 (copolymers that are present)
    y_true_masked = tf.where(y_true > 1e-3, y_true, -1)
    y_pred_masked = tf.where(y_pred > 1e-3, y_pred, -1)
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true_masked, y_pred_masked)

    return .5*ce_loss + .5*mse_loss

@tf.keras.utils.register_keras_serializable(package="Custom", name="ce_huber_pun_loss")
def ce_huber_pun_loss(y_true, y_pred):
    epsilon = 1e-7  # Small constant to avoid division by zero and log(0)
    # Clip y_pred to prevent log(0) and division by zero errors
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
    ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true, y_pred)

    # Mask where y_true >= 0 (copolymers that are present)
    y_true_masked = tf.where(y_true > 1e-3, y_true, -1)
    y_pred_masked = tf.where(y_pred > 1e-3, y_pred, -1)

    # Replace MSE with Huber Loss
    huber_loss = tf.keras.losses.Huber(delta=.5)(y_true_masked, y_pred_masked)

    return 0.5 * ce_loss + 0.5 * huber_loss

@tf.keras.utils.register_keras_serializable(package="Custom", name="masked_mse_loss")
def masked_mse_loss(y_true, y_pred):
    # Mask where y_true >= 0 (copolymers that are present)
    mask = tf.cast(y_true >= 0, dtype=tf.float32)
    # Compute squared differences
    sq_diff = tf.square(y_true - y_pred)
    # Apply mask
    masked_sq_diff = sq_diff * mask

    # Calculate mean squared error over valid entries
    sum_sq_diff = tf.reduce_sum(masked_sq_diff, axis=1)
    # sum_sq_diff = tf.reduce_sum(sq_diff, axis=1)

    valid_counts = tf.reduce_sum(mask, axis=1)
    valid_counts = tf.maximum(valid_counts, 1)  # Avoid division by zero
    mse = sum_sq_diff / valid_counts
    # Average over the batch
    loss = tf.reduce_mean(mse)
    return loss

@tf.keras.utils.register_keras_serializable(package="Custom", name="masked_mae_loss")
def masked_mae_loss(y_true, y_pred):
    # Create a mask where y_true >= 0 (valid entries)
    mask = tf.cast(y_true >= 0, dtype=tf.float32)
    # Compute absolute differences
    abs_diff = tf.abs(y_true - y_pred)

    # Apply the mask to the absolute differences
    masked_abs_diff = abs_diff * mask

    # Sum the absolute differences over the last axis (features)
    sum_abs_diff = tf.reduce_sum(masked_abs_diff, axis=1)
    # sum_abs_diff = tf.reduce_sum(abs_diff, axis=1)

    # Count the number of valid entries per sample
    valid_counts = tf.reduce_sum(mask, axis=1)
    # Avoid division by zero
    valid_counts = tf.maximum(valid_counts, 1)
    # Compute mean absolute error for each sample
    mae = sum_abs_diff / valid_counts
    # Compute the average loss over the batch
    loss = tf.reduce_mean(mae)
    return loss

@tf.keras.utils.register_keras_serializable(package="Custom", name="mse_masked_mse_loss")
def mse_masked_mse_loss(y_true, y_pred):
    # Mask where y_true >= 0 (copolymers that are present)
    mask = tf.cast(y_true >= 0, dtype=tf.float32)
    # Compute squared differences
    sq_diff = tf.square(y_true - y_pred)
    # Apply mask
    masked_sq_diff = sq_diff * mask

    # Calculate mean squared error over valid entries
    sum_sq_diff = tf.reduce_sum(masked_sq_diff, axis=1)
    # sum_sq_diff = tf.reduce_sum(sq_diff, axis=1)

    valid_counts = tf.reduce_sum(mask, axis=1)
    valid_counts = tf.maximum(valid_counts, 1)  # Avoid division by zero
    mse = sum_sq_diff / valid_counts
    # Average over the batch
    loss = tf.reduce_mean(mse)
    return loss + tf.keras.losses.MeanSquaredError()(y_true, y_pred)

@tf.keras.utils.register_keras_serializable(package="Custom", name="huber_loss")
def huber_loss(y_true, y_pred):
    return tf.keras.losses.Huber(delta=0.05)(y_true, y_pred)

@tf.keras.utils.register_keras_serializable(package="Custom", name="mqe")
def mqe(y_true, y_pred):
    error = y_true - y_pred
    return tf.reduce_mean(tf.pow(error, 4))

@tf.keras.utils.register_keras_serializable(package="Custom", name="huber_masked_huber_loss")
def huber_masked_huber_loss(y_true, y_pred, delta=.5):
    # Mask where y_true >= 0 (copolymers that are present)
    mask = tf.cast(y_true >= 0, dtype=tf.float32)

    # Compute absolute differences (needed for Huber Loss calculation)
    abs_diff = tf.abs(y_true - y_pred)
    # Compute Huber Loss (element-wise)
    huber_elementwise = tf.where(
        abs_diff <= delta, 
        0.5 * tf.square(abs_diff),  # Quadratic for small errors
        delta * (abs_diff - 0.5 * delta)  # Linear for large errors
    )
    # Apply mask
    masked_huber = huber_elementwise * mask

    # Calculate the mean Huber loss over valid entries
    sum_huber = tf.reduce_sum(masked_huber, axis=1)
    valid_counts = tf.reduce_sum(mask, axis=1)
    valid_counts = tf.maximum(valid_counts, 1)  # Avoid division by zero
    huber_loss = sum_huber / valid_counts

    # Average over the batch
    loss = tf.reduce_mean(huber_loss)

    # Optionally add a secondary loss component
    return loss + tf.keras.losses.Huber(delta=delta)(y_true, y_pred)

@tf.keras.utils.register_keras_serializable(package="Custom", name="focal_mse_loss")
def focal_mse_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    epsilon = 1e-7  # Small constant to avoid division by zero and log(0)
    # Clip y_pred to prevent log(0) and division by zero errors
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
    
    # Compute focal loss
    cross_entropy = -y_true * tf.math.log(y_pred)  # Element-wise cross-entropy
    weights = alpha * tf.pow(1 - y_pred, gamma)  # Modulating factor
    focal_loss = tf.reduce_sum(weights * cross_entropy, axis=-1)  # Sum over classes
    
    # Compute MSE loss
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    
    # Combine the losses
    combined_loss = 0.5 * focal_loss + 0.5 * mse_loss
    return combined_loss

@tf.keras.utils.register_keras_serializable(package="Custom", name="focal_loss")
def focal_loss(y_true, y_pred, gamma=1.0, alpha=1.0):
    y_pred = tf.nn.softmax(y_pred, axis=-1)

    # Convert y_true to one-hot encoding if not already
    if len(y_true.shape) == 1 or y_true.shape[-1] != y_pred.shape[-1]:
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
    else:
        y_true_one_hot = y_true

    # Probabilità associate alla classe corretta
    p_t = tf.reduce_sum(y_pred * y_true_one_hot, axis=-1)  # Prob. for the true class

    # Focal Loss (element-wise)
    focal_elementwise = -alpha * tf.pow(1 - p_t, gamma) * tf.math.log(tf.maximum(p_t, 1e-7))

    # Media della focal loss nella batch
    loss = tf.reduce_mean(focal_elementwise)

    return loss
