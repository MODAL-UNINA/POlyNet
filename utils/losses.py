#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:47:46 2024

@author: MODAL
"""

# %%
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom", name="neg_mse_autobalanced")
def neg_mse_autobalanced(y_true, y_pred):
    epsilon = 1e-8 # Per evitare divisioni per zero

    # 1. Identifica le maschere per componenti presenti e assenti
    is_absent = tf.equal(y_true, -1.0)
    is_present = tf.logical_not(is_absent)

    # 2. Calcola dinamicamente il numero di presenti e assenti nel batch
    num_present = tf.reduce_sum(tf.cast(is_present, tf.float32))
    num_absent = tf.reduce_sum(tf.cast(is_absent, tf.float32))
    num_total = num_present + num_absent

    # 3. Calcola i pesi inversamente proporzionali alla frequenza
    # La formula standard è (num_total / num_classi) / num_campioni_per_classe
    # Qui usiamo una versione semplificata: num_total / (2 * num_campioni_per_classe)
    weight_for_present = num_total / (2.0 * num_present + epsilon)
    weight_for_absent = num_total / (2.0 * num_absent + epsilon)

    # 4. Calcola le loss per le due classi (come prima)
    present_loss = tf.square(y_true - y_pred)
    absent_loss = tf.nn.relu(y_pred + 0.05)
    
    # 5. Applica i pesi e combina le loss
    # Moltiplica ogni errore per il peso della sua classe
    weighted_loss_per_entry = tf.where(
        is_present,
        weight_for_present * present_loss,
        weight_for_absent * absent_loss
    )
    
    # 6. Restituisci la media della loss pesata
    return tf.reduce_mean(weighted_loss_per_entry)

@tf.keras.utils.register_keras_serializable(package="Custom", name="neg_mse")
def neg_mse(y_true, y_pred):
    is_absent = tf.equal(y_true, -1.)
    is_present = tf.logical_not(is_absent)

    present_loss = tf.square(y_true - y_pred)
    absent_loss = tf.nn.relu(y_pred+0.05)

    loss_per_entry = tf.where(is_present, present_loss, absent_loss)
    # return tf.reduce_mean(tf.reduce_sum(loss_per_entry, axis=1))
    return tf.reduce_mean(loss_per_entry)

@tf.keras.utils.register_keras_serializable(package="Custom", name="neg2_mse_hybrid")
def neg2_mse_hybrid(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Canali speciali: LDPE, PE, PP
    y_true_mono = y_true[:, :3]
    y_pred_mono = y_pred[:, :3]

    mono_absent = tf.equal(y_true_mono, -1.0)
    mono_present = tf.logical_not(mono_absent)

    # Se presenti, target fissato a 1
    mono_present_loss = tf.square(1.0 - y_pred_mono)

    # Se assenti, devono restare negativi
    mono_absent_loss = tf.square(tf.nn.relu(y_pred_mono + 0.05))

    mono_loss = tf.where(mono_present, mono_present_loss, mono_absent_loss)

    # Canali dei veri copolimeri: EH, EO, EB, RACO, EPR
    y_true_copo = y_true[:, 3:]
    y_pred_copo = y_pred[:, 3:]

    copo_absent = tf.equal(y_true_copo, -1.0)
    copo_present = tf.logical_not(copo_absent)

    copo_present_loss = tf.square(y_true_copo - y_pred_copo)
    copo_absent_loss = tf.square(tf.nn.relu(y_pred_copo + 0.05))

    copo_loss = tf.where(copo_present, copo_present_loss, copo_absent_loss)

    loss_per_entry = tf.concat([mono_loss, copo_loss], axis=1)
    return tf.reduce_mean(loss_per_entry)

@tf.keras.utils.register_keras_serializable(package="Custom", name="neg2_mse")
def neg2_mse(y_true, y_pred):
    is_absent = tf.equal(y_true, -1.)
    is_present = tf.logical_not(is_absent)

    present_loss = tf.square(y_true - y_pred)
    absent_loss = tf.square(tf.nn.relu(y_pred + 0.05))

    loss_per_entry = tf.where(is_present, present_loss, absent_loss)
    # return tf.reduce_mean(tf.reduce_sum(loss_per_entry, axis=1))
    return tf.reduce_mean(loss_per_entry)

@tf.keras.utils.register_keras_serializable(package="Custom", name="zero_aware_mse")
def zero_aware_mse(y_true, y_pred):
    is_present = y_true > 1e-3
    is_absent = tf.logical_not(is_present)

    present_loss = tf.square(y_true - y_pred)
    absent_loss = tf.square(y_pred)

    loss_per_entry = tf.where(is_present, present_loss, absent_loss)
    return tf.reduce_mean(loss_per_entry)

@tf.keras.utils.register_keras_serializable(package="Custom", name="neg_mse2")
def neg_mse2(y_true, y_pred):
    is_absent = tf.equal(y_true, -1.)
    is_present = tf.logical_not(is_absent)

    present_loss = tf.square(y_true - y_pred)
    absent_loss = tf.nn.relu(y_pred+0.05)

    loss_per_entry = tf.where(is_present, present_loss, absent_loss)
    return tf.reduce_mean(tf.reduce_sum(loss_per_entry, axis=1))
    # return tf.reduce_mean(loss_per_entry)


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

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    bce = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, 
        reduction=tf.keras.losses.Reduction.NONE
    )(y_true, y_pred)
    
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_weight = (1 - p_t) ** gamma

    alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    focal_loss_value = alpha_factor * focal_weight * bce
    
    return tf.reduce_mean(focal_loss_value)

@tf.keras.utils.register_keras_serializable(package="Custom", name="kl_mse_fl_loss")
def kl_mse_fl_loss(y_true, y_pred):

    epsilon = 1e-7  

    y_pred_clipped_kld = tf.clip_by_value(y_pred, epsilon, 1.0)
    kld_loss = tf.keras.losses.KLDivergence()(y_true, y_pred_clipped_kld)
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    
    mask_y_true = tf.cast(y_true > 1e-3, tf.float32)
    detection_loss = focal_loss(mask_y_true, y_pred)

    return 0.4 * kld_loss + 0.4 * mse_loss + 0.2 * detection_loss

@tf.keras.utils.register_keras_serializable(package="Custom", name="kl_mse_loss")
def kl_mse_loss(y_true, y_pred):
    epsilon = 1e-7  # Small constant to avoid division by zero and log(0)
    # Clip y_pred to prevent log(0) and division by zero errors
    y_pred_clipped_kld = tf.clip_by_value(y_pred, epsilon, 1.0)
    kld_loss = tf.keras.losses.KLDivergence()(y_true, y_pred_clipped_kld)
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
