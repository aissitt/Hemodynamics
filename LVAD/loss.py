import tensorflow as tf

# Data-Driven Loss: Component-Wise Huber Loss
def component_wise_huber_loss(y_true, y_pred, delta):
    # Computes Huber loss for each velocity component (u, v, w)
    huber_u = tf.keras.losses.Huber(delta, reduction=tf.keras.losses.Reduction.SUM)(y_true[..., 0], y_pred[..., 0])
    huber_v = tf.keras.losses.Huber(delta, reduction=tf.keras.losses.Reduction.SUM)(y_true[..., 1], y_pred[..., 1])
    huber_w = tf.keras.losses.Huber(delta, reduction=tf.keras.losses.Reduction.SUM)(y_true[..., 2], y_pred[..., 2])
    
    # Averages the Huber loss over the three velocity components
    return (huber_u + huber_v + huber_w) / 3.0

def data_driven_loss(y_true, y_pred, config):
    # Extract delta for Huber loss from the config
    delta = config['loss_parameters']['data_driven']['huber_delta']
    
    # Compute and return the component-wise Huber loss
    return component_wise_huber_loss(y_true, y_pred, delta)

# Physics-Informed Loss Functions
def compute_vorticity(predictedY):
    # Compute the vorticity (curl of velocity field) of the predicted velocities
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(predictedY)
        u, v, w = predictedY[..., 0], predictedY[..., 1], predictedY[..., 2]

    # Compute the spatial derivatives of the velocity components
    du_dy = tape.gradient(u, predictedY)[..., 1]
    du_dz = tape.gradient(u, predictedY)[..., 2]
    dv_dx = tape.gradient(v, predictedY)[..., 0]
    dv_dz = tape.gradient(v, predictedY)[..., 2]
    dw_dx = tape.gradient(w, predictedY)[..., 0]
    dw_dy = tape.gradient(w, predictedY)[..., 1]

    # Compute the components of the vorticity vector
    omega_x = dw_dy - dv_dz
    omega_y = du_dz - dw_dx
    omega_z = dv_dx - du_dy

    # Return the vorticity as a stack of its components
    return tf.stack([omega_x, omega_y, omega_z], axis=-1)

def compute_vorticity_magnitude(vorticity):
    # Compute the magnitude of the vorticity vector
    return tf.sqrt(tf.reduce_sum(tf.square(vorticity), axis=-1) + 1e-6)

def focused_vorticity_loss(y_true, y_pred, threshold):
    # Compute the vorticity loss focusing on high vorticity regions
    vorticity_true = compute_vorticity(y_true)
    vorticity_pred = compute_vorticity(y_pred)
    
    # Compute the magnitude of the vorticity
    vorticity_magnitude_true = compute_vorticity_magnitude(vorticity_true)
    high_vorticity_mask = vorticity_magnitude_true > threshold
    high_vorticity_mask = tf.expand_dims(high_vorticity_mask, axis=-1)

    # Compute squared error for vorticity
    vorticity_error = tf.square(vorticity_true - vorticity_pred)
    focused_error = tf.where(high_vorticity_mask, vorticity_error, tf.zeros_like(vorticity_error))
    high_vorticity_area = tf.reduce_sum(tf.cast(high_vorticity_mask, tf.float32))
    total_focused_error = tf.reduce_sum(focused_error)

    # Return the average focused error in high vorticity regions
    loss = tf.where(high_vorticity_area > 0, total_focused_error / high_vorticity_area, 0.0)
    return loss

def compute_continuity_loss(predictedY):
    # Compute continuity loss to enforce mass conservation
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(predictedY)
        u, v, w = predictedY[..., 0], predictedY[..., 1], predictedY[..., 2]
        du_dx = tape.gradient(u, predictedY)[..., 0]
        dv_dy = tape.gradient(v, predictedY)[..., 1]
        dw_dz = tape.gradient(w, predictedY)[..., 2]
    
    # Compute continuity residual (should be zero for incompressible flow)
    continuity_residual = du_dx + dv_dy + dw_dz
    continuity_loss = tf.reduce_mean(tf.square(continuity_residual))
    return continuity_loss
    
def compute_momentum_loss(predictedY, nu):
    # Compute momentum loss to enforce Navier-Stokes equations (incomplete)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(predictedY)
        u, v, w = predictedY[..., 0], predictedY[..., 1], predictedY[..., 2]

        grad_u = tape.gradient(u, predictedY)
        grad_v = tape.gradient(v, predictedY)
        grad_w = tape.gradient(w, predictedY)

        du_dx, du_dy, du_dz = grad_u[..., 0], grad_u[..., 1], grad_u[..., 2]
        dv_dx, dv_dy, dv_dz = grad_v[..., 0], grad_v[..., 1], grad_v[..., 2]
        dw_dx, dw_dy, dw_dz = grad_w[..., 0], grad_w[..., 1], grad_w[..., 2]

        # Compute convective and diffusive terms
        conv_u = u * du_dx + v * du_dy + w * du_dz
        conv_v = u * dv_dx + v * dv_dy + w * dv_dz
        conv_w = u * dw_dx + v * dw_dy + w * dw_dz
        convective_acceleration = tf.stack([conv_u, conv_v, conv_w], axis=-1)

        lap_u = du_dx + du_dy + du_dz
        lap_v = dv_dx + dv_dy + dv_dz
        lap_w = dw_dx + dw_dy + dw_dz
        diffusion_term = tf.stack([lap_u, lap_v, lap_w], axis=-1)

        momentum_residual = convective_acceleration - nu * diffusion_term

    # Return mean squared momentum residual
    momentum_loss = tf.reduce_mean(tf.square(momentum_residual))
    return momentum_loss

def physics_informed_loss(y_true, y_pred, config):
    # Extract parameters for physics-informed loss from config
    hp = config['loss_parameters']['physics_informed']
    
    # Compute individual loss components
    data_loss = component_wise_huber_loss(y_true, y_pred, delta=hp['huber_delta'])
    continuity_loss = compute_continuity_loss(y_pred)
    vorticity_loss = focused_vorticity_loss(y_true, y_pred, threshold=hp['threshold_vorticity'])
    momentum_loss = compute_momentum_loss(y_pred, hp['nu'])
    
    # Combine the loss components with their respective weights
    total_loss = (hp['lambda_data'] * data_loss +
                  hp['lambda_continuity'] * continuity_loss +
                  hp['lambda_vorticity_focused'] * vorticity_loss +
                  hp['lambda_momentum'] * momentum_loss)
    return total_loss
