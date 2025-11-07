import tensorflow as tf
import numpy as np

EPS = 1e-8

# -------------------------
# GAN Losses
# -------------------------
def hinge_d(real_logits, fake_logits):
    """Hinge loss for discriminator."""
    return tf.reduce_mean(tf.nn.relu(1.0 - real_logits)) + tf.reduce_mean(tf.nn.relu(1.0 + fake_logits))

def hinge_g(fake_logits):
    """Hinge loss for generator."""
    return -tf.reduce_mean(fake_logits)

def l1_loss(x, y):
    """L1 pixel-wise loss."""
    return tf.reduce_mean(tf.abs(tf.cast(x, tf.float32) - tf.cast(y, tf.float32)))

def feature_matching(real_feats, fake_feats):
    """Feature matching loss from discriminator features."""
    loss = 0.0
    for r, f in zip(real_feats, fake_feats):
        loss += tf.reduce_mean(tf.abs(tf.stop_gradient(r) - f))
    return loss

# -------------------------
# Physics-based losses
# -------------------------
@tf.function
def divergence_loss_norm_tf(u, dx):
    """Normalized divergence loss for 3D velocity field."""
    u_x, u_y, u_z = u[...,0], u[...,1], u[...,2]
    dudx = tf.pad(u_x[:,1:,:,:]-u_x[:,:-1,:,:], [[0,0],[0,1],[0,0],[0,0]]) / dx
    dvdy = tf.pad(u_y[:,:,1:,:]-u_y[:,:,:-1,:], [[0,0],[0,0],[0,1],[0,0]]) / dx
    dwdz = tf.pad(u_z[:,:,:,1:]-u_z[:,:,:,:-1], [[0,0],[0,0],[0,0],[0,1]]) / dx
    div = dudx + dvdy + dwdz
    div_l2 = tf.sqrt(tf.reduce_mean(div**2) + EPS)
    energy = tf.sqrt(tf.reduce_mean(u_x**2 + u_y**2 + u_z**2) + EPS)
    return div_l2 / (energy + EPS)

@tf.function
def spectrum_loss_safe_tf(sr, hr, dx=66.4/512.0):
    """High-wavenumber spectrum loss."""
    B = tf.shape(sr)[0]
    k_safe_val = 2.0/3.0*np.pi/ dx * tf.sqrt(3.0)

    def per_sample(i):
        # compute normalized 3D spectrum
        s_spec = tf.signal.fft3d(tf.cast(sr[i]-tf.reduce_mean(sr[i]), tf.complex64))
        h_spec = tf.signal.fft3d(tf.cast(hr[i]-tf.reduce_mean(hr[i]), tf.complex64))
        s_spec = tf.math.abs(s_spec)**2
        h_spec = tf.math.abs(h_spec)**2
        # safe range
        safe_idx = tf.constant(0, dtype=tf.int32) # simplified placeholder
        s_log = tf.math.log(s_spec[safe_idx:]+EPS)
        h_log = tf.math.log(h_spec[safe_idx:]+EPS)
        return tf.reduce_mean(tf.abs(s_log - h_log))

    losses = tf.map_fn(per_sample, tf.range(B), dtype=tf.float32)
    return tf.reduce_mean(losses)
