import tensorflow as tf
import numpy as np
from loss import hinge_d, hinge_g, l1_loss, feature_matching, divergence_loss_norm_tf, spectrum_loss_safe_tf
from utils import quarter_jumble_tf
import os
import pickle
import matplotlib.pyplot as plt


upsampler = tf.keras.layers.UpSampling3D(size=(2,2,2))

def quarter_jumble_tf(x):
    """Rearrange 3D tensor quarters for augmentation."""
    shape = tf.shape(x)
    H = shape[2]; W = shape[3]
    h2 = H//2; w2 = W//2
    q00 = x[:,:, :h2, :w2, :]
    q01 = x[:,:, :h2, w2:, :]
    q10 = x[:,:, h2:, :w2, :]
    q11 = x[:,:, h2:, w2:, :]
    top = tf.concat([q11,q10], axis=3)
    bot = tf.concat([q01,q00], axis=3)
    return tf.concat([top, bot], axis=2)



def build_pairs(arr):
    """Build consecutive frame pairs."""
    return np.stack([arr[:-1], arr[1:]], axis=1)

def make_dataset(lr_pairs, hr_pairs, batch, shuffle=True):
    """Create TensorFlow dataset from LR-HR pairs."""
    ds = tf.data.Dataset.from_tensor_slices((lr_pairs, hr_pairs))
    if shuffle:
        ds = ds.shuffle(buffer_size=max(128, lr_pairs.shape[0]))
    ds = ds.batch(batch, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# -------------------------
# Training step (tf.function)
# -------------------------
@tf.function
def train_step(lr_pair, hr_pair,
               G, Ds, Dt,
               g_opt, d_opt,
               lambda_l1, lambda_fm, lambda_adv_s, lambda_adv_t,
               lambda_div, lambda_spec,
               use_physics, use_jumble, jumble_prob, train_discriminator_flag, use_temporal):
    # lr_pair: (B,2,D,H,W,C)
    lr0 = lr_pair[:,0]
    lr1 = lr_pair[:,1]
    hr0 = hr_pair[:,0]
    hr1 = hr_pair[:,1]

    # upsample LR conditions
    lr0_up = upsampler(lr0)
    lr1_up = upsampler(lr1)

    # maybe jumble (use tf.random.uniform)
    cond0 = tf.where(tf.random.uniform([]) < jumble_prob, quarter_jumble_tf(lr0_up), lr0_up) if use_jumble else lr0_up
    cond1 = tf.where(tf.random.uniform([]) < jumble_prob, quarter_jumble_tf(lr1_up), lr1_up) if use_jumble else lr1_up

    # --------- Discriminator update ----------
    d_loss_s_val = tf.constant(0.0, dtype=tf.float32)
    d_loss_t_val = tf.constant(0.0, dtype=tf.float32)
    if train_discriminator_flag:
        with tf.GradientTape() as tape_d:
            sr0 = G(lr0_up, training=True)
            sr1 = G(lr1_up, training=True)

            d_real_logits, *d_real_feats = Ds(tf.concat([hr0, cond0], axis=-1), training=True)
            d_fake_logits, *d_fake_feats = Ds(tf.concat([sr0, cond0], axis=-1), training=True)
            d_loss_s = hinge_d(d_real_logits, d_fake_logits)

            d_loss_t = tf.constant(0.0, dtype=tf.float32)
            if use_temporal:
                dt_real_logits, *dt_real_feats = Dt(tf.concat([hr0, hr1, cond0, cond1], axis=-1), training=True)
                dt_fake_logits, *dt_fake_feats = Dt(tf.concat([sr0, sr1, cond0, cond1], axis=-1), training=True)
                d_loss_t = hinge_d(dt_real_logits, dt_fake_logits)

            d_loss_total = d_loss_s + d_loss_t

        d_vars = Ds.trainable_variables + (Dt.trainable_variables if use_temporal else [])
        d_grads = tape_d.gradient(d_loss_total, d_vars)
        d_opt.apply_gradients(zip(d_grads, d_vars))
        d_loss_s_val = tf.cast(d_loss_s, tf.float32)
        d_loss_t_val = tf.cast(d_loss_t, tf.float32)
    else:
        # evaluate without update
        sr0 = G(lr0_up, training=False)
        sr1 = G(lr1_up, training=False)
        d_real_logits, *_ = Ds(tf.concat([hr0, cond0], axis=-1), training=False)
        d_fake_logits, *_ = Ds(tf.concat([sr0, cond0], axis=-1), training=False)
        d_loss_s_val = tf.cast(hinge_d(d_real_logits, d_fake_logits), tf.float32)
        if use_temporal:
            d_t_real_logits, *_ = Dt(tf.concat([hr0, hr1, cond0, cond1], axis=-1), training=False)
            d_t_fake_logits, *_ = Dt(tf.concat([sr0, sr1, cond0, cond1], axis=-1), training=False)
            d_loss_t_val = tf.cast(hinge_d(d_t_real_logits, d_t_fake_logits), tf.float32)

    # --------- Generator update ----------
    with tf.GradientTape() as tape_g:
        sr0 = G(lr0_up, training=True)
        sr1 = G(lr1_up, training=True)

        g_adv_s = tf.constant(0.0, dtype=tf.float32)
        g_adv_t = tf.constant(0.0, dtype=tf.float32)
        g_fm_s = tf.constant(0.0, dtype=tf.float32)
        g_fm_t = tf.constant(0.0, dtype=tf.float32)
        g_div = tf.constant(0.0, dtype=tf.float32)
        g_spec = tf.constant(0.0, dtype=tf.float32)

        if lambda_adv_s > 0.0 or lambda_adv_t > 0.0:
            d_fake_logits, *d_fake_feats = Ds(tf.concat([sr0, cond0], axis=-1), training=False)
            d_real_logits, *d_real_feats = Ds(tf.concat([hr0, cond0], axis=-1), training=False)
            g_adv_s = tf.cast(lambda_adv_s, tf.float32) * tf.cast(hinge_g(d_fake_logits), tf.float32)
            g_fm_s = tf.cast(lambda_fm, tf.float32) * feature_matching(d_real_feats, d_fake_feats)

            if use_temporal:
                dt_fake_logits, *dt_fake_feats = Dt(tf.concat([sr0, sr1, cond0, cond1], axis=-1), training=False)
                dt_real_logits, *dt_real_feats = Dt(tf.concat([hr0, hr1, cond0, cond1], axis=-1), training=False)
                g_adv_t = tf.cast(lambda_adv_t, tf.float32) * tf.cast(hinge_g(dt_fake_logits), tf.float32)
                g_fm_t = tf.cast(lambda_fm, tf.float32) * feature_matching(dt_real_feats, dt_fake_feats)

        g_l1 = tf.cast(lambda_l1, tf.float32) * l1_loss(sr0, hr0)

        if use_physics:
            # sr0 might be float16 if mixed precision; cast to float32 for physics
            sr0_f32 = tf.cast(sr0, tf.float32)
            hr0_f32 = tf.cast(hr0, tf.float32)
            g_div = tf.cast(lambda_div, tf.float32) * divergence_loss_norm_tf(sr0_f32, dx=66.4/512.0)
            g_spec = tf.cast(lambda_spec, tf.float32) * spectrum_loss_safe_tf(sr0_f32, hr0_f32, dx=66.4/512.0)


        g_total = g_adv_s + g_adv_t + g_l1 + g_fm_s + g_fm_t + g_div + g_spec

    g_grads = tape_g.gradient(g_total, G.trainable_variables)
    g_opt.apply_gradients(zip(g_grads, G.trainable_variables))

    return {
        "d_loss_s": d_loss_s_val,
        "d_loss_t": d_loss_t_val,
        "g_total": tf.cast(g_total, tf.float32),
        "g_adv_s": tf.cast(g_adv_s, tf.float32),
        "g_adv_t": tf.cast(g_adv_t, tf.float32),
        "g_l1": tf.cast(g_l1, tf.float32),
        "g_fm_s": tf.cast(g_fm_s, tf.float32),
        "g_fm_t": tf.cast(g_fm_t, tf.float32),
        "g_div": tf.cast(g_div, tf.float32),
        "g_spec": tf.cast(g_spec, tf.float32),
    }


# for predict
# ============================ 1. Load normalization parameters ============================
def load_normalization_params(path):
    """Load global mean and std from pickle file."""
    with open(path, 'rb') as f:
        params = pickle.load(f)
    return params['global_mean'], params['global_std']

# ============================ 2. Load training/validation data ============================
def load_data(data_dir='./processed_data_seq'):
    """Load LR and HR data for train and validation sets."""
    train_lr = np.load(os.path.join(data_dir, 'train_lr.npy'))
    train_hr = np.load(os.path.join(data_dir, 'train_hr.npy'))
    val_lr = np.load(os.path.join(data_dir, 'val_lr.npy'))
    val_hr = np.load(os.path.join(data_dir, 'val_hr.npy'))
    print(f"Train LR:{train_lr.shape}, HR:{train_hr.shape}")
    print(f"Validation LR:{val_lr.shape}, HR:{val_hr.shape}")
    return train_lr, train_hr, val_lr, val_hr

# ============================ 3. Denormalize ============================
def denormalize(data, mean, std):
    """Reverse normalization."""
    return data * std + mean

# ============================ 4. Compute relative L2 errors ============================
def compute_relative_errors(pred, truth):
    """Compute per-frame relative L2 error."""
    errors = []
    for i in range(len(truth)):
        mse = np.mean((pred[i] - truth[i]) ** 2)
        norm = np.mean(truth[i] ** 2)
        errors.append(np.sqrt(mse / norm))
    return np.array(errors)

# ============================ 5. Energy spectrum ============================
def energy_spectrum_phys(u, dx):
    """Compute energy spectrum of 3D velocity field."""
    u = u - np.mean(u)
    uk = np.fft.fftn(u, axes=(0, 1, 2))
    E_k = np.abs(uk) ** 2

    nx, ny, nz = u.shape
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dx) * 2 * np.pi
    kz = np.fft.fftfreq(nz, d=dx) * 2 * np.pi
    k_mag = np.sqrt(kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2)

    k_bins = np.arange(0.5, np.max(k_mag) + 1, 1.0)
    E_shell = np.zeros_like(k_bins)
    counts = np.zeros_like(k_bins)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                k_val = k_mag[i, j, k]
                k_idx = int(np.floor(k_val))
                if k_idx < len(k_bins):
                    E_shell[k_idx] += E_k[i, j, k]
                    counts[k_idx] += 1

    E_shell = np.divide(E_shell, counts, out=np.zeros_like(E_shell), where=counts > 0)
    return k_bins, E_shell

# ============================ 6. Plotting ============================
def plot_and_save_curve(y, save_path, xlabel='Frame', ylabel='Relative L2 Error', title=''):
    """Plot and save curve."""
    plt.figure()
    plt.plot(y, marker='o', color='b')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved curve: {save_path}")

def plot_and_save_spectrum(k, E_true, E_pred, save_path, title='Energy Spectrum'):
    """Plot and save energy spectrum."""
    plt.figure(figsize=(6, 4))
    plt.loglog(k, E_true, label='True')
    plt.loglog(k, E_pred, '--', label='Pred')
    plt.xlabel('Wavenumber k')
    plt.ylabel('Energy Spectrum')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved spectrum: {save_path}")

def plot_and_save_slice(pred, truth, save_path, slice_idx=32, title='Prediction vs Truth Slice'):
    """Plot a single z-slice of prediction and ground truth."""
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(truth[:, :, slice_idx, 0], cmap='jet')
    plt.title('True HR')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 2, 2)
    plt.imshow(pred[:, :, slice_idx, 0], cmap='jet')
    plt.title('Predicted SR')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved slice image: {save_path}")
