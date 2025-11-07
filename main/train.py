import os
import numpy as np
import tensorflow as tf
from tempoGAN import build_generator, build_Ds, build_Dt
from utils import make_dataset, build_pairs
from utils import upsampler
from utils import train_step

# -------------------------
# User configuration
# -------------------------
DATA_DIR = "data/processed_data_seq"
CKPT_DIR = "checkpoints"
EPOCHS = 900
BATCH = 4
MIXED_PRECISION = True

# -------------------------
# GPU + Mixed precision
# -------------------------
if MIXED_PRECISION:
    try:
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled")
    except Exception as e:
        print("Mixed precision unavailable:", e)
        MIXED_PRECISION=False

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# -------------------------
# Load data
# -------------------------
train_lr = np.load(os.path.join(DATA_DIR, "train_lr.npy"))
train_hr = np.load(os.path.join(DATA_DIR, "train_hr.npy"))
val_lr = np.load(os.path.join(DATA_DIR, "val_lr.npy"))
val_hr = np.load(os.path.join(DATA_DIR, "val_hr.npy"))

train_lr_pairs = build_pairs(train_lr)
train_hr_pairs = build_pairs(train_hr)
val_lr_pairs = build_pairs(val_lr)
val_hr_pairs = build_pairs(val_hr)

train_ds = make_dataset(train_lr_pairs, train_hr_pairs, BATCH, shuffle=True)
val_ds = make_dataset(val_lr_pairs, val_hr_pairs, BATCH, shuffle=False)

# -------------------------
def main():
    G = build_generator()
    Ds = build_Ds()
    Dt = build_Dt()

    # Optimizers (if mixed precision, wrap with LossScaleOptimizer)
    g_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)
    d_opt = tf.keras.optimizers.Adam(5e-5, beta_1=0.0, beta_2=0.9)
    if MIXED_PRECISION:
        try:
            g_opt = tf.keras.mixed_precision.LossScaleOptimizer(g_opt, loss_scale='dynamic')
            d_opt = tf.keras.mixed_precision.LossScaleOptimizer(d_opt, loss_scale='dynamic')
        except Exception:
            pass

    # Warm-up forward pass to build variables and avoid UnknownVariable errors on restore
    dummy_lr = tf.zeros((1, 32, 32, 32, 3), dtype=tf.float32)
    _ = G(dummy_lr)
    dummy_hr = tf.zeros((1, 64, 64, 64, 3), dtype=tf.float32)
    _ = Ds(tf.concat([dummy_hr, upsampler(dummy_lr)], axis=-1))
    _ = Dt(tf.concat([dummy_hr, dummy_hr, upsampler(dummy_lr), upsampler(dummy_lr)], axis=-1))

    # hyperparams (same as your original logic)
    lambda_l1_stage1 = 20.0
    lambda_fm_stage1 = 3.0
    lambda_adv_s_max_stage1 = 0.2
    lambda_adv_t_max_stage1 = 0.15

    lambda_l1_stage2 = 2.0
    lambda_fm_stage2 = 1.0
    lambda_adv_s_max_stage2 = 0.1
    lambda_adv_t_max_stage2 = 0.05
    lambda_div_max = 0.2
    lambda_spec_max = 1.0

    STAGE1_EPOCHS = 400
    PHYSICS_WARMUP = 50
    adv_warmup_epochs = 50

    freeze_threshold = 0.15
    freeze_window = 5
    freeze_epochs_remaining = 0
    epoch_d_losses_s = []

    use_jumble = True
    jumble_prob = 0.15

    CKPT_DIR = os.path.join(DATA_DIR, "CKPT_DIR")
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt = tf.train.Checkpoint(generator=G, discriminator_s=Ds, discriminator_t=Dt,
                               g_optimizer=g_opt, d_optimizer=d_opt)
    manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=3)

    # resume
    best_val = np.inf
    if manager.latest_checkpoint:
          ckpt.restore(manager.latest_checkpoint).expect_partial()
          print("✅ Restored from:", manager.latest_checkpoint)
          start_epoch = int(manager.latest_checkpoint.split('-')[-1]) * 100
          start_epoch += 1
    else:
          print("🚀 No checkpoint found. Starting from epoch 1.")
          start_epoch = 1


    # training loop
    for epoch in range(start_epoch, EPOCHS+1):
        # compute stage-wise lambdas
        if epoch <= STAGE1_EPOCHS:
            lambda_l1 = lambda_l1_stage1
            lambda_fm = lambda_fm_stage1
            lambda_adv_s_max = lambda_adv_s_max_stage1
            lambda_adv_t_max = lambda_adv_t_max_stage1
            lambda_div = 0.0
            lambda_spec = 0.0
            use_physics = False
            stage_name = "Stage1-tempoGAN"
        else:
            stage_name = "Stage2-Physics"
            physics_progress = min((epoch - STAGE1_EPOCHS) / PHYSICS_WARMUP, 1.0)
            lambda_l1 = lambda_l1_stage1 + (lambda_l1_stage2 - lambda_l1_stage1) * physics_progress
            lambda_fm = lambda_fm_stage1 + (lambda_fm_stage2 - lambda_fm_stage1) * physics_progress
            lambda_adv_s_max = lambda_adv_s_max_stage1 + (lambda_adv_s_max_stage2 - lambda_adv_s_max_stage1) * physics_progress
            lambda_adv_t_max = lambda_adv_t_max_stage1 + (lambda_adv_t_max_stage2 - lambda_adv_t_max_stage1) * physics_progress
            lambda_div = lambda_div_max * physics_progress
            lambda_spec = lambda_spec_max * physics_progress
            use_physics = True

        scale = min(epoch / adv_warmup_epochs, 1.0)
        lambda_adv_s = lambda_adv_s_max * scale
        lambda_adv_t = lambda_adv_t_max * scale
        use_temporal = True
        use_adv = True

        # freeze logic
        if freeze_epochs_remaining > 0:
            freeze_epochs_remaining -= 1
            train_discriminator_flag = False
        else:
            train_discriminator_flag = True

        # metrics accumulators
        d_s_vals = []
        d_t_vals = []
        g_vals = []
        g_adv_s_vals = []
        g_adv_t_vals = []
        g_l1_vals = []
        g_fm_s_vals = []
        g_fm_t_vals = []
        g_div_vals = []
        g_spec_vals = []

        # iterate dataset
        for batch_lr, batch_hr in train_ds:
            stats = train_step(batch_lr, batch_hr,
                               G, Ds, Dt, g_opt, d_opt,
                               lambda_l1, lambda_fm, lambda_adv_s, lambda_adv_t,
                               lambda_div, lambda_spec,
                               use_physics, use_jumble, jumble_prob,
                               train_discriminator_flag, use_temporal)

            d_s_vals.append(stats["d_loss_s"].numpy())
            d_t_vals.append(stats["d_loss_t"].numpy())
            g_vals.append(stats["g_total"].numpy())
            g_adv_s_vals.append(stats["g_adv_s"].numpy())
            g_adv_t_vals.append(stats["g_adv_t"].numpy())
            g_l1_vals.append(stats["g_l1"].numpy())
            g_fm_s_vals.append(stats["g_fm_s"].numpy())
            g_fm_t_vals.append(stats["g_fm_t"].numpy())
            g_div_vals.append(float(stats["g_div"].numpy()))
            g_spec_vals.append(float(stats["g_spec"].numpy()))

        # epoch-level freeze check
        epoch_avg_d_loss_s = float(np.mean(d_s_vals)) if d_s_vals else 0.0
        epoch_d_losses_s.append(epoch_avg_d_loss_s)
        if train_discriminator_flag and len(epoch_d_losses_s) >= freeze_window:
            recent_avg = float(np.mean(epoch_d_losses_s[-freeze_window:]))
            if recent_avg < freeze_threshold:
                freeze_epochs_remaining = freeze_window
                print(f"⚠️ Freezing discriminators for {freeze_window} epochs (epoch {epoch}), recent avg d_loss_s={recent_avg:.4f}")

        # logging (compact)
        print(f"{stage_name} | Epoch {epoch}/{EPOCHS} | d_loss_s={epoch_avg_d_loss_s:.4f}, d_loss_t={np.mean(d_t_vals):.4f}, g_total={np.mean(g_vals):.4f}")
        print(f"  g: adv_s={np.mean(g_adv_s_vals):.4f}, adv_t={np.mean(g_adv_t_vals):.4f}, l1={np.mean(g_l1_vals):.4f}, fm_s={np.mean(g_fm_s_vals):.4f}, fm_t={np.mean(g_fm_t_vals):.4f}"
              + (f", div={np.mean(g_div_vals):.4f}, spec={np.mean(g_spec_vals):.4f}" if use_physics else ""))

        # checkpointing
        if epoch % 100 == 0:
            manager.save()
            print("✅ Saved checkpoint at epoch", epoch)

        # best generator saving on train metric
        mean_g = np.mean(g_vals) if g_vals else np.inf
        if mean_g < best_val:
            best_val = mean_g
            G.save_weights(os.path.join(CKPT_DIR, "G_best.weights.h5"))
            print(f"🌟 New best generator saved at epoch {epoch} (g_total={best_val:.4f})")

if __name__ == "__main__":
    main()
