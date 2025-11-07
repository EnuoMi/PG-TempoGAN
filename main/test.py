from utils import *
from tempoGAN import build_generator

# ============================ Sequential prediction ============================
def predict_sequentially(model, lr_data):
    """Perform frame-by-frame inference using upsampling + generator."""
    preds = []
    upsampler = tf.keras.layers.UpSampling3D(size=(2, 2, 2))  # Same as training
    for i in range(len(lr_data)):
        lr_frame = tf.convert_to_tensor(lr_data[i][np.newaxis, ...], dtype=tf.float32)
        lr_up_tf = upsampler(lr_frame)
        sr = model(lr_up_tf, training=False)
        preds.append(sr.numpy()[0])
    return np.array(preds)




# ============================ Main testing flow ============================
def main():
    DATA_DIR = 'data/processed_data_seq'
    CKPT_DIR = os.path.join(DATA_DIR, 'checkpoints')
    OUTPUT_DIR = './processed_data_seq/results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data and normalization
    train_lr, train_hr, val_lr, val_hr = load_data(DATA_DIR)
    global_mean, global_std = load_normalization_params(os.path.join(DATA_DIR, 'normalization_params.pkl'))

    # 2. Build model
    G = build_generator()

    # 3. Load specific checkpoint
    epoch_to_load = 6
    ckpt_prefix = os.path.join(CKPT_DIR, f"ckpt-{epoch_to_load}")
    ckpt = tf.train.Checkpoint(generator=G)
    ckpt.restore(ckpt_prefix).expect_partial()
    print(f"✅ Loaded generator checkpoint: {ckpt_prefix}")

    # 4. Perform inference
    train_pred = predict_sequentially(G, train_lr)
    val_pred = predict_sequentially(G, val_lr)

    # 5. Denormalize
    train_pred_denorm = denormalize(train_pred, global_mean, global_std)
    val_pred_denorm = denormalize(val_pred, global_mean, global_std)
    train_hr_denorm = denormalize(train_hr, global_mean, global_std)
    val_hr_denorm = denormalize(val_hr, global_mean, global_std)

    # 6. Save reconstruction
    np.savez(os.path.join(OUTPUT_DIR, 'reconstruction_results.npz'),
             train_pred=train_pred_denorm,
             train_hr=train_hr_denorm,
             val_pred=val_pred_denorm,
             val_hr=val_hr_denorm)
    print(f"✅ Saved reconstruction results.")

    # 7. Compute errors and plot
    train_errors = compute_relative_errors(train_pred_denorm, train_hr_denorm)
    val_errors = compute_relative_errors(val_pred_denorm, val_hr_denorm)
    plot_and_save_curve(train_errors, os.path.join(OUTPUT_DIR, 'train_relative_error.png'),
                        title='Train Relative Error per Frame')
    plot_and_save_curve(val_errors, os.path.join(OUTPUT_DIR, 'val_relative_error.png'),
                        title='Validation Relative Error per Frame')
    print(f"Average train error: {np.mean(train_errors):.5f}")
    print(f"Average validation error: {np.mean(val_errors):.5f}")

    # 8. Energy spectrum example
    dx = 66.4 / 512
    k_train, E_train_pred = energy_spectrum_phys(train_pred_denorm[49, :, :, :, 0], dx)
    _, E_train_true = energy_spectrum_phys(train_hr_denorm[49, :, :, :, 0], dx)
    plot_and_save_spectrum(k_train, E_train_true, E_train_pred,
                           os.path.join(OUTPUT_DIR, 'train_energy_spectrum.png'),
                           title='Train Energy Spectrum (Frame 50)')

    k_val, E_val_pred = energy_spectrum_phys(val_pred_denorm[1, :, :, :, 0], dx)
    _, E_val_true = energy_spectrum_phys(val_hr_denorm[1, :, :, :, 0], dx)
    plot_and_save_spectrum(k_val, E_val_true, E_val_pred,
                           os.path.join(OUTPUT_DIR, 'val_energy_spectrum.png'),
                           title='Validation Energy Spectrum (Frame 2)')

    # 9. Slice visualization
    plot_and_save_slice(val_pred_denorm[1], val_hr_denorm[1],
                        os.path.join(OUTPUT_DIR, 'val_frame2_slice49.png'),
                        slice_idx=49,
                        title='Validation Frame 2 - Slice z=49')

if __name__ == "__main__":
    main()
