import os
import h5py
import numpy as np
import pickle

# ============================ 1. Load H5 data ============================
def load_h5_data(hr_file_path, lr_file_path, num_frames=140):
    """
    Load HR and LR velocity fields from HDF5 files.

    Args:
        hr_file_path (str): Path to high-resolution H5 file.
        lr_file_path (str): Path to low-resolution H5 file.
        num_frames (int): Number of frames to load.

    Returns:
        hr_data: np.array of shape (num_frames, nx, ny, nz, 3)
        lr_data: np.array of shape (num_frames, nx_lr, ny_lr, nz_lr, 3)
    """
    print("Loading H5 files...")

    # Load HR
    with h5py.File(hr_file_path, 'r') as f:
        U = f['U'][:num_frames]
        V = f['V'][:num_frames]
        W = f['W'][:num_frames]
        hr_data = np.stack([U, V, W], axis=-1)
        print(f"HR shape: {hr_data.shape}, range: [{np.min(hr_data):.6f}, {np.max(hr_data):.6f}]")

    # Load LR
    with h5py.File(lr_file_path, 'r') as f:
        lr_data = f['flow'][:num_frames]
        print(f"LR shape: {lr_data.shape}, range: [{np.min(lr_data):.6f}, {np.max(lr_data):.6f}]")

    assert hr_data.shape[0] == lr_data.shape[0], "Number of frames mismatch between HR and LR."
    print(f"Using {hr_data.shape[0]} frames.")
    return hr_data, lr_data


# ============================ 2. Normalize velocity data ============================
def normalize_velocity_data(hr_data, lr_data):
    """
    Normalize velocity fields per component using HR statistics.

    Args:
        hr_data (np.array): High-resolution data (num_frames, nx, ny, nz, 3)
        lr_data (np.array): Low-resolution data (num_frames, nx_lr, ny_lr, nz_lr, 3)

    Returns:
        normalized_hr: normalized HR data
        normalized_lr: normalized LR data
        normalization_params: dict with 'global_mean' and 'global_std'
    """
    print("\nStarting per-component normalization...")

    global_mean = np.mean(hr_data, axis=(0,1,2,3))
    global_std = np.std(hr_data, axis=(0,1,2,3))

    normalized_hr = (hr_data - global_mean) / global_std
    normalized_lr = (lr_data - global_mean) / global_std

    print("\nNormalization parameters (per component):")
    for i, comp in enumerate(['u','v','w']):
        print(f"{comp}: mean={global_mean[i]:.6f}, std={global_std[i]:.6f}")

    print("\nPost-normalization statistics:")
    for name, data in zip(['HR','LR'], [normalized_hr, normalized_lr]):
        print(f"{name} mean: {np.mean(data, axis=(0,1,2,3))}")
        print(f"{name} std:  {np.std(data, axis=(0,1,2,3))}")
        print(f"{name} range: [min={np.min(data):.3f}, max={np.max(data):.3f}]")

    normalization_params = {
        'global_mean': global_mean,
        'global_std': global_std
    }
    return normalized_hr, normalized_lr, normalization_params


# ============================ 3. Split training/validation data ============================
def prepare_training_data(normalized_hr, normalized_lr, train_frames=120, val_frames=20):
    """
    Sequentially split training and validation data.

    Args:
        normalized_hr: normalized HR data
        normalized_lr: normalized LR data
        train_frames: number of training frames
        val_frames: number of validation frames

    Returns:
        train_data: dict with 'lr' and 'hr'
        val_data: dict with 'lr' and 'hr'
    """
    total_frames = train_frames + val_frames
    assert normalized_hr.shape[0] >= total_frames, "Not enough frames in dataset."

    train_lr = normalized_lr[:train_frames]
    train_hr = normalized_hr[:train_frames]

    val_lr = normalized_lr[train_frames:total_frames]
    val_hr = normalized_hr[train_frames:total_frames]

    print(f"\nTrain set: {train_lr.shape[0]} frames, Validation set: {val_lr.shape[0]} frames")
    print(f"Train indices: 0-{train_frames-1}, Validation indices: {train_frames}-{total_frames-1}")

    return {'lr': train_lr, 'hr': train_hr}, {'lr': val_lr, 'hr': val_hr}


# ============================ 4. Save processed data ============================
def save_processed_data(train_data, val_data, normalization_params, output_dir='./processed_data_seq'):
    """
    Save processed numpy arrays and normalization parameters.

    Args:
        train_data: dict with 'lr' and 'hr'
        val_data: dict with 'lr' and 'hr'
        normalization_params: dict with 'global_mean' and 'global_std'
        output_dir: folder to save data
    """
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'train_lr.npy'), train_data['lr'])
    np.save(os.path.join(output_dir, 'train_hr.npy'), train_data['hr'])
    np.save(os.path.join(output_dir, 'val_lr.npy'), val_data['lr'])
    np.save(os.path.join(output_dir, 'val_hr.npy'), val_data['hr'])

    with open(os.path.join(output_dir, 'normalization_params.pkl'), 'wb') as f:
        pickle.dump(normalization_params, f)

    print(f"\nProcessed data saved to: {output_dir}")


# ============================ 5. Full preprocessing pipeline ============================
def preprocess_dns_data_from_h5(hr_file_path, lr_file_path, train_frames=120, val_frames=20, output_dir='./processed_data_seq'):
    """
    Complete preprocessing pipeline: load, normalize, split, save.
    """
    print("=" * 50)
    print("DNS Data Preprocessing Pipeline")
    print("=" * 50)

    total_frames = train_frames + val_frames
    hr_data, lr_data = load_h5_data(hr_file_path, lr_file_path, num_frames=total_frames)
    normalized_hr, normalized_lr, normalization_params = normalize_velocity_data(hr_data, lr_data)
    train_data, val_data = prepare_training_data(normalized_hr, normalized_lr, train_frames, val_frames)
    save_processed_data(train_data, val_data, normalization_params, output_dir=output_dir)

    print("\nData preprocessing completed!")
    return train_data, val_data, normalization_params


# ============================ 6. Example usage ============================
if __name__ == "__main__":
    LR_H5_PATH = "cube0_all_frames_LR.h5"
    HR_H5_PATH = "cube0_all_frames.h5"

    train_data, val_data, norm_params = preprocess_dns_data_from_h5(
        HR_H5_PATH, LR_H5_PATH, train_frames=120, val_frames=20
    )
