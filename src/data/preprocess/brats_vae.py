import os
import random
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

def prepare_vae_data(src_dir: str, output_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Prepare BraTS data for VAE training by splitting into train/val/test sets.
    
    Args:
        src_dir: Path to MICCAI_BraTS2020_TrainingData folder
        output_dir: Path to output directory for processed data
        train_ratio: Ratio of slices for training (default 0.8)
        val_ratio: Ratio of slices for validation (default 0.1)
    """
    # Setup output directories
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val"
    test_dir = Path(output_dir) / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # First pass: collect all slices
    all_slices = []
    patients = sorted([p for p in Path(src_dir).iterdir() if p.is_dir()])
    print(f"Found {len(patients)} patients")

    print("Collecting slices...")
    for patient_dir in tqdm(patients):
        # Load modality images
        flair = nib.load(patient_dir / f"{patient_dir.name}_flair.nii").get_fdata()
        t1 = nib.load(patient_dir / f"{patient_dir.name}_t1.nii").get_fdata()
        t1ce = nib.load(patient_dir / f"{patient_dir.name}_t1ce.nii").get_fdata()
        t2 = nib.load(patient_dir / f"{patient_dir.name}_t2.nii").get_fdata()

        # Process middle slices (80-129)
        for slice_idx in range(80, 129):
            # Stack modalities
            slice_data = np.stack([
                flair[..., slice_idx],
                t1[..., slice_idx], 
                t1ce[..., slice_idx],
                t2[..., slice_idx]
            ], axis=-1)

            # Crop from 240x240 to 224x224
            slice_data = slice_data[8:-8, 8:-8, :]
            
            # # Normalize to [0,1]
            # for i in range(4):
            #     modality = slice_data[..., i]
            #     min_val, max_val = modality.min(), modality.max()
            #     if max_val > min_val:
            #         slice_data[..., i] = (modality - min_val) / (max_val - min_val)

            all_slices.append(slice_data.astype(np.float32))

    # Shuffle and split slices
    random.seed(42)
    random.shuffle(all_slices)
    
    total_slices = len(all_slices)
    n_train = int(total_slices * train_ratio)
    n_val = int(total_slices * val_ratio)
    
    train_slices = all_slices[:n_train]
    val_slices = all_slices[n_train:n_train+n_val]
    test_slices = all_slices[n_train+n_val:]

    # Save slices to respective directories
    def save_slices(slices, output_dir):
        for idx, slice_data in enumerate(slices):
            np.save(
                output_dir / f"image_slice_{idx}.npy",
                slice_data
            )

    print("Saving training slices...")
    save_slices(train_slices, train_dir)
    
    print("Saving validation slices...")
    save_slices(val_slices, val_dir)
    
    print("Saving test slices...")
    save_slices(test_slices, test_dir)

    print("\nDataset split complete:")
    print(f"Train: {len(train_slices)} slices")
    print(f"Val: {len(val_slices)} slices")
    print(f"Test: {len(test_slices)} slices")

if __name__ == "__main__":
    src_dir = "data/brats-2020/MICCAI_BraTS2020_TrainingData"
    output_dir = "data/brats-2020"
    prepare_vae_data(src_dir, output_dir)