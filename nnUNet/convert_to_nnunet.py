import os
import json
import numpy as np
from PIL import Image
import random

# ================= Configuration Path =================
# Your original dataset root directory.
ORIGINAL_DATASET_ROOT = os.environ.get('LUS7SEG_DATA', './data')

# The basic environment directory required by nnU-Net v2 (we first build it in the project directory, and later configure the environment variables to point here).
NNUNET_RAW_DIR = os.environ.get('NNUNET_RAW', './nnUNet_raw')

# Assign a unique ID to your dataset. The nnU-Net specification is Dataset + three-digit number + name.
DATASET_NAME = 'Dataset501_LUS7Seg'
TARGET_DIR = os.path.join(NNUNET_RAW_DIR, DATASET_NAME)

# Subdirectories mandatory inside nnU-Net.
IMAGES_TR = os.path.join(TARGET_DIR, 'imagesTr') # Original images of the training set.
LABELS_TR = os.path.join(TARGET_DIR, 'labelsTr') # Masks of the training set.
IMAGES_TS = os.path.join(TARGET_DIR, 'imagesTs') # Original images of the independent test set.

def setup_directories():
    for d in [IMAGES_TR, LABELS_TR, IMAGES_TS]:
        os.makedirs(d, exist_ok=True)
    print(f"Created nnU-Net target directory structure: {TARGET_DIR}")

def convert_mask_to_grayscale(rgb_mask_path, target_path):
    mask_img = Image.open(rgb_mask_path).convert("RGB")
    mask_np = np.array(mask_img)
    
    # Initialize all 0 background.
    out_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.uint8)
    
    # Extract color channels.
    r, g, b = mask_np[:,:,0], mask_np[:,:,1], mask_np[:,:,2]
    
    # Strictly match your semantic colors.
    liver_idx = (r > 200) & (g > 200) & (b > 200)     # White -> Class 1.
    gb_idx = (r < 50) & (g > 200) & (b < 50)          # Green -> Class 2.
    
    out_mask[liver_idx] = 1
    out_mask[gb_idx] = 2
    
    # Save as single-channel PNG.
    Image.fromarray(out_mask, mode='L').save(target_path)

def process_and_split_data():
    # 1. Get all patient directories.
    all_patients = [os.path.join(ORIGINAL_DATASET_ROOT, d) for d in os.listdir(ORIGINAL_DATASET_ROOT) 
                    if os.path.isdir(os.path.join(ORIGINAL_DATASET_ROOT, d)) and d.startswith('Patient')]
    
    valid_classes = ['FPH', 'GBH', 'LHA', 'LHP', 'LHV', 'RH', 'SPH']
    
    # 2. Scan and count the valid image and mask paths for each patient.
    patient_img_paths = {} 
    total_images = 0
    
    for patient_dir in all_patients:
        paths = []
        for class_name in valid_classes:
            class_dir = os.path.join(patient_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith('.jpg'):
                        img_path = os.path.join(class_dir, img_name)
                        mask_path = img_path.replace('.jpg', '.png')
                        if os.path.exists(mask_path):
                            paths.append((img_path, mask_path))
        patient_img_paths[patient_dir] = paths
        total_images += len(paths)

    print(f"Found a total of {total_images} valid sample pairs, starting partitioning by patient...")

    # 3. Set the target image capacity for each set (strictly aligned with U-Net's 70:10:20).
    target_train_imgs = total_images * 0.7
    target_val_imgs = total_images * 0.1
    target_test_imgs = total_images * 0.2

    # 4. Shuffle the order of patients (using the same random seed 42 to ensure the allocation results are consistent with U-Net).
    random.seed(42)
    shuffled_patients = list(all_patients)
    random.shuffle(shuffled_patients)

    # 5. Greedy algorithm for patient allocation.
    train_patients, val_patients, test_patients = [], [], []
    current_train_imgs, current_val_imgs, current_test_imgs = 0, 0, 0

    for patient in shuffled_patients:
        img_count = len(patient_img_paths[patient])
        if img_count == 0:
            continue
            
        train_ratio = current_train_imgs / target_train_imgs if target_train_imgs > 0 else 1
        val_ratio = current_val_imgs / target_val_imgs if target_val_imgs > 0 else 1
        test_ratio = current_test_imgs / target_test_imgs if target_test_imgs > 0 else 1
        
        if train_ratio <= val_ratio and train_ratio <= test_ratio:
            train_patients.append(patient)
            current_train_imgs += img_count
        elif val_ratio <= train_ratio and val_ratio <= test_ratio:
            val_patients.append(patient)
            current_val_imgs += img_count
        else:
            test_patients.append(patient)
            current_test_imgs += img_count

    # nnU-Net needs to merge our Train and Val as its training set.
    nnunet_train_patients = train_patients + val_patients
    nnunet_test_patients = test_patients

    print("Starting conversion and copying of data (due to the large amount of data, please be patient)...")
    
    num_train_images = 0
    num_test_images = 0

    # 6. Process the training set (Train + Val -> imagesTr & labelsTr).
    for patient in nnunet_train_patients:
        for orig_img_path, orig_mask_path in patient_img_paths[patient]:
            base_name = os.path.basename(orig_img_path).replace('.jpg', '')
            
            target_img_path = os.path.join(IMAGES_TR, f"{base_name}_0000.png")
            target_mask_path = os.path.join(LABELS_TR, f"{base_name}.png")
            
            Image.open(orig_img_path).convert("RGB").save(target_img_path)
            convert_mask_to_grayscale(orig_mask_path, target_mask_path)
            num_train_images += 1

    # 7. Process the independent test set (Test -> imagesTs).
    for patient in nnunet_test_patients:
        for orig_img_path, _ in patient_img_paths[patient]:
            base_name = os.path.basename(orig_img_path).replace('.jpg', '')
            
            target_img_path = os.path.join(IMAGES_TS, f"{base_name}_0000.png")
            Image.open(orig_img_path).convert("RGB").save(target_img_path)
            num_test_images += 1

    print(f"Data processing complete!")
    print(f"Merged training set (imagesTr): {num_train_images} images")
    print(f"Independent test set (imagesTs): {num_test_images} images")
    
    return num_train_images

def generate_dataset_json(num_training):
    dataset_info = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"
        },
        "labels": {
            "background": 0,
            "liver": 1,
            "gallbladder": 2
        },
        "numTraining": num_training,
        "file_ending": ".png"
    }
    
    json_path = os.path.join(TARGET_DIR, 'dataset.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4)
    print(f"Generated dataset configuration file: {json_path}")

if __name__ == '__main__':
    setup_directories()
    num_train = process_and_split_data()
    generate_dataset_json(num_train)
    print("\n✅ Format conversion all complete! Your data now conforms to the nnU-Net v2 standard.")