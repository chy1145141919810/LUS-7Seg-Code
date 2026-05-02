import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from config import DATASET_ROOT, BATCH_SIZE

class PatientSegDataset(Dataset):
    def __init__(self, patient_folders, is_train=False):
        self.samples = []
        self.is_train = is_train
        
        # Color jitter and normalization only apply to the original image.
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Define the 7 core standard sections of LUS-7Seg.
        valid_classes = ['FPH', 'GBH', 'LHA', 'LHP', 'LHV', 'RH', 'SPH']

        # Traverse the patient folders assigned to this dataset.
        for patient_folder in patient_folders:
            for class_name in os.listdir(patient_folder):
                if class_name not in valid_classes:
                    continue
                class_dir = os.path.join(patient_folder, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith('.jpg'):
                            img_path = os.path.join(class_dir, img_name)
                            # Mask has the same name as the original image but with a .png suffix.
                            mask_path = img_path.replace('.jpg', '.png')
                            
                            if os.path.exists(mask_path):
                                self.samples.append((img_path, mask_path))
                            else:
                                print(f"Warning: Mask file missing, skipping sample: {mask_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # Uniformly converted to PIL Image in RGB mode.
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # --- Step 1: Synchronized Resize (original image uses bilinear, mask must use nearest neighbor) ---
        image = TF.resize(image, (224, 224), interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (224, 224), interpolation=InterpolationMode.NEAREST)

       # --- Step 2: Training set exclusive synchronized geometric transformation ---
        if self.is_train:
            # 1. Random horizontal flip (50% probability)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # 2. Random rotation (-15 degrees to 15 degrees)
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            
            # 3. Color jitter (only applied to the original image, simulating gain differences between different devices)
            image = self.color_jitter(image)

        # --- Step 3: Original image converted to Tensor and normalized ---
        image = TF.to_tensor(image)
        image = self.normalize(image)

        # --- Step 4: Mask RGB converted to category index Tensor ---
        # Semantic definition: white=liver parenchyma area, green=gallbladder area, unlabeled and artifact areas are uniformly kept as black.
        mask_np = np.array(mask)
        mask_tensor = torch.zeros((224, 224), dtype=torch.long)
        
        r, g, b = mask_np[:,:,0], mask_np[:,:,1], mask_np[:,:,2]
        
        # Threshold matching to prevent label loss caused by small color differences.
        liver_idx = (r > 200) & (g > 200) & (b > 200) 
        gb_idx = (r < 50) & (g > 200) & (b < 50)
        
        mask_tensor[liver_idx] = 1
        mask_tensor[gb_idx] = 2

        return image, mask_tensor, str(img_path)

# Dataset loading and partition entry.
def get_data_loaders():
    # 1. Get all valid patient folder paths.
    all_patients = [os.path.join(DATASET_ROOT, d) for d in os.listdir(DATASET_ROOT) 
                    if os.path.isdir(os.path.join(DATASET_ROOT, d)) and d.startswith('Patient')]
    
    valid_classes = ['FPH', 'GBH', 'LHA', 'LHP', 'LHV', 'RH', 'SPH']
    
    # 2. Count the total number of valid images owned by each patient.
    patient_image_counts = {}
    total_images = 0
    
    for patient_dir in all_patients:
        count = 0
        for class_name in valid_classes:
            class_dir = os.path.join(patient_dir, class_name)
            if os.path.isdir(class_dir):
                count += len([f for f in os.listdir(class_dir) if f.lower().endswith('.jpg')])
        patient_image_counts[patient_dir] = count
        total_images += count

    # 3. Set the target image capacity for each set (70% Train : 10% Val : 20% Test).
    target_train_imgs = total_images * 0.7
    target_val_imgs = total_images * 0.1
    target_test_imgs = total_images * 0.2

    # Shuffle the order of patients to ensure consistency of randomness in each run.
    random.seed(42) 
    shuffled_patients = list(all_patients)
    random.shuffle(shuffled_patients)

    # 4. Greedy algorithm for patient allocation.
    train_patients, val_patients, test_patients = [], [], []
    current_train_imgs, current_val_imgs, current_test_imgs = 0, 0, 0

    for patient in shuffled_patients:
        img_count = patient_image_counts[patient]
        
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

    print(f"Segmentation dataset allocation complete! (Total Images: {total_images})")
    print(f"Actual image count -> Train: {current_train_imgs} ({current_train_imgs/total_images:.2%}) | "
          f"Val: {current_val_imgs} ({current_val_imgs/total_images:.2%}) | "
          f"Test: {current_test_imgs} ({current_test_imgs/total_images:.2%})")

    # 5. Instantiate the custom segmentation dataset, data augmentation is only enabled for the training set.
    train_dataset = PatientSegDataset(train_patients, is_train=True)
    val_dataset = PatientSegDataset(val_patients, is_train=False)
    test_dataset = PatientSegDataset(test_patients, is_train=False)

    # 6. Create DataLoader.
    num_workers = 0 if os.name == 'nt' else 8 
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    class_names = ['Background', 'Liver', 'Gallbladder']

    return train_loader, val_loader, test_loader, class_names