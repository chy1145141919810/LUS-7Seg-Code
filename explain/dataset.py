import os
import cv2
import numpy as np
import random
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from config import DATASET_ROOT, BATCH_SIZE

class PatientCAMDataset(Dataset):
    def __init__(self, patient_folders, class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = class_to_idx

        # Strictly restricted to 7 standard sections.
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
                            self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 1. Get the original PIL image and apply the preprocessing pipeline.
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = image
            
        # 2. Derive and load the corresponding Ground Truth Mask (.png).
        mask_path = img_path.replace('.jpg', '.png')
        
        if os.path.exists(mask_path):
            # Read the mask in RGB format to retain white (liver) and green (gallbladder) features.
            gt_mask = cv2.imread(mask_path)
            gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)
            
            # Uniformly scale the mask to 224x224.
            # Nearest neighbor interpolation (INTER_NEAREST) must be used, and the original pixel color values must never be destroyed.
            gt_mask = cv2.resize(gt_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        else:
            # Fault tolerance processing: if no mask is found, return a 224x224 all-black empty mask.
            gt_mask = np.zeros((224, 224, 3), dtype=np.uint8)
            
        # Return: preprocessed image tensor, category label, real RGB mask, original image path.
        return image_tensor, label, gt_mask, img_path

def get_data_loaders():
    # Image preprocessing and data augmentation.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),               
        transforms.RandomRotation(degrees=15),                
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. Get all valid patient folder paths.
    all_patients = [os.path.join(DATASET_ROOT, d) for d in os.listdir(DATASET_ROOT) 
                    if os.path.isdir(os.path.join(DATASET_ROOT, d)) and d.startswith('Patient')]
    
    class_names = ['FPH', 'GBH', 'LHA', 'LHP', 'LHV', 'RH', 'SPH']
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    # 2. Count the total number of valid images owned by each patient.
    patient_image_counts = {}
    total_images = 0
    
    for patient_dir in all_patients:
        count = 0
        for class_name in class_names:
            class_dir = os.path.join(patient_dir, class_name)
            if os.path.isdir(class_dir):
                count += len([f for f in os.listdir(class_dir) if f.lower().endswith('.jpg')])
        patient_image_counts[patient_dir] = count
        total_images += count

    # 3. Set the target image capacity for each set (70% : 10% : 20%).
    target_train_imgs = total_images * 0.7
    target_val_imgs = total_images * 0.1
    target_test_imgs = total_images * 0.2

    # 4. Fix the random seed and shuffle the order of patients.
    random.seed(42)
    shuffled_patients = list(all_patients)
    random.shuffle(shuffled_patients)

    # 5. Greedy algorithm for patient allocation.
    train_patients, val_patients, test_patients = [], [], []
    current_train_imgs, current_val_imgs, current_test_imgs = 0, 0, 0

    for patient in shuffled_patients:
        img_count = patient_image_counts[patient]
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

    print(f"CAM dataset allocation complete! (Total number of images: {total_images})")
    print(f"Actual allocation -> Train: {current_train_imgs} | Val: {current_val_imgs} | Test: {current_test_imgs}")

    # 6. Instantiate the customized dataset.
    train_dataset = PatientCAMDataset(train_patients, class_to_idx, transform=train_transform)
    val_dataset = PatientCAMDataset(val_patients, class_to_idx, transform=test_transform)
    test_dataset = PatientCAMDataset(test_patients, class_to_idx, transform=test_transform)

    # 7. Create DataLoader.
    # When running on the server, you can set num_workers to 4 or 8 to speed up data loading.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    
    return train_loader, val_loader, test_loader, class_names