import os
import random
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from config import DATASET_ROOT, BATCH_SIZE

class PatientUltrasoundDataset(Dataset):
    def __init__(self, patient_folders, class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = class_to_idx

        # Traverse the patient folders assigned to this dataset.
        for patient_folder in patient_folders:
            # Traverse the 7 standard section folders.
            for class_name in os.listdir(patient_folder):
                class_dir = os.path.join(patient_folder, class_name)
                # Ensure it is a folder and belongs to the 7 categories we defined.
                if os.path.isdir(class_dir) and class_name in class_to_idx:
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith('.jpg'): 
                            img_path = os.path.join(class_dir, img_name)
                            self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

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
    
    # 2. Define category mapping for 7 standard sections.
    class_names = ['FPH', 'GBH', 'LHA', 'LHP', 'LHV', 'RH', 'SPH']
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    print(f"Successfully loaded category dictionary: {class_to_idx}")

    # 3. Count the total number of valid images owned by each patient.
    patient_image_counts = {}
    total_images = 0
    
    for patient_dir in all_patients:
        count = 0
        for class_name in class_names:
            class_dir = os.path.join(patient_dir, class_name)
            if os.path.isdir(class_dir):
                # Count the number of .jpg images in this section folder.
                count += len([f for f in os.listdir(class_dir) if f.lower().endswith('.jpg')])
        patient_image_counts[patient_dir] = count
        total_images += count

    # 4. Set target image capacity for each set (70% : 10% : 20%).
    target_train_imgs = total_images * 0.7
    target_val_imgs = total_images * 0.1
    target_test_imgs = total_images * 0.2

    # Shuffle patient order to ensure randomness.
    import random
    random.seed(42) 
    shuffled_patients = list(all_patients)
    random.shuffle(shuffled_patients)

    # 5. Greedy algorithm for patient allocation.
    train_patients, val_patients, test_patients = [], [], []
    current_train_imgs, current_val_imgs, current_test_imgs = 0, 0, 0

    for patient in shuffled_patients:
        img_count = patient_image_counts[patient]
        
        # Calculate the filling progress of each set from the target capacity (the smaller the ratio, the more data is missing).
        train_ratio = current_train_imgs / target_train_imgs if target_train_imgs > 0 else 1
        val_ratio = current_val_imgs / target_val_imgs if target_val_imgs > 0 else 1
        test_ratio = current_test_imgs / target_test_imgs if target_test_imgs > 0 else 1
        
        # Assign the patient, along with all their images, to the set that currently lacks data the most.
        if train_ratio <= val_ratio and train_ratio <= test_ratio:
            train_patients.append(patient)
            current_train_imgs += img_count
        elif val_ratio <= train_ratio and val_ratio <= test_ratio:
            val_patients.append(patient)
            current_val_imgs += img_count
        else:
            test_patients.append(patient)
            current_test_imgs += img_count

    print(f"Allocation complete!")
    print(f"Expected image ratio -> Train: 70% | Val: 10% | Test: 20%")
    print(f"Actual image count -> Train: {current_train_imgs} ({current_train_imgs/total_images:.2%}) | "
          f"Val: {current_val_imgs} ({current_val_imgs/total_images:.2%}) | "
          f"Test: {current_test_imgs} ({current_test_imgs/total_images:.2%})")

    # 6. Instantiate the custom dataset.
    train_dataset = PatientUltrasoundDataset(train_patients, class_to_idx, transform=train_transform)
    val_dataset = PatientUltrasoundDataset(val_patients, class_to_idx, transform=test_transform)
    test_dataset = PatientUltrasoundDataset(test_patients, class_to_idx, transform=test_transform)

    print(f"Total image count statistics -> Training set: {len(train_dataset)} | Validation set: {len(val_dataset)} | Test set: {len(test_dataset)}")

    # 7. Create DataLoader.
    # When running on the server, you can set num_workers to 4 or 8 to speed up data loading.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader, class_names