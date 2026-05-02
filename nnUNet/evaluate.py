import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from medpy import metric

# ================= Path Configuration =================
# The directory where nnU-Net just outputted prediction results.
PREDICT_DIR = os.environ.get('NNUNET_PREDICT_DIR', './nnUNet_results/predictions_ensemble')
# The root directory of the original dataset containing your real RGB masks.
ORIGINAL_DATASET_ROOT = os.environ.get('LUS7SEG_DATA', './data')

NUM_CLASSES = 3  # 0: Background, 1: Liver, 2: Gallbladder.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_file_index(dataset_root):
    print("Building global file hash index, please wait...")
    file_index = {}
    for root, _, files in os.walk(dataset_root):
        for f in files:
            if f.endswith(('.png', '.jpg')):
                file_index[f] = os.path.join(root, f)
    print(f"Index construction complete! Recorded a total of {len(file_index)} file paths.")
    return file_index

def calculate_metrics_gpu(pred_tensor, true_tensor, num_classes):
    ious, dices, hd95s, sens = [], [], [],[]
    
    for cls in range(1, num_classes):
        pred_inds = (pred_tensor == cls)
        target_inds = (true_tensor == cls)
        
        target_sum = target_inds.sum().item()
        pred_sum = pred_inds.sum().item()
        
        if target_sum == 0:
            if pred_sum == 0:
                ious.append(np.nan); dices.append(np.nan); hd95s.append(np.nan); sens.append(np.nan)
            else:
                ious.append(0.0); dices.append(0.0); hd95s.append(np.nan); sens.append(np.nan)
        else:
            intersection = (pred_inds & target_inds).sum().item()
            union = pred_sum + target_sum - intersection
            
            ious.append(intersection / union)
            dices.append(2.0 * intersection / (pred_sum + target_sum))
            sens.append(intersection / target_sum)
            
            if pred_sum > 0:
                pred_np_bool = pred_inds.cpu().numpy()
                target_np_bool = target_inds.cpu().numpy()
                try:
                    hd95_val = metric.binary.hd95(pred_np_bool, target_np_bool)
                    hd95s.append(hd95_val)
                except Exception:
                    hd95s.append(np.nan)
            else:
                hd95s.append(np.nan)
                
    return ious, dices, hd95s, sens

def decode_mask_to_rgb(mask_np):
    h, w = mask_np.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Liver (Class 1) -> White.
    rgb_mask[mask_np == 1] = [255, 255, 255]
    # Gallbladder (Class 2) -> Green.
    rgb_mask[mask_np == 2] = [0, 255, 0]
    
    return rgb_mask 


def evaluate_nnunet():
    print(f"Environment detection: Using {DEVICE} for accelerated evaluation...")
    print("Starting evaluation of nnU-Net test set prediction results...")

    pred_filenames = [f for f in os.listdir(PREDICT_DIR) if f.endswith('.png')]
    if not pred_filenames:
        print(f"Error: No prediction results found under {PREDICT_DIR}.")
        return
    
    # 1. Establish file speed index.
    file_index = build_file_index(ORIGINAL_DATASET_ROOT)
    
    all_ious_liver, all_dices_liver, all_hd95_liver, all_sens_liver = [], [], [],[]
    all_ious_gb, all_dices_gb, all_hd95_gb, all_sens_gb = [], [], [],[]
    
    # Directory for saving visualizations.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vis_dir = os.path.join(current_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    max_save_images = 10
    save_count = 0

    print("Starting high-speed calculation of metrics...")
    start_time = time.time()

    for pred_filename in pred_filenames:
        # --- O(1) high-speed path matching ---
        orig_base_name = pred_filename.replace('.png', '')
        orig_mask_name = f"{orig_base_name}.png"
        orig_img_name = f"{orig_base_name}.jpg"
        
        exact_mask_path = file_index.get(orig_mask_name)
        exact_img_path = file_index.get(orig_img_name)
        
        if not exact_mask_path or not exact_img_path:
            print(f"Warning: Cannot find the original Ground Truth or image for {orig_base_name}, skipped.")
            continue
            
        class_dir_found = os.path.basename(os.path.dirname(exact_mask_path))

        # --- Data reading and conversion ---
        pred_path = os.path.join(PREDICT_DIR, pred_filename)
        pred_np = np.array(Image.open(pred_path))
        
        true_rgb = Image.open(exact_mask_path).convert("RGB")
        true_np = np.array(true_rgb)
        
        true_mask_np = np.zeros((true_np.shape[0], true_np.shape[1]), dtype=np.uint8)
        r, g, b = true_np[:,:,0], true_np[:,:,1], true_np[:,:,2]
        # White -> Class 1.
        liver_idx = (r > 200) & (g > 200) & (b > 200)
        # Green -> Class 2.
        gb_idx = (r < 50) & (g > 200) & (b < 50)
        true_mask_np[liver_idx] = 1
        true_mask_np[gb_idx] = 2
        
        # --- GPU tensorization calculation ---
        # Transfer to GPU for high-speed calculation.
        pred_tensor = torch.tensor(pred_np, dtype=torch.uint8, device=DEVICE)
        true_tensor = torch.tensor(true_mask_np, dtype=torch.uint8, device=DEVICE)
        
        ious, dices, hd95s, sens = calculate_metrics_gpu(pred_tensor, true_tensor, NUM_CLASSES)
        
        if not np.isnan(ious[0]): all_ious_liver.append(ious[0])
        if not np.isnan(dices[0]): all_dices_liver.append(dices[0])
        if not np.isnan(hd95s[0]): all_hd95_liver.append(hd95s[0])
        if not np.isnan(sens[0]): all_sens_liver.append(sens[0])
        
        if not np.isnan(ious[1]): all_ious_gb.append(ious[1])
        if not np.isnan(dices[1]): all_dices_gb.append(dices[1])
        if not np.isnan(hd95s[1]): all_hd95_gb.append(hd95s[1])
        if not np.isnan(sens[1]): all_sens_gb.append(sens[1])
    
        # --- Qualitative visualization ---
        if save_count < max_save_images:
            orig_img = Image.open(exact_img_path).convert("RGB")
            true_rgb_vis = decode_mask_to_rgb(true_mask_np)
            pred_rgb_vis = decode_mask_to_rgb(pred_np)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(orig_img)
            axes[0].set_title("Original Ultrasound", fontsize=28, weight='bold', pad=15)
            axes[0].axis('off')
            
            axes[1].imshow(true_rgb_vis)
            axes[1].set_title("Ground Truth Mask", fontsize=28, weight='bold', pad=15)
            axes[1].axis('off')
            
            axes[2].imshow(pred_rgb_vis)
            axes[2].set_title("nnU-Net Prediction", fontsize=28, weight='bold', pad=15)
            axes[2].axis('off')

            plt.tight_layout(w_pad=0.5)
            vis_path = os.path.join(vis_dir, f'nnunet_comparison_{save_count+1}_{class_dir_found}.png')
            plt.savefig(vis_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)
            save_count += 1

    end_time = time.time()

    print("\n" + "="*40)
    print(f"====== Quantitative Evaluation Results on Test Set (Task 2) ======")
    print("="*40)
    print(f"[Liver - Liver Parenchyma]")
    print(f"Mean IoU:  {np.mean(all_ious_liver):.4f}")
    print(f"Mean Dice: {np.mean(all_dices_liver):.4f}")
    print(f"Mean HD95: {np.mean(all_hd95_liver):.4f} px")
    print(f"Mean Sens: {np.mean(all_sens_liver):.4f}")
    print("-" * 40)
    print(f"[Gallbladder - Gallbladder]")
    if len(all_ious_gb) > 0:
        print(f"Mean IoU:  {np.mean(all_ious_gb):.4f}")
        print(f"Mean Dice: {np.mean(all_dices_gb):.4f}")
        print(f"Mean HD95: {np.mean(all_hd95_gb):.4f} px")
        print(f"Mean Sens: {np.mean(all_sens_gb):.4f}")
    else:
        print("Warning: No valid gallbladder positive samples included in the calculation.")
    print("="*40)
    print(f"Qualitative visualization results saved to: {vis_dir}")

if __name__ == '__main__':
    evaluate_nnunet()