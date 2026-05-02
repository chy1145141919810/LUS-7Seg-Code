import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from medpy import metric
from config import DEVICE, MODEL_SAVE_PATH, NUM_CLASSES
from dataset import get_data_loaders
from model import get_unet_model 

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
            sens.append(intersection / target_sum) # Sensitivity (Recall)
            
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

def decode_mask_to_rgb(mask_tensor):
    mask_np = mask_tensor.cpu().numpy()
    h, w = mask_np.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_mask[mask_np == 1] = [255, 255, 255]
    rgb_mask[mask_np == 2] =[0, 255, 0]
    return rgb_mask

def evaluate():
    print(f"Loading test set data...")
    _, _, test_loader, _ = get_data_loaders()
    
    print(f"Loading optimal U-Net model weights...")
    model = get_unet_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    
    all_ious_liver, all_dices_liver, all_hd95_liver, all_sens_liver = [], [], [],[]
    all_ious_gb, all_dices_gb, all_hd95_gb, all_sens_gb = [], [], [],[]

    # Directory for saving visualizations.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vis_dir = os.path.join(current_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set how many comparison charts to save for paper presentation.
    save_count = 0
    max_save_images = 10 

    print("Starting Evaluation: Interpolating 224x224 predictions back to Original Resolution...")
    with torch.no_grad():
        for inputs, masks, paths in test_loader:
            inputs = inputs.to(DEVICE)
            masks = masks.to(DEVICE) 
            
            outputs = model(inputs)
            preds_224 = torch.argmax(outputs, dim=1) # Shape: (B, 224, 224)
            
            for i in range(inputs.size(0)):
                img_path = paths[i]
                
                mask_path = img_path.replace('.jpg', '.png')
                if not os.path.exists(mask_path):
                    continue
                    
                true_rgb = Image.open(mask_path).convert("RGB")
                true_np = np.array(true_rgb)
                orig_H, orig_W = true_np.shape[0], true_np.shape[1]
                
                true_mask_np = np.zeros((orig_H, orig_W), dtype=np.uint8)
                r, g, b = true_np[:,:,0], true_np[:,:,1], true_np[:,:,2]
                true_mask_np[(r > 200) & (g > 200) & (b > 200)] = 1
                true_mask_np[(r < 50) & (g > 200) & (b < 50)] = 2
                true_tensor = torch.tensor(true_mask_np, dtype=torch.uint8, device=DEVICE)
                
                pred_single = preds_224[i].unsqueeze(0).unsqueeze(0).float()
                
                pred_upsampled = F.interpolate(pred_single, size=(orig_H, orig_W), mode='nearest')
                
                pred_tensor_orig_size = pred_upsampled.squeeze().byte() 
                
                ious, dices, hd95s, sens = calculate_metrics_gpu(pred_tensor_orig_size, true_tensor, NUM_CLASSES)
                
                if not np.isnan(ious[0]): all_ious_liver.append(ious[0])
                if not np.isnan(dices[0]): all_dices_liver.append(dices[0])
                if not np.isnan(hd95s[0]): all_hd95_liver.append(hd95s[0])
                if not np.isnan(sens[0]): all_sens_liver.append(sens[0])
                
                if not np.isnan(ious[1]): all_ious_gb.append(ious[1])
                if not np.isnan(dices[1]): all_dices_gb.append(dices[1])
                if not np.isnan(hd95s[1]): all_hd95_gb.append(hd95s[1])
                if not np.isnan(sens[1]): all_sens_gb.append(sens[1])
                
                # Randomly save the first few images for qualitative visualization analysis in the paper.
                if save_count < max_save_images:
                    # Restore the original input image for display (denormalization).
                    img_np = inputs[i].cpu().numpy().transpose(1, 2, 0)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = std * img_np + mean
                    img_np = np.clip(img_np, 0, 1)
                    
                    # Convert mask to RGB.
                    true_rgb_vis = decode_mask_to_rgb(masks[i])
                    pred_rgb_vis = decode_mask_to_rgb(preds_224[i])
                    
                    # Use matplotlib to splice original image, ground truth mask, and prediction result.
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(img_np)
                    axes[0].set_title("Original Ultrasound", fontsize=28, weight='bold', pad=15)
                    axes[0].axis('off')
                    
                    axes[1].imshow(true_rgb_vis)
                    axes[1].set_title("Ground Truth Mask", fontsize=28, weight='bold', pad=15)
                    axes[1].axis('off')
                    
                    axes[2].imshow(pred_rgb_vis)
                    axes[2].set_title("U-Net Prediction", fontsize=28, weight='bold', pad=15) 
                    axes[2].axis('off')

                    # Compact layout and save high-resolution images (add bbox_inches='tight' to strictly align cropped edges).
                    plt.tight_layout(w_pad=0.5)
                    vis_path = os.path.join(vis_dir, f'unet_comparison_{save_count+1}.png')
                    plt.savefig(vis_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
                    plt.close(fig)
                    save_count += 1

    # Calculate and print global average metrics.
    print("\n" + "="*40)
    print("====== Quantitative Evaluation Results on Test Set (Task 2) ======")
    print("="*40)
    print(f"[Liver - Liver Parenchyma]")
    print(f"Mean IoU:  {np.mean(all_ious_liver):.4f}")
    print(f"Mean Dice: {np.mean(all_dices_liver):.4f}")
    print(f"Mean HD95: {np.mean(all_hd95_liver):.4f} px")
    print(f"Mean Sens: {np.mean(all_sens_liver):.4f}")
    print("-" * 40)
    print(f"[Gallbladder - Gallbladder]")
    # Avoid errors caused by no gallbladder data in the test set.
    if len(all_ious_gb) > 0:
        print(f"Mean IoU:  {np.mean(all_ious_gb):.4f}")
        print(f"Mean Dice: {np.mean(all_dices_gb):.4f}")
        print(f"Mean HD95: {np.mean(all_hd95_gb):.4f} px")
        print(f"Mean Sens: {np.mean(all_sens_gb):.4f}")
    else:
        print("No gallbladder samples included in the test set.")
    print("="*40)
    print(f"Qualitative visualization results saved to: {vis_dir}")

if __name__ == '__main__':
    evaluate()