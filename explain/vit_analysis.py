import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
from dataset import get_data_loaders
from metrics import binarize_heatmap_otsu, binarize_heatmap_threshold, calculate_alignment_iou, generate_random_mask, generate_center_mask
from cam_utils import VITAttentionRollout, get_vit_cams_multi
from config import VIT_WEIGHT_PATH, SAVE_DIR_VIT

def run_vit_interpretability():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for inference...")

    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, 7)
    
    print(f"Loading weights: {VIT_WEIGHT_PATH}")
    state_dict = torch.load(VIT_WEIGHT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    rollout = VITAttentionRollout(model)

    print("Loading dataset...")
    _, _, test_loader, class_names = get_data_loaders()

    # Record ViT multi-threshold and LCC data.
    thresholds = np.arange(0.1, 1.0, 0.1)
    methods = ["Attention Rollout", "Grad-CAM (ViT)", "EigenCAM (ViT)"]

    threshold_ious = {m: {t: [] for t in thresholds} for m in methods}

    # baseline IoU
    baseline_random_ious = []
    baseline_center_ious = []

    # 3. Traverse the test set.
    for images, labels, gt_masks, img_paths in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
        correct_mask = (preds == labels)
        
        for i in range(len(labels)):
            if not correct_mask[i]:
                continue 
                
            img_tensor = images[i:i+1] 
            true_class = labels[i].item()
            gt_mask_img = gt_masks[i].numpy() 

            # Calculate baseline IoU for random and center masks
            random_mask = generate_random_mask(gt_mask_img)
            center_mask = generate_center_mask(gt_mask_img)
            
            baseline_random_ious.append(calculate_alignment_iou(random_mask, gt_mask_img))
            baseline_center_ious.append(calculate_alignment_iou(center_mask, gt_mask_img))

            rollout_map = rollout(img_tensor, start_layer=0)
            
            cam_maps = get_vit_cams_multi(model, img_tensor, true_class)
            
            current_heatmaps = {
                "Attention Rollout": rollout_map,
                "Grad-CAM (ViT)": cam_maps["Grad-CAM (ViT)"],
                "EigenCAM (ViT)": cam_maps["EigenCAM (ViT)"]
            }

            for m_name, heatmap in current_heatmaps.items():
                for t in thresholds:
                    pred_t = binarize_heatmap_threshold(heatmap, t)
                    iou_t = calculate_alignment_iou(pred_t, gt_mask_img)
                    threshold_ious[m_name][t].append(iou_t)

    os.makedirs(SAVE_DIR_VIT, exist_ok=True)

    # Aggregate baseline IoUs
    avg_random_iou = np.mean(baseline_random_ious) if baseline_random_ious else 0.0
    avg_center_iou = np.mean(baseline_center_ious) if baseline_center_ious else 0.0

    # ---------------------------------------------------------
    # Multi-threshold IoU sensitivity line chart.
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8)) 
    
    modern_colors = ['#1E88E5', '#FF3366', '#00B4D8', '#8A2BE2', '#FFC300']
    
    methods = ["Attention Rollout", "Grad-CAM (ViT)", "EigenCAM (ViT)"]
    
    for idx, m_name in enumerate(methods):
        avg_ious_per_t = [np.mean(threshold_ious[m_name][t]) for t in thresholds]
        
        plt.plot(thresholds, avg_ious_per_t, 
                 marker='o', markersize=12, linewidth=4, 
                 label=m_name, 
                 color=modern_colors[idx % len(modern_colors)])

    plt.axhline(y=avg_random_iou, color='gray', linestyle='--', linewidth=3, 
                label=f'Random Baseline ({avg_random_iou:.3f})')
    plt.axhline(y=avg_center_iou, color='#333333', linestyle='-.', linewidth=3, 
                label=f'Center Baseline ({avg_center_iou:.3f})')
        
    plt.title("ViT-B/16", fontsize=28, pad=20, weight='bold')
    plt.xlabel("Binarization Threshold", fontsize=24, labelpad=15, weight='bold')
    plt.ylabel("Alignment IoU", fontsize=24, labelpad=15, weight='bold')
    
    plt.xticks(fontsize=20, weight='bold')
    plt.yticks(fontsize=20, weight='bold')
    
    plt.legend(loc='best', prop={'size': 18, 'weight': 'bold'})
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    
    plt.tight_layout() 
    
    curve_save_path = os.path.join(SAVE_DIR_VIT, "iou_threshold_curve.png")
    
    plt.savefig(curve_save_path, dpi=300, bbox_inches='tight')
    print(f"\n[Multi-threshold line chart] Saved to: {curve_save_path}")

if __name__ == '__main__':
    run_vit_interpretability()