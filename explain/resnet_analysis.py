import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
from dataset import get_data_loaders
from metrics import binarize_heatmap_otsu, binarize_heatmap_threshold, calculate_alignment_iou
from cam_utils import get_all_cams
from config import RESNET_WEIGHT_PATH, SAVE_DIR_RESNET, CAM_METHODS
from torchvision import transforms
from pytorch_grad_cam.utils.image import show_cam_on_image

def run_interpretability_analysis():
    
    # 1. Initialization settings and data loading.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for inference...")

    model = models.resnet50(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    
    print(f"Loading weights: {RESNET_WEIGHT_PATH}")
    state_dict = torch.load(RESNET_WEIGHT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print("Loading dataset...")
    _, _, test_loader, class_names = get_data_loaders()
    target_layers = [model.layer3[-1]] 

    # Initialize data recording dictionary.
    thresholds = np.arange(0.1, 1.0, 0.1) # 0.1, 0.2 ... 0.9
    threshold_ious = {method: {t: [] for t in thresholds} for method in CAM_METHODS}
    
    
    # 2. Traverse the test set.
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
                
            img_tensor = images[i].unsqueeze(0)
            true_class = labels[i].item()
            gt_mask_img = gt_masks[i].numpy() 
            
            # 3. Batch extract 4 types of heatmaps.
            heatmaps_dict = get_all_cams(model, img_tensor, true_class, target_layers)
            
            # 4. Process each method in a loop.
            for method_name, heatmap in heatmaps_dict.items():
                
                for t in thresholds:
                    pred_t = binarize_heatmap_threshold(heatmap, t)
                    iou_t = calculate_alignment_iou(pred_t, gt_mask_img)
                    threshold_ious[method_name][t].append(iou_t)
                

    os.makedirs(SAVE_DIR_RESNET, exist_ok=True)

    # ---------------------------------------------------------
    # Multi-threshold IoU sensitivity line chart (Threshold Sensitivity Curve).
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8)) 
    
    modern_colors =['#1E88E5', '#FF3366', '#00B4D8', '#8A2BE2', '#FFC300']
    
    for idx, method in enumerate(CAM_METHODS):
        avg_ious_per_t =[np.mean(threshold_ious[method][t]) for t in thresholds]
        plt.plot(thresholds, avg_ious_per_t, marker='o', markersize=12, linewidth=4, 
                 label=method, color=modern_colors[idx % len(modern_colors)])
        
    plt.title("ResNet50", fontsize=28, pad=20, weight='bold')
    plt.xlabel("Binarization Threshold", fontsize=24, labelpad=15, weight='bold')
    plt.ylabel("Alignment IoU", fontsize=24, labelpad=15, weight='bold')
    
    plt.xticks(fontsize=20, weight='bold')
    plt.yticks(fontsize=20, weight='bold')
    
    plt.legend(loc='best', prop={'size': 18, 'weight': 'bold'})
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    
    plt.tight_layout() 
    curve_save_path = os.path.join(SAVE_DIR_RESNET, "iou_threshold_curve.png")
    plt.savefig(curve_save_path, dpi=300, bbox_inches='tight')
    print(f"\n[Multi-threshold line chart] Saved to: {curve_save_path}")

  
if __name__ == '__main__':
    run_interpretability_analysis()