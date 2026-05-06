import cv2
import numpy as np
import random

# Binarize the continuous heatmap using the Otsu method.
def binarize_heatmap_otsu(heatmap: np.ndarray) -> np.ndarray:      
    if heatmap.dtype != np.uint8:
        heatmap = (heatmap * 255).astype(np.uint8)
        
    _, binary_mask = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (binary_mask > 0).astype(np.uint8)

# Binarize the continuous heatmap using the specified threshold.
def binarize_heatmap_threshold(heatmap: np.ndarray, threshold: float) -> np.ndarray:
    return (heatmap > threshold).astype(np.uint8)

# Morphological post-processing: keep the largest connected component in the binary mask (remove isolated noise).
def keep_largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    # cv2.connectedComponentsWithStats requires input in uint8 format.
    binary_uint8 = (binary_mask * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_uint8, connectivity=8)
    
    # If there is only background (num_labels == 1) or it is completely empty, return the original image directly.
    if num_labels <= 1:
        return binary_mask
        
    # Find the connected component with the largest area (excluding background label 0).
    # Each row of stats is [x, y, w, h, area].
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Generate a mask containing only the largest connected component.
    lcc_mask = (labels == largest_label).astype(np.uint8)
    return lcc_mask

# Calculate the Intersection over Union (IoU) between the predicted region of interest and the doctor's labeled mask.
def calculate_alignment_iou(pred_binary_mask: np.ndarray, gt_mask: np.ndarray) -> float:    
    gt_binary = (np.sum(gt_mask, axis=-1) > 0).astype(np.uint8)
    
    if pred_binary_mask.shape != gt_binary.shape:
        pred_binary_mask = cv2.resize(pred_binary_mask, (gt_binary.shape[1], gt_binary.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    intersection = np.logical_and(pred_binary_mask, gt_binary).sum()
    union = np.logical_or(pred_binary_mask, gt_binary).sum()
    
    if union == 0:
        return 0.0 
        
    return float(intersection / union)

# Generate a random mask as a baseline for random placement
def generate_random_mask(gt_mask: np.ndarray) -> np.ndarray:
    gt_binary = (np.sum(gt_mask, axis=-1) > 0).astype(np.uint8)
    area = np.sum(gt_binary)
    if area == 0:
        return np.zeros_like(gt_binary)
        
    radius = int(np.sqrt(area / np.pi))
    h, w = gt_binary.shape
    
    # Restrict the center of the circle to avoid boundary issues
    center_x = random.randint(radius, max(radius, w - radius - 1))
    center_y = random.randint(radius, max(radius, h - radius - 1))
    
    random_mask = np.zeros_like(gt_binary)
    cv2.circle(random_mask, (center_x, center_y), radius, 1, thickness=-1)
    return random_mask

# Generate a center mask as a baseline for human operation prior knowledge
def generate_center_mask(gt_mask: np.ndarray) -> np.ndarray:
    gt_binary = (np.sum(gt_mask, axis=-1) > 0).astype(np.uint8)
    area = np.sum(gt_binary)
    if area == 0:
        return np.zeros_like(gt_binary)
        
    radius = int(np.sqrt(area / np.pi))
    h, w = gt_binary.shape
    
    # Fixed center in the field of view
    center_x = w // 2
    center_y = h // 2
    
    center_mask = np.zeros_like(gt_binary)
    cv2.circle(center_mask, (center_x, center_y), radius, 1, thickness=-1)
    return center_mask