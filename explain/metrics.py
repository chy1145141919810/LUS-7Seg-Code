import cv2
import numpy as np

def binarize_heatmap_otsu(heatmap: np.ndarray) -> np.ndarray:
    """
    使用 Otsu 方法将连续的热力图二值化
    # Binarize the continuous heatmap using the Otsu method.
    :param heatmap: 归一化到 [0, 255] 的热力图数组 (H, W), dtype=np.uint8
    # :param heatmap: Heatmap array normalized to [0, 255] (H, W), dtype=np.uint8.
    """
    if heatmap.dtype != np.uint8:
        heatmap = (heatmap * 255).astype(np.uint8)
        
    _, binary_mask = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (binary_mask > 0).astype(np.uint8)

def binarize_heatmap_threshold(heatmap: np.ndarray, threshold: float) -> np.ndarray:
    """
    [新增] 使用指定阈值将连续的热力图二值化
    # [Added] Binarize the continuous heatmap using the specified threshold.
    :param heatmap: 归一化到 [0, 1] 的热力图数组
    # :param heatmap: Heatmap array normalized to [0, 1].
    :param threshold: 阈值 (0.0 - 1.0)
    # :param threshold: Threshold (0.0 - 1.0).
    """
    return (heatmap > threshold).astype(np.uint8)

def keep_largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    [新增] 形态学后处理：保留二值掩码中的最大连通域（去除孤立噪点）
    # [Added] Morphological post-processing: keep the largest connected component in the binary mask (remove isolated noise).
    :param binary_mask: 0 和 1 的二维 numpy 数组
    # :param binary_mask: Two-dimensional numpy array of 0s and 1s.
    """
    # cv2.connectedComponentsWithStats 要求输入为 uint8 格式
    # cv2.connectedComponentsWithStats requires input in uint8 format.
    binary_uint8 = (binary_mask * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_uint8, connectivity=8)
    
    # 如果只有背景 (num_labels == 1) 或全空，直接返回原图
    # If there is only background (num_labels == 1) or it is completely empty, return the original image directly.
    if num_labels <= 1:
        return binary_mask
        
    # 找到面积最大的连通域 (排除背景标签 0)
    # Find the connected component with the largest area (excluding background label 0).
    # stats 每一行是 [x, y, w, h, area]
    # Each row of stats is [x, y, w, h, area].
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # 生成仅包含最大连通域的掩码
    # Generate a mask containing only the largest connected component.
    lcc_mask = (labels == largest_label).astype(np.uint8)
    return lcc_mask

def calculate_alignment_iou(pred_binary_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    计算预测关注区域与医生标注掩码的交并比 (IoU)
    # Calculate the Intersection over Union (IoU) between the predicted region of interest and the doctor's labeled mask.
    """
    gt_binary = (np.sum(gt_mask, axis=-1) > 0).astype(np.uint8)
    
    if pred_binary_mask.shape != gt_binary.shape:
        pred_binary_mask = cv2.resize(pred_binary_mask, (gt_binary.shape[1], gt_binary.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    intersection = np.logical_and(pred_binary_mask, gt_binary).sum()
    union = np.logical_or(pred_binary_mask, gt_binary).sum()
    
    if union == 0:
        return 0.0 
        
    return float(intersection / union)