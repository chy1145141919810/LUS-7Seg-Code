import os

# ==========================================
# LUS-7Seg dataset and training configuration.
# ==========================================

# 1. Path configuration.
DATASET_ROOT = os.environ.get('LUS7SEG_DATA', './data')

RESNET_WEIGHT_PATH = os.environ.get('RESNET_WEIGHT', './checkpoints/best_resnet50.pth')
DENSENET_WEIGHT_PATH = os.environ.get('DENSENET_WEIGHT', './checkpoints/best_densenet121.pth')
VIT_WEIGHT_PATH = os.environ.get('VIT_WEIGHT', './checkpoints/best_vit.pth')
EXPLAIN_BASE_DIR = os.environ.get('EXPLAIN_DIR', './explain_results')
SAVE_DIR_RESNET = os.path.join(EXPLAIN_BASE_DIR, 'resnet_results')
SAVE_DIR_DENSENET = os.path.join(EXPLAIN_BASE_DIR, 'densenet_results')
SAVE_DIR_VIT = os.path.join(EXPLAIN_BASE_DIR, 'vit_results')

# 2. Hyperparameter configuration.
BATCH_SIZE = 16  
NUM_WORKERS = 8

# ==========================================
# Clinical semantics and category configuration.
# ==========================================

# 3. List of standard section categories (must be consistent with the dictionary order read by ImageFolder).
# 7 section abbreviations corresponding to Table 1.
CLASSES = [
    "FPH", 
    "GBH", 
    "LHA", 
    "LHP", 
    "LHV", 
    "RH",  
    "SPH"  
]

# 4. Mask RGB color mapping (refer to paper Section 3.3).
MASK_COLORS = {
    "Liver": (255, 255, 255),       # White: Liver parenchyma area.
    "Gallbladder": (0, 255, 0),     # Green: Gallbladder area.
    "Background": (0, 0, 0)         # Black: Background and artifact area.
}

# 5. List of CAM methods for interpretability analysis.
CAM_METHODS = [
    "Grad-CAM", 
    "Grad-CAM++", 
    "XGrad-CAM", 
    "LayerCAM"
]