import torch
import os

# ================= Dataset Path =================
DATASET_ROOT = os.environ.get('LUS7SEG_DATA', './data')

# ================= Hyperparameter Settings =================
# When testing on a notebook, set BATCH_SIZE smaller (e.g., 4 or 8) first, then increase it (e.g., 32, 64) after going to the server.
BATCH_SIZE = 128  
EPOCHS = 100       
LEARNING_RATE = 1e-4
NUM_CLASSES = 7  

# ================= Hardware Configuration =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to save the model
MODEL_SAVE_DIR = os.environ.get('MODEL_DIR', './checkpoints')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'best_densenet121.pth')
