import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES

def get_vit_model():
    # Load pre-trained Vision Transformer (ViT-Base, Patch-16, 224x224).
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    
    # Replace the last fully connected layer, change the number of output features to 7.
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, NUM_CLASSES)
    
    return model