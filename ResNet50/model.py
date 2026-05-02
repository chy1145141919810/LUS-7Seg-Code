import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES

def get_resnet50_model():
    # Load pre-trained ResNet50 weights.
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Replace the last fully connected layer, change the number of output features to 7.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    return model