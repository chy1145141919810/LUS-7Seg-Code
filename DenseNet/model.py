import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES

def get_densenet_model():
    # Load pre-trained DenseNet weights.
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
      
    # Replace the last fully connected layer, change the number of output features to 7.
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    
    return model