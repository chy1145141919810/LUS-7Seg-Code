import segmentation_models_pytorch as smp
from config import NUM_CLASSES

def get_unet_model():
    model = smp.Unet(
        encoder_name="resnet50",        
        encoder_weights="imagenet",         
        in_channels=3,                     
        classes=NUM_CLASSES,              
    )
    return model