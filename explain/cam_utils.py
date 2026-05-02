from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, LayerCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch

def get_all_cams(model, input_tensor, target_class, target_layers):
    targets = [ClassifierOutputTarget(target_class)]
    heatmaps = {}
    
    # Note: What is stored here is the "class name (Class)" rather than the instantiated object.
    cam_classes = {
        "Grad-CAM": GradCAM,
        "Grad-CAM++": GradCAMPlusPlus,
        "XGrad-CAM": XGradCAM,
        "LayerCAM": LayerCAM
    }
    
    # Instantiate during the loop and use with statement to manage the lifecycle.
    for name, cam_class in cam_classes.items():
        # Automatically mount hooks when entering the with block, and automatically unmount hooks when leaving.
        with cam_class(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            heatmaps[name] = grayscale_cam[0, :]
            
    return heatmaps

import torch
import cv2
import numpy as np

def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def get_vit_cams_multi(model, input_tensor, target_class):
    target_layers = [model.encoder.layers[-1].ln_1]
    targets = [ClassifierOutputTarget(target_class)]
    heatmaps = {}
    with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform_vit) as cam:
        heatmaps["Grad-CAM (ViT)"] = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
    with EigenCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform_vit) as cam:
        heatmaps["EigenCAM (ViT)"] = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
    return heatmaps

class VITAttentionRollout:
    def __init__(self, model):
        self.model = model
        self.attentions = []
        
        # Intercept the attention weight of each Transformer layer.
        for block in self.model.encoder.layers:
            self._patch_forward(block)

    def _patch_forward(self, block):
        original_forward = block.forward
        
        def new_forward(input):
            x = block.ln_1(input)
            # Force set need_weights=True to return the attention matrix.
            # torchvision returns the weights of the average head by default, shape: (batch, seq_len, seq_len).
            attn_output, attn_weights = block.self_attention(x, x, x, need_weights=True)
            block.attn_weights = attn_weights 
            
            x = input + block.dropout(attn_output)
            x = x + block.mlp(block.ln_2(x))
            return x
            
        block.forward = new_forward

    def __call__(self, input_tensor, start_layer=0):
        self.attentions = []
        
        # Forward propagation, trigger hook and collect attention.
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        for block in self.model.encoder.layers:
            # Extract attention matrix with shape (1, 197, 197).
            self.attentions.append(block.attn_weights)

        # Pass start_layer to the underlying calculation function.
        return self._rollout(start_layer)

    def _rollout(self, start_layer=0):
        # 1. Initialize the Rollout matrix as an identity matrix (diagonal is 1, representing self-attention).
        result = torch.eye(self.attentions[0].size(-1)).to(self.attentions[0].device)
        
        for attention in self.attentions[start_layer:]:
            # 2. Add residual connection (Identity) to simulate the retention of information in different layers.
            attention_fused = attention[0] + torch.eye(attention.size(-1)).to(attention.device)
            
            # 3. Normalize by row.
            attention_fused = attention_fused / attention_fused.sum(dim=-1, keepdim=True)
            
            # 4. Matrix multiplication and accumulation.
            result = torch.matmul(attention_fused, result)
            
        # 5. Extract the attention distribution of the Classification Token (CLS, index 0) to the remaining 196 image patches (index 1:).
        mask = result[0, 1:]
        
        # 6. Spatial reconstruction: 1D 196 restored to 2D 14x14 feature map.
        width = int(mask.size(0)**0.5)
        mask = mask.reshape(width, width).cpu().numpy()
        
        # 7. Normalize to the [0, 1] interval.
        mask = mask / np.max(mask)
        
        # 8. Upsample back to the original image resolution 224x224.
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_LINEAR)
        return mask