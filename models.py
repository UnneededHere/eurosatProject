import torch
import torch.nn as nn
from torchvision import models

def getModel(modelName, numClasses=10, pretrained=True):
    """
    Factory function to initialize models.
    Args:
        modelName: 'resnet50' or 'vit_b_16'
        numClasses: 10 (for EuroSAT)
    """
    print(f"Initializing {modelName} with pretrained={pretrained}...")

    if modelName == 'resnet50':
        # Load ResNet50
        # weights='DEFAULT' loads the best available ImageNet weights
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        
        # Replace the final fully connected layer (fc)
        # ResNet50 fc input features is 2048
        inFeatures = model.fc.in_features
        model.fc = nn.Linear(inFeatures, numClasses)
        
    elif modelName == 'vit_b_16':
        # Load Vision Transformer (Base, patch size 16)
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        
        # Replace the final head
        # ViT head structure is typically model.heads.head
        inFeatures = model.heads.head.in_features
        model.heads.head = nn.Linear(inFeatures, numClasses)
        
    else:
        raise ValueError(f"Model {modelName} not supported. Choose 'resnet50' or 'vit_b_16'.")

    return model

if __name__ == "__main__":
    # Quick sanity check
    m = getModel('resnet50')
    print("ResNet50 initialized successfully.")
    m = getModel('vit_b_16')
    print("ViT initialized successfully.")
