import torch
import json
import os
from torchvision.datasets import EuroSAT
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

def getDataloaders(dataRoot='./data', splitKey='train100', batchSize=32):
    """
    Args:
        splitKey: Which training set to use? 
                   Options: 'train100', 'train50', 'train25', 'train10'
    """
    
    # 1. Define Transforms
    # We must resize to 224 for standard Pre-trained models (ResNet/ViT)
    # We normalize using ImageNet statistics because we are using pre-trained models.
    imagenetStats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    trainTransform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(), # Basic Augmentation
        transforms.RandomVerticalFlip(),   # Good for satellite imagery
        transforms.ToTensor(),
        transforms.Normalize(*imagenetStats)
    ])

    # Validation/Test data should NOT be augmented (only resized and normalized)
    evalTransform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(*imagenetStats)
    ])

    # 2. Load the Dataset
    # We load it twice: once for training (with aug), once for eval (clean)
    trainDataset = EuroSAT(root=dataRoot, transform=trainTransform, download=True)
    valDataset   = EuroSAT(root=dataRoot, transform=evalTransform, download=True)

    # 3. Load the Indices we created in Step 2
    with open(os.path.join(dataRoot, 'splitIndices.json'), 'r') as f:
        indices = json.load(f)

    # 4. Create Subsets
    trainSubset = Subset(trainDataset, indices[splitKey])
    valSubset   = Subset(valDataset, indices['val'])
    testSubset  = Subset(valDataset, indices['test']) # Use valDataset source for transform consistency

    # 5. Create Loaders
    trainLoader = DataLoader(trainSubset, batch_size=batchSize, shuffle=True, num_workers=2)
    valLoader   = DataLoader(valSubset, batch_size=batchSize, shuffle=False, num_workers=2)
    testLoader  = DataLoader(testSubset, batch_size=batchSize, shuffle=False, num_workers=2)

    return trainLoader, valLoader, testLoader

# Quick test to see if it works
if __name__ == "__main__":
    t, v, te = getDataloaders(splitKey='train10')
    print(f"Loaded {len(t.dataset)} training images for 10% split.")
    batch = next(iter(t))
    print(f"Batch shape: {batch[0].shape}") # Should be [32, 3, 224, 224]