import torch
import json
import os
from torchvision.datasets import EuroSAT
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

def getDataLoaders(dataRoot='./data', splitKey='train100', augMethod = 'basic', batchSize=32):
    """
    Args:
        splitKey: Which training set to use? 
                   Options: 'train100', 'train50', 'train25', 'train10'
        augMethod: which augmentation method to use? Note, 'none' is always applied to eval and test data
                   Options: 'none', 'basic', 'Mixup', 'AugReg'
    """

    if isinstance(augMethod, str):
        augMethods = [augMethod]
    else:
        augMethods = augMethod
        
    if isinstance(splitKey, str):
        splitKeys = [splitKey]
    else:
        splitKeys = splitKey
    
    # 1. Define Transforms
    # We must resize to 224 for standard Pre-trained models (ResNet/ViT)
    # We normalize using ImageNet statistics because we are using pre-trained models.
    imagenetStats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def transformMethod(method):
        if method   == 'none':
            transform = transforms.Compose([])
        elif method == 'basic':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(), # Basic Augmentation
                transforms.RandomVerticalFlip()    # Good for satellite imagery
            ])
        elif method == 'Mixup':
            raise NotImplementedError
        elif method == 'AugReg':
            raise NotImplementedError
        
        return transforms.Compose([
            transforms.Resize(224),
            transform,
            transforms.ToTensor(),
            transforms.Normalize(*imagenetStats)
        ])
                                   

    trainTransforms = list(map(transformMethod, augMethods))

    # Validation/Test data should NOT be augmented (only resized and normalized)
    evalTransform = transformMethod('none')


    def makeLoader(indices, transform):
        dataset = EuroSAT(root=dataRoot, transform=transform, download=True)
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batchSize, shuffle=True, num_workers=2)
        return loader

    # 3. Load the Indices we created in Step 2
    with open(os.path.join(dataRoot, 'splitIndices.json'), 'r') as f:
        indices = json.load(f)
    
    trainLoaders = [[makeLoader(indices[key], method) for method in trainTransforms] for key in splitKeys]
    valLoader = makeLoader(indices['val'], evalTransform)
    testLoader = makeLoader(indices['test'], evalTransform) # Use valDataset source for transform consistency

    if isinstance(augMethod, str):
        trainLoaders = [loader[0] for loader in trainLoaders]
    if isinstance(splitKey, str):
        trainLoaders = trainLoaders[0]
    return trainLoaders, valLoader, testLoader

# Quick test to see if it works
if __name__ == "__main__":
    t, v, te = getDataloaders(splitKey='train10')
    print(f"Loaded {len(t.dataset)} training images for 10% split.")
    batch = next(iter(t))
    print(f"Batch shape: {batch[0].shape}") # Should be [32, 3, 224, 224]
