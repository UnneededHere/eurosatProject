import torch
import json
import os
from typing import Iterable, Any, Tuple
from itertools import starmap
from torchvision.datasets import EuroSAT
from torchvision import transforms
from torchvision.transforms import v2  # Required for MixUp
from torch.utils.data import DataLoader, Subset, default_collate

def getDataLoaders(dataRoot='./data',
                   splitKey: str|Iterable[str]='train100',
                   augMethod: str|Iterable[str]|Iterable[Tuple[str, dict[str, Any]]]='basic',
                   batchSize=32
                   ):
    """
    Args:
        splitKey: Which training set to use? Options: 'train100', 'train50', 'train25', 'train10'
        augMethod: which augmentation method to use? Options: 'none', 'basic', 'MixUp', 'RandAug', 'MixUpRandAug'
    """

    # 1. Input Parsing & Unboxing tracking
    is_single_aug = False
    if isinstance(augMethod, str):
        augMethods = [(augMethod, {})]
        is_single_aug = True
    elif isinstance(augMethod, tuple) and isinstance(augMethod[0], str):
        augMethods = [augMethod]
        is_single_aug = True
    elif isinstance(augMethod, list) and isinstance(augMethod[0], str):
        augMethods = [(method, {}) for method in augMethod]
    else:
        augMethods = augMethod
        
    is_single_split = isinstance(splitKey, str)
    splitKeys = [splitKey] if is_single_split else splitKey
    
    imagenetStats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # 2. Transform & Collation Factory
    def get_transform_and_collation(method: str, methodArgs: dict):
        collation = default_collate
        image_transforms = []
        
        # A. Image-level transforms (Applied per image)
        if method == 'none':
            pass
        elif method == 'basic' or method == 'MixUp':
            image_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ]
        elif method == 'RandAug':
            image_transforms = [transforms.RandAugment(fill=128, **methodArgs)]
        elif method == 'MixUpRandAug':
            # Extract only the RandAug parameters
            randArgs = {k: v for k, v in methodArgs.items() if k != 'alpha'}
            image_transforms = [transforms.RandAugment(fill=128, **randArgs)]
        else:
            raise ValueError("Unrecognised augmentation strategy: " + method)
            
        # B. Batch-level transforms (Applied per batch in the collate_fn)
        if 'MixUp' in method:
            mixArgs = {'alpha': methodArgs['alpha']} if 'alpha' in methodArgs else {}
            # Initialize MixUp outside the lambda so it doesn't re-init on every batch
            mixup_obj = v2.MixUp(num_classes=10, **mixArgs)
            collation = lambda batch: mixup_obj(*default_collate(batch))

        # C. Compose the final image pipeline
        full_transform = transforms.Compose([
            transforms.Resize(224),
            *image_transforms,
            transforms.ToTensor(),
            transforms.Normalize(*imagenetStats)
        ])
        
        return full_transform, collation
                                   
    trainTransforms = list(starmap(get_transform_and_collation, augMethods))
    evalTransform, evalCollation = get_transform_and_collation('none', {})

    # 3. Safe DataLoader Builder
    def makeLoader(indices, transform, collation, shuffle):
        dataset = EuroSAT(root=dataRoot, transform=transform, download=True)
        subset = Subset(dataset, indices)
        # FIX: collate_fn is now correctly passed to the DataLoader
        loader = DataLoader(subset, batch_size=batchSize, shuffle=shuffle, num_workers=2, collate_fn=collation)
        return loader

    # 4. Load the Indices
    with open(os.path.join(dataRoot, 'splitIndices.json'), 'r') as f:
        indices = json.load(f)
    
    # 5. Build the Loaders
    trainLoaders = [[makeLoader(indices[key], t, c, shuffle=True) for t, c in trainTransforms] for key in splitKeys]
    
    # Eval loaders must NOT be shuffled
    valLoader = makeLoader(indices['val'], evalTransform, evalCollation, shuffle=False)
    testLoader = makeLoader(indices['test'], evalTransform, evalCollation, shuffle=False)

    # 6. Unboxing logic to match input shapes
    if is_single_aug:
        trainLoaders = [loaders[0] for loaders in trainLoaders]
    if is_single_split:
        trainLoaders = trainLoaders[0]
        
    return trainLoaders, valLoader, testLoader

# Quick test to see if it works
if __name__ == "__main__":
    ts, v, te = getDataLoaders(splitKey='train10', augMethod=['basic', 'MixUp'])
    print(f"Loaded {len(ts[0].dataset)} training images for 10% split (Basic).")
    print(f"Loaded {len(ts[1].dataset)} training images for 10% split (MixUp).")
    
    batch = next(iter(ts[0]))
    print(f"Basic Batch shape: {batch[0].shape}, Targets shape: {batch[1].shape}") 
    
    batch_mixup = next(iter(ts[1]))
    print(f"MixUp Batch shape: {batch_mixup[0].shape}, Targets shape: {batch_mixup[1].shape}")