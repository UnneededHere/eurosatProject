import os
import torch
import numpy as np
from torchvision.datasets import EuroSAT
from sklearn.model_selection import train_test_split
import json

# Configuration
dataRoot = './data'
seed = 42  # The answer to life, the universe, and reproducible research

def generateSplits():
    print("Downloading EuroSAT (this might take a moment)...")
    # We use the torchvision version for ease of use
    dataset = EuroSAT(root=dataRoot, download=True)
    targets = dataset.targets
    indices = np.arange(len(dataset))

    print(f"Total images: {len(indices)}")

    # 1. Create the Main Split (60% Train, 20% Val, 20% Test)
    # We use 'stratify' to ensure every class (River, Forest, etc.) is represented equally
    trainIdx, tempIdx, yTrain, yTemp = train_test_split(
        indices, targets, test_size=0.4, stratify=targets, random_state=seed
    )
    valIdx, testIdx, yVal, yTest = train_test_split(
        tempIdx, yTemp, test_size=0.5, stratify=yTemp, random_state=seed
    )

    # 2. Create the "Data Scarcity" Subsets for analysis
    # We will grab 10%, 25%, and 50% of the TRAINING set.
    # We nest them: 10% is inside 25%, which is inside 50%.
    
    # 50% subset
    train50Idx, _, _, _ = train_test_split(
        trainIdx, yTrain, train_size=0.5, stratify=yTrain, random_state=seed
    )
    
    # 25% subset (sampled from the full train set, ensuring stratification)
    train25Idx, _, _, _ = train_test_split(
        trainIdx, yTrain, train_size=0.25, stratify=yTrain, random_state=seed
    )
    
    # 10% subset
    train10Idx, _, _, _ = train_test_split(
        trainIdx, yTrain, train_size=0.10, stratify=yTrain, random_state=seed
    )

    # 3. Save these indices to a file
    # This JSON file will allow us to be consistent across experiments
    splits = {
        "train100": trainIdx.tolist(),
        "train50": train50Idx.tolist(),
        "train25": train25Idx.tolist(),
        "train10": train10Idx.tolist(),
        "val": valIdx.tolist(),
        "test": testIdx.tolist()
    }

    with open(os.path.join(dataRoot, 'splitIndices.json'), 'w') as f:
        json.dump(splits, f)
    
    print("Success! 'splitIndices.json' created in ./data")
    print(f"Train (100%): {len(trainIdx)}")
    print(f"Train (10%):  {len(train10Idx)}")
    print(f"Val:          {len(valIdx)}")
    print(f"Test:         {len(testIdx)}")

if __name__ == "__main__":
    generateSplits()