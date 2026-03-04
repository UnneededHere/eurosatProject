import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import csv
import os
import time
import itertools

from datasetFactory import getDataLoaders
from models import getModel
from trainer import trainOneEpoch, evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batchSize', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model', type=str, nargs='+', default=['resnet50'], choices=['resnet50', 'vit_b_16'])
    parser.add_argument('--split', type=str, nargs='+', default=['train100'], help='Dataset split key', choices=['train10', 'train25', 'train50', 'train100'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.set_defaults(hyper=[])

    augMethods = parser.add_subparsers(dest='augCommand')
    
    aug_basic = augMethods.add_parser('basic', conflict_handler='resolve')
    aug_basic.set_defaults(augMethodUsed='basic', hyper=[])
    
    aug_none = augMethods.add_parser('none', conflict_handler='resolve')
    aug_none.set_defaults(augMethodUsed='none', hyper=[])
    
    aug_mixup = augMethods.add_parser('MixUp', conflict_handler='resolve')
    aug_mixup.add_argument('--alpha', type=float, default=[0.2], nargs='+')
    aug_mixup.set_defaults(augMethodUsed='MixUp', hyper=['alpha'])
                           
    aug_randaug = augMethods.add_parser('RandAug', conflict_handler='resolve')
    aug_randaug.add_argument('--num_ops', type=int, default=[2], nargs='+')
    aug_randaug.add_argument('--magnitude', type=int, default=[9], nargs='+')
    aug_randaug.set_defaults(augMethodUsed='RandAug', hyper=['num_ops', 'magnitude'])
    
    aug_mixup_and_randaug = augMethods.add_parser('MixUpRandAug', conflict_handler='resolve')
    aug_mixup_and_randaug.add_argument('--alpha', type=float, default=[0.2], nargs='+')
    aug_mixup_and_randaug.add_argument('--num_ops', type=int, default=[2], nargs='+')
    aug_mixup_and_randaug.add_argument('--magnitude', type=int, default=[9], nargs='+')
    aug_mixup_and_randaug.set_defaults(augMethodUsed='MixUpRandAug', hyper=['alpha', 'num_ops', 'magnitude'])
    
    args = parser.parse_args()

    augMethod = getattr(args, 'augMethodUsed', 'none')
    paramNames = getattr(args, 'hyper', [])
    
    paramValues = [vars(args)[param] for param in paramNames]
    if paramNames:
        augMethodDetails = [(augMethod, dict(zip(paramNames, permut))) for permut in itertools.product(*paramValues)]
    else:
        augMethodDetails = [(augMethod, {})]

    for model_name in args.model:
        for split_name in args.split:
            for augName, augDeets in augMethodDetails:
                
                # Format the parameters for the filename (e.g., alpha0.2_ops2)
                if augDeets:
                    param_str = "_".join([f"{k}{v}" for k, v in augDeets.items()])
                else:
                    param_str = "base"
                    
                # The Golden Rule Naming Convention
                file_prefix = f"{model_name}-{augName}-{param_str}-{split_name}"
                
                print(f"\n=======================================================")
                print(f"Starting experiment: {file_prefix} on {args.device}")
                print(f"=======================================================")
                
                hitIt(model_name, split_name, augName, augDeets, args, file_prefix)

def hitIt(model_name, split_name, augName, augDeets, args, file_prefix):
    trainLoader, valLoader, _ = getDataLoaders(splitKey=split_name, batchSize=args.batchSize)

    model = getModel(model_name).to(args.device)

    # Setup Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    os.makedirs('results', exist_ok=True)
    logFile = os.path.join('results', f"{file_prefix}.csv")
    weightsFile = os.path.join('results', f"{file_prefix}.pth")

    with open(logFile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'TrainLoss', 'TrainAcc', 'ValLoss', 'ValAcc', 'Time'])

    bestAcc = 0.0
    startTime = time.time()

    for epoch in range(1, args.epochs + 1):
        epochStart = time.time()
        
        trainLoss, trainAcc = trainOneEpoch(model, trainLoader, criterion, optimizer, args.device)
        valLoss, valAcc = evaluate(model, valLoader, criterion, args.device)
        
        scheduler.step()
        epochDur = time.time() - epochStart

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {trainLoss:.4f} Acc: {trainAcc:.2f}% | "
              f"Val Loss: {valLoss:.4f} Acc: {valAcc:.2f}% | "
              f"Time: {epochDur:.1f}s")

        with open(logFile, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, trainLoss, trainAcc, valLoss, valAcc, epochDur])

        if valAcc > bestAcc:
            bestAcc = valAcc
            # Saving weights properly in the flat format
            torch.save(model.state_dict(), weightsFile)

    totalTime = (time.time() - startTime) / 60
    print(f"Finished {file_prefix} in {totalTime:.1f} mins. Best Val Acc: {bestAcc:.2f}%")

if __name__ == "__main__":
    main()