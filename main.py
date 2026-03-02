import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor

# Import your modules
from datasetFactory import getDataLoaders
from models import getModel
from trainer import trainOneEpoch, evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batchSize', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'vit_b_16'])
    parser.add_argument('--split', type=str, default='train100', help='Dataset split key (e.g. train10)', choices=['train10', 'train25', 'train50', 'train100'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    augMethods = parser.add_subparsers(dest='augMethodUsed')
    aug_basic = augMethods.add_parser('basic')
    aug_none = augMethods.add_parser('none')
    aug_mixup = augMethods.add_parser('MixUp')
    aug_mixup.add_argument('--alpha', type=float, default=[1.0], nargs='+')
    aug_randaug = augMethods.add_parser('RandAug')
    aug_randaug.add_argument('--num-ops', type=int, default=[2], nargs='+')
    aug_randaug.add_argument('--magnitude', type=int, default=[9], nargs='+')
    aug_randaug.add_argument('--num-magnitude-bins', type=int, default=31)
    args = parser.parse_args()

    print(args)
    augMethod = args.augMethodUsed or 'basic'
    if augMethod == 'RandAug':
        augMethodDetails = [('RandAug', {'num_ops': num_ops, 'magnitude': magnitude, 'num_magnitude_bins': args.num_magnitude_bins})
                            for num_ops in args.num_ops
                            for magnitude in args.magnitude
                            ]
    elif augMethod == 'MixUp':
        augMethodDetails = [('MixUp', {'alpha': alpha}) for alpha in args.alpha]
    else:
        augMethodDetails = [augMethod]

        
    print(f"Starting experiment: {args.model} on {args.split} augmented with method {augMethod} using {args.device}")

    # 1. Setup Data
    trainLoaders, valLoader, _ = getDataLoaders(splitKey=args.split, augMethod=augMethodDetails, batchSize=args.batchSize)
    with ProcessPoolExecutor() as pool:
        for trainLoader in trainLoaders:
            pool.submit(hitIt, trainLoader, valLoader, args)

def hitIt(trainLoader, valLoader, args):
    # 2. Setup Model
    model = getModel(args.model).to(args.device)
    model.compile()

    # 3. Setup Optimizer & Loss
    # ViTs often prefer AdamW, but SGD is fine for a baseline
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # Scheduler: Decay LR by factor of 0.1 every 7 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 4. Results directory and logger setup
    time_str = time.strftime('%Y%m%d-%H%M%S')
    lr_str = str(args.lr).replace('.', 'p')
    results_dir = os.path.join('results', f"{time_str}")
    os.makedirs(results_dir, exist_ok=True)

    # Save run parameters
    with open(os.path.join(results_dir, 'params.txt'), 'w') as pf:
        for k, v in vars(args).items():
            pf.write(f"{k}: {v}\n")

    logFile = os.path.join(results_dir, f"results.csv")
    with open(logFile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'TrainLoss', 'TrainAcc', 'ValLoss', 'ValAcc', 'Time'])

    # 5. Training Loop
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

        # Log to CSV
        with open(logFile, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, trainLoss, trainAcc, valLoss, valAcc, epochDur])

        # Save Checkpoint if best
        if valAcc > bestAcc:
            bestAcc = valAcc
            torch.save(model.state_dict(), os.path.join(results_dir, f"checkpoint_{args.model}_{args.split}.pth"))

    totalTime = (time.time() - startTime) / 60
    print(f"Training finished in {totalTime:.1f} minutes. Best Val Acc: {bestAcc:.2f}%")
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()
