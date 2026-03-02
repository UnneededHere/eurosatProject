import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
import itertools

# Import your modules
from datasetFactory import getDataLoaders
from models import getModel
from trainer import trainOneEpoch, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batchSize', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model', type=str, nargs='+', default=['resnet50'], choices=['resnet50', 'vit_b_16'])
    parser.add_argument('--split', type=str, nargs='+', default=['train100'], help='Dataset split key (e.g. train10)', choices=['train10', 'train25', 'train50', 'train100'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.set_defaults(hyper=[])

    augMethods = parser.add_subparsers()
    
    aug_basic = augMethods.add_parser('basic', parents=[parser], conflict_handler='resolve')
    aug_basic.set_defaults(augMethodUsed='basic', hyper=[])
    
    aug_none = augMethods.add_parser('none', parents=[parser], conflict_handler='resolve')
    aug_none.set_defaults(augMethodUsed='none', hyper=[])
    
    aug_mixup = augMethods.add_parser('MixUp', parents=[parser], conflict_handler='resolve')
    aug_mixup.add_argument('--alpha', type=float, default=[1.0], nargs='+')
    aug_mixup.set_defaults(augMethodUsed='MixUp', hyper=['alpha'])
                           
    aug_randaug = augMethods.add_parser('RandAug', parents=[parser], conflict_handler='resolve')
    aug_randaug.add_argument('--num-ops', type=int, default=[2], nargs='+')
    aug_randaug.add_argument('--magnitude', type=int, default=[9], nargs='+')
    aug_randaug.add_argument('--num-magnitude-bins', type=int, default=[31], nargs='+') # probably don't bother with changing this
    aug_randaug.set_defaults(augMethodUsed='RandAug', hyper=['num_ops', 'magnitude', 'num_magnitude_bins'])
    
    aug_mixup_and_randaug = augMethods.add_parser('MixUpRandAug', parents=[parser], conflict_handler='resolve')
    aug_mixup_and_randaug.add_argument('--alpha', type=float, default=[1.0], nargs='+')
    aug_mixup_and_randaug.add_argument('--num-ops', type=int, default=[2], nargs='+')
    aug_mixup_and_randaug.add_argument('--magnitude', type=int, default=[9], nargs='+')
    aug_mixup_and_randaug.add_argument('--num-magnitude-bins', type=int, default=[31], nargs='+')
    aug_mixup_and_randaug.set_defaults(hyper=['num_ops', 'magnitude', 'num_magnitude_bins'])
    aug_mixup_and_randaug.set_defaults(augMethodUsed='MixUpRandAug', hyper=['alpha', 'num_ops', 'magnitude', 'num_magnitude_bins'])
    
    args = parser.parse_args()


    augMethod = args.augMethodUsed or 'basic'
    paramNames = args.hyper
    paramValues = [vars(args)[param] for param in paramNames]
    if paramNames:
        augMethodDetails = [(augMethod, dict(zip(paramNames, permut))) for permut in itertools.product(*paramValues)]
    else:
        augMethodDetails = [(augMethod, {})]


    # 1. Setup Data
    trainLoaders, valLoader, _ = getDataLoaders(splitKey=args.split, augMethod=augMethodDetails, batchSize=args.batchSize)
    with ProcessPoolExecutor() as pool:
        for model in args.model:
            for split, trainLoadersWithSplit in zip(args.split, trainLoaders):
                for (augName, augDeets), trainLoader in zip(augMethodDetails, trainLoadersWithSplit):
                    print(f"Starting experiment: {model} on {split} augmented with method {augMethod} using {args.device}")
                    pool.submit(hitIt, trainLoader, valLoader, args, "_".join([model, augName, *map(str, augDeets.values()), split]))

def hitIt(trainLoader, valLoader, args, dirName: str):
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
    results_dir = os.path.join('results', dirName)
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
            torch.save(model.state_dict(), os.path.join(results_dir, f"checkpoint.pth"))

    totalTime = (time.time() - startTime) / 60
    print(f"Training finished in {totalTime:.1f} minutes. Best Val Acc: {bestAcc:.2f}%")
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()
