import pandas as pd
import matplotlib.pyplot as plt
import sys

def plotLog(csvFile):
    try:
        data = pd.read_csv(csvFile)
    except FileNotFoundError:
        print(f"Error: Could not find {csvFile}. Run main.py first!")
        return

    plt.figure(figsize=(10, 5))

    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(data['Epoch'], data['TrainLoss'], label='Train Loss')
    plt.plot(data['Epoch'], data['ValLoss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Subplot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(data['Epoch'], data['TrainAcc'], label='Train Acc')
    plt.plot(data['Epoch'], data['ValAcc'], label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(csvFile.replace('.csv', '.png'))
    print(f"Plot saved to {csvFile.replace('.csv', '.png')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plotResults.py results_resnet50_train100.csv")
    else:
        plotLog(sys.argv[1])