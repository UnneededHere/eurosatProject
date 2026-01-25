import ssl
from torchvision.datasets import EuroSAT

# Global config
dataRoot = './data'

def downloadOnly():
    print(f"Checking for EuroSAT data in {dataRoot}...")
    
    # This automatically handles downloading and extracting.
    # If the data is already there, it does nothing.
    # It does NOT touch the JSON files.
    EuroSAT(root=dataRoot, download=True)
    
    print("Download complete. You are ready to train.")

if __name__ == "__main__":
    # Fix for some university networks blocking SSL certificates
    ssl._create_default_https_context = ssl._create_unverified_context
    downloadOnly()