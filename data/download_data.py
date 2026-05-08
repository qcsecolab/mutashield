"""
download_data.py — Download CICIDS2017 and CSE-CIC-IDS2018 datasets.
Place this script in the data/ directory and run it.

Official pages:
  CICIDS2017:     https://www.unb.ca/cic/datasets/ids-2017.html
  CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html

Kaggle mirror (recommended for automated download):
  https://www.kaggle.com/datasets/cicdataset/cicids2017
"""

import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def download_via_kaggle(dataset: str, out_dir: str):
    """Download a Kaggle dataset using the kaggle CLI."""
    try:
        import kaggle
        os.makedirs(out_dir, exist_ok=True)
        print(f"Downloading {dataset} → {out_dir}")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset, path=out_dir, unzip=True)
        print("Done!")
    except ImportError:
        print("kaggle package not found. Install with:  pip install kaggle")
        print("Then set up ~/.kaggle/kaggle.json with your API key.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        manual_instructions()


def manual_instructions():
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print()
    print("Option A — CICIDS2017 (Official, ~2.2GB):")
    print("  1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html")
    print("  2. Click 'Download' and extract CSV files")
    print(f"  3. Place all CSV files in:  {os.path.join(DATA_DIR, 'cicids2017/')}")
    print()
    print("Option B — Kaggle CLI (Automated):")
    print("  pip install kaggle")
    print("  # Place your kaggle.json in ~/.kaggle/")
    print("  kaggle datasets download -d cicdataset/cicids2017 \\")
    print(f"    --path {os.path.join(DATA_DIR, 'cicids2017')}")
    print()
    print("Expected folder structure after download:")
    print(f"  {DATA_DIR}/")
    print("  └── cicids2017/")
    print("      ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    print("      ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    print("      ├── Friday-WorkingHours-Morning.pcap_ISCX.csv")
    print("      ├── Monday-WorkingHours.pcap_ISCX.csv")
    print("      ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
    print("      ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
    print("      ├── Tuesday-WorkingHours.pcap_ISCX.csv")
    print("      └── Wednesday-workingHours.pcap_ISCX.csv")
    print("="*60)


if __name__ == "__main__":
    choice = input("Download via Kaggle API? [y/N]: ").strip().lower()
    if choice == 'y':
        out = os.path.join(DATA_DIR, "cicids2017")
        download_via_kaggle("cicdataset/cicids2017", out)
    else:
        manual_instructions()
