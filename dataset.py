"""
dataset.py — MutaShield-Net Data Pipeline
Dataset: CICIDS2017 / CSE-CIC-IDS2018  (Section IV-A)
Download URL: https://www.unb.ca/cic/datasets/ids-2017.html
            https://www.kaggle.com/datasets/cicdataset/cicids2017
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

from config import (DATA_DIR, LABEL_COLUMN, NUM_FEATURES, TRAIN_RATIO,
                    VAL_RATIO, RANDOM_SEED, CAGRT, TRAIN, CLASS_NAMES)


# ─── Download helper ──────────────────────────────────────────────────────────
def download_cicids2017(target_dir: str = DATA_DIR):
    """
    CICIDS2017 download instructions.
    Official: https://www.unb.ca/cic/datasets/ids-2017.html
    Kaggle mirror (requires kaggle API key):
        kaggle datasets download -d cicdataset/cicids2017 -p <target_dir>
    """
    print("=" * 60)
    print("CICIDS2017 Dataset Download")
    print("=" * 60)
    print("Option 1 – Official (manual):")
    print("  https://www.unb.ca/cic/datasets/ids-2017.html")
    print("  Place all CSV files in:  data/cicids2017/")
    print()
    print("Option 2 – Kaggle CLI:")
    print("  pip install kaggle")
    print("  kaggle datasets download -d cicdataset/cicids2017 \\")
    print(f"    --path {os.path.join(target_dir, 'cicids2017')}")
    print("  unzip the archive in that folder")
    print("=" * 60)

    # Try kaggle download automatically if API is configured
    try:
        import kaggle
        save_path = os.path.join(target_dir, "cicids2017")
        os.makedirs(save_path, exist_ok=True)
        kaggle.api.dataset_download_files(
            "cicdataset/cicids2017", path=save_path, unzip=True
        )
        print(f"Downloaded to {save_path}")
    except Exception:
        print("Kaggle auto-download skipped — place CSV files manually.")


# ─── Loading & preprocessing ──────────────────────────────────────────────────
def load_cicids2017(data_dir: str = None) -> pd.DataFrame:
    """
    Load all CICIDS2017 CSV files in data/cicids2017/.
    Returns a cleaned DataFrame.  (Section IV-A)
    """
    data_dir = data_dir or os.path.join(DATA_DIR, "cicids2017")
    csv_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
    if not csv_files:
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Run download_cicids2017() first."
        )

    print(f"Found {len(csv_files)} CSV file(s). Loading…")
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            frames.append(df)
            print(f"  {os.path.basename(f)}: {len(df):,} rows")
        except Exception as e:
            print(f"  Skipping {f}: {e}")
    df = pd.concat(frames, ignore_index=True)
    print(f"Total loaded: {len(df):,} rows, {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame, fit: bool = True,
               scaler: StandardScaler = None,
               encoder: LabelEncoder  = None):
    """
    Section IV-A preprocessing:
      1. Drop inf / NaN
      2. Encode labels
      3. Select 80 numeric features
      4. StandardScaler normalisation
    Returns X (np.float32), y (np.int64), scaler, encoder
    """
    # ── 1. Clean ──────────────────────────────────────────────────────────────
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # ── 2. Labels ─────────────────────────────────────────────────────────────
    label_col = LABEL_COLUMN if LABEL_COLUMN in df.columns else "Label"
    if label_col not in df.columns:
        label_col = [c for c in df.columns if c.strip().lower() == "label"][0]

    df[label_col] = df[label_col].str.strip()
    if encoder is None:
        encoder = LabelEncoder()
        y = encoder.fit_transform(df[label_col].values)
    else:
        y = encoder.transform(df[label_col].values)

    # ── 3. Features ───────────────────────────────────────────────────────────
    drop_cols = [label_col] + [c for c in df.columns
                               if df[c].dtype == object]
    X_df = df.drop(columns=drop_cols, errors="ignore")
    # Keep at most 80 numeric columns
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = numeric_cols[:NUM_FEATURES]
    X = X_df[numeric_cols].values.astype(np.float32)

    # ── 4. Scale ──────────────────────────────────────────────────────────────
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X.astype(np.float32), y.astype(np.int64), scaler, encoder


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────
class CICIDSDataset(Dataset):
    """
    Returns (packet_features, semantic_features, label).
    The 80 features are split into two 40-dim domains as per Section III-C.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 seq_len: int = None, augment: bool = False):
        self.seq_len = seq_len or CAGRT["sequence_length"]
        self.augment = augment
        self.X = torch.from_numpy(X)           # (N, 80)
        self.y = torch.from_numpy(y)           # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]                        # (80,)
        if self.augment:
            x = x + 0.01 * torch.randn_like(x)  # light Gaussian noise augmentation

        # Section III-C: split into packet-domain and semantic-domain features
        half = x.shape[0] // 2
        p = x[:half]     # packet-domain   (40,)
        s = x[half:]     # semantic-domain (40,)

        # Expand to sequence dimension: repeat along time axis
        p_seq = p.unsqueeze(0).expand(self.seq_len, -1)   # (T, 40)
        s_seq = s.unsqueeze(0).expand(self.seq_len, -1)   # (T, 40)

        return p_seq, s_seq, self.y[idx]


# ─── Full pipeline builder ────────────────────────────────────────────────────
def build_dataloaders(data_dir: str = None, batch_size: int = None,
                      num_workers: int = None):
    """
    End-to-end: load → preprocess → split → DataLoaders.
    Returns (train_loader, val_loader, test_loader, scaler, encoder)
    """
    batch_size  = batch_size  or TRAIN["batch_size"]
    num_workers = num_workers or TRAIN["num_workers"]

    df = load_cicids2017(data_dir)
    X, y, scaler, encoder = preprocess(df, fit=True)

    n = len(y)
    idx = np.arange(n)
    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=y
    )
    idx_train, idx_val = train_test_split(
        idx_train,
        test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        random_state=RANDOM_SEED,
        stratify=y[idx_train],
    )

    print(f"Split — train: {len(idx_train):,}  val: {len(idx_val):,}  "
          f"test: {len(idx_test):,}")

    # Scale using training statistics only
    scaler_fit = StandardScaler().fit(X[idx_train])
    X_train = scaler_fit.transform(X[idx_train]).astype(np.float32)
    X_val   = scaler_fit.transform(X[idx_val]).astype(np.float32)
    X_test  = scaler_fit.transform(X[idx_test]).astype(np.float32)

    ds_train = CICIDSDataset(X_train, y[idx_train], augment=True)
    ds_val   = CICIDSDataset(X_val,   y[idx_val],   augment=False)
    ds_test  = CICIDSDataset(X_test,  y[idx_test],  augment=False)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=TRAIN["pin_memory"])
    train_loader = DataLoader(ds_train, shuffle=True,  **kwargs)
    val_loader   = DataLoader(ds_val,   shuffle=False, **kwargs)
    test_loader  = DataLoader(ds_test,  shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, scaler_fit, encoder


if __name__ == "__main__":
    download_cicids2017()
