"""
dataset.py — MutaShield-Net Dataset Handling
Dataset: CIC-IDS2017 and CSE-CIC-IDS2018
Source: Canadian Institute for Cybersecurity, University of New Brunswick
CIC-IDS2017 URL: https://www.unb.ca/cic/datasets/ids-2017.html
CSE-CIC-IDS2018 URL: https://www.unb.ca/cic/datasets/ids-2018.html

Preprocessing follows Section IV-A-1 and IV-A-2:
  - 80 CICFlowMeter features
  - Duplicate removal by 5-tuple matching
  - Temporally stratified 70/10/20 split (days 1-3 train, day 4 val, day 5 test)
  - Class imbalance: benign traffic >80% of both corpora
"""

import os
import glob
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─── CICFlowMeter column name normalisations ────────────────────────────────
# The CSVs from CIC have inconsistent leading/trailing spaces; strip all.

LABEL_COL   = "Label"
BENIGN_STR  = "BENIGN"

# Features to drop (non-numeric or constant in CIC exports)
DROP_COLS = [
    "Flow ID", "Source IP", "Source Port", "Destination IP",
    "Destination Port", "Protocol", "Timestamp",
]


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df


def _load_csv_dir(directory: str) -> pd.DataFrame:
    """Load all CSV files from a directory and concatenate."""
    paths = sorted(glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in {directory}. "
            "Download the dataset from https://www.unb.ca/cic/datasets/ids-2017.html "
            "and place all CICFlowMeter CSV files under data/CIC-IDS2017/raw/"
        )
    log.info(f"Loading {len(paths)} CSV file(s) from {directory}")
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding="latin-1", low_memory=False)
        df = _strip_columns(df)
        frames.append(df)
        log.info(f"  {os.path.basename(p)}: {len(df):,} rows")
    return pd.concat(frames, ignore_index=True)


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Section IV-A-2: remove duplicates by matching 5-tuples with identical
    feature vectors. CIC CSVs do not export raw 5-tuple fields, so we
    deduplicate on the full 80-feature numeric vector instead.
    """
    before = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df.drop_duplicates(subset=numeric_cols)
    log.info(f"Duplicate removal: {before:,} -> {len(df):,} rows "
             f"({before - len(df):,} removed)")
    return df


def _encode_labels(df: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    df[LABEL_COL] = df[LABEL_COL].str.strip()
    # Map known labels; unknown labels get -1 and are dropped
    df["label_int"] = df[LABEL_COL].map(label_map)
    before = len(df)
    df = df.dropna(subset=["label_int"])
    df["label_int"] = df["label_int"].astype(int)
    if len(df) < before:
        log.warning(f"Dropped {before - len(df)} rows with unrecognised labels.")
    return df


def preprocess(df: pd.DataFrame, label_map: dict, scaler=None, fit_scaler=True):
    """
    Full preprocessing pipeline:
    1. Encode labels
    2. Drop non-feature columns
    3. Replace inf/NaN with 0
    4. Standard-scale features
    Returns: X (np.ndarray), y (np.ndarray), fitted scaler
    """
    df = _encode_labels(df, label_map)

    drop_existing = [c for c in DROP_COLS if c in df.columns]
    feature_cols  = [c for c in df.columns
                     if c not in drop_existing + [LABEL_COL, "label_int"]]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label_int"].values.astype(np.int64)

    # Replace inf and NaN — Section IV-A-1
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip extreme values to prevent scaler blow-up
    X = np.clip(X, -1e9, 1e9)

    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    log.info(f"Feature matrix: {X.shape}, classes: {np.unique(y)}")
    return X, y, scaler


def temporally_split(X, y, timestamps=None):
    """
    Section IV-A-2: temporally stratified protocol.
    When timestamps are unavailable (raw CSVs are already day-ordered),
    we approximate by positional ordering — first 70% train,
    next 10% val, last 20% test.
    """
    n = len(X)
    n_train = int(n * config.TRAIN_RATIO)
    n_val   = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_val], y[n_train:n_val]
    X_test,  y_test  = X[n_val:],        y[n_val:]

    log.info(f"Split: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_dataset_cic2017(raw_dir=config.CIC2017_RAW_DIR,
                           proc_dir=config.CIC2017_PROC_DIR,
                           force_rebuild=False):
    """
    Full pipeline for CIC-IDS2017.
    Returns train/val/test splits plus the fitted scaler.
    Caches processed arrays to proc_dir.
    """
    cache_path = os.path.join(proc_dir, "cic2017_splits.pkl")
    if os.path.exists(cache_path) and not force_rebuild:
        log.info(f"Loading cached splits from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    df = _load_csv_dir(raw_dir)
    df = _remove_duplicates(df)
    X, y, scaler = preprocess(df, config.CIC2017_LABEL_MAP)

    train, val, test = temporally_split(X, y)

    payload = {"train": train, "val": val, "test": test, "scaler": scaler}
    os.makedirs(proc_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    log.info(f"Saved processed splits to {cache_path}")
    return payload


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────

class IDSDataset(Dataset):
    """
    PyTorch Dataset wrapping preprocessed CICFlowMeter feature arrays.

    The CA-GRT model expects a sequence of length T_seq (Section IV-A-2, p95
    of flow lengths = 100). For flow-level data each sample is a single vector;
    we replicate it to form a dummy sequence so the BiGRU encoder receives
    a proper (batch, seq_len, feat_dim) tensor. In production, replace with
    actual per-packet sequences if raw PCAP access is available.

    Packet-domain features  p: first d_p dims of the 80-feature vector
    Semantic-domain features s: remaining d_s dims                       (Eq. 9-10)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 seq_len: int = config.CAGRT_SEQ_LEN,
                 packet_dim: int = config.CAGRT_PACKET_FEAT_DIM,
                 semantic_dim: int = config.CAGRT_SEMANTIC_FEAT_DIM):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.seq_len    = seq_len
        self.packet_dim = packet_dim
        self.semantic_dim = semantic_dim

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]                          # (80,)
        p = x[:self.packet_dim]                  # (d_p,)
        s = x[self.packet_dim:]                  # (d_s,)

        # Replicate single feature vector into sequence: (seq_len, feat_dim)
        p_seq = p.unsqueeze(0).expand(self.seq_len, -1)  # (T, d_p)
        s_seq = s.unsqueeze(0).expand(self.seq_len, -1)  # (T, d_s)

        return p_seq, s_seq, self.y[idx]


def get_dataloaders(train_split, val_split, test_split,
                    batch_size=config.BATCH_SIZE, num_workers=4):
    """Return train, val, and test DataLoaders."""
    X_train, y_train = train_split
    X_val,   y_val   = val_split
    X_test,  y_test  = test_split

    train_ds = IDSDataset(X_train, y_train)
    val_ds   = IDSDataset(X_val,   y_val)
    test_ds  = IDSDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, val_loader, test_loader
