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

LABEL_COL   = "Label"
BENIGN_STR  = "BENIGN"

DROP_COLS = [
    "Flow ID", "Source IP", "Source Port", "Destination IP",
    "Destination Port", "Protocol", "Timestamp",
]


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df


def _load_csv_dir(directory: str) -> pd.DataFrame:
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

    before = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df.drop_duplicates(subset=numeric_cols)
    log.info(f"Duplicate removal: {before:,} -> {len(df):,} rows "
             f"({before - len(df):,} removed)")
    return df


def _encode_labels(df: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    df[LABEL_COL] = df[LABEL_COL].str.strip()
    df["label_int"] = df[LABEL_COL].map(label_map)
    before = len(df)
    df = df.dropna(subset=["label_int"])
    df["label_int"] = df["label_int"].astype(int)
    if len(df) < before:
        log.warning(f"Dropped {before - len(df)} rows with unrecognised labels.")
    return df


def preprocess(df: pd.DataFrame, label_map: dict, scaler=None, fit_scaler=True):

    df = _encode_labels(df, label_map)

    drop_existing = [c for c in DROP_COLS if c in df.columns]
    feature_cols  = [c for c in df.columns
                     if c not in drop_existing + [LABEL_COL, "label_int"]]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label_int"].values.astype(np.int64)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

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
        x = self.X[idx]                          
        p = x[:self.packet_dim]                  
        s = x[self.packet_dim:]                  

        
        p_seq = p.unsqueeze(0).expand(self.seq_len, -1)  
        s_seq = s.unsqueeze(0).expand(self.seq_len, -1)  

        return p_seq, s_seq, self.y[idx]


def get_dataloaders(train_split, val_split, test_split,
                    batch_size=config.BATCH_SIZE, num_workers=4):
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
