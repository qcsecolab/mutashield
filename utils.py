"""
utils.py — MutaShield-Net Utilities
Helper functions for metrics, visualization, seeding, etc.
"""

import os, random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score)


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Metric computation ───────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_probs: np.ndarray = None) -> dict:
    """
    Compute all metrics reported in Table II & V of the paper.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # False Positive Rate (Eq. 29)
    cm      = confusion_matrix(y_true, y_pred)
    fp_sum  = cm.sum(axis=0) - np.diag(cm)
    tn_sum  = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    fpr     = np.mean(fp_sum / (fp_sum + tn_sum + 1e-8))

    metrics = dict(accuracy=acc, precision=prec, recall=rec,
                   f1=f1, fpr=fpr)

    if y_probs is not None:
        try:
            auc = roc_auc_score(y_true, y_probs,
                                multi_class='ovr', average='weighted')
            metrics["auc"] = auc
        except Exception:
            metrics["auc"] = float('nan')

    return metrics


# ─── Logging ─────────────────────────────────────────────────────────────────

def log_epoch(epoch: int, metrics_tr: dict, metrics_val: dict,
              log_file: str = "results/train_log.csv"):
    """Append one row to the CSV training log."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    header = not os.path.exists(log_file)
    with open(log_file, 'a') as f:
        if header:
            f.write("epoch,tr_loss,tr_acc,val_loss,val_acc,val_f1,val_fpr\n")
        f.write(f"{epoch},"
                f"{metrics_tr.get('loss',0):.6f},"
                f"{metrics_tr.get('accuracy',0):.6f},"
                f"{metrics_val.get('loss',0):.6f},"
                f"{metrics_val.get('accuracy',0):.6f},"
                f"{metrics_val.get('f1',0):.6f},"
                f"{metrics_val.get('fpr',0):.6f}\n")


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_training_curves(log_file: str, save_path: str = "results/training_curves.png"):
    """Plot loss and accuracy from training log CSV."""
    import pandas as pd
    df = pd.read_csv(log_file)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df['epoch'], df['tr_loss'],  label='Train Loss',    lw=2)
    axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', lw=2, ls='--')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(df['epoch'], df['tr_acc']*100,  label='Train Acc', lw=2)
    axes[1].plot(df['epoch'], df['val_acc']*100, label='Val Acc',   lw=2, ls='--')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {save_path}")


def plot_mss_history(mss_history: list, save_path: str = "results/mss_history.png"):
    """Plot MSS evolution (Figure 7 equivalent)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    gens = np.arange(1, len(mss_history) + 1) * 5
    ax.plot(gens, mss_history, 'b-o', ms=5, lw=2, label='MSS')
    ax.axhline(0.03, color='r', ls='--', label='Convergence threshold')
    ax.set_xlabel('Training Epoch'); ax.set_ylabel('Mutation Survival Score')
    ax.set_title('AMFEL — MSS Evolution during Training')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Augmentation pipeline ────────────────────────────────────────────────────

class GaussianNoise:
    """Light augmentation: add Gaussian noise to feature vector."""
    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.std * torch.randn_like(x)


class FeatureDropout:
    """Randomly zero out features (simulates missing sensors)."""
    def __init__(self, p: float = 0.05):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p))
        return x * mask


# ─── Model summary ───────────────────────────────────────────────────────────

def model_summary(model: torch.nn.Module) -> str:
    lines = [f"{'Layer':<40} {'Params':>12}"]
    lines.append("-" * 55)
    total = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        lines.append(f"{name:<40} {n:>12,}")
    lines.append("-" * 55)
    lines.append(f"{'TOTAL':<40} {total:>12,}")
    return "\n".join(lines)


# ─── MSS Computation (standalone, for post-hoc analysis) ─────────────────────

def compute_mss(model, test_loader, device, n_batches: int = 10) -> float:
    """
    Eq. 24: Compute Mutation Survival Score over test batches.
    """
    model.eval()
    survived = 0; total = 0
    for i, (p_seq, s_seq, labels) in enumerate(test_loader):
        if i >= n_batches:
            break
        p_seq = p_seq.to(device); s_seq = s_seq.to(device)
        x_flat = torch.cat([p_seq[:,0,:], s_seq[:,0,:]], dim=-1)
        mutants = model.smoe.generate_pool(x_flat)
        with torch.no_grad():
            for mut in mutants:
                half = mut.shape[-1] // 2
                T    = p_seq.shape[1]
                p_m  = mut[:, :half].unsqueeze(1).expand_as(p_seq)
                s_m  = mut[:, half:].unsqueeze(1).expand_as(s_seq)
                preds = model.detector(p_m, s_m).argmax(dim=-1)
                survived += (preds == 0).sum().item()
                total    += preds.numel()
    return survived / max(total, 1)
