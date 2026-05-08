"""
evaluate.py — MutaShield-Net Evaluation
Computes all metrics reported in Section IV (Table II–VI).
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config  import (BEST_CKPT, RESULTS_DIR, CONF_MATRIX_PATH,
                     ROC_CURVE_PATH, METRICS_CSV, CLASS_NAMES, RANDOM_SEED)
from dataset import build_dataloaders
from model   import MutaShieldNet
from utils   import set_seed


# ─── Adversarial attacks ──────────────────────────────────────────────────────

def fgsm_attack(model, p_seq, s_seq, labels, eps=0.03, device='cpu'):
    """FGSM adversarial example generation (Section IV-E)."""
    p_seq = p_seq.clone().detach().requires_grad_(True).to(device)
    s_seq = s_seq.clone().detach().requires_grad_(True).to(device)
    labels = labels.to(device)
    logits = model(p_seq, s_seq)
    loss   = F.cross_entropy(logits, labels)
    loss.backward()
    p_adv  = p_seq + eps * p_seq.grad.sign()
    s_adv  = s_seq + eps * s_seq.grad.sign()
    return p_adv.detach(), s_adv.detach()


def pgd_attack(model, p_seq, s_seq, labels, eps=0.03, alpha=0.007, steps=10, device='cpu'):
    """PGD adversarial attack (Section IV-E)."""
    p_adv = p_seq.clone().detach().to(device)
    s_adv = s_seq.clone().detach().to(device)
    labels = labels.to(device)
    for _ in range(steps):
        p_adv.requires_grad_(True); s_adv.requires_grad_(True)
        logits = model(p_adv, s_adv)
        loss   = F.cross_entropy(logits, labels)
        loss.backward()
        p_adv  = (p_adv + alpha * p_adv.grad.sign()).detach()
        s_adv  = (s_adv + alpha * s_adv.grad.sign()).detach()
        p_adv  = torch.max(torch.min(p_adv, p_seq + eps), p_seq - eps)
        s_adv  = torch.max(torch.min(s_adv, s_seq + eps), s_seq - eps)
    return p_adv, s_adv


# ─── Core evaluation function ─────────────────────────────────────────────────

@torch.no_grad()
def run_eval(model, loader, device, adversarial: str = None):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for p_seq, s_seq, labels in loader:
        p_seq, s_seq, labels = p_seq.to(device), s_seq.to(device), labels.to(device)

        if adversarial == "fgsm":
            p_seq, s_seq = fgsm_attack(model, p_seq, s_seq, labels, device=device)
        elif adversarial == "pgd":
            p_seq, s_seq = pgd_attack(model, p_seq, s_seq, labels, device=device)

        with torch.no_grad():
            logits = model(p_seq, s_seq)
            probs  = F.softmax(logits, dim=-1)
            preds  = logits.argmax(dim=-1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    y_true  = torch.cat(all_labels).numpy()
    y_pred  = torch.cat(all_preds).numpy()
    y_probs = torch.cat(all_probs).numpy()
    return y_true, y_pred, y_probs


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = cm == 0
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, linewidths=0.5, ax=ax, mask=mask)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, linewidths=0.5, ax=ax,
                cbar=False, alpha=0)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('MutaShield-Net Confusion Matrix — CICIDS2017', fontsize=13)
    plt.xticks(rotation=30, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def plot_roc_curve(y_true, y_probs, class_names, save_path):
    n_classes = y_probs.shape[1]
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, (cls, c) in enumerate(zip(class_names, colors)):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
        auc = roc_auc_score(y_bin, y_probs[:, i])
        ax.plot(fpr, tpr, color=c, lw=1.5, label=f'{cls} (AUC={auc:.4f})')
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('MutaShield-Net ROC Curves — CICIDS2017', fontsize=13)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved: {save_path}")


# ─── Main evaluation ──────────────────────────────────────────────────────────

def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if not os.path.exists(BEST_CKPT):
        raise FileNotFoundError(f"Checkpoint not found: {BEST_CKPT}\nRun train.py first.")
    ckpt  = torch.load(BEST_CKPT, map_location=device)
    model = MutaShieldNet().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch','?')}")

    # Load test data
    _, _, test_loader, _, encoder = build_dataloaders()
    class_names = list(encoder.classes_) if hasattr(encoder, 'classes_') else CLASS_NAMES

    # ── Clean evaluation ──────────────────────────────────────────────────────
    print("\n── Clean Evaluation ──")
    y_true, y_pred, y_probs = run_eval(model, test_loader, device)

    acc   = accuracy_score(y_true, y_pred) * 100
    prec  = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    rec   = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    f1    = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    # FPR: mean per-class FPR
    cm = confusion_matrix(y_true, y_pred)
    fp_per = cm.sum(axis=0) - np.diag(cm)
    tn_per = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    fpr = np.mean(fp_per / (fp_per + tn_per + 1e-8)) * 100

    try:
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted') * 100
    except Exception:
        auc = float('nan')

    print(f"\nResults (Table II — CICIDS2017):")
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  Precision : {prec:.2f}%")
    print(f"  Recall    : {rec:.2f}%")
    print(f"  F1-Score  : {f1:.2f}%")
    print(f"  AUC-ROC   : {auc:.2f}%")
    print(f"  FPR       : {fpr:.2f}%")

    print("\n" + classification_report(y_true, y_pred,
                                       target_names=class_names[:len(set(y_true))],
                                       zero_division=0))

    # ── Adversarial evaluation ────────────────────────────────────────────────
    print("\n── Adversarial Robustness (Table V) ──")
    for atk in ["fgsm", "pgd"]:
        yt, yp, _ = run_eval(model, test_loader, device, adversarial=atk)
        adv_acc = accuracy_score(yt, yp) * 100
        print(f"  {atk.upper():8s}: {adv_acc:.2f}%")

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame([{
        "Method": "MutaShield-Net",
        "Accuracy": round(acc, 2),
        "Precision": round(prec, 2),
        "Recall": round(rec, 2),
        "F1": round(f1, 2),
        "AUC": round(auc, 2),
        "FPR": round(fpr, 2),
    }])
    results_df.to_csv(METRICS_CSV, index=False)
    print(f"\nMetrics saved: {METRICS_CSV}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, class_names[:len(set(y_true))], CONF_MATRIX_PATH)
    plot_roc_curve(y_true, y_probs, class_names[:y_probs.shape[1]], ROC_CURVE_PATH)


if __name__ == "__main__":
    main()
