"""
evaluate.py — MutaShield-Net Evaluation
Reproduces Tables VI, VII, VIII from the paper.
Metrics: Accuracy, Precision, Recall, F1, FPR, AUC-ROC (Equations 31-35).
Adversarial robustness under FGSM, PGD, C&W, AutoAttack (Table VIII).
"""

import os
import logging
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "Times New Roman"
rcParams["font.size"]   = 11

import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize

import config
from dataset import build_dataset_cic2017, get_dataloaders, IDSDataset
from model import MutaShieldNet
from utils import compute_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_model(ckpt_path: str, n_classes: int, device: str):
    model = MutaShieldNet(n_classes=n_classes).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
             f"val_acc={ckpt.get('val_acc', 0):.2f}%")
    return model


@torch.no_grad()
def run_inference(model, loader, device):
    all_preds, all_labels, all_probs = [], [], []
    for p_seq, s_seq, labels in loader:
        p_seq  = p_seq.to(device, non_blocking=True)
        s_seq  = s_seq.to(device, non_blocking=True)
        logits, _ = model(p_seq, s_seq)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_probs, axis=0))


def fgsm_attack(model, p_seq, s_seq, labels, epsilon, device):
    """FGSM perturbation on input sequence features."""
    p_seq = p_seq.clone().detach().requires_grad_(True).to(device)
    s_seq = s_seq.clone().detach().requires_grad_(True).to(device)
    labels = labels.to(device)

    logits, _ = model(p_seq, s_seq)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    p_adv = p_seq + epsilon * p_seq.grad.sign()
    s_adv = s_seq + epsilon * s_seq.grad.sign()
    return p_adv.detach(), s_adv.detach()


@torch.no_grad()
def evaluate_adversarial(model, loader, device, epsilon=0.03):
    """
    Table VIII: evaluate accuracy under FGSM.
    Full PGD, C&W, AutoAttack require the torchattacks library.
    """
    correct = 0
    total   = 0
    for p_seq, s_seq, labels in loader:
        with torch.enable_grad():
            model.train()  # needed for grad through BN
            p_adv, s_adv = fgsm_attack(model, p_seq, s_seq, labels, epsilon, device)
        model.eval()
        logits, _ = model(p_adv, s_adv)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels.to(device)).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm  = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title("MutaShield-Net — Normalised Confusion Matrix (CIC-IDS2017)",
                 fontname="Times New Roman", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    log.info(f"Confusion matrix saved: {save_path}")


def plot_roc_curves(labels, probs, n_classes, save_path):
    """One-vs-rest ROC curves for each class."""
    y_bin = label_binarize(labels, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.tab20(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"Class {i} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontname="Times New Roman")
    ax.set_ylabel("True Positive Rate",  fontname="Times New Roman")
    ax.set_title("MutaShield-Net — ROC Curves (CIC-IDS2017)",
                 fontname="Times New Roman", fontsize=13)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    log.info(f"ROC curve saved: {save_path}")


def print_results_table(metrics, dataset_name="CIC-IDS2017"):
    """Reproduce the format of Table VI from the paper."""
    print(f"\n{'='*65}")
    print(f"  Results on {dataset_name}")
    print(f"{'='*65}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}%")
    print(f"  Precision : {metrics['precision']:.4f}%")
    print(f"  Recall    : {metrics['recall']:.4f}%")
    print(f"  F1-Score  : {metrics['f1']:.4f}%")
    print(f"  FPR       : {metrics['fpr']:.4f}%")
    print(f"  AUC-ROC   : {metrics.get('auc', 0.0):.4f}%")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    default=config.BEST_CKPT)
    parser.add_argument("--dataset", default="cic2017")
    parser.add_argument("--adv",     action="store_true",
                        help="Run FGSM adversarial accuracy evaluation")
    args = parser.parse_args()

    device = config.DEVICE

    # ── Load data ────────────────────────────────────────────────────────
    splits = build_dataset_cic2017()
    _, _, test_loader = get_dataloaders(
        splits["train"], splits["val"], splits["test"]
    )
    n_classes = len(np.unique(splits["train"][1]))

    # ── Load model ───────────────────────────────────────────────────────
    if not os.path.exists(args.ckpt):
        log.error(f"Checkpoint not found: {args.ckpt}. Run train.py first.")
        return
    model = load_model(args.ckpt, n_classes, device)

    # ── Inference ────────────────────────────────────────────────────────
    preds, labels, probs = run_inference(model, test_loader, device)

    # ── Metrics ─────────────────────────────────────────────────────────
    metrics = compute_metrics(labels, preds, probs=probs, n_classes=n_classes)
    print_results_table(metrics)

    # ── Per-class report (Table VII format) ─────────────────────────────
    print(classification_report(labels, preds, digits=4))

    # ── Confusion matrix ─────────────────────────────────────────────────
    cm_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    class_names = [str(i) for i in range(n_classes)]
    plot_confusion_matrix(labels, preds, class_names, cm_path)

    # ── ROC curves ───────────────────────────────────────────────────────
    roc_path = os.path.join(config.RESULTS_DIR, "roc_curves.png")
    plot_roc_curves(labels, probs, n_classes, roc_path)

    # ── Adversarial evaluation (FGSM) ───────────────────────────────────
    if args.adv:
        fgsm_acc = evaluate_adversarial(model, test_loader, device, epsilon=0.03)
        log.info(f"FGSM adversarial accuracy: {fgsm_acc:.2f}% "
                 f"(paper reports 94.23% for MutaShield-Net)")


if __name__ == "__main__":
    main()
