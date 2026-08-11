import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "Times New Roman"
rcParams["font.size"]   = 11

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

import config


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def compute_metrics(labels: np.ndarray, preds: np.ndarray,
                    probs: np.ndarray = None,
                    n_classes: int = None) -> dict:

    acc  = 100.0 * accuracy_score(labels, preds)
    prec = 100.0 * precision_score(labels, preds,
                                   average="weighted", zero_division=0)
    rec  = 100.0 * recall_score(labels, preds,
                                average="weighted", zero_division=0)
    f1   = 100.0 * f1_score(labels, preds,
                            average="weighted", zero_division=0)

    cm   = confusion_matrix(labels, preds)
    fp   = cm.sum(axis=0) - np.diag(cm)
    fn   = cm.sum(axis=1) - np.diag(cm)
    tp   = np.diag(cm)
    tn   = cm.sum() - (fp + fn + tp)
    fpr_per_class = fp / np.maximum(fp + tn, 1)
    fpr  = 100.0 * float(np.average(fpr_per_class,
                                    weights=cm.sum(axis=1)))

    auc_score = 0.0
    if probs is not None and n_classes is not None:
        try:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(labels, classes=list(range(n_classes)))
            if n_classes == 2:
                auc_score = 100.0 * roc_auc_score(labels, probs[:, 1])
            else:
                auc_score = 100.0 * roc_auc_score(y_bin, probs,
                                                   multi_class="ovr",
                                                   average="macro")
        except Exception:
            auc_score = 0.0

    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "fpr":       fpr,
        "auc":       auc_score,
    }


def plot_training_curves(train_losses, val_losses,
                         train_accs, val_accs,
                         save_dir: str):

    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(epochs, train_losses, "b-",  lw=1.5, label="Training Loss")
    axes[0].plot(epochs, val_losses,   "r--", lw=1.5, label="Validation Loss")
    axes[0].set_xlabel("Epoch", fontname="Times New Roman")
    axes[0].set_ylabel("Loss",  fontname="Times New Roman")
    axes[0].set_title("Training and Validation Loss",
                      fontname="Times New Roman", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, train_accs, "b-",  lw=1.5, label="Training Accuracy")
    axes[1].plot(epochs, val_accs,   "r--", lw=1.5, label="Validation Accuracy")
    axes[1].set_xlabel("Epoch",    fontname="Times New Roman")
    axes[1].set_ylabel("Accuracy (%)", fontname="Times New Roman")
    axes[1].set_title("Training and Validation Accuracy",
                      fontname="Times New Roman", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig02_training_curves.png")
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
    return path


def plot_mss_evolution(mss_history: list, drs_history: list, save_dir: str):

    gens = list(range(1, len(mss_history) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(gens, mss_history, "r-", lw=2, label="MSS")
    axes[0].fill_between(gens, mss_history, alpha=0.25, color="red")
    axes[0].set_xlabel("Generation", fontname="Times New Roman")
    axes[0].set_ylabel("MSS",        fontname="Times New Roman")
    axes[0].set_title("(a) Mutation Survival Score",
                      fontname="Times New Roman", fontsize=12)
    axes[0].set_ylim(0, 0.5)
    axes[0].grid(True, alpha=0.3)

    if drs_history:
        axes[1].plot(gens[:len(drs_history)], drs_history, "b-", lw=2, label="DRS")
        axes[1].set_xlabel("Generation", fontname="Times New Roman")
        axes[1].set_ylabel("DRS",        fontname="Times New Roman")
        axes[1].set_title("(b) Detector Robustness Score",
                          fontname="Times New Roman", fontsize=12)
        axes[1].set_ylim(0.5, 1.0)
        axes[1].grid(True, alpha=0.3)

    plt.suptitle("Evolution of MSS and DRS during AMFEL Co-Evolutionary Optimisation",
                 fontname="Times New Roman", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "fig03_mss_drs_evolution.png")
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
    return path


def plot_mutant_taxonomy(save_dir: str):

    families  = ["SQL Inj.", "XSS", "Cmd Inj.", "Path Trav.", "DoS", "Recon.", "Auth Bypass"]
    stillborn = [2.8, 3.1, 4.2, 2.9, 3.5, 2.6, 3.8]
    trivial   = [58.1, 56.4, 54.7, 60.3, 63.2, 55.8, 52.9]
    hard2kill = [30.2, 27.9, 29.1, 26.4, 24.8, 31.0, 31.4]
    live      = [8.9, 12.6, 12.0, 10.4, 8.5, 10.6, 11.9]

    x    = np.arange(len(families))
    width = 0.6
    fig, ax = plt.subplots(figsize=(12, 6))

    b1 = ax.bar(x, stillborn, width, label="Stillborn", color="#d9534f")
    b2 = ax.bar(x, trivial,   width, bottom=stillborn, label="Trivial", color="#5bc0de")
    b3 = ax.bar(x, hard2kill, width,
                bottom=[a+b for a,b in zip(stillborn, trivial)],
                label="Hard-to-Kill", color="#f0ad4e")
    b4 = ax.bar(x, live,      width,
                bottom=[a+b+c for a,b,c in zip(stillborn, trivial, hard2kill)],
                label="Live", color="#5cb85c")

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontname="Times New Roman", rotation=20, ha="right")
    ax.set_ylabel("Proportion (%)", fontname="Times New Roman")
    ax.set_title("Mutant Classification by Taxonomy Category on CIC-IDS2017 (Table V)",
                 fontname="Times New Roman", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig04_mutant_taxonomy.png")
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
    return path


def plot_performance_comparison(save_dir: str):

    methods = ["RF", "DNN", "CNN-IDS", "LSTM-IDS", "GAN-IDS",
               "CNN-LSTM", "GWO-GA", "CrossAttn", "GAT-IDS",
               "FL-WGAN", "MutaShield"]
    acc  = [95.23, 96.12, 96.78, 97.23, 97.56, 97.89,
            98.12, 98.45, 98.78, 99.01, 99.47]
    f1   = [94.15, 95.28, 95.95, 96.61, 96.95, 97.28,
            97.61, 98.00, 98.28, 98.50, 98.92]
    fpr  = [2.34, 1.89, 1.56, 1.34, 1.12, 0.98,
            0.87, 0.72, 0.61, 0.52, 0.31]

    x     = np.arange(len(methods))
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(14, 6))

    bars1 = ax1.bar(x - width, acc, width, label="Accuracy (%)",
                    color="#1f77b4", edgecolor="black", linewidth=0.5)
    bars2 = ax1.bar(x,         f1,  width, label="F1-Score (%)",
                    color="#ff7f0e", edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Score (%)", fontname="Times New Roman")
    ax1.set_ylim(93, 101)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontname="Times New Roman", rotation=25, ha="right")
    ax1.set_title("Performance Comparison on CIC-IDS2017 (Table VI)",
                  fontname="Times New Roman", fontsize=12)

    ax2 = ax1.twinx()
    ax2.bar(x + width, fpr, width, label="FPR (%)",
            color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("False Positive Rate (%)", fontname="Times New Roman")
    ax2.set_ylim(0, 4)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig05_performance_comparison.png")
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
    return path


def plot_adversarial_robustness(save_dir: str):

    methods  = ["CNN-IDS", "LSTM-IDS", "GAN-IDS", "GAT-IDS", "FL-WGAN", "MutaShield"]
    clean    = [96.78, 97.23, 97.56, 98.78, 99.01, 99.47]
    fgsm     = [67.23, 71.56, 78.34, 82.45, 85.67, 94.23]
    pgd      = [58.45, 63.78, 71.23, 76.89, 79.34, 91.56]
    cw       = [52.34, 56.89, 65.67, 71.23, 74.56, 88.89]
    autoatk  = [48.67, 52.12, 61.45, 67.56, 70.89, 85.67]

    x     = np.arange(len(methods))
    width = 0.14
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - 2*width, clean,   width, label="Clean",      color="#1f77b4", edgecolor="black", lw=0.5)
    ax.bar(x - 1*width, fgsm,    width, label="FGSM",       color="#ff7f0e", edgecolor="black", lw=0.5)
    ax.bar(x,           pgd,     width, label="PGD",        color="#2ca02c", edgecolor="black", lw=0.5)
    ax.bar(x + 1*width, cw,      width, label="C&W",        color="#d62728", edgecolor="black", lw=0.5)
    ax.bar(x + 2*width, autoatk, width, label="AutoAttack", color="#9467bd", edgecolor="black", lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontname="Times New Roman")
    ax.set_ylabel("Accuracy (%)", fontname="Times New Roman")
    ax.set_ylim(40, 105)
    ax.set_title("Adversarial Robustness Comparison (Table VIII)",
                 fontname="Times New Roman", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig06_adversarial_robustness.png")
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
    return path


def plot_ablation_study(save_dir: str):

    configs_   = ["Full", "w/o SMOE", "w/o CrossAttn", "w/o Gating",
                  "w/o AMFEL", "w/o BiGRU", "w/o Semantic"]
    acc_vals   = [99.47, 98.56, 98.89, 99.12, 98.34, 99.01, 98.78]
    f1_vals    = [98.92, 97.89, 98.23, 98.45, 97.56, 98.12, 97.89]
    adv_vals   = [88.89, 71.23, 82.45, 84.67, 68.89, 81.23, 79.56]

    x     = np.arange(len(configs_))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(x - width, acc_vals, width, label="Accuracy (%)",
           color="#1f77b4", edgecolor="black", lw=0.5)
    ax.bar(x,         f1_vals,  width, label="F1-Score (%)",
           color="#ff7f0e", edgecolor="black", lw=0.5)
    ax.bar(x + width, adv_vals, width, label="Adv. Acc. (%)",
           color="#d62728", edgecolor="black", lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(configs_, fontname="Times New Roman", rotation=20, ha="right")
    ax.set_ylabel("Score (%)", fontname="Times New Roman")
    ax.set_ylim(60, 102)
    ax.set_title("Ablation Study Results on CIC-IDS2017 (Table IX)",
                 fontname="Times New Roman", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig07_ablation_study.png")
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
    return path


if __name__ == "__main__":
    out_dir = config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    plot_mutant_taxonomy(out_dir)
    plot_performance_comparison(out_dir)
    plot_adversarial_robustness(out_dir)
    plot_ablation_study(out_dir)

    print(f"All figures saved to {out_dir}")
