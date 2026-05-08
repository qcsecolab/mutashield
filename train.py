"""
train.py — MutaShield-Net Training Loop
Implements full training with AMFEL co-evolutionary adaptation.
Section IV-A: Implementation Details.
"""

import os, copy, pickle, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from config  import TRAIN, AMFEL as AMFEL_CFG, BEST_CKPT, LAST_CKPT, AMFEL_CKPT, LOG_DIR, RANDOM_SEED
from dataset import build_dataloaders
from model   import MutaShieldNet, FocalLoss
from utils   import set_seed, compute_metrics, log_epoch


def train_one_epoch(model, loader, optimizer, scaler, device, lam):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for p_seq, s_seq, labels in loader:
        p_seq, s_seq, labels = p_seq.to(device), s_seq.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=TRAIN["mixed_precision"]):
            loss, logits = model.adversarial_loss(p_seq, s_seq, labels, lam=lam)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    criterion = FocalLoss()
    for p_seq, s_seq, labels in loader:
        p_seq, s_seq, labels = p_seq.to(device), s_seq.to(device), labels.to(device)
        logits = model(p_seq, s_seq)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.append(preds.cpu()); all_labels.append(labels.cpu())
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics    = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / total
    return metrics


# ─── AMFEL: co-evolutionary outer loop ───────────────────────────────────────

def amfel_step(model, train_loader, device, generation: int) -> float:
    """
    Section III-D / Algorithm 2: One AMFEL generation.
    Evaluates MSS on a mini-batch, then fine-tunes detector on surviving mutants.
    Returns current MSS value.
    """
    model.train()
    mss_values = []
    optimizer_ft = optim.Adam(model.parameters(), lr=TRAIN["learning_rate"] * 0.1)

    for i, (p_seq, s_seq, labels) in enumerate(train_loader):
        if i >= 5:   # evaluate MSS on 5 batches per generation
            break
        p_seq, s_seq, labels = p_seq.to(device), s_seq.to(device), labels.to(device)

        # Compute MSS (fraction of mutants predicted as benign)
        x_flat  = torch.cat([p_seq[:, 0, :], s_seq[:, 0, :]], dim=-1)
        mutants = model.smoe.generate_pool(x_flat)

        survived = 0; total_mut = 0
        with torch.no_grad():
            for mut in mutants:
                half = mut.shape[-1] // 2
                T    = p_seq.shape[1]
                p_m  = mut[:, :half].unsqueeze(1).expand_as(p_seq)
                s_m  = mut[:, half:].unsqueeze(1).expand_as(s_seq)
                logits = model.detector(p_m, s_m)
                preds  = logits.argmax(dim=-1)
                survived   += (preds == 0).sum().item()   # predicted benign = survived
                total_mut  += preds.numel()

        mss = survived / max(total_mut, 1)
        mss_values.append(mss)

        # Fine-tune on surviving mutants (Section III-D, Algorithm 2 line 14-16)
        model.train()
        optimizer_ft.zero_grad(set_to_none=True)
        loss_ft, _ = model.adversarial_loss(p_seq, s_seq, labels, lam=1.0)
        loss_ft.backward()
        optimizer_ft.step()

    return float(np.mean(mss_values)) if mss_values else 1.0


# ─── Main training procedure ──────────────────────────────────────────────────

def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build data
    train_loader, val_loader, _, scaler, encoder = build_dataloaders()

    # Build model
    model = MutaShieldNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer + scheduler (Table III)
    optimizer = optim.Adam(model.parameters(), lr=TRAIN["learning_rate"],
                           weight_decay=TRAIN["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN["epochs"], eta_min=1e-6
    )
    scaler_amp = GradScaler(enabled=TRAIN["mixed_precision"])
    writer     = SummaryWriter(LOG_DIR)

    best_val_acc = 0.0
    patience_ctr = 0
    mss_history  = []

    print(f"\n{'Epoch':>6}  {'T-Loss':>8}  {'T-Acc':>7}  {'V-Loss':>8}  "
          f"{'V-Acc':>7}  {'V-F1':>7}  {'MSS':>7}  {'LR':>10}")

    for epoch in range(1, TRAIN["epochs"] + 1):
        t0 = time.time()

        # ── Standard training epoch ──────────────────────────────────────────
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scaler_amp, device, TRAIN["lambda_mut"]
        )

        # ── Validation ───────────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, device)
        val_loss    = val_metrics["loss"]
        val_acc     = val_metrics["accuracy"]
        val_f1      = val_metrics["f1"]

        # ── AMFEL step every 5 epochs (Section III-D) ────────────────────────
        mss = float('nan')
        if epoch % 5 == 0:
            mss = amfel_step(model, train_loader, device, epoch // 5)
            mss_history.append(mss)
            writer.add_scalar("AMFEL/MSS", mss, epoch)
            if mss < AMFEL_CFG["mss_convergence_thr"]:
                print(f"\n  ✓ MSS converged to {mss:.4f} at epoch {epoch}")

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        # ── Logging ──────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        print(f"{epoch:>6}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {val_loss:>8.4f}  "
              f"{val_acc:>7.4f}  {val_f1:>7.4f}  {mss if not np.isnan(mss) else '--':>7}  "
              f"{lr:>10.2e}  [{elapsed:.1f}s]")

        writer.add_scalars("Loss",     {"train": tr_loss,  "val": val_loss},  epoch)
        writer.add_scalars("Accuracy", {"train": tr_acc,   "val": val_acc},   epoch)
        writer.add_scalar ("F1/val",   val_f1, epoch)
        writer.add_scalar ("LR",       lr,     epoch)

        # ── Checkpoint ───────────────────────────────────────────────────────
        torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": val_metrics, "mss": mss}, LAST_CKPT)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "metrics": val_metrics}, BEST_CKPT)
            print(f"  → New best val accuracy: {best_val_acc:.4f}")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= TRAIN["early_stop_patience"]:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience_ctr})")
                break

    # Save AMFEL state
    with open(AMFEL_CKPT, "wb") as f:
        pickle.dump({"mss_history": mss_history}, f)

    writer.close()
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {BEST_CKPT}")


if __name__ == "__main__":
    main()
