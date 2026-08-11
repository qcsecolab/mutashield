
import os
import time
import random
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import build_dataset_cic2017, get_dataloaders
from model import MutaShieldNet, AMFEL
from utils import compute_metrics, AverageMeter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(n_classes: int = 15) -> MutaShieldNet:
    model = MutaShieldNet(n_classes=n_classes)
    log.info(f"Model parameters: {model.count_parameters():,}")
    return model


def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    device, lambda_adv=config.LAMBDA):

    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total   = 0

    for p_seq, s_seq, labels in loader:
        p_seq  = p_seq.to(device, non_blocking=True)
        s_seq  = s_seq.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=config.USE_AMP):
            logits, _ = model(p_seq, s_seq)
            loss_clean = criterion(logits, labels)

            flat_p = p_seq[:, 0, :]    # (batch, d_p) — take first time step
            flat_s = s_seq[:, 0, :]    # (batch, d_s)
            flat_x = torch.cat([flat_p, flat_s], dim=-1)    # (batch, 80)
            noise  = torch.randn_like(flat_x) * config.SMOE_PERTURBATION_BOUND
            x_tilde = flat_x + noise

            d_p = config.CAGRT_PACKET_FEAT_DIM
            T   = config.CAGRT_SEQ_LEN
            p_t = x_tilde[:, :d_p].unsqueeze(1).expand(-1, T, -1)
            s_t = x_tilde[:, d_p:].unsqueeze(1).expand(-1, T, -1)
            logits_adv, _ = model(p_t, s_t)
            loss_adv   = criterion(logits_adv, labels)

            loss = loss_clean + lambda_adv * loss_adv

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), p_seq.size(0))
        preds    = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    acc = 100.0 * correct / total
    return loss_meter.avg, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    for p_seq, s_seq, labels in loader:
        p_seq  = p_seq.to(device, non_blocking=True)
        s_seq  = s_seq.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, _ = model(p_seq, s_seq)
        loss = criterion(logits, labels)
        loss_meter.update(loss.item(), p_seq.size(0))

        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics    = compute_metrics(all_labels, all_preds)
    return loss_meter.avg, metrics


def train(seed: int = config.DEFAULT_SEED,
          dataset: str = "cic2017",
          run_amfel: bool = True):
    """
    Full training pipeline (Section IV-A-2):
    1. Build dataset
    2. Train CA-GRT with inline adversarial loss
    3. Run AMFEL co-evolutionary loop every 10 epochs
    4. Save best checkpoint
    """
    set_seed(seed)
    device = config.DEVICE
    log.info(f"Training on {device}, seed={seed}")

    # ── Data ──────────────────────────────────────────────────────────────
    splits = build_dataset_cic2017()
    train_loader, val_loader, test_loader = get_dataloaders(
        splits["train"], splits["val"], splits["test"]
    )
    n_classes = len(np.unique(splits["train"][1]))
    log.info(f"Classes in training set: {n_classes}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(n_classes=n_classes).to(device)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer  = torch.optim.Adam(model.parameters(),
                                  lr=config.LEARNING_RATE,
                                  weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-6
    )
    scaler     = GradScaler(enabled=config.USE_AMP)

    # ── AMFEL ─────────────────────────────────────────────────────────────
    amfel = None
    if run_amfel:
        amfel = AMFEL(model.detector, model.smoe, device=device)

    # ── TensorBoard ───────────────────────────────────────────────────────
    tb_dir = os.path.join(config.TENSORBOARD_DIR, f"seed{seed}")
    writer = SummaryWriter(log_dir=tb_dir)

    best_val_acc  = 0.0
    patience_ctr  = 0
    mss_history   = []

    log.info(f"Starting training for {config.EPOCHS} epochs")
    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        val_acc = val_metrics["accuracy"]

        scheduler.step()
        elapsed = time.time() - t0

        log.info(
            f"Epoch {epoch:3d}/{config.EPOCHS} | "
            f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.2f}% | "
            f"ValLoss={val_loss:.4f} ValAcc={val_acc:.2f}% | "
            f"F1={val_metrics['f1']:.4f} FPR={val_metrics['fpr']:.4f} | "
            f"{elapsed:.1f}s"
        )

        writer.add_scalar("Loss/train",    train_loss, epoch)
        writer.add_scalar("Loss/val",      val_loss,   epoch)
        writer.add_scalar("Acc/train",     train_acc,  epoch)
        writer.add_scalar("Acc/val",       val_acc,    epoch)
        writer.add_scalar("F1/val",        val_metrics["f1"], epoch)
        writer.add_scalar("FPR/val",       val_metrics["fpr"], epoch)

        # ── AMFEL generation every 2 epochs ───────────────────────────────
        if run_amfel and epoch % 2 == 0:
            # Fetch a stratified batch for fitness evaluation
            X_batch_np, y_batch_np = splits["val"]
            idx = np.random.choice(len(X_batch_np),
                                   min(config.AMFEL_FITNESS_BATCH, len(X_batch_np)),
                                   replace=False)
            X_batch    = torch.from_numpy(X_batch_np[idx]).float().to(device)
            attack_cats = torch.from_numpy(y_batch_np[idx]).long().to(device)

            best_D, mss, drs = amfel.step(
                X_batch, attack_cats, val_acc / 100.0,
                g=epoch // 2
            )
            mss_history.append(mss)
            writer.add_scalar("AMFEL/MSS", mss, epoch)
            writer.add_scalar("AMFEL/DRS", drs, epoch)
            log.info(f"  AMFEL gen {epoch//2}: MSS={mss:.4f}, DRS={drs:.4f}")

            # Replace detector with best evolved candidate
            model.detector.load_state_dict(best_D.state_dict())

        # ── Checkpointing ─────────────────────────────────────────────────
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc, "seed": seed},
                   config.LAST_CKPT)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": val_acc, "seed": seed},
                       config.BEST_CKPT)
            log.info(f"  New best val acc: {best_val_acc:.2f}% — checkpoint saved.")
        else:
            patience_ctr += 1
            if patience_ctr >= config.EARLY_STOP_PAT:
                log.info(f"Early stopping at epoch {epoch}.")
                break

    writer.close()
    log.info(f"Training complete. Best val acc: {best_val_acc:.2f}%")

    # ── Final test evaluation ──────────────────────────────────────────────
    ckpt = torch.load(config.BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, test_metrics = evaluate(model, test_loader, criterion, device)
    log.info("Test metrics:")
    for k, v in test_metrics.items():
        log.info(f"  {k}: {v:.4f}")

    return test_metrics, mss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",      type=int,  default=config.DEFAULT_SEED)
    parser.add_argument("--no-amfel",  action="store_true")
    parser.add_argument("--dataset",   type=str,  default="cic2017")
    args = parser.parse_args()

    train(seed=args.seed, dataset=args.dataset, run_amfel=not args.no_amfel)
