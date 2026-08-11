import os
import argparse
import logging
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "Times New Roman"
rcParams["font.size"]   = 11

import config
from model import MutaShieldNet

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CLASS_NAMES = {
    0:  "BENIGN",
    1:  "FTP-Patator",
    2:  "SSH-Patator",
    3:  "DoS-Slowloris",
    4:  "DoS-Slowhttptest",
    5:  "DoS-Hulk",
    6:  "DoS-GoldenEye",
    7:  "Heartbleed",
    8:  "Web-BruteForce",
    9:  "Web-XSS",
    10: "Web-SQLi",
    11: "Infiltration",
    12: "Botnet",
    13: "PortScan",
    14: "DDoS",
}


def load_model(ckpt_path: str, n_classes: int = 15, device: str = config.DEVICE):
    model = MutaShieldNet(n_classes=n_classes).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def preprocess_sample(raw_features: np.ndarray, scaler=None) -> tuple:

    x = raw_features.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -1e9, 1e9)

    if scaler is not None:
        x = scaler.transform(x.reshape(1, -1)).squeeze()

    d_p = config.CAGRT_PACKET_FEAT_DIM
    T   = config.CAGRT_SEQ_LEN

    p = torch.from_numpy(x[:d_p]).float().unsqueeze(0).unsqueeze(0)  
    s = torch.from_numpy(x[d_p:]).float().unsqueeze(0).unsqueeze(0)  

    p_seq = p.expand(-1, T, -1) 
    s_seq = s.expand(-1, T, -1) 
    return p_seq, s_seq


@torch.no_grad()
def predict(model, p_seq: torch.Tensor, s_seq: torch.Tensor,
            device: str = config.DEVICE):

    p_seq = p_seq.to(device)
    s_seq = s_seq.to(device)
    logits, h_fused = model(p_seq, s_seq)
    probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred   = int(probs.argmax())
    return pred, probs, h_fused.cpu()


def visualize_prediction(probs: np.ndarray, pred: int, save_path: str):
    classes = [CLASS_NAMES.get(i, str(i)) for i in range(len(probs))]
    colors  = ["#d62728" if i == pred else "#1f77b4" for i in range(len(probs))]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(classes, probs * 100, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Confidence (%)", fontname="Times New Roman")
    ax.set_title(f"MutaShield-Net Prediction: {CLASS_NAMES.get(pred, pred)} "
                 f"({probs[pred]*100:.1f}% confidence)",
                 fontname="Times New Roman", fontsize=12)
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=45)
    for bar, p in zip(bars, probs):
        if p > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, f"{p*100:.1f}",
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    log.info(f"Prediction chart saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      default=config.BEST_CKPT)
    parser.add_argument("--input",     type=str, default=None,
                        help="JSON file with 80 CICFlowMeter features, or omit for demo")
    parser.add_argument("--n-classes", type=int, default=15)
    parser.add_argument("--save-dir",  type=str, default=config.RESULTS_DIR)
    args = parser.parse_args()

    device = config.DEVICE
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────────
    if not os.path.exists(args.ckpt):
        log.warning(f"Checkpoint not found: {args.ckpt}. Using untrained model for demo.")
        model = MutaShieldNet(n_classes=args.n_classes).to(device)
        model.eval()
    else:
        model = load_model(args.ckpt, args.n_classes, device)

    # ── Load or generate input ───────────────────────────────────────────
    if args.input and os.path.exists(args.input):
        with open(args.input) as f:
            features = np.array(json.load(f), dtype=np.float32)
        log.info(f"Loaded features from {args.input}: shape={features.shape}")
    else:
        log.info("No input file provided — using random demo feature vector.")
        features = np.random.randn(config.N_FEATURES).astype(np.float32)

    assert features.shape == (config.N_FEATURES,), \
        f"Expected shape ({config.N_FEATURES},), got {features.shape}"

    # ── Inference ────────────────────────────────────────────────────────
    p_seq, s_seq = preprocess_sample(features)
    pred, probs, _ = predict(model, p_seq, s_seq, device)

    print(f"\nPredicted class : {CLASS_NAMES.get(pred, pred)} (index {pred})")
    print(f"Confidence      : {probs[pred]*100:.2f}%")
    print(f"Threat detected : {'YES — ATTACK' if pred != 0 else 'NO — BENIGN'}\n")

    for i, p in enumerate(probs):
        bar = "#" * int(p * 50)
        print(f"  {CLASS_NAMES.get(i, i):20s} | {bar:<50} {p*100:5.1f}%")

    # ── Visualise ────────────────────────────────────────────────────────
    vis_path = os.path.join(args.save_dir, "inference_result.png")
    visualize_prediction(probs, pred, vis_path)


if __name__ == "__main__": main()
