"""
inference.py — MutaShield-Net Single-Sample Inference
Accepts a CSV row or a numpy feature vector and predicts the traffic class.
"""

import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config  import BEST_CKPT, RESULTS_DIR, CAGRT, CLASS_NAMES, RANDOM_SEED
from model   import MutaShieldNet
from utils   import set_seed


def load_model(ckpt_path: str = BEST_CKPT, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\nRun train.py first."
        )
    model = MutaShieldNet().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, device


def preprocess_sample(feature_vector: np.ndarray,
                      scaler=None, seq_len: int = None) -> tuple:
    """
    Convert a raw 80-dim feature vector to (p_seq, s_seq) tensors.
    scaler: sklearn StandardScaler fitted on training data (optional).
    """
    x = feature_vector.astype(np.float32)
    if scaler is not None:
        x = scaler.transform(x.reshape(1, -1)).flatten()

    # Pad/trim to 80 features
    if x.shape[0] < 80:
        x = np.pad(x, (0, 80 - x.shape[0]))
    else:
        x = x[:80]

    half = 40
    T    = seq_len or CAGRT["sequence_length"]
    p    = torch.from_numpy(x[:half]).unsqueeze(0).unsqueeze(0).expand(1, T, -1)  # (1,T,40)
    s    = torch.from_numpy(x[half:]).unsqueeze(0).unsqueeze(0).expand(1, T, -1)
    return p, s


def predict(model, p_seq: torch.Tensor, s_seq: torch.Tensor,
            device, class_names=None) -> dict:
    """Run forward pass and return prediction dict."""
    class_names = class_names or CLASS_NAMES
    model.eval()
    with torch.no_grad():
        logits = model(p_seq.to(device), s_seq.to(device))
        probs  = F.softmax(logits, dim=-1).cpu().squeeze(0)
        pred   = probs.argmax().item()
    return {
        "predicted_class": pred,
        "predicted_label": class_names[pred] if pred < len(class_names) else str(pred),
        "confidence":      float(probs[pred]),
        "probabilities":   probs.numpy(),
    }


def visualize_prediction(result: dict, class_names=None,
                          save_path: str = None):
    """Bar chart of class probabilities."""
    class_names = class_names or CLASS_NAMES
    probs = result["probabilities"]
    n     = len(probs)
    names = class_names[:n]

    colors = ['#e74c3c' if i == result["predicted_class"] else '#3498db'
              for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, probs * 100, color=colors, alpha=0.85)
    ax.set_xlabel('Probability (%)', fontsize=11)
    ax.set_title(
        f'MutaShield-Net Prediction\n'
        f'Predicted: {result["predicted_label"]}  '
        f'(Confidence: {result["confidence"]*100:.1f}%)',
        fontsize=12
    )
    ax.set_xlim(0, 105)
    for bar, p in zip(bars, probs):
        ax.text(p*100 + 0.5, bar.get_y() + bar.get_height()/2,
                f'{p*100:.1f}%', va='center', fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    else:
        plt.savefig(os.path.join(RESULTS_DIR, "inference_result.png"),
                    dpi=150, bbox_inches='tight')
    plt.close()


def main():
    set_seed(RANDOM_SEED)
    model, device = load_model()
    print(f"Model loaded on {device}")

    # ── Demo: random feature vector ───────────────────────────────────────────
    print("\nRunning demo inference on random 80-dim feature vector…")
    x_demo = np.random.randn(80).astype(np.float32)
    p_seq, s_seq = preprocess_sample(x_demo)

    result = predict(model, p_seq, s_seq, device)
    print(f"\nPrediction:")
    print(f"  Class  : {result['predicted_label']} (index {result['predicted_class']})")
    print(f"  Confidence: {result['confidence']*100:.2f}%")
    print(f"  All probabilities:")
    for i, (name, p) in enumerate(zip(CLASS_NAMES[:len(result['probabilities'])],
                                       result['probabilities'])):
        print(f"    {name:20s}: {p*100:6.2f}%")

    save_path = os.path.join(RESULTS_DIR, "inference_result.png")
    visualize_prediction(result, save_path=save_path)

    # ── From CSV row ──────────────────────────────────────────────────────────
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    if csv_path and os.path.exists(csv_path):
        import pandas as pd
        row = pd.read_csv(csv_path).iloc[0]
        # Drop non-numeric columns
        nums = row.select_dtypes(include=[float, int]).values.astype(np.float32)
        p_seq, s_seq = preprocess_sample(nums)
        result = predict(model, p_seq, s_seq, device)
        print(f"\nCSV sample prediction: {result['predicted_label']} "
              f"({result['confidence']*100:.1f}%)")
        visualize_prediction(result,
                             save_path=os.path.join(RESULTS_DIR, "inference_csv.png"))


if __name__ == "__main__":
    main()
