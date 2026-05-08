# MutaShield-Net

**A Hybrid Mutation-Guided Adversarial Evolution Framework for Adaptive Web Intrusion Detection Systems**

> MutaShield-Net integrates mutation testing principles from software engineering with adversarial evolutionary learning to continuously harden web intrusion detection models. It achieves **99.47% accuracy** and **67.3% reduction in false positive rate** on CICIDS2017, with **85.67% accuracy under AutoAttack** adversarial perturbations.

---

## Architecture

MutaShield-Net comprises three interdependent components:

1. **SMOE — Spectral Mutation Operator Engine** (Section III-B)
   - Constructs HTTP flow graphs and performs graph Laplacian spectral decomposition
   - 28 atomic mutation operators across 7 attack families (SQL Injection, XSS, Command Injection, Path Traversal, DoS, Reconnaissance, Auth Bypass)
   - Semantic preservation constraint: `S(x̃, c) ≥ δ = 0.8`

2. **CA-GRT — Cross-Domain Attention-Gated Recurrent Transformer** (Section III-C)
   - Dual BiGRU encoders for packet-domain and semantic-domain features
   - Multi-head cross-domain attention (8 heads, d_k=64)
   - Learnable domain-specific gating mechanism
   - ~8.7M parameters, 2.8ms inference latency

3. **AMFEL — Adversarial Mutation Fitness Evolutionary Loop** (Section III-D)
   - Co-evolutionary min-max optimization between mutant generators and detector
   - Mutation Survival Score (MSS) — detection-domain analog of software mutation score
   - Reduces live mutant proportion from 11.4% → 2.1% over 50 generations

```
Web Traffic → [SMOE: Mutation Pool] → [CA-GRT: Detection] → Output
                      ↑__________________________|
                    Feedback (surviving mutants harden detector)
```

---

## Dataset

| Dataset | Samples | Features | Attack Types | URL |
|---------|---------|----------|-------------|-----|
| **CICIDS2017** | 2,830,743 | 80 | 7 | [Link](https://www.unb.ca/cic/datasets/ids-2017.html) |
| **CSE-CIC-IDS2018** | 16,233,002 | 80 | 14 | [Link](https://www.unb.ca/cic/datasets/ids-2018.html) |

**Download via Kaggle:**
```bash
pip install kaggle
kaggle datasets download -d cicdataset/cicids2017 --path data/cicids2017/
unzip data/cicids2017/*.zip -d data/cicids2017/
```

**Folder structure:**
```
data/
└── cicids2017/
    ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    ├── Monday-WorkingHours.pcap_ISCX.csv
    └── ...
```

---

## Installation

```bash
git clone https://github.com/yourusername/MutaShieldNet.git
cd MutaShieldNet
pip install -r requirements.txt
```

**GPU requirements:** NVIDIA GPU with ≥16GB VRAM recommended (paper used dual A100 80GB).

---

## Usage

### Train
```bash
python train.py
```
Logs to `runs/` (TensorBoard), saves checkpoints to `checkpoints/`.

```bash
tensorboard --logdir runs/
```

### Evaluate
```bash
python evaluate.py
```
Outputs metrics table, confusion matrix (`results/confusion_matrix.png`), and ROC curves (`results/roc_curve.png`).

### Inference
```bash
# Demo (random sample)
python inference.py

# Single CSV row
python inference.py path/to/sample.csv
```

---

## Results

### Table I — CICIDS2017 Overall Performance

| Method | Acc (%) | Prec (%) | Rec (%) | F1 (%) | FPR (%) |
|--------|---------|----------|---------|--------|---------|
| Random Forest | 95.23 | 94.87 | 93.45 | 94.15 | 2.34 |
| CNN-IDS | 96.78 | 96.23 | 95.67 | 95.95 | 1.56 |
| LSTM-IDS | 97.23 | 96.89 | 96.34 | 96.61 | 1.34 |
| FL-WGAN-IDS | 99.01 | 98.67 | 98.34 | 98.50 | 0.52 |
| **MutaShield-Net** | **99.47** | **99.12** | **98.73** | **98.92** | **0.31** |

### Table II — Adversarial Robustness

| Method | Clean | FGSM | PGD | C&W | AutoAttack |
|--------|-------|------|-----|-----|------------|
| CNN-IDS | 96.78 | 67.23 | 58.45 | 52.34 | 48.67 |
| FL-WGAN-IDS | 99.01 | 85.67 | 79.34 | 74.56 | 70.89 |
| **MutaShield-Net** | **99.47** | **94.23** | **91.56** | **88.89** | **85.67** |

---

## Folder Structure

```
MutaShield-Net_Implementation/
├── figures/                  ← Paper figures (HD PNG)
│   ├── fig01_mutashield_architecture.png
│   ├── fig02_mss_evolution.png
│   ├── fig03_performance_comparison.png
│   ├── fig04_ablation_study.png
│   ├── fig05_training_loss.png
│   ├── fig06_adversarial_robustness.png
│   ├── table01_dataset_statistics.png
│   ├── table02_overall_performance.png
│   ├── table03_mutant_taxonomy.png
│   └── table04_ablation.png
├── data/                     ← Place CICIDS2017 CSVs here
│   └── cicids2017/
├── checkpoints/              ← Saved model weights
├── results/                  ← Evaluation outputs
├── notebooks/
│   └── demo.ipynb
├── config.py                 ← All hyperparameters
├── dataset.py                ← Data loading & preprocessing
├── model.py                  ← SMOE + CA-GRT + AMFEL
├── train.py                  ← Training loop
├── evaluate.py               ← Test evaluation & plots
├── inference.py              ← Single-sample prediction
├── utils.py                  ← Helper functions
├── requirements.txt
└── README.md
```

---

## Hyperparameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| SMOE | Spectral dim k | 64 |
| SMOE | Perturbation bound ε | 0.1 |
| SMOE | Semantic threshold δ | 0.8 |
| CA-GRT | GRU hidden dim | 256 |
| CA-GRT | Attention heads | 8 |
| CA-GRT | Dropout | 0.3 |
| AMFEL | Population size | 20 |
| AMFEL | Generations | 50 |
| Training | Learning rate | 1e-4 |
| Training | Batch size | 256 |
| Training | Epochs | 100 |

---

## Citation

```bibtex
@article{mutashieldnet2026,
  title   = {A Hybrid Mutation-Guided Adversarial Evolution Framework for
             Adaptive Web Intrusion Detection Systems},
  journal = {IEEE Transactions on ...},
  year    = {2026}
}
```
