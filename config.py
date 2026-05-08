"""
config.py — MutaShield-Net Configuration
Paper: "A Hybrid Mutation-Guided Adversarial Evolution Framework for
        Adaptive Web Intrusion Detection Systems"
All hyperparameters match Section IV-A (Experimental Setup) of the paper.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
FIGURES_DIR     = os.path.join(BASE_DIR, "figures")
LOG_DIR         = os.path.join(BASE_DIR, "runs")          # TensorBoard

for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME    = "CICIDS2017"   # or "CSE-CIC-IDS2018"
CICIDS2017_URL  = "https://www.unb.ca/cic/datasets/ids-2017.html"
CSECIC_URL      = "https://www.unb.ca/cic/datasets/ids-2018.html"

# Kaggle mirror (programmatic download):
KAGGLE_DATASET  = "cicdataset/cicids2017"   # kaggle datasets download
CSV_FILENAME    = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"  # example file

LABEL_COLUMN    = " Label"           # CICFlowMeter column name (note leading space)
NUM_FEATURES    = 80                 # Section IV-A: 80 features (CICFlowMeter)
NUM_CLASSES     = 8                  # 1 benign + 7 attack categories (CICIDS2017)

# Train / Val / Test split
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15
RANDOM_SEED     = 42

# ─── SMOE Hyperparameters (Section III-B & Table III) ────────────────────────
SMOE = dict(
    spectral_dim        = 64,    # k — grid search {32,64,128}
    perturbation_bound  = 0.1,   # ε — semantic fidelity sweep
    semantic_threshold  = 0.8,   # δ — kills <5% trivial mutants
    max_iterations      = 100,   # T in Algorithm 1
    num_families        = 7,
    num_operators       = 28,    # 4 per family × 7 families
)

# ─── CA-GRT Hyperparameters (Section III-C & Table III) ──────────────────────
CAGRT = dict(
    packet_feat_dim     = 40,    # d_p — half of 80 features → packet domain
    semantic_feat_dim   = 40,    # d_s — other half → semantic domain
    gru_hidden_dim      = 256,   # d_h — standard scaling rule
    num_attention_heads = 8,     # H   — standard Transformer
    key_dim             = 64,    # d_k = d_h / H
    sequence_length     = 100,   # T_seq — dataset flow length p95
    dropout_rate        = 0.3,   # val-set tuning
    num_classes         = NUM_CLASSES,
    fc_hidden_dim       = 128,
)

# ─── AMFEL Hyperparameters (Section III-D & Table III) ───────────────────────
AMFEL = dict(
    mutation_pop_size   = 20,    # N_M
    detector_pop_size   = 20,    # N_D
    generations         = 50,    # G — MSS convergence criterion
    tournament_size     = 5,     # k_tour
    crossover_rate      = 0.8,   # GA standard
    mutation_sigma      = 0.01,  # σ in Eq. 25
    gamma               = 0.5,   # balance robustness vs. accuracy in DRS
    mss_convergence_thr = 0.03,  # stop early if MSS < this
)

# ─── Training Hyperparameters (Table III) ────────────────────────────────────
TRAIN = dict(
    learning_rate       = 1e-4,  # Adam warm-up sweep
    batch_size          = 256,   # GPU memory constraint
    epochs              = 100,
    early_stop_patience = 10,
    weight_decay        = 1e-5,
    lr_scheduler        = "cosine",   # cosine annealing
    mixed_precision     = True,
    lambda_mut          = 0.5,   # λ in Eq. 1 — balance original vs mutated loss
    num_workers         = 4,
    pin_memory          = True,
)

# ─── Checkpoint paths ─────────────────────────────────────────────────────────
BEST_CKPT  = os.path.join(CHECKPOINT_DIR, "mutashield_best.pt")
LAST_CKPT  = os.path.join(CHECKPOINT_DIR, "mutashield_last.pt")
AMFEL_CKPT = os.path.join(CHECKPOINT_DIR, "amfel_populations.pkl")

# ─── Results ──────────────────────────────────────────────────────────────────
CONF_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
ROC_CURVE_PATH   = os.path.join(RESULTS_DIR, "roc_curve.png")
METRICS_CSV      = os.path.join(RESULTS_DIR, "metrics.csv")

# ─── Class labels (CICIDS2017) ────────────────────────────────────────────────
CLASS_NAMES = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "Web Attack",
]

# Attack family mapping for SMOE operators
ATTACK_FAMILIES = [
    "SQL Injection",
    "XSS",
    "Command Injection",
    "Path Traversal",
    "DoS",
    "Reconnaissance",
    "Auth Bypass",
]
