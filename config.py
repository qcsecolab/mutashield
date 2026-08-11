"""
config.py — MutaShield-Net Configuration
Paper: "A Hybrid Mutation-Guided Adversarial Evolution Framework for
       Adaptive Web Intrusion Detection Systems"
Authors: Faisal Alhwikem, Syed Mohd Saqib
All hyperparameters taken directly from Table IV (Section IV-A-2).
"""

import os

# ─── Dataset ────────────────────────────────────────────────────────────────
DATASET_CIC2017_URL = "https://www.unb.ca/cic/datasets/ids-2017.html"
DATASET_CIC2018_URL = "https://www.unb.ca/cic/datasets/ids-2018.html"

DATA_DIR          = os.path.join(os.path.dirname(__file__), "data")
CIC2017_RAW_DIR   = os.path.join(DATA_DIR, "CIC-IDS2017", "raw")
CIC2018_RAW_DIR   = os.path.join(DATA_DIR, "CSE-CIC-IDS2018", "raw")
CIC2017_PROC_DIR  = os.path.join(DATA_DIR, "CIC-IDS2017", "processed")
CIC2018_PROC_DIR  = os.path.join(DATA_DIR, "CSE-CIC-IDS2018", "processed")

# Dataset split — Section IV-A-2: temporally stratified 70/10/20
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20

# Number of CICFlowMeter features — Section IV-A-1
N_FEATURES = 80

# Attack category counts
N_ATTACK_CATEGORIES_2017 = 7
N_ATTACK_CATEGORIES_2018 = 14

# ─── SMOE Hyperparameters (Table IV) ────────────────────────────────────────
SMOE_SPECTRAL_DIM_K     = 64      # Grid search over {32, 64, 128}
SMOE_PERTURBATION_BOUND = 0.1     # ε_c — semantic fidelity sweep
SMOE_SEMANTIC_THRESHOLD = 0.8     # δ_c — kills <5% trivial mutants
SMOE_MAX_ITER           = 100     # T in Algorithm 1 (number of mutation attempts)
SMOE_N_OPERATORS        = 28      # 28 atomic operators across 7 families
SMOE_N_FAMILIES         = 7

# ─── CA-GRT Hyperparameters (Table IV) ──────────────────────────────────────
CAGRT_GRU_HIDDEN_DIM    = 256     # d_h — standard scaling rule
CAGRT_ATTN_HEADS        = 8       # H — standard Transformer
CAGRT_KEY_DIM           = 64      # d_k = d_h / H = 256 / 8
CAGRT_DROPOUT           = 0.3     # Val-set tuning
CAGRT_SEQ_LEN           = 100     # T_seq — dataset flow length p95
CAGRT_PACKET_FEAT_DIM   = 40      # d_p — first 40 of 80 CICFlowMeter features
CAGRT_SEMANTIC_FEAT_DIM = 40      # d_s — remaining 40 features

# ─── AMFEL Hyperparameters (Table IV) ───────────────────────────────────────
AMFEL_POP_MUTATION  = 20          # N_M — co-evolutionary standard practice
AMFEL_POP_DETECTOR  = 20          # N_D — co-evolutionary standard practice
AMFEL_GENERATIONS   = 50          # G — MSS convergence criterion
AMFEL_TOURNAMENT_K  = 5           # k_tour — selection pressure balance
AMFEL_CROSSOVER_RATE = 0.8        # GA standard
AMFEL_MUTATION_SIGMA = 0.01       # σ for parameter perturbation (Eq. 27)
AMFEL_GAMMA         = 0.5         # γ in DRS equation (Eq. 24)
AMFEL_FITNESS_BATCH = 2000        # Stratified mini-batch size for fitness eval

# ─── Training Hyperparameters (Table IV) ────────────────────────────────────
LEARNING_RATE    = 1e-4           # Adam warm-up sweep
BATCH_SIZE       = 256            # GPU memory constraint
EPOCHS           = 100            # Early stopping patience = 10
EARLY_STOP_PAT   = 10
LAMBDA           = 0.5            # λ — balances original vs. mutated loss (Eq. 1)
RANDOM_SEEDS     = [42, 123, 456, 789, 1024]  # Section IV-A-2: 5 runs

# ─── Checkpoint and Output Paths ────────────────────────────────────────────
CKPT_DIR         = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR      = os.path.join(os.path.dirname(__file__), "results")
BEST_CKPT        = os.path.join(CKPT_DIR, "mutashieldnet_best.pt")
LAST_CKPT        = os.path.join(CKPT_DIR, "mutashieldnet_last.pt")
TENSORBOARD_DIR  = os.path.join(RESULTS_DIR, "tensorboard")

# ─── Device ─────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()  # Mixed precision when GPU available

# ─── Reproducibility ────────────────────────────────────────────────────────
DEFAULT_SEED = 42

# ─── Mutation operator family labels ────────────────────────────────────────
ATTACK_FAMILIES = [
    "SQL Injection",
    "XSS",
    "Command Injection",
    "Path Traversal",
    "DoS",
    "Reconnaissance",
    "Auth Bypass",
]

# ─── CIC-IDS2017 class label mapping ────────────────────────────────────────
CIC2017_LABEL_MAP = {
    "BENIGN":         0,
    "FTP-Patator":    1,
    "SSH-Patator":    2,
    "DoS slowloris":  3,
    "DoS Slowhttptest": 4,
    "DoS Hulk":       5,
    "DoS GoldenEye":  6,
    "Heartbleed":     7,
    "Web Attack -- Brute Force": 8,
    "Web Attack -- XSS":        9,
    "Web Attack -- Sql Injection": 10,
    "Infiltration":   11,
    "Bot":            12,
    "PortScan":       13,
    "DDoS":           14,
}

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)
os.makedirs(CIC2017_PROC_DIR, exist_ok=True)
os.makedirs(CIC2018_PROC_DIR, exist_ok=True)
