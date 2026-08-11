import os

# ─── Dataset ────────────────────────────────────────────────────────────────
DATASET_CIC2017_URL = "https://www.unb.ca/cic/datasets/ids-2017.html"
DATASET_CIC2018_URL = "https://www.unb.ca/cic/datasets/ids-2018.html"

DATA_DIR          = os.path.join(os.path.dirname(__file__), "data")
CIC2017_RAW_DIR   = os.path.join(DATA_DIR, "CIC-IDS2017", "raw")
CIC2018_RAW_DIR   = os.path.join(DATA_DIR, "CSE-CIC-IDS2018", "raw")
CIC2017_PROC_DIR  = os.path.join(DATA_DIR, "CIC-IDS2017", "processed")
CIC2018_PROC_DIR  = os.path.join(DATA_DIR, "CSE-CIC-IDS2018", "processed")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20

N_FEATURES = 80

N_ATTACK_CATEGORIES_2017 = 7
N_ATTACK_CATEGORIES_2018 = 14

SMOE_SPECTRAL_DIM_K     = 64      
SMOE_PERTURBATION_BOUND = 0.1     
SMOE_SEMANTIC_THRESHOLD = 0.8     
SMOE_MAX_ITER           = 100     
SMOE_N_OPERATORS        = 28      
SMOE_N_FAMILIES         = 7

CAGRT_GRU_HIDDEN_DIM    = 256     
CAGRT_ATTN_HEADS        = 8       
CAGRT_KEY_DIM           = 64      
CAGRT_DROPOUT           = 0.3     
CAGRT_SEQ_LEN           = 100     
CAGRT_PACKET_FEAT_DIM   = 40      
CAGRT_SEMANTIC_FEAT_DIM = 40      

AMFEL_POP_MUTATION  = 20     
AMFEL_POP_DETECTOR  = 20     
AMFEL_GENERATIONS   = 50     
AMFEL_TOURNAMENT_K  = 5      
AMFEL_CROSSOVER_RATE = 0.8   
AMFEL_MUTATION_SIGMA = 0.01  
AMFEL_GAMMA         = 0.5    
AMFEL_FITNESS_BATCH = 2000   

LEARNING_RATE    = 1e-4           
BATCH_SIZE       = 256            
EPOCHS           = 100            
EARLY_STOP_PAT   = 10
LAMBDA           = 0.5            
RANDOM_SEEDS     = [42, 123, 456, 789, 1024]  

CKPT_DIR         = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR      = os.path.join(os.path.dirname(__file__), "results")
BEST_CKPT        = os.path.join(CKPT_DIR, "mutashieldnet_best.pt")
LAST_CKPT        = os.path.join(CKPT_DIR, "mutashieldnet_last.pt")
TENSORBOARD_DIR  = os.path.join(RESULTS_DIR, "tensorboard")

# ─── Device ─────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()  # Mixed precision when GPU available

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
