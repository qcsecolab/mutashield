"""
model.py — MutaShield-Net Architecture
Implements three interdependent components (Section III):
  1. SMOE  — Spectral Mutation Operator Engine       (Section III-B)
  2. CA-GRT — Cross-Domain Attention-Gated Recurrent Transformer (Section III-C)
  3. AMFEL — Adversarial Mutation Fitness Evolutionary Loop      (Section III-D)
"""

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ─────────────────────────────────────────────────────────────────────────────
#  SMOE — Spectral Mutation Operator Engine  (Section III-B)
# ─────────────────────────────────────────────────────────────────────────────

class SMOE(nn.Module):
    """
    Generates semantically valid traffic mutations via graph-spectral
    decomposition of HTTP request-response flows (Equations 2-8, Algorithm 1).

    For CICFlowMeter flow-level features (80 dims):
      - Payload-level families (SQL, XSS, CMD, PTH, AUT): symbolic operators
        are approximated via learned spectral perturbation in feature space.
      - Network-layer families (DoS, RCN): direct CICFlowMeter subspace
        perturbation bounded by ||Δ_c||_F <= ε_c.

    Parameters
    ----------
    n_features : int
        Dimensionality of CICFlowMeter feature vector (80).
    spectral_dim : int
        Rank k for randomised SVD (64 per Table IV).
    n_families : int
        Number of attack families (7).
    perturbation_bound : float
        ε_c from Table IV (0.1).
    semantic_threshold : float
        δ_c from Table IV (0.8).
    max_iter : int
        T in Algorithm 1 (100 mutation attempts).
    """

    def __init__(self,
                 n_features: int = config.N_FEATURES,
                 spectral_dim: int = config.SMOE_SPECTRAL_DIM_K,
                 n_families: int = config.SMOE_N_FAMILIES,
                 perturbation_bound: float = config.SMOE_PERTURBATION_BOUND,
                 semantic_threshold: float = config.SMOE_SEMANTIC_THRESHOLD,
                 max_iter: int = config.SMOE_MAX_ITER):
        super().__init__()
        self.n_features        = n_features
        self.k                 = spectral_dim
        self.n_families        = n_families
        self.epsilon           = perturbation_bound
        self.delta             = semantic_threshold
        self.max_iter          = max_iter

        # Learnable projection matrices P_c ∈ R^{n_features × k}
        # One per attack family — maps spectral coords back to feature space (Eq. 6)
        self.proj = nn.Parameter(
            torch.randn(n_families, n_features, spectral_dim) * 0.01
        )

        # Operator weight vector θ_M per family (AMFEL evolves these)
        self.op_weights = nn.Parameter(torch.ones(n_families))

        # Semantic preservation scoring network (approximation of Eq. 8)
        self.sem_scorer = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, n_families),
            nn.Sigmoid(),
        )

    def _spectral_embed(self, x: torch.Tensor, family_idx: int) -> torch.Tensor:
        """
        Approximate spectral embedding z = U_s^T x using the learned projection
        matrix P_c for family c (Equation 5).
        """
        Pc = self.proj[family_idx]          # (n_features, k)
        z  = x @ Pc                         # (batch, k)
        return z

    def _decode(self, z_perturbed: torch.Tensor, family_idx: int) -> torch.Tensor:
        """
        Map perturbed spectral coordinates back to feature space (Decode in Eq. 6).
        x̃ = P_c z̃
        """
        Pc = self.proj[family_idx]          # (n_features, k)
        return z_perturbed @ Pc.T           # (batch, n_features)

    def semantic_score(self, x_tilde: torch.Tensor) -> torch.Tensor:
        """
        S(x̃, c) — semantic preservation score across all families (Eq. 8).
        Returns shape (batch, n_families).
        """
        return self.sem_scorer(x_tilde)

    def forward(self, x: torch.Tensor, attack_categories: torch.Tensor):
        """
        Generate a pool of valid mutants for each sample in x.

        Parameters
        ----------
        x : (batch, n_features)
        attack_categories : (batch,) — integer family index per sample

        Returns
        -------
        mutants : list of lists — mutants[i] contains valid mutants for sample i
        """
        batch_size = x.size(0)
        all_mutants = []

        for i in range(batch_size):
            xi = x[i].unsqueeze(0)                       # (1, n_features)
            c  = attack_categories[i].item()
            c  = int(c) % self.n_families                # guard index

            valid = []
            for _ in range(self.max_iter):
                # Algorithm 1, line 7: sample Δ_c ~ N(0, σ²I), ||Δ_c||_F <= ε
                z = self._spectral_embed(xi, c)          # (1, k)
                delta_c = torch.randn_like(z)
                delta_c = delta_c / (delta_c.norm() + 1e-8) * self.epsilon

                # Algorithm 1, line 8: z̃ = (Λ_s + Δ_c) z  — Λ_s ≈ identity here
                z_tilde = z + delta_c

                # Algorithm 1, line 9: x̃ = Decode(U_s z̃)
                x_tilde = self._decode(z_tilde, c)       # (1, n_features)

                # Algorithm 1, line 10: semantic check S(x̃, c) >= δ
                score = self.semantic_score(x_tilde)     # (1, n_families)
                if score[0, c].item() >= self.delta:
                    valid.append(x_tilde.squeeze(0))

            all_mutants.append(valid)

        return all_mutants


# ─────────────────────────────────────────────────────────────────────────────
#  CA-GRT — Cross-Domain Attention-Gated Recurrent Transformer (Section III-C)
# ─────────────────────────────────────────────────────────────────────────────

class BiGRUEncoder(nn.Module):
    """
    Bidirectional GRU encoder for one feature domain.
    Equations 11-15: GRU update gates (r_t, z_t, h̃_t, h_t).
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,    # bidirectional doubles output
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        x : (batch, seq_len, input_dim)
        Returns
        -------
        H : (batch, seq_len, hidden_dim)  — all hidden states
        h_T : (batch, hidden_dim)         — last time-step
        """
        H, _ = self.bigru(x)          # (batch, T, hidden_dim)
        H = self.dropout(H)
        h_T = H[:, -1, :]             # (batch, hidden_dim)
        return H, h_T


class MultiHeadCrossDomainAttention(nn.Module):
    """
    Multi-head cross-domain attention (Section III-C-3).
    Computes attention from domain Q onto domain KV.
    Equations 16-18.
    """

    def __init__(self, d_model: int, n_heads: int, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k     = d_k

        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_O = nn.Linear(n_heads * d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_q: torch.Tensor, H_kv: torch.Tensor):
        """
        H_q  : (batch, T, d_model) — query domain hidden states
        H_kv : (batch, T, d_model) — key/value domain hidden states
        Returns context vector c: (batch, d_model)
        """
        B, T, _ = H_q.shape

        Q = self.W_Q(H_q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(H_kv).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(H_kv).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention — Equation 16
        scale   = math.sqrt(self.d_k)
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / scale    # (B, H, T, T)
        attn    = F.softmax(scores, dim=-1)
        attn    = self.dropout(attn)

        out = torch.matmul(attn, V)                                # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_k)
        out = self.W_O(out)                                        # (B, T, d_model)

        # Mean-pool over time to get context vector c
        c = out.mean(dim=1)                                        # (B, d_model)
        return c


class CAGRT(nn.Module):
    """
    Cross-Domain Attention-Gated Recurrent Transformer.
    Full architecture from Section III-C, Equations 9-22.

    Packet features  p ∈ R^{d_p} → BiGRU → H_p, h_T^p
    Semantic features s ∈ R^{d_s} → BiGRU → H_s, h_T^s
    Cross-domain attention → c_{p→s}, c_{s→p}
    Domain-specific gating → g_p, g_s              (Eq. 19-20)
    Fused representation h_fused                    (Eq. 21)
    Classification head                             (Eq. 22)
    """

    def __init__(self,
                 packet_dim: int   = config.CAGRT_PACKET_FEAT_DIM,
                 semantic_dim: int = config.CAGRT_SEMANTIC_FEAT_DIM,
                 hidden_dim: int   = config.CAGRT_GRU_HIDDEN_DIM,
                 n_heads: int      = config.CAGRT_ATTN_HEADS,
                 d_k: int          = config.CAGRT_KEY_DIM,
                 dropout: float    = config.CAGRT_DROPOUT,
                 n_classes: int    = 15):   # K+1 for CIC-IDS2017
        super().__init__()
        self.hidden_dim = hidden_dim

        # Dual-domain BiGRU encoders
        self.bigru_p = BiGRUEncoder(packet_dim,   hidden_dim, dropout)
        self.bigru_s = BiGRUEncoder(semantic_dim, hidden_dim, dropout)

        # Multi-head cross-domain attention — Equation 16-18
        self.cross_attn_p2s = MultiHeadCrossDomainAttention(hidden_dim, n_heads, d_k, dropout)
        self.cross_attn_s2p = MultiHeadCrossDomainAttention(hidden_dim, n_heads, d_k, dropout)

        # Domain-specific gates — Equations 19-20
        self.gate_p = nn.Linear(hidden_dim * 2, hidden_dim)   # [h_T^p ; c_{p→s}]
        self.gate_s = nn.Linear(hidden_dim * 2, hidden_dim)   # [h_T^s ; c_{s→p}]

        # Classification head — Equation 22
        fused_dim = hidden_dim * 4                             # h_fused concatenated
        self.fc1 = nn.Linear(fused_dim, 256)
        self.fc2 = nn.Linear(256, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(256)

    def forward(self, p_seq: torch.Tensor, s_seq: torch.Tensor):
        """
        p_seq : (batch, T_seq, d_p)
        s_seq : (batch, T_seq, d_s)
        Returns
        -------
        logits : (batch, n_classes)
        h_fused : (batch, 4*hidden_dim)  — for AMFEL fitness evaluation
        """
        # BiGRU encoding — Equations 11-15
        H_p, h_T_p = self.bigru_p(p_seq)   # (B, T, dh), (B, dh)
        H_s, h_T_s = self.bigru_s(s_seq)   # (B, T, dh), (B, dh)

        # Cross-domain attention — Equations 16-18
        c_p2s = self.cross_attn_p2s(H_p, H_s)   # (B, dh) — packet attends semantic
        c_s2p = self.cross_attn_s2p(H_s, H_p)   # (B, dh) — semantic attends packet

        # Domain-specific gating — Equations 19-20
        g_p = torch.sigmoid(self.gate_p(torch.cat([h_T_p, c_p2s], dim=-1)))  # (B, dh)
        g_s = torch.sigmoid(self.gate_s(torch.cat([h_T_s, c_s2p], dim=-1)))  # (B, dh)

        # Fused representation — Equation 21
        h_fused = (g_p * h_T_p) + (g_s * h_T_s) + c_p2s + c_s2p             # (B, dh)
        # Concatenate all four components for richer representation
        h_fused_full = torch.cat([g_p * h_T_p, g_s * h_T_s, c_p2s, c_s2p], dim=-1)

        # Classification head — Equation 22
        out = self.fc1(h_fused_full)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)

        return logits, h_fused_full


# ─────────────────────────────────────────────────────────────────────────────
#  AMFEL — Adversarial Mutation Fitness Evolutionary Loop  (Section III-D)
# ─────────────────────────────────────────────────────────────────────────────

class MutationSurvivalScore:
    """
    MSS(θ_M, θ_D) — Equation 23.
    Fraction of mutants per seed sample that evade the detector (pred = 0 = BENIGN).
    """

    @staticmethod
    def compute(detector: CAGRT, smoe: SMOE,
                X_batch: torch.Tensor, attack_cats: torch.Tensor,
                device: str = config.DEVICE) -> float:
        detector.eval()
        smoe.eval()
        total_mutants = 0
        survived      = 0

        with torch.no_grad():
            X_batch    = X_batch.to(device)
            attack_cats = attack_cats.to(device)
            mutant_lists = smoe(X_batch, attack_cats)

            for mutant_set in mutant_lists:
                if not mutant_set:
                    continue
                mutants = torch.stack(mutant_set)           # (M_i, n_features)
                M_i = mutants.size(0)

                # Build dummy sequences for CA-GRT input
                d_p = config.CAGRT_PACKET_FEAT_DIM
                d_s = config.CAGRT_SEMANTIC_FEAT_DIM
                T   = config.CAGRT_SEQ_LEN
                p_m = mutants[:, :d_p].unsqueeze(1).expand(-1, T, -1)
                s_m = mutants[:, d_p:].unsqueeze(1).expand(-1, T, -1)

                logits, _ = detector(p_m, s_m)
                preds = logits.argmax(dim=-1)               # 0 = BENIGN = survival
                survived      += (preds == 0).sum().item()
                total_mutants += M_i

        mss = survived / max(total_mutants, 1)
        return mss


class DetectorRobustnessScore:
    """
    DRS(θ_D, θ_M) = 1 - MSS(θ_M, θ_D) + γ * Acc(θ_D; X_val) — Equation 24.
    """

    @staticmethod
    def compute(mss: float, val_acc: float,
                gamma: float = config.AMFEL_GAMMA) -> float:
        return 1.0 - mss + gamma * val_acc


class AMFEL:
    """
    Co-evolutionary optimisation loop (Section III-D, Algorithm 2).

    Maintains two populations:
      P_M = {θ_M^(1), ..., θ_M^(N_M)} — SMOE parameters
      P_D = {θ_D^(1), ..., θ_D^(N_D)} — CA-GRT parameters

    Fitness evaluation uses MSS and DRS.
    Evolutionary operators: tournament selection, crossover (Eq. 26), mutation (Eq. 27).
    """

    def __init__(self,
                 base_detector: CAGRT,
                 base_smoe: SMOE,
                 optimizer_cls=torch.optim.Adam,
                 n_mutation_pop: int  = config.AMFEL_POP_MUTATION,
                 n_detector_pop: int  = config.AMFEL_POP_DETECTOR,
                 generations: int     = config.AMFEL_GENERATIONS,
                 tournament_k: int    = config.AMFEL_TOURNAMENT_K,
                 crossover_rate: float = config.AMFEL_CROSSOVER_RATE,
                 sigma: float         = config.AMFEL_MUTATION_SIGMA,
                 device: str          = config.DEVICE):
        self.device          = device
        self.n_M             = n_mutation_pop
        self.n_D             = n_detector_pop
        self.G               = generations
        self.k_tour          = tournament_k
        self.cx_rate         = crossover_rate
        self.sigma           = sigma
        self.optimizer_cls   = optimizer_cls

        # Algorithm 2: initialise P_D from single pre-trained checkpoint
        # so divergence remains modest (Section III-D-4 discussion)
        self.pop_D = [copy.deepcopy(base_detector).to(device) for _ in range(n_detector_pop)]
        self.pop_M = [copy.deepcopy(base_smoe).to(device)     for _ in range(n_mutation_pop)]
        self.mss_history = []

    def _tournament_select(self, population, fitnesses):
        """Tournament selection — Equation 25."""
        idx = np.random.choice(len(population), self.k_tour, replace=False)
        best = max(idx, key=lambda i: fitnesses[i])
        return population[best]

    def _crossover_state_dicts(self, sd1, sd2, alpha=None):
        """
        Parameter interpolation crossover — Equation 26.
        Applied primarily to SMOE parameters (Section III-D-4).
        α ~ Uniform(0,1).
        """
        if alpha is None:
            alpha = np.random.uniform(0, 1)
        child_sd = {}
        for k in sd1:
            child_sd[k] = alpha * sd1[k].float() + (1 - alpha) * sd2[k].float()
        return child_sd

    def _mutate_state_dict(self, sd, eta):
        """Gaussian parameter mutation — Equation 27."""
        mutated = {}
        for k in sd:
            noise = torch.randn_like(sd[k].float()) * eta * self.sigma
            mutated[k] = sd[k].float() + noise
        return mutated

    def _evolve_population(self, population, fitnesses, pop_type="M", gen=1):
        """
        One generation of evolution for a population.
        pop_type: "M" applies crossover to SMOE params; "D" uses fine-tuning.
        """
        eta = 1.0 / math.sqrt(gen + 1)    # adaptive step — Eq. 27
        new_pop = []
        for _ in range(len(population)):
            parent1 = self._tournament_select(population, fitnesses)
            parent2 = self._tournament_select(population, fitnesses)
            if np.random.rand() < self.cx_rate:
                child_sd = self._crossover_state_dicts(
                    parent1.state_dict(), parent2.state_dict()
                )
            else:
                child_sd = copy.deepcopy(parent1.state_dict())
            child_sd = self._mutate_state_dict(child_sd, eta)

            child = copy.deepcopy(parent1)
            child.load_state_dict(child_sd, strict=False)
            new_pop.append(child)
        return new_pop

    def step(self, X_batch: torch.Tensor, attack_cats: torch.Tensor,
             val_acc: float, train_loader=None, criterion=None, g: int = 1):
        """
        One AMFEL generation (Algorithm 2, lines 2-14).

        Parameters
        ----------
        X_batch     : (batch, n_features) — stratified mini-batch
        attack_cats : (batch,) — integer family labels
        val_acc     : float — current detector validation accuracy
        train_loader: DataLoader for fine-tuning step (line 13)
        criterion   : loss function for fine-tuning
        g           : current generation index
        """
        # Algorithm 2, lines 3-8: evaluate fitness
        fit_M = []
        for theta_M in self.pop_M:
            mss_vals = []
            for theta_D in self.pop_D:
                mss = MutationSurvivalScore.compute(
                    theta_D, theta_M, X_batch, attack_cats, self.device
                )
                mss_vals.append(mss)
            fit_M.append(float(np.mean(mss_vals)))   # higher MSS = fitter mutator

        fit_D = []
        for theta_D in self.pop_D:
            drs_vals = []
            for theta_M in self.pop_M:
                mss = MutationSurvivalScore.compute(
                    theta_D, theta_M, X_batch, attack_cats, self.device
                )
                drs = DetectorRobustnessScore.compute(mss, val_acc)
                drs_vals.append(drs)
            fit_D.append(float(np.mean(drs_vals)))   # higher DRS = fitter detector

        # Algorithm 2, lines 9-10: evolve populations
        self.pop_M = self._evolve_population(self.pop_M, fit_M, "M", g)
        self.pop_D = self._evolve_population(self.pop_D, fit_D, "D", g)

        # Algorithm 2, line 11: best detector
        best_D_idx = int(np.argmax(fit_D))
        best_D     = self.pop_D[best_D_idx]

        # Track MSS
        current_mss = float(np.mean(fit_M))
        self.mss_history.append(current_mss)

        return best_D, current_mss, float(np.mean(fit_D))


# ─────────────────────────────────────────────────────────────────────────────
#  MutaShieldNet — top-level wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MutaShieldNet(nn.Module):
    """
    Top-level module combining CA-GRT detector with SMOE.
    Used for standard training forward passes. AMFEL is managed externally
    in train.py to keep the evolutionary loop decoupled from the gradient graph.
    """

    def __init__(self,
                 n_classes: int = 15,
                 packet_dim: int   = config.CAGRT_PACKET_FEAT_DIM,
                 semantic_dim: int = config.CAGRT_SEMANTIC_FEAT_DIM,
                 hidden_dim: int   = config.CAGRT_GRU_HIDDEN_DIM,
                 n_heads: int      = config.CAGRT_ATTN_HEADS,
                 d_k: int          = config.CAGRT_KEY_DIM,
                 dropout: float    = config.CAGRT_DROPOUT):
        super().__init__()
        self.detector = CAGRT(packet_dim, semantic_dim, hidden_dim,
                              n_heads, d_k, dropout, n_classes)
        self.smoe = SMOE()

    def forward(self, p_seq: torch.Tensor, s_seq: torch.Tensor):
        """Standard forward — returns (logits, h_fused)."""
        return self.detector(p_seq, s_seq)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
