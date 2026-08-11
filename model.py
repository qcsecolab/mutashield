import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class SMOE(nn.Module):

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

        self.proj = nn.Parameter(
            torch.randn(n_families, n_features, spectral_dim) * 0.01
        )

        self.op_weights = nn.Parameter(torch.ones(n_families))

        self.sem_scorer = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, n_families),
            nn.Sigmoid(),
        )

    def _spectral_embed(self, x: torch.Tensor, family_idx: int) -> torch.Tensor:

        Pc = self.proj[family_idx]          
        z  = x @ Pc                         
        return z

    def _decode(self, z_perturbed: torch.Tensor, family_idx: int) -> torch.Tensor:

        Pc = self.proj[family_idx]          # (n_features, k)
        return z_perturbed @ Pc.T           # (batch, n_features)

    def semantic_score(self, x_tilde: torch.Tensor) -> torch.Tensor:

        return self.sem_scorer(x_tilde)

    def forward(self, x: torch.Tensor, attack_categories: torch.Tensor):

        batch_size = x.size(0)
        all_mutants = []

        for i in range(batch_size):
            xi = x[i].unsqueeze(0)                      
            c  = attack_categories[i].item()
            c  = int(c) % self.n_families               

            valid = []
            for _ in range(self.max_iter):
               
                z = self._spectral_embed(xi, c)        
                delta_c = torch.randn_like(z)
                delta_c = delta_c / (delta_c.norm() + 1e-8) * self.epsilon

                z_tilde = z + delta_c

                x_tilde = self._decode(z_tilde, c)      

                score = self.semantic_score(x_tilde)    
                if score[0, c].item() >= self.delta:
                    valid.append(x_tilde.squeeze(0))

            all_mutants.append(valid)

        return all_mutants


class BiGRUEncoder(nn.Module):


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

        H, _ = self.bigru(x)         
        H = self.dropout(H)
        h_T = H[:, -1, :]            
        return H, h_T


class MultiHeadCrossDomainAttention(nn.Module):


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

        B, T, _ = H_q.shape

        Q = self.W_Q(H_q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(H_kv).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(H_kv).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scale   = math.sqrt(self.d_k)
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / scale    
        attn    = F.softmax(scores, dim=-1)
        attn    = self.dropout(attn)

        out = torch.matmul(attn, V)                               
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_k)
        out = self.W_O(out)                                       

        c = out.mean(dim=1)                                        
        return c


class CAGRT(nn.Module):


    def __init__(self,
                 packet_dim: int   = config.CAGRT_PACKET_FEAT_DIM,
                 semantic_dim: int = config.CAGRT_SEMANTIC_FEAT_DIM,
                 hidden_dim: int   = config.CAGRT_GRU_HIDDEN_DIM,
                 n_heads: int      = config.CAGRT_ATTN_HEADS,
                 d_k: int          = config.CAGRT_KEY_DIM,
                 dropout: float    = config.CAGRT_DROPOUT,
                 n_classes: int    = 15):  
        super().__init__()
        self.hidden_dim = hidden_dim

        self.bigru_p = BiGRUEncoder(packet_dim,   hidden_dim, dropout)
        self.bigru_s = BiGRUEncoder(semantic_dim, hidden_dim, dropout)

        self.cross_attn_p2s = MultiHeadCrossDomainAttention(hidden_dim, n_heads, d_k, dropout)
        self.cross_attn_s2p = MultiHeadCrossDomainAttention(hidden_dim, n_heads, d_k, dropout)

        self.gate_p = nn.Linear(hidden_dim * 2, hidden_dim)   
        self.gate_s = nn.Linear(hidden_dim * 2, hidden_dim)   

        fused_dim = hidden_dim * 4                             
        self.fc1 = nn.Linear(fused_dim, 256)
        self.fc2 = nn.Linear(256, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(256)

    def forward(self, p_seq: torch.Tensor, s_seq: torch.Tensor):

        H_p, h_T_p = self.bigru_p(p_seq)   
        H_s, h_T_s = self.bigru_s(s_seq)   

        c_p2s = self.cross_attn_p2s(H_p, H_s)   
        c_s2p = self.cross_attn_s2p(H_s, H_p)   

        g_p = torch.sigmoid(self.gate_p(torch.cat([h_T_p, c_p2s], dim=-1))) 
        g_s = torch.sigmoid(self.gate_s(torch.cat([h_T_s, c_s2p], dim=-1)))  

        h_fused = (g_p * h_T_p) + (g_s * h_T_s) + c_p2s + c_s2p            
        h_fused_full = torch.cat([g_p * h_T_p, g_s * h_T_s, c_p2s, c_s2p], dim=-1)

        out = self.fc1(h_fused_full)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)

        return logits, h_fused_full

class MutationSurvivalScore:


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
                mutants = torch.stack(mutant_set)          
                M_i = mutants.size(0)

                # Build dummy sequences for CA-GRT input
                d_p = config.CAGRT_PACKET_FEAT_DIM
                d_s = config.CAGRT_SEMANTIC_FEAT_DIM
                T   = config.CAGRT_SEQ_LEN
                p_m = mutants[:, :d_p].unsqueeze(1).expand(-1, T, -1)
                s_m = mutants[:, d_p:].unsqueeze(1).expand(-1, T, -1)

                logits, _ = detector(p_m, s_m)
                preds = logits.argmax(dim=-1)              
                survived      += (preds == 0).sum().item()
                total_mutants += M_i

        mss = survived / max(total_mutants, 1)
        return mss


class DetectorRobustnessScore:

    @staticmethod
    def compute(mss: float, val_acc: float,
                gamma: float = config.AMFEL_GAMMA) -> float:
        return 1.0 - mss + gamma * val_acc


class AMFEL:


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


        self.pop_D = [copy.deepcopy(base_detector).to(device) for _ in range(n_detector_pop)]
        self.pop_M = [copy.deepcopy(base_smoe).to(device)     for _ in range(n_mutation_pop)]
        self.mss_history = []

    def _tournament_select(self, population, fitnesses):
        idx = np.random.choice(len(population), self.k_tour, replace=False)
        best = max(idx, key=lambda i: fitnesses[i])
        return population[best]

    def _crossover_state_dicts(self, sd1, sd2, alpha=None):

        if alpha is None:
            alpha = np.random.uniform(0, 1)
        child_sd = {}
        for k in sd1:
            child_sd[k] = alpha * sd1[k].float() + (1 - alpha) * sd2[k].float()
        return child_sd

    def _mutate_state_dict(self, sd, eta):
        mutated = {}
        for k in sd:
            noise = torch.randn_like(sd[k].float()) * eta * self.sigma
            mutated[k] = sd[k].float() + noise
        return mutated

    def _evolve_population(self, population, fitnesses, pop_type="M", gen=1):

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

        self.pop_M = self._evolve_population(self.pop_M, fit_M, "M", g)
        self.pop_D = self._evolve_population(self.pop_D, fit_D, "D", g)

        best_D_idx = int(np.argmax(fit_D))
        best_D     = self.pop_D[best_D_idx]

        current_mss = float(np.mean(fit_M))
        self.mss_history.append(current_mss)

        return best_D, current_mss, float(np.mean(fit_D))

class MutaShieldNet(nn.Module):


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
        return self.detector(p_seq, s_seq)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
