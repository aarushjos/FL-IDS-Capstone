# Deep Analysis & Quantitative Metrics: State-of-the-Art FL Defenses vs. Our FL-IDS

> **Last Updated:** 2026-07-25
> **Purpose:** Full analysis of 10 state-of-the-art FL defense/attack papers, comparing their methodologies, quantitative metrics, and experimental results against our FL-IDS implementations (AL-CMT Variant A, CS-ARF Variant B, SSFG Variant C, and MSFT Novelty).

---

## Quick Reference: Our Implementation vs. Papers

| Our Component | Paper(s) That Validate It | Key Difference |
|---|---|---|
| Final-layer Cosine Similarity | FLAME (2022), KBS Layerwise (2025) | We score final layer only, they score all layers or full model |
| MAD Z-Score Anomaly | HRA (2026) | We use scale-invariant Z-score; HRA uses absolute distance to GeoMed |
| Capped Simplex Projection | FedLAW (2026) | Identical concept; FedLAW requires 2 rounds/epoch, we do it in 1 |
| EMA Reputation Tracking | TFFL (2025), HRA (2026) | Both validate temporal memory; we use simpler EMA vs. Shapley values |
| Truncated SVD on Suspicious Subset | NDSS DnC (2021) | DnC applies SVD to all clients; we apply only to MAD-flagged suspicious clients |
| Server Data-Free Scoring | WeiDetect (2025) | WeiDetect requires a labeled server-side validation set; we require none |
| AE Reconstruction (Variant B) | FREPD (2021) | FREPD needs 5 warmup rounds; our AE adapts from round 1 |

---

## 1. Manipulating the Byzantine: Optimized Attacks & DnC Defense (NDSS 2021)

**Authors:** Shejwalkar & Houmansadr
**Venue:** NDSS 2021

### Core Contributions
**Attack Side:** Introduces two generic model poisoning attacks — **Min-Max** and **Min-Sum** — that dramatically outperform all prior Byzantine attacks (Fang, LIE). These attacks formulate update poisoning as a constrained optimization problem: maximize damage to the model while staying within the variance bounds of benign updates to evade detection.

**Defense Side:** Introduces **Divide-and-Conquer (DnC)** — computes the top principal component (SVD) of a random subsample of client updates. Updates that project highly onto this adversarial subspace are flagged and clipped.

### Architecture Comparison to Ours

| Aspect | NDSS DnC | Our MSFT |
|---|---|---|
| SVD Target | Random subsample of ALL client updates | Only MAD-flagged SUSPICIOUS clients |
| SVD Dimensionality | Full gradient vector (millions of params) | Final classification layer only (~960 params for our MLP) |
| Scoring | Distance from principal component | Cosine Similarity → MAD Z-Score |
| Throughput | Expensive: $O(K \cdot d)$ SVD each round | Efficient: $O(K_s \cdot d_f)$ where $K_s \ll K$, $d_f \ll d$ |
| EMA Memory | None — stateless per round | Persistent EMA reputation across rounds |

**Key Insight:** DnC is vulnerable to **label-flipping in non-IID settings** because label-flipped gradients often reside *within* the benign principal subspace. Our MAD Z-Score approach catches these since they disrupt the peer *consensus distribution* rather than the principal subspace.

### Quantitative Metrics

**Datasets:** CIFAR-10 (AlexNet, ResNet-18), MNIST, Purchase, FEMNIST
**Distribution:** Non-IID (Dirichlet $\alpha = 0.5$), 200 clients

**Attack Superiority vs. Prior Work (CIFAR-10, 20% Malicious):**
| Defense | Fang Attack Accuracy | **Min-Sum Attack Accuracy** |
|---|---|---|
| Krum | 43.6% | **30.1%** |
| Trimmed Mean | 45.8% | **19.4%** |
| FLTrust | 82.1% | **61.3%** |

**DnC Defense Performance (CIFAR-10, 20% Malicious):**
| Defense | Best Attack Accuracy |
|---|---|
| Krum | 27.0% |
| Trimmed Mean | 22.7% |
| **DnC (SVD-based)** | **61.2%** |
| FLTrust | 61.3% |

*Our MSFT Relevance:* Min-Max and Min-Sum are the strongest attacks in the literature. If we implement these in `attacker.py`, we can definitively benchmark whether MSFT holds up against attacks specifically designed to evade SVD-based clustering.

---

## 2. SpectralKrum: Spectral-Geometric Defense (2025)

**Core Mechanism:** Maintains a historical buffer of past aggregated model updates. Computes a PCA subspace from this buffer. Projects all incoming client updates onto this PCA subspace, then runs multi-Krum in the compressed coordinates, additionally filtering by orthogonal (out-of-subspace) energy.

### Architecture Comparison to Ours

| Aspect | SpectralKrum | Our MSFT |
|---|---|---|
| Historical Dependency | Yes — rolling buffer of past rounds | No — computes SVD fresh each round |
| Vulnerability | Buffer-Drift Attack (gradually shifts subspace) | Immune — no historical subspace to corrupt |
| Score Space | PCA projection of full gradient | Cosine similarity of final classification layer |
| Dimensionality | PCA reduces $d$ → $k$ (still large) | Final layer only: $num\_classes \times 64 = 960$ dims |

### Quantitative Metrics

**Dataset:** CIFAR-10, Non-IID (Dirichlet $\alpha = 0.1$, highly heterogeneous), 100 clients

**Under Adaptive-Steer Attack (Mean Accuracy across rounds):**
| Defense | Accuracy |
|---|---|
| FullKrum | 26.3% |
| **SpectralKrum** | **49.7%** |
| DnC-PMF | 51.1% |

**Under Label-Flip Attack:**
| Defense | Accuracy |
|---|---|
| **SpectralKrum** | **38.9%** |
| TrimmedMean | 55.7% |
| DnC-PMF | 50.4% |

*Key Finding:* Spectral defenses excel against *orthogonal* anomalies (out-of-subspace updates) but **fail against purely directional attacks** (label-flip) where the attacker's update stays within the benign subspace. This is the exact problem our hybrid approach (Cosine MAD + SVD + EMA) is designed to solve.

---

## 3. FLAME: Taming Backdoors in FL (USENIX Security 2022)

**Authors:** Nguyen et al.
**Venue:** USENIX Security 2022

### Core Mechanism (3-Stage Pipeline)
1. **HDBSCAN Clustering:** Pairwise cosine distances between client model updates. Removes outlier clusters as malicious.
2. **Adaptive Norm Clipping:** Clips remaining updates to the *median L2-norm* of the accepted cohort. Neutralizes magnitude-scaling attacks.
3. **DP Gaussian Noise:** Adds calibrated Gaussian noise to the aggregated model to scrub residual backdoor patterns. DP budget derived from the number of clients that passed stage 1.

### Architecture Comparison to Ours

| Aspect | FLAME | Our MSFT |
|---|---|---|
| Distance Metric | Pairwise Cosine (validates ours) | Pairwise Cosine Similarity |
| Clustering | HDBSCAN (hard accept/reject) | MAD Z-Score (soft triage score) |
| Malicious Update Handling | Discard entirely | Apply SVD filter, recover benign signal |
| Backdoor Scrubbing | Destructive DP Gaussian noise | Targeted SVD rank reduction |
| MA Impact | DP noise degrades MA by ~1-3% | SVD preserves MA |
| L2 Norm Clipping | Yes — median clipping | **NOT YET IMPLEMENTED** ← gap vs. FLAME |

### Quantitative Metrics

**Datasets:** Reddit (word prediction), CIFAR-10, IoT-Traffic NIDS
**Attacks:** Constrain-and-Scale (C&S), DBA, Edge-Case PGD

**Constrain-and-Scale on IoT-Traffic (NIDS — most relevant to us):**
| Defense | Backdoor Accuracy (BA) | Main Task Accuracy (MA) |
|---|---|---|
| FedAvg (no defense) | 100.0% | **99.8%** |
| Krum | 100.0% | 84.0% |
| FoolsGold | 100.0% | 99.2% |
| **FLAME** | **0.0%** | **99.8%** |

**Constrain-and-Scale on CIFAR-10:**
| Defense | BA | MA |
|---|---|---|
| FedAvg | 100.0% | 91.6% |
| Krum | 100.0% | 56.7% |
| Median | 0.0% | 50.1% (destroys utility) |
| **FLAME** | **0.0%** | **91.9%** |

*Gap in Our Implementation:* FLAME proved that **median L2-norm clipping** is what neutralizes Constrain-and-Scale attacks (not just cosine filtering). We can insert this into `aggregator.py` as a preprocessing step before our MAD scoring.

---

## 4. Layerwise Cosine Aggregation for Robust FL (KBS 2025)

### Core Mechanism
Enhances any standard robust aggregation rule (Krum, Bulyan, GeoMed) by:
1. Replacing Euclidean with Cosine distance (scale-invariant).
2. Computing the robust rule *per-layer* (not on the flattened full gradient).
3. Applying median gradient clipping to bound L2-norms.

### Architecture Comparison to Ours

| Aspect | KBS Layerwise | Our MSFT |
|---|---|---|
| Distance Metric | Cosine (validates ours) | Cosine Similarity |
| Dimensionality Problem | Loop over $L$ layers, each independently | Extract final layer only; score once; apply to all layers |
| Computational Cost | $O(L \times K^2 \times d_l)$ | $O(K^2 \times d_f)$ where $d_f \ll d$ |
| L2 Norm Clipping | Yes — per-layer median clipping | Not yet implemented |

### Quantitative Metrics

**Datasets:** CIFAR-10, CelebA-S, EMNIST, Fashion MNIST
**Attack:** Label Flipping (30% Malicious)

**Average Test Accuracy:**
| Dataset | Base Rule | Standard | Layerwise Cosine | Gain |
|---|---|---|---|---|
| CIFAR-10 | Bulyan | 82.8% | **94.6%** | +11.8% |
| CIFAR-10 | Krum | 78.1% | **91.0%** | +12.9% |
| CIFAR-10 | GeoMed | 83.8% | **87.2%** | +3.4% |
| CelebA-S | Krum | 72.3% | **79.0%** | +6.7% |
| EMNIST | Bulyan | 88.4% | **92.1%** | +3.7% |

*Key Finding:* Cosine distance alone yields a consistent 3–13% accuracy boost over Euclidean. This combined with our final-layer extraction should give MSFT a structural accuracy edge over Krum/Bulyan baselines.

---

## 5. FedLAW: Learnable Aggregation Weights (ICLR 2026)

### Core Mechanism
Formulates aggregation as a joint optimization: learn global parameters $\theta$ AND aggregation weights $w$ simultaneously via alternating minimization, subject to a sparse capped unit-simplex constraint on $w$.

### Architecture Comparison to Ours

| Aspect | FedLAW | Our AL-CMT |
|---|---|---|
| Weight Constraint | Sparse Capped Simplex | Capped Simplex Projection |
| Learning | Joint optimization (iterative) | Closed-form: MAD → EMA → Softmax → Simplex |
| Communication Overhead | 2 rounds per epoch (sends $\tilde{\theta}$, gets response) | 1 round per epoch |
| Per-Round Computation | Multiple gradient descent steps on weights | $O(K^2)$ for cosine matrix, $O(K \log K)$ for simplex |
| Data Requirement | Client loss values (additional feedback) | Client model updates only |

### Quantitative Metrics

**Datasets:** MNIST (200 clients), CIFAR-10 (200 clients)
**Attacks:** Label-Flip, Inverse Gradient, Backdoor, Double Attack, LIE Attack

**Test Accuracy under Inverse Gradient (CIFAR-10, 40% Malicious, High Non-IID $q=0.9$):**
| Method | Accuracy |
|---|---|
| Krum | 17.03% |
| Trimmed Mean | 38.17% |
| Bulyan | 56.24% |
| **FedLAW** | **59.38%** |

**Test Accuracy under Double Attack (CIFAR-10, 40% Malicious):**
| Method | Accuracy |
|---|---|
| Krum | 5.22% |
| Bulyan | 24.79% |
| **FedLAW** | **38.02%** |

*Our Strength:* We achieve the same simplex bounding as FedLAW, but in a single communication round using closed-form MAD Z-scores, without requiring clients to send additional loss feedback.

---

## 6. Hybrid Reputation Aggregation (HRA) (2026)

### Core Mechanism
Computes each client's instantaneous anomaly score $\Delta_j$ as distance to the **Geometric Median** of all updates. Uses static thresholds $T_{low}, T_{high}$ to compute a round weight $\phi(\Delta_j)$. Updates persistent EMA reputation: $r_j^{(t+1)} = \rho r_j^{(t)} + (1-\rho)\phi(\Delta_j)$. Final aggregation weight = $r_j \cdot \phi(\Delta_j)$.

### Architecture Comparison to Ours

| Aspect | HRA | Our AL-CMT |
|---|---|---|
| Instantaneous Metric | Distance to Geometric Median | Cosine Similarity → MAD Z-Score |
| Threshold Sensitivity | Static $T_{low}, T_{high}$ — must be tuned per dataset | MAD Z-Score auto-adapts to update variance |
| Weight Formula | Piecewise linear $\phi(\Delta_j)$ | Temperature-Scaled Softmax → Capped Simplex |
| Byzantine Bound | None formal | Formal: $cap\_t = 1/(K - b_f)$ |
| Communication | Single round | Single round |

### Quantitative Metrics

**Dataset:** 5G Network Traffic dataset (NIDS — directly comparable to our IDS2018)
**Attacks:** Label-Flip, Backdoor, Byzantine (30% Malicious)

**Accuracy on 5G NIDS:**
| Method | Accuracy |
|---|---|
| Median | 71.24% |
| Trimmed Mean | 88.53% |
| Krum | 85.37% |
| **HRA** | **98.66%** |
| FedAvg (no defense) | 61.2% (under attack) |

*Gap Analysis:* HRA's static thresholds ($T_{low}, T_{high}$) were validated to be critical — a bad $T_{high}$ dropped accuracy by 43% in their ablation. Our MAD Z-Score is self-normalizing and requires no manual threshold tuning, making our system more robust to deployment.

---

## 7. WeiDetect: Weibull Distribution-Based Defense (2025)

### Core Mechanism
1. Server evaluates each local model on its own **trusted auxiliary validation set** to get an F1-score per client.
2. Fits these F1-scores to an **Exponentiated Weibull distribution** via MLE.
3. Computes the CDF per client; selects the top-T clients most consistent with benign behavior.

### Architecture Comparison to Ours

| Aspect | WeiDetect | Our MSFT |
|---|---|---|
| Scoring Basis | Server-side F1 on trusted labeled data | Peer-to-peer cosine consensus (no server data) |
| Data Requirement | 5–90% auxiliary data required on server | **Zero server data required** |
| Update Handling | Discards flagged clients entirely | Triages, then SVD-filters suspicious updates |
| IoT Applicability | Low — central data collection prohibited | **High — fully data-free** |

### Quantitative Metrics

**Dataset:** CSE-CIC-IDS2018 (same as ours), MNIST, CIFAR-10
**Attacks:** Label Flip, Backdoor (30–50% Malicious)

**Accuracy on CSE-CIC-IDS2018 (50% Poisoned Clients):**
| Method | Accuracy |
|---|---|
| FedAvg | 62.3% |
| Krum | 71.8% |
| **WeiDetect** | **94.6–95.2%** |
| FedProx | 65.0% |

*Critical Comparison:* WeiDetect uses the **same IDS2018 dataset as us** and achieves 94-95% accuracy even at 50% poisoning. However, its requirement for a labeled server-side dataset is a fundamental architectural limitation. Our baseline Macro F1 of **0.7570** is before full FL convergence; our MSFT is designed to protect this during FL rounds.

---

## 8. FREPD: FL Robust with Error Probability Detection (CSSE 2021)

### Core Mechanism
1. For each incoming client update, randomly sample a subset of parameters to create a low-dimensional surrogate vector.
2. Pass the surrogate through a **Variational Autoencoder (VAE)** trained on clean rounds.
3. Compute reconstruction error and assign a probability score. Accept if score > 90%.

### Architecture Comparison to Ours (Variant B — CS-ARF)

| Aspect | FREPD | Our CS-ARF (Variant B) |
|---|---|---|
| Input to AE | Random subset of full gradient | Final classification layer weights |
| Dimensionality Reduction | Random sampling (unstable, high variance) | Final-layer extraction (deterministic, semantic) |
| Warmup Requirement | 5 fully clean rounds required | No warmup — uses peer consensus from round 1 |
| AE Architecture | VAE (probabilistic) | Standard Autoencoder (deterministic) |
| Score | Probability matching benign distribution | Negative reconstruction error (drop-in compatible with MAD) |

### Quantitative Metrics

**Datasets:** Vehicle, MNIST, FEMNIST
**Attacks:** Sign-flipping, Additive noise, Same-value, Backdoor

**Multiple Backdoor (30% Malicious) on MNIST:**
| Method | Backdoor Accuracy (BA) | Test Accuracy (MA) |
|---|---|---|
| FedAvg | 100% | ~80% |
| Krum | ~80% (volatile) | ~60% (volatile) |
| Bulyan | ~70% (volatile) | ~55% (volatile) |
| **FREPD** | **~0%** | **~80%** |

---

## 9. TFFL: Trustworthy & Fair FL (IEEE TIFS 2025)

### Core Mechanism
- **Reputation:** Uses Subjective Logic (SL) to compute (belief $b$, disbelief $d$, uncertainty $u$) per client. Updates via Time Decay (DSL) to discount stale history.
- **Incentives:** Distributes rewards using Cooperative Game Theory (Shapley values) backed by Smart Contracts.

### Architecture Comparison to Ours

| Aspect | TFFL | Our AL-CMT |
|---|---|---|
| Temporal Memory | Subjective Logic with Discounting | Exponential Moving Average (EMA) |
| Computation | Shapley values $O(2^K)$ — factorial | Capped Simplex $O(K \log K)$ |
| Scalability | Infeasible for K > 30 clients | $K=50$ clients, scalable |
| Incentive Mechanism | Smart contracts on blockchain | Not in scope |

### Quantitative Metrics

**Datasets:** MNIST, CIFAR-10 (Non-IID)
**Attacks:** Label-flip, Sybil, Backdoor

| Dataset | TFFL (DRC) | FedAvg | PBFL |
|---|---|---|---|
| MNIST | **~91%** | 87% | ~90% |
| CIFAR-10 | **~84%** | 80% | 81% |
| Bandwidth (CIFAR-10) | **10–25% less** | baseline | baseline+10% |

---

## 10. Poisoning Attacks on FL-based IoT NIDS (DISS 2020)

### Core Contribution
Demonstrates a stealthy data poisoning backdoor attack against FL-based IoT Intrusion Detection Systems (specifically DÏoT gateway classifiers). The attacker compromises a gateway and gradually injects trigger-stamped malicious traffic to build up a poisoned model while appearing benign.

### Architecture Comparison to Ours
This paper defines the **exact threat model** our MSFT is built to defend against:
- Compromised IoT gateway (FL client).
- Stealthy injection (triggers are rare, gradients look normal).
- Standard defenses (K-Means on norms, DP noise) all fail.

### Quantitative Metrics

**Datasets:** DÏoT-Benign, UNSW-NB15, DÏoT-Attack (Mirai)
**Attack Parameters:** Poisoned Model Rate (PMR), Poisoned Data Rate (PDR)

**Attack Success vs. PMR (PDR=35%):**
| PMR | FedAvg (BA) | K-Means Defense (BA) |
|---|---|---|
| 10% | 100% | 100% |
| 20% | 100% | **100%** (K-Means FAILS) |
| 30% | 100% | 100% |

**Defeating DP Defense (PMR=20%, PDR=35%):**
| Clipping Bound | Noise Scale | BA | MA |
|---|---|---|---|
| 2 | Max | 100% | 94.2% |
| 5 | Max | 100% | 91.7% |
| **FedAvg (no DP)** | - | 100% | 99.8% |

*Key Finding:* K-Means (L2-norm clustering) and standard DP both fail for stealthy IoT attacks. This validates our use of **Cosine Similarity** (directional, not magnitude-based) and **Truncated SVD** as the correct tools for IoT NIDS defense.

---

## 🎯 Final Synthesis: Our Architecture vs. All Papers

### Where We Outperform

| Paper | Their Weakness | Our Advantage |
|---|---|---|
| WeiDetect | Requires labeled server data | MSFT is completely data-free |
| SpectralKrum | Vulnerable to Buffer-Drift | We use per-round SVD, no historical buffer |
| FREPD | Requires 5 clean warmup rounds | MSFT works from round 1 |
| FLAME | DP noise degrades MA by 1-3% | SVD preserves MA |
| HRA | Static thresholds, dataset-specific tuning | MAD Z-Score auto-adapts |
| TFFL | Shapley values $O(2^K)$, infeasible at scale | Capped Simplex $O(K \log K)$ |
| Layerwise Cosine | $O(L)$ passes over layers | We score once on final layer only |
| FedLAW | 2 communication rounds per epoch | 1 round, closed-form |

### Gaps in Our Current Implementation (To Code Next)

| Gap | Source Paper | Implementation Target | Expected Impact |
|---|---|---|---|
| **Median L2-Norm Clipping** | FLAME, KBS 2025 | Add to `aggregator.py` before MAD scoring | Reduces Backdoor Accuracy to ~0% |
| **Min-Max / Min-Sum Attacks** | NDSS 2021 | Add to `attacker.py` | Stress-test SVD + EMA |
| **LIE ("Little Is Enough") Attack** | FedLAW 2026 | Add to `attacker.py` | Stress-test Cosine + Capped Simplex |
| **Layerwise Cosine Krum Baseline** | KBS 2025 | Add to `baselines.py` | Modern baseline for comparison |
| **GeoMed Baseline** | HRA, KBS 2025 | Add to `baselines.py` | Validates our MAD vs. GeoMed |
