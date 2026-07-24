# MSFT Approach & Pipeline Analysis

This document provides a deep, structured analysis of the Multi-Stage Final-Layer Triage (MSFT) approach implemented in our FL-IDS system. It breaks down the entire pipeline component by component, explaining both the intuitive "simple terms" concept and the underlying technical mathematics, followed by a comprehensive experiment plan.

---

## 1. Overall Pipeline Architecture

**In Simple Terms:**
Instead of blindly trusting all clients (like FedAvg) or running heavy security checks on the entire AI model (which is too slow), our approach only looks at the very last layer of the AI model. It puts clients through a series of "gates." First, we stop any client trying to shout too loudly (clipping). Next, we see who is pointing in the wrong direction compared to the crowd (Cosine+MAD). If a client looks suspicious, we put them under a microscope (SVD). Finally, we track their reputation over time (EMA) and give them a voting weight based on how much we trust them, but we never let any small group have too much power (Capped Simplex).

**Technical Details:**
The MSFT (`TriageAggregator`) operates purely on the server-side during `aggregate_fit`. It processes a list of client weight updates (NumPy arrays) through a 5-stage pipeline:
1. **Norm Clipping (Stage 0):** Bound gradient magnitudes.
2. **Initial Scoring (Stage 1):** O(K²) pairwise Cosine Similarity + Median Absolute Deviation (MAD) on the final layer.
3. **Triage & Refinement (Stage 2):** Truncated SVD anomaly scoring, but *strictly* applied only to the subset of clients where MAD < threshold.
4. **Trust & Projection (Stage 3):** Exponential Moving Average (EMA) of scores mapped through a temperature-scaled Softmax, then bounded via a Capped Simplex projection.
5. **Aggregation (Stage 4):** Final convex combination of the full model weights.

---

## 2. Component-by-Component Breakdown

### Stage 0: Median L2-Norm Clipping (FLAME, USENIX 2022)
**Simple Terms:** 
Attackers sometimes multiply their malicious update by a huge number (like 100x) so that when the server averages the models, the attacker's update completely overwrites the good clients. We stop this by looking at the "size" of everyone's update and forcing anyone who is above average to shrink down to the middle size.

**Technical Details:** 
- **Mechanism:** We compute the L2-norm of the fully concatenated parameter vector for each client. We find the median norm across the cohort. If a client's norm exceeds the median, their entire update is scaled down by `median / client_norm`. 
- **Why it's here:** Cosine similarity (used in Stage 1) only measures the *angle* of the update, not the *magnitude*. Without clipping, a Constrain-and-Scale attacker can bypass cosine defenses. 

### Stage 1: Final-Layer Cosine Similarity + MAD Scoring
**Simple Terms:** 
Different IoT devices have different normal traffic (Non-IID data), so their updates will naturally look a bit different. If we compare the whole model, good clients might look like attackers. Instead, we only compare the very last part of the model (the classification head). We measure the angle between everyone's updates (Cosine). Then, we use a robust statistical tool (MAD) to find the "middle of the pack" and flag anyone who is too far away from it.

**Technical Details:**
- **Mechanism:** Extract only the final weight matrix. Compute pairwise cosine distances using `scipy.spatial.distance.pdist`.
- **MAD (Median Absolute Deviation):** Calculate the sum of similarities for each client. Let these be $S_i$. We compute $Z_i = (S_i - \text{median}(S)) / \text{MAD}(S)$. 
- **Advantage over HRA:** HRA (2026) uses static distance thresholds which must be manually tuned per dataset. MAD Z-scores are self-normalizing, adapting dynamically to the cohort's natural variance.

### Stage 2: SVD Hybrid on Suspicious Subset (Extends DnC, NDSS 2021)
**Simple Terms:** 
Some advanced attackers (like the LIE attack) know how to trick the MAD math to stay just barely inside the "normal" range. To catch them, if the MAD score says a client is even slightly suspicious, we isolate that group and use a heavy-duty mathematical microscope (SVD) to look for hidden malicious patterns. We don't use this microscope on everyone because it's computationally expensive.

**Technical Details:**
- **Mechanism:** Filter clients where $Z_i < \text{soft\_threshold}$. On this subset matrix $M_{sub}$, perform SVD: $U, \Sigma, V^T = \text{SVD}(M_{sub})$.
- **Hybrid Score:** We calculate a projection score measuring alignment with the top left-singular vector: $P_i = 1 - (|U_{i,0}| / \max(|U_{*,0}|))$. 
- **Blending:** The final score for suspicious clients is a blend: `0.6 * re_MAD + 0.4 * P_i`.
- **Advantage:** SVD identifies the primary axis of variation in the suspicious subset (likely the attack vector). By blending it, we catch variance-optimized attacks (LIE) that evade 1D statistical bounds.

### Stage 3: EMA Reputation + Capped Simplex Projection
**Simple Terms:** 
A client shouldn't be banned forever if they send one weird update (which happens naturally in wireless IoT). Instead, we keep a running "credit score" (EMA). When converting this credit score into voting power, we use a strict cap. If 20 clients are voting, we mathematically guarantee that no single client (or small cartel) can ever hold more than, say, 10% of the total voting power, even if they have perfect credit.

**Technical Details:**
- **EMA:** $R_{i, t} = \rho R_{i, t-1} + (1-\rho) Z_{i, t}$. Adds temporal memory, defeating Adaptive Mimicry attackers who act benign to build trust before striking.
- **Softmax:** Map unbounded EMA scores to a probability distribution using temperature scaling: $w_i = \exp(R_i / \tau) / \sum \exp(R_j / \tau)$.
- **Capped Simplex:** Project $w$ onto a bounded simplex where $0 \le w_i \le \frac{1}{K-f}$, where $f$ is the estimated number of Byzantine clients. Solved in $O(K \log K)$ via a continuous Lagrange multiplier binary search.

---

## 3. Comprehensive Experiment & Testing Plan

Our goal is to prove that MSFT achieves State-of-the-Art (SOTA) performance. We measure this primarily via **Macro F1** (classification performance on minority attack classes) and **Attack Success Rate (ASR)** (how often a backdoor succeeds).

### Experiment 1: Clean Baseline (Sanity Check)
* **What to run:** 0% attackers across all strategies (FedAvg, HRA, Layerwise Krum, MSFT).
* **Goal:** Prove that MSFT does not hurt the model when there are no attackers.
* **Target Metric:** Macro F1 should be within ~1% of FedAvg.

### Experiment 2: The "Constrain-and-Scale" Backdoor
* **What to run:** Backdoor attack with `scale_to_benign_norm=False` (Attacker amplifies norm by 50x) vs `scale_to_benign_norm=True` (Attacker hides within benign norm).
* **Component Tested:** Stage 0 (Median L2-Norm Clipping).
* **Expected Result:** Without clipping, FedAvg ASR goes to 100%. With clipping, ASR should drop to near 0%. MSFT should survive both.
* **Target Metric:** Low ASR (< 5%).

### Experiment 3: Label-Flipping Sweep (Standard Byzantine)
* **What to run:** `label_flip` attack at 10%, 30%, and 50% attacker ratios.
* **Component Tested:** Stage 1 (Cosine + MAD) and Stage 4 (Capped Simplex).
* **Expected Result:** As attackers hit 50%, Krum and TrimmedMean will collapse. HRA and MSFT should survive. MSFT should beat HRA because HRA's static thresholds will struggle with Non-IID variance.
* **Target Metric:** Maintain Macro F1 > 0.65 even at 30% attackers.

### Experiment 4: Min-Max and Min-Sum Attacks (NDSS 2021)
* **What to run:** `min_max` and `min_sum` attacks. These are agnostic attacks designed specifically to defeat clustering (Cosine) defenses.
* **Component Tested:** Stage 1 + Stage 2.
* **Expected Result:** Standard Krum/GeoMed will degrade. MSFT's Stage 2 SVD Hybrid should detect the tightly clustered adversarial vectors that Min-Sum creates.
* **Target Metric:** MSFT Macro F1 > GeoMed / LayerwiseCosineKrum Macro F1.

### Experiment 5: The LIE Attack (FedLAW, ICLR 2026)
* **What to run:** `lie` attack with `lie_z_clip=2.0`. LIE injects `mean + z*std` to perfectly evade variance-based scoring (like pure MAD).
* **Component Tested:** Stage 2 (SVD Projection).
* **Expected Result:** Pure MAD (Variant A) might fail here. MSFT's Stage 2 applies SVD to the suspicious subset. The LIE perturbation creates a massive principal component in the SVD, causing the projection score to plummet and catch the attacker.
* **Target Metric:** MSFT survives with High F1; Variant A (robust) degrades.

### Experiment 6: Adaptive Mimicry / Trust-then-Strike (HRA 2026)
* **What to run:** `trust_then_strike` attack (`trust_rounds=5`, `strike=min_max`).
* **Component Tested:** Stage 3 (EMA Reputation).
* **Expected Result:** Stateless defenses (GeoMed, Krum, TrimmedMean) will be instantly compromised on round 6. EMA will delay the compromise, and because the strike is harsh, the EMA score will crash rapidly in round 6/7, restricting the attacker's voting weight via the simplex cap.
* **Target Metric:** Minimal F1 drop during the strike rounds.

### Experiment 7: Ablation Study
* **What to run:** Run the hardest attack (e.g., 30% `min_max` or `lie`) on MSFT (`triage`), Variant A (`robust`), `full_model_cosine`, and `final_no_simplex`.
* **Goal:** Prove every stage of MSFT is mathematically necessary.
* **Expected Result:** 
  - `full_model_cosine` fails due to Non-IID diversity.
  - `final_no_simplex` fails because attackers monopolize the softmax without the simplex cap.
  - Variant A (`robust`) performs well but fails on `lie`.
  - MSFT (`triage`) wins globally.
