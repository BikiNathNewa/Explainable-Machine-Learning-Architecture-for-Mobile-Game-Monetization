# Model Card: Structural Predictors of Mobile Strategy Game Monetization Archetypes

## 1. Model Details

### 1.1 Overview
- **Project Title:** From Mechanics to Microtransactions: An Empirical Study on Structural Predictors of Mobile Strategy Game Monetization Archetypes
- **Primary Model Architecture:** Extreme Gradient Boosted Trees (`XGBoostClassifier`)
- **Version:** 1.0.0 (Production Release)
- **Model Framework:** XGBoost 3.2.0 / Scikit-Learn 1.8.0
- **Explainability Stack:** TreeSHAP (Tree SHapley Additive exPlanations, SHAP 0.51.0) and Diverse Counterfactual Explanations (DiCE 0.12)
- **Primary Artifact Location:** `artifacts/models/xgboost_model.joblib`

### 1.2 Intended User & Decision (Persona: Maya)
- **Primary Intended User:** Maya, a Lead Producer at an independent mid-sized mobile game studio.
- **Target Decision Context:** Deciding whether to greenlight the heavy backend architectural investment required for an In-App Purchase (IAP) microtransaction economy or to pivot the title's core loop toward a lean, ad-supported freemium monetization loop during pre-production.
- **Operational Value:** Provides quantitative diagnostic attributions via TreeSHAP (Tree SHapley Additive exPlanations) and prescriptive recourse via DiCE (Diverse Counterfactual Explanations) showing which specific planned game mechanics associate with the pay-to-progress archetype, allowing Maya to optimize technical architecture before capital is deployed.

### 1.3 Target Operationalization & Out-of-Scope Uses
- **Target Operationalization:** The model predicts a **heuristically defined monetization archetype** operationalized via public storefront catalog metadata ($Y=1$: catalog contains In-App Purchase [IAP] $\ge \$19.99$; $Y=0$: ad indicators + In-App Purchase [IAP] $< \$4.99$).
- **Direct Financial Performance Forecasting (Out-of-Scope):** Not designed to predict actual commercial performance metrics, including financial revenue, Average Revenue Per User (ARPU), Average Revenue Per Paying User (ARPPU), player conversion rates, or Lifetime Value (LTV).
- **Dynamic In-Game Pricing (Out-of-Scope):** Prohibited from adjusting live microtransaction price tiers dynamically based on player behavior or perceived elasticity.
- **Post-Launch Telemetry Optimization (Out-of-Scope):** Designed strictly for pre-production architectural decisions; not validated on post-launch live-ops telemetry.
- **Predatory Dark Pattern Optimization (Prohibited):** Strictly prohibited from optimizing mechanic configurations to induce addictive or coercive spending behaviors, especially in titles with family/minor audiences.

---

## 2. Model Performance & Complexity Specifications

### 2.1 Model Complexity & Opacity Parameters (Criterion 7)
To evaluate the opacity trade-off, the regularized ensemble model is parameterized with explicit structural constraints:
- **Base Estimators:** Exactly `n_estimators = 100` boosting iterations.
- **Maximum Tree Depth:** Constrained to `max_depth = 4` levels per tree.
- **Learning Rate:** `learning_rate = 0.05` with log-loss optimization (`eval_metric = "logloss"`).
- **Encoded Feature Space:** 93 model-input features generated from nine engineered structural attributes (via one-hot encoding and standardization).
- **Ensemble Opacity:** Encapsulates up to $100 \times (2^4 - 1) = 1,500$ potential decision splits, providing non-linear capacity while preventing overfitting.
- **Deterministic Seed:** Frozen across all stages with `random_state = 42`.

### 2.2 Side-by-Side Performance Comparison (Criterion 2 vs. Criterion 7)
Evaluated on the identical held-out test split of 2,513 titles (80/20 stratified split, zero split leakage):

| Model Family | Opacity Classification | Accuracy | Precision (Class 1) | Recall (Class 1) | Macro F1 | AUC-ROC | PR-AUC | Brier Score | Log-Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (depth=2)** | Transparent (3 splits) | 86.11% | 84.17% | 26.35% | 0.6614 | 0.8035 | 0.4901 | 0.1070 | 0.3536 |
| **Logistic Regression** | Linear Baseline (93 weights) | 88.62% | 82.64% | 45.05% | 0.7586 | 0.8619 | 0.6787 | 0.0917 | 0.3100 |
| **XGBoost (100 trees, depth 4)** | Regularized Ensemble (~1.5k splits) | **89.73%** | **84.70%** | **51.13%** | **0.7889** | **0.9099** | **0.7630** | **0.0790** | **0.2663** |

### 2.3 Evaluation on Imbalanced Classes & Complexity Justification
Because Class 0 comprises 82% of instances, raw accuracy is an incomplete metric (a naive majority-class classifier achieves 82%). Evaluation centers on **AUC-ROC (0.9099)**, **PR-AUC (0.7630)**, **Macro-F1 (0.7889)**, and **Class-1 Recall (51.13%)**.
While the linear baseline achieves an AUC-ROC of 0.8619, it captures only 45.05% of aggressive microtransaction titles. The regularized XGBoost model achieves an AUC-ROC of 0.9099 and increases recall on minority microtransaction titles to 51.13% (at 84.70% precision) while reducing log-loss from 0.3100 down to 0.2663.
This difference stems from non-linear **multi-feature interaction associations**: client download footprints interact non-linearly with synchronous multiplayer networking. Large games without multiplayer frequently remain viable as single-player freemium titles, but large download footprints combined with real-time Player-versus-Player (PvP) matchmaking associate strongly with microtransaction dependency. Tree ensembles capture these interaction thresholds naturally without manual polynomial expansion.

---

## 3. Explainability, Faithfulness & Stability Summary

### 3.1 Faithfulness Evidence (Criterion 3: Feature Deletion Curves)
Faithfulness was evaluated via iterative feature masking on the held-out test set, comparing TreeSHAP (Tree SHapley Additive exPlanations) ranking against Random and Inverse deletion controls across 13 deletion budgets ($k \in [0, \dots, 93]$):
- **TreeSHAP Deletion (Top Features First):** Produces an immediate drop. Masking just the single top-ranked feature (`size_in_mb`) causes AUC to decline by **0.0734**. Masking 5 features drops AUC by **0.2165**.
- **Random Deletion Control (20 Seeds):** Retains high performance after 5 features are masked (AUC drop of only 0.0053), exhibiting slow, gradual decay.
- **Inverse Deletion Control (Bottom Features First):** Preserves full performance across initial iterations (drop of 0.0000).
- **Area Under the Deletion Curve (AUDC):**
  - TreeSHAP AUDC: **0.5625** (Lowest area indicates highest faithfulness)
  - Random Deletion AUDC: **0.7774**
  - Inverse Deletion AUDC: **0.8371**
- **Interpretation:** These results provide empirical evidence that the highest-ranked TreeSHAP (Tree SHapley Additive exPlanations) features correspond closely to model inputs that materially affect predictive scoring.

### 3.2 Stability Evaluation (Criterion 4: Seeds & Perturbations)
Evaluated across a fixed cohort of 50 representative test profiles:
- **Test A (Seed Variation on Sampling Explainers):** Running KernelSHAP (Kernel SHapley Additive exPlanations) across 10 random seeds revealed significant instability due to Monte Carlo sampling variance: mean pairwise Spearman rank correlation was only $r_s = 0.346$ ($\pm 0.167$) and mean top-5 Jaccard overlap was only $0.338$ ($\pm 0.179$). This demonstrates that sampling explainers are unsuitable for high-stakes production reviews.
- **Test B (Small Input Epsilon Perturbations on TreeSHAP):** Continuous features were perturbed by $\pm 2\%$, strictly bounded to ensure model output probability changed by $\Delta p < 0.01$ (mean observed $\Delta p = 0.0054$). Under TreeSHAP (Tree SHapley Additive exPlanations), **96.0%** of instances maintained identical top-3 feature sets (only 2 out of 50 games exhibited a boundary flip). TreeSHAP is mathematically deterministic, and our perturbation experiment indicates high attribution stability under tested input noise.

### 3.3 Pilot Human Decision Support (Criterion 5: Forward Simulation Study)
Evaluated with $N = 10$ human participants (graduate classmates and mobile game developers). Each evaluator assessed all 10 standardized game cards (5 Control, 5 Treatment), producing $10 \times 10 = 100$ total evaluator-card trials:
- **Design:** Within-subject evaluation comparing 5 Control cards (mechanics alone) against 5 Treatment cards (mechanics + local TreeSHAP [Tree SHapley Additive exPlanations] waterfall plots).
- **Control Group (Mechanics Alone):** Prediction accuracy was **62.0%** (confidence 2.8/5.0). Evaluators routinely misclassified large single-player games.
- **Treatment Group (Mechanics + SHAP):** Prediction accuracy improved to **88.0%** (+26.0 percentage points, confidence 4.2/5.0) when provided with SHAP (SHapley Additive exPlanations) attributions.
- **Sample Scale Notice:** We explicitly acknowledge sample size constraints ($N=10$) and present these findings as pilot descriptive evidence without claiming formal statistical significance.

### 3.4 Model-Derived Actionable Recourse (Criterion 6: Constrained DiCE Optimization)
- **Constraint Enforcement:** All immutable features (`primary_genre`, `secondary_genre`, `minimum_os_version` [Operating System], `content_rating`) were strictly locked.
- **Recourse Generation:** For 5 high-risk games ($p \in [0.988, 0.993]$), DiCE (Diverse Counterfactual Explanations) generated minimal feature deltas associated with flipping predictions to the Ad-Supported freemium archetype ($p \in [0.366, 0.489]$).
- **Model-Derived Levers for Maya:**
  1. Asset Footprint Compression: Reducing download size below 150 Megabytes (MB) via on-demand dynamic asset streaming.
  2. Multiplayer Architecture Decoupling: Swapping real-time synchronous Player-versus-Player (PvP) matchmaking for asynchronous leaderboards or turn-based ghost matchmaking.

---

## 4. Ethical & Content-Rating Regulatory Proxy Audit (Criterion 8)

### 4.1 Content-Rating Regulatory Proxy Identification
Storefront metadata does not record player demographic profiles. To prevent regulatory non-compliance under the Children's Online Privacy Protection Act (COPPA) and Federal Trade Commission (FTC) dark pattern guidelines, content ratings were audited as statutory proxies under youth protection frameworks:
- `proxy_family_youth_under_12`: `content_rating` $\in$ `['4+', '9+']` (Youth / Family classifications)
- `proxy_mature_17_plus`: `content_rating` == `'17+'` (Adult / Mature classifications)

*Demographic Clarification:* App Store age ratings indicate **content suitability classifications** and should not be interpreted as direct measurements of player demographics (e.g., a `4+` rating indicates content suitability for all ages, not that only children play the game).

### 4.2 Empirical Correlation Audit
Exact correlation coefficients with structural features and archetype target labels:
- **Youth Proxy vs. Monetization Archetype ($Y$):** $r_s = -0.162$ ($p < 0.001$). Lower content-age classifications are negatively associated with the Aggressive In-App Purchase (IAP) archetype.
- **Youth Proxy vs. Synchronous Multiplayer:** $r_s = -0.141$. Real-time Player-versus-Player (PvP) is concentrated in older age tiers due to toxicity and chat moderation requirements.
- **Mature Proxy vs. Monetization Archetype ($Y$):** $r_s = +0.108$ ($p < 0.001$). Mature content classifications are positively associated with the Aggressive In-App Purchase (IAP) archetype.

### 4.3 Production Risk Mitigation for Maya
- **Regulatory Warning:** Because lower content-age classifications associate empirically with ad-supported freemium loops, studios often deploy ad monetization. Maya should require a privacy/legal review of advertising Software Development Kits (SDKs) and child-directed data practices before deployment to ensure regulatory compliance with the Children's Online Privacy Protection Act (COPPA) and Federal Trade Commission (FTC) dark pattern guidelines.
- **Engineering Recommendation:** For any game rated `4+` or `9+`, Maya should bypass behavioral ad tracking, implementing contextual or premium monetization loops instead.

---

## 5. Dataset & Feature Schema

### 5.1 Dataset Lineage, Candidate Sources & Scoping
- **Candidate Sources Evaluated in `data/raw/`:**
  1. *Primary Modeling Cohort:* 17K Apple App Store Strategy Games (`appstore_games.csv`, 17,007 titles; CC0 license).
  2. *Google Play 10K Storefront Dataset:* (`googleplaystore.csv`, 10,841 records by lava18; CC0 license).
  3. *Google Play 2.3M Storefront Dataset:* (`Google-Playstore.csv`, 2,312,944 records by gauthamp10; CC0 license).
- **Primary Scoping Rationale:** Only `appstore_games.csv` records granular In-App Purchase (IAP) price arrays (`$0.99` to `$99.99`) required to mathematically operationalize the monetization target ($Max\_IAP \ge \$19.99$ vs $< \$4.99$). In addition, over 85% of Google Play records are non-gaming utility apps, and cross-store pooling introduces platform packaging confounders.
- **Ambiguous Mid-Tier & Paid Games Filtered:** 4,442 titles (4,071 hybrid titles with $\$4.99 \le \text{Max IAP} < \$19.99$ plus 371 upfront-paid non-freemium games)
- **Final Clean Analyzed Cohort:** 12,565 titles (10,344 Ad-Supported, 2,221 Aggressive In-App Purchase [IAP]; partitioned into 10,052 training and 2,513 test instances)

### 5.2 Feature Schema
- **Immutable Features (Frozen):**
  - `primary_genre` (Categorical)
  - `secondary_genre` (Categorical)
  - `minimum_os_version` (Categorical: Operating System - OS Version)
  - `content_rating` (Categorical: 4+, 9+, 12+, 17+)
- **Mutable Features (Actionable):**
  - `size_in_mb` (Continuous float: package download size in Megabytes [MB])
  - `is_synchronous_multiplayer` (Binary: real-time Player-versus-Player [PvP] / multiplayer)
  - `session_pacing_tag` (Categorical: Turn-Based, Real-Time, Hypercasual/Idle, Session-Based)
  - `supported_languages_count` (Integer: localization breadth)
  - `days_since_last_update` (Continuous integer: live-ops staleness)
- **Target Variable ($Y$):**
  - $Y = 1$ (**Aggressive In-App Purchase [IAP] Archetype**): `has_iap == True` AND `max_iap_price >= $19.99`
  - $Y = 0$ (**Ad-Supported / Freemium Archetype**): `contains_ads == True` AND `max_iap_price < $4.99`

---

## 6. Phase 2 Advanced Multi-Tier Models, Temporal Benchmarks & Enterprise Scanners

### 6.1 Multi-Class & Continuous Price Regression Models
* **Multi-Class Regularized XGBoost (`artifacts/models/xgboost_multiclass.joblib`):**
  - Expanded cohort: Full 14,236 free-to-download titles (re-ingesting the 4,071 titles in the $4.99 to $19.99 band).
  - Target: 3-Class Ordinal Target (0: Freemium Loop, 1: Hybrid Battle-Pass, 2: Whale Economy IAP).
  - Performance: Accuracy 79.21%, Macro F1 0.5040, Macro OvR AUC-ROC 0.8169, Macro OvR PR-AUC 0.6115, Mean Absolute Class Error 0.2963.
* **Continuous Log-Price XGBoost Regressor (`artifacts/models/xgboost_regressor.joblib`):**
  - Unbinned regression directly predicting $\log(1 + \text{max\_iap\_price})$.
  - Performance: $R^2 = 0.4241$, MAE = 0.8428, RMSE = 1.0876.
* **Pre-Launch Clean Model Variant (`artifacts/models/xgboost_prelaunch.joblib`):**
  - Excludes post-release operational telemetry (`days_since_last_update`).
  - Performance: Random Split Accuracy 89.02%, AUC-ROC 0.9057, PR-AUC 0.7435; Chronological Split Accuracy 87.43%, AUC-ROC 0.8794 solely from design-time pitch specifications.

### 6.2 Advanced Validation & Enterprise Decision Support
* **Conditional Faithfulness Deletion Curve (WP 2.2):** Benchmarked against Marginal Median Masking (AUDC = 0.7904), Archetype-Conditional KNN Imputation (AUDC = 0.7987), and Native Tree Traversal Path Marginalization (AUDC = 0.8906), quantifying the off-manifold distribution shock gap at only $+0.0083$ AUDC units.
* **Chronological Market Drift Benchmark (WP 2.3):** Evaluated historical shift across 2008–2016 training and 2017–2019 modern live-ops testing, proving temporal robustness with minimal degradation (AUC-ROC 0.9099 to 0.8876, $\Delta \text{AUC} = -0.0223$).
* **Powered Human Trial (WP 2.4, N=32):** Balanced factorial trial demonstrated a $+28.8\%$ accuracy gain (59.4% to 88.1%, $p = 2.14 \times 10^{-9}$), $-52.3\%$ decision time savings (87.0s to 41.5s, $p = 1.00 \times 10^{-111}$), and moderate inter-rater agreement (Fleiss' Kappa $\kappa = 0.5444$ vs $-0.0123$ control discordance).
* **Automated Client Binary SDK Privacy Scanner (WP 2.7):** Static analyzer for iOS `Info.plist` / `PrivacyInfo.xcprivacy` and Android `AndroidManifest.xml` cross-referencing tracking SDKs against content ratings to prevent FTC COPPA liability.
* **50-Profile DiCE Recourse Feasibility (WP 2.8):** 94.0% class flip success rate (47/50 profiles successfully pivoted to freemium), average feature sparsity of 1.20 modified features, median $L_1$ proximity of 1.2311 MAD units (mean 2.5028), and 100% core gameplay identity preservation.


