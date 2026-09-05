# 🛡️ Self-Baselining LLM Substitution Detection
## The Complete, End-to-End Technical & Analytical Final Report

> **Document Name**: `FINNNNNNAAAAALreport.md`  
> **Author**: AI Infrastructure & Model Authenticity Team  
> **Target Audience**: Anyone seeking a complete, clear, and comprehensive explanation of LLM substitution detection—from basic intuition to advanced statistical proofs and empirical results.

---

## 📖 Executive Summary & Intuition (Start Here!)

### 1. The Real-World Problem: Silent Model Downgrades
Imagine you pay a company for access to a high-performance 70-Billion parameter AI model (like `llama3.2:3b` or `GPT-4`). To save hosting costs and increase profit margins, the cloud provider secretly switches your API endpoint to serve a smaller, cheaper, or lower-quality model (like `qwen2.5:3b` or a compressed 4-bit quantized version). 

Because LLM APIs are **black boxes**—they only give you text back—you cannot look inside their server memory to check the model weights. Traditional API keys and security certificates only verify *who owns the server*, not *which AI model generated the text*.

### 2. The Solution: Zero-Shot "Fingerprinting" Probes
Every AI model has unique internal preferences when asked open-ended questions. For example, if you repeatedly ask a model:
> *"Pick a random number between 1 and 100."*

Model A (`llama3.2:3b`) might favor numbers around 40–50 with a variance of 15, while Model B (`qwen2.5:3b`) might favor numbers around 60–70 with a tighter variance. 

By sending these simple, cheap single-token probes alongside regular user traffic, **ModelAuth** monitors the stream of numbers over time. When a provider secretly swaps the model, the probability distribution of these random numbers shifts. Our statistical detectors pick up this shift automatically and raise an alarm!

---

## 🏛️ Enterprise System Architecture & Corporate Directory Layout

The project adheres to a clean corporate hierarchy separating core code, technical documentation, raw specifications, and visual analytics:

```
modelauth/
├── README.md                          # Repository landing page & executive guide
├── FINNNNNNAAAAALreport.md            # THIS REPORT: Master end-to-end guide & analytical report
├── .gitignore                         # Repository git exclusion rules
├── docs/                              # Project Documentation Hub
│   ├── COMPLETE_PROJECT_REPORT.md     # Technical reference manual
│   ├── TEAM_REFERENCE_PROGRESS_REPORT.md # Team implementation reference
│   ├── EXPERIMENT_EVALUATION_GUIDE.md # JSONL stream schema & metric guide
│   └── source_docs/                   # Raw specification files (.docx)
│       ├── Final Steps.docx
│       └── gaps.docx
├── substitution-sim/                  # Core Simulation & Detection Engine Package
│   ├── config.py                      # Global experiment hyperparameters & model pairs
│   ├── probe_client.py                # Ollama REST API client
│   ├── simulator.py                   # Stream generator & switch point simulator
│   ├── run_experiments.py             # Resumable experiment suite runner
│   ├── run_cold_start_experiment.py   # Cold-start contamination stream generator
│   ├── data_loader.py                 # Regex numeric answer parser & stream loader
│   ├── detector_v1.py                 # Sliding-window 2-sample KS test detector
│   ├── detector_cusum.py              # Adaptive CUSUM detector
│   ├── detector_das_cusum.py          # DAS-CUSUM variance-sensitive detector
│   ├── detector_fixed_reference.py    # Static reference baseline detector
│   └── evaluate.py                    # Multi-tier evaluation & benchmark suite
└── final-analysis/                    # Analytics & Visual Reporting Package
    ├── sanity_checks.py               # Data completeness & model separability audit
    ├── visualizations.py              # Matplotlib trace, ROC, & contamination plots
    ├── interactive_dashboard.py       # HTML / Chart.js dashboard generator
    ├── run_final_steps.py             # Master analysis runner
    └── figures/                       # Output visual figures, CSV tables, & dashboards
        ├── example_trace_easy_rep0.png
        ├── roc_comparison_easy.png
        ├── cold_start_boundary.png
        ├── summary_table.csv
        ├── summary_table_all_tiers.csv
        └── dashboard.html
```

---

## 🔬 3. Mathematical Detector Formulations

```mermaid
flowchart TD
    A[Raw Probe Responses] --> B[data_loader.py: Extract Numeric Values X_t]
    B --> C{Detector Engine}
    C --> D["detector_v1.py<br/>Sliding-Window KS"]
    C --> E["detector_cusum.py<br/>Adaptive CUSUM"]
    C --> F["detector_das_cusum.py<br/>DAS-CUSUM"]
    C --> G["detector_fixed_reference.py<br/>Fixed Reference"]

    D --> H[2-sample Kolmogorov-Smirnov test on rolling windows]
    E --> I[Accumulate standardized z-score mean shifts]
    F --> J[Accumulate z-score mean & squared variance (z^2-1) shifts]
    G --> K[2-sample KS test against held-out clean reference stream]

    H --> L[Flag if p_value < 0.01]
    I --> M[Flag if S_t^+ > h or |S_t^-| > h]
    J --> N[Flag if S_t^das > h]
    K --> O[Flag if p_value < 0.01]
```

### 3.1 Kolmogorov-Smirnov Sliding Window (`detector_v1.py`)
Compares two adjacent trailing windows of size $W=20$:
$$W_1 = [X_{t-2W}, \dots, X_{t-W}], \quad W_2 = [X_{t-W}, \dots, X_t]$$
Using empirical cumulative distribution functions $F_{W_1}(x)$ and $F_{W_2}(x)$, it computes:
$$D = \sup_x |F_{W_1}(x) - F_{W_2}(x)|$$
Flags substitution when $p\text{-value} < \alpha = 0.01$.

### 3.2 Adaptive CUSUM (`detector_cusum.py`)
Maintains dynamic rolling estimates of mean ($\hat{\mu}$) and standard deviation ($\hat{\sigma}$):
$$z_t = \frac{X_t - \hat{\mu}}{\hat{\sigma} + \epsilon}$$
$$S_t^+ = \max(0, S_{t-1}^+ + z_t - k), \quad S_t^- = \min(0, S_{t-1}^- + z_t + k)$$
Flags when cumulative deviation exceeds decision threshold $h=5.0$.

### 3.3 Variance-Sensitive DAS-CUSUM (`detector_das_cusum.py`)
Tracks combined mean and variance shifts using a quadratic statistic:
$$v_t = 0.5 \cdot (z_t^2 - 1)$$
$$S_t^{\text{das}} = \max(0, S_{t-1}^{\text{das}} + v_t - k)$$
Flags when $S_t^{\text{das}} > h$.

### 3.4 Fixed-Reference Baseline Detector (`detector_fixed_reference.py`)
Compares incoming batches against a pre-collected, held-out reference dataset generated from a clean baseline run (`easy_null_rep14.jsonl`).

---

## 🛠️ 4. Key Engineering Fixes & Closed Gaps

### 1. Fix for Negative Detection Delays (Resolved Bug)
- **Problem**: Initial evaluations reported negative mean detection delays (e.g. $-80.0$ requests), which occurred because the evaluator picked up pre-switch false flags before $t=200$.
- **Fix**: Updated `compute_metrics` in `evaluate.py` to count only the **first valid flag occurring AT or AFTER the true switch point** ($t \ge 200$). Post-fix detection delays are strictly positive and accurate.

### 2. Multi-Tier & Held-Out Baseline Evaluation (RQ1 & RQ3)
- Integrated `fixed_reference_detector` using held-out baseline dataset `easy_null_rep14.jsonl` (ensuring test data isolation).
- Benchmark metrics now evaluate all 4 detectors side-by-side across repetitions.

### 3. Real Cold-Start Contamination Boundary (RQ2)
- Added `generate_contaminated_stream` to `simulator.py` and built `run_cold_start_experiment.py`.
- Generated 75 cold-start contamination stream logs across fractions $[0.0, 0.25, 0.5, 0.75, 1.0]$.
- Evaluated actual post-warmup recovery power ($>85\%$ recovery power up to $25\%$ initial history contamination).

---

## 📊 5. Empirical Results & Complete 3-Tier Performance Benchmark

| Difficulty Tier | Model Pair ($A \rightarrow B$) | Nature of Substitution | Detector Method | Mean Detection Delay ($\tau - T$) | Detection Rate (Power) | False Alarm Rate ($\alpha$) | Performance Assessment |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Easy Tier** | `llama3.2:3b` $\rightarrow$ `qwen2.5:3b` | Cross-Architecture | **`v1 naive`** *(Sliding Window KS)* | **+15.33 probes** | **85.71%** | **0.00%** | **Fastest & Zero False Alarms** |
| **Easy Tier** | `llama3.2:3b` $\rightarrow$ `qwen2.5:3b` | Cross-Architecture | **`adaptive CUSUM`** | **+11.00 probes** | **78.57%** | **0.42%** | **Lowest Delay Post-Switch** |
| **Easy Tier** | `llama3.2:3b` $\rightarrow$ `qwen2.5:3b` | Cross-Architecture | **`DAS-CUSUM`** | **+53.00 probes** | **57.14%** | **0.38%** | Robust to Variance Shifts |
| **Easy Tier** | `llama3.2:3b` $\rightarrow$ `qwen2.5:3b` | Cross-Architecture | **`fixed-reference`** *(Held-Out)* | **+20.00 probes** | **100.00%** | **0.36%** | **100% Detection Power** |
| **Medium Tier** | `llama3.2:1b` $\rightarrow$ `llama3.2:3b` | Capacity/Scale Shift | **`v1 naive`** *(Sliding Window KS)* | **+14.50 probes** | **14.29%** | **0.16%** | Power collapses on subtle shift |
| **Medium Tier** | `llama3.2:1b` $\rightarrow$ `llama3.2:3b` | Capacity/Scale Shift | **`adaptive CUSUM`** | **+41.15 probes** | **92.86%** | **0.08%** | **Top Self-Baselined Power (92.86%)** |
| **Medium Tier** | `llama3.2:1b` $\rightarrow$ `llama3.2:3b` | Capacity/Scale Shift | **`DAS-CUSUM`** | **+83.55 probes** | **78.57%** | **0.00%** | **Zero False Alarms (0.00%)** |
| **Medium Tier** | `llama3.2:1b` $\rightarrow$ `llama3.2:3b` | Capacity/Scale Shift | **`fixed-reference`** *(Held-Out)* | **+22.86 probes** | **100.00%** | **0.75%** | **100% Detection Power** |
| **Hard Tier** | `llama3.2:3b-q4` $\rightarrow$ `3b-q8` | Quantization Shift | **`v1 naive`** *(Sliding Window KS)* | **+126.00 probes** | **28.57%** | **0.00%** | High Delay on subtle precision drift |
| **Hard Tier** | `llama3.2:3b-q4` $\rightarrow$ `3b-q8` | Quantization Shift | **`adaptive CUSUM`** | **+71.20 probes** | **71.43%** | **0.58%** | **Top Quantization Power (71.43%)** |
| **Hard Tier** | `llama3.2:3b-q4` $\rightarrow$ `3b-q8` | Quantization Shift | **`DAS-CUSUM`** | **+88.75 probes** | **57.14%** | **0.54%** | Variance-Sensitive Drift Tracking |
| **Hard Tier** | `llama3.2:3b-q4` $\rightarrow$ `3b-q8` | Quantization Shift | **`fixed-reference`** *(Held-Out)* | **+90.00 probes** | **14.29%** | **0.36%** | Requires larger batch integration |

> [!IMPORTANT]
> **Key Scientific Insights across Tiers**:
> 1. **Cross-Architecture Shifts (Easy: LLaMA $\rightarrow$ Qwen)**: High distribution separability (KS = 0.659, $p < 10^{-270}$) allows local sliding-window detectors (`v1 naive`) to detect the switch rapidly (+15.33 probes) with 85.71% power.
> 2. **Intra-Family Parameter Shifts (Medium: 1B $\rightarrow$ 3B)**: More subtle distribution shift (KS = 0.402, $p < 10^{-96}$). The naive sliding window KS lacks memory and drops to 14.29% power. In contrast, **Adaptive CUSUM** accumulates small persistent standardized drifts, achieving **92.86% detection power** with only **0.08% false alarms**.
> 3. **Quantization Precision Shifts (Hard: Q4_K_M $\rightarrow$ Q8_0)**: The most challenging regime where model weights share the exact same architecture and parameter counts, varying only by quantization compression. **Adaptive CUSUM** maintains the highest self-baselined power (**71.43%**, +71.2 probes), whereas non-parametric sliding windows suffer severe delays (+126 probes).
> 4. **Fixed-Reference Upper Bound**: When clean held-out reference distributions are stored, `fixed-reference` achieves **100% detection power** across Easy (+20 probes) and Medium (+22.86 probes) tiers with near-zero false alarms.

---

## 🖼️ 6. Visualizations & Multi-Tier Analytics

### 6.1 Single Stream Response Traces & Switch Points

| Easy Tier (`llama3.2:3b` $\rightarrow$ `qwen2.5:3b`) | Medium Tier (`llama3.2:1b` $\rightarrow$ `llama3.2:3b`) | Hard Tier (`llama3.2:3b-q4` $\rightarrow$ `3b-q8`) |
| :---: | :---: | :---: |
| ![Example Trace — Easy](final-analysis/figures/example_trace_easy_rep0.png) | ![Example Trace — Medium](final-analysis/figures/example_trace_medium_rep0.png) | ![Example Trace — Hard](final-analysis/figures/example_trace_hard_rep0.png) |

*Figure 1: Numerical response streams across 400 probes for Easy, Medium, and Hard tiers. The red dashed line marks the ground-truth substitution point ($t=200$), and the green dotted line marks the detector's automated flag.*

---

### 6.2 ROC Delay vs. False Alarm Rate Trade-Off Curves

| Easy Tier ROC Curve | Medium Tier ROC Curve | Hard Tier ROC Curve |
| :---: | :---: | :---: |
| ![ROC Curve — Easy](final-analysis/figures/roc_comparison_easy.png) | ![ROC Curve — Medium](final-analysis/figures/roc_comparison_medium.png) | ![ROC Curve — Hard](final-analysis/figures/roc_comparison_hard.png) |

*Figure 2: Receiver Operating Characteristic (ROC) trade-off curves mapping False Alarm Rate ($X$-axis) against Mean Detection Delay ($Y$-axis) across all difficulty tiers.*

---

### 6.3 Complete Multi-Tier Benchmark Comparison (Power & Delay)

![Multi-Tier Benchmark Comparison](final-analysis/figures/multi_tier_benchmark_comparison.png)

*Figure 3: Side-by-side grouped bar chart comparing Detection Power (%) and Mean Detection Delay (probes) across all 4 detectors for Easy, Medium, and Hard difficulty tiers.*

---

### 6.4 Model Output Distribution Separability (Architecture vs Scale vs Quantization)

![Distribution Separability All Tiers](final-analysis/figures/distribution_separability_all_tiers.png)

*Figure 4: Empirical probability density distributions of single-token probe responses across all 3 tiers. Easy Tier (cross-architecture) shows distinct modal separation ($KS=0.659, p<10^{-270}$), Medium Tier (parameter shift) exhibits moderate density shifts ($KS=0.402, p<10^{-96}$), and Hard Tier (quantization shift) captures subtle precision differences.*

---

### 6.5 Cold-Start Baseline Contamination Boundary

![Cold-Start Contamination Power Boundary](final-analysis/figures/cold_start_boundary.png)

*Figure 5: Cold-start contamination boundary showing recovery power as a function of pre-monitoring baseline contamination ($0\%$ to $100\%$).*

---

## 🌐 7. Interactive Dashboard

An interactive dashboard with Chart.js analytics widgets is available in your browser:
🔗 [Open Interactive Dashboard HTML](file:///d:/Praneeth/Work/modelauth/final-analysis/figures/dashboard.html)

---

## 🚀 8. How to Run the System

To re-run the entire pipeline from scratch on Windows 11:

```powershell
# 1. Run full simulation experiments across Easy, Medium, or Hard tiers
cd substitution-sim
python run_experiments.py easy
python run_experiments.py medium
python run_experiments.py hard

# 2. Run detector evaluations and statistical audits
python evaluate.py

# 3. Generate all figures, summary tables, and dashboard
cd ../final-analysis
python run_final_steps.py
python interactive_dashboard.py
```

All summary tables (`summary_table_all_tiers.csv`), figures (`.png`), and interactive dashboards (`dashboard.html`) will update automatically.
