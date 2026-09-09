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

## 📊 5. Empirical Results & Easy Tier Performance Benchmark

| Detector Method | Mean Detection Delay ($\tau - T$) | Detection Rate (Power) | False Alarm Rate ($\alpha$) | Performance Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **`v1 naive`** *(Sliding Window KS)* | **+15.33 probes** | **85.71%** | **0.00%** | **Fastest & Zero False Alarms** |
| **`adaptive CUSUM`** | **+11.00 probes** | **78.57%** | **0.42%** | **Lowest Delay Post-Switch** |
| **`DAS-CUSUM`** | **+53.00 probes** | **57.14%** | **0.38%** | Robust to Variance Shifts |
| **`fixed-reference`** *(Held-Out)* | **+20.00 probes** | **100.00%** | **0.36%** | **100% Detection Power** |

> [!IMPORTANT]
> **Key Scientific Insights for Easy Tier (`llama3.2:3b` $\rightarrow$ `qwen2.5:3b`)**:
> 1. **Cross-Architecture Shifts**: High distribution separability ($KS = 0.659, p < 10^{-270}$) allows local sliding-window detectors (`v1 naive`) to detect the switch rapidly (+15.33 probes) with 85.71% power and zero false alarms.
> 2. **Adaptive CUSUM Responsiveness**: Adaptive CUSUM registers the fastest initial post-switch flag at an average of **+11.00 probes**, making it the premier choice for low-latency alerting.
> 3. **Fixed-Reference Upper Bound**: When clean held-out reference distributions are stored (`easy_null_rep14.jsonl`), `fixed-reference` achieves **100% detection power** (+20.0 probes) with near-zero false alarms ($0.36\%$).

---

## 🖼️ 6. Visualizations & Easy Tier Analytics

### 6.1 Single Stream Response Trace & Switch Point

![Example Trace — Easy](final-analysis/figures/example_trace_easy_rep0.png)

*Figure 1: Numerical response stream across 400 probes for Easy Tier. The red dashed line marks the ground-truth substitution point ($t=200$), and the green dotted line marks the detector's automated flag.*

---

### 6.2 ROC Delay vs. False Alarm Rate Trade-Off Curve

![ROC Curve — Easy](final-analysis/figures/roc_comparison_easy.png)

*Figure 2: Receiver Operating Characteristic (ROC) trade-off curve mapping False Alarm Rate ($X$-axis) against Mean Detection Delay ($Y$-axis) for Easy Tier.*

---

### 6.3 4-Detector Benchmark Comparison (Power & Delay)

![Easy Tier Detector Benchmark Comparison](final-analysis/figures/detector_comparison_easy.png)

*Figure 3: Side-by-side bar chart comparing Detection Power (%) and Mean Detection Delay (probes) across all 4 detectors for the Easy Tier.*

---

### 6.4 Model Output Distribution Separability

![Distribution Separability Easy](final-analysis/figures/distribution_separability_easy.png)

*Figure 4: Empirical probability density distributions of single-token probe responses between Model A (`llama3.2:3b`) and Model B (`qwen2.5:3b`), demonstrating distinct modal separation ($KS=0.659, p<10^{-270}$).*

---

### 6.5 Cold-Start Baseline Contamination Boundary

![Cold-Start Contamination Power Boundary](final-analysis/figures/cold_start_boundary.png)

*Figure 5: Cold-start contamination boundary showing recovery power as a function of pre-monitoring baseline contamination ($0\%$ to $100\%$).*

---

## 🌐 7. Interactive Dashboard

An interactive dashboard with Chart.js analytics widgets is available in your browser:
🔗 `final-analysis/figures/dashboard.html`

---

## 🚀 8. How to Run the System

To re-run the entire Easy Tier pipeline:

```bash
# 1. Run detector evaluations and statistical audits
cd substitution-sim
python evaluate.py

# 2. Generate all figures, summary tables, and dashboard
cd ../final-analysis
python run_final_steps.py
python interactive_dashboard.py
```

All summary tables (`summary_table.csv`), figures (`.png`), and interactive dashboards (`dashboard.html`) will update automatically.

