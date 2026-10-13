# Presentation-Ready Results Package: DNN vs. DNN-Feature ELM for ASL-MRI

Welcome to the definitive, presentation-ready results package for the **DNN vs. DNN-Feature ELM** Arterial Spin Labeling (ASL) MRI parameter estimation project.

---

## 1. Quick Navigation: What to Open First

1. **Executive Summary & Report:**
   - [`FINAL_RESULTS_REPORT.md`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/FINAL_RESULTS_REPORT.md): The full, authoritative scientific presentation report covering background, methodology, statistical verification, and conclusions.
2. **Authoritative Metric Tables:**
   - [`model_comparison.md`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/model_comparison.md) and [`tables/final_metrics.csv`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/tables/final_metrics.csv): Side-by-side locked-test metrics for Baseline A (DNN-G) and the 5-Seed Ensemble Feature ELM.
   - [`tables/snr_stratified_metrics.csv`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/tables/snr_stratified_metrics.csv): Performance broken down across all 12 acquisition SNR levels.
   - [`tables/att_interval_metrics.csv`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/tables/att_interval_metrics.csv): Breakdown across physiological transit windows.

---

## 2. Top Five Graphs for Project Presentation / Slides

If preparing a 5-to-10 minute presentation or a PowerPoint slide deck, we recommend featuring these five figures:

1. **[`graphs/01_cbf_rmse_comparison.png`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/graphs/01_cbf_rmse_comparison.png)**  
   *Slide Topic: CBF Accuracy Benchmark.* Shows exact CBF RMSE ($3.7266$ down to $3.6849\text{ mL}/(100\text{g}\cdot\text{min})$, a $+1.12\%$ improvement).
2. **[`graphs/02_att_rmse_comparison.png`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/graphs/02_att_rmse_comparison.png)**  
   *Slide Topic: ATT Accuracy Benchmark.* Shows exact ATT RMSE ($0.3058\text{ s}$ down to $0.3048\text{ s}$, an absolute improvement of $1\text{ millisecond}$ or $+0.31\%$), demonstrating that downstream readouts do not meaningfully improve transit estimation.
3. **[`graphs/05_percentage_improvement.png`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/graphs/05_percentage_improvement.png)**  
   *Slide Topic: Balanced Honest Summary.* Displays all percentage improvements side-by-side with a clear zero reference line, highlighting positive gains on CBF alongside the slight negative degradation in early-ATT ($-0.64\%$).
4. **[`graphs/06_cbf_prediction_scatter.png`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/graphs/06_cbf_prediction_scatter.png)**  
   *Slide Topic: Prediction Fidelity & Bias Removal.* Contrasts 5,000 ground truth vs. predicted CBF points, showing how Feature ELM centers the distribution and eliminates negative bias ($-0.173$ to $-0.008$).
5. **[`graphs/10_att_interval_performance.png`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/graphs/10_att_interval_performance.png)**  
   *Slide Topic: Physical Limitations of Late-PLD Acquisitions.* Demonstrates that early bolus arrival ($[0.5, 1.0)\text{ s}$) cannot be recovered by either model because the earliest PLD is $1.525\text{ s}$.

---

## 3. Directory Layout

```text
results/BEST_RESULTS_PRESENTATION/
├── README.md                           # This quick start guide
├── FINAL_RESULTS_REPORT.md             # Authoritative scientific presentation report
├── model_comparison.csv                # Primary metric comparison table (CSV)
├── model_comparison.md                 # Primary metric comparison table (Markdown)
├── reproduce_all.py                    # One-click reproduction script
├── graphs/                             # 11 individual 300-DPI publication PNGs
│   ├── 01_cbf_rmse_comparison.png
│   ├── 02_att_rmse_comparison.png
│   ├── 03_cbf_mae_comparison.png
│   ├── 04_att_mae_comparison.png
│   ├── 05_percentage_improvement.png
│   ├── 06_cbf_prediction_scatter.png
│   ├── 07_att_prediction_scatter.png
│   ├── 08_error_distribution.png
│   ├── 09_snr_robustness.png
│   ├── 10_att_interval_performance.png
│   └── 11_paired_error_difference.png
├── tables/                             # CSV exports of locked-test evaluation
│   ├── final_metrics.csv
│   ├── snr_stratified_metrics.csv
│   └── att_interval_metrics.csv
├── selected_models/                    # Checkpoint paths and architecture manifests
│   ├── model_manifest.json
│   └── README.md
└── reproducibility/                    # Environment details & reproduction guide
    ├── environment.txt
    ├── evaluation_config.json
    ├── reproduce_all.py
    ├── reproduction_instructions.md
    └── source_commit.txt
```

