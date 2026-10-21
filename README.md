# Baseline vs CV-tuned tree model on OpenML KC1

Question: does a CV-tuned tree model rank KC1 software modules by defect risk (ROC-AUC on a held-out split) better than a standard random forest?

Run: `python analysis.py` (writes `results.json` and `figures/baseline_vs_improved.png`).

Method:
- Data: OpenML `kc1`, one stratified 75/25 split (seed 42). Median imputation fitted on the training part.
- Baseline: 300-tree random forest, raw predicted probabilities.
- Improved: LightGBM and random forest, each tuned by 5-fold ROC-AUC CV on the training part only. Grids include unweighted and class-balanced options. The final form (LightGBM, RF, or their average) is chosen by out-of-fold CV ROC-AUC on the training part.
- The test set is used only for the final scoring. Seeds 42, 43, 44.

Caveat: KC1's defect label is a proxy for costly maintenance. The dataset is small, so differences may not be significant.

## Results (latest run)

On the public OpenML KC1 dataset (defect label as a proxy for costly maintenance), a standard 300-tree random forest on raw probabilities reached a test ROC-AUC of 0.836, while my class-balanced, CV-tuned LightGBM/RF ensemble reached 0.837, a negligible gain of 0.001 within seed-to-seed spread.

| Metric | Value |
|---|---|
| baseline_score | 0.8365 |
| improved_score | 0.8375 |
| improvement | 0.001003 |
| baseline_score_std | 0.001715 |
| improved_score_std | 0.001343 |
| baseline_brier | 0.09884 |
| improved_brier | 0.09811 |
| baseline_f1 | 0.7006 |
| improved_f1 | 0.6535 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: KC1 defect label is a proxy for costly maintenance, not the paper's own data. Single train/test split; seeds 42-44 vary only model randomness. Test set used once, at the end, never for tuning or model selection. Model-form selection (LightGBM vs RF vs average) uses out-of-fold CV AUC on the training split only. Isotonic calibration is not used. Baseline and shared preprocessing are unchanged. Timeout is best-effort via socket default timeout, since openml exposes no per-request timeout. Small dataset (about 2k rows): differences may not be statistically significant.

## Reference

This project is motivated by the following paper:

> R Ahmad, A Hussain, F Baharom (2017). *Embedding the concept of service oriented architecture into software sustainability evaluation model*. AIP Conference Proceedings. https://pubs.aip.org/aip/acp/article-abstract/1891/1/020023/887417

Paper author: Prof. Fauziah Baharom, Universiti Utara Malaysia (fauziah@uum.edu.my)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
