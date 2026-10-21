# Baseline vs CV-tuned tree model on OpenML KC1

Question: does a CV-tuned tree model rank KC1 software modules by defect risk (ROC-AUC on a held-out split) better than a standard random forest?

Run: `python analysis.py` (writes `results.json` and `figures/baseline_vs_improved.png`).

Method:
- Data: OpenML `kc1`, one stratified 75/25 split (seed 42). Median imputation fitted on the training part.
- Baseline: 300-tree random forest, raw predicted probabilities.
- Improved: LightGBM and random forest, each tuned by 5-fold ROC-AUC CV on the training part only. Grids include unweighted and class-balanced options. The final form (LightGBM, RF, or their average) is chosen by out-of-fold CV ROC-AUC on the training part.
- The test set is used only for the final scoring. Seeds 42, 43, 44.

Caveat: KC1's defect label is a proxy for costly maintenance. The dataset is small, so differences may not be significant.

## Reference

This project is motivated by the following paper:

> R Ahmad, A Hussain, F Baharom (2017). *Embedding the concept of service oriented architecture into software sustainability evaluation model*. AIP Conference Proceedings. https://pubs.aip.org/aip/acp/article-abstract/1891/1/020023/887417

Paper author: Prof. Fauziah Baharom, Universiti Utara Malaysia (fauziah@uum.edu.my)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
