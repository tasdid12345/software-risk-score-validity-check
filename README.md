# KC1 costly-maintenance risk: baseline RF vs CV-tuned ensemble

Stand-in experiment on OpenML `kc1` (defect label used as a proxy for costly maintenance).

- Split: one stratified 75/25 split, seed 42. Median imputation fitted on train only.
- Baseline (unchanged): 300-tree random forest, raw predicted probabilities as the risk score.
- Improved: equal-weight probability average of a class-balanced LightGBM and a class-balanced random forest. Each is tuned by 5-fold ROC-AUC cross-validation on the training split only. The isotonic step was removed.
- Metric: ROC-AUC on the test split, averaged over seeds 42, 43, 44.

Run: `pip install -r requirements.txt && python analysis.py`. This writes `results.json` and `figures/baseline_vs_improved.png`.

The test set is used only for the final evaluation. Small dataset (about 2k rows), so differences may not be statistically significant.

## Reference

This project is motivated by the following paper:

> R Ahmad, A Hussain, F Baharom (2017). *Embedding the concept of service oriented architecture into software sustainability evaluation model*. AIP Conference Proceedings. https://pubs.aip.org/aip/acp/article-abstract/1891/1/020023/887417

Paper author: Prof. Fauziah Baharom, Universiti Utara Malaysia (fauziah@uum.edu.my)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
