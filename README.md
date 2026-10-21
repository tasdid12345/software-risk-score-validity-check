# Class-balanced LightGBM vs random-forest risk ranking on NASA KC1

Software sustainability models, such as the service-oriented approach in "Embedding the concept of service oriented architecture into software sustainability evaluation model" (2017), need ways to flag modules likely to become costly to maintain. A ranking score that orders modules by risk helps teams focus review and refactoring effort. This experiment asks whether a better-calibrated, class-balanced gradient-boosting score ranks modules better than a standard random-forest score.

**Question:** Does a class-balanced, calibrated gradient-boosting model rank software modules by future costly-maintenance risk better than a standard random-forest score on the public NASA KC1 defect dataset?

**Dataset:** OpenML "kc1" (static code metrics per module, defect label used as a proxy for costly maintenance), fetched with the `openml` Python package. This is a public stand-in dataset, not the paper's own data. One stratified 75/25 split with seed 42.

**Baseline:** a standard 300-tree random forest. Its raw predicted probabilities are the risk score.

**Improved model:** a class-balanced LightGBM. Hyper-parameters are chosen by 5-fold cross-validation on the training part only. Isotonic calibration is fitted on out-of-fold training predictions.

**Metrics:** main metric is ROC-AUC on the test split. Secondary metrics are the Brier score and macro-F1 at a 0.5 threshold. Both models are repeated over seeds 42, 43 and 44.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Requires internet access to openml.org. Runtime is a few minutes on CPU.

## Results

Results are written to `results.json`. A chart is saved to `figures/baseline_vs_improved.png`.

## Reference

This project is motivated by the following paper:

> R Ahmad, A Hussain, F Baharom (2017). *Embedding the concept of service oriented architecture into software sustainability evaluation model*. AIP Conference Proceedings. https://pubs.aip.org/aip/acp/article-abstract/1891/1/020023/887417

Paper author: Prof. Fauziah Baharom, Universiti Utara Malaysia (fauziah@uum.edu.my)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
