"""Stand-in experiment: CV-tuned ensemble (balanced LightGBM + CV-tuned random forest,
probability average, no isotonic step) vs a standard random-forest risk score on the
public NASA KC1 defect data (OpenML 'kc1').
The defect label is used as a proxy for costly maintenance. Seeds 42, 43, 44."""
import json, socket, sys, time
import numpy as np
import pandas as pd
import openml
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

SEED, SEEDS, TEST_SIZE, N_MAX = 42, [42, 43, 44], 0.25, 200_000
SOURCE = "https://www.openml.org/search?type=data&q=kc1"
# Hyper-parameter grids, searched ONLY with 5-fold CV on the training split (scoring = ROC-AUC).
LGB_GRID = {"num_leaves": [7, 15], "n_estimators": [100, 300],
            "learning_rate": [0.03, 0.1], "min_child_samples": [10, 30],
            "colsample_bytree": [0.5, 1.0], "reg_lambda": [0.0, 5.0]}
RF_GRID = {"min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 0.3]}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def with_retry(fn, tries=5):
    """Call fn(); retry with exponential backoff on HTTP 429/5xx or transient errors."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            log(f"download failed ({e}); retrying in {2 ** i}s")
            time.sleep(2 ** i)

def load_kc1():
    socket.setdefaulttimeout(60)  # best-effort 60 s timeout for openml's HTTP calls
    table = with_retry(lambda: openml.datasets.list_datasets(data_name="kc1", output_format="dataframe"))
    table = table[table["name"] == "kc1"].sort_values("version")
    ds = with_retry(lambda: openml.datasets.get_dataset(int(table.iloc[0]["did"])))
    target = ds.default_target_attribute
    X, y, _, _ = with_retry(lambda: ds.get_data(dataset_format="dataframe", target=target))
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    y = y.astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)  # 1 = defective / costly
    return X, y

def metrics(y, p):
    return {"auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p),
            "f1": f1_score(y, (p >= 0.5).astype(int), average="macro")}

def baseline_probs(Xtr, ytr, Xte, seed):
    """Standard 300-tree random forest; raw predicted probabilities are the risk score."""
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    return rf.fit(Xtr, ytr).predict_proba(Xte)[:, 1]

def improved_probs(Xtr, ytr, Xte, seed):
    """Ensemble: class-balanced LightGBM and class-balanced random forest, each tuned by
    5-fold CV on the training part only (ROC-AUC). Final score = mean of the two
    predicted probabilities. No isotonic step (it is monotone but creates tied scores,
    which hurts ranking-based AUC on a small dataset)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    lgb_base = lgb.LGBMClassifier(class_weight="balanced", subsample=0.8, subsample_freq=1,
                                  random_state=seed, verbose=-1)
    gs_lgb = GridSearchCV(lgb_base, LGB_GRID, cv=skf, scoring="roc_auc", n_jobs=1).fit(Xtr, ytr)

    rf_base = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                     random_state=seed, n_jobs=-1)
    gs_rf = GridSearchCV(rf_base, RF_GRID, cv=skf, scoring="roc_auc", n_jobs=1).fit(Xtr, ytr)

    # Both best estimators are already refit on the full training part only (refit=True).
    p_lgb = gs_lgb.best_estimator_.predict_proba(Xte)[:, 1]
    p_rf = gs_rf.best_estimator_.predict_proba(Xte)[:, 1]
    p = 0.5 * p_lgb + 0.5 * p_rf
    params = {"lgbm": gs_lgb.best_params_, "rf": gs_rf.best_params_,
              "cv_auc_lgbm": float(gs_lgb.best_score_), "cv_auc_rf": float(gs_rf.best_score_)}
    return p, params

def main():
    log("loading OpenML 'kc1'")
    X, y = load_kc1()
    if len(X) > N_MAX:  # subsample very large data (not triggered for KC1)
        idx = X.sample(N_MAX, random_state=SEED).index
        X, y = X.loc[idx], y.loc[idx]
    if y.nunique() < 2 or X.shape[1] == 0:
        log("no usable data / single class"); sys.exit(1)
    # One stratified 75/25 split; median imputation fitted on train only, shared by both models
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med).values, Xte.fillna(med).values
    ytr, yte = ytr.values, yte.values
    log(f"rows={len(X)} train={len(ytr)} test={len(yte)} defect rate={y.mean():.3f}")

    res = {"baseline": [], "improved": []}
    for s in SEEDS:
        log(f"seed {s}: baseline random forest")
        res["baseline"].append(metrics(yte, baseline_probs(Xtr, ytr, Xte, s)))
        log(f"seed {s}: improved CV-tuned LightGBM + random forest ensemble")
        p, best = improved_probs(Xtr, ytr, Xte, s)
        res["improved"].append(metrics(yte, p))
        log(f"seed {s}: best CV params {best}")

    agg = {k: {m: (float(np.mean([r[m] for r in v])), float(np.std([r[m] for r in v])))
               for m in ["auc", "brier", "f1"]} for k, v in res.items()}
    vals = [agg[k][m][0] for k in agg for m in agg[k]]
    if not np.all(np.isfinite(vals)):
        log("non-finite metrics"); sys.exit(1)

    b, i = agg["baseline"], agg["improved"]
    out = {
        "question": "Does a class-balanced, CV-tuned tree ensemble rank software modules by future costly-maintenance risk better than a standard random-forest score on the public NASA KC1 defect dataset?",
        "method": "Stand-in experiment on OpenML 'kc1' (defect label as proxy for costly maintenance). One stratified 75/25 split (seed 42). Baseline: 300-tree random forest raw probabilities. Improved: average of a class-balanced LightGBM and a class-balanced random forest, each tuned by 5-fold ROC-AUC CV on the training split only. Models repeated over seeds 42, 43, 44.",
        "dataset": {"name": "kc1 (OpenML)", "source_url": SOURCE, "n_rows": int(len(X)),
                    "n_train": int(len(ytr)), "n_test": int(len(yte)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "ROC-AUC on test split",
        "higher_is_better": True,
        "baseline": "Standard 300-tree random forest; raw predicted defect probability used as risk score.",
        "improved": "Equal-weight probability average of a class-balanced LightGBM and a class-balanced random forest, hyper-parameters chosen by 5-fold CV ROC-AUC on training data only.",
        "metrics": {
            "baseline_score": b["auc"][0], "improved_score": i["auc"][0],
            "improvement": i["auc"][0] - b["auc"][0],
            "baseline_score_std": b["auc"][1], "improved_score_std": i["auc"][1],
            "baseline_brier": b["brier"][0], "improved_brier": i["brier"][0],
            "baseline_f1": b["f1"][0], "improved_f1": i["f1"][0],
        },
        "notes": ("KC1 defect label is a proxy for costly maintenance, not the paper's own data. "
                  "Single train/test split; seeds 42-44 vary only model randomness. Test set used once, at the end, never for tuning. "
                  "Isotonic calibration was dropped from the improved model because it produces tied scores that reduce ROC-AUC resolution. "
                  "Baseline and shared preprocessing are unchanged. "
                  "Timeout is best-effort via socket default timeout, since openml exposes no per-request timeout. "
                  "Small dataset (about 2k rows): differences may not be statistically significant."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"AUC baseline={b['auc'][0]:.4f} improved={i['auc'][0]:.4f}; wrote results.json")

    # Optional chart: AUC and Brier, baseline vs improved (mean +/- std over seeds)
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    names = ["baseline RF", "improved LGBM+RF ens."]
    ax[0].bar(names, [b["auc"][0], i["auc"][0]], yerr=[b["auc"][1], i["auc"][1]], capsize=4)
    ax[0].set_title("Test ROC-AUC (higher better)")
    ax[1].bar(names, [b["brier"][0], i["brier"][0]], yerr=[b["brier"][1], i["brier"][1]], capsize=4)
    ax[1].set_title("Test Brier score (lower better)")
    fig.tight_layout()
    fig.savefig("figures/baseline_vs_improved.png", dpi=120)

if __name__ == "__main__":
    main()
