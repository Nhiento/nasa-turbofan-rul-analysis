"""
================================================================================
NASA C-MAPSS Turbofan Engine RUL Analysis
Step 4: Remaining Useful Life (RUL) prediction model
================================================================================
Question: "How many cycles does this engine have left?"

The Health Index (step 2) says HOW WORN an engine is. This step turns that into
a forecast in cycles, which is what a maintenance planner actually schedules by.

How it works, in plain terms
  1. Inputs  : the 84 rolling sensor features, the Health Index, how fast the
               Health Index is rising (its 10-cycle slope), and the cycle count.
               Nothing that depends on the failure date is used as an input.
  2. Target  : RUL capped at 125 cycles (early wear is invisible, see step 2).
  3. Models  : a ladder, each one has to beat the one before it
                 a. Baseline         always guess the average   (the floor)
                 b. Linear (Ridge)   a weighted sum of the inputs
                 c. Gradient boosting many small decision trees, each fixing
                                     the previous trees' mistakes
  4. Checking: 5-fold cross-validation GROUPED BY ENGINE, so an engine is never
               in training and validation at the same time.
  5. Final test: each of the 100 test engines is predicted ONCE, at its last
               recorded cycle, and compared with NASA's answer key.
               Scored with RMSE (typical miss, in cycles) and NASA's scoring
               function, which punishes LATE predictions harder than early ones.
  6. Range  : two extra models give a likely range (10th to 90th percentile),
              e.g. "about 18 cycles left, likely 10 to 30".

Inputs : data/processed/cleaned_master.csv, cleaned_test.csv, test_rul.csv
Outputs: data/processed/rul_predictions.csv, model_results.json,
         data/figures/06–08_*.png, table rul_predictions in data/turbofan.db
================================================================================
"""

import json
import sqlite3
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "data" / "figures"
DB_PATH = BASE_DIR / "data" / "turbofan.db"

RUL_CAP = 125
SEED = 42
SLOPE_WINDOW = 10      # cycles used to measure how fast the Health Index is rising

# Maintenance actions from the predicted RUL (same bands as the Health Index alerts)
SCHEDULE_NOW = 25      # predicted < 25 cycles left  -> "Schedule now"
PLAN_AHEAD = 50        # predicted 25–49 cycles left -> "Plan within next window"
PERSISTENCE = 3        # same "3 cycles in a row" rule the Health Index alerts use

SENSOR_NAMES = {
    "sensor_02": "LPC outlet temp", "sensor_03": "HPC outlet temp",
    "sensor_04": "LPT outlet temp", "sensor_07": "HPC outlet pressure",
    "sensor_08": "Fan speed", "sensor_09": "Core speed",
    "sensor_11": "HPC static pressure", "sensor_12": "Fuel flow ratio",
    "sensor_13": "Corrected fan speed", "sensor_14": "Corrected core speed",
    "sensor_15": "Bypass ratio", "sensor_17": "Bleed enthalpy",
    "sensor_20": "HPT coolant bleed", "sensor_21": "LPT coolant bleed",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA AND FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def add_hi_slope(df: pd.DataFrame) -> pd.DataFrame:
    """Health Index change per cycle over the last 10 cycles (trailing only).
    Engines that wear fast are caught by the slope even before the level is high."""
    df = df.sort_values(["engine_id", "cycle"]).copy()
    change = df.groupby("engine_id")["health_index"].diff(SLOPE_WINDOW)
    df["hi_slope_10"] = (change / SLOPE_WINDOW).fillna(0.0)   # first cycles: no trend yet
    return df


def load():
    train = add_hi_slope(pd.read_csv(PROC / "cleaned_master.csv"))
    test = add_hi_slope(pd.read_csv(PROC / "cleaned_test.csv"))
    answer_key = pd.read_csv(PROC / "test_rul.csv")
    src = (PROC / "data_source.txt").read_text().strip() if (PROC / "data_source.txt").exists() else "unknown"

    rolling = [c for c in train.columns if "_roll" in c]
    features = rolling + ["health_index", "hi_slope_10", "cycle"]
    leaky = {"RUL", "RUL_capped", "rul_band", "cycle_pct"}
    assert not leaky & set(features), "a failure-time column slipped into the inputs"

    print(f"  Data source: {src}")
    print(f"  Train: {train.engine_id.nunique()} engines, {len(train):,} rows | "
          f"Test: {test.engine_id.nunique()} engines | {len(features)} input features")
    return train, test, answer_key, features, src


# ─────────────────────────────────────────────────────────────────────────────
# 2. THE MODELS
# ─────────────────────────────────────────────────────────────────────────────
def make_models():
    return {
        "Baseline (average)": DummyRegressor(strategy="mean"),
        "Linear (Ridge)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Gradient boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=50, random_state=SEED),
    }


def gb_quantile(q: float):
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=300, learning_rate=0.05,
        max_leaf_nodes=15, min_samples_leaf=50, random_state=SEED)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """NASA's scoring function (Saxena et al. 2008). d = predicted − true.
    Late predictions (d > 0) grow faster (÷10) than early ones (÷13). Lower is better."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13), np.exp(d / 10)) - 1))


# ─────────────────────────────────────────────────────────────────────────────
# 3. CROSS-VALIDATION, GROUPED BY ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def cross_validate(train, features):
    X, y, groups = train[features], train["RUL_capped"], train["engine_id"]
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))
    results, oof = {}, {}
    for name, model in make_models().items():
        pred = np.zeros(len(train))
        for tr, va in folds:
            model.fit(X.iloc[tr], y.iloc[tr])
            pred[va] = np.clip(model.predict(X.iloc[va]), 0, RUL_CAP)
        oof[name] = pred
        per_fold = [rmse(y.iloc[va], pred[va]) for _, va in folds]
        results[name] = {"cv_rmse": round(rmse(y, pred), 2),
                         "cv_rmse_fold_std": round(float(np.std(per_fold)), 2)}
        print(f"  [CV] {name:<20} RMSE {results[name]['cv_rmse']:6.2f} cycles "
              f"(± {results[name]['cv_rmse_fold_std']:.2f} across folds)")
    return results, oof["Gradient boosting"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. FINAL TEST: ONE PREDICTION PER TEST ENGINE, AT ITS LAST CYCLE
# ─────────────────────────────────────────────────────────────────────────────
def test_evaluation(train, test, answer_key, features, results):
    X, y = train[features], train["RUL_capped"]
    last = (test.sort_values("cycle").groupby("engine_id").tail(1)
            .merge(answer_key, on="engine_id").sort_values("engine_id"))
    truth = last["true_rul"].to_numpy()
    fitted = {}
    for name, model in make_models().items():
        model.fit(X, y)
        pred = np.clip(model.predict(last[features]), 0, RUL_CAP)
        fitted[name] = (model, pred)
        results[name].update({
            "test_rmse": round(rmse(truth, pred), 2),
            "test_rmse_vs_capped_truth": round(rmse(np.minimum(truth, RUL_CAP), pred), 2),
            "test_nasa_score": round(nasa_score(truth, pred), 1),
            "test_mean_error": round(float(np.mean(pred - truth)), 2),
        })
        r = results[name]
        print(f"  [TEST] {name:<20} RMSE {r['test_rmse']:6.2f} | NASA score {r['test_nasa_score']:9.1f} "
              f"| avg error {r['test_mean_error']:+.1f} (+ = late)")

    best_model, best_pred = fitted["Gradient boosting"]
    lo = np.clip(gb_quantile(0.10).fit(X, y).predict(last[features]), 0, RUL_CAP)
    hi = np.clip(gb_quantile(0.90).fit(X, y).predict(last[features]), 0, RUL_CAP)
    lo, hi = np.minimum(lo, best_pred), np.maximum(hi, best_pred)

    out = pd.DataFrame({
        "engine_id": last["engine_id"].to_numpy(),
        "last_cycle": last["cycle"].to_numpy(),
        "health_index": last["health_index"].round(4).to_numpy(),
        "hi_alert_level": last["alert_level"].to_numpy(),
        "predicted_rul": best_pred.round(1),
        "rul_low_p10": lo.round(1),
        "rul_high_p90": hi.round(1),
        "true_rul": truth,
    })
    out["error"] = (out["predicted_rul"] - out["true_rul"]).round(1)
    out["model_action"] = np.select(
        [out["predicted_rul"] < SCHEDULE_NOW, out["predicted_rul"] < PLAN_AHEAD],
        ["Schedule now", "Plan within next window"], default="No action")
    capped = np.minimum(truth, RUL_CAP)
    coverage = float(np.mean((capped >= out["rul_low_p10"]) & (capped <= out["rul_high_p90"])))
    print(f"  [RANGE] true RUL (capped at {RUL_CAP}) inside the 10–90% range for "
          f"{coverage:.0%} of test engines")
    return out, best_model, last, coverage


# ─────────────────────────────────────────────────────────────────────────────
# 5. DOES THE MODEL FIX THE HEALTH-INDEX MISSES?
# ─────────────────────────────────────────────────────────────────────────────
def compare_with_alerts(train, oof, preds):
    summary = {}

    # Model alert rule, held to the SAME 3-cycles-in-a-row rule as the Health Index:
    # "predicted RUL < 25 for 3 cycles in a row". Uses out-of-fold predictions, so
    # every training engine is judged by a model that never saw it.
    t = train[["engine_id", "cycle", "RUL"]].copy()
    t["below"] = (oof < SCHEDULE_NOW).astype(int)
    run = t.groupby("engine_id")["below"].transform(
        lambda b: b.groupby((b != b.shift()).cumsum()).cumsum())
    first = t[run >= PERSISTENCE].groupby("engine_id")["RUL"].max()   # RUL at first sustained alert
    warning = first.reindex(sorted(t.engine_id.unique()))
    summary["model_alert_rule"] = {
        "engines_alerted": int(warning.notna().sum()),
        "median_warning_cycles": float(warning.median()),
        "warning_iqr": [float(warning.quantile(.25)), float(warning.quantile(.75))],
        "warning_min_max": [float(warning.min()), float(warning.max())],
    }
    m = summary["model_alert_rule"]
    print(f"  [MODEL ALERTS] predicted < {SCHEDULE_NOW} for {PERSISTENCE} cycles in a row: "
          f"{m['engines_alerted']}/100 training engines alerted, median warning "
          f"{m['median_warning_cycles']:.0f} cycles (IQR {m['warning_iqr'][0]:.0f}–{m['warning_iqr'][1]:.0f}, "
          f"min {m['warning_min_max'][0]:.0f}, max {m['warning_min_max'][1]:.0f})")

    # Training engines that never raised a sustained Health Index alert
    eng = pd.read_csv(PROC / "powerbi_engine_summary.csv")
    missed = eng.loc[eng["first_alert_cycle"].isna(), "engine_id"].tolist()
    caught = [{"engine_id": int(e), "flagged": bool(pd.notna(warning.get(e))),
               "cycles_of_warning": int(warning[e]) if pd.notna(warning.get(e)) else 0}
              for e in missed]
    summary["train_engines_missed_by_hi"] = caught
    print(f"  [MISSES] training engines the Health Index never alerted on: {missed}")
    for c in caught:
        print(f"           engine {c['engine_id']}: model alert {c['cycles_of_warning']} cycles "
              f"before failure" if c["flagged"] else f"           engine {c['engine_id']}: model also missed it")

    # Test engines the Health Index called "Normal" with < 25 cycles left
    n_late = preds[(preds.hi_alert_level == "Normal") & (preds.true_rul < SCHEDULE_NOW)]
    summary["test_normal_but_under_25"] = n_late[["engine_id", "true_rul", "predicted_rul",
                                                  "model_action"]].to_dict("records")
    for _, r in n_late.iterrows():
        print(f"           test engine {r.engine_id}: HI said Normal, true RUL {r.true_rul}, "
              f"model predicts {r.predicted_rul:.0f} → {r.model_action}")

    # Same check the SQL view vw_alert_check does, but for the model's actions
    band = pd.cut(preds["true_rul"], [-1, 24, 49, np.inf], labels=["<25", "25-49", "50+"])
    table = pd.crosstab(preds["model_action"], band)
    print("  [ACTIONS vs TRUTH] rows = model action, columns = true cycles left")
    print("    " + table.to_string().replace("\n", "\n    "))
    summary["action_vs_truth"] = {a: {str(k): int(v) for k, v in row.items()}
                                  for a, row in table.iterrows()}

    # ── Combined alert: raise it when EITHER the Health Index alert OR the model alert fires ──
    # Training engines: warning time = whichever of the two fired first.
    hi_lead = eng.set_index("engine_id")["alert_lead_cycles"].reindex(warning.index)
    combined = pd.concat([hi_lead, warning], axis=1).max(axis=1)
    # Test engines: "urgent" = HI level Alert OR model says Schedule now
    urgent = (preds["hi_alert_level"] == "Alert") | (preds["model_action"] == "Schedule now")
    under25, over50 = preds["true_rul"] < SCHEDULE_NOW, preds["true_rul"] >= PLAN_AHEAD
    summary["combined_alert"] = {
        "rule": "Health Index Alert (3 in a row) OR predicted RUL < 25 (3 in a row)",
        "train_engines_alerted": int(combined.notna().sum()),
        "train_median_warning_cycles": float(combined.median()),
        "train_warning_iqr": [float(combined.quantile(.25)), float(combined.quantile(.75))],
        "train_min_warning_cycles": float(combined.min()),
        "test_urgent_flags": int(urgent.sum()),
        "test_urgent_with_under_25_left": int((urgent & under25).sum()),
        "test_engines_with_under_25_left": int(under25.sum()),
        "test_urgent_with_50_plus_left": int((urgent & over50).sum()),
    }
    c = summary["combined_alert"]
    print(f"  [COMBINED] HI alert OR model alert: {c['train_engines_alerted']}/100 training engines alerted, "
          f"median warning {c['train_median_warning_cycles']:.0f} cycles "
          f"(IQR {c['train_warning_iqr'][0]:.0f}–{c['train_warning_iqr'][1]:.0f}, min {c['train_min_warning_cycles']:.0f})")
    print(f"             test: {c['test_urgent_flags']} urgent flags, {c['test_urgent_with_under_25_left']} of "
          f"{c['test_engines_with_under_25_left']} engines with < 25 left caught, "
          f"{c['test_urgent_with_50_plus_left']} flagged with 50+ left")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 6. WHICH SENSORS DRIVE THE PREDICTION?
# ─────────────────────────────────────────────────────────────────────────────
def sensor_importance(train, features):
    """Permutation importance on held-out engines: shuffle one input and see how
    much worse the predictions get. Rolling features are summed per sensor."""
    X, y, groups = train[features], train["RUL_capped"], train["engine_id"]
    tr, va = next(GroupKFold(n_splits=5).split(X, y, groups))
    model = make_models()["Gradient boosting"].fit(X.iloc[tr], y.iloc[tr])
    imp = permutation_importance(model, X.iloc[va], y.iloc[va], n_repeats=3,
                                 random_state=SEED, scoring="neg_root_mean_squared_error")
    s = pd.Series(imp.importances_mean, index=features)

    def group(f):
        if f.startswith("sensor_"):
            key = f[:9]
            return f"{key[-2:]} · {SENSOR_NAMES.get(key, key)}"
        return {"health_index": "Health Index", "hi_slope_10": "Health Index slope",
                "cycle": "Cycle count"}[f]
    return s.groupby(group).sum().sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 7. FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def plot(results, preds, importance, src):
    tag = "" if src.startswith("REAL") else "  [SYNTHETIC DATA]"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    # 06: model ladder
    names = list(results)
    vals = [results[n]["test_rmse"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names[::-1], vals[::-1], color=["#1F6E6A", "#8FA1AC", "#C9C3B6"])
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center")
    ax.set_xlabel("Test RMSE, cycles (lower is better)")
    ax.set_title(f"Each model has to beat the one before it{tag}", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG_DIR / "06_model_comparison.png", dpi=150); plt.close(fig)

    # 07: predicted vs true, with the likely range
    p = preds.sort_values("true_rul")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.vlines(p["true_rul"], p["rul_low_p10"], p["rul_high_p90"], color="#B9C7CF", lw=2,
              label="Likely range (10th–90th percentile)")
    ax.scatter(p["true_rul"], p["predicted_rul"], s=22, color="#1F6E6A", zorder=3,
               label="Prediction")
    lim = max(p["true_rul"].max(), RUL_CAP) + 5
    ax.plot([0, lim], [0, lim], ls="--", color="#13202B", lw=1, label="Perfect prediction")
    ax.axhline(RUL_CAP, color="#9A5410", lw=1, ls=":", label=f"Cap at {RUL_CAP}")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("True cycles left (NASA answer key)")
    ax.set_ylabel("Predicted cycles left")
    ax.set_title(f"100 unseen test engines: predicted vs true{tag}", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor="white", edgecolor="#DCD8CE")
    fig.tight_layout(); fig.savefig(FIG_DIR / "07_predicted_vs_true.png", dpi=150); plt.close(fig)

    # 08: what drives the prediction
    top = importance.head(8)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(top.index, top.values, color="#1F6E6A")
    ax.set_xlabel("RMSE increase when this input is shuffled (cycles)")
    ax.set_title(f"What the model relies on most{tag}", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG_DIR / "08_feature_importance.png", dpi=150); plt.close(fig)
    print("  [FIGURES] 06_model_comparison.png, 07_predicted_vs_true.png, 08_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. SAVE
# ─────────────────────────────────────────────────────────────────────────────
def save(results, preds, coverage, misses, importance, features, src):
    preds.to_csv(PROC / "rul_predictions.csv", index=False)
    (PROC / "model_results.json").write_text(json.dumps({
        "data_source": src, "rul_cap": RUL_CAP, "n_features": len(features),
        "cv": "5-fold GroupKFold by engine_id on all training rows (target capped at 125)",
        "test": "one prediction per test engine at its last cycle vs RUL_FD001.txt",
        "models": results,
        "range_coverage_p10_p90": round(coverage, 3),
        "comparison_with_hi_alerts": misses,
        "sensor_importance_top8": {k: round(float(v), 2) for k, v in importance.head(8).items()},
    }, indent=2))
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as con:
            preds.to_sql("rul_predictions", con, if_exists="replace", index=False)
        print("  [SQL] table rul_predictions written to data/turbofan.db")
    print("  [EXPORT] rul_predictions.csv, model_results.json")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print(" NASA C-MAPSS — Remaining Useful Life model")
    print("=" * 65)
    train, test, answer_key, features, src = load()
    results, oof = cross_validate(train, features)
    preds, model, last, coverage = test_evaluation(train, test, answer_key, features, results)
    misses = compare_with_alerts(train, oof, preds)
    importance = sensor_importance(train, features)
    print("  [IMPORTANCE] " + ", ".join(f"{k} ({v:.1f})" for k, v in importance.head(5).items()))
    plot(results, preds, importance, src)
    save(results, preds, coverage, misses, importance, features, src)
    print("\n  ✓ Done.\n")
