"""
================================================================================
NASA C-MAPSS Turbofan Engine RUL Analysis
Step 2: Data Cleaning, Feature Engineering, Health Index & EDA
================================================================================
Pipeline (parameters are LEARNED ON TRAIN, then APPLIED to test):
   1. Load raw train / test
   2. Validate schema and types
   3. Handle nulls (forward-fill per engine)
   4. Drop flat sensors (no variation in train)
   5. Remove single-cycle spikes without flattening the wear trend
   6. Label RUL (train only) and RUL bands
   7. Rolling mean / std features (trailing windows, no look-ahead)
   8. Health Index (0 = like new, 1 = typical failure state)
   9. Data-driven Warning / Alert thresholds + alert lead time
  10. Test-engine current status (what a maintenance planner would see)
  11. EDA figures and exports for SQL / Power BI

Leakage note: RUL, RUL_capped, rul_band and cycle_pct all require knowing when
the engine failed. They are fine for EDA and labels but must never be used as
model inputs.
================================================================================
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR  = BASE_DIR / "data" / "processed"
FIG_DIR  = BASE_DIR / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Settings ─────────────────────────────────────────────────────────────────
WINDOWS = [5, 10, 20]          # rolling windows (cycles)
SPIKE_WINDOW = 11              # trailing median window used to find spikes
SPIKE_IQR_FACTOR = 3.0
ALERT_PERSISTENCE = 3          # cycles in a row above threshold to raise an alert

# RUL cap: early in life the wear is too small to show up in the sensors, so
# "300 cycles left" and "200 cycles left" look identical. Capping the target at
# 125 (piecewise-linear RUL, Heimes 2008) stops models from being punished for
# something the data cannot reveal. 120–130 is the common range in the literature.
RUL_CAP = 125

# RUL bands (cycles remaining). These are LABELS derived from RUL, not a
# measurement of health: by construction every engine spends exactly 25 cycles
# in "Critical", 25 in "Severe" and 50 in "Moderate".
BAND_EDGES  = [0, 25, 50, 100, RUL_CAP, np.inf]
BAND_LABELS = ["Critical", "Severe", "Moderate", "Early", "Healthy"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    paths = {k: OUT_DIR / f"{k}.csv" for k in ["raw_train", "raw_test", "test_rul"]}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{missing} not found — run 01_data_acquisition.py first.")
    train = pd.read_csv(paths["raw_train"])
    test = pd.read_csv(paths["raw_test"])
    rul = pd.read_csv(paths["test_rul"])
    src = (OUT_DIR / "data_source.txt").read_text().strip() if (OUT_DIR / "data_source.txt").exists() else "unknown"
    print(f"  Data source: {src}")
    print(f"  Train {train.shape[0]:,} rows | Test {test.shape[0]:,} rows")
    return train, test, rul, src


# ─────────────────────────────────────────────────────────────────────────────
# 2. VALIDATE
# ─────────────────────────────────────────────────────────────────────────────
def validate_schema(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.dropna(subset=["engine_id", "cycle"]).copy()
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    for col in [c for c in df.columns if c.startswith(("sensor", "op_setting"))]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    bad = (df["cycle"] <= 0).sum()
    dupes = df.duplicated(["engine_id", "cycle"]).sum()
    df = df[df["cycle"] > 0].drop_duplicates(["engine_id", "cycle"])
    gaps = (df.sort_values(["engine_id", "cycle"]).groupby("engine_id")["cycle"].diff() > 1).sum()
    print(f"  [VALIDATE {name}] invalid cycles={bad}, duplicates={dupes}, timeline gaps={gaps}")
    return df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. NULLS
# ─────────────────────────────────────────────────────────────────────────────
def handle_nulls(df: pd.DataFrame, name: str) -> pd.DataFrame:
    sensor_cols = [c for c in df.columns if c.startswith("sensor")]
    n = int(df[sensor_cols].isnull().sum().sum())
    if n:
        df[sensor_cols] = df.groupby("engine_id")[sensor_cols].transform(lambda g: g.ffill().bfill())
        df[sensor_cols] = df[sensor_cols].fillna(df[sensor_cols].median())
    print(f"  [NULLS {name}] {n} missing sensor values {'imputed' if n else '(none)'}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. FLAT SENSORS (decided on train only)
# ─────────────────────────────────────────────────────────────────────────────
def find_flat_sensors(train: pd.DataFrame) -> list:
    """
    A sensor is dropped when it barely changes AT ALL (<= 2 distinct values or
    std ~ 0). We deliberately do NOT use a coefficient-of-variation cutoff:
    sensors 08 and 13 read ~2388 with a std of ~0.07, so their CV is tiny even
    though they carry a clear wear trend.
    """
    sensor_cols = [c for c in train.columns if c.startswith("sensor")]
    flat = [c for c in sensor_cols if train[c].nunique() <= 2 or train[c].std() < 1e-6]
    print(f"  [SENSORS] Dropping {len(flat)} flat sensors: {flat}")
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# 5. SPIKE REMOVAL (per engine, keeps the trend)
# ─────────────────────────────────────────────────────────────────────────────
def remove_spikes(df: pd.DataFrame, sensors: list, name: str) -> pd.DataFrame:
    """
    Global IQR clipping would treat end-of-life readings as "outliers" and
    flatten exactly the signal we care about. Instead, compare each reading to
    its engine's trailing rolling median and clip only the residual.
    """
    total = 0
    for s in sensors:
        med = df.groupby("engine_id")[s].transform(
            lambda x: x.rolling(SPIKE_WINDOW, min_periods=1).median())
        resid = df[s] - med
        q1, q3 = resid.quantile([0.25, 0.75])
        lo, hi = q1 - SPIKE_IQR_FACTOR * (q3 - q1), q3 + SPIKE_IQR_FACTOR * (q3 - q1)
        mask = (resid < lo) | (resid > hi)
        total += int(mask.sum())
        df.loc[mask, s] = med[mask] + resid[mask].clip(lo, hi)
    print(f"  [SPIKES {name}] softened {total:,} single-cycle spikes "
          f"({100 * total / (len(df) * len(sensors)):.2f}% of readings)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. RUL LABELS (train)
# ─────────────────────────────────────────────────────────────────────────────
def add_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    life = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = life - df["cycle"]
    df["RUL_capped"] = df["RUL"].clip(upper=RUL_CAP)
    df["rul_band"] = pd.cut(df["RUL"], bins=BAND_EDGES, labels=BAND_LABELS, right=False)
    df["cycle_pct"] = (df["cycle"] / life).round(4)   # EDA only (uses failure time)
    print(f"  [RUL] RUL 0–{df['RUL'].max()} | capped at {RUL_CAP}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. ROLLING FEATURES (trailing → usable in real time)
# ─────────────────────────────────────────────────────────────────────────────
def add_rolling_features(df: pd.DataFrame, sensors: list) -> pd.DataFrame:
    new = {}
    for s in sensors:
        grp = df.groupby("engine_id")[s]
        for w in WINDOWS:
            new[f"{s}_roll{w}_mean"] = grp.transform(lambda x, w=w: x.rolling(w, min_periods=1).mean())
            new[f"{s}_roll{w}_std"] = grp.transform(lambda x, w=w: x.rolling(w, min_periods=1).std().fillna(0))
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 8. HEALTH INDEX
# ─────────────────────────────────────────────────────────────────────────────
def fit_health_index(train: pd.DataFrame, sensors: list) -> dict:
    """
    Learn, from TRAIN only:
      - each sensor's wear direction: sign of its correlation with cycle number
        (does it go up or down as the engine ages?). Uses time, not the RUL label.
      - each sensor's healthy and failed reference levels: median of the
        10-cycle smoothed value over the first 20 cycles vs. the last 5 cycles.
    """
    params = {}
    first = train[train["cycle"] <= 20]
    last = train[train["RUL"] < 5]
    for s in sensors:
        col = f"{s}_roll10_mean"
        direction = int(np.sign(train[[s, "cycle"]].corr().iloc[0, 1]))
        params[s] = {"direction": direction,
                     "healthy": float(first[col].median()),
                     "failed": float(last[col].median())}
    rising = [s for s in sensors if params[s]["direction"] > 0]
    falling = [s for s in sensors if params[s]["direction"] < 0]
    print(f"  [HEALTH INDEX] rising with wear: {rising}")
    print(f"                 falling with wear: {falling}")
    return params


def apply_health_index(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Each sensor is scaled so the fleet's healthy level -> 0 and its typical
    failure level -> 1; the HI is the average across sensors. It is not clipped:
    below 0 = healthier than the fleet's typical new engine, above 1 = further
    gone than the typical failure point (engines differ in how much wear they
    tolerate before failing).
    """
    scaled = pd.DataFrame(index=df.index)
    for s, p in params.items():
        span = p["failed"] - p["healthy"]
        scaled[s] = (df[f"{s}_roll10_mean"] - p["healthy"]) / (span if span else 1.0)
    df["health_index"] = scaled.mean(axis=1).round(4)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 9. THRESHOLDS + LEAD TIME
# ─────────────────────────────────────────────────────────────────────────────
def derive_thresholds(train: pd.DataFrame) -> dict:
    """
    Thresholds come from the data instead of being picked by eye:
      Alert   = 25th percentile of HI over each engine's final 25 cycles
                (75% of 'Critical' readings are at or above it)
      Warning = 25th percentile of HI in the 25–50 cycles-left window
    """
    alert = float(train.loc[train["RUL"] < 25, "health_index"].quantile(0.25))
    warn = float(train.loc[(train["RUL"] >= 25) & (train["RUL"] < 50), "health_index"].quantile(0.25))
    # Rapid-rise flag (used by the SQL anomaly log): the top 1% of 5-cycle HI jumps
    jump5 = train.groupby("engine_id")["health_index"].diff(5)
    rapid = float(jump5.quantile(0.99))
    print(f"  [THRESHOLDS] Warning HI >= {warn:.3f} | Alert HI >= {alert:.3f} | "
          f"rapid 5-cycle rise >= {rapid:.3f}")
    return {"warning": round(warn, 4), "alert": round(alert, 4), "rapid_rise_5cy": round(rapid, 4)}


def add_alert_level(df: pd.DataFrame, th: dict) -> pd.DataFrame:
    df["alert_level"] = np.select(
        [df["health_index"] >= th["alert"], df["health_index"] >= th["warning"]],
        ["Alert", "Warning"], default="Normal")
    return df


def first_sustained_cross(g: pd.DataFrame, level: float):
    above = (g["health_index"] >= level).astype(int)
    run = above.groupby((above != above.shift()).cumsum()).cumsum()
    hit = g.loc[run >= ALERT_PERSISTENCE, "cycle"]
    return int(hit.iloc[0]) if len(hit) else np.nan


def alert_lead_times(train: pd.DataFrame, th: dict) -> pd.DataFrame:
    """How many cycles of warning would the Alert threshold have given each engine?"""
    rows = []
    for eng, g in train.groupby("engine_id"):
        life = g["cycle"].max()
        a = first_sustained_cross(g, th["alert"])
        w = first_sustained_cross(g, th["warning"])
        rows.append({"engine_id": eng,
                     "first_warning_cycle": w,
                     "first_alert_cycle": a,
                     "alert_lead_cycles": life - a if pd.notna(a) else np.nan})
    lead = pd.DataFrame(rows)
    lt = lead["alert_lead_cycles"]
    print(f"  [LEAD TIME] alerts raised for {lt.notna().sum()}/{len(lead)} engines | "
          f"median lead {lt.median():.0f} cycles (IQR {lt.quantile(.25):.0f}–{lt.quantile(.75):.0f}, "
          f"min {lt.min():.0f}, max {lt.max():.0f})")
    return lead


# ─────────────────────────────────────────────────────────────────────────────
# 10. TEST ENGINES — CURRENT STATUS
# ─────────────────────────────────────────────────────────────────────────────
def test_engine_status(test: pd.DataFrame, rul: pd.DataFrame) -> pd.DataFrame:
    """
    Test engines are still running when their records stop, so their LAST row
    is the 'current state' a maintenance planner would see. true_rul is the
    answer key from RUL_FD001.txt, kept only to check how well HI ranks risk.
    """
    last = test.sort_values("cycle").groupby("engine_id").tail(1)
    status = last[["engine_id", "cycle", "health_index", "alert_level"]].rename(
        columns={"cycle": "last_cycle"}).merge(rul, on="engine_id", how="left")
    rho = status[["health_index", "true_rul"]].corr(method="spearman").iloc[0, 1]
    print(f"  [TEST STATUS] Spearman(HI, true RUL) = {rho:.2f} "
          f"(negative = higher HI means less life left)")
    print(f"                current alert levels: {status['alert_level'].value_counts().to_dict()}")
    return status, rho


# ─────────────────────────────────────────────────────────────────────────────
# 11. EDA FIGURES
# ─────────────────────────────────────────────────────────────────────────────
BAND_COLORS = {"Healthy": "#22c55e", "Early": "#84cc16", "Moderate": "#f59e0b",
               "Severe": "#f97316", "Critical": "#ef4444"}


def plot_eda(train: pd.DataFrame, sensors: list, th: dict, lead: pd.DataFrame, src: str) -> None:
    tag = "" if src.startswith("REAL") else "  [SYNTHETIC DATA]"

    # A: lifespan distribution
    life = train.groupby("engine_id")["cycle"].max()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(life, bins=30, color="#1d4ed8", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(life.median(), color="#0f172a", ls="--", lw=1, label=f"Median {life.median():.0f} cycles")
    ax.set_title(f"Engine Lifespan Distribution (Train, n={len(life)}){tag}", fontweight="bold")
    ax.set_xlabel("Total cycles to failure"); ax.set_ylabel("Engine count"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "01_lifespan_distribution.png", dpi=150); plt.close(fig)

    # B: health index trajectories with data-driven thresholds
    fig, ax = plt.subplots(figsize=(10, 4))
    for eng in train["engine_id"].unique()[:10]:
        sub = train[train["engine_id"] == eng]
        ax.plot(sub["cycle_pct"], sub["health_index"], alpha=0.5, lw=1.2, color="#ef4444")
    ax.axhline(th["warning"], color="#f59e0b", ls="--", lw=1, label=f"Warning ({th['warning']:.2f})")
    ax.axhline(th["alert"], color="#b91c1c", ls="--", lw=1, label=f"Alert ({th['alert']:.2f})")
    ax.set_title(f"Health Index Trajectory — 10 Sample Engines{tag}", fontweight="bold")
    ax.set_xlabel("Life position (0 = new, 1 = failure)"); ax.set_ylabel("Health Index")
    ax.set_xlim(0, 1); ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG_DIR / "02_health_index_trajectories.png", dpi=150); plt.close(fig)

    # C: sensor correlation with capped RUL (signed)
    corr = train[sensors + ["RUL_capped"]].corr()["RUL_capped"].drop("RUL_capped")
    order = corr.abs().sort_values(ascending=True).index
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(order, corr[order].abs(), color=["#1d4ed8" if corr[s] > 0 else "#ef4444" for s in order])
    ax.axvline(0.3, color="#94a3b8", ls="--", lw=0.8)
    ax.set_title(f"Sensor Correlation with Capped RUL ({len(sensors)} sensors){tag}", fontweight="bold")
    ax.set_xlabel("|Pearson r|   (blue = falls as engine wears, red = rises)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "03_sensor_rul_correlations.png", dpi=150); plt.close(fig)

    # D: health index by RUL band — does the HI actually separate the bands?
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order_b = ["Healthy", "Early", "Moderate", "Severe", "Critical"]
    data = [train.loc[train["rul_band"] == b, "health_index"] for b in order_b]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)   # works on old + new matplotlib
    ax.set_xticks(range(1, len(order_b) + 1)); ax.set_xticklabels(order_b)
    for patch, b in zip(bp["boxes"], order_b):
        patch.set_facecolor(BAND_COLORS[b]); patch.set_alpha(0.8)
    ax.axhline(th["alert"], color="#b91c1c", ls="--", lw=1, label="Alert threshold")
    ax.set_title(f"Health Index by Cycles-Remaining Band{tag}", fontweight="bold")
    ax.set_xlabel("RUL band (≥125, 100–124, 50–99, 25–49, <25 cycles left)")
    ax.set_ylabel("Health Index"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "04_health_index_by_rul_band.png", dpi=150); plt.close(fig)

    # E: alert lead time
    lt = lead["alert_lead_cycles"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lt, bins=20, color="#0f766e", alpha=0.85, edgecolor="white")
    ax.axvline(lt.median(), color="#0f172a", ls="--", lw=1, label=f"Median {lt.median():.0f} cycles")
    ax.set_title(f"Warning Time Before Failure from the Alert Threshold{tag}", fontweight="bold")
    ax.set_xlabel("Cycles between first sustained alert and failure"); ax.set_ylabel("Engine count")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "05_alert_lead_time.png", dpi=150); plt.close(fig)

    print(f"  [EDA] 5 figures saved to {FIG_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def export(train, test, sensors, lead, status, th, hi_params, flat, rho, src):
    train.to_csv(OUT_DIR / "cleaned_master.csv", index=False)
    # Test engines with the same features + Health Index, used by 04_rul_model.py
    test.to_csv(OUT_DIR / "cleaned_test.csv", index=False)

    main_cols = (["engine_id", "cycle", "cycle_pct", "RUL", "RUL_capped", "rul_band",
                  "health_index", "alert_level"] + sensors)
    train[main_cols].to_csv(OUT_DIR / "powerbi_main.csv", index=False)

    summary = train.groupby("engine_id").agg(
        life_cycles=("cycle", "max"),
        start_health=("health_index", "first"),
        final_health=("health_index", "last"),
        mean_health=("health_index", "mean"),
    ).reset_index().merge(lead, on="engine_id")
    summary.to_csv(OUT_DIR / "powerbi_engine_summary.csv", index=False)

    train[["engine_id", "cycle", "cycle_pct", "RUL_capped"] + sensors].to_csv(
        OUT_DIR / "powerbi_sensor_trends.csv", index=False)
    status.to_csv(OUT_DIR / "test_engine_status.csv", index=False)
    pd.DataFrame([{"name": k, "value": v} for k, v in th.items()]).to_csv(
        OUT_DIR / "hi_thresholds.csv", index=False)

    (OUT_DIR / "pipeline_params.json").write_text(json.dumps({
        "data_source": src, "rul_cap": RUL_CAP, "dropped_flat_sensors": flat,
        "sensors_used": sensors, "rolling_windows": WINDOWS,
        "rolling_feature_count": len(sensors) * len(WINDOWS) * 2,
        "health_index_params": hi_params, "thresholds": th,
        "alert_persistence_cycles": ALERT_PERSISTENCE,
        "test_spearman_hi_vs_true_rul": round(float(rho), 3),
    }, indent=2))
    print(f"  [EXPORT] cleaned_master.csv ({train.shape[0]:,} × {train.shape[1]}), cleaned_test.csv, powerbi_main.csv, "
          f"powerbi_engine_summary.csv, powerbi_sensor_trends.csv, test_engine_status.csv, "
          f"hi_thresholds.csv, pipeline_params.json")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print(" NASA C-MAPSS — Cleaning, Features & Health Index")
    print("=" * 65)

    train, test, test_rul, src = load_data()
    train, test = validate_schema(train, "train"), validate_schema(test, "test")
    train, test = handle_nulls(train, "train"), handle_nulls(test, "test")

    flat = find_flat_sensors(train)
    sensors = [c for c in train.columns if c.startswith("sensor") and c not in flat]
    train, test = train.drop(columns=flat), test.drop(columns=flat)
    print(f"  [SENSORS] Keeping {len(sensors)}: {sensors}")

    train = remove_spikes(train, sensors, "train")
    test = remove_spikes(test, sensors, "test")

    train = add_rul_labels(train)
    train = add_rolling_features(train, sensors)
    test = add_rolling_features(test, sensors)
    print(f"  [FEATURES] {len(sensors) * len(WINDOWS) * 2} rolling features per dataset")

    hi_params = fit_health_index(train, sensors)
    train, test = apply_health_index(train, hi_params), apply_health_index(test, hi_params)

    th = derive_thresholds(train)
    train, test = add_alert_level(train, th), add_alert_level(test, th)
    lead = alert_lead_times(train, th)
    status, rho = test_engine_status(test, test_rul)

    plot_eda(train, sensors, th, lead, src)
    export(train, test, sensors, lead, status, th, hi_params, flat, rho, src)

    print("\n  ✓ Done. Next: python python/03_load_sqlite.py\n")
