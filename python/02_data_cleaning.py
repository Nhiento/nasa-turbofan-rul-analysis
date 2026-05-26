"""
================================================================================
NASA C-MAPSS Turbofan Engine RUL Analysis
Step 2: Data Cleaning, EDA & Feature Engineering
================================================================================
Cleaning pipeline:
  1. Load raw data
  2. Validate schema and types
  3. Handle nulls / outliers
  4. Drop constant/near-constant sensors
  5. Compute RUL labels
  6. Add rolling statistics (mean, std — lag windows)
  7. Add health index (normalised degradation indicator)
  8. Add operational cluster label
  9. Export clean CSV for Power BI and SQL
================================================================================
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = BASE_DIR / "data" / "processed"
FIG_DIR  = BASE_DIR / "data" / "figures"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ── Sensor groups ────────────────────────────────────────────────────────────
INFORMATIVE = [
    "sensor_02", "sensor_03", "sensor_04", "sensor_07", "sensor_08",
    "sensor_09", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]
CONSTANT_SENSORS = [
    "sensor_01", "sensor_05", "sensor_06", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]
OPERATIONAL_COLS  = ["op_setting_1", "op_setting_2", "op_setting_3"]

# Rolling window sizes (cycles)
WINDOWS = [5, 10, 20]

# RUL cap — engines are practically failed at 125 cycles from failure
RUL_CAP = 125


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load the raw CSV produced by 01_data_acquisition.py."""
    path = OUT_DIR / "raw_train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"'{path}' not found — run 01_data_acquisition.py first."
        )
    df = pd.read_csv(path)
    print(f"  Loaded raw_train.csv  →  {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  VALIDATE
# ─────────────────────────────────────────────────────────────────────────────
def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce expected dtypes and report anomalies."""
    print("\n  [VALIDATE] Schema check...")

    # Ensure cycle and engine_id are integers
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"]     = df["cycle"].astype(int)

    # All sensor + op columns should be float
    float_cols = [c for c in df.columns if c.startswith(("sensor", "op_setting"))]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # Check for negative cycles (data integrity)
    neg_cycles = (df["cycle"] <= 0).sum()
    if neg_cycles:
        print(f"    ⚠  {neg_cycles} rows with cycle ≤ 0 — dropping.")
        df = df[df["cycle"] > 0].copy()

    # Check for missing engine IDs
    null_eng = df["engine_id"].isna().sum()
    if null_eng:
        print(f"    ⚠  {null_eng} rows with null engine_id — dropping.")
        df = df[df["engine_id"].notna()].copy()

    print(f"    ✓  Schema valid.  Shape after validation: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Report and impute missing sensor readings via forward-fill per engine."""
    print("\n  [NULLS] Checking for missing values...")

    null_summary = df.isnull().sum()
    null_cols = null_summary[null_summary > 0]

    if null_cols.empty:
        print("    ✓  No missing values found.")
        return df

    print(f"    Found nulls in {len(null_cols)} columns:")
    for col, cnt in null_cols.items():
        pct = 100 * cnt / len(df)
        print(f"      {col}: {cnt} ({pct:.1f}%)")

    # Forward-fill within each engine, then back-fill edge case at start
    sensor_cols = [c for c in df.columns if c.startswith("sensor")]
    df[sensor_cols] = (
        df.groupby("engine_id")[sensor_cols]
          .transform(lambda g: g.ffill().bfill())
    )

    remaining = df[sensor_cols].isnull().sum().sum()
    if remaining:
        print(f"    ⚠  {remaining} nulls remain after imputation — filling with column median.")
        df[sensor_cols] = df[sensor_cols].fillna(df[sensor_cols].median())

    print(f"    ✓  All nulls resolved.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DROP CONSTANT / LOW-VARIANCE SENSORS
# ─────────────────────────────────────────────────────────────────────────────
def drop_constant_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Remove sensors that carry no useful signal (variance ≈ 0)."""
    print("\n  [SENSORS] Removing constant/near-constant sensors...")

    # Also auto-detect any sensor with CV < 0.001
    sensor_cols = [c for c in df.columns if c.startswith("sensor")]
    low_var = []
    for col in sensor_cols:
        mean_val = df[col].mean()
        cv = df[col].std() / (abs(mean_val) + 1e-9)
        if cv < 0.001:
            low_var.append(col)

    to_drop = list(set(CONSTANT_SENSORS + low_var))
    actual_drop = [c for c in to_drop if c in df.columns]

    df = df.drop(columns=actual_drop)
    print(f"    Dropped {len(actual_drop)} sensors: {actual_drop}")
    print(f"    Remaining informative sensors: {sorted([c for c in df.columns if c.startswith('sensor')])}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  OUTLIER TREATMENT (IQR clip per sensor per engine)
# ─────────────────────────────────────────────────────────────────────────────
def clip_outliers(df: pd.DataFrame, iqr_factor: float = 3.0) -> pd.DataFrame:
    """Clip extreme outliers using IQR method (global, per sensor column)."""
    print(f"\n  [OUTLIERS] Clipping outliers (IQR × {iqr_factor})...")

    sensor_cols = [c for c in df.columns if c.startswith("sensor")]
    clipped_total = 0

    for col in sensor_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo  = Q1 - iqr_factor * IQR
        hi  = Q3 + iqr_factor * IQR
        mask = (df[col] < lo) | (df[col] > hi)
        clipped_total += mask.sum()
        df[col] = df[col].clip(lower=lo, upper=hi)

    print(f"    Clipped {clipped_total:,} extreme values across all sensors.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6.  COMPUTE RUL LABELS
# ─────────────────────────────────────────────────────────────────────────────
def compute_rul(df: pd.DataFrame, rul_cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Calculate RUL for each row.
    RUL = max_cycle_for_that_engine - current_cycle
    Also apply piece-wise linear cap: RUL > rul_cap → capped at rul_cap.
    """
    print(f"\n  [RUL] Computing Remaining Useful Life (cap={rul_cap})...")

    max_cycle = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    df = df.merge(max_cycle, on="engine_id")
    df["RUL"]     = df["max_cycle"] - df["cycle"]
    df["RUL_capped"] = df["RUL"].clip(upper=rul_cap)

    # Degradation stage labels (used in Power BI)
    bins   = [0, 25, 50, 100, rul_cap, df["RUL"].max() + 1]
    labels = ["Critical", "Severe", "Moderate", "Early", "Healthy"]
    df["health_stage"] = pd.cut(
        df["RUL_capped"], bins=bins, labels=labels, right=False
    )

    print(f"    RUL range: {df['RUL'].min()} – {df['RUL'].max()} cycles")
    print(f"    RUL_capped range: {df['RUL_capped'].min()} – {df['RUL_capped'].max()} cycles")
    print(f"    Health stage distribution:\n{df['health_stage'].value_counts().to_string()}")

    df = df.drop(columns=["max_cycle"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ROLLING STATISTICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean and std for each informative sensor over multiple windows.
    This captures temporal degradation trends and is critical for accurate RUL.
    """
    print(f"\n  [FEATURES] Adding rolling statistics (windows={WINDOWS})...")

    active_sensors = [c for c in INFORMATIVE if c in df.columns]
    df = df.sort_values(["engine_id", "cycle"]).copy()

    for sensor in active_sensors:
        grp = df.groupby("engine_id")[sensor]
        for w in WINDOWS:
            df[f"{sensor}_roll{w}_mean"] = grp.transform(
                lambda x, w=w: x.rolling(w, min_periods=1).mean()
            )
            df[f"{sensor}_roll{w}_std"] = grp.transform(
                lambda x, w=w: x.rolling(w, min_periods=1).std().fillna(0)
            )

    feat_count = len(active_sensors) * len(WINDOWS) * 2
    print(f"    Added {feat_count} rolling features.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8.  HEALTH INDEX (normalised composite degradation score)
# ─────────────────────────────────────────────────────────────────────────────
def add_health_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a single Health Index (0 = new, 1 = failed) for each row.
    Formula: mean of min-max normalised informative sensor readings.
    Sensors trending down are inverted.
    """
    print("\n  [FEATURES] Computing Health Index...")

    active_sensors = [c for c in INFORMATIVE if c in df.columns]
    normalised = pd.DataFrame(index=df.index)

    # Sensors that decrease as engine degrades (RUL decreases)
    DECREASING = {"sensor_07", "sensor_08", "sensor_09",
                  "sensor_11", "sensor_15"}

    for sensor in active_sensors:
        col = df[sensor]
        min_v, max_v = col.min(), col.max()
        rng = max_v - min_v if max_v != min_v else 1.0
        norm = (col - min_v) / rng
        # Flip decreasing sensors so all increase → degraded
        if sensor in DECREASING:
            norm = 1 - norm
        normalised[sensor] = norm

    df["health_index"] = normalised.mean(axis=1).round(4)
    print(f"    Health Index: mean={df['health_index'].mean():.3f}  "
          f"std={df['health_index'].std():.3f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 9.  OPERATIONAL REGIME CLUSTERING (simple rule-based)
# ─────────────────────────────────────────────────────────────────────────────
def add_operational_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each row with one of 6 operational regimes based on op_settings.
    In real C-MAPSS there are 6 discrete flight conditions.
    """
    print("\n  [FEATURES] Labelling operational regimes...")

    # Bin op_setting_1 and op_setting_2 into 2 × 3 grid → 6 regimes
    df["op1_bin"] = pd.cut(df["op_setting_1"], bins=3, labels=["lo", "mid", "hi"])
    df["op2_bin"] = pd.cut(df["op_setting_2"], bins=2, labels=["lo", "hi"])
    df["op_regime"] = (df["op1_bin"].astype(str) + "_" + df["op2_bin"].astype(str))
    df = df.drop(columns=["op1_bin", "op2_bin"])

    print(f"    Regime distribution:\n{df['op_regime'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 10.  CYCLE-NORMALISED POSITION (0→1 per engine)
# ─────────────────────────────────────────────────────────────────────────────
def add_cycle_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Add cycle_pct = fraction of engine life elapsed (0 = start, 1 = failure)."""
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["cycle_pct"] = (df["cycle"] / max_cycle).round(4)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 11.  EDA PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_eda(df: pd.DataFrame) -> None:
    """Save key EDA visualisations."""
    print("\n  [EDA] Generating plots...")

    # --- A: RUL distribution ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df.groupby("engine_id")["RUL"].max(), bins=30,
            color="#1d4ed8", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_title("Engine Lifespan Distribution (Max RUL per Engine)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Total Cycles to Failure", fontsize=11)
    ax.set_ylabel("Engine Count", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_rul_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- B: Health index over cycle_pct (sample 10 engines) ----------------
    sample_engines = df["engine_id"].unique()[:10]
    fig, ax = plt.subplots(figsize=(10, 4))
    for eng in sample_engines:
        sub = df[df["engine_id"] == eng].sort_values("cycle_pct")
        ax.plot(sub["cycle_pct"], sub["health_index"],
                alpha=0.5, linewidth=1.2, color="#ef4444")
    ax.set_title("Health Index Trajectory — Sample 10 Engines",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Normalised Life Position (0 = new, 1 = failure)", fontsize=11)
    ax.set_ylabel("Health Index", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axhline(0.6, color="#64748b", linestyle="--",
               linewidth=0.8, label="Alert threshold (0.60)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_health_index_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- C: Sensor correlation with RUL_capped (top 14) --------------------
    sensor_cols = [c for c in df.columns if c.startswith("sensor_") and "_roll" not in c]
    if len(sensor_cols) >= 6:
        correlations = df[sensor_cols + ["RUL_capped"]].corr()["RUL_capped"].drop("RUL_capped")
        top_corr = correlations.abs().nlargest(14)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#1d4ed8" if correlations[s] > 0 else "#ef4444" for s in top_corr.index]
        bars = ax.barh(top_corr.index, top_corr.values, color=colors, alpha=0.85)
        ax.set_title("Sensor Correlation with RUL (Absolute Value)",
                     fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("|Pearson r| with RUL_capped", fontsize=11)
        ax.invert_yaxis()
        ax.axvline(0.3, color="#94a3b8", linestyle="--",
                   linewidth=0.8, label="r = 0.30")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "03_sensor_rul_correlations.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- D: Health stage pie chart -----------------------------------------
    if "health_stage" in df.columns:
        stage_counts = df["health_stage"].value_counts()
        PALETTE = {
            "Healthy": "#22c55e", "Early": "#84cc16",
            "Moderate": "#f59e0b", "Severe": "#f97316",
            "Critical": "#ef4444",
        }
        colors = [PALETTE.get(s, "#94a3b8") for s in stage_counts.index]

        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            stage_counts.values, labels=stage_counts.index,
            autopct="%1.1f%%", colors=colors, startangle=90,
            pctdistance=0.82, wedgeprops={"linewidth": 1, "edgecolor": "white"}
        )
        for t in texts + autotexts:
            t.set_fontsize(10)
        ax.set_title("Distribution of Health Stages", fontsize=13,
                     fontweight="bold", pad=12)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "04_health_stage_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"    Saved 4 EDA figures to {FIG_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# 12.  EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def export_datasets(df: pd.DataFrame) -> None:
    """
    Export multiple CSVs:
      - cleaned_master.csv        → full dataset for analysis / ML
      - powerbi_main.csv          → Power BI primary fact table
      - powerbi_engine_summary.csv → engine-level summary for KPI cards
    """
    print("\n  [EXPORT] Writing output files...")

    # Master (all rows, all features)
    df.to_csv(OUT_DIR / "cleaned_master.csv", index=False)
    print(f"    cleaned_master.csv  →  {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Power BI primary fact table (select columns for readability)
    pbi_cols = (
        ["engine_id", "cycle", "cycle_pct", "RUL", "RUL_capped",
         "health_index", "health_stage", "op_regime",
         "op_setting_1", "op_setting_2", "op_setting_3"]
        + [c for c in INFORMATIVE if c in df.columns]
    )
    df[pbi_cols].to_csv(OUT_DIR / "powerbi_main.csv", index=False)
    print(f"    powerbi_main.csv    →  {df.shape[0]:,} rows × {len(pbi_cols)} cols")

    # Engine-level summary (one row per engine)
    eng_summary = df.groupby("engine_id").agg(
        max_cycle     = ("cycle",        "max"),
        total_rul     = ("RUL",          "max"),
        final_health  = ("health_index", "last"),
        min_health    = ("health_index", "min"),
        mean_health   = ("health_index", "mean"),
        primary_regime= ("op_regime",    lambda x: x.mode()[0]),
        failure_stage = ("health_stage", "last"),
    ).reset_index()
    eng_summary.to_csv(OUT_DIR / "powerbi_engine_summary.csv", index=False)
    print(f"    powerbi_engine_summary.csv  →  {eng_summary.shape[0]} engines")

    # Sensor trends (for trend-line visuals in Power BI)
    sensor_cols = [c for c in INFORMATIVE if c in df.columns]
    trends = df[["engine_id", "cycle", "cycle_pct", "RUL_capped"] + sensor_cols]
    trends.to_csv(OUT_DIR / "powerbi_sensor_trends.csv", index=False)
    print(f"    powerbi_sensor_trends.csv   →  {trends.shape[0]:,} rows × {trends.shape[1]} cols")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*65)
    print(" NASA C-MAPSS — Data Cleaning & Feature Engineering")
    print("="*65)

    df = load_data()
    df = validate_schema(df)
    df = handle_nulls(df)
    df = drop_constant_sensors(df)
    df = clip_outliers(df)
    df = compute_rul(df)
    df = add_rolling_features(df)
    df = add_health_index(df)
    df = add_operational_regime(df)
    df = add_cycle_norm(df)

    plot_eda(df)
    export_datasets(df)

    print("\n" + "="*65)
    print(" ✓  Cleaning complete!")
    print(f"    Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("="*65)
    print("\nNext step: run SQL scripts in /sql/ then connect Power BI to CSVs.\n")
