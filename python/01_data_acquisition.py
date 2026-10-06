"""
================================================================================
NASA C-MAPSS Turbofan Engine RUL Analysis
Step 1: Data Acquisition
================================================================================
Dataset: NASA C-MAPSS FD001
  - Train: 100 engines run from healthy to failure (20,631 rows)
  - Test : 100 engines whose records stop BEFORE failure (13,096 rows)
  - RUL  : true remaining cycles for each test engine at its last record
  - One operating condition (sea level), one fault mode (HPC degradation)

Put train_FD001.txt, test_FD001.txt and RUL_FD001.txt in data/raw/.
If they are missing, a synthetic stand-in is generated so the pipeline still
runs, and every output is clearly marked as synthetic (data_source.txt).
================================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Directory setup ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
OUT_DIR  = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Column schema (matches the NASA text files) ──────────────────────────────
COLS = (
    ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i:02d}" for i in range(1, 22)]
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback (only used when the real files are missing)
# ─────────────────────────────────────────────────────────────────────────────
# Baselines and degradation directions mirror real FD001 behaviour:
# +1 = rises as the engine wears, -1 = falls, 0 = constant.
SENSOR_BASELINES = {
    "sensor_01": 518.67, "sensor_02": 642.0,  "sensor_03": 1589.0,
    "sensor_04": 1400.0, "sensor_05": 14.62,  "sensor_06": 21.61,
    "sensor_07": 554.0,  "sensor_08": 2388.0, "sensor_09": 9046.0,
    "sensor_10": 1.30,   "sensor_11": 47.47,  "sensor_12": 521.0,
    "sensor_13": 2388.0, "sensor_14": 8138.0, "sensor_15": 8.4195,
    "sensor_16": 0.03,   "sensor_17": 392.0,  "sensor_18": 2388.0,
    "sensor_19": 100.0,  "sensor_20": 38.86,  "sensor_21": 23.39,
}
SENSOR_TREND = {
    "sensor_01": 0, "sensor_02": +1, "sensor_03": +1, "sensor_04": +1,
    "sensor_05": 0, "sensor_06": 0,  "sensor_07": -1, "sensor_08": +1,
    "sensor_09": +1, "sensor_10": 0, "sensor_11": +1, "sensor_12": -1,
    "sensor_13": +1, "sensor_14": +1, "sensor_15": +1, "sensor_16": 0,
    "sensor_17": +1, "sensor_18": 0, "sensor_19": 0,  "sensor_20": -1,
    "sensor_21": -1,
}


def generate_synthetic_cmapss(n_engines: int = 100, min_life: int = 128,
                              max_life: int = 362, seed: int = 42) -> pd.DataFrame:
    """Run-to-failure engines with an exponential wear curve plus noise."""
    rng = np.random.default_rng(seed)
    records = []
    for eng_id in range(1, n_engines + 1):
        life = int(rng.integers(min_life, max_life + 1))
        initial_wear = rng.uniform(0.0, 0.3)          # engines start unequal
        for t in range(1, life + 1):
            wear = initial_wear + (1 - initial_wear) * (np.exp(4 * t / life) - 1) / (np.exp(4) - 1)
            row = {"engine_id": eng_id, "cycle": t,
                   "op_setting_1": rng.normal(0, 0.002),
                   "op_setting_2": rng.normal(0, 0.0003),
                   "op_setting_3": 100.0}
            for s, base in SENSOR_BASELINES.items():
                trend = SENSOR_TREND[s] * wear * base * 0.004
                noise = rng.normal(0, base * 0.0008) if SENSOR_TREND[s] else 0.0
                row[s] = round(base + trend + noise, 4)
            records.append(row)
    return pd.DataFrame(records, columns=COLS)


def truncate_for_test(df: pd.DataFrame, seed: int = 99):
    """Cut each engine at a random point to mimic the NASA test set."""
    rng = np.random.default_rng(seed)
    parts, rul = [], []
    for eng_id, g in df.groupby("engine_id"):
        life = len(g)
        stop = int(rng.integers(30, life - 5))
        parts.append(g.iloc[:stop])
        rul.append(life - stop)
    return pd.concat(parts, ignore_index=True), pd.Series(rul, name="RUL")


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
def read_nasa_file(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / name, sep=r"\s+", header=None, names=COLS)


def load_all():
    """Return (train, test, test_rul, source) — real data if present."""
    files = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
    if all((RAW_DIR / f).exists() for f in files):
        train = read_nasa_file("train_FD001.txt")
        test  = read_nasa_file("test_FD001.txt")
        rul   = pd.read_csv(RAW_DIR / "RUL_FD001.txt", sep=r"\s+",
                            header=None, names=["RUL"])["RUL"]
        return train, test, rul, "REAL NASA C-MAPSS FD001"

    print("  [!] Real FD001 files not found in data/raw/ — using SYNTHETIC data.")
    print("      Results are for testing the pipeline only; do not report them.")
    train = generate_synthetic_cmapss(seed=42)
    test_full = generate_synthetic_cmapss(seed=7)
    test, rul = truncate_for_test(test_full)
    return train, test, rul, "SYNTHETIC (not real NASA data)"


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print(" NASA C-MAPSS — Data Acquisition")
    print("=" * 65)

    df_train, df_test, test_rul, source = load_all()
    print(f"\n  Data source: {source}")
    print(f"  Train: {df_train.shape[0]:,} rows | {df_train['engine_id'].nunique()} engines "
          f"| life {df_train.groupby('engine_id')['cycle'].max().min()}–"
          f"{df_train.groupby('engine_id')['cycle'].max().max()} cycles")
    print(f"  Test : {df_test.shape[0]:,} rows | {df_test['engine_id'].nunique()} engines")
    print(f"  Test RUL labels: {len(test_rul)}")

    test_rul_df = pd.DataFrame({"engine_id": range(1, len(test_rul) + 1),
                                "true_rul": test_rul.values})

    df_train.to_csv(OUT_DIR / "raw_train.csv", index=False)
    df_test.to_csv(OUT_DIR / "raw_test.csv", index=False)
    test_rul_df.to_csv(OUT_DIR / "test_rul.csv", index=False)
    (OUT_DIR / "data_source.txt").write_text(source + "\n")

    print(f"\n  [✓] Saved raw_train.csv, raw_test.csv, test_rul.csv to {OUT_DIR}")
    print("  Run 02_data_cleaning.py next.\n")
