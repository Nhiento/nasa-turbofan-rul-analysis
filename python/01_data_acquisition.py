"""
================================================================================
NASA C-MAPSS Turbofan Engine RUL Analysis
Step 1: Data Acquisition & Synthetic Data Generation
================================================================================
Dataset: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
================================================================================
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── Directory setup ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
OUT_DIR  = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Column schema (matches NASA C-MAPSS format exactly) ─────────────────────
COLS = (
    ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i:02d}" for i in range(1, 22)]
)

# Sensors known to be informative for RUL prediction (from literature)
INFORMATIVE_SENSORS = [
    "sensor_02", "sensor_03", "sensor_04", "sensor_07", "sensor_08",
    "sensor_09", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

# Sensors with near-zero variance (constant noise — to be dropped)
CONSTANT_SENSORS = ["sensor_01", "sensor_05", "sensor_06", "sensor_10",
                     "sensor_16", "sensor_18", "sensor_19"]


def generate_synthetic_cmapss(n_engines: int = 100,
                               min_life: int = 100,
                               max_life: int = 350,
                               noise_scale: float = 0.03,
                               seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic C-MAPSS-style run-to-failure data.

    Each engine degrades according to a nonlinear wear curve:
        degradation(t) = 1 / (1 + exp(-k * (t/T - 0.5)))
    with additive Gaussian sensor noise.
    """
    rng = np.random.default_rng(seed)
    records = []

    # Sensor baseline values (approximate real C-MAPSS ranges)
    SENSOR_BASELINES = {
        "sensor_01": 518.67,  "sensor_02": 642.0,   "sensor_03": 1589.0,
        "sensor_04": 1400.0,  "sensor_05": 14.62,   "sensor_06": 21.61,
        "sensor_07": 554.0,   "sensor_08": 2388.0,  "sensor_09": 9046.0,
        "sensor_10": 1.30,    "sensor_11": 47.47,   "sensor_12": 521.0,
        "sensor_13": 2388.0,  "sensor_14": 8138.0,  "sensor_15": 8.4195,
        "sensor_16": 0.03,    "sensor_17": 392.0,   "sensor_18": 2388.0,
        "sensor_19": 100.0,   "sensor_20": 38.86,   "sensor_21": 23.39,
    }
    # Direction each sensor moves as degradation increases (+1 or -1)
    SENSOR_TREND = {
        "sensor_01":  0, "sensor_02": +1, "sensor_03": +1, "sensor_04": +1,
        "sensor_05":  0, "sensor_06":  0, "sensor_07": -1, "sensor_08": -1,
        "sensor_09": -1, "sensor_10":  0, "sensor_11": -1, "sensor_12": +1,
        "sensor_13": +1, "sensor_14": +1, "sensor_15": -1, "sensor_16":  0,
        "sensor_17": +1, "sensor_18":  0, "sensor_19":  0, "sensor_20": +1,
        "sensor_21": +1,
    }
    SENSOR_SCALE = {k: abs(v) * 0.05 for k, v in SENSOR_BASELINES.items()}

    for eng_id in range(1, n_engines + 1):
        max_cycle = rng.integers(min_life, max_life + 1)
        op1 = rng.choice([0.0, 0.2, 0.4, 0.6, 1.0])  # discrete operating modes
        op2 = rng.choice([0.0, 0.25, 0.5, 0.75])
        op3 = rng.integers(60, 101) * 1.0

        k = rng.uniform(6.0, 10.0)  # steepness of degradation curve

        for t in range(1, max_cycle + 1):
            deg = 1 / (1 + np.exp(-k * (t / max_cycle - 0.5)))

            row = {
                "engine_id":    eng_id,
                "cycle":        t,
                "op_setting_1": op1 + rng.normal(0, 0.005),
                "op_setting_2": op2 + rng.normal(0, 0.005),
                "op_setting_3": op3 + rng.normal(0, 0.1),
            }
            for sensor, baseline in SENSOR_BASELINES.items():
                trend = SENSOR_TREND[sensor] * deg * SENSOR_SCALE[sensor] * 20
                noise = rng.normal(0, noise_scale * baseline * 0.02)
                row[sensor] = round(baseline + trend + noise, 4)

            records.append(row)

    df = pd.DataFrame(records, columns=COLS)
    return df


def load_or_generate(split: str = "train") -> pd.DataFrame:
    """Load real C-MAPSS file or fall back to synthetic generation."""
    real_path = RAW_DIR / f"{split}_FD001.txt"

    if real_path.exists():
        print(f"  [✓] Loading REAL NASA data from {real_path}")
        df = pd.read_csv(real_path, sep=r"\s+", header=None, names=COLS)
    else:
        print(f"  [~] Real file not found — generating synthetic C-MAPSS data...")
        n = 100 if split == "train" else 20
        df = generate_synthetic_cmapss(n_engines=n, seed=42 if split == "train" else 99)

    return df


def load_rul_labels() -> pd.Series:
    """Load ground-truth RUL for test set, or return None."""
    rul_path = RAW_DIR / "RUL_FD001.txt"
    if rul_path.exists():
        return pd.read_csv(rul_path, header=None, names=["RUL"])["RUL"]
    return None


if __name__ == "__main__":
    print("\n" + "="*65)
    print(" NASA C-MAPSS — Data Acquisition")
    print("="*65)

    print("\n[1/2] Loading training data...")
    df_train = load_or_generate("train")
    print(f"      Shape: {df_train.shape}  |  Engines: {df_train['engine_id'].nunique()}")

    print("\n[2/2] Loading test data...")
    df_test = load_or_generate("test")
    print(f"      Shape: {df_test.shape}  |  Engines: {df_test['engine_id'].nunique()}")

    # Save raw copies to processed dir for next steps
    df_train.to_csv(OUT_DIR / "raw_train.csv", index=False)
    df_test.to_csv(OUT_DIR / "raw_test.csv", index=False)

    print(f"\n[✓] Raw data saved to {OUT_DIR}")
    print("\nColumn overview:")
    print(df_train.dtypes.to_string())
    print("\nFirst 3 rows:")
    print(df_train.head(3).to_string(index=False))
    print("\nRun 02_data_cleaning.py next.\n")
