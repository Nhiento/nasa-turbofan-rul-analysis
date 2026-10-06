# NASA Turbofan Engine Remaining Useful Life (RUL) Analysis
### Predictive maintenance analytics on NASA C-MAPSS FD001 — Python · SQL · Power BI

---

## Project overview

Jet engines are maintained on fixed schedules, which means servicing some engines
too early and missing others that wear out fast. This project uses engine sensor
data to answer two questions:

1. **Can we tell how worn an engine is from its sensors alone?** → a Health Index
2. **How much warning does that give before failure?** → alert thresholds and lead time

**Status**

| Stage | Status |
|---|---|
| Data cleaning, validation, feature engineering | ✅ Done |
| Health Index, data-driven alert thresholds, lead-time analysis | ✅ Done |
| SQLite layer: integrity checks, anomaly log, analytical views | ✅ Done |
| RUL prediction model (baseline → gradient boosting, engine-grouped CV) | 🔜 Next |
| Power BI dashboard | 🔜 Planned |

---

## Dataset: NASA C-MAPSS FD001

C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) is NASA's simulated
turbofan degradation dataset (Saxena et al., 2008) and a standard benchmark for
prognostics research.

| Property | Value |
|---|---|
| Source | NASA Ames Prognostics Center of Excellence |
| Training set | 100 engines, run from healthy to **failure** — 20,631 rows |
| Test set | 100 engines, records stop **before** failure — 13,096 rows |
| Answer key | `RUL_FD001.txt`: true cycles left for each test engine |
| Operating conditions | 1 (sea level) |
| Fault mode | 1 — high-pressure compressor (HPC) degradation |
| Columns | engine id, cycle, 3 operational settings, 21 sensors |
| Engine life (train) | 128–362 cycles, median ≈ 199 |

Each engine starts with a different, unknown amount of wear and manufacturing
variation — so engines do not start at the same health level.

Place `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` in `data/raw/`.
(If they are missing, the scripts generate a clearly labelled synthetic stand-in
so the pipeline can still be tested; its results are not reported here.)

### What the sensors measure

| Sensor | Measures | As the HPC wears |
|---|---|---|
| 02 / 03 / 04 | LPC / HPC / LPT outlet temperature | Rise |
| 07 | HPC outlet pressure | Falls |
| 08 / 13 | Physical / corrected fan speed | Rise |
| 09 / 14 | Physical / corrected core speed | Rise (weaker, varies by engine) |
| 11 | HPC outlet static pressure | Rises |
| 12 | Fuel flow ÷ static pressure | Falls |
| 15 | Bypass ratio | Rises |
| 17 | Bleed enthalpy | Rises |
| 20 / 21 | HPT / LPT coolant bleed | Fall |

The physical story: a worn compressor is less efficient, so the engine runs hotter
and at lower pressure to do the same work.

---

## Pipeline

```
python/01_data_acquisition.py   raw .txt  ──►  raw_train / raw_test / test_rul .csv
python/02_data_cleaning.py      clean, features, Health Index, thresholds, figures
                                ──►  cleaned_master.csv, powerbi_*.csv,
                                     test_engine_status.csv, hi_thresholds.csv,
                                     pipeline_params.json, data/figures/*.png
python/03_load_sqlite.py        sql/01_schema.sql → load CSVs →
                                sql/02_checks_anomalies_views.sql
                                ──►  data/turbofan.db
```

Run from the project folder:

```bash
pip install pandas numpy matplotlib
python python/01_data_acquisition.py
python python/02_data_cleaning.py
python python/03_load_sqlite.py
```

**Train/test discipline:** every parameter (which sensors to drop, each sensor's
wear direction and reference levels, the alert thresholds) is learned from the
training engines only and then applied unchanged to the test engines.

---

## Cleaning and feature engineering (`02_data_cleaning.py`)

| Step | Method | Result on FD001 |
|---|---|---|
| Schema validation | Types, invalid cycles, duplicates, timeline gaps | 0 issues |
| Nulls | Forward-fill within engine, median fallback | 0 missing values |
| Flat sensors | Dropped if ≤ 2 distinct values or std ≈ 0 | 7 dropped (01, 05, 06, 10, 16, 18, 19) → **14 kept** |
| Spike removal | Residual from each engine's trailing 11-cycle median, clipped at IQR × 3 | 204 readings softened (0.07%) |
| RUL label | life − cycle; capped at 125 for modelling | Target variable |
| RUL bands | ≥125 / 100–124 / 50–99 / 25–49 / <25 cycles left | Labels for filtering |
| Rolling features | Trailing mean + std, 5/10/20-cycle windows | **84 features** |
| Health Index | See below | 0 ≈ healthy, 1 ≈ typical failure |
| Alert thresholds | Learned from the data | Warning 0.40, Alert 0.67 |

**Design decisions worth knowing**

- **Why not a coefficient-of-variation cutoff for flat sensors?** Sensors 08 and 13
  read ~2,388 with a standard deviation of ~0.07, so their CV is tiny — but they
  have a clear wear trend. A CV rule would wrongly drop them.
- **Why not global outlier clipping?** End-of-life readings are the extreme values,
  and they are the signal. Clipping against each engine's own recent median
  removes one-off spikes without flattening the wear trend.
- **Why cap RUL at 125?** Early in life, wear is too small to show in the sensors —
  an engine with 300 cycles left looks the same as one with 200. Capping
  (piecewise-linear RUL) avoids punishing a model for what the data cannot show.
- **Why trailing windows?** A centred window would use future readings, which
  a live monitoring system would not have.
- **Leakage:** `RUL`, `RUL_capped`, `rul_band` and `cycle_pct` all depend on the
  failure time. They are labels / EDA columns and must not be model inputs.

### Health Index

1. Each sensor's wear direction comes from its correlation with **cycle number**
   (time, not the RUL label): 10 sensors rise with wear, 4 fall (07, 12, 20, 21).
2. Each sensor is rescaled so the fleet's healthy level (median of each engine's
   first 20 cycles, 10-cycle smoothed) maps to **0** and its typical failure
   level (median over the last 5 cycles) maps to **1**.
3. The Health Index is the average across the 14 sensors. It is not clipped:
   values above 1 mean an engine has gone past the typical failure level.

### Alert thresholds

Chosen from the data, not by eye:

- **Alert = 0.674**: 75% of readings in engines' final 25 cycles are at or above it.
- **Warning = 0.399**: 75% of readings with 25–49 cycles left are at or above it.
- An alert is only raised after **3 cycles in a row** above the threshold, to
  avoid false alarms from noise.

---

## Results

| Finding | Value |
|---|---|
| Training engines that raised an Alert before failing | **96 / 100** |
| Warning time from first Alert to failure | median **26 cycles** (IQR 10–35, range 1–60) |
| Test engines: rank correlation of current HI with true RUL | **Spearman −0.82** |
| Test engines in **Alert** that truly had < 25 cycles left | 10 of 14 (none had ≥ 50 left) |
| Test engines in **Normal** that truly had ≥ 50 cycles left | 66 of 70 |

**Limits of a single threshold.** 4 training engines failed without a sustained
alert (their final HI was 0.66–0.70, just below 0.674), and 2 test engines read
"Normal" with fewer than 25 cycles left. Engines tolerate different amounts of
wear before failing, so a fleet-wide level misses some of them. This is the
motivation for the next stage: a model that uses the **trend and rate** of
degradation, not just its current level.

### Figures (`data/figures/`)

| File | Shows |
|---|---|
| `01_lifespan_distribution.png` | Engine life varies 128–362 cycles → fixed-interval maintenance is wasteful |
| `02_health_index_trajectories.png` | Flat early life, accelerating wear late — justifies the RUL cap |
| `03_sensor_rul_correlations.png` | All 14 kept sensors track wear; 11, 04, 12, 07 strongest |
| `04_health_index_by_rul_band.png` | The HI separates the RUL bands cleanly |
| `05_alert_lead_time.png` | How much warning the Alert threshold gives per engine |

---

## SQL layer (`sql/`, SQLite)

| Object | Purpose |
|---|---|
| `turbofan_readings` | Fact table: one row per training engine per cycle |
| `engine_summary` | Dimension table: life, start/final health, first alert, lead time |
| `test_engine_status` | Current state of the 100 in-service test engines |
| `hi_thresholds` | Thresholds learned in Python (single source of truth) |
| `anomaly_log` | `RAPID_DEGRADATION` (top 1% of 5-cycle HI jumps) and `ALERT_RAISED` events |
| `vw_fleet_kpis` | Fleet life statistics, engines alerted, average lead time |
| `vw_engine_health_trajectory` | Line-chart data |
| `vw_sensor_degradation` | Average sensor profile in 5% life buckets |
| `vw_degradation_rate` | 10-cycle Health Index change per engine |
| `vw_maintenance_queue` | Test engines ranked by current HI with a recommended action |
| `vw_alert_check` | Alert level vs. true remaining life — how well the alerts work |

Integrity checks (duplicates, timeline gaps, nulls, RUL stepping down by exactly 1,
every training engine ending at RUL 0) all pass on FD001.

The maintenance queue uses the **test** engines on purpose: training engines have
already failed, so ranking them would be meaningless.

---

## Next steps

1. **RUL model** — linear baseline, then gradient boosting on the rolling features;
   cross-validation grouped by engine so no engine appears in both train and
   validation folds; evaluated on the test set with RMSE and NASA's asymmetric
   score (late predictions penalised more than early ones).
2. **Power BI** — fleet overview, engine deep dive, maintenance queue, built on
   `data/turbofan.db` or the processed CSVs.
3. **FD002–FD004** — six operating conditions and a second fault mode; sensors
   would need normalising per operating condition first.

---

## Project structure

```
nasa_rul_project/
├── data/
│   ├── raw/            ← NASA .txt files
│   ├── processed/      ← cleaned CSVs, thresholds, pipeline_params.json
│   ├── figures/        ← EDA figures
│   └── turbofan.db     ← SQLite database (built by step 3)
├── python/
│   ├── 01_data_acquisition.py
│   ├── 02_data_cleaning.py
│   └── 03_load_sqlite.py
├── sql/
│   ├── 01_schema.sql
│   └── 02_checks_anomalies_views.sql
└── README.md
```

## Skills demonstrated

`Python` · `pandas` · `NumPy` · `Matplotlib` · `Data validation` · `Feature engineering`
`Time-series smoothing` · `SQL (DDL, CTEs, window functions)` · `SQLite`
`Predictive maintenance` · `Threshold design` · `Leakage-aware train/test design`
