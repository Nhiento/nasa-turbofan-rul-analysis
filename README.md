# NASA Turbofan Engine Remaining Useful Life (RUL) Analysis
### Predictive maintenance analytics on NASA C-MAPSS FD001 — Python · SQL · scikit-learn

---

## Project overview

Jet engines are maintained on fixed schedules, which means servicing some engines
too early and missing others that wear out fast. This project uses engine sensor
data to answer three questions:

1. **Can we tell how worn an engine is from its sensors alone?** → a Health Index
2. **How much warning does that give before failure?** → alert thresholds and lead time
3. **How many cycles does each engine have left?** → a Remaining Useful Life (RUL) model

**Status**

| Stage | Status |
|---|---|
| Data cleaning, validation, feature engineering | ✅ Done |
| Health Index, data-driven alert thresholds, lead-time analysis | ✅ Done |
| SQLite layer: integrity checks, anomaly log, analytical views | ✅ Done |
| RUL prediction model (baseline → gradient boosting, engine-grouped CV) | ✅ Done |
| Power BI data model and measures (`05_powerbi_export.py`, `powerbi/measures.dax`) | ✅ Done |
| Power BI report pages | 🔜 In progress |

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
python/04_rul_model.py          RUL model: cross-validation, test scoring,
                                likely range, figures 06–08
                                ──►  rul_predictions.csv, model_results.json,
                                     table rul_predictions in data/turbofan.db
python/05_powerbi_export.py     one Excel workbook, seven tables, ready for Power BI
                                ──►  data/powerbi/turbofan_powerbi.xlsx
```

Run from the project folder:

```bash
pip install pandas numpy matplotlib scikit-learn
python python/01_data_acquisition.py
python python/02_data_cleaning.py
python python/03_load_sqlite.py
python python/04_rul_model.py
python python/05_powerbi_export.py
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
motivation for the RUL model below: it uses the **trend and rate** of
degradation, not just its current level.

### Figures (`data/figures/`)

| File | Shows |
|---|---|
| `01_lifespan_distribution.png` | Engine life varies 128–362 cycles → fixed-interval maintenance is wasteful |
| `02_health_index_trajectories.png` | Flat early life, accelerating wear late — justifies the RUL cap |
| `03_sensor_rul_correlations.png` | All 14 kept sensors track wear; 11, 04, 12, 07 strongest |
| `04_health_index_by_rul_band.png` | The HI separates the RUL bands cleanly |
| `05_alert_lead_time.png` | How much warning the Alert threshold gives per engine |
| `06_model_comparison.png` | Test error of each model in the ladder |
| `07_predicted_vs_true.png` | Predicted vs true cycles left for the 100 test engines, with the likely range |
| `08_feature_importance.png` | Which inputs the model relies on most |

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

## RUL prediction model (`04_rul_model.py`)

**In one sentence:** the model looks at an engine's recent sensor history and its
Health Index and estimates how many cycles it has left, with a likely range.

### How it works

| Step | What happens | Why |
|---|---|---|
| Inputs | 84 rolling features, Health Index, Health Index slope (change per cycle over the last 10 cycles), cycle count — **87 inputs** | The slope catches engines that are wearing fast, which a fixed threshold misses |
| Target | RUL capped at 125 cycles | Early wear is invisible in the sensors (see above) |
| Model ladder | 1) always guess the average → 2) linear (Ridge) → 3) gradient boosting (scikit-learn `HistGradientBoostingRegressor`) | Each model must beat the simpler one before it to earn its place |
| Validation | 5-fold cross-validation **grouped by engine** | An engine is never in training and validation at once, so the score is honest |
| Final test | One prediction per test engine, at its last recorded cycle, vs NASA's answer key | The engine-level forecast a planner would actually use |
| Likely range | Two quantile models (10th and 90th percentile) | "About 18 cycles left, likely 10 to 30" is easier to plan around than one number |

**Scoring.** RMSE is the typical miss in cycles. NASA's score punishes a late
prediction (engine fails before you act) harder than an early one; lower is better.

### Results

| Model | Cross-validation RMSE | Test RMSE | NASA score |
|---|---|---|---|
| Baseline (always the average) | 41.7 | 43.1 | 33,629 |
| Linear (Ridge) | 16.8 | 18.8 | 568 |
| **Gradient boosting** | **14.9** | **17.6** | **543** |

- Gradient boosting misses by **17.6 cycles** on a typical test engine
  (16.3 if the answer key is also capped at 125, as many papers do), with almost
  no bias: the average error is +0.5 cycles.
- Predictions are tight near failure, where decisions are made, and wide early in
  life, where wear is not yet visible (figure 07).
- The true remaining life fell inside the 10–90% range for **80%** of test engines.
- What the model uses most: the Health Index, the cycle count, and the Health Index
  slope (figure 08), so the step-2 Health Index is the backbone of the forecast.

### Does it fix the Health Index misses?

Holding the model to the **same "3 cycles in a row" rule** ("predicted RUL under
25 for 3 cycles in a row"), using predictions from models that never saw that engine:

| Check | Health Index threshold | RUL model |
|---|---|---|
| Training engines alerted before failure | 96 / 100 | **100 / 100** |
| Median warning before failure | **26 cycles** (IQR 10–35) | 20 cycles (IQR 17–25) |
| The 4 engines the Health Index missed (57, 65, 72, 90) | missed | caught, 12–15 cycles ahead |
| Test engines with < 25 cycles left, flagged "Schedule now" / "Alert" | 10 of 19 | **15 of 19** (other 4 → "Plan") |
| Test engines flagged "Schedule now" / "Alert" that had 50+ cycles left | 0 | 0 |

**Takeaway:** the two work best together. The Health Index gives earlier warning
on most engines; the model catches the engines that tolerate more wear before
failing, and gives a planner a number of cycles instead of a color.

## Power BI (`05_powerbi_export.py`, `powerbi/measures.dax`)

`05_powerbi_export.py` writes `data/powerbi/turbofan_powerbi.xlsx`: one Excel table
per sheet, already shaped as a small star schema, so Power BI needs no clean-up.

| Table | Rows | Used for |
|---|---|---|
| `Engines` | 100 | Training fleet: life, health, alert timing |
| `Readings` | 20,631 | Health Index per engine per cycle |
| `SensorReadings` | 288,834 | Sensor values (raw and wear-scaled 0–1) for the deep dive |
| `Sensors` | 14 | Sensor names and healthy / failed levels |
| `MaintenanceQueue` | 100 | Test engines: Health Index, model forecast and range, action, truth |
| `Thresholds` | 3 | Warning / Alert / rapid-rise levels |
| `ModelMetrics` | 3 | Error of each model in the ladder |

Relationships: `Engines[engine_id]` → `Readings` and `SensorReadings`;
`Sensors[sensor_id]` → `SensorReadings`. `powerbi/measures.dax` holds every measure,
each with the value it should show, for three pages: fleet overview, engine deep
dive, and maintenance queue. The same three pages map to an AVEVA PI Vision display.

## Next steps

1. **Dashboard** — finish the three Power BI report pages on the workbook above.
2. **Combined alert** — raise an alert when either the Health Index or the model
   fires, and measure lead time and false alarms together.
3. **FD002–FD004** — six operating conditions and a second fault mode; sensors
   would need normalising per operating condition first.

---

## Project structure

```
nasa_rul_project/
├── data/
│   ├── raw/            ← NASA .txt files
│   ├── processed/      ← cleaned CSVs, thresholds, pipeline_params.json
│   ├── figures/        ← EDA and model figures
│   ├── powerbi/        ← turbofan_powerbi.xlsx (built by step 5)
│   └── turbofan.db     ← SQLite database (built by step 3)
├── python/
│   ├── 01_data_acquisition.py
│   ├── 02_data_cleaning.py
│   ├── 03_load_sqlite.py
│   ├── 04_rul_model.py
│   └── 05_powerbi_export.py
├── powerbi/
│   └── measures.dax
├── sql/
│   ├── 01_schema.sql
│   └── 02_checks_anomalies_views.sql
└── README.md
```

## Skills demonstrated

`Python` · `pandas` · `NumPy` · `Matplotlib` · `Data validation` · `Feature engineering`
`Time-series smoothing` · `SQL (DDL, CTEs, window functions)` · `SQLite`
`Predictive maintenance` · `Threshold design` · `Leakage-aware train/test design`
`scikit-learn` · `Gradient boosting` · `Grouped cross-validation` · `Quantile regression`
