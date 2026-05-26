# NASA C-MAPSS Turbofan Engine RUL
## Power BI Dashboard — Complete Setup Guide

> **Resume-worthy deliverable**: An end-to-end predictive maintenance analytics
> dashboard covering fleet health monitoring, sensor degradation analysis,
> engine risk ranking, and anomaly detection.

---

## Files to Import into Power BI

| File | Role in Data Model |
|---|---|
| `powerbi_main.csv` | Fact table — all sensor readings per engine per cycle |
| `powerbi_engine_summary.csv` | Engine dimension — one row per engine with KPIs |
| `powerbi_sensor_trends.csv` | Trend table — sensor readings over normalised life |

---

## Step 1: Load Data

1. Open **Power BI Desktop**
2. **Get Data → Text/CSV** → import each of the 3 CSV files above
3. In **Power Query Editor**, verify these column types:

| Column | Type |
|---|---|
| `engine_id` | Whole Number |
| `cycle` | Whole Number |
| `cycle_pct` | Decimal Number |
| `rul`, `rul_capped` | Decimal Number |
| `health_index` | Decimal Number |
| `health_stage` | Text |
| `op_regime` | Text |
| All `sensor_XX` columns | Decimal Number |

4. Click **Close & Apply**

---

## Step 2: Data Model Relationships

In the **Model** view, create these relationships:

```
powerbi_main.engine_id  ──────►  powerbi_engine_summary.engine_id
powerbi_sensor_trends.engine_id  ──►  powerbi_engine_summary.engine_id
```

Set both as **Many-to-One**, **Cross-filter: Single**.

---

## Step 3: DAX Measures

Create a dedicated **Measures Table** (new empty table named `_Measures`).

### Core KPI Measures

```dax
-- Total engines in fleet
Total Engines =
DISTINCTCOUNT(powerbi_engine_summary[engine_id])

-- Average RUL across current selection
Avg RUL =
AVERAGE(powerbi_main[rul_capped])

-- Engines in critical health stage
Critical Engines =
CALCULATE(
    DISTINCTCOUNT(powerbi_main[engine_id]),
    powerbi_main[health_stage] = "Critical"
)

-- Fleet average health index
Fleet Health Score =
AVERAGE(powerbi_main[health_index])

-- Engines below 50% health (alert threshold)
Engines At Risk =
CALCULATE(
    DISTINCTCOUNT(powerbi_main[engine_id]),
    powerbi_main[health_index] > 0.50
)

-- % of fleet in warning or worse
Fleet Risk % =
DIVIDE(
    CALCULATE(
        DISTINCTCOUNT(powerbi_main[engine_id]),
        powerbi_main[health_stage] IN { "Critical", "Severe" }
    ),
    [Total Engines],
    0
)
```

### Trend Measures

```dax
-- Rolling 10-cycle degradation rate (for selected engine)
Degradation Rate 10cy =
VAR CurrentCycle  = MAX(powerbi_main[cycle])
VAR PreviousCycle = CurrentCycle - 10
VAR CurrentHealth =
    CALCULATE(
        AVERAGE(powerbi_main[health_index]),
        powerbi_main[cycle] = CurrentCycle
    )
VAR PreviousHealth =
    CALCULATE(
        AVERAGE(powerbi_main[health_index]),
        powerbi_main[cycle] = PreviousCycle
    )
RETURN
    IF(
        NOT ISBLANK(PreviousHealth),
        CurrentHealth - PreviousHealth,
        BLANK()
    )

-- Predicted remaining cycles to threshold (health_index > 0.75)
Cycles to Critical =
VAR AvgDegRate =
    AVERAGEX(
        FILTER(
            powerbi_main,
            powerbi_main[engine_id] = MAX(powerbi_main[engine_id])
                && powerbi_main[cycle] >= MAX(powerbi_main[cycle]) - 20
        ),
        powerbi_main[health_index]
    )
VAR CurrentHI = MAX(powerbi_main[health_index])
RETURN
    IF(
        AvgDegRate > 0 && CurrentHI < 0.75,
        INT((0.75 - CurrentHI) / (AvgDegRate / 20)),
        "N/A"
    )

-- Health stage label with colour indicator (used in conditional formatting)
Health Stage Color =
SWITCH(
    MAX(powerbi_main[health_stage]),
    "Critical",  "#ef4444",
    "Severe",    "#f97316",
    "Moderate",  "#f59e0b",
    "Early",     "#84cc16",
    "Healthy",   "#22c55e",
    "#94a3b8"
)
```

### Comparison Measures

```dax
-- Engine RUL vs fleet average (positive = better than average)
RUL vs Fleet Avg =
    MAX(powerbi_engine_summary[total_rul])
    - CALCULATE(AVERAGE(powerbi_engine_summary[total_rul]), ALL(powerbi_engine_summary))

-- % of life elapsed for selected engine
Life Elapsed % =
DIVIDE(
    MAX(powerbi_main[cycle]),
    MAX(powerbi_engine_summary[max_cycle]),
    0
) * 100
```

---

## Step 4: Dashboard Pages

### Page 1 — Fleet Overview (Executive Summary)

**Layout**: 4 KPI cards across the top, large visuals below.

| Visual | Type | Fields |
|---|---|---|
| Total Engines | Card | `[Total Engines]` |
| Critical Engines | Card | `[Critical Engines]` with red conditional formatting |
| Fleet Health Score | Gauge | `[Fleet Health Score]`, min=0, max=1, target=0.5 |
| Avg RUL | Card | `[Avg RUL]` |
| Health Stage Distribution | Donut chart | Legend: `health_stage`, Values: count of `engine_id` |
| Fleet Health Over Time | Line chart | X: `cycle_pct` buckets, Y: `[Fleet Health Score]` |
| Operational Regime Comparison | Clustered Bar | Axis: `op_regime`, Values: `[Avg RUL]` |
| Engine Risk Scatter | Scatter chart | X: `total_rul`, Y: `final_health`, Legend: `failure_stage` |

**Slicers**: `health_stage`, `op_regime`

---

### Page 2 — Engine Deep Dive

**Layout**: Engine selector at top-left, multi-panel detail.

| Visual | Type | Fields / Config |
|---|---|---|
| Engine Selector | Slicer (dropdown) | `engine_id` |
| Health Index Trend | Line chart | X: `cycle`, Y: `health_index`; add 0.60 constant line |
| Sensor Degradation Panel | Small multiples Line | Each sensor trend over `cycle_pct` |
| RUL Countdown | Card | `[Avg RUL]` with RAG colour formatting |
| Cycles to Critical | Card | `[Cycles to Critical]` |
| Anomaly Timeline | Table | `cycle`, `anomaly_type`, `severity` from anomaly data |
| Sensor Correlation | Clustered bar | Pre-computed correlation values |

**Interactions**: Cross-filter between all visuals by engine_id.

---

### Page 3 — Sensor Analytics

| Visual | Type | Fields |
|---|---|---|
| Sensor vs RUL Scatter | Scatter | X: sensor value, Y: `rul_capped`, Legend: `health_stage` |
| Sensor Health Heatmap | Matrix | Rows: `engine_id`, Columns: sensors, Values: normalised sensor value |
| Sensor Distribution Box-plot | Boxplot or violin | Sensor values grouped by `health_stage` |
| Sensor Trend Lines | Line chart | Multiple sensors overlaid, X: `cycle_pct` |

---

### Page 4 — Maintenance Scheduling

| Visual | Type | Fields |
|---|---|---|
| Priority Queue Table | Table | `engine_id`, `total_rul`, `final_health`, `failure_stage`, `anomaly_count` |
| Maintenance Urgency | Bar chart | `maintenance_window` category counts |
| Fleet Risk Trend | Area chart | % of engines in each health stage over `cycle_pct` |
| Engines Nearing Failure | Conditional table | Filtered: `rul_capped` < 50, sorted by `final_health` DESC |

---

## Step 5: Visual Formatting Best Practices

### Colour Theme (import as JSON via `View → Themes → Customize`)

```json
{
  "name": "NASA RUL Dashboard",
  "dataColors": [
    "#1d4ed8", "#ef4444", "#f59e0b", "#22c55e",
    "#8b5cf6", "#06b6d4", "#f97316", "#84cc16"
  ],
  "background": "#f8fafc",
  "foreground": "#1e293b",
  "tableAccent": "#1d4ed8"
}
```

### Conditional Formatting for Health Stage

In **Format → Conditional Formatting → Background colour → Rules**:

| Value | Colour |
|---|---|
| = "Critical" | `#ef4444` (red) |
| = "Severe" | `#f97316` (orange) |
| = "Moderate" | `#f59e0b` (amber) |
| = "Early" | `#84cc16` (lime) |
| = "Healthy" | `#22c55e` (green) |

---

## Step 6: Report-Level Filters (RLS — optional for demo)

To simulate department-level access control:

1. **Modelling → Manage roles → New role: "Fleet Manager"**
2. Add filter: `powerbi_engine_summary[primary_regime] = "mid_lo"`
3. In **View As Roles**, test the dashboard experience

---

## Resume Talking Points for This Project

### Technical Skills Demonstrated

- **Python**: End-to-end data pipeline (pandas, numpy, matplotlib), feature engineering (rolling statistics, RUL labelling, health index), outlier detection, EDA visualisation
- **SQL**: DDL/DML, window functions (LAG, PARTITION BY), CTEs, analytical views, data integrity auditing, anomaly logging
- **Power BI**: Data modelling (star schema), DAX (window measures, conditional measures, CALCULATE), conditional formatting, slicers, multi-page dashboard design

### Domain Knowledge Points

- Predictive Maintenance (PdM) vs Preventive Maintenance — cost reduction framing
- Remaining Useful Life (RUL) prediction as a regression problem
- C-MAPSS dataset context: 4 operational subsets, 21 sensor channels, run-to-failure simulation
- Health Index as a composite degradation indicator
- CMAPSS is cited in 1,000+ academic papers — industry-recognised benchmark

### Impact Statement Template (for resume bullet)

> "Engineered an end-to-end predictive maintenance analytics platform using NASA C-MAPSS turbofan engine data (21,000+ readings, 100 engines), reducing maintenance scheduling decisions from heuristic to data-driven. Built Python ETL pipeline with RUL feature engineering, SQL-layer anomaly detection, and a 4-page Power BI fleet health dashboard — enabling identification of engines 25+ cycles before critical failure."
