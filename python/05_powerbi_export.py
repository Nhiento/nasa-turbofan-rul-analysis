"""
================================================================================
NASA C-MAPSS Turbofan Engine RUL Analysis
Step 5: Power BI data workbook
================================================================================
Builds ONE Excel workbook that Power BI (Desktop or the web service) can load
directly. Every sheet holds one Excel table, already shaped as a small star
schema, so no Power Query clean-up is needed.

  Table            Rows     What it is
  Engines           100     One row per training engine (life, health, alert timing)
  Readings       20,631     Training engines, one row per engine per cycle
  SensorReadings ~289k      Same cycles, one row per sensor (for the sensor chart)
  Sensors            14     Sensor names, what they measure, healthy / failed levels
  MaintenanceQueue  100     Test engines: Health Index, model forecast, action, truth
  Thresholds          3     Warning / Alert / rapid-rise levels
  ModelMetrics        3     Error of each model in the ladder

Relationships to create in Power BI (all one-to-many, single direction):
  Engines[engine_id]  → Readings[engine_id]
  Engines[engine_id]  → SensorReadings[engine_id]
  Sensors[sensor_id]  → SensorReadings[sensor_id]

Inputs : data/processed/*.csv, pipeline_params.json, model_results.json
Output : data/powerbi/turbofan_powerbi.xlsx  (rebuild any time; not committed)
================================================================================
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "data" / "powerbi"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "turbofan_powerbi.xlsx"

SENSOR_INFO = {
    "sensor_02": ("LPC outlet temperature", "Temperature"),
    "sensor_03": ("HPC outlet temperature", "Temperature"),
    "sensor_04": ("LPT outlet temperature", "Temperature"),
    "sensor_07": ("HPC outlet pressure", "Pressure"),
    "sensor_08": ("Physical fan speed", "Speed"),
    "sensor_09": ("Physical core speed", "Speed"),
    "sensor_11": ("HPC outlet static pressure", "Pressure"),
    "sensor_12": ("Fuel flow / static pressure", "Flow"),
    "sensor_13": ("Corrected fan speed", "Speed"),
    "sensor_14": ("Corrected core speed", "Speed"),
    "sensor_15": ("Bypass ratio", "Ratio"),
    "sensor_17": ("Bleed enthalpy", "Energy"),
    "sensor_20": ("HPT coolant bleed", "Flow"),
    "sensor_21": ("LPT coolant bleed", "Flow"),
}


def build_tables():
    params = json.loads((PROC / "pipeline_params.json").read_text())
    results = json.loads((PROC / "model_results.json").read_text())
    hi_p = params["health_index_params"]
    sensors = params["sensors_used"]

    # Engines (training fleet)
    eng = pd.read_csv(PROC / "powerbi_engine_summary.csv")
    eng["alerted_before_failure"] = np.where(eng["first_alert_cycle"].notna(), "Yes", "No")
    eng["engine_label"] = "Engine " + eng["engine_id"].astype(str).str.zfill(3)
    eng = eng[["engine_id", "engine_label", "life_cycles", "start_health", "final_health",
               "mean_health", "first_warning_cycle", "first_alert_cycle", "alert_lead_cycles",
               "alerted_before_failure"]].round(4)

    # Readings (training fleet, per cycle)
    master = pd.read_csv(PROC / "cleaned_master.csv")
    readings = master[["engine_id", "cycle", "cycle_pct", "RUL", "RUL_capped", "rul_band",
                       "health_index", "alert_level"]].rename(columns={
                           "cycle_pct": "life_pct", "RUL": "cycles_left",
                           "RUL_capped": "cycles_left_capped"})
    band_order = {"Healthy": 1, "Early": 2, "Moderate": 3, "Severe": 4, "Critical": 5}
    readings["rul_band_order"] = readings["rul_band"].map(band_order)
    readings["life_pct_bucket"] = (np.floor(readings["life_pct"] * 20) * 5).clip(upper=95).astype(int)

    # SensorReadings (long): raw value + wear-scaled value (0 = healthy, 1 = typical failure)
    long_parts = []
    for s in sensors:
        p = hi_p[s]
        span = (p["failed"] - p["healthy"]) or 1.0
        part = master[["engine_id", "cycle"]].copy()
        part["sensor_id"] = s
        part["value"] = master[s].round(4)
        part["wear_scaled"] = ((master[f"{s}_roll10_mean"] - p["healthy"]) / span).round(4)
        long_parts.append(part)
    sensor_long = pd.concat(long_parts, ignore_index=True)

    # Sensors (dimension)
    sens = pd.DataFrame([{
        "sensor_id": s,
        "sensor_number": int(s[-2:]),
        "sensor_name": SENSOR_INFO[s][0],
        "measures": SENSOR_INFO[s][1],
        "as_engine_wears": "Rises" if hi_p[s]["direction"] > 0 else "Falls",
        "healthy_level": round(hi_p[s]["healthy"], 4),
        "failed_level": round(hi_p[s]["failed"], 4),
    } for s in sensors])

    # MaintenanceQueue (test engines, what a planner sees now)
    q = pd.read_csv(PROC / "rul_predictions.csv")
    q["engine_label"] = "Engine " + q["engine_id"].astype(str).str.zfill(3)
    q["queue_rank"] = q["predicted_rul"].rank(method="first").astype(int)
    q["action_order"] = q["model_action"].map(
        {"Schedule now": 1, "Plan within next window": 2, "No action": 3})
    q["alert_order"] = q["hi_alert_level"].map({"Alert": 1, "Warning": 2, "Normal": 3})
    q["true_band"] = pd.cut(q["true_rul"], [-1, 24, 49, np.inf],
                            labels=["Under 25", "25 to 49", "50 or more"]).astype(str)
    q["true_band_order"] = q["true_band"].map({"Under 25": 1, "25 to 49": 2, "50 or more": 3})
    q = q[["queue_rank", "engine_id", "engine_label", "last_cycle", "health_index",
           "hi_alert_level", "alert_order", "predicted_rul", "rul_low_p10", "rul_high_p90",
           "model_action", "action_order", "true_rul", "true_band", "true_band_order",
           "error"]].sort_values("queue_rank")

    th = pd.read_csv(PROC / "hi_thresholds.csv").rename(columns={"name": "threshold"})
    th["label"] = th["threshold"].map({"warning": "Warning", "alert": "Alert",
                                       "rapid_rise_5cy": "Rapid rise (5 cycles)"})

    mm = pd.DataFrame([{"model": k, "ladder_step": i + 1, "cv_rmse": v["cv_rmse"],
                        "test_rmse": v["test_rmse"], "nasa_score": v["test_nasa_score"],
                        "mean_error": v["test_mean_error"]}
                       for i, (k, v) in enumerate(results["models"].items())])

    return {"Engines": eng, "Readings": readings, "SensorReadings": sensor_long,
            "Sensors": sens, "MaintenanceQueue": q, "Thresholds": th, "ModelMetrics": mm}


def write_workbook(tables, src):
    with pd.ExcelWriter(OUT, engine="xlsxwriter") as xw:
        wb = xw.book
        head = wb.add_format({"bold": True, "font_name": "Arial", "font_size": 11})
        body = wb.add_format({"font_name": "Arial", "font_size": 10, "text_wrap": True,
                              "valign": "top"})

        # About sheet: what is in the file and where it came from
        about = wb.add_worksheet("About")
        lines = [
            ("NASA turbofan predictive maintenance: Power BI data", head),
            (f"Data source: {src}. Built by python/05_powerbi_export.py from the "
             "project's processed outputs; rebuild by re-running steps 1–5.", body),
            ("Each sheet below holds one Excel table with the same name. In Power BI, "
             "load the tables (not the sheets) and skip this About sheet.", body),
            ("Relationships: Engines[engine_id] → Readings[engine_id]; "
             "Engines[engine_id] → SensorReadings[engine_id]; "
             "Sensors[sensor_id] → SensorReadings[sensor_id].", body),
            ("Training engines (Engines, Readings, SensorReadings) ran to failure and are "
             "history. Test engines (MaintenanceQueue) are still running; true_rul is "
             "NASA's answer key, included only to check accuracy.", body),
            ("wear_scaled: each sensor's 10-cycle mean rescaled so the fleet's healthy "
             "level = 0 and its typical failure level = 1 (the Health Index ingredients).", body),
        ]
        about.set_column("A:A", 110)
        for i, (text, fmt) in enumerate(lines):
            about.write(i * 2, 0, text, fmt)

        for name, df in tables.items():
            df = df.replace({np.nan: None})
            ws = wb.add_worksheet(name)
            cols = [{"header": c} for c in df.columns]
            ws.add_table(0, 0, len(df), len(df.columns) - 1, {
                "name": name, "columns": cols, "data": df.values.tolist(),
                "style": "Table Style Light 9"})
            for j, c in enumerate(df.columns):
                ws.set_column(j, j, max(12, min(30, len(c) + 4)))
            ws.freeze_panes(1, 0)


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print(" NASA C-MAPSS — Power BI data workbook")
    print("=" * 65)
    src = (PROC / "data_source.txt").read_text().strip() if (PROC / "data_source.txt").exists() else "unknown"
    tables = build_tables()
    for n, df in tables.items():
        print(f"  {n:<17} {len(df):>8,} rows × {df.shape[1]} columns")
    write_workbook(tables, src)
    print(f"  [EXPORT] {OUT.relative_to(BASE_DIR)} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print("\n  ✓ Done. Open it in Power BI: Get data → Excel workbook.\n")
