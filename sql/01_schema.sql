-- ================================================================================
-- NASA C-MAPSS Turbofan Engine RUL Analysis
-- SQL Layer, part 1: Schema  (SQLite)
-- Run by python/03_load_sqlite.py, which then loads the CSVs into these tables.
-- ================================================================================

DROP VIEW  IF EXISTS vw_fleet_kpis;
DROP VIEW  IF EXISTS vw_engine_health_trajectory;
DROP VIEW  IF EXISTS vw_sensor_degradation;
DROP VIEW  IF EXISTS vw_degradation_rate;
DROP VIEW  IF EXISTS vw_maintenance_queue;
DROP VIEW  IF EXISTS vw_alert_check;
DROP TABLE IF EXISTS anomaly_log;
DROP TABLE IF EXISTS turbofan_readings;
DROP TABLE IF EXISTS engine_summary;
DROP TABLE IF EXISTS test_engine_status;
DROP TABLE IF EXISTS hi_thresholds;

-- Fact table: one row per training engine per cycle (source: powerbi_main.csv)
CREATE TABLE turbofan_readings (
    engine_id     INTEGER NOT NULL,
    cycle         INTEGER NOT NULL,
    cycle_pct     REAL,          -- share of life elapsed (EDA only: needs failure time)
    rul           INTEGER,       -- cycles until failure (label)
    rul_capped    INTEGER,       -- RUL capped at 125 (model target)
    rul_band      TEXT,          -- Healthy / Early / Moderate / Severe / Critical
    health_index  REAL,          -- 0 = healthy baseline, 1 = typical failure level
    alert_level   TEXT,          -- Normal / Warning / Alert (from hi_thresholds)
    sensor_02 REAL, sensor_03 REAL, sensor_04 REAL, sensor_07 REAL,
    sensor_08 REAL, sensor_09 REAL, sensor_11 REAL, sensor_12 REAL,
    sensor_13 REAL, sensor_14 REAL, sensor_15 REAL, sensor_17 REAL,
    sensor_20 REAL, sensor_21 REAL,
    PRIMARY KEY (engine_id, cycle)
);

-- Dimension table: one row per training engine (source: powerbi_engine_summary.csv)
CREATE TABLE engine_summary (
    engine_id            INTEGER PRIMARY KEY,
    life_cycles          INTEGER,
    start_health         REAL,
    final_health         REAL,
    mean_health          REAL,
    first_warning_cycle  INTEGER,   -- first cycle of 3 in a row at/above Warning
    first_alert_cycle    INTEGER,   -- first cycle of 3 in a row at/above Alert
    alert_lead_cycles    INTEGER    -- life_cycles - first_alert_cycle
);

-- Current state of the 100 test engines (still running when data stops)
CREATE TABLE test_engine_status (
    engine_id     INTEGER PRIMARY KEY,
    last_cycle    INTEGER,
    health_index  REAL,
    alert_level   TEXT,
    true_rul      INTEGER        -- answer key from RUL_FD001.txt, for validation only
);

-- Data-driven thresholds learned in Python (single source of truth)
CREATE TABLE hi_thresholds (
    name   TEXT PRIMARY KEY,         -- warning | alert | rapid_rise_5cy
    value  REAL NOT NULL
);

-- Events detected by part 2
CREATE TABLE anomaly_log (
    anomaly_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    engine_id     INTEGER NOT NULL,
    cycle         INTEGER NOT NULL,
    anomaly_type  TEXT NOT NULL,     -- RAPID_DEGRADATION | ALERT_RAISED
    severity      TEXT NOT NULL,     -- WARNING | CRITICAL
    detail        TEXT
);
