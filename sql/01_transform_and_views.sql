-- ================================================================================
-- NASA C-MAPSS Turbofan Engine RUL Analysis
-- SQL Data Layer — Transformations, Integrity Checks & Analytical Views
-- ================================================================================
-- Compatible with: SQLite (default), PostgreSQL, MySQL (minor syntax changes noted)
-- Run order:
--   1. CREATE tables
--   2. Import CSVs (via tool or COPY/import wizard)
--   3. Run transformations
--   4. Create views used by Power BI
-- ================================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- SECTION 1: CREATE TABLES
-- ─────────────────────────────────────────────────────────────────────────────

-- Core sensor readings (fact table)
CREATE TABLE IF NOT EXISTS turbofan_readings (
    reading_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    engine_id       INTEGER     NOT NULL,
    cycle           INTEGER     NOT NULL,
    cycle_pct       REAL,
    rul             REAL,
    rul_capped      REAL,
    health_index    REAL,
    health_stage    TEXT,
    op_regime       TEXT,
    op_setting_1    REAL,
    op_setting_2    REAL,
    op_setting_3    REAL,
    sensor_02       REAL,
    sensor_03       REAL,
    sensor_04       REAL,
    sensor_07       REAL,
    sensor_08       REAL,
    sensor_09       REAL,
    sensor_11       REAL,
    sensor_12       REAL,
    sensor_13       REAL,
    sensor_14       REAL,
    sensor_15       REAL,
    sensor_17       REAL,
    sensor_20       REAL,
    sensor_21       REAL,
    UNIQUE (engine_id, cycle)
);

-- Engine-level summary (dimension table)
CREATE TABLE IF NOT EXISTS engine_summary (
    engine_id       INTEGER PRIMARY KEY,
    max_cycle       INTEGER,
    total_rul       REAL,
    final_health    REAL,
    min_health      REAL,
    mean_health     REAL,
    primary_regime  TEXT,
    failure_stage   TEXT
);

-- Anomaly log (populated by queries below)
CREATE TABLE IF NOT EXISTS anomaly_log (
    anomaly_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    engine_id       INTEGER,
    cycle           INTEGER,
    anomaly_type    TEXT,
    severity        TEXT,    -- 'WARNING' | 'CRITICAL'
    detected_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    detail          TEXT
);


-- ─────────────────────────────────────────────────────────────────────────────
-- SECTION 2: DATA INTEGRITY CHECKS
-- ─────────────────────────────────────────────────────────────────────────────

-- Check 1: Duplicate readings (same engine + cycle)
SELECT
    engine_id,
    cycle,
    COUNT(*) AS duplicate_count
FROM turbofan_readings
GROUP BY engine_id, cycle
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC;

-- Check 2: Engines with missing cycle sequences (gaps in timeline)
WITH cycle_gaps AS (
    SELECT
        engine_id,
        cycle,
        LAG(cycle) OVER (PARTITION BY engine_id ORDER BY cycle) AS prev_cycle,
        cycle - LAG(cycle) OVER (PARTITION BY engine_id ORDER BY cycle) AS gap
    FROM turbofan_readings
)
SELECT
    engine_id,
    prev_cycle AS gap_after_cycle,
    cycle AS gap_before_cycle,
    gap AS missing_cycles
FROM cycle_gaps
WHERE gap > 1
ORDER BY engine_id, cycle;

-- Check 3: Null values per column (quick audit)
SELECT
    SUM(CASE WHEN engine_id    IS NULL THEN 1 ELSE 0 END) AS null_engine_id,
    SUM(CASE WHEN cycle        IS NULL THEN 1 ELSE 0 END) AS null_cycle,
    SUM(CASE WHEN rul          IS NULL THEN 1 ELSE 0 END) AS null_rul,
    SUM(CASE WHEN health_index IS NULL THEN 1 ELSE 0 END) AS null_health_index,
    SUM(CASE WHEN sensor_02    IS NULL THEN 1 ELSE 0 END) AS null_sensor_02
FROM turbofan_readings;

-- Check 4: RUL consistency (RUL should decrease as cycle increases per engine)
WITH rul_check AS (
    SELECT
        engine_id,
        cycle,
        rul,
        LAG(rul) OVER (PARTITION BY engine_id ORDER BY cycle) AS prev_rul
    FROM turbofan_readings
)
SELECT
    engine_id,
    cycle,
    prev_rul,
    rul,
    (prev_rul - rul) AS rul_delta
FROM rul_check
WHERE prev_rul IS NOT NULL
  AND rul > prev_rul      -- RUL increased — anomaly
ORDER BY engine_id, cycle;

-- Check 5: Health index out of expected range [0, 1]
SELECT
    engine_id,
    cycle,
    health_index
FROM turbofan_readings
WHERE health_index < 0 OR health_index > 1
ORDER BY engine_id, cycle;


-- ─────────────────────────────────────────────────────────────────────────────
-- SECTION 3: POPULATE ANOMALY LOG
-- ─────────────────────────────────────────────────────────────────────────────

-- Flag engines whose health index drops rapidly (>0.10 in 5 cycles = acute fault)
INSERT INTO anomaly_log (engine_id, cycle, anomaly_type, severity, detail)
SELECT
    curr.engine_id,
    curr.cycle,
    'RAPID_DEGRADATION',
    CASE
        WHEN (curr.health_index - prev.health_index) > 0.20 THEN 'CRITICAL'
        ELSE 'WARNING'
    END,
    'Health index rose ' ||
        ROUND((curr.health_index - prev.health_index) * 100, 1) ||
        '% within 5 cycles'
FROM turbofan_readings AS curr
JOIN turbofan_readings AS prev
    ON  curr.engine_id = prev.engine_id
    AND curr.cycle     = prev.cycle + 5
WHERE (curr.health_index - prev.health_index) > 0.10;

-- Flag engines entering critical zone (RUL < 25 cycles)
INSERT INTO anomaly_log (engine_id, cycle, anomaly_type, severity, detail)
SELECT
    engine_id,
    cycle,
    'CRITICAL_RUL',
    'CRITICAL',
    'Engine RUL below 25 cycles — immediate maintenance required'
FROM turbofan_readings
WHERE rul_capped < 25
  AND NOT EXISTS (
      SELECT 1 FROM anomaly_log al
      WHERE al.engine_id = turbofan_readings.engine_id
        AND al.anomaly_type = 'CRITICAL_RUL'
        AND al.cycle <= turbofan_readings.cycle
  );


-- ─────────────────────────────────────────────────────────────────────────────
-- SECTION 4: ANALYTICAL VIEWS (referenced by Power BI)
-- ─────────────────────────────────────────────────────────────────────────────

-- VIEW 1: Fleet-level KPI summary
CREATE VIEW IF NOT EXISTS vw_fleet_kpis AS
SELECT
    COUNT(DISTINCT engine_id)                               AS total_engines,
    ROUND(AVG(total_rul), 1)                               AS avg_engine_life_cycles,
    ROUND(MIN(total_rul), 1)                               AS shortest_life,
    ROUND(MAX(total_rul), 1)                               AS longest_life,
    SUM(CASE WHEN failure_stage = 'Critical' THEN 1 ELSE 0 END)
                                                            AS engines_critical,
    SUM(CASE WHEN failure_stage = 'Severe'   THEN 1 ELSE 0 END)
                                                            AS engines_severe,
    ROUND(AVG(mean_health), 3)                             AS avg_fleet_health,
    ROUND(AVG(final_health), 3)                            AS avg_health_at_failure
FROM engine_summary;

-- VIEW 2: Engine health trajectory (for Power BI line charts)
CREATE VIEW IF NOT EXISTS vw_engine_health_trajectory AS
SELECT
    engine_id,
    cycle,
    cycle_pct,
    rul_capped,
    health_index,
    health_stage,
    op_regime,
    CASE
        WHEN health_index < 0.30 THEN 'Healthy'
        WHEN health_index < 0.55 THEN 'Degrading'
        WHEN health_index < 0.75 THEN 'Alert'
        ELSE 'Critical'
    END AS alert_level
FROM turbofan_readings
ORDER BY engine_id, cycle;

-- VIEW 3: Sensor degradation profile (average sensor value per cycle_pct bucket)
CREATE VIEW IF NOT EXISTS vw_sensor_degradation AS
SELECT
    ROUND(cycle_pct * 20) / 20.0   AS life_pct_bucket,
    ROUND(AVG(sensor_02), 3)       AS avg_sensor_02,
    ROUND(AVG(sensor_03), 3)       AS avg_sensor_03,
    ROUND(AVG(sensor_04), 3)       AS avg_sensor_04,
    ROUND(AVG(sensor_07), 3)       AS avg_sensor_07,
    ROUND(AVG(sensor_11), 3)       AS avg_sensor_11,
    ROUND(AVG(sensor_12), 3)       AS avg_sensor_12,
    ROUND(AVG(sensor_15), 3)       AS avg_sensor_15,
    ROUND(AVG(health_index), 3)    AS avg_health_index,
    COUNT(DISTINCT engine_id)      AS engine_count
FROM turbofan_readings
GROUP BY ROUND(cycle_pct * 20) / 20.0
ORDER BY life_pct_bucket;

-- VIEW 4: Operational regime comparison
CREATE VIEW IF NOT EXISTS vw_regime_analysis AS
SELECT
    op_regime,
    COUNT(DISTINCT engine_id)               AS engine_count,
    ROUND(AVG(rul_capped), 1)               AS avg_rul,
    ROUND(AVG(health_index), 3)             AS avg_health,
    ROUND(MIN(rul_capped), 1)               AS min_rul,
    ROUND(MAX(rul_capped), 1)               AS max_rul,
    COUNT(*)                                AS total_readings,
    SUM(CASE WHEN health_stage = 'Critical'
             THEN 1 ELSE 0 END)             AS critical_readings
FROM turbofan_readings
GROUP BY op_regime
ORDER BY avg_rul;

-- VIEW 5: Engine maintenance priority queue
CREATE VIEW IF NOT EXISTS vw_maintenance_queue AS
SELECT
    es.engine_id,
    es.total_rul                                            AS life_cycles,
    es.final_health,
    es.failure_stage,
    es.primary_regime,
    COALESCE(al.alert_count, 0)                            AS anomaly_count,
    CASE
        WHEN es.final_health    > 0.80 THEN 1
        WHEN es.final_health    > 0.60 THEN 2
        WHEN es.final_health    > 0.40 THEN 3
        ELSE 4
    END                                                     AS priority_rank,
    CASE
        WHEN es.final_health    > 0.80 THEN 'Immediate'
        WHEN es.final_health    > 0.60 THEN 'Within 50 cycles'
        WHEN es.final_health    > 0.40 THEN 'Within 100 cycles'
        ELSE 'Scheduled'
    END                                                     AS maintenance_window
FROM engine_summary es
LEFT JOIN (
    SELECT engine_id, COUNT(*) AS alert_count
    FROM anomaly_log
    GROUP BY engine_id
) al ON es.engine_id = al.engine_id
ORDER BY priority_rank DESC, es.final_health DESC;

-- VIEW 6: Weekly/rolling 10-cycle degradation rate per engine
CREATE VIEW IF NOT EXISTS vw_degradation_rate AS
SELECT
    engine_id,
    cycle,
    health_index,
    LAG(health_index, 10) OVER (
        PARTITION BY engine_id ORDER BY cycle
    )                                                       AS health_10_cycles_ago,
    ROUND(
        health_index - LAG(health_index, 10) OVER (
            PARTITION BY engine_id ORDER BY cycle
        ),
    4)                                                      AS degradation_rate_10cy,
    rul_capped
FROM turbofan_readings
WHERE cycle > 10
ORDER BY engine_id, cycle;


-- ─────────────────────────────────────────────────────────────────────────────
-- SECTION 5: SUMMARY QUERIES (run to validate your data)
-- ─────────────────────────────────────────────────────────────────────────────

-- How many readings per health stage?
SELECT
    health_stage,
    COUNT(*)                                               AS row_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)   AS pct_of_total,
    ROUND(AVG(rul_capped), 1)                             AS avg_rul
FROM turbofan_readings
GROUP BY health_stage
ORDER BY avg_rul;

-- Top 10 engines closest to failure
SELECT
    engine_id,
    MAX(cycle)          AS last_cycle,
    MIN(rul_capped)     AS current_rul,
    MAX(health_index)   AS peak_health_index,
    failure_stage
FROM turbofan_readings tr
JOIN engine_summary    es USING (engine_id)
GROUP BY tr.engine_id, failure_stage
ORDER BY current_rul ASC
LIMIT 10;

-- Anomaly summary
SELECT
    anomaly_type,
    severity,
    COUNT(*)    AS occurrences,
    COUNT(DISTINCT engine_id) AS engines_affected
FROM anomaly_log
GROUP BY anomaly_type, severity
ORDER BY occurrences DESC;

-- Sensor statistics overview
SELECT
    'sensor_02' AS sensor, MIN(sensor_02) AS min_val, MAX(sensor_02) AS max_val, ROUND(AVG(sensor_02), 2) AS avg_val, ROUND(STDEV(sensor_02), 3) AS std_val FROM turbofan_readings
UNION ALL
SELECT 'sensor_04', MIN(sensor_04), MAX(sensor_04), ROUND(AVG(sensor_04), 2), ROUND(STDEV(sensor_04), 3) FROM turbofan_readings
UNION ALL
SELECT 'sensor_07', MIN(sensor_07), MAX(sensor_07), ROUND(AVG(sensor_07), 2), ROUND(STDEV(sensor_07), 3) FROM turbofan_readings
UNION ALL
SELECT 'sensor_11', MIN(sensor_11), MAX(sensor_11), ROUND(AVG(sensor_11), 2), ROUND(STDEV(sensor_11), 3) FROM turbofan_readings
UNION ALL
SELECT 'sensor_15', MIN(sensor_15), MAX(sensor_15), ROUND(AVG(sensor_15), 2), ROUND(STDEV(sensor_15), 3) FROM turbofan_readings;
-- Note: STDEV() is PostgreSQL/MySQL syntax; use STDEV() in SQLite with an extension
-- or replace with: SQRT(AVG(sensor_02*sensor_02) - AVG(sensor_02)*AVG(sensor_02))
