-- ================================================================================
-- NASA C-MAPSS Turbofan Engine RUL Analysis
-- SQL Layer, part 2: Integrity checks, anomaly log, analytical views  (SQLite)
-- Run AFTER the tables are loaded (python/03_load_sqlite.py does both).
-- ================================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- SECTION 1: DATA INTEGRITY CHECKS  (each should return 0 rows / zeros)
-- ─────────────────────────────────────────────────────────────────────────────

-- Check 1: duplicate readings (same engine + cycle)
SELECT engine_id, cycle, COUNT(*) AS duplicate_count
FROM turbofan_readings
GROUP BY engine_id, cycle
HAVING COUNT(*) > 1;

-- Check 2: gaps in an engine's cycle timeline
WITH cycle_gaps AS (
    SELECT engine_id, cycle,
           cycle - LAG(cycle) OVER (PARTITION BY engine_id ORDER BY cycle) AS gap
    FROM turbofan_readings
)
SELECT engine_id, cycle, gap - 1 AS missing_cycles
FROM cycle_gaps
WHERE gap > 1;

-- Check 3: nulls in key columns
SELECT
    SUM(engine_id    IS NULL) AS null_engine_id,
    SUM(cycle        IS NULL) AS null_cycle,
    SUM(rul          IS NULL) AS null_rul,
    SUM(health_index IS NULL) AS null_health_index
FROM turbofan_readings;

-- Check 4: RUL must fall by exactly 1 each cycle
WITH rul_check AS (
    SELECT engine_id, cycle, rul,
           LAG(rul) OVER (PARTITION BY engine_id ORDER BY cycle) AS prev_rul
    FROM turbofan_readings
)
SELECT engine_id, cycle, prev_rul, rul
FROM rul_check
WHERE prev_rul IS NOT NULL AND prev_rul - rul <> 1;

-- Check 5: every engine must end at RUL = 0 (run-to-failure)
SELECT engine_id, MIN(rul) AS final_rul
FROM turbofan_readings
GROUP BY engine_id
HAVING MIN(rul) <> 0;


-- ─────────────────────────────────────────────────────────────────────────────
-- SECTION 2: ANOMALY LOG
-- ─────────────────────────────────────────────────────────────────────────────

-- Rapid degradation: health index ROSE (got worse) by at least the
-- 'rapid_rise_5cy' threshold (top 1% of 5-cycle jumps) within 5 cycles.
-- CRITICAL when the jump is at least 1.5x that threshold.
INSERT INTO anomaly_log (engine_id, cycle, anomaly_type, severity, detail)
WITH jumps AS (
    SELECT engine_id, cycle,
           health_index - LAG(health_index, 5)
               OVER (PARTITION BY engine_id ORDER BY cycle) AS rise_5cy
    FROM turbofan_readings
),
t AS (SELECT value AS rapid FROM hi_thresholds WHERE name = 'rapid_rise_5cy')
SELECT j.engine_id, j.cycle, 'RAPID_DEGRADATION',
       CASE WHEN j.rise_5cy >= 1.5 * t.rapid THEN 'CRITICAL' ELSE 'WARNING' END,
       'Health index rose ' || ROUND(j.rise_5cy, 3) || ' points in 5 cycles'
FROM jumps j CROSS JOIN t
WHERE j.rise_5cy >= t.rapid;

-- Alert raised: one row per engine, at its first sustained Alert
-- (taken from engine_summary so it matches the Python lead-time analysis).
INSERT INTO anomaly_log (engine_id, cycle, anomaly_type, severity, detail)
SELECT engine_id, first_alert_cycle, 'ALERT_RAISED', 'CRITICAL',
       'Health index above Alert threshold for 3 cycles; failed '
           || alert_lead_cycles || ' cycles later'
FROM engine_summary
WHERE first_alert_cycle IS NOT NULL;


-- ─────────────────────────────────────────────────────────────────────────────
-- SECTION 3: ANALYTICAL VIEWS (for Power BI)
-- ─────────────────────────────────────────────────────────────────────────────

-- VIEW 1: Fleet KPIs (training fleet = historical, run-to-failure engines)
CREATE VIEW vw_fleet_kpis AS
SELECT
    COUNT(*)                                           AS total_engines,
    ROUND(AVG(life_cycles), 1)                         AS avg_life_cycles,
    MIN(life_cycles)                                   AS shortest_life,
    MAX(life_cycles)                                   AS longest_life,
    SUM(first_alert_cycle IS NOT NULL)                 AS engines_alerted_before_failure,
    ROUND(AVG(alert_lead_cycles), 1)                   AS avg_alert_lead_cycles,
    MIN(alert_lead_cycles)                             AS min_alert_lead_cycles,
    ROUND(AVG(start_health), 3)                        AS avg_start_health,
    ROUND(AVG(final_health), 3)                        AS avg_health_at_failure
FROM engine_summary;

-- VIEW 2: Health trajectory per engine (line charts)
CREATE VIEW vw_engine_health_trajectory AS
SELECT engine_id, cycle, cycle_pct, rul_capped, health_index, rul_band, alert_level
FROM turbofan_readings;

-- VIEW 3: Average sensor profile across the life of an engine (5% buckets)
CREATE VIEW vw_sensor_degradation AS
SELECT
    ROUND(cycle_pct * 20) / 20.0  AS life_pct_bucket,
    ROUND(AVG(sensor_02), 3) AS avg_sensor_02, ROUND(AVG(sensor_03), 3) AS avg_sensor_03,
    ROUND(AVG(sensor_04), 3) AS avg_sensor_04, ROUND(AVG(sensor_07), 3) AS avg_sensor_07,
    ROUND(AVG(sensor_08), 3) AS avg_sensor_08, ROUND(AVG(sensor_11), 3) AS avg_sensor_11,
    ROUND(AVG(sensor_12), 3) AS avg_sensor_12, ROUND(AVG(sensor_13), 3) AS avg_sensor_13,
    ROUND(AVG(sensor_15), 3) AS avg_sensor_15, ROUND(AVG(sensor_20), 3) AS avg_sensor_20,
    ROUND(AVG(sensor_21), 3) AS avg_sensor_21,
    ROUND(AVG(health_index), 3) AS avg_health_index,
    COUNT(DISTINCT engine_id)    AS engine_count
FROM turbofan_readings
GROUP BY life_pct_bucket;

-- VIEW 4: 10-cycle health change per engine
-- (window computed first, filter after, so no cycles are skipped)
CREATE VIEW vw_degradation_rate AS
WITH w AS (
    SELECT engine_id, cycle, health_index, rul_capped,
           LAG(health_index, 10) OVER (PARTITION BY engine_id ORDER BY cycle) AS health_10_ago
    FROM turbofan_readings
)
SELECT engine_id, cycle, health_index, health_10_ago,
       ROUND(health_index - health_10_ago, 4) AS change_10cy,
       rul_capped
FROM w
WHERE health_10_ago IS NOT NULL;

-- VIEW 5: Maintenance queue — the TEST engines (still in service).
-- Ranked by current health index; rank 1 = most urgent.
-- true_rul is shown only so the ranking can be checked against the answer key.
CREATE VIEW vw_maintenance_queue AS
SELECT
    RANK() OVER (ORDER BY health_index DESC)            AS priority_rank,
    engine_id,
    last_cycle,
    ROUND(health_index, 3)                              AS health_index,
    alert_level,
    CASE alert_level
        WHEN 'Alert'   THEN 'Schedule now'
        WHEN 'Warning' THEN 'Plan within next maintenance window'
        ELSE 'Monitor'
    END                                                 AS recommended_action,
    true_rul                                            AS true_rul_answer_key
FROM test_engine_status;

-- VIEW 6: Does the alert level match how much life was actually left?
CREATE VIEW vw_alert_check AS
SELECT
    alert_level,
    COUNT(*)                                          AS engines,
    SUM(true_rul < 25)                                AS fail_within_25,
    SUM(true_rul BETWEEN 25 AND 49)                   AS fail_in_25_to_49,
    SUM(true_rul >= 50)                               AS fail_after_50,
    ROUND(AVG(true_rul), 1)                           AS avg_true_rul,
    MIN(true_rul)                                     AS min_true_rul
FROM test_engine_status
GROUP BY alert_level;


-- ─────────────────────────────────────────────────────────────────────────────
-- SECTION 4: SUMMARY QUERIES
-- ─────────────────────────────────────────────────────────────────────────────

-- Readings per RUL band (bands are cut from RUL, so these shares follow from
-- engine lifespans: every engine has exactly 25 'Critical' cycles)
SELECT rul_band, COUNT(*) AS row_count,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_total,
       ROUND(AVG(health_index), 3) AS avg_health_index
FROM turbofan_readings
GROUP BY rul_band
ORDER BY avg_health_index;

-- Top 10 most urgent test engines
SELECT * FROM vw_maintenance_queue ORDER BY priority_rank LIMIT 10;

-- Anomaly summary
SELECT anomaly_type, severity, COUNT(*) AS occurrences,
       COUNT(DISTINCT engine_id) AS engines_affected
FROM anomaly_log
GROUP BY anomaly_type, severity
ORDER BY occurrences DESC;

-- Sensor statistics (population standard deviation via a mean CTE —
-- avoids STDEV(), which SQLite does not have)
WITH m AS (
    SELECT AVG(sensor_02) a02, AVG(sensor_04) a04, AVG(sensor_07) a07,
           AVG(sensor_11) a11, AVG(sensor_15) a15
    FROM turbofan_readings
)
SELECT 'sensor_02' AS sensor, MIN(sensor_02) AS min_val, MAX(sensor_02) AS max_val,
       ROUND(m.a02, 3) AS avg_val, ROUND(SQRT(AVG((sensor_02 - m.a02) * (sensor_02 - m.a02))), 4) AS std_val
FROM turbofan_readings, m
UNION ALL
SELECT 'sensor_04', MIN(sensor_04), MAX(sensor_04), ROUND(m.a04, 3),
       ROUND(SQRT(AVG((sensor_04 - m.a04) * (sensor_04 - m.a04))), 4) FROM turbofan_readings, m
UNION ALL
SELECT 'sensor_07', MIN(sensor_07), MAX(sensor_07), ROUND(m.a07, 3),
       ROUND(SQRT(AVG((sensor_07 - m.a07) * (sensor_07 - m.a07))), 4) FROM turbofan_readings, m
UNION ALL
SELECT 'sensor_11', MIN(sensor_11), MAX(sensor_11), ROUND(m.a11, 3),
       ROUND(SQRT(AVG((sensor_11 - m.a11) * (sensor_11 - m.a11))), 4) FROM turbofan_readings, m
UNION ALL
SELECT 'sensor_15', MIN(sensor_15), MAX(sensor_15), ROUND(m.a15, 3),
       ROUND(SQRT(AVG((sensor_15 - m.a15) * (sensor_15 - m.a15))), 4) FROM turbofan_readings, m;
