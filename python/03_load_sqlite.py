"""
================================================================================
NASA C-MAPSS Turbofan Engine RUL Analysis
Step 3: Build the SQLite database
================================================================================
  1. Run sql/01_schema.sql (creates empty tables)
  2. Load the processed CSVs into those tables
  3. Run sql/02_checks_anomalies_views.sql (integrity checks, anomaly log, views)
  4. Print the integrity-check results and key view outputs

Output: data/turbofan.db  — open it in DB Browser for SQLite, or connect
Power BI through an SQLite ODBC driver (or just import the CSVs directly).
================================================================================
"""

import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
SQL_DIR = BASE_DIR / "sql"
DB_PATH = BASE_DIR / "data" / "turbofan.db"

LOADS = {
    "turbofan_readings": ("powerbi_main.csv", {"RUL": "rul", "RUL_capped": "rul_capped"}),
    "engine_summary": ("powerbi_engine_summary.csv", {}),
    "test_engine_status": ("test_engine_status.csv", {}),
    "hi_thresholds": ("hi_thresholds.csv", {}),
}


def split_statements(sql: str):
    """Split a script on ';' at line ends, keeping statements whole."""
    stmts, buf = [], []
    for line in sql.splitlines():
        if line.strip().startswith("--") and not buf:
            continue
        buf.append(line)
        if line.rstrip().endswith(";"):
            stmt = "\n".join(buf).strip()
            if stmt and sqlite3.complete_statement(stmt):
                stmts.append(stmt)
                buf = []
    return stmts


def main():
    print("\n" + "=" * 65)
    print(" NASA C-MAPSS — Build SQLite database")
    print("=" * 65)

    con = sqlite3.connect(DB_PATH)
    con.executescript((SQL_DIR / "01_schema.sql").read_text())

    for table, (csv, rename) in LOADS.items():
        df = pd.read_csv(PROC / csv).rename(columns=rename)
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        df[[c for c in cols if c in df.columns]].to_sql(table, con, if_exists="append", index=False)
        print(f"  Loaded {table:<20} {len(df):>6,} rows")

    print("\n  Running checks, anomaly detection and views...")
    for stmt in split_statements((SQL_DIR / "02_checks_anomalies_views.sql").read_text()):
        cur = con.execute(stmt)
        if cur.description:                       # a SELECT → show what came back
            label = next((l.strip("- ").strip() for l in stmt.splitlines()
                          if l.strip().startswith("--") and any(ch.isalpha() for ch in l)
                          and "SECTION" not in l), stmt.splitlines()[0])
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            print(f"\n  ▸ {label}")
            if not rows:
                print("    (0 rows — check passed)")
            else:
                print(pd.DataFrame(rows, columns=cols).to_string(index=False, max_rows=12)
                      .replace("\n", "\n    ").join(["    ", ""]))
    con.commit()

    for view in ["vw_fleet_kpis", "vw_alert_check"]:
        print(f"\n  ▸ {view}")
        print("    " + pd.read_sql(f"SELECT * FROM {view}", con).to_string(index=False)
              .replace("\n", "\n    "))
    con.close()
    print(f"\n  ✓ Database written to {DB_PATH}\n")


if __name__ == "__main__":
    main()
