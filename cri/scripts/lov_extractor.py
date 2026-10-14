import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from collections import defaultdict
from config.db import get_db_connection

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Config ───────────────────────────────────────────────────────────────
VALUESET_PARAM_TABLE  = "ACM_L_VALUESET_PARAM"   # link table: Valueset_Param_ID -> Valueset_Parameter
VALUESET_PARAM_SCHEMA = "dbo"

ONLY_ACTIVE_STATUS = True   # if True, only pulls master rows where LAMI_Status = 1

# Data types worth checking for LOV-style FK values. Numeric types cover the
# normal case; the char/varchar types are included because some columns
# store the numeric ID as text. Large text/binary/date/float columns are
# skipped since LOV IDs are always small integers.
CANDIDATE_DATA_TYPES = (
    "int", "bigint", "smallint", "tinyint", "decimal", "numeric",
    "char", "varchar", "nchar", "nvarchar",
)

LOV_MAPPING = defaultdict(dict)

with get_db_connection() as conn:
    cursor = conn.cursor()

    # ── Step 1: Load master lookup from ACM_L_VALUESET_PARAM ────────────────
    # Valueset_Param_ID → Valueset_Parameter
    print("📦 Loading LOV master data from ACM_L_VALUESET_PARAM ...")

    status_filter = "AND LAMI_Status = 1" if ONLY_ACTIVE_STATUS else ""

    cursor.execute(f"""
        SELECT Valueset_Param_ID, Valueset_Parameter
        FROM [{VALUESET_PARAM_SCHEMA}].[{VALUESET_PARAM_TABLE}]
        WHERE Valueset_Parameter IS NOT NULL
        {status_filter}
    """)
    lov_master = {}   # Valueset_Param_ID (int) → Valueset_Parameter (str)
    for row in cursor.fetchall():
        lov_master[int(row.Valueset_Param_ID)] = row.Valueset_Parameter.strip()

    print(f"   ✅ Loaded {len(lov_master)} Valueset_Param entries.")

    # ── Step 2: Get every candidate column across every user table ─────────
    # No name filtering — we check every column of a plausible data type and
    # let the value overlap with lov_master decide whether it's LOV-like.
    print("🔍 Scanning all candidate columns in DB ...")
    type_placeholders = ", ".join(["?"] * len(CANDIDATE_DATA_TYPES))
    cursor.execute(f"""
        SELECT c.TABLE_SCHEMA, c.TABLE_NAME, c.COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS c
        JOIN INFORMATION_SCHEMA.TABLES t
          ON t.TABLE_SCHEMA = c.TABLE_SCHEMA AND t.TABLE_NAME = c.TABLE_NAME
        WHERE t.TABLE_TYPE = 'BASE TABLE'
          AND c.DATA_TYPE IN ({type_placeholders})
        ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.COLUMN_NAME
    """, CANDIDATE_DATA_TYPES)

    table_cols = defaultdict(list)
    for row in cursor.fetchall():
        table_cols[(row.TABLE_SCHEMA, row.TABLE_NAME)].append(row.COLUMN_NAME)

    total_tables = len(table_cols)
    print(f"   ✅ {total_tables} tables have candidate columns to check.")

    # ── Step 3: For each table+column, pull distinct values & map via master ───
    processed = 0
    for (schema_name, table_name), cols in table_cols.items():
        for col_name in cols:
            try:
                cursor.execute(f"""
                    SELECT DISTINCT [{col_name}]
                    FROM [{schema_name}].[{table_name}]
                    WHERE [{col_name}] IS NOT NULL
                """)
                raw_values = [row[0] for row in cursor.fetchall()]
                if not raw_values:
                    continue

                # Only keep values that convert cleanly to int
                numeric_ids = []
                for v in raw_values:
                    try:
                        numeric_ids.append(int(v))
                    except (ValueError, TypeError):
                        pass

                if not numeric_ids:
                    continue

                # Keep only IDs that exist in the Valueset_Param master
                col_mapping = {}
                for nid in numeric_ids:
                    if nid in lov_master:
                        col_mapping[str(nid)] = {"description": lov_master[nid]}

                if col_mapping:
                    LOV_MAPPING[table_name][col_name] = col_mapping

            except Exception as e:
                # Some columns may have access issues / conversion errors
                pass  # silently skip; uncomment below for verbose debug
                # print(f"   ⚠️  Skipped [{schema_name}].[{table_name}].[{col_name}]: {e}")

        processed += 1
        if processed % 50 == 0:
            print(f"   ... processed {processed}/{total_tables} tables")

# ── Step 4: Save result ────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), "data", "lov_mapping_dynamic.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(LOV_MAPPING, f, indent=4, ensure_ascii=False)

total_tables_mapped = len(LOV_MAPPING)
total_cols_mapped   = sum(len(cols) for cols in LOV_MAPPING.values())
total_ids_mapped    = sum(
    len(ids)
    for cols in LOV_MAPPING.values()
    for ids in cols.values()
)

print(f"\n✅ LOV mapping saved → {out_path}")
print(f"   Tables  : {total_tables_mapped}")
print(f"   Columns : {total_cols_mapped}")
print(f"   IDs     : {total_ids_mapped}")