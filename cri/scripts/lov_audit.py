import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from collections import defaultdict
from config.db import get_db_connection
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
# ── Load extracted JSON ────────────────────────────────────────────────────────
json_path = os.path.join(os.path.dirname(__file__), "data", "lov_mapping_dynamic.json")
with open(json_path, "r", encoding="utf-8") as f:
    EXTRACTED = json.load(f)

extracted_tables  = set(EXTRACTED.keys())
extracted_cols    = {(tbl, col) for tbl, cols in EXTRACTED.items() for col in cols}
extracted_labels  = sum(len(labels) for cols in EXTRACTED.values() for labels in cols.values())

print("=" * 65)
print("  LOV EXTRACTION AUDIT REPORT")
print("=" * 65)

with get_db_connection() as conn:
    cursor = conn.cursor()

    # ── 1. DB master: total LOV entries ───────────────────────────────────────
    cursor.execute("SELECT COUNT(*) FROM COM_L_LOV_DETAILS WHERE LOV_Details_Name IS NOT NULL")
    db_total_lov_entries = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT LOV_ID) FROM COM_L_LOV_DETAILS")
    db_total_lov_groups = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT LOV_Details_ID) FROM COM_L_LOV_DETAILS")
    db_total_lov_ids = cursor.fetchone()[0]

    print(f"\n📋 COM_L_LOV_DETAILS (master)")
    print(f"   Total entries (rows)       : {db_total_lov_entries}")
    print(f"   Distinct LOV groups        : {db_total_lov_groups}")
    print(f"   Distinct LOV_Details_IDs   : {db_total_lov_ids}")

    # ── 2. DB: tables that HAVE LOV-like columns ──────────────────────────────
    cursor.execute("""
        SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME IN (
            SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE'
        )
        AND (
            COLUMN_NAME LIKE '%_LOV_ID%'
            OR COLUMN_NAME LIKE '%Status%'
            OR COLUMN_NAME LIKE '%STATUS%'
        )
    """)
    rows = cursor.fetchall()

    db_lov_cols_by_table = defaultdict(list)
    for row in rows:
        db_lov_cols_by_table[row.TABLE_NAME].append((row.TABLE_SCHEMA, row.COLUMN_NAME))

    db_tables_with_lov   = set(db_lov_cols_by_table.keys())
    db_total_lov_cols    = sum(len(v) for v in db_lov_cols_by_table.values())

    print(f"\n🗄️  DB tables with LOV/Status columns")
    print(f"   Tables found in schema     : {len(db_tables_with_lov)}")
    print(f"   Total LOV-like columns     : {db_total_lov_cols}")

    # ── 3. Extraction summary ─────────────────────────────────────────────────
    print(f"\n✅ Extracted into JSON")
    print(f"   Tables extracted           : {len(extracted_tables)}")
    print(f"   Columns extracted          : {len(extracted_cols)}")
    print(f"   Label→ID pairs extracted   : {extracted_labels}")

    # ── 4. Coverage % ─────────────────────────────────────────────────────────
    pct_tables = len(extracted_tables) / len(db_tables_with_lov) * 100 if db_tables_with_lov else 0
    pct_cols   = len(extracted_cols)   / db_total_lov_cols          * 100 if db_total_lov_cols   else 0
    pct_labels = extracted_labels      / db_total_lov_entries        * 100 if db_total_lov_entries else 0

    print(f"\n📊 Coverage")
    print(f"   Table  coverage            : {len(extracted_tables)}/{len(db_tables_with_lov)}  ({pct_tables:.1f}%)")
    print(f"   Column coverage            : {len(extracted_cols)}/{db_total_lov_cols}  ({pct_cols:.1f}%)")
    print(f"   Label  coverage            : {extracted_labels}/{db_total_lov_entries}  ({pct_labels:.1f}%)")

    # ── 5. MISSED tables (in DB but not in JSON) ──────────────────────────────
    missed_tables = db_tables_with_lov - extracted_tables
    print(f"\n⚠️  Tables with LOV columns NOT extracted ({len(missed_tables)})")
    if missed_tables:
        for tbl in sorted(missed_tables):
            cols = db_lov_cols_by_table[tbl]
            col_names = ", ".join(c for _, c in cols)
            print(f"   • {tbl}  →  [{col_names}]")
    else:
        print("   None — full coverage!")

    # ── 6. MISSED columns per extracted table ─────────────────────────────────
    print(f"\n⚠️  Columns missed inside already-extracted tables")
    any_missed_col = False
    for tbl in sorted(extracted_tables):
        db_cols_for_tbl  = {c for _, c in db_lov_cols_by_table.get(tbl, [])}
        ext_cols_for_tbl = set(EXTRACTED[tbl].keys())
        missed_cols      = db_cols_for_tbl - ext_cols_for_tbl
        if missed_cols:
            any_missed_col = True
            print(f"   • {tbl}  →  {sorted(missed_cols)}")
    if not any_missed_col:
        print("   None — all columns captured for extracted tables!")

    # ── 7. Diagnose WHY columns were missed ───────────────────────────────────
    print(f"\n🔬 Diagnosing missed columns (checking if they hold non-numeric or all-NULL values) ...")
    diagnose_tables = missed_tables | {
        tbl for tbl in extracted_tables
        if (set(c for _, c in db_lov_cols_by_table.get(tbl, [])) - set(EXTRACTED[tbl].keys()))
    }

    no_data_cols      = []
    non_numeric_cols  = []
    no_lov_match_cols = []

    for tbl in sorted(diagnose_tables):
        schema = db_lov_cols_by_table[tbl][0][0]  # grab schema from first entry
        ext_cols = set(EXTRACTED.get(tbl, {}).keys())
        all_db_cols = [c for _, c in db_lov_cols_by_table[tbl]]
        missed = [c for c in all_db_cols if c not in ext_cols]

        for col in missed:
            try:
                cursor.execute(f"SELECT TOP 10 [{col}] FROM [{schema}].[{tbl}] WHERE [{col}] IS NOT NULL")
                samples = [row[0] for row in cursor.fetchall()]
                if not samples:
                    no_data_cols.append(f"{tbl}.{col}")
                else:
                    try:
                        [int(v) for v in samples]
                        # numeric but no LOV match
                        no_lov_match_cols.append((f"{tbl}.{col}", samples[:5]))
                    except (ValueError, TypeError):
                        non_numeric_cols.append((f"{tbl}.{col}", samples[:5]))
            except Exception as e:
                no_data_cols.append(f"{tbl}.{col}  (query error: {e})")

    if no_data_cols:
        print(f"\n   🟡 All-NULL / empty columns ({len(no_data_cols)}) — nothing to extract:")
        for c in no_data_cols:
            print(f"      - {c}")

    if non_numeric_cols:
        print(f"\n   🟠 Non-numeric values ({len(non_numeric_cols)}) — not LOV IDs, skipped correctly:")
        for c, samples in non_numeric_cols:
            print(f"      - {c}  e.g. {samples}")

    if no_lov_match_cols:
        print(f"\n   🔴 Numeric but NO match in COM_L_LOV_DETAILS ({len(no_lov_match_cols)}) — stale/unmapped IDs:")
        for c, samples in no_lov_match_cols:
            print(f"      - {c}  IDs: {samples}")

print("\n" + "=" * 65)
print("  AUDIT COMPLETE")
print("=" * 65)
