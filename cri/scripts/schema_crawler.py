import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import re
from collections import defaultdict
from config.db import get_db_connection

# =================================================
# TEXT NORMALIZATION
# =================================================
def humanize(text: str) -> str:
    text = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(
        r'\b([A-Z])(?:\s+[A-Z])+\b',
        lambda m: m.group(0).replace(" ", ""),
        text
    )

    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


# =================================================
# COLUMN MEANING
# =================================================
def generate_column_meaning(column_name: str, semantic_role: str) -> str:
    base = humanize(column_name)

    if semantic_role == "primary_id":
        return "Unique identifier for each record."

    if semantic_role == "foreign_id":
        return "Identifier linking to another table."

    if semantic_role == "date_field":
        return f"Stores date/time related to {base}."

    if semantic_role == "duration_field":
        return f"Stores duration related to {base}."

    if semantic_role == "boolean_flag":
        return f"True/false flag for {base}."

    if semantic_role == "numeric_field":
        return f"Numeric value for {base}."

    return f"Text information about {base}."


# =================================================
# METADATA FETCH
# =================================================
def fetch_tables(cursor):
    cursor.execute("""
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
    """)
    return [row.TABLE_NAME for row in cursor.fetchall()]


def fetch_columns(cursor):
    cursor.execute("""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
    """)
    columns = defaultdict(list)

    for row in cursor.fetchall():
        columns[row.TABLE_NAME].append({
            "column_name": row.COLUMN_NAME,
            "data_type": row.DATA_TYPE.lower(),
            "nullable": row.IS_NULLABLE == "YES"
        })

    return columns


def fetch_primary_keys(cursor):
    cursor.execute("""
        SELECT KU.TABLE_NAME, KU.COLUMN_NAME
        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS TC
        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE KU
          ON TC.CONSTRAINT_NAME = KU.CONSTRAINT_NAME
        WHERE TC.CONSTRAINT_TYPE = 'PRIMARY KEY'
    """)
    pk_map = defaultdict(list)

    for row in cursor.fetchall():
        pk_map[row.TABLE_NAME].append(row.COLUMN_NAME)

    return pk_map


def fetch_foreign_keys(cursor):
    cursor.execute("""
        SELECT fk_tab.name AS table_name,
               fk_col.name AS column_name
        FROM sys.foreign_keys fk
        JOIN sys.foreign_key_columns fkc
          ON fk.object_id = fkc.constraint_object_id
        JOIN sys.tables fk_tab
          ON fk_tab.object_id = fkc.parent_object_id
        JOIN sys.columns fk_col
          ON fk_col.column_id = fkc.parent_column_id
         AND fk_col.object_id = fk_tab.object_id
    """)
    fk_map = defaultdict(list)

    for row in cursor.fetchall():
        fk_map[row.table_name].append(row.column_name)

    return fk_map


# =================================================
# PROFILING FUNCTIONS
# =================================================
def profile_column(cursor, table, column):
    profile = {}

    try:
        cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
        total = cursor.fetchone()[0]

        cursor.execute(f"""
            SELECT COUNT(DISTINCT [{column}])
            FROM [{table}]
        """)
        distinct = cursor.fetchone()[0]

        profile["distinct_count"] = int(distinct)

        cursor.execute(f"""
            SELECT COUNT(*)
            FROM [{table}]
            WHERE [{column}] IS NULL
        """)
        nulls = cursor.fetchone()[0]

        profile["null_ratio"] = round(nulls / total, 3) if total else 0

        # Sample values (enum hint)
        if distinct <= 20:
            cursor.execute(f"""
                SELECT TOP 10 [{column}]
                FROM [{table}]
                WHERE [{column}] IS NOT NULL
                GROUP BY [{column}]
            """)
            profile["sample_values"] = [
                str(r[0]) for r in cursor.fetchall()
            ]

    except:
        pass

    return profile


# =================================================
# SEMANTIC ROLE INFERENCE
# =================================================
def infer_semantic_role(column_name, data_type, table,
                        primary_keys, foreign_keys):

    if column_name in primary_keys.get(table, []):
        return "primary_id"

    if column_name in foreign_keys.get(table, []):
        return "foreign_id"

    if data_type in ["bit", "boolean"]:
        return "boolean_flag"

    if data_type in ["date","datetime","datetime2","timestamp"]:
        return "date_field"

    if data_type == "time":
        return "duration_field"

    if data_type in [
        "int","bigint","smallint",
        "decimal","numeric","float"
    ]:
        return "numeric_field"

    return "text_field"


# =================================================
# MAIN CRAWLER
# =================================================
def crawl_schema():
    schema = {}

    with get_db_connection() as conn:
        cursor = conn.cursor()

        tables = fetch_tables(cursor)
        columns = fetch_columns(cursor)
        primary_keys = fetch_primary_keys(cursor)
        foreign_keys = fetch_foreign_keys(cursor)

        for table in tables:
            enriched = []

            for col in columns.get(table, []):
                role = infer_semantic_role(
                    col["column_name"],
                    col["data_type"],
                    table,
                    primary_keys,
                    foreign_keys
                )

                col["semantic_role"] = role
                col["meaning"] = generate_column_meaning(
                    col["column_name"], role
                )

                # ⭐ Profiling added here
                profile = profile_column(
                    cursor,
                    table,
                    col["column_name"]
                )
                col.update(profile)

                enriched.append(col)

            schema[table] = {
                "columns": enriched,
                "primary_keys": primary_keys.get(table, []),
                "foreign_keys": foreign_keys.get(table, [])
            }

    return schema


# =================================================
# SAVE
# =================================================
def save_schema(schema, filename="data/schema_raw.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=4)


# =================================================
# ENTRY
# =================================================
if __name__ == "__main__":
    print("📡 Crawling schema with profiling...")
    schema = crawl_schema()
    save_schema(schema)
    print("✅ Schema saved to schema_raw.json")
