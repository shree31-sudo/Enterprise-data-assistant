import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=10.225.30.26;"
    "DATABASE=LAMI_CNC_PROD0302;"
    "UID=sa;"
    "PWD=sql@123;"
    "TrustServerCertificate=yes;"
    "Connection Timeout=5;"
)

print("✅ Connected to SQL Server")
conn.close()
