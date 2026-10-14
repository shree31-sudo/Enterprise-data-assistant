import bcrypt
import pyodbc

username = input("Username: ")
password = input("Password: ")
role = input("Role (ADMIN/USER): ")

password_hash = bcrypt.hashpw(
    password.encode(),
    bcrypt.gensalt()
).decode()

conn = pyodbc.connect(
    "DRIVER=ODBC Driver 17 for SQL Server;"
    "SERVER=10.225.30.26;"
    "DATABASE=LAMI_DB;"
    "UID=sa;"
    "PWD=sql@123;"
    "TrustServerCertificate=yes;"
)

cursor = conn.cursor()

cursor.execute("""
INSERT INTO dbo.docmind_user_cred
(
    Username,
    PasswordHash,
    Role,
    IsActive
)
VALUES
(
    ?,?,?,1
)
""",
username,
password_hash,
role
)

conn.commit()

print("User Created Successfully")