import sqlite3

# ✅ Create database connection
conn = sqlite3.connect("emotionsense.db")
cursor = conn.cursor()

# ✅ Create table for storing emotion history
cursor.execute("""
CREATE TABLE IF NOT EXISTS emotion_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emotion TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("✅ emotion_history table created successfully.")