import sqlite3

conn = sqlite3.connect("emotion_data.db")
cursor = conn.cursor()

# ✅ Create text_analysis table for storing text sentiment history
cursor.execute("""
CREATE TABLE IF NOT EXISTS text_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    sentiment TEXT,
    confidence REAL,
    analysis_type TEXT,
    timestamp TEXT
)
""")

cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
''')

conn.commit()
conn.close()

print("✅ text_analysis table created successfully!")