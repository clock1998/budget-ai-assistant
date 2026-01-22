import sqlite3
import sqlite_vss

db = sqlite3.connect("my_db.db")

# Enable loading extensions and load vss
db.enable_load_extension(True)
sqlite_vss.load(db)

# Verify installation
version = db.execute("select vss_version()").fetchone()[0]
print(f"sqlite-vss version: {version}")