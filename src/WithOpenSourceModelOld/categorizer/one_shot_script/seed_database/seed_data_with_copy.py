import os
import io
import csv
import psycopg2

# Connect to PostgreSQL
db = psycopg2.connect(
    host=os.environ["POSTGRES_HOST"],
    port=int(os.environ.get("POSTGRES_PORT", 5432)),
    database=os.environ.get("POSTGRES_DATABASE", "default"),
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"]
)

cursor = db.cursor()
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

# Drop table if exists
cursor.execute("DROP TABLE IF EXISTS business")

# Create table (without fts_vector — will be populated after import)
cursor.execute("""
    CREATE TABLE business (
        id SERIAL PRIMARY KEY,
        business_name TEXT,
        business_domain TEXT,
        business_niche_description TEXT,
        business_name_embedding bit(128)
    )
""")
db.commit()

# Bulk import using COPY
csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data_with_embeddings.csv')
csv_path = os.path.abspath(csv_path)
print(f"Importing data from {csv_path} ...")

with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header

    # Re-serialize with QUOTE_ALL so embedded commas are safely quoted
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    for row in reader:
        writer.writerow(row[1:])  # skip Id column
    buf.seek(0)

    cursor.copy_expert(
        "COPY business (business_name, business_domain, business_niche_description, business_name_embedding) "
        "FROM STDIN WITH (FORMAT csv)",
        buf
    )
db.commit()

cursor.execute("SELECT count(*) FROM business")
count = cursor.fetchone()[0]
print(f"Imported {count} rows")

# Add fts_vector column and populate it from existing text columns
print("Updating fts_vector column...")
cursor.execute("ALTER TABLE business ADD COLUMN fts_vector tsvector")
cursor.execute("""
    UPDATE business 
    SET fts_vector = 
    setweight(to_tsvector('simple', coalesce(business_name, '')), 'A') || 
    setweight(to_tsvector('french', coalesce(business_name, '')), 'B') 
""")
db.commit()

# Create indexes after data is loaded (faster than creating before)
print("Creating indexes...")
cursor.execute("CREATE INDEX business_search_idx ON business USING GIN (fts_vector)")
db.commit()
print("Indexes created")

# Verify data
cursor.execute("SELECT id, business_name, business_domain FROM business LIMIT 5")
print("\nSample data:")
for row in cursor.fetchall():
    print(row)

db.close()
print("Done.")


