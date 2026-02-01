import os
import psycopg2
import pandas as pd

# Connect to PostgreSQL
conn = psycopg2.connect(
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    port=int(os.environ.get("POSTGRES_PORT", 5432)),
    database=os.environ.get("POSTGRES_DATABASE", "default"),
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"]
)

cursor = conn.cursor()

# Drop table if exists
cursor.execute("DROP TABLE IF EXISTS fts_documents")

# Create table with tsvector column for full-text search
cursor.execute("""
    CREATE TABLE fts_documents (
        id SERIAL PRIMARY KEY,
        business_name TEXT,
        business_domain TEXT,
        business_niche_description TEXT,
        search_vector tsvector
    )
""")
conn.commit()

# Create GIN index for fast full-text search
cursor.execute("""
    CREATE INDEX fts_documents_search_idx ON fts_documents USING GIN (search_vector)
""")
conn.commit()

businesses = pd.read_csv('data.csv')

for i, row in businesses.iterrows():
    # Insert with tsvector generated from business_name, domain, and description
    cursor.execute('''
        INSERT INTO fts_documents (
            business_name, 
            business_domain, 
            business_niche_description,
            search_vector
        ) VALUES (%s, %s, %s, 
            setweight(to_tsvector('french', coalesce(%s, '')), 'A') ||
            setweight(to_tsvector('french', coalesce(%s, '')), 'B') ||
            setweight(to_tsvector('french', coalesce(%s, '')), 'C')
        )''',
        (row['business_name'], row['business_domain'], row['business_niche_description'],
         row['business_name'], row['business_domain'], row['business_niche_description']))

conn.commit()

# Verify data
cursor.execute("SELECT id, business_name, business_domain FROM fts_documents LIMIT 5")
print("Sample data:")
for row in cursor.fetchall():
    print(row)

# Example FTS query
cursor.execute("""
    SELECT id, business_name, business_domain,
           ts_rank(search_vector, query) AS rank
    FROM fts_documents, plainto_tsquery('english', 'restaurant') query
    WHERE search_vector @@ query
    ORDER BY rank DESC
    LIMIT 5
""")
print("\nExample FTS search for 'restaurant':")
for row in cursor.fetchall():
    print(row)

conn.close()


