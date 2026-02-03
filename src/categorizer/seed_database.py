import os
import psycopg2
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2", 
    device="cuda"
)
# Connect to PostgreSQL
db = psycopg2.connect(
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    port=int(os.environ.get("POSTGRES_PORT", 5432)),
    database=os.environ.get("POSTGRES_DATABASE", "default"),
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"]
)

cursor = db.cursor()
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
# Drop table if exists
cursor.execute("DROP TABLE IF EXISTS business")
# Create table with tsvector column for full-text search
cursor.execute("""
    CREATE TABLE business (
        id SERIAL PRIMARY KEY,
        business_name TEXT,
        business_domain TEXT,
        business_niche_description TEXT,
        business_name_embedding vector(768),
        fts_vector tsvector
    )
""")
# Create GIN index for fast full-text search
cursor.execute("""
    CREATE INDEX business_search_idx ON business USING GIN (fts_vector)
""")
db.commit()

businesses = pd.read_csv('data.csv')

# Handle NaN values in business_name column - fill with empty string or drop rows
businesses['business_name'] = businesses['business_name'].fillna('')
businesses['business_domain'] = businesses['business_domain'].fillna('')
businesses['business_niche_description'] = businesses['business_niche_description'].fillna('')

# Filter out rows with empty business names for encoding
business_names = businesses['business_name'].tolist()
embeddings = model.encode(business_names)

for i, row in businesses.iterrows():
    embedding_list = embeddings[i].astype(np.float32).tolist()
    # Insert with tsvector generated from business_name, domain, and description
    cursor.execute('''
        INSERT INTO business (
            business_name, 
            business_domain, 
            business_niche_description,
            fts_vector,
            business_name_embedding
        ) VALUES (
            %s,
            %s,
            %s, 
            setweight(to_tsvector('french', coalesce(%s, '')), 'A'),
            %s
        )''',
        (
            row['business_name'], 
            row['business_domain'], 
            row['business_niche_description'],
            row['business_name'], 
            embedding_list
        )
    )

db.commit()

# Verify data
cursor.execute("SELECT id, business_name, business_domain FROM business LIMIT 5")
print("Sample data:")
for row in cursor.fetchall():
    print(row)

db.close()


