import os
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
import pandas as pd
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

# Enable pgvector extension (must be done BEFORE registering vector type)
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
db.commit()
# Register pgvector extension (after extension is created)
register_vector(db)

# Verify installation
cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
vec_version = cursor.fetchone()
print(f"pgvector_version={vec_version[0] if vec_version else 'not installed'}")

# Drop table if exists
cursor.execute("DROP TABLE IF EXISTS business_vec")

# Create table with vector column (768 dimensions for all-mpnet-base-v2)
cursor.execute("""
    CREATE TABLE business_vec (
        id SERIAL PRIMARY KEY,
        business_name_embedding vector(768),
        business_name TEXT,
        business_domain TEXT,
        business_niche_description TEXT
    )
""")
db.commit()

businesses = pd.read_csv('data.csv')
embeddings = model.encode(businesses['business_name'].tolist())

for i, row in businesses.iterrows():
    embedding_list = embeddings[i].astype(np.float32).tolist()
    # Insert text into documents
    cursor.execute('''INSERT INTO business_vec (
                   business_name_embedding,
                   business_name, 
                   business_domain, 
                   business_niche_description) VALUES (%s, %s, %s, %s)''',
                   (embedding_list, row['business_name'], row['business_domain'], row['business_niche_description']))
    
db.commit()

# Create an index for faster similarity search
cursor.execute("""
    CREATE INDEX ON business_vec USING hnsw (business_name_embedding vector_cosine_ops)
""")
db.commit()

cursor.execute("""
SELECT
    id,
    business_name,
    business_name_embedding
FROM business_vec
LIMIT 2
""")
print(cursor.fetchall())
db.close()
