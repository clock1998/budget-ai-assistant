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
conn = psycopg2.connect(
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    port=int(os.environ.get("POSTGRES_PORT", 5432)),
    database=os.environ.get("POSTGRES_DATABASE", "default"),
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"]
)

cursor = conn.cursor()

# Enable pgvector extension (must be done BEFORE registering vector type)
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.commit()

# Register pgvector extension (after extension is created)
register_vector(conn)

# Verify installation
cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
vec_version = cursor.fetchone()
print(f"pgvector_version={vec_version[0] if vec_version else 'not installed'}")

# Drop table if exists
cursor.execute("DROP TABLE IF EXISTS vec_documents")

# Create table with vector column (768 dimensions for all-mpnet-base-v2)
cursor.execute("""
    CREATE TABLE vec_documents (
        id SERIAL PRIMARY KEY,
        business_name_embedding vector(768),
        business_name TEXT,
        business_domain TEXT,
        business_niche_description TEXT
    )
""")
conn.commit()

businesses = pd.read_csv('data.csv')
embeddings = model.encode(businesses['business_name'].tolist())

for i, row in businesses.iterrows():
    embedding_list = embeddings[i].astype(np.float32).tolist()
    # Insert text into documents
    cursor.execute('''INSERT INTO vec_documents (
                   business_name_embedding,
                   business_name, 
                   business_domain, 
                   business_niche_description) VALUES (%s, %s, %s, %s)''',
                   (embedding_list, row['business_name'], row['business_domain'], row['business_niche_description']))
    
conn.commit()

# Create an index for faster similarity search
cursor.execute("""
    CREATE INDEX ON vec_documents USING hnsw (business_name_embedding vector_cosine_ops)
""")
conn.commit()

cursor.execute("""
SELECT
    id,
    business_name,
    business_name_embedding
FROM vec_documents
LIMIT 2
""")
print(cursor.fetchall())
conn.close()
