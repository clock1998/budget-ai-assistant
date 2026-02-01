import numpy as np
from pgvector.psycopg2 import register_vector
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B", 
    device="cuda", 
    model_kwargs={"dtype": "bfloat16"}
)

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="quebec_business",
    user="postgres",
    password="postgres"
)

# Register pgvector extension
register_vector(conn)

cursor = conn.cursor()

# Enable pgvector extension
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.commit()

# Verify installation
cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
vec_version = cursor.fetchone()
print(f"pgvector_version={vec_version[0] if vec_version else 'not installed'}")

# Drop table if exists
cursor.execute("DROP TABLE IF EXISTS vec_documents")

# Create table with vector column
cursor.execute("""
    CREATE TABLE vec_documents (
        id SERIAL PRIMARY KEY,
        business_name_embedding vector(4096),
        business_name TEXT,
        business_domain TEXT,
        business_niche_description TEXT
    )
""")
conn.commit()

businesses = pd.read_csv('data.csv')
embeddings = model.encode(businesses['business_name'].tolist(), prompt_name="document")

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

# Create an index for faster similarity search (optional but recommended)
cursor.execute("""
    CREATE INDEX ON vec_documents USING ivfflat (business_name_embedding vector_cosine_ops)
    WITH (lists = 100)
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
