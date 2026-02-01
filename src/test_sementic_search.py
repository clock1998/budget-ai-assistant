import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2", 
    device="cuda", 
    model_kwargs={"dtype": "bfloat16"}
)

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="default",
    user="secret",
    password="secret"
)

cursor = conn.cursor()

# Register pgvector extension
register_vector(conn)

# Verify installation
cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
vec_version = cursor.fetchone()
print(f"pgvector_version={vec_version[0] if vec_version else 'not installed'}")

embedding = model.encode("LIGN'aasdasdasd PLUS INC", prompt_name="document").astype(np.float32).tolist()

cursor.execute("""
SELECT
    id,
    business_name,
    business_name_embedding <=> %s::vector AS distance
FROM vec_documents
ORDER BY business_name_embedding <=> %s::vector
LIMIT 5;
""", (embedding, embedding))

print(cursor.fetchall())
conn.close()