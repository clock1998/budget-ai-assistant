import os
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
db = psycopg2.connect(
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    port=int(os.environ.get("POSTGRES_PORT", 5432)),
    database=os.environ.get("POSTGRES_DATABASE", "default"),
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"]
)

cursor = db.cursor()

# Register pgvector extension
register_vector(db)
search_term = 'PLOMBERIE CARL ST-AMOUR INC.'
embedding = model.encode(search_term, prompt_name="document").astype(np.float32).tolist()

cursor.execute("""
SELECT
    id,
    business_name,
    business_name_embedding <=> %s::vector AS distance
FROM business
ORDER BY business_name_embedding <=> %s::vector
LIMIT 5;
""", (embedding, embedding))

print(cursor.fetchall())
db.close()