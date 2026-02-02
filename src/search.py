import os
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# Connect to PostgreSQL
db = psycopg2.connect(
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    port=int(os.environ.get("POSTGRES_PORT", 5432)),
    database=os.environ.get("POSTGRES_DATABASE", "default"),
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"]
)

cursor = db.cursor()

search_term = 'PLOMBERIE CARL ST-AMOUR INC.'

# PostgreSQL FTS query with ranking
cursor.execute("""
    SELECT 
        business_name,
        business_domain, 
        business_niche_description,
        ts_rank(fts_vector, query) AS rank
    FROM business, plainto_tsquery('french', %s) query
    WHERE fts_vector @@ query
    ORDER BY rank DESC
    LIMIT 3;
""", (search_term,))
results = cursor.fetchall()
if(results == []):
    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2", 
        device="cuda", 
        model_kwargs={"dtype": "bfloat16"}
    )

    # Register pgvector extension
    register_vector(db)
    search_term = 'PLOMBERIE CARL ST-AMOUR INC.'
    embedding = model.encode(search_term, prompt_name="document").astype(np.float32).tolist()

    cursor.execute("""
    SELECT
        id,
        business_name,
        business_domain, 
        business_niche_description,
        business_name_embedding <=> %s::vector AS distance
    FROM business
    ORDER BY business_name_embedding <=> %s::vector
    LIMIT 3;
    """, (embedding, embedding))

    results = cursor.fetchall()
db.close()