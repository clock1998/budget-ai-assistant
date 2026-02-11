import os
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings

model = SentenceTransformer(
    "jinaai/jina-embeddings-v3",
    device="cuda",
    trust_remote_code=True,
    truncate_dim=128
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
search_term = 'Marché C&T'

# Generate float embedding, then binary quantize it
float_embedding = model.encode(search_term, task="retrieval.query")
binary_embedding = quantize_embeddings(
    float_embedding.reshape(1, -1), precision="binary"
)
# Convert to bit string for PostgreSQL bit type
binary_bits = ''.join(format(byte, '08b') for byte in binary_embedding.tobytes())

cursor.execute("""
SELECT
    id,
    business_name,
    business_name_embedding <~> %s::bit(128) AS hamming_distance
FROM business
ORDER BY business_name_embedding <~> %s::bit(128)
LIMIT 5;
""", (binary_bits, binary_bits))

print(cursor.fetchall())
db.close()