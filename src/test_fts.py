import os
import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    port=int(os.environ.get("POSTGRES_PORT", 5432)),
    database=os.environ.get("POSTGRES_DATABASE", "default"),
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"]
)

cursor = conn.cursor()

search_term = 'PLOMBERIE CARL ST-AMOUR INC.'

# PostgreSQL FTS query with ranking
query = """
    SELECT business_name,
           business_domain, 
           business_niche_description,
           ts_rank(search_vector, query) AS rank
    FROM fts_documents, plainto_tsquery('french', %s) query
    WHERE search_vector @@ query
    ORDER BY rank DESC;
"""

cursor.execute(query, (search_term,))
results = cursor.fetchall()

for row in results:
    print(row)

conn.close()