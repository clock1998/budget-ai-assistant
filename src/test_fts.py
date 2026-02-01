import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="default",
    user="secret",
    password="secret"
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