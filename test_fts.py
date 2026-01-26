
import sqlite3

db = sqlite3.connect("my_db.db")
search_term = 'business_name: "PLOMBERIE CARL ST-AMOUR INC."'
query = """
    SELECT business_name,
            business_domain, 
            business_niche_description
    FROM fts_documents 
    WHERE fts_documents MATCH ? 
    ORDER BY bm25(fts_documents);
"""

print(db.execute(query, (search_term,)).fetchall())