import sqlite3
import sqlite_vss
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B", 
    device="cuda", 
    model_kwargs={"torch_dtype": "bfloat16"}
)

db = sqlite3.connect("my_db.db")

# Enable loading extensions and load vss
db.enable_load_extension(True)
sqlite_vss.load(db)

# Verify installation
version = db.execute("select vss_version()").fetchone()[0]
print(f"sqlite-vss version: {version}")

cursor = db.cursor()

# Create a standard table for metadata (text, titles, etc.)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        row_id INTEGER PRIMARY KEY,
        business_name TEXT,
        business_domain TEXT,
        business_niche_description TEXT
    )
''')

cursor.execute("""
    CREATE VIRTUAL TABLE vss_documents USING vss0(business_name_embedding(384));
""")

businesses = pd.read_csv('quebec_business_full_export.csv')
embeddings = model.encode(businesses['business_name'].tolist(), prompt_name="document")


for i, row in businesses.iterrows():
    name = row['business_name']
    vector = embeddings[i].tobytes() # Convert numpy array to BLOB
    
    # Insert text into documents
    cursor.execute('''INSERT INTO documents (
                   row_id, 
                   business_name, 
                   business_domain, 
                   business_niche_description) VALUES (?, ?, ?, ?)'''
                   , (i, name, row['business_domain'], row['business_niche_description']))
    row_id = cursor.lastrowid
    
    # Insert vector into index using the SAME row_id
    cursor.execute("""
        INSERT INTO vss_documents(row_id, business_name_embedding) VALUES (?, ?); 
    """, (row_id, vector))
    
db.commit()
db.close()