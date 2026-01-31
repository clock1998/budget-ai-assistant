import sqlite3
import numpy as np
import sqlite_vec
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B", 
    device="cuda", 
    model_kwargs={"dtype": "bfloat16"}
)

db = sqlite3.connect("my_db.db")

# Enable loading extensions and load vec0
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

# Verify installation
vec_version, = db.execute("select vec_version()").fetchone()
print(f"vec_version={vec_version}")

cursor = db.cursor()

cursor.execute("""
    DELETE VIRTUAL TABLE IF EXISTS vec_documents;
""")

cursor.execute("""
    CREATE VIRTUAL TABLE vec_documents USING 
               vec0(
                business_name_embedding float[4096],
                +business_name text, 
                +business_domain text, 
                +business_niche_description text
               );
""")

businesses = pd.read_csv('data.csv')
embeddings = model.encode(businesses['business_name'].tolist(), prompt_name="document")

for i, row in businesses.iterrows():
    vector_bytes = embeddings[i].astype(np.float32).tobytes()
    # Insert text into documents
    cursor.execute('''INSERT INTO vec_documents (
                   rowid, 
                   business_name_embedding,
                   business_name, 
                   business_domain, 
                   business_niche_description) VALUES (?, ?, ?, ?, ?)'''
                   , (i, vector_bytes, row['business_name'], row['business_domain'], row['business_niche_description']))
    
db.commit()
print(cursor.execute("""
select
    rowid,
    business_name,
    business_name_embedding
from vec_documents
limit 2;
""").fetchall())
db.close()