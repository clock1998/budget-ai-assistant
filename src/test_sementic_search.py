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

embedding = model.encode("LIGN'aasdasdasd PLUS INC", prompt_name="document").astype(np.float32).tobytes()

print(cursor.execute("""
select
    rowid,
    business_name,
    distance
from vec_documents
where business_name_embedding match ? and k=5
order by distance;
""", (embedding,)).fetchall())
db.close()