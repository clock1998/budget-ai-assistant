import os
import psycopg2
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2", 
    device="cuda"
)

# Configuration
BATCH_SIZE = 512  # Batch size for embedding generation
CHUNK_SIZE = 10000  # Number of rows to process at a time
DB_COMMIT_BATCH = 1000  # Commit to DB every N rows
# Connect to PostgreSQL
db = psycopg2.connect(
    host=os.environ["POSTGRES_HOST"],
    port=int(os.environ.get("POSTGRES_PORT", 5432)),
    database=os.environ.get("POSTGRES_DATABASE", "default"),
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"]
)

cursor = db.cursor()
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
# Drop table if exists
cursor.execute("DROP TABLE IF EXISTS business")
# Create table with tsvector column for full-text search
cursor.execute("""
    CREATE TABLE business (
        id SERIAL PRIMARY KEY,
        business_name TEXT,
        business_domain TEXT,
        business_niche_description TEXT,
        business_name_embedding vector(768),
        fts_vector tsvector
    )
""")
# Create GIN index for fast full-text search
cursor.execute("""
    CREATE INDEX business_search_idx ON business USING GIN (fts_vector)
""")
db.commit()

# Count total rows for progress tracking
total_rows = sum(1 for _ in open('data.csv')) - 1  # Subtract header
print(f"Total rows to process: {total_rows}")

# Process CSV in chunks to handle large files
chunk_iterator = pd.read_csv('data.csv', chunksize=CHUNK_SIZE)

total_inserted = 0

for chunk_num, businesses in enumerate(chunk_iterator):
    print(f"\nProcessing chunk {chunk_num + 1}...")
    
    # Handle NaN values
    businesses['business_name'] = businesses['business_name'].fillna('')
    businesses['business_domain'] = businesses['business_domain'].fillna('')
    businesses['business_niche_description'] = businesses['business_niche_description'].fillna('')
    
    # Generate embeddings in batches for better GPU utilization
    business_names = businesses['business_name'].tolist()
    
    print(f"Generating embeddings for {len(business_names)} items...")
    embeddings = model.encode(
        business_names,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Prepare batch insert data
    print("Inserting into database...")
    insert_data = []
    
    for i, row in businesses.iterrows():
        embedding_list = embeddings[i - businesses.index[0]].astype(np.float32).tolist()
        insert_data.append((
            row['business_name'], 
            row['business_domain'], 
            row['business_niche_description'],
            row['business_name'], 
            embedding_list
        ))
    
    # Bulk insert with progress bar
    for i in tqdm(range(0, len(insert_data), DB_COMMIT_BATCH), desc="Inserting batches"):
        batch = insert_data[i:i + DB_COMMIT_BATCH]
        
        # Use executemany for bulk insert
        cursor.executemany('''
            INSERT INTO business (
                business_name, 
                business_domain, 
                business_niche_description,
                fts_vector,
                business_name_embedding
            ) VALUES (
                %s,
                %s,
                %s, 
                setweight(to_tsvector('french', coalesce(%s, '')), 'A'),
                %s
            )''',
            batch
        )
        db.commit()
        total_inserted += len(batch)
    
    print(f"Chunk {chunk_num + 1} complete. Total inserted: {total_inserted}")

print(f"\n✓ All data inserted successfully! Total: {total_inserted} rows")

# Verify data
cursor.execute("SELECT id, business_name, business_domain FROM business LIMIT 5")
print("Sample data:")
for row in cursor.fetchall():
    print(row)

db.close()


