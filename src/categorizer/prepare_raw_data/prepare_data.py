import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2", 
    device="cuda"
)

# Configuration
BATCH_SIZE = 2048  # Batch size for embedding generation
CHUNK_SIZE = 50000  # Number of rows to process at a time (RAM-friendly)
INPUT_FILE = 'data.csv'
OUTPUT_FILE = 'data_with_embeddings.csv'

# Count total rows
total_rows = sum(1 for _ in open(INPUT_FILE)) - 1
print(f"Total rows to process: {total_rows}")

# Process CSV in chunks
chunk_iterator = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE)
first_chunk = True

for chunk_num, businesses in enumerate(tqdm(chunk_iterator, total=(total_rows // CHUNK_SIZE) + 1, desc="Processing chunks")):
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

    # Convert embeddings to list format and add as column
    businesses['business_name_embedding'] = [
        emb.astype(np.float32).tolist() for emb in embeddings
    ]

    # Save to CSV (append mode after first chunk)
    if first_chunk:
        businesses.to_csv(OUTPUT_FILE, index=False, mode='w')
        first_chunk = False
    else:
        businesses.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)
    
    # Free memory
    del embeddings
    print(f"Chunk {chunk_num + 1} saved.")

print(f"\n✓ All data processed successfully!")
print(f"Output saved to: {OUTPUT_FILE}")


