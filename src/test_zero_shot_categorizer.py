import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

translation_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507"
)

def translate_local(text):
    messages = [
        {"role": "system", "content": "Translate to English. Only output the translation."},
        {"role": "user", "content": text}
    ]
    
    # Qwen3 uses a specific chat template for better results
    text_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True)
    
    model_inputs = tokenizer([text_input], return_tensors="pt").to(translation_model.device)

    generated_ids = translation_model.generate(
        **model_inputs,
        max_new_tokens=50,
        temperature=0.1  # Low temperature for precise translation
    )
    
    # Trim the input from the output
    response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return response.strip()



model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B", 
    device="cuda", 
    model_kwargs={"dtype": "bfloat16"}
)

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="default",
    user="secret",
    password="secret"
)

cursor = conn.cursor()

# Register pgvector extension
register_vector(conn)

# Verify installation
cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
vec_version = cursor.fetchone()
print(f"pgvector_version={vec_version[0] if vec_version else 'not installed'}")

embedding = model.encode("MULTI-SERVICES D'ENTRETIEN CARL ST-AMOUR INC", prompt_name="document").astype(np.float32).tolist()

cursor.execute("""
SELECT
    id,
    business_name,
    business_domain,
    business_niche_description,
    business_name_embedding <=> %s::vector AS distance
FROM vec_documents
ORDER BY business_name_embedding <=> %s::vector
LIMIT 5;
""", (embedding, embedding))

results = cursor.fetchall()
conn.close()



from transformers import pipeline

print(results[0])
translated_business = translate_local(results[0][2] +', '+ results[0][3])

text = translated_business
hypothesis_template = "This description of a business domain or activity is in {} budget category."
classes_verbalized = ['Housing', 'Utilities', 'Transportation', 'Groceries', 'Insurance', 'Healthcare', 'Debt Payments', 'Savings & Investments', 'Personal Care', 'Entertainment', 'Dining Out', 'Household Supplies', 'Education', 'Gifts & Donations', 'Miscellaneous']
zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")  # change the model identifier here
output = zeroshot_classifier(text, classes_verbalized, hypothesis_template=hypothesis_template, multi_label=False)
print(output)

#{'labels': ['travel', 'dancing', 'cooking'],
# 'scores': [0.9938651323318481, 0.0032737774308770895, 0.002861034357920289],
# 'sequence': 'one day I will see the world'}
