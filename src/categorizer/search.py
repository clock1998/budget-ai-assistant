import os
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

embedding_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2", 
    device="cuda", 
    model_kwargs={"dtype": "bfloat16"}
)


# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

translation_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507"
)


def translate_local(text):
    messages = [
        {"role": "system", "content": "Translate to English. Only output the translation. If the text is already in English, just repeat it."},
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


# Connect to PostgreSQL
db = psycopg2.connect(
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    port=int(os.environ.get("POSTGRES_PORT", 5432)),
    database=os.environ.get("POSTGRES_DATABASE", "default"),
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"]
)

cursor = db.cursor()

search_term = 'PLOMBERIE CARL ST-AMOUR INC.'

# PostgreSQL FTS query with ranking
cursor.execute("""
    SELECT 
        business_name,
        business_domain, 
        business_niche_description,
        ts_rank(fts_vector, query) AS rank
    FROM business, plainto_tsquery('french', %s) query
    WHERE fts_vector @@ query
    ORDER BY rank DESC
    LIMIT 3;
""", (search_term,))
results = cursor.fetchall()
print(results)
if(results == []):
    # Register pgvector extension
    register_vector(db)
    search_term = 'PLOMBERIE CARL ST-AMOUR INC.'
    embedding = embedding_model.encode(search_term, prompt_name="document").astype(np.float32).tolist()

    cursor.execute("""
    SELECT
        id,
        business_name,
        business_domain, 
        business_niche_description,
        business_name_embedding <=> %s::vector AS distance
    FROM business
    ORDER BY business_name_embedding <=> %s::vector
    LIMIT 3;
    """, (embedding, embedding))

    results = cursor.fetchall()
db.close()

if(results != []):
    translated_business = translate_local(results[0][1] +', '+ results[0][2])
    hypothesis_template = "This description of a business domain or activity is in {} budget category."
    classes_verbalized = [
        'Groceries & Supermarkets',
        'Dining & Restaurants',
        'Fast Food & Coffee Shops',
        'Gas & Fuel',
        'Travel (Airfare, Hotels, Car Rentals)',
        'Transit (Rideshare, Trains, Tolls)',
        'Bills & Utilities (Recurring)',
        'Entertainment (Movies, Events, Hobbies)',
        'Streaming & Digital Subscriptions',
        'Health & Wellness (Gym, Pharmacy, Copays)',
        'Shopping & Department Stores',
        'Home Improvement & Decor',
        'Automotive (Parts & Service)',
        'Personal Care (Salon, Barber, Spa)',
        'Professional Services (Legal, Tax, Business)',
        'Insurance Premiums',
        'Charity & Donations',
        'Miscellaneous & Fees'
    ]
    zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")  # change the model identifier here
    output = zeroshot_classifier(translated_business, classes_verbalized, hypothesis_template=hypothesis_template, multi_label=False)
    print(output["labels"][0])
