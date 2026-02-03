import os
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class TransactionCategorizer:
    """Categorizes business transactions using search and zero-shot classification."""
    
    BUDGET_CATEGORIES = [
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
    
    HYPOTHESIS_TEMPLATE = "This description of a business domain or activity is in {} budget category."
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        translation_model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        classifier_model_name: str = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        device: str = "cuda"
    ):
        """Initialize the categorizer with required models."""
        self.device = device
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            device=device,
            model_kwargs={"dtype": "bfloat16"}
        )
        
        # Load translation model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
        self.translation_model = AutoModelForCausalLM.from_pretrained(translation_model_name)
        
        # Load zero-shot classifier
        self.zeroshot_classifier = pipeline(
            "zero-shot-classification",
            model=classifier_model_name
        )
        
        # Database connection (lazy initialization)
        self._db = None
        self._cursor = None
    
    def _get_db_connection(self):
        """Get or create database connection."""
        if self._db is None or self._db.closed:
            self._db = psycopg2.connect(
                host=os.environ.get("POSTGRES_HOST", "localhost"),
                port=int(os.environ.get("POSTGRES_PORT", 5432)),
                database=os.environ.get("POSTGRES_DATABASE", "default"),
                user=os.environ["POSTGRES_USER"],
                password=os.environ["POSTGRES_PASSWORD"]
            )
            register_vector(self._db)
            self._cursor = self._db.cursor()
        return self._db, self._cursor
    
    def close(self):
        """Close database connection."""
        if self._db is not None and not self._db.closed:
            self._db.close()
            self._db = None
            self._cursor = None
    
    def translate_to_english(self, text: str) -> str:
        """Translate text to English using the local translation model."""
        messages = [
            {"role": "system", "content": "Translate to English. Only output the translation. If the text is already in English, just repeat it."},
            {"role": "user", "content": text}
        ]
        
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.translation_model.device)
        
        generated_ids = self.translation_model.generate(
            **model_inputs,
            max_new_tokens=50,
            temperature=0.1
        )
        
        response = self.tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
        return response.strip()
    
    def search_fts(self, search_term: str, limit: int = 3) -> list:
        """Search using PostgreSQL Full-Text Search."""
        _, cursor = self._get_db_connection()
        
        cursor.execute("""
            SELECT 
                business_name,
                business_domain, 
                business_niche_description,
                ts_rank(fts_vector, query) AS rank
            FROM business, plainto_tsquery('french', %s) query
            WHERE fts_vector @@ query
            ORDER BY rank DESC
            LIMIT %s;
        """, (search_term, limit))
        
        return cursor.fetchall()
    
    def search_semantic(self, search_term: str, limit: int = 3) -> list:
        """Search using semantic vector similarity."""
        _, cursor = self._get_db_connection()
        
        embedding = self.embedding_model.encode(
            search_term,
            prompt_name="document"
        ).astype(np.float32).tolist()
        
        cursor.execute("""
            SELECT
                id,
                business_name,
                business_domain, 
                business_niche_description,
                business_name_embedding <=> %s::vector AS distance
            FROM business
            ORDER BY business_name_embedding <=> %s::vector
            LIMIT %s;
        """, (embedding, embedding, limit))
        
        return cursor.fetchall()
    
    def search_business(self, search_term: str, limit: int = 3) -> list:
        """Search for business using FTS first, then fallback to semantic search."""
        results = self.search_fts(search_term, limit)
        
        if not results:
            results = self.search_semantic(search_term, limit)
        
        return results
    
    def classify_category(self, text: str) -> dict:
        """Classify text into a budget category using zero-shot classification."""
        output = self.zeroshot_classifier(
            text,
            self.BUDGET_CATEGORIES,
            hypothesis_template=self.HYPOTHESIS_TEMPLATE,
            multi_label=False
        )
        return output
    
    def categorize(self, search_term: str) -> dict | None:
        """
        Main method to categorize a transaction.
        
        Args:
            search_term: The business name or transaction description to categorize.
            
        Returns:
            Dictionary with category, confidence, and search results, or None if no match found.
        """
        results = self.search_business(search_term)
        
        if not results:
            return None
        
        # Extract business info (handle both FTS and semantic search result formats)
        if len(results[0]) == 4:  # FTS result
            business_name, business_domain = results[0][0], results[0][1]
        else:  # Semantic search result (has id as first column)
            business_name, business_domain = results[0][1], results[0][2]
        
        translated_business = self.translate_to_english(f"{business_name}, {business_domain}")
        classification = self.classify_category(translated_business)
        
        return {
            "category": classification["labels"][0],
            "confidence": classification["scores"][0],
            "all_labels": classification["labels"],
            "all_scores": classification["scores"],
            "matched_business": business_name,
            "business_domain": business_domain,
            "translated_text": translated_business
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    # Example usage
    with TransactionCategorizer() as categorizer:
        search_term = 'PLOMBERIE CARL ST-AMOUR INC.'
        result = categorizer.categorize(search_term)
        
        if result:
            print(f"Category: {result['category']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Matched Business: {result['matched_business']}")
        else:
            print("No matching business found.")
