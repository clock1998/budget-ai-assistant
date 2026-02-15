import os
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer, quantize_embeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from ddgs import DDGS


DEFAULT_BUDGET_CATEGORIES = [
    "Groceries and Supermarkets",
    "Restaurants and Dining",
    "Fast Food and Coffee Shops",
    "Rent and Mortgage Payments",
    "Utilities and Monthly Bills",
    "Phone and Internet Services",
    "Gas Stations and Automotive Fuel",
    "Public Transit and Rideshare",
    "Airfare and Hotel Travel",
    "Clothing and Fashion Accessories",
    "Department Stores and Big Box Retailers",  # Walmart, Target, Costco
    "Online Marketplaces and E-commerce",      # Amazon, eBay, Etsy
    "Electronics and Computer Hardware",       # Apple, Best Buy, Newegg
    "Home Improvement and Hardware Stores",    # Home Depot, Lowe's, IKEA
    "Clothing, Apparel, and Footwear",         # Nike, Zara, Gap
    "Books, Hobby, and Stationery",            # Barnes & Noble, Michaels
    "Health, Beauty, and Drugstores",          # CVS, Walgreens, Sephora
    "Pet Supplies and Veterinary Services",    # Chewy, Petco
    "Sporting Goods and Outdoor Gear",         # REI, Dick's Sporting Goods
    "Discount and Variety Stores",             # Dollar General, Five Below
    "Household Goods and Furniture",
    "Health, Pharmacy, and Medical",
    "Fitness, Gyms, and Sports",
    "Entertainment, Streaming, and Movies",
    "Personal Care, Beauty, and Barber",
    "Insurance and Financial Services",
    "Education and Learning",
    "Government Fees and Taxes",
    "Charity and Donations",
    "Uncategorized or Miscellaneous"
]


class TransactionCategorizer:
    """Categorizes business transactions using search and zero-shot classification."""
    
    HYPOTHESIS_TEMPLATE = "This description of a business domain or activity is in {} budget category."
    
    def __init__(
        self,
        embedding_model_name: str = "jinaai/jina-embeddings-v3",
        translation_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        classifier_model_name: str = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        categories: list[str] | None = None,
        device: str = "cuda"
    ):
        """Initialize the categorizer with required models.
        
        Args:
            embedding_model_name: Name of the sentence-transformer embedding model.
            translation_model_name: Name of the translation model.
            classifier_model_name: Name of the zero-shot classification model.
            categories: List of budget categories for classification. Uses DEFAULT_BUDGET_CATEGORIES if None.
            device: Device to run models on.
        """
        self.device = device
        self.categories = categories if categories is not None else DEFAULT_BUDGET_CATEGORIES
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            device="cuda",
            trust_remote_code=True,
            truncate_dim=128
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
            FROM business, websearch_to_tsquery('simple', %s) query
            WHERE fts_vector @@ query
            ORDER BY rank DESC
            LIMIT %s;
        """, (search_term, limit))
        
        return cursor.fetchall()
    
    def search_semantic(self, search_term: str, limit: int = 3) -> list:
        """Search using semantic vector similarity."""
        _, cursor = self._get_db_connection()
        
        float_embedding = self.embedding_model.encode(search_term, task="retrieval.query")
        binary_embedding = quantize_embeddings(
            float_embedding.reshape(1, -1), precision="binary"
        )
        # Convert to bit string for PostgreSQL bit type
        binary_bits = ''.join(format(byte, '08b') for byte in binary_embedding.tobytes())

        cursor.execute("""
        SELECT
            id,
            business_name,
            business_domain, 
            business_niche_description,
            business_name_embedding <~> %s::bit(128) AS hamming_distance
        FROM business
        WHERE business_name_embedding <~> %s::bit(128) < 17
        ORDER BY business_name_embedding <~> %s::bit(128)
        LIMIT %s;
        """, (binary_bits, binary_bits, binary_bits, limit))
        
        return cursor.fetchall()
    
    def search_business(self, search_term: str, limit: int = 3) -> list:
        """Search for business using FTS first, then fallback to semantic search."""
        results = self.search_fts(search_term, limit)
        
        if not results:
            results = self.search_semantic(search_term, limit)
        
        return results

    def search_ddgs(self, search_term: str, max_results: int = 3) -> str | None:
        """Search for a business using DuckDuckGo and return a short description.
        
        Args:
            search_term: The business name to search for.
            max_results: Maximum number of search results to consider.
            
        Returns:
            A short description of the business, or None if nothing found.
        """
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(f"{search_term} business", max_results=max_results))
            
            if not results:
                return None
            
            # Combine the top result snippets into a concise description
            snippets = [r.get("body", "") for r in results if r.get("body")]
            if not snippets:
                return None
            
            # Use the first snippet as the primary description
            return snippets[0] + " " + " ".join(snippets[1:])  # Combine with others for more context 
        except Exception as e:
            print(f"DDGS search failed for '{search_term}': {e}")
            return None

    def categorize(self, search_term: str) -> dict | None:
        """
        Main method to categorize a transaction.
        
        Args:
            search_term: The business name or transaction description to categorize.
            
        Returns:
            Dictionary with category, confidence, and search results, or None if no match found.
        """
        results = self.search_business(search_term)
        
        if results:
            # Extract business info (handle both FTS and semantic search result formats)
            if len(results[0]) == 4:  # FTS result
                business_name, business_domain = results[0][0], results[0][1]
            else:  # Semantic search result (has id as first column)
                business_name, business_domain = results[0][1], results[0][2]

            distance_info = results[0][-1]  # Could be rank or hamming distance 
            print(distance_info)
            translated_business = self.translate_to_english(f"{business_name}, {business_domain}")
            classification = self.zeroshot_classifier(
                translated_business,
                self.categories,
                hypothesis_template=self.HYPOTHESIS_TEMPLATE,
                multi_label=False
            )
            
            return {
                "category": classification["labels"][0],
                "confidence": classification["scores"][0],
                "all_labels": classification["labels"],
                "all_scores": classification["scores"],
                "matched_business": business_name,
                "business_domain": business_domain,
                "translated_text": translated_business,
                "source": "database"
            }
        
        # Fallback: search the business on the internet via DuckDuckGo
        ddgs_description = self.search_ddgs(search_term)
        
        if not ddgs_description:
            return None
        
        translated_description = self.translate_to_english(ddgs_description)
        print(f"DDGS fallback for '{search_term}': {translated_description}")
        
        classification = self.zeroshot_classifier(
            translated_description,
            self.categories,
            hypothesis_template=self.HYPOTHESIS_TEMPLATE,
            multi_label=False
        )
        
        return {
            "category": classification["labels"][0],
            "confidence": classification["scores"][0],
            "all_labels": classification["labels"],
            "all_scores": classification["scores"],
            "matched_business": search_term,
            "business_domain": ddgs_description,
            "translated_text": translated_description,
            "source": "ddgs"
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
        search_term = 'Esso'
        result = categorizer.categorize(search_term)
        
        if result:
            print(f"Category: {result['category']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Matched Business: {result['matched_business']}")
        else:
            print("No matching business found.")
