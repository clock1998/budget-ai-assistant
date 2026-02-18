import os
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer, quantize_embeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from ddgs import DDGS
from src.categorizer.categories import DEFAULT_BUDGET_CATEGORIES

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
        WHERE business_name_embedding <~> %s::bit(128) < 14
        ORDER BY business_name_embedding <~> %s::bit(128)
        LIMIT %s;
        """, (binary_bits, binary_bits, binary_bits, limit))
        
        return cursor.fetchall()

    def search_ddgs(self, search_term: str, max_results: int = 5, context: str | None = None) -> str | None:
        """Search for a business using DuckDuckGo and return the most relevant snippet.
        
        Fetches multiple results from DDGS, then uses in-memory semantic search
        (cosine similarity with the embedding model) to pick the snippet most
        relevant to the search term.
        
        Args:
            search_term: The business name to search for.
            max_results: Maximum number of search results to consider.
            context: Optional context string to refine the search query.
            
        Returns:
            The most semantically relevant snippet, or None if nothing found.
        """
        try:
            query = f"what is {search_term} business about?"
            if context:
                query += f" {context}"
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results, backend="google"))
            
            if not results:
                return None
            
            snippets = [r.get("body", "") for r in results if r.get("body")]
            if not snippets:
                return None
            
            if len(snippets) == 1:
                print(f"DDGS result for '{search_term}': {snippets[0]}")
                return snippets[0]
            
            # In-memory semantic search: rank snippets by cosine similarity to the search term
            query_embedding = self.embedding_model.encode(search_term, task="retrieval.query")
            snippet_embeddings = self.embedding_model.encode(snippets, task="retrieval.passage")
            
            # Cosine similarity: dot product of normalized vectors
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            snippet_norms = snippet_embeddings / np.linalg.norm(snippet_embeddings, axis=1, keepdims=True)
            similarities = snippet_norms @ query_norm
            
            best_idx = int(np.argmax(similarities))
            print(f"DDGS best snippet for '{search_term}' (similarity={similarities[best_idx]:.3f}): {snippets[best_idx]}")
            return snippets[best_idx]
        except Exception as e:
            print(f"DDGS search failed for '{search_term}': {e}")
            return None

    def _gather_candidates(self, search_term: str, context: str | None = None) -> tuple[list[str], list[dict]]:
        """Gather candidate texts and metadata from all search sources for a single term.
        
        Args:
            search_term: The business name to search for.
            context: Optional context string passed to search_ddgs.
            
        Returns:
            Tuple of (texts_to_classify, candidates_meta) for this search term.
        """
        texts = []
        metas = []
        
        # 1. FTS search
        fts_results = self.search_fts(search_term)
        if fts_results:
            business_name, business_domain = fts_results[0][0], fts_results[0][1]
            translated = self.translate_to_english(f"{business_name}, {business_domain}")
            texts.append(translated)
            metas.append({
                "matched_business": business_name,
                "business_domain": business_domain,
                "translated_text": translated,
                "source": "fts"
            })
        
        # 2. Semantic search
        # semantic_results = self.search_semantic(search_term)
        # if semantic_results:
        #     business_name, business_domain = semantic_results[0][1], semantic_results[0][2]
        #     translated = self.translate_to_english(f"{business_name}, {business_domain}")
        #     texts.append(translated)
        #     metas.append({
        #         "matched_business": business_name,
        #         "business_domain": business_domain,
        #         "translated_text": translated,
        #         "source": "semantic"
        #     })
        
        # 3. DuckDuckGo search
        ddgs_description = self.search_ddgs(search_term, context=context)
        if ddgs_description:
            translated = self.translate_to_english(ddgs_description)
            texts.append(translated)
            metas.append({
                "matched_business": search_term,
                "business_domain": ddgs_description,
                "translated_text": translated,
                "source": "ddgs"
            })
        
        return texts, metas

    def categorize(self, transactions: list, context: str | None = None) -> list:
        """
        Categorize a list of transactions in place.
        
        Each transaction must have a 'description' attribute (used for search)
        and a 'category' attribute (set by this method).
        
        Runs FTS, semantic search, and DuckDuckGo search for each transaction,
        batches all zero-shot classifications in a single pipeline call,
        and assigns the best category to each transaction.
        
        Args:
            transactions: List of transaction objects with 'description' and 'category' attributes.
            context: Optional context string to help refine DuckDuckGo searches.
            
        Returns:
            The same list of transactions with 'category' populated.
        """
        # Gather all candidates across all transactions
        all_texts = []
        # Track which texts belong to which transaction: list of (start_idx, count, metas)
        term_ranges = []
        
        for txn in transactions:
            texts, metas = self._gather_candidates(txn.description, context=context)
            start = len(all_texts)
            all_texts.extend(texts)
            term_ranges.append((start, len(texts), metas))
        
        # Batch zero-shot classification in a single pipeline call
        if all_texts:
            classifications = self.zeroshot_classifier(
                all_texts,
                self.categories,
                hypothesis_template=self.HYPOTHESIS_TEMPLATE,
                multi_label=False,
                batch_size=len(all_texts)
            )
            if isinstance(classifications, dict):
                classifications = [classifications]
        else:
            classifications = []
        
        # Assign categories to transactions in order
        for txn, (start, count, metas) in zip(transactions, term_ranges):
            if count == 0:
                continue
            
            candidates = []
            for i, meta in enumerate(metas):
                clf = classifications[start + i]
                candidates.append({
                    **meta,
                    "category": clf["labels"][0],
                    "confidence": clf["scores"][0],
                })
            
            best = max(candidates, key=lambda c: c["confidence"])
            txn.category = best["category"]
        
        return transactions
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    from dataclasses import dataclass, field

    @dataclass
    class MockTransaction:
        description: str
        category: str | None = None

    with TransactionCategorizer() as categorizer:
        transactions = [MockTransaction(description='Steam')]
        categorizer.categorize(transactions)
        
        for txn in transactions:
            print(f"'{txn.description}' -> Category: {txn.category}")
