from ast import Dict
import torch
from transformers import pipeline
import pandas as pd
import os

class TransactionClassifier:
    def __init__(self):
        """
        Initialize zero-shot transaction classification
        """
        print("Loading zero-shot classification model...")
        
        # Use a reliable zero-shot model
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Define transaction categories
        self.candidate_labels = [
            "groceries", "dining", "entertainment", "shopping", 
            "transportation", "utilities", "healthcare", "travel",
            "income", "transfer", "fees", "other"
        ]
        
        print("Transaction classifier initialized successfully!")
    
    def classify_transaction(self, description: str) -> Dict:
        """
        Classify a single transaction
        """
        try:
            result = self.classifier(
                description, 
                candidate_labels=self.candidate_labels,
                hypothesis_template="This transaction is about {}."
            )
            
            return {
                "description": description,
                "category": result['labels'][0],  # Top prediction
                "confidence": result['scores'][0],
                "all_scores": dict(zip(result['labels'], result['scores']))
            }
        except Exception as e:
            return {
                "description": description,
                "error": str(e),
                "category": "UNKNOWN",
                "confidence": 0.0
            }