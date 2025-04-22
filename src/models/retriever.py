from ..data_ingestion.vectorizer import Vectorizer
from typing import List, Dict

class Retriever:
    def __init__(self, vectorizer: Vectorizer):
        """
        Initialize the retriever with a vectorizer
        Args:
            vectorizer (Vectorizer): Initialized vectorizer with loaded index
        """
        self.vectorizer = vectorizer
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        """
        Retrieve relevant chunks for a query
        Args:
            query (str): User's question
            k (int): Number of chunks to retrieve
        Returns:
            List[Dict[str, str]]: List of relevant chunks with metadata
        """
        return self.vectorizer.search(query, k=k)
    
    def retrieve_with_scores(self, query: str, k: int = 3) -> List[Dict[str, any]]:
        """
        Retrieve relevant chunks with similarity scores
        Args:
            query (str): User's question
            k (int): Number of chunks to retrieve
        Returns:
            List[Dict]: List of chunks with scores
        """
        results = self.vectorizer.search(query, k=k)
        # Add a mock relevance score (you can implement proper scoring if needed)
        for i, result in enumerate(results):
            result['relevance_score'] = 1.0 - (i * 0.1)  # Simple declining score
        return results 