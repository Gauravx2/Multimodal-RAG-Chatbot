from typing import Dict, List
from .retriever import Retriever
from .generator import Generator

class RAG:
    def __init__(self, retriever: Retriever, generator: Generator):
        """
        Initialize the RAG system
        Args:
            retriever (Retriever): Document retriever
            generator (Generator): Text generator
        """
        self.retriever = retriever
        self.generator = generator
    
    def process_query(self, query: str, k: int = 3) -> Dict[str, any]:
        """
        Process a query through the RAG pipeline
        Args:
            query (str): User's question
            k (int): Number of chunks to retrieve
        Returns:
            Dict: Contains generated answer and metadata
        """
        # Retrieve relevant chunks
        context_chunks = self.retriever.retrieve(query, k=k)
        
        # Generate response with metadata
        response = self.generator.generate_response_with_metadata(
            query, 
            context_chunks
        )
        
        return {
            "query": query,
            "answer": response["answer"],
            "sources": response["sources"],
            "num_chunks_used": response["num_chunks_used"]
        }
    
    def process_query_with_context(self, query: str, k: int = 3) -> Dict[str, any]:
        """
        Process a query and return both response and context
        Args:
            query (str): User's question
            k (int): Number of chunks to retrieve
        Returns:
            Dict: Contains generated answer, context, and metadata
        """
        # Retrieve relevant chunks with scores
        context_chunks = self.retriever.retrieve_with_scores(query, k=k)
        
        # Generate response
        response = self.generator.generate_response_with_metadata(
            query, 
            context_chunks
        )
        
        return {
            "query": query,
            "answer": response["answer"],
            "context_chunks": context_chunks,
            "sources": response["sources"],
            "num_chunks_used": response["num_chunks_used"]
        } 