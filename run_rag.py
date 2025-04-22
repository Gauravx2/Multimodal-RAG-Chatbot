from src.models.retriever import Retriever
from src.models.generator import Generator
from src.models.rag import RAG
from src.data_ingestion.vectorizer import Vectorizer

def test_rag():
    # Initialize components
    print("Loading vectorizer...")
    vectorizer = Vectorizer()
    vectorizer.load("data")  # Load pre-created index
    
    print("Initializing retriever...")
    retriever = Retriever(vectorizer)
    
    print("Initializing generator...")
    generator = Generator()
    
    print("Creating RAG system...")
    rag = RAG(retriever, generator)
    
    # Test query
    test_query = "What are the main symptoms of COVID-19?"
    print(f"\nProcessing query: {test_query}")
    
    result = rag.process_query_with_context(test_query)
    
    print("\nGenerated Response:")
    print("-" * 50)
    print(result["answer"])
    print("-" * 50)
    print("\nSources used:")
    for source in result["sources"]:
        print(f"- {source}")
    print(f"\nNumber of chunks used: {result['num_chunks_used']}")

if __name__ == "__main__":
    test_rag() 