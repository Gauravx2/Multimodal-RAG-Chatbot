from pdf_processor import PDFProcessor
from vectorizer import Vectorizer
import os

def main():
    # Initialize PDF processor
    pdf_dir = "pmc_pdfs"
    processor = PDFProcessor(pdf_dir)
    
    # Process all PDFs
    print("Processing PDFs...")
    chunks = processor.process_all_pdfs()
    print(f"Extracted {len(chunks)} chunks from PDFs")
    
    # Create vector index
    print("Creating FAISS index...")
    vectorizer = Vectorizer()
    vectorizer.create_index(chunks)
    
    # Save the index
    print("Saving index...")
    os.makedirs("data", exist_ok=True)
    vectorizer.save("data")
    print("Done!")

if __name__ == "__main__":
    main() 