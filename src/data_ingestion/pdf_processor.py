import fitz  # PyMuPDF
import os
from typing import List, Dict, Tuple
import re
from tqdm import tqdm

class PDFProcessor:
    def __init__(self, pdf_dir: str):
        """
        Initialize the PDF processor
        Args:
            pdf_dir (str): Directory containing PDF files
        """
        self.pdf_dir = pdf_dir
        
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and special characters
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep necessary punctuation
        text = re.sub(r'[^\w\s.,;?!-]', '', text)
        return text.strip()
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        Extract text from a PDF file and chunk it into sections
        Args:
            pdf_path (str): Path to PDF file
        Returns:
            List[Dict[str, str]]: List of chunks with metadata
        """
        chunks = []
        try:
            doc = fitz.open(pdf_path)
            
            # Extract document metadata
            metadata = doc.metadata
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Clean the extracted text
                cleaned_text = self.clean_text(text)
                
                # Skip empty pages
                if not cleaned_text:
                    continue
                
                # Create chunks of approximately 1000 characters
                words = cleaned_text.split()
                current_chunk = []
                current_length = 0
                
                for word in words:
                    current_length += len(word) + 1
                    current_chunk.append(word)
                    
                    if current_length >= 1000:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append({
                            'text': chunk_text,
                            'metadata': {
                                'source': os.path.basename(pdf_path),
                                'page': page_num + 1,
                                'title': metadata.get('title', ''),
                                'author': metadata.get('author', ''),
                            }
                        })
                        current_chunk = []
                        current_length = 0
                
                # Add remaining text as a chunk
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'source': os.path.basename(pdf_path),
                            'page': page_num + 1,
                            'title': metadata.get('title', ''),
                            'author': metadata.get('author', ''),
                        }
                    })
            
            doc.close()
            return chunks
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return []
    
    def process_all_pdfs(self) -> List[Dict[str, str]]:
        """
        Process all PDFs in the directory
        Returns:
            List[Dict[str, str]]: List of all chunks from all PDFs
        """
        all_chunks = []
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            chunks = self.extract_text_from_pdf(pdf_path)
            all_chunks.extend(chunks)
            
        return all_chunks 