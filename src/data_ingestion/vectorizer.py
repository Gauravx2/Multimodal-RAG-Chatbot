import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict
import pickle
import os
from tqdm import tqdm

class Vectorizer:
    def __init__(self):
        """
        Initialize the vectorizer with a biomedical-specific transformer model
        """
        # Using PubMedBERT as it's specifically trained on biomedical text
        model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.index = None
        self.chunks = None

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts
        """
        # Tokenize sentences
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Return embeddings as numpy array
        return embeddings.cpu().numpy()

    def create_index(self, chunks: List[Dict[str, str]]):
        """
        Create FAISS index from text chunks
        """
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Process in batches to avoid memory issues
        batch_size = 32
        embeddings_list = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.get_embedding(batch_texts)
            embeddings_list.append(batch_embeddings)
        
        # Concatenate all embeddings
        embeddings = np.vstack(embeddings_list)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        print(f"Created index with {len(chunks)} vectors")

    def save(self, save_dir: str):
        """
        Save the FAISS index and chunks
        """
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, 'faiss_index.bin'))
        with open(os.path.join(save_dir, 'chunks.pkl'), 'wb') as f:
            pickle.dump(self.chunks, f)

    def load(self, save_dir: str):
        """
        Load the FAISS index and chunks
        """
        self.index = faiss.read_index(os.path.join(save_dir, 'faiss_index.bin'))
        with open(os.path.join(save_dir, 'chunks.pkl'), 'rb') as f:
            self.chunks = pickle.load(f)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar chunks
        """
        query_embedding = self.get_embedding([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        return [self.chunks[idx] for idx in indices[0]] 