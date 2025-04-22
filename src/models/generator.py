import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

class Generator:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize the local LLM for text generation
        Args:
            model_name (str): Name of the model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto"
        )
        
    def generate_response(self, query: str, context_chunks: List[Dict[str, str]], 
                         max_length: int = 512) -> str:
        """
        Generate a response using the LLM based on the query and context
        Args:
            query (str): User's question
            context_chunks (List[Dict[str, str]]): Retrieved context chunks
            max_length (int): Maximum length of generated response
        Returns:
            str: Generated response
        """
        # Format context and query into a prompt
        context_text = "\n".join([chunk["text"] for chunk in context_chunks])
        
        prompt = f"""Below is a biomedical question and relevant context from scientific papers. 
Please provide a detailed, accurate answer based on the context provided.

Context:
{context_text}

Question: {query}

Answer:"""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=3,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part
        answer = response.split("Answer:")[-1].strip()
        return answer

    def generate_response_with_metadata(self, query: str, context_chunks: List[Dict[str, str]], 
                                     max_length: int = 512) -> Dict[str, any]:
        """
        Generate a response with metadata about sources
        Args:
            query (str): User's question
            context_chunks (List[Dict[str, str]]): Retrieved context chunks
            max_length (int): Maximum length of generated response
        Returns:
            Dict: Contains response and source metadata
        """
        answer = self.generate_response(query, context_chunks, max_length)
        
        # Collect source metadata
        sources = []
        for chunk in context_chunks:
            if chunk["metadata"]["source"] not in sources:
                sources.append(chunk["metadata"]["source"])
        
        return {
            "answer": answer,
            "sources": sources,
            "num_chunks_used": len(context_chunks)
        } 