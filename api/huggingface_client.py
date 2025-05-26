
"""
Integration with Hugging Face models for recommendation systems.
"""
import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

try:
    from transformers import AutoModel, AutoTokenizer
    from transformers import pipeline
except ImportError:
    print("Warning: transformers package not installed. HuggingFaceClient will not work.")


class HuggingFaceEmbedding:
    """
    Client for generating embeddings using Hugging Face models.
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the Hugging Face embedding client.
        
        Args:
            model_name: Name of the model to use
            device: Device to use ('cpu', 'cuda', or None for auto-detection)
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            cache_dir: Directory to cache models
        """
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("transformers package is required for HuggingFaceEmbedding")
        
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"Loading model: {model_name} on {device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.to(device)
        
        # Set to evaluation mode
        self.model.eval()
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling of token embeddings, weighted by attention mask.
        """
        token_embeddings = model_output[0]  # First element of model_output contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, texts: List[str], normalize: bool = True, show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of strings to encode
            normalize: Whether to L2-normalize the embeddings
            show_progress: Whether to show a progress bar
            
        Returns:
            Array of embeddings
        """
        # Process in batches to handle large datasets
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        progress_bar = tqdm(range(0, len(texts), self.batch_size), disable=not show_progress)
        progress_bar.set_description("Encoding texts")
        
        for i in progress_bar:
            # Prepare batch
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                batch_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                
                if normalize:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all batches
        all_embeddings = np.concatenate(embeddings, axis=0)
        return all_embeddings
    
    @staticmethod
    def similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings (n × dim)
            embeddings2: Second set of embeddings (m × dim)
            
        Returns:
            Similarity matrix (n × m)
        """
        # Normalize embeddings if they aren't already
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        if not np.allclose(norm1, 1.0) or not np.allclose(norm2, 1.0):
            embeddings1 = embeddings1 / norm1
            embeddings2 = embeddings2 / norm2
        
        # Calculate similarity
        return np.dot(embeddings1, embeddings2.T)


class HuggingFaceTextGeneration:
    """
    Client for generating text using Hugging Face models.
    """
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        max_length: int = 512,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the Hugging Face text generation client.
        
        Args:
            model_name: Name of the model to use
            device: Device to use ('cpu', 'cuda', or None for auto-detection)
            max_length: Maximum sequence length
            cache_dir: Directory to cache models
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("transformers package is required for HuggingFaceTextGeneration")
        
        # Determine device
        if device is None:
            device = 0 if torch.cuda.is_available() else -1  # 0 for cuda:0, -1 for CPU
        
        # Convert string device to int for pipeline
        if isinstance(device, str):
            if device.startswith('cuda'):
                device = 0 if device == 'cuda' else int(device.split(':')[1])
            else:
                device = -1  # CPU
        
        print(f"Loading text generation model: {model_name}")
        self.generator = pipeline(
            "text-generation", 
            model=model_name, 
            device=device,
            max_length=max_length,
            cache_dir=cache_dir
        )
    
    def generate(
        self, 
        prompt: str, 
        max_length: Optional[int] = None,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs
    ) -> List[str]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Text prompt
            max_length: Maximum length of generated text
            num_return_sequences: Number of different sequences to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional arguments to pass to the generator
            
        Returns:
            List of generated text sequences
        """
        outputs = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )
        
        # Extract generated text
        generated_texts = [output['generated_text'] for output in outputs]
        
        # Remove the prompt from the beginning if desired
        # generated_texts = [text[len(prompt):] for text in generated_texts]
        
        return generated_texts


class HuggingFaceRecommender:
    """
    Recommendation system using Hugging Face models.
    """
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        generation_model: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the Hugging Face recommender.
        
        Args:
            embedding_model: Model for generating embeddings
            generation_model: Optional model for generating text
            device: Device to use ('cpu', 'cuda', or None for auto-detection)
            cache_dir: Directory to cache models
        """
        # Initialize embedding model
        self.embedder = HuggingFaceEmbedding(
            model_name=embedding_model,
            device=device,
            cache_dir=cache_dir
        )
        
        # Optionally initialize text generation model
        self.generator = None
        if generation_model:
            self.generator = HuggingFaceTextGeneration(
                model_name=generation_model,
                device=device,
                cache_dir=cache_dir
            )
    
    def index_items(self, items: List[Dict[str, Any]], text_key: str = 'text') -> np.ndarray:
        """
        Index items for recommendation.
        
        Args:
            items: List of item dictionaries
            text_key: Key for text content in the item dictionaries
            
        Returns:
            Array of item embeddings
        """
        # Extract text content
        texts = [item[text_key] for item in items]
        
        # Generate embeddings
        return self.embedder.encode(texts)
    

    def get_user_embedding(self, user_text: str) -> np.ndarray:
            """
            Generate embedding for a user based on their description or preferences.
            
            Args:
                user_text: Text describing the user or their preferences
                
            Returns:
                User embedding vector
            """
            return self.embedder.encode([user_text])[0]
        
    def recommend(
        self, 
        user_embedding: np.ndarray, 
        item_embeddings: np.ndarray, 
        items: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recommend items for a user.
        
        Args:
            user_embedding: User embedding vector
            item_embeddings: Item embedding matrix
            items: List of item dictionaries
            top_k: Number of recommendations to return
            
        Returns:
            List of recommended item dictionaries with scores
        """
        # Calculate similarity scores
        scores = self.embedder.similarity(user_embedding.reshape(1, -1), item_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Prepare recommendations
        recommendations = []
        for idx in top_indices:
            item = items[idx].copy()
            item['score'] = float(scores[idx])
            recommendations.append(item)
        
        return recommendations
    
    def generate_explanation(self, user_text: str, item: Dict[str, Any], text_key: str = 'text') -> str:
        """
        Generate an explanation for a recommendation.
        
        Args:
            user_text: Text describing the user or their preferences
            item: Recommended item dictionary
            text_key: Key for text content in the item dictionary
            
        Returns:
            Generated explanation
        """
        if not self.generator:
            return "No explanation available (text generation model not loaded)"
        
        # Create a prompt for the explanation
        prompt = f"User preferences: {user_text}\n\nRecommended item: {item[text_key]}\n\nWhy this item matches the user:"
        
        # Generate explanation
        explanation = self.generator.generate(prompt, max_length=200, num_return_sequences=1)[0]
        
        # Extract just the explanation part (after the prompt)
        explanation = explanation.split("Why this item matches the user:")[1].strip()
        
        return explanation