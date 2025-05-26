"""
Integration with OpenAI models for recommendation systems.
"""
import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm

try:
    import openai
    from openai import OpenAI
except ImportError:
    print("Warning: openai package not installed. OpenAIClient will not work.")
    openai = None
    OpenAI = None


class OpenAIEmbedding:
    """
    Client for generating embeddings using OpenAI models.
    """
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 20,
        max_retries: int = 5,
        retry_delay: float = 1.0
    ):
        """
        Initialize the OpenAI embedding client.
        
        Args:
            model: OpenAI embedding model to use
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            batch_size: Batch size for API requests
            max_retries: Maximum number of retries for API calls
            retry_delay: Initial delay between retries (with exponential backoff)
        """
        if openai is None:
            raise ImportError("openai package is required for OpenAIEmbedding")
        
        # Set API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key)
        
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Embedding dimensions for different models
        self.dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        
        # Get dimension for the selected model (default to 1536 if unknown)
        self.embedding_dim = self.dimensions.get(model, 1536)
    
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
        if not texts:
            return np.array([])
        
        all_embeddings = []
        
        # Process in batches
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        
        progress_bar = tqdm(batches, disable=not show_progress)
        progress_bar.set_description(f"Encoding with {self.model}")
        
        for batch in progress_bar:
            # Call API with exponential backoff for rate limits
            batch_embeddings = self._call_embedding_api_with_retry(batch)
            all_embeddings.extend(batch_embeddings)
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings)
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / norms
        
        return embeddings_array
    
    def _call_embedding_api_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Call OpenAI embedding API with retry logic.
        
        Args:
            texts: Batch of texts to encode
            
        Returns:
            List of embedding vectors
        """
        retry_count = 0
        delay = self.retry_delay
        
        while retry_count < self.max_retries:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                
                # Extract embeddings from response
                embeddings = [data.embedding for data in response.data]
                return embeddings
                
            except (openai.RateLimitError, openai.APITimeoutError) as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    raise Exception(f"Max retries exceeded: {e}")
                
                print(f"Rate limit or timeout hit. Retrying in {delay:.2f}s...")
                time.sleep(delay)
                # Exponential backoff
                delay *= 2
                
            except Exception as e:
                raise Exception(f"Error calling OpenAI API: {e}")
    
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


class OpenAICompletion:
    """
    Client for text generation using OpenAI models.
    """
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        max_retries: int = 5,
        retry_delay: float = 1.0
    ):
        """
        Initialize the OpenAI completion client.
        
        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            max_retries: Maximum number of retries for API calls
            retry_delay: Initial delay between retries (with exponential backoff)
        """
        if openai is None:
            raise ImportError("openai package is required for OpenAICompletion")
        
        # Set API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key)
        
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def generate(
        self, 
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: int = 300,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: User prompt text
            system_message: Optional system message to guide the response
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Generated text
        """
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        # Call API with exponential backoff for rate limits
        return self._call_completion_api_with_retry(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
    
    def generate_json(
        self, 
        prompt: str,
        system_message: Optional[str] = None,
        response_format: Dict[str, str] = {"type": "json_object"},
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response from a prompt.
        
        Args:
            prompt: User prompt text
            system_message: Optional system message to guide the response
            response_format: Format specification for the response
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Generated response as a Python dictionary
        """
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        # Call API with JSON response format
        response_text = self._call_completion_api_with_retry(
            messages=messages,
            response_format=response_format,
            **kwargs
        )
        
        # Parse JSON response
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response_text}")
    
    def _call_completion_api_with_retry(
        self, 
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Call OpenAI completion API with retry logic.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Generated text
        """
        retry_count = 0
        delay = self.retry_delay
        
        while retry_count < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs
                )
                
                # Extract response text
                return response.choices[0].message.content
                
            except (openai.RateLimitError, openai.APITimeoutError) as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    raise Exception(f"Max retries exceeded: {e}")
                
                print(f"Rate limit or timeout hit. Retrying in {delay:.2f}s...")
                time.sleep(delay)
                # Exponential backoff
                delay *= 2
                
            except Exception as e:
                raise Exception(f"Error calling OpenAI API: {e}")


class OpenAIRecommender:
    """
    Advanced recommendation system using OpenAI models.
    """
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        completion_model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None
    ):
        """
        Initialize the OpenAI recommender.
        
        Args:
            embedding_model: OpenAI embedding model to use
            completion_model: OpenAI completion model to use
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        # Initialize embedding client
        self.embedder = OpenAIEmbedding(
            model=embedding_model,
            api_key=api_key
        )
        
        # Initialize completion client
        self.generator = OpenAICompletion(
            model=completion_model,
            api_key=api_key
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
    
    def generate_explanation(
        self, 
        user_text: str, 
        item: Dict[str, Any], 
        text_key: str = 'text',
        explanation_format: str = 'paragraph'  # 'paragraph', 'bullet_points', 'json'
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate an explanation for a recommendation.
        
        Args:
            user_text: Text describing the user or their preferences
            item: Recommended item dictionary
            text_key: Key for text content in the item dictionary
            explanation_format: Format for the explanation
            
        Returns:
            Generated explanation as a string or dictionary
        """
        # Create system message for LLM
        system_message = (
            "You are a helpful recommendation system assistant. Your job is to explain why a "
            "particular item was recommended to a user based on their preferences."
        )
        
        # Create prompt for the explanation
        prompt = f"""
User preferences: {user_text}

Recommended item: {item[text_key]}

Please explain why this item matches the user's preferences.
"""
        
        if explanation_format == 'bullet_points':
            prompt += "\nFormat your explanation as bullet points, focusing on the top 3-5 reasons."
            
            # Call completion API
            explanation = self.generator.generate(
                prompt=prompt,
                system_message=system_message,
                temperature=0.7,
                max_tokens=300
            )
            
            return explanation
            
        elif explanation_format == 'json':
            # Add specific JSON formatting instructions
            system_message += (
                " Return your explanation as a structured JSON with the following fields: "
                "'match_score' (1-100), 'main_reason', and 'specific_matches' (an array of specific "
                "elements that match between the user and item)."
            )
            
            # Call JSON-specific completion API
            explanation = self.generator.generate_json(
                prompt=prompt,
                system_message=system_message,
                temperature=0.7,
                max_tokens=500
            )
            
            return explanation
            
        else:  # Default paragraph format
            prompt += "\nProvide a concise, natural-sounding explanation in one paragraph."
            
            # Call completion API
            explanation = self.generator.generate(
                prompt=prompt,
                system_message=system_message,
                temperature=0.7,
                max_tokens=200
            )
            
            return explanation
    
    def batch_generate_explanations(
        self,
        user_text: str,
        items: List[Dict[str, Any]],
        text_key: str = 'text',
        explanation_format: str = 'paragraph',
        max_items: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple recommended items.
        
        Args:
            user_text: Text describing the user or their preferences
            items: List of recommended item dictionaries
            text_key: Key for text content in the item dictionary
            explanation_format: Format for the explanations
            max_items: Maximum number of items to explain
            
        Returns:
            List of item dictionaries with added explanations
        """
        # Limit number of items to process
        items_to_process = items[:max_items]
        
        results = []
        for item in tqdm(items_to_process, desc="Generating explanations"):
            item_copy = item.copy()
            explanation = self.generate_explanation(
                user_text=user_text,
                item=item,
                text_key=text_key,
                explanation_format=explanation_format
            )
            item_copy['explanation'] = explanation
            results.append(item_copy)
            
            # Slight delay to avoid rate limits
            time.sleep(0.5)
        
        return results