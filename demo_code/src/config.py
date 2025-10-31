"""
Configuration module for managing API keys and LLM provider selection.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for managing LLM providers and API keys."""
    
    def __init__(self, llm_provider: Optional[str] = None, use_local_embeddings: bool = False):
        """
        Initialize configuration.
        
        Args:
            llm_provider: Either "openai" or "gemini". If None, reads from env.
            use_local_embeddings: If True, use local sentence-transformers instead of API embeddings
        """
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "openai")
        self.llm_provider = self.llm_provider.lower()
        self.use_local_embeddings = use_local_embeddings
        
        if self.llm_provider not in ["openai", "gemini"]:
            raise ValueError(f"Invalid LLM provider: {self.llm_provider}. Must be 'openai' or 'gemini'")
        
        # API Keys (only validate if not using local embeddings for everything)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Model names
        self.model_name = os.getenv("MODEL_NAME")
        if not self.model_name:
            self.model_name = "gpt-4o" if self.llm_provider == "openai" else "gemini-pro"
        
        # Validate API keys (only if needed)
        if not use_local_embeddings:
            self._validate_api_keys()
        
        # Random seed for reproducibility
        self.random_seed = 224
        
    def _validate_api_keys(self):
        """Validate that required API keys are present."""
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if self.llm_provider == "gemini" and not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    def get_embedding_model(self):
        """Get the appropriate embedding model based on provider."""
        if self.use_local_embeddings:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            print("🔧 Using local embeddings (sentence-transformers/all-MiniLM-L6-v2)")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        elif self.llm_provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        else:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.google_api_key
            )
    
    def get_llm_model(self, temperature: float = 0, max_tokens: int = 1024):
        """
        Get the appropriate LLM model based on provider.
        
        Args:
            temperature: Temperature for generation (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM model instance
        """
        if self.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                temperature=temperature,
                model=self.model_name,
                openai_api_key=self.openai_api_key,
                max_tokens=max_tokens
            )
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            # Workaround for Gemini max_retries issue
            # The underlying Google client doesn't accept max_retries, but LangChain tries to pass it
            # We need to create a custom wrapper that intercepts the _generate call
            class GeminiWrapper(ChatGoogleGenerativeAI):
                def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                    # Remove max_retries from kwargs before passing to parent
                    kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'max_retries'}
                    return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs_filtered)
            
            return GeminiWrapper(
                model=self.model_name,
                temperature=temperature,
                google_api_key=self.google_api_key,
                max_output_tokens=max_tokens
            )
    
    def __repr__(self):
        return f"Config(provider={self.llm_provider}, model={self.model_name})"
