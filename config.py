"""
Configuration module for the Travel Plan Agent.
This module contains configuration parameters and environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the Travel Plan Agent application."""
    
    def __init__(self):
        """Initialize configuration with default values and environment variables."""
        # Base paths
        self.BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = self.BASE_DIR / "data"
        self.DOCUMENTS_PATH = self.DATA_DIR / "documents"
        self.CHROMA_DB_PATH = self.DATA_DIR / "chromadb"
        
        # Create directories if they don't exist
        self.DATA_DIR.mkdir(exist_ok=True)
        self.DOCUMENTS_PATH.mkdir(exist_ok=True)
        self.CHROMA_DB_PATH.mkdir(exist_ok=True)
        
        # Gemini API configuration
        self.GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
        self.GEMINI_MODEL = "gemini-1.5-pro"
        
        # Database configuration
        self.CHUNK_SIZE = 250  # Reduced from 1000 for smaller chunks
        self.CHUNK_OVERLAP = 50  # Reduced from 200 for smaller chunks
        self.USE_MILVUS = False  # Whether to use Milvus/Zilliz Cloud instead of ChromaDB
        
        # Milvus/Zilliz configuration
        self.MILVUS_URI = os.environ.get("MILVUS_URI", "https://in03-56ea31fb714e029.serverless.gcp-us-west1.cloud.zilliz.com")
        self.MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")
        self.MILVUS_COLLECTION = "travel_documents"
        
        # Agent configuration
        self.MAX_TOKENS = 8192
        self.TEMPERATURE = 0.7
        self.TOP_K = 40
        self.TOP_P = 0.95
        
        # API usage optimization
        self.SKIP_METADATA_GENERATION = False  # Set to True to reduce API calls by 20%
        self.OPTIMIZE_KEYWORD_EXTRACTION = False  # Set to True to skip second keyword extraction API call
        self.SKIP_DRAFT_PLAN = False  # Set to True to skip draft plan and go directly to detailed plan
        self.USE_CACHED_KEYWORDS = True  # Use cached keywords for similar queries
        self.OFFLINE_MODE = False  # Set to True to skip all rate limiting (now controlled by --force-mode)
        self.USE_CHAT_MODE = False  # Set to True to use multi-section plan generation
        
        # RAG optimization settings
        self.MAX_CONTEXT_TOKENS = 3000  # Maximum number of tokens to include in the context
        self.ENABLE_CONTEXT_OPTIMIZATION = True  # Enable context optimization to save tokens
        self.MAX_WORDS_PER_DOCUMENT = 100  # Maximum words to include per document in optimized context (reduced for smaller chunks)
        
        # Validate required environment variables
        self._validate_config()
    
    def _validate_config(self):
        """Validate that all required configuration values are set."""
        if not self.GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY environment variable is not set.")
            print("Please set it in the .env file or using: export GEMINI_API_KEY='your-api-key'")
            print(f"Remember to place your travel documents in: {self.DOCUMENTS_PATH}")
            
        # Check Google Generative AI library version
        try:
            import google.generativeai as genai
            version = genai.__version__
            if version < "0.8.4" and self.USE_CHAT_MODE:
                print(f"\nWARNING: Your google-generativeai version ({version}) may not fully support chat mode features.")
                print("For best results with chat mode, please update to version 0.8.4 or higher:")
                print("pip install --upgrade google-generativeai>=0.8.4")
                print("For now, a compatibility mode will be used.")
        except (ImportError, AttributeError):
            pass
    
    def __str__(self):
        """Return a string representation of the configuration."""
        # Exclude API keys from string representation
        return (
            f"Config:\n"
            f"  DOCUMENTS_PATH: {self.DOCUMENTS_PATH}\n"
            f"  CHROMA_DB_PATH: {self.CHROMA_DB_PATH}\n"
            f"  GEMINI_MODEL: {self.GEMINI_MODEL}\n"
            f"  CHUNK_SIZE: {self.CHUNK_SIZE}\n"
            f"  CHUNK_OVERLAP: {self.CHUNK_OVERLAP}\n"
            f"  MAX_TOKENS: {self.MAX_TOKENS}\n"
            f"  TEMPERATURE: {self.TEMPERATURE}\n"
        )