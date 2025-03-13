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
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        
        # Agent configuration
        self.MAX_TOKENS = 8192
        self.TEMPERATURE = 0.7
        self.TOP_K = 40
        self.TOP_P = 0.95
        
        # Validate required environment variables
        self._validate_config()
    
    def _validate_config(self):
        """Validate that all required configuration values are set."""
        if not self.GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY environment variable is not set.")
            print("Please set it in the .env file or using: export GEMINI_API_KEY='your-api-key'")
            print(f"Remember to place your travel documents in: {self.DOCUMENTS_PATH}")
    
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