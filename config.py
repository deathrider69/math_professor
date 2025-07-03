import os
from typing import Dict, Any

class MathAgentConfig:
    """Configuration class for Math Agent system"""
    
    # API Configuration
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "")
    
    # Model Configuration
    DEFAULT_MODEL = "deepseek-chat"
    MAX_TOKENS = 4096
    TEMPERATURE = 0.1
    
    # Knowledge Base Configuration
    KB_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    KB_SIMILARITY_THRESHOLD = 0.7
    KB_MAX_RESULTS = 5
    
    # Web Search Configuration
    SEARCH_TIMEOUT = 10
    MAX_SEARCH_RESULTS = 10
    
    # Feedback Configuration
    FEEDBACK_CONFIDENCE_THRESHOLD = 0.6
    MAX_FEEDBACK_ITERATIONS = 3
    
    # Guardrails Configuration
    INPUT_MAX_LENGTH = 2000
    OUTPUT_MAX_LENGTH = 8000
    
    # Cache Configuration
    SOLUTION_CACHE_SIZE = 1000
    CACHE_TTL_HOURS = 24
    
    # Streamlit Configuration
    STREAMLIT_CONFIG = {
        "theme": {
            "primaryColor": "#1f77b4",
            "backgroundColor": "#ffffff",
            "secondaryBackgroundColor": "#f0f2f6",
            "textColor": "#262730"
        },
        "server": {
            "port": 8501,
            "enableCORS": False,
            "enableXsrfProtection": False
        }
    }
    
    @classmethod
    def get_dspy_config(cls) -> Dict[str, Any]:
        """Get DSPy configuration"""
        return {
            "max_tokens": cls.MAX_TOKENS,
            "temperature": cls.TEMPERATURE,
            "model": cls.DEFAULT_MODEL
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        required_vars = []
        
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        
        if missing_vars:
            print(f"Warning: Missing configuration variables: {missing_vars}")
            return False
        
        return True