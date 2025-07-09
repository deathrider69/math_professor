import os
from typing import Dict, Any

class MathAgentConfig:
    """Configuration class for Math Agent system with LangGraph and DSPy"""
    
    # API Configuration
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    NIM_API_KEY = os.getenv("NIM_API_KEY", "")
    
    # LangGraph Configuration
    LANGGRAPH_CONFIG = {
        "checkpointer": "memory",
        "thread_config": {"configurable": {"thread_id": "default"}},
        "max_iterations": 10,
        "debug": False
    }
    
    # DSPy Configuration  
    DSPY_CONFIG = {
        "model": "gemini/gemini-2.0-flash",  # LiteLLM format for Gemini
        "max_tokens": 4096,
        "temperature": 0.3,
        "top_p": 0.9
    }
    
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
    DSPY_FEEDBACK_ENABLED = True
    
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
        return cls.DSPY_CONFIG
    
    @classmethod
    def get_langgraph_config(cls) -> Dict[str, Any]:
        """Get LangGraph configuration"""
        return cls.LANGGRAPH_CONFIG
    
    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """Validate configuration settings"""
        validation_results = {}
        
        # Check required API keys
        validation_results["gemini_api_key"] = bool(cls.GEMINI_API_KEY)
        validation_results["nim_api_key"] = bool(cls.NIM_API_KEY)
        validation_results["tavily_api_key"] = bool(cls.TAVILY_API_KEY)
        
        # Check optional configurations
        validation_results["dspy_feedback"] = cls.DSPY_FEEDBACK_ENABLED and bool(cls.GEMINI_API_KEY)
        validation_results["langgraph_workflow"] = True  # Always available
        
        return validation_results
    
    @classmethod
    def get_missing_requirements(cls) -> list:
        """Get list of missing requirements"""
        validation = cls.validate_config()
        missing = []
        
        if not validation["gemini_api_key"]:
            missing.append("GEMINI_API_KEY (required for DSPy feedback)")
        if not validation["nim_api_key"]:
            missing.append("NIM_API_KEY (required for DeepSeek solver)")
        if not validation["tavily_api_key"]:
            missing.append("TAVILY_API_KEY (required for web search)")
        
        return missing