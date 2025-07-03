# config.py
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 30
    retry_attempts: int = 3

@dataclass
class SystemConfig:
    """Main system configuration"""
    
    # API Configuration
    gemini_api_key: str = os.getenv('GEMINI_API_KEY')
    
    # Model Configuration
    model_name: str = "gemini-1.5-flash"
    
    # Agent Configuration
    agent_config: AgentConfig = AgentConfig()
    
    # Knowledge Base Configuration
    kb_similarity_threshold: float = 0.8
    kb_max_results: int = 5
    
    # Web Search Configuration
    web_search_enabled: bool = True
    web_search_timeout: int = 10
    
    # Solver Configuration
    solver_fallback_enabled: bool = True
    
    # Human Feedback Configuration
    feedback_confidence_threshold: float = 0.8
    feedback_storage_enabled: bool = True
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "math_agent.log"
    
    # Streamlit Configuration
    streamlit_title: str = "Mathematical AI Professor"
    streamlit_icon: str = "🧮"
    max_query_history: int = 100
    
    # Performance Configuration
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        if self.agent_config.temperature < 0 or self.agent_config.temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")
        
        if self.kb_similarity_threshold < 0 or self.kb_similarity_threshold > 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        
        return True

# Global configuration instance
config = SystemConfig()

# Environment-specific configurations
class DevelopmentConfig(SystemConfig):
    """Development environment configuration"""
    log_level: str = "DEBUG"
    enable_caching: bool = False
    agent_config: AgentConfig = AgentConfig(temperature=0.2)

class ProductionConfig(SystemConfig):
    """Production environment configuration"""
    log_level: str = "WARNING"
    enable_caching: bool = True
    agent_config: AgentConfig = AgentConfig(temperature=0.1)

def get_config(environment: str = "development") -> SystemConfig:
    """Get configuration based on environment"""
    if environment.lower() == "production":
        return ProductionConfig()
    else:
        return DevelopmentConfig()