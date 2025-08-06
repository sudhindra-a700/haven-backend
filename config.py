"""
HAVEN Platform Configuration - Complete Fixed Version
Comprehensive configuration management with all required settings
"""

from pydantic import BaseSettings, Field, validator
from typing import Optional, List
import os
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with all required configurations"""
    
    # Application Settings
    app_name: str = Field(default="HAVEN Crowdfunding Platform", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="production", env="ENVIRONMENT")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Security Settings
    secret_key: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    session_secret_key: str = Field(default="dev-session-secret-change-in-production", env="SESSION_SECRET_KEY")
    
    # JWT Configuration - FIXED: Added all missing JWT settings
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_secret_key: str = Field(default="dev-jwt-secret-change-in-production", env="JWT_SECRET_KEY")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")  # ADDED: Missing setting
    jwt_access_token_expire_minutes: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    jwt_refresh_token_expire_days: int = Field(default=7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./haven.db", env="DATABASE_URL")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    @property
    def database_url_constructed(self) -> str:
        """Construct database URL - FIXED: Proper implementation"""
        return self.database_url
    
    # OAuth Configuration - Google
    google_client_id: str = Field(default="", env="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field(default="", env="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str = Field(default="http://localhost:8000/auth/google/callback", env="GOOGLE_REDIRECT_URI")
    
    # OAuth Configuration - Facebook
    facebook_client_id: str = Field(default="", env="FACEBOOK_CLIENT_ID")
    facebook_client_secret: str = Field(default="", env="FACEBOOK_CLIENT_SECRET")
    facebook_redirect_uri: str = Field(default="http://localhost:8000/auth/facebook/callback", env="FACEBOOK_REDIRECT_URI")
    
    # Email Service Configuration (Brevo)
    brevo_api_key: str = Field(default="", env="BREVO_API_KEY")
    brevo_sender_email: str = Field(default="noreply@haven.com", env="BREVO_SENDER_EMAIL")
    brevo_sender_name: str = Field(default="HAVEN Platform", env="BREVO_SENDER_NAME")
    
    # Payment Service Configuration (Instamojo)
    instamojo_api_key: str = Field(default="", env="INSTAMOJO_API_KEY")
    instamojo_auth_token: str = Field(default="", env="INSTAMOJO_AUTH_TOKEN")
    instamojo_endpoint: str = Field(default="https://test.instamojo.com/api/1.1/", env="INSTAMOJO_ENDPOINT")
    
    # Search Service Configuration (Algolia)
    algolia_app_id: str = Field(default="", env="ALGOLIA_APP_ID")
    algolia_api_key: str = Field(default="", env="ALGOLIA_API_KEY")
    algolia_index_name: str = Field(default="haven_campaigns", env="ALGOLIA_INDEX_NAME")
    
    # Translation Service Configuration
    translation_service_enabled: bool = Field(default=True, env="TRANSLATION_SERVICE_ENABLED")
    translation_cache_ttl: int = Field(default=3600, env="TRANSLATION_CACHE_TTL")  # 1 hour
    
    # Simplification Service Configuration
    simplification_service_enabled: bool = Field(default=True, env="SIMPLIFICATION_SERVICE_ENABLED")
    simplification_cache_ttl: int = Field(default=3600, env="SIMPLIFICATION_CACHE_TTL")  # 1 hour
    
    # Fraud Detection Configuration
    fraud_detection_enabled: bool = Field(default=True, env="FRAUD_DETECTION_ENABLED")
    fraud_detection_threshold: float = Field(default=0.7, env="FRAUD_DETECTION_THRESHOLD")
    fraud_detection_model_path: str = Field(default="./models/fraud_detection", env="FRAUD_DETECTION_MODEL_PATH")
    
    # ML Services Configuration
    ml_services_enabled: bool = Field(default=True, env="ML_SERVICES_ENABLED")
    torch_device: str = Field(default="cpu", env="TORCH_DEVICE")  # cpu or cuda
    
    # Rate Limiting Configuration
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests_per_minute: int = Field(default=60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    
    # CORS Configuration
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    
    # File Upload Configuration
    max_file_size: int = Field(default=10485760, env="MAX_FILE_SIZE")  # 10MB
    allowed_file_types: List[str] = Field(default=["jpg", "jpeg", "png", "pdf"], env="ALLOWED_FILE_TYPES")
    upload_directory: str = Field(default="./uploads", env="UPLOAD_DIRECTORY")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    
    # Cache Configuration
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_ttl: int = Field(default=300, env="CACHE_TTL")  # 5 minutes
    
    # Monitoring Configuration
    monitoring_enabled: bool = Field(default=True, env="MONITORING_ENABLED")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")  # seconds
    
    # Validators
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('allowed_file_types', pre=True)
    def parse_allowed_file_types(cls, v):
        """Parse allowed file types from string or list"""
        if isinstance(v, str):
            return [file_type.strip() for file_type in v.split(',')]
        return v
    
    @validator('database_url')
    def validate_database_url(cls, v):
        """Validate database URL format"""
        if not v:
            raise ValueError("Database URL cannot be empty")
        return v
    
    @validator('jwt_secret_key')
    def validate_jwt_secret_key(cls, v):
        """Validate JWT secret key"""
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        return v
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
_settings: Optional[Settings] = None

@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

# Configuration manager for backward compatibility
class ConfigManager:
    """Configuration manager for backward compatibility"""
    
    def __init__(self):
        self.settings = get_settings()
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return getattr(self.settings, key, default)
    
    def get_database_url(self) -> str:
        """Get database URL"""
        return self.settings.database_url_constructed
    
    def get_jwt_config(self) -> dict:
        """Get JWT configuration"""
        return {
            "secret_key": self.settings.jwt_secret_key,
            "algorithm": self.settings.jwt_algorithm,
            "expiration_hours": self.settings.jwt_expiration_hours,
            "access_token_expire_minutes": self.settings.jwt_access_token_expire_minutes,
            "refresh_token_expire_days": self.settings.jwt_refresh_token_expire_days
        }
    
    def get_oauth_config(self) -> dict:
        """Get OAuth configuration"""
        return {
            "google": {
                "client_id": self.settings.google_client_id,
                "client_secret": self.settings.google_client_secret,
                "redirect_uri": self.settings.google_redirect_uri
            },
            "facebook": {
                "client_id": self.settings.facebook_client_id,
                "client_secret": self.settings.facebook_client_secret,
                "redirect_uri": self.settings.facebook_redirect_uri
            }
        }

# Create global config manager instance
config_manager = ConfigManager()

# Export commonly used functions
__all__ = [
    "Settings",
    "get_settings",
    "ConfigManager",
    "config_manager"
]

