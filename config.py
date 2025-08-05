"""
Configuration Management for HAVEN Crowdfunding Platform
Handles all environment variables and application settings
Updated to match existing environment configuration
"""

import os
import logging
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import json
import base64

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Core Application Settings
    environment: str = Field(default="development", env="ENVIRONMENT")
    secret_key: str = Field(env="SECRET_KEY")
    jwt_secret_key: str = Field(env="JWT_SECRET_KEY")
    session_secret_key: str = Field(env="SESSION_SECRET_KEY")
    
    # Service URLs
    backend_url: str = Field(default="http://haven-fastapi-backend:10000", env="BACKEND_URL")
    frontend_base_uri: str = Field(default="http://haven-streamlit-frontend:10000", env="FRONTEND_BASE_URI")
    
    # Database Configuration
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    database_host: str = Field(default="localhost", env="DATABASE_HOST")
    database_port: int = Field(default=5432, env="DATABASE_PORT")
    database_name: str = Field(default="haven_db", env="DATABASE_NAME")
    database_user: str = Field(default="postgres", env="DATABASE_USER")
    database_password: str = Field(default="", env="DATABASE_PASSWORD")
    
    # OAuth Configuration - Google
    google_client_id: str = Field(env="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field(env="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str = Field(env="GOOGLE_REDIRECT_URI")
    
    # OAuth Configuration - Facebook
    facebook_client_id: str = Field(env="FACEBOOK_CLIENT_ID")
    facebook_client_secret: str = Field(env="FACEBOOK_CLIENT_SECRET")
    facebook_redirect_uri: str = Field(env="FACEBOOK_REDIRECT_URI")
    
    # Payment Integration - Instamojo
    instamojo_api_key: str = Field(env="INSTAMOJO_API_KEY")
    instamojo_auth_token: str = Field(env="INSTAMOJO_AUTH_TOKEN")
    instamojo_sandbox: bool = Field(default=True, env="INSTAMOJO_SANDBOX")
    instamojo_private_salt: Optional[str] = Field(default=None, env="INSTAMOJO_PRIVATE_SALT")
    
    # Email Service - Brevo
    brevo_api_key: str = Field(env="BREVO_API_KEY")
    brevo_sender_name: str = Field(default="HAVEN Crowdfunding", env="BREVO_SENDER_NAME")
    brevo_sender_email: str = Field(default="noreply@haven.org", env="BREVO_SENDER_EMAIL")
    brevo_mailing_list_id: int = Field(default=1, env="BREVO_MAILING_LIST_ID")
    
    # Search Service - Algolia
    algolia_app_id: str = Field(env="ALGOLIA_APP_ID")
    algolia_api_key: str = Field(env="ALGOLIA_API_KEY")
    algolia_admin_api_key: Optional[str] = Field(default=None, env="ALGOLIA_ADMIN_API_KEY")
    
    # Firebase Configuration
    firebase_project_id: str = Field(env="FIREBASE_PROJECT_ID")
    firebase_client_email: str = Field(env="FIREBASE_CLIENT_EMAIL")
    firebase_private_key: str = Field(env="FIREBASE_PRIVATE_KEY")
    firebase_service_account_key_json_base64: str = Field(env="FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64")
    
    # Feature Flags
    features_oauth_enabled: bool = Field(default=True, env="FEATURES_OAUTH_ENABLED")
    features_translation_enabled: bool = Field(default=True, env="FEATURES_TRANSLATION_ENABLED")
    features_simplification_enabled: bool = Field(default=True, env="FEATURES_SIMPLIFICATION_ENABLED")
    features_fraud_detection_enabled: bool = Field(default=True, env="FEATURES_FRAUD_DETECTION_ENABLED")
    features_analytics_enabled: bool = Field(default=True, env="FEATURES_ANALYTICS_ENABLED")
    features_file_upload_enabled: bool = Field(default=True, env="FEATURES_FILE_UPLOAD_ENABLED")
    features_batch_operations_enabled: bool = Field(default=True, env="FEATURES_BATCH_OPERATIONS_ENABLED")
    
    # Translation Service Configuration
    translation_enabled: bool = Field(default=True, env="TRANSLATION_ENABLED")
    translation_default_language: str = Field(default="en", env="TRANSLATION_DEFAULT_LANGUAGE")
    translation_max_text_length: int = Field(default=5000, env="TRANSLATION_MAX_TEXT_LENGTH")
    translation_timeout: int = Field(default=30, env="TRANSLATION_TIMEOUT")
    translation_batch_size: int = Field(default=8, env="TRANSLATION_BATCH_SIZE")
    translation_cache_ttl: int = Field(default=3600, env="TRANSLATION_CACHE_TTL")
    indictrans2_model_path: str = Field(default="facebook/m2m100_418M", env="INDICTRANS2_MODEL_PATH")
    
    # Simplification Service Configuration
    simplification_enabled: bool = Field(default=True, env="SIMPLIFICATION_ENABLED")
    simplification_default_level: str = Field(default="simple", env="SIMPLIFICATION_DEFAULT_LEVEL")
    simplification_max_terms: int = Field(default=1000, env="SIMPLIFICATION_MAX_TERMS")
    simplification_cache_ttl: int = Field(default=7200, env="SIMPLIFICATION_CACHE_TTL")
    simplification_model_path: str = Field(default="t5-base", env="SIMPLIFICATION_MODEL_PATH")
    
    # Performance Configuration
    performance_api_timeout: int = Field(default=30, env="PERFORMANCE_API_TIMEOUT")
    performance_cache_ttl: int = Field(default=3600, env="PERFORMANCE_CACHE_TTL")
    performance_max_requests_per_minute: int = Field(default=60, env="PERFORMANCE_MAX_REQUESTS_PER_MINUTE")
    batch_processing: bool = Field(default=True, env="BATCH_PROCESSING")
    
    # Machine Learning Configuration
    torch_device: str = Field(default="cpu", env="TORCH_DEVICE")
    use_quantization: bool = Field(default=True, env="USE_QUANTIZATION")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Redis Configuration
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # File Upload Configuration
    upload_folder: str = Field(default="uploads", env="UPLOAD_FOLDER")
    max_file_size: int = Field(default=10485760, env="MAX_FILE_SIZE")  # 10MB
    allowed_extensions: str = Field(default="jpg,jpeg,png,gif,pdf,doc,docx", env="ALLOWED_EXTENSIONS")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    
    # CORS Configuration
    cors_origins: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501", 
        env="CORS_ORIGINS"
    )
    
    # JWT Configuration
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    jwt_refresh_token_expire_days: int = Field(default=7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    
    # Webhook Configuration
    webhook_secret: Optional[str] = Field(default=None, env="WEBHOOK_SECRET")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @validator('cors_origins')
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('allowed_extensions')
    def parse_allowed_extensions(cls, v):
        """Parse allowed extensions from comma-separated string"""
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(',')]
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"
    
    @property
    def database_url_constructed(self) -> str:
        """Construct database URL if not provided"""
        if self.database_url:
            return self.database_url
        
        return f"postgresql://{self.database_user}:{self.database_password}@{self.database_host}:{self.database_port}/{self.database_name}"
    
    @property
    def redis_url_constructed(self) -> str:
        """Construct Redis URL if not provided"""
        if self.redis_url:
            return self.redis_url
        
        auth_part = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth_part}{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @property
    def firebase_service_account_dict(self) -> dict:
        """Decode Firebase service account JSON from base64"""
        try:
            decoded_json = base64.b64decode(self.firebase_service_account_key_json_base64).decode('utf-8')
            return json.loads(decoded_json)
        except Exception as e:
            logger.error(f"Error decoding Firebase service account JSON: {e}")
            return {}
    
    def get_feature_flags(self) -> dict:
        """Get all feature flags as a dictionary"""
        return {
            "oauth_enabled": self.features_oauth_enabled,
            "translation_enabled": self.features_translation_enabled,
            "simplification_enabled": self.features_simplification_enabled,
            "fraud_detection_enabled": self.features_fraud_detection_enabled,
            "analytics_enabled": self.features_analytics_enabled,
            "file_upload_enabled": self.features_file_upload_enabled,
            "batch_operations_enabled": self.features_batch_operations_enabled,
        }

# Global settings instance
settings = Settings()

# Configuration manager for backward compatibility
class ConfigManager:
    """Configuration manager for the application"""
    
    def __init__(self):
        self.settings = settings
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return getattr(self.settings, key.lower(), default)
    
    def get_database_url(self) -> str:
        """Get database URL"""
        return self.settings.database_url_constructed
    
    def get_redis_url(self) -> str:
        """Get Redis URL"""
        return self.settings.redis_url_constructed
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins"""
        return self.settings.cors_origins
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled"""
        feature_flags = self.settings.get_feature_flags()
        return feature_flags.get(f"{feature}_enabled", False)
    
    def get_oauth_config(self, provider: str) -> dict:
        """Get OAuth configuration for a provider"""
        if provider.lower() == "google":
            return {
                "client_id": self.settings.google_client_id,
                "client_secret": self.settings.google_client_secret,
                "redirect_uri": self.settings.google_redirect_uri,
            }
        elif provider.lower() == "facebook":
            return {
                "client_id": self.settings.facebook_client_id,
                "client_secret": self.settings.facebook_client_secret,
                "redirect_uri": self.settings.facebook_redirect_uri,
            }
        return {}
    
    def get_instamojo_config(self) -> dict:
        """Get Instamojo configuration"""
        return {
            "api_key": self.settings.instamojo_api_key,
            "auth_token": self.settings.instamojo_auth_token,
            "sandbox": self.settings.instamojo_sandbox,
            "private_salt": self.settings.instamojo_private_salt,
        }
    
    def get_brevo_config(self) -> dict:
        """Get Brevo configuration"""
        return {
            "api_key": self.settings.brevo_api_key,
            "sender_name": self.settings.brevo_sender_name,
            "sender_email": self.settings.brevo_sender_email,
            "mailing_list_id": self.settings.brevo_mailing_list_id,
        }
    
    def get_algolia_config(self) -> dict:
        """Get Algolia configuration"""
        return {
            "app_id": self.settings.algolia_app_id,
            "api_key": self.settings.algolia_api_key,
            "admin_api_key": self.settings.algolia_admin_api_key,
        }

# Global configuration manager instance
config_manager = ConfigManager()

