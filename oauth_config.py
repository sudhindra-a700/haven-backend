"""
OAuth 2.0 Configuration Module - Fixed Version
Handles Google and Facebook OAuth configuration with proper error handling
"""

import os
import logging
import secrets
from typing import Optional, Dict, Any
from authlib.integrations.requests_client import OAuth2Session
from pydantic import BaseModel
from enum import Enum

logger = logging.getLogger(__name__)

class OAuthProvider(Enum):
    """OAuth Provider Enum"""
    GOOGLE = "google"
    FACEBOOK = "facebook"

class OAuthUser(BaseModel):
    """OAuth User Model"""
    id: str
    email: str
    name: str
    picture: Optional[str] = None
    provider: str
    provider_id: str

class OAuthConfig:
    """OAuth configuration and client management - Fixed Version"""
    
    def __init__(self):
        # Google OAuth configuration
        self.google_client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self.google_redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
        
        # Facebook OAuth configuration
        self.facebook_client_id = os.getenv("FACEBOOK_CLIENT_ID")
        self.facebook_client_secret = os.getenv("FACEBOOK_CLIENT_SECRET")
        self.facebook_redirect_uri = os.getenv("FACEBOOK_REDIRECT_URI", "http://localhost:8000/auth/facebook/callback")
        
        # OAuth endpoints
        self.google_auth_url = "https://accounts.google.com/o/oauth2/auth"
        self.google_token_url = "https://oauth2.googleapis.com/token"
        self.google_userinfo_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        
        self.facebook_auth_url = "https://www.facebook.com/v18.0/dialog/oauth"
        self.facebook_token_url = "https://graph.facebook.com/v18.0/oauth/access_token"
        self.facebook_userinfo_url = "https://graph.facebook.com/v18.0/me"
        
        # OAuth scopes
        self.google_scopes = ["openid", "email", "profile"]
        self.facebook_scopes = ["email", "public_profile"]
        
        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate OAuth configuration"""
        missing_vars = []
        
        # Check Google configuration
        if not self.google_client_id:
            missing_vars.append("GOOGLE_CLIENT_ID")
        if not self.google_client_secret:
            missing_vars.append("GOOGLE_CLIENT_SECRET")
        if not self.google_redirect_uri:
            missing_vars.append("GOOGLE_REDIRECT_URI")
            
        # Check Facebook configuration
        if not self.facebook_client_id:
            missing_vars.append("FACEBOOK_CLIENT_ID")
        if not self.facebook_client_secret:
            missing_vars.append("FACEBOOK_CLIENT_SECRET")
        if not self.facebook_redirect_uri:
            missing_vars.append("FACEBOOK_REDIRECT_URI")
            
        if missing_vars:
            logger.warning(f"Missing OAuth environment variables: {', '.join(missing_vars)}")
            logger.warning("OAuth functionality will be limited. Please set the missing variables.")

    def create_google_oauth_session(self, state: Optional[str] = None) -> OAuth2Session:
        """Create Google OAuth session"""
        if not state:
            state = secrets.token_urlsafe(32)
            
        return OAuth2Session(
            client_id=self.google_client_id,
            client_secret=self.google_client_secret,
            redirect_uri=self.google_redirect_uri,
            scope=self.google_scopes,
            state=state
        )

    def create_facebook_oauth_session(self, state: Optional[str] = None) -> OAuth2Session:
        """Create Facebook OAuth session"""
        if not state:
            state = secrets.token_urlsafe(32)
            
        return OAuth2Session(
            client_id=self.facebook_client_id,
            client_secret=self.facebook_client_secret,
            redirect_uri=self.facebook_redirect_uri,
            scope=self.facebook_scopes,
            state=state
        )

    def get_google_auth_url(self, state: str) -> str:
        """Generate Google OAuth authorization URL - FIXED"""
        try:
            oauth_session = self.create_google_oauth_session(state)
            
            # Use create_authorization_url method correctly
            auth_url, _ = oauth_session.create_authorization_url(
                self.google_auth_url,
                access_type="offline",
                include_granted_scopes="true"
            )
            
            return auth_url
        
        except Exception as e:
            logger.error(f"Failed to create Google auth URL: {e}")
            raise ValueError(f"Google OAuth URL generation failed: {e}")

    def get_facebook_auth_url(self, state: str) -> str:
        """Generate Facebook OAuth authorization URL - FIXED"""
        try:
            oauth_session = self.create_facebook_oauth_session(state)
            
            # Use create_authorization_url method correctly
            auth_url, _ = oauth_session.create_authorization_url(self.facebook_auth_url)
            
            return auth_url
        
        except Exception as e:
            logger.error(f"Failed to create Facebook auth URL: {e}")
            raise ValueError(f"Facebook OAuth URL generation failed: {e}")

    def exchange_google_code_for_token(self, code: str, state: str) -> Dict[str, Any]:
        """Exchange Google authorization code for access token - FIXED"""
        try:
            oauth_session = self.create_google_oauth_session(state)
            
            token = oauth_session.fetch_token(
                self.google_token_url,
                code=code,
                client_secret=self.google_client_secret
            )
            
            return token
        
        except Exception as e:
            logger.error(f"Failed to exchange Google code for token: {e}")
            raise ValueError(f"Google token exchange failed: {e}")

    def exchange_facebook_code_for_token(self, code: str, state: str) -> Dict[str, Any]:
        """Exchange Facebook authorization code for access token - FIXED"""
        try:
            oauth_session = self.create_facebook_oauth_session(state)
            
            token = oauth_session.fetch_token(
                self.facebook_token_url,
                code=code,
                client_secret=self.facebook_client_secret
            )
            
            return token
        
        except Exception as e:
            logger.error(f"Failed to exchange Facebook code for token: {e}")
            raise ValueError(f"Facebook token exchange failed: {e}")

    @property
    def is_google_configured(self) -> bool:
        """Check if Google OAuth is properly configured"""
        return bool(
            self.google_client_id and 
            self.google_client_secret and 
            self.google_redirect_uri
        )

    @property
    def is_facebook_configured(self) -> bool:
        """Check if Facebook OAuth is properly configured"""
        return bool(
            self.facebook_client_id and 
            self.facebook_client_secret and 
            self.facebook_redirect_uri
        )

    def get_provider_config(self, provider: OAuthProvider) -> Dict[str, Any]:
        """Get configuration for specific provider"""
        if provider == OAuthProvider.GOOGLE:
            return {
                "client_id": self.google_client_id,
                "client_secret": self.google_client_secret,
                "redirect_uri": self.google_redirect_uri,
                "auth_url": self.google_auth_url,
                "token_url": self.google_token_url,
                "userinfo_url": self.google_userinfo_url,
                "scopes": self.google_scopes,
                "configured": self.is_google_configured
            }
        elif provider == OAuthProvider.FACEBOOK:
            return {
                "client_id": self.facebook_client_id,
                "client_secret": self.facebook_client_secret,
                "redirect_uri": self.facebook_redirect_uri,
                "auth_url": self.facebook_auth_url,
                "token_url": self.facebook_token_url,
                "userinfo_url": self.facebook_userinfo_url,
                "scopes": self.facebook_scopes,
                "configured": self.is_facebook_configured
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def validate_redirect_uri(self, provider: OAuthProvider, redirect_uri: str) -> bool:
        """Validate redirect URI for security"""
        allowed_domains = [
            "localhost",
            "127.0.0.1",
            "haven-streamlit-frontend.onrender.com",
            "haven-frontend.vercel.app"
        ]
        
        from urllib.parse import urlparse
        parsed = urlparse(redirect_uri)
        
        # Check if domain is allowed
        domain = parsed.netloc.split(':')[0]  # Remove port if present
        return domain in allowed_domains

    def get_oauth_status(self) -> Dict[str, Any]:
        """Get OAuth configuration status"""
        return {
            "google": {
                "configured": self.is_google_configured,
                "client_id_set": bool(self.google_client_id),
                "client_secret_set": bool(self.google_client_secret),
                "redirect_uri": self.google_redirect_uri
            },
            "facebook": {
                "configured": self.is_facebook_configured,
                "client_id_set": bool(self.facebook_client_id),
                "client_secret_set": bool(self.facebook_client_secret),
                "redirect_uri": self.facebook_redirect_uri
            }
        }

# Global configuration instance
_oauth_config = None

def get_oauth_config() -> OAuthConfig:
    """Get the global OAuth configuration instance"""
    global _oauth_config
    if _oauth_config is None:
        _oauth_config = OAuthConfig()
    return _oauth_config

# Utility functions for OAuth operations
def create_oauth_user_from_google(user_info: Dict[str, Any]) -> OAuthUser:
    """Create OAuthUser from Google user info"""
    return OAuthUser(
        id=user_info["id"],
        email=user_info["email"],
        name=user_info["name"],
        picture=user_info.get("picture"),
        provider="google",
        provider_id=user_info["id"]
    )

def create_oauth_user_from_facebook(user_info: Dict[str, Any]) -> OAuthUser:
    """Create OAuthUser from Facebook user info"""
    picture_url = None
    if "picture" in user_info and "data" in user_info["picture"]:
        picture_url = user_info["picture"]["data"].get("url")
    
    return OAuthUser(
        id=user_info["id"],
        email=user_info.get("email", ""),
        name=user_info["name"],
        picture=picture_url,
        provider="facebook",
        provider_id=user_info["id"]
    )

# OAuth error handling
class OAuthError(Exception):
    """Custom OAuth error"""
    def __init__(self, message: str, provider: str = None, error_code: str = None):
        self.message = message
        self.provider = provider
        self.error_code = error_code
        super().__init__(self.message)

def handle_oauth_error(error: Exception, provider: str) -> OAuthError:
    """Handle and standardize OAuth errors"""
    if "invalid_grant" in str(error):
        return OAuthError(
            "Authorization code has expired or is invalid",
            provider,
            "invalid_grant"
        )
    elif "access_denied" in str(error):
        return OAuthError(
            "User denied access",
            provider,
            "access_denied"
        )
    elif "invalid_client" in str(error):
        return OAuthError(
            "Invalid client credentials",
            provider,
            "invalid_client"
        )
    else:
        return OAuthError(
            f"OAuth error: {str(error)}",
            provider,
            "unknown_error"
        )

# Configuration validation
def validate_oauth_environment() -> Dict[str, Any]:
    """Validate OAuth environment configuration"""
    config = get_oauth_config()
    
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check Google configuration
    if not config.is_google_configured:
        validation_result["errors"].append("Google OAuth not properly configured")
        validation_result["valid"] = False
    
    # Check Facebook configuration
    if not config.is_facebook_configured:
        validation_result["errors"].append("Facebook OAuth not properly configured")
        validation_result["valid"] = False
    
    # Check redirect URIs
    if config.google_redirect_uri and not config.validate_redirect_uri(OAuthProvider.GOOGLE, config.google_redirect_uri):
        validation_result["warnings"].append("Google redirect URI may not be secure")
    
    if config.facebook_redirect_uri and not config.validate_redirect_uri(OAuthProvider.FACEBOOK, config.facebook_redirect_uri):
        validation_result["warnings"].append("Facebook redirect URI may not be secure")
    
    return validation_result

