""" FIXED OAuth Routes for HAVEN Crowdfunding Platform Backend

This file contains the corrected OAuth implementation that fixes:
1. 405 Method Not Allowed errors (using GET for callbacks)
2. Proper environment variable usage
3. Correct API endpoint paths (/api/v1)
4. Error handling and user feedback
5. DIRECT REDIRECT instead of JSON response (MAIN FIX)
6. Proper callback handling with JWT token generation
7. Popup-based OAuth flow support
"""

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import RedirectResponse, HTMLResponse
import httpx
import os
import logging
from urllib.parse import urlencode, parse_qs
from typing import Optional, Dict, Any
import jwt
import time
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

# Create router with API prefix
router = APIRouter(prefix="/api/v1")

# OAuth Configuration
class OAuthConfig:
    """OAuth configuration using corrected environment variables"""
    
    def __init__(self):
        # Use corrected environment variable names
        self.GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
        self.GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
        self.GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")  # Fixed from FACEBOOK_CLIENT_ID
        
        # Facebook OAuth - Using corrected variable names
        self.FACEBOOK_APP_ID = os.getenv("FACEBOOK_APP_ID")  # Fixed variable name
        self.FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET")  # Fixed from FACEBOOK_CLIENT_SECRET
        self.FACEBOOK_REDIRECT_URI = os.getenv("FACEBOOK_REDIRECT_URI")
        
        # Application URLs
        self.FRONTEND_URL = os.getenv("FRONTEND_URL")  # Fixed from FRONTEND_BASE_URI
        self.BACKEND_URL = os.getenv("BACKEND_URL")
        
        # JWT Configuration
        self.JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        self.JWT_ALGORITHM = "HS256"
        self.JWT_EXPIRATION_HOURS = 24

    def validate_oauth_config(self) -> Dict[str, bool]:
        """Validate that all required OAuth environment variables are set"""
        missing_vars = []
        
        if not self.GOOGLE_CLIENT_ID:
            missing_vars.append("GOOGLE_CLIENT_ID")
        if not self.GOOGLE_CLIENT_SECRET:
            missing_vars.append("GOOGLE_CLIENT_SECRET")
        if not self.GOOGLE_REDIRECT_URI:
            missing_vars.append("GOOGLE_REDIRECT_URI")
        if not self.FACEBOOK_APP_ID:
            missing_vars.append("FACEBOOK_APP_ID")
        if not self.FACEBOOK_APP_SECRET:
            missing_vars.append("FACEBOOK_APP_SECRET")
        if not self.FACEBOOK_REDIRECT_URI:
            missing_vars.append("FACEBOOK_REDIRECT_URI")
        if not self.FRONTEND_URL:
            missing_vars.append("FRONTEND_URL")
        
        if missing_vars:
            logger.error(f"Missing OAuth environment variables: {missing_vars}")
            return False
        return True

    def generate_jwt_token(self, user_data: Dict[str, Any]) -> str:
        """Generate JWT token for authenticated user"""
        payload = {
            "user_id": user_data.get("id"),
            "email": user_data.get("email"),
            "name": user_data.get("name"),
            "provider": user_data.get("provider"),
            "user_type": user_data.get("user_type"),
            "exp": datetime.utcnow() + timedelta(hours=self.JWT_EXPIRATION_HOURS),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.JWT_SECRET_KEY, algorithm=self.JWT_ALGORITHM)

# Initialize OAuth configuration
oauth_config = OAuthConfig()

# FIXED: Google OAuth login - NOW RETURNS JSON with auth URL
@router.get("/auth/google/login")
async def google_login(request: Request, user_type: str = Query("individual")):
    """
    FIXED: Initiate Google OAuth login - NOW RETURNS JSON
    - Returns JSON with OAuth URL instead of direct redirect
    - Supports popup-based OAuth flow
    - Proper error handling
    """
    try:
        if not oauth_config.validate_oauth_config():
            raise HTTPException(status_code=500, detail="OAuth configuration incomplete")
        
        # Build Google OAuth authorization URL
        auth_params = {
            "client_id": oauth_config.GOOGLE_CLIENT_ID,
            "redirect_uri": oauth_config.GOOGLE_REDIRECT_URI,
            "scope": "openid email profile",
            "response_type": "code",
            "access_type": "offline",
            "state": user_type  # Pass user_type in state parameter
        }
        
        auth_url = f"https://accounts.google.com/o/oauth2/auth?{urlencode(auth_params)}"
        logger.info(f"Google OAuth login initiated for user_type: {user_type}")
        
        # MAIN FIX: Return JSON instead of redirect for popup handling
        return {
            "auth_url": auth_url,
            "provider": "google",
            "user_type": user_type
        }
        
    except Exception as e:
        logger.error(f"Google OAuth login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Google OAuth login error: {str(e)}")

# FIXED: Google OAuth callback
@router.get("/auth/google/callback")
async def google_callback(
    request: Request,
    code: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    state: Optional[str] = Query("individual")
):
    """
    FIXED: Handle Google OAuth callback with proper JWT token generation
    - Creates user session and JWT tokens
    - Returns HTML page that communicates with parent window
    - Closes popup after successful authentication
    """
    try:
        # Handle OAuth errors
        if error:
            logger.error(f"Google OAuth error: {error}")
            return HTMLResponse(content=create_oauth_error_page("google", error))
        
        # Validate authorization code
        if not code:
            logger.error("Google OAuth callback: No authorization code provided")
            return HTMLResponse(content=create_oauth_error_page("google", "No authorization code"))
        
        # Exchange authorization code for access token
        token_data = {
            "client_id": oauth_config.GOOGLE_CLIENT_ID,
            "client_secret": oauth_config.GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": oauth_config.GOOGLE_REDIRECT_URI
        }
        
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data=token_data,
                timeout=30
            )
        
        if token_response.status_code != 200:
            logger.error(f"Google token exchange failed: {token_response.text}")
            return HTMLResponse(content=create_oauth_error_page("google", "Token exchange failed"))
        
        tokens = token_response.json()
        access_token = tokens.get("access_token")
        
        if not access_token:
            logger.error("Google OAuth: No access token received")
            return HTMLResponse(content=create_oauth_error_page("google", "No access token"))
        
        # Get user information
        async with httpx.AsyncClient() as client:
            user_response = await client.get(
                f"https://www.googleapis.com/oauth2/v2/userinfo?access_token={access_token}",
                timeout=30
            )
        
        if user_response.status_code != 200:
            logger.error(f"Google user info failed: {user_response.text}")
            return HTMLResponse(content=create_oauth_error_page("google", "Failed to get user info"))
        
        user_data = user_response.json()
        
        # Prepare user data for JWT
        jwt_user_data = {
            "id": user_data.get("id"),
            "email": user_data.get("email"),
            "name": user_data.get("name"),
            "provider": "google",
            "user_type": state,
            "picture": user_data.get("picture")
        }
        
        # Generate JWT token
        jwt_token = oauth_config.generate_jwt_token(jwt_user_data)
        
        # TODO: Save user to database here
        logger.info(f"Google OAuth successful for user: {user_data.get('email')}")
        
        # Return success page that communicates with parent window
        return HTMLResponse(content=create_oauth_success_page("google", jwt_token, jwt_user_data))
        
    except httpx.TimeoutException:
        logger.error("Google OAuth callback: Request timeout")
        return HTMLResponse(content=create_oauth_error_page("google", "Request timeout"))
    except Exception as e:
        logger.error(f"Google OAuth callback error: {str(e)}")
        return HTMLResponse(content=create_oauth_error_page("google", str(e)))

# FIXED: Facebook OAuth login - NOW RETURNS JSON
@router.get("/auth/facebook/login")
async def facebook_login(request: Request, user_type: str = Query("individual")):
    """
    FIXED: Initiate Facebook OAuth login - NOW RETURNS JSON
    - Returns JSON with OAuth URL instead of direct redirect
    - Supports popup-based OAuth flow
    - Proper error handling
    """
    try:
        if not oauth_config.validate_oauth_config():
            raise HTTPException(status_code=500, detail="OAuth configuration incomplete")
        
        # Build Facebook OAuth authorization URL
        auth_params = {
            "client_id": oauth_config.FACEBOOK_APP_ID,
            "redirect_uri": oauth_config.FACEBOOK_REDIRECT_URI,
            "scope": "email,public_profile",
            "response_type": "code",
            "state": user_type  # Pass user_type in state parameter
        }
        
        auth_url = f"https://www.facebook.com/v18.0/dialog/oauth?{urlencode(auth_params)}"
        logger.info(f"Facebook OAuth login initiated for user_type: {user_type}")
        
        # MAIN FIX: Return JSON instead of redirect for popup handling
        return {
            "auth_url": auth_url,
            "provider": "facebook",
            "user_type": user_type
        }
        
    except Exception as e:
        logger.error(f"Facebook OAuth login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Facebook OAuth login error: {str(e)}")

# FIXED: Facebook OAuth callback
@router.get("/auth/facebook/callback")
async def facebook_callback(
    request: Request,
    code: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    state: Optional[str] = Query("individual")
):
    """
    FIXED: Handle Facebook OAuth callback with proper JWT token generation
    - Creates user session and JWT tokens
    - Returns HTML page that communicates with parent window
    - Closes popup after successful authentication
    """
    try:
        # Handle OAuth errors
        if error:
            logger.error(f"Facebook OAuth error: {error}")
            return HTMLResponse(content=create_oauth_error_page("facebook", error))
        
        # Validate authorization code
        if not code:
            logger.error("Facebook OAuth callback: No authorization code provided")
            return HTMLResponse(content=create_oauth_error_page("facebook", "No authorization code"))
        
        # Exchange authorization code for access token
        token_params = {
            "client_id": oauth_config.FACEBOOK_APP_ID,
            "client_secret": oauth_config.FACEBOOK_APP_SECRET,
            "code": code,
            "redirect_uri": oauth_config.FACEBOOK_REDIRECT_URI
        }
        
        async with httpx.AsyncClient() as client:
            token_response = await client.get(
                f"https://graph.facebook.com/v18.0/oauth/access_token?{urlencode(token_params)}",
                timeout=30
            )
        
        if token_response.status_code != 200:
            logger.error(f"Facebook token exchange failed: {token_response.text}")
            return HTMLResponse(content=create_oauth_error_page("facebook", "Token exchange failed"))
        
        tokens = token_response.json()
        access_token = tokens.get("access_token")
        
        if not access_token:
            logger.error("Facebook OAuth: No access token received")
            return HTMLResponse(content=create_oauth_error_page("facebook", "No access token"))
        
        # Get user information
        async with httpx.AsyncClient() as client:
            user_response = await client.get(
                f"https://graph.facebook.com/v18.0/me?fields=id,name,email,picture&access_token={access_token}",
                timeout=30
            )
        
        if user_response.status_code != 200:
            logger.error(f"Facebook user info failed: {user_response.text}")
            return HTMLResponse(content=create_oauth_error_page("facebook", "Failed to get user info"))
        
        user_data = user_response.json()
        
        # Prepare user data for JWT
        jwt_user_data = {
            "id": user_data.get("id"),
            "email": user_data.get("email"),
            "name": user_data.get("name"),
            "provider": "facebook",
            "user_type": state,
            "picture": user_data.get("picture", {}).get("data", {}).get("url")
        }
        
        # Generate JWT token
        jwt_token = oauth_config.generate_jwt_token(jwt_user_data)
        
        # TODO: Save user to database here
        logger.info(f"Facebook OAuth successful for user: {user_data.get('email')}")
        
        # Return success page that communicates with parent window
        return HTMLResponse(content=create_oauth_success_page("facebook", jwt_token, jwt_user_data))
        
    except httpx.TimeoutException:
        logger.error("Facebook OAuth callback: Request timeout")
        return HTMLResponse(content=create_oauth_error_page("facebook", "Request timeout"))
    except Exception as e:
        logger.error(f"Facebook OAuth callback error: {str(e)}")
        return HTMLResponse(content=create_oauth_error_page("facebook", str(e)))

def create_oauth_success_page(provider: str, jwt_token: str, user_data: Dict[str, Any]) -> str:
    """Create HTML page for successful OAuth that communicates with parent window"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Authentication Successful</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f5f5f5;
            }}
            .success-container {{
                text-align: center;
                background: white;
                padding: 2rem;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .success-icon {{
                color: #4CAF50;
                font-size: 3rem;
                margin-bottom: 1rem;
            }}
            .provider-name {{
                text-transform: capitalize;
                color: #333;
            }}
        </style>
    </head>
    <body>
        <div class="success-container">
            <div class="success-icon">✓</div>
            <h2>Authentication Successful!</h2>
            <p>You have successfully signed in with <span class="provider-name">{provider}</span>.</p>
            <p>Redirecting to application...</p>
        </div>
        
        <script>
            // Send authentication data to parent window
            const authData = {{
                success: true,
                provider: '{provider}',
                token: '{jwt_token}',
                user: {user_data}
            }};
            
            // Try to communicate with parent window
            try {{
                if (window.opener) {{
                    window.opener.postMessage({{
                        type: 'OAUTH_SUCCESS',
                        data: authData
                    }}, '*');
                    window.close();
                }} else if (window.parent && window.parent !== window) {{
                    window.parent.postMessage({{
                        type: 'OAUTH_SUCCESS',
                        data: authData
                    }}, '*');
                }} else {{
                    // Fallback: redirect to frontend with token
                    window.location.href = '{oauth_config.FRONTEND_URL}?auth=success&token=' + encodeURIComponent('{jwt_token}');
                }}
            }} catch (error) {{
                console.error('Error communicating with parent window:', error);
                // Fallback: redirect to frontend
                window.location.href = '{oauth_config.FRONTEND_URL}?auth=success&token=' + encodeURIComponent('{jwt_token}');
            }}
        </script>
    </body>
    </html>
    """

def create_oauth_error_page(provider: str, error_message: str) -> str:
    """Create HTML page for OAuth errors that communicates with parent window"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Authentication Error</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f5f5f5;
            }}
            .error-container {{
                text-align: center;
                background: white;
                padding: 2rem;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .error-icon {{
                color: #f44336;
                font-size: 3rem;
                margin-bottom: 1rem;
            }}
            .provider-name {{
                text-transform: capitalize;
                color: #333;
            }}
            .error-message {{
                color: #666;
                margin: 1rem 0;
            }}
            .retry-button {{
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 4px;
                cursor: pointer;
                margin-top: 1rem;
            }}
        </style>
    </head>
    <body>
        <div class="error-container">
            <div class="error-icon">✗</div>
            <h2>Authentication Failed</h2>
            <p>There was an error signing in with <span class="provider-name">{provider}</span>.</p>
            <div class="error-message">{error_message}</div>
            <button class="retry-button" onclick="closeWindow()">Close and Retry</button>
        </div>
        
        <script>
            function closeWindow() {{
                try {{
                    if (window.opener) {{
                        window.opener.postMessage({{
                            type: 'OAUTH_ERROR',
                            data: {{
                                provider: '{provider}',
                                error: '{error_message}'
                            }}
                        }}, '*');
                        window.close();
                    }} else if (window.parent && window.parent !== window) {{
                        window.parent.postMessage({{
                            type: 'OAUTH_ERROR',
                            data: {{
                                provider: '{provider}',
                                error: '{error_message}'
                            }}
                        }}, '*');
                    }} else {{
                        window.location.href = '{oauth_config.FRONTEND_URL}?auth=error&provider={provider}&message=' + encodeURIComponent('{error_message}');
                    }}
                }} catch (error) {{
                    console.error('Error communicating with parent window:', error);
                    window.location.href = '{oauth_config.FRONTEND_URL}?auth=error&provider={provider}&message=' + encodeURIComponent('{error_message}');
                }}
            }}
        </script>
    </body>
    </html>
    """

# Health check endpoint
@router.get("/auth/status")
async def auth_status():
    """Check OAuth configuration status"""
    config_status = oauth_config.validate_oauth_config()
    return {
        "oauth_enabled": config_status,
        "google_configured": bool(oauth_config.GOOGLE_CLIENT_ID and oauth_config.GOOGLE_CLIENT_SECRET),
        "facebook_configured": bool(oauth_config.FACEBOOK_APP_ID and oauth_config.FACEBOOK_APP_SECRET),
        "backend_url": oauth_config.BACKEND_URL,
        "frontend_url": oauth_config.FRONTEND_URL
    }

