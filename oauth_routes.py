""" FIXED OAuth Routes for HAVEN Crowdfunding Platform Backend

This file contains the corrected OAuth implementation that fixes:
1. 405 Method Not Allowed errors (using GET for callbacks)
2. Correct environment variable names
3. Proper API prefix (/api/v1)
4. Error handling and redirects
5. DIRECT REDIRECT instead of JSON response (MAIN FIX)
"""

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import RedirectResponse
import httpx
import os
import logging
from urllib.parse import urlencode, parse_qs
from typing import Optional, Dict, Any

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
        self.GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
        
        # Facebook OAuth - Using corrected variable names
        self.FACEBOOK_APP_ID = os.getenv("FACEBOOK_APP_ID")  # Fixed from FACEBOOK_CLIENT_ID
        self.FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET")  # Fixed from FACEBOOK_CLIENT_SECRET
        self.FACEBOOK_REDIRECT_URI = os.getenv("FACEBOOK_REDIRECT_URI")
        
        # Application URLs
        self.FRONTEND_URL = os.getenv("FRONTEND_URL")  # Fixed from FRONTEND_BASE_URI
        self.BACKEND_URL = os.getenv("BACKEND_URL")

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

# Initialize OAuth configuration
oauth_config = OAuthConfig()

@router.get("/auth/google/login")
async def google_login(request: Request, user_type: str = Query("individual")):
    """
    FIXED: Initiate Google OAuth login - NOW REDIRECTS DIRECTLY
    - Uses correct API endpoint path (/api/v1)
    - Proper error handling
    - Direct redirect to Google (no JSON response)
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
        
        # MAIN FIX: Return redirect response instead of JSON
        return RedirectResponse(url=auth_url)
        
    except Exception as e:
        logger.error(f"Google OAuth login error: {str(e)}")
        # Redirect to frontend with error
        error_url = f"{oauth_config.FRONTEND_URL}?auth=error&provider=google&message={str(e)}"
        return RedirectResponse(url=error_url)

@router.get("/auth/google/callback")
async def google_callback(
    request: Request,
    code: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    state: Optional[str] = Query("individual")
):
    """
    Handle OAuth errors
    """
    try:
        # Handle OAuth errors
        if error:
            logger.error(f"Google OAuth error: {error}")
            return RedirectResponse(
                url=f"{oauth_config.FRONTEND_URL}?auth=error&provider=google&message={error}"
            )
        
        # Validate authorization code
        if not code:
            logger.error("Google OAuth callback: No authorization code provided")
            return RedirectResponse(
                url=f"{oauth_config.FRONTEND_URL}?auth=error&provider=google&message=No authorization code"
            )
        
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
            return RedirectResponse(
                url=f"{oauth_config.FRONTEND_URL}?auth=error&provider=google&message=Token exchange failed"
            )
        
        tokens = token_response.json()
        access_token = tokens.get("access_token")
        
        if not access_token:
            logger.error("Google OAuth: No access token received")
            return RedirectResponse(
                url=f"{oauth_config.FRONTEND_URL}?auth=error&provider=google&message=No access token"
            )
        
        # Get user information
        user_response = await client.get(
            f"https://www.googleapis.com/oauth2/v2/userinfo?access_token={access_token}",
            timeout=30
        )
        
        if user_response.status_code != 200:
            logger.error(f"Google user info failed: {user_response.text}")
            return RedirectResponse(
                url=f"{oauth_config.FRONTEND_URL}?auth=error&provider=google&message=Failed to get user info"
            )
        
        user_data = user_response.json()
        # TODO: Process user data and create/update user in database
        # For now, we'll just log the successful authentication
        logger.info(f"Google OAuth successful for user: {user_data.get('email')}")
        
        # Redirect to frontend with success
        return RedirectResponse(
            url=f"{oauth_config.FRONTEND_URL}?auth=success&provider=google&user_type={state}"
        )
        
    except httpx.TimeoutException:
        logger.error("Google OAuth callback: Request timeout")
        return RedirectResponse(
            url=f"{oauth_config.FRONTEND_URL}?auth=error&provider=google&message=Request timeout"
        )
    except Exception as e:
        logger.error(f"Google OAuth callback error: {str(e)}")
        return RedirectResponse(
            url=f"{oauth_config.FRONTEND_URL}?auth=error&provider=google&message={str(e)}"
        )

@router.get("/auth/facebook/login")
async def facebook_login(request: Request, user_type: str = Query("individual")):
    """
    FIXED: Initiate Facebook OAuth login - NOW REDIRECTS DIRECTLY
    - Uses correct API endpoint path (/api/v1)
    - Proper error handling
    - Direct redirect to Facebook (no JSON response)
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
        
        # MAIN FIX: Return redirect response instead of JSON
        return RedirectResponse(url=auth_url)
        
    except Exception as e:
        logger.error(f"Facebook OAuth login error: {str(e)}")
        # Redirect to frontend with error
        error_url = f"{oauth_config.FRONTEND_URL}?auth=error&provider=facebook&message={str(e)}"
        return RedirectResponse(url=error_url)

@router.get("/auth/facebook/callback")
async def facebook_callback(
    request: Request,
    code: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    state: Optional[str] = Query("individual")
):
    """
    Handle Facebook OAuth callback
    """
    try:
        # Handle OAuth errors
        if error:
            logger.error(f"Facebook OAuth error: {error}")
            return RedirectResponse(
                url=f"{oauth_config.FRONTEND_URL}?auth=error&provider=facebook&message={error}"
            )
        
        # Validate authorization code
        if not code:
            logger.error("Facebook OAuth callback: No authorization code provided")
            return RedirectResponse(
                url=f"{oauth_config.FRONTEND_URL}?auth=error&provider=facebook&message=No authorization code"
            )
        
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
            return RedirectResponse(
                url=f"{oauth_config.FRONTEND_URL}?auth=error&provider=facebook&message=Token exchange failed"
            )
        
        tokens = token_response.json()
        access_token = tokens.get("access_token")
        
        if not access_token:
            logger.error("Facebook OAuth: No access token received")
            return RedirectResponse(
                url=f"{oauth_config.FRONTEND_URL}?auth=error&provider=facebook&message=No access token"
            )
        
        # Get user information
        user_response = await client.get(
            f"https://graph.facebook.com/me?fields=id,name,email&access_token={access_token}",
            timeout=30
        )
        
        if user_response.status_code != 200:
            logger.error(f"Facebook user info failed: {user_response.text}")
            return RedirectResponse(
                url=f"{oauth_config.FRONTEND_URL}?auth=error&provider=facebook&message=Failed to get user info"
            )
        
        user_data = user_response.json()
        # TODO: Process user data and create/update user in database
        # For now, we'll just log the successful authentication
        logger.info(f"Facebook OAuth successful for user: {user_data.get('email')}")
        
        # Redirect to frontend with success
        return RedirectResponse(
            url=f"{oauth_config.FRONTEND_URL}?auth=success&provider=facebook&user_type={state}"
        )
        
    except httpx.TimeoutException:
        logger.error("Facebook OAuth callback: Request timeout")
        return RedirectResponse(
            url=f"{oauth_config.FRONTEND_URL}?auth=error&provider=facebook&message=Request timeout"
        )
    except Exception as e:
        logger.error(f"Facebook OAuth callback error: {str(e)}")
        return RedirectResponse(
            url=f"{oauth_config.FRONTEND_URL}?auth=error&provider=facebook&message={str(e)}"
        )

# Health check endpoint
@router.get("/auth/status")
async def check_auth_status() -> Dict[str, Any]:
    """Check authentication status from backend"""
    try:
        config_status = oauth_config.validate_oauth_config()
        return {
            "oauth_enabled": config_status,
            "backend_url_set": bool(oauth_config.BACKEND_URL),
            "google_configured": bool(oauth_config.GOOGLE_CLIENT_ID),
            "facebook_configured": bool(oauth_config.FACEBOOK_APP_ID)
        }
    except Exception as e:
        logger.error(f"Auth status check failed: {str(e)}")
        return {"error": f"Authentication status check failed: {str(e)}"}

