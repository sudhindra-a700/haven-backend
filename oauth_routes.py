"""
FIXED OAuth Routes for HAVEN Crowdfunding Platform
This file contains the corrected OAuth implementation that fixes:
1. 405 Method Not Allowed errors (using GET for callbacks)
2. Correct environment variable names
3. Proper API prefix (/api/v1)
4. Error handling and redirects
"""
from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import RedirectResponse
import httpx
import os
import logging
from urllib.parse import urlencode
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

# Create router with API prefix
router = APIRouter(prefix="/api/v1")

# OAuth Configuration
class OAuthConfig:
    """OAuth configuration using corrected environment variables"""
    # Google OAuth
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
    # Facebook OAuth - Using corrected variable names
    FACEBOOK_APP_ID = os.getenv("FACEBOOK_APP_ID")  # Fixed from FACEBOOK_CLIENT_ID
    FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET")  # Fixed from FACEBOOK_CLIENT_SECRET
    FACEBOOK_REDIRECT_URI = os.getenv("FACEBOOK_REDIRECT_URI")
    # Application URLs
    FRONTEND_URL = os.getenv("FRONTEND_URL")  # Fixed from FRONTEND_BASE_URI
    BACKEND_URL = os.getenv("BACKEND_URL")

# Validate OAuth configuration
def validate_oauth_config():
    """Validate that all required OAuth environment variables are set"""
    missing_vars = []
    if not OAuthConfig.GOOGLE_CLIENT_ID:
        missing_vars.append("GOOGLE_CLIENT_ID")
    if not OAuthConfig.GOOGLE_CLIENT_SECRET:
        missing_vars.append("GOOGLE_CLIENT_SECRET")
    if not OAuthConfig.GOOGLE_REDIRECT_URI:
        missing_vars.append("GOOGLE_REDIRECT_URI")
    if not OAuthConfig.FACEBOOK_APP_ID:
        missing_vars.append("FACEBOOK_APP_ID")
    if not OAuthConfig.FACEBOOK_APP_SECRET:
        missing_vars.append("FACEBOOK_APP_SECRET")
    if not OAuthConfig.FACEBOOK_REDIRECT_URI:
        missing_vars.append("FACEBOOK_REDIRECT_URI")
    if not OAuthConfig.FRONTEND_URL:
        missing_vars.append("FRONTEND_URL")
    if missing_vars:
        logger.error(f"Missing OAuth environment variables: {missing_vars}")
        return False
    return True

@router.get("/auth/google/login")
async def google_login(request: Request, user_type: str = Query("individual")):
    try:
        if not validate_oauth_config():
            raise HTTPException(status_code=500, detail="OAuth configuration incomplete")
        # Build Google OAuth authorization URL
        auth_params = {
            "client_id": OAuthConfig.GOOGLE_CLIENT_ID,
            "redirect_uri": OAuthConfig.GOOGLE_REDIRECT_URI,
            "scope": "openid email profile",
            "response_type": "code",
            "access_type": "offline",
            "state": user_type  # Pass user type in state parameter
        }
        auth_url = f"https://accounts.google.com/o/oauth2/auth?{urlencode(auth_params)}"
        logger.info(f"Google OAuth login initiated for user_type: {user_type}")
        return {"auth_url": auth_url, "provider": "google", "user_type": user_type}
    except Exception as e:
        logger.error(f"Google OAuth login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate Google login: {str(e)}")

@router.get("/auth/google/callback")
async def google_callback(
    request: Request,
    code: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    state: Optional[str] = Query("individual")
):
    try:
        # Handle OAuth errors
        if error:
            logger.error(f"Google OAuth error: {error}")
            return RedirectResponse(
                url=f"{OAuthConfig.FRONTEND_URL}?auth=error&provider=google&message={error}"
            )
        # Validate authorization code
        if not code:
            logger.error("Google OAuth callback: No authorization code provided")
            return RedirectResponse(
                url=f"{OAuthConfig.FRONTEND_URL}?auth=error&provider=google&message=No authorization code"
            )
        # Exchange authorization code for access token
        token_data = {
            "client_id": OAuthConfig.GOOGLE_CLIENT_ID,
            "client_secret": OAuthConfig.GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": OAuthConfig.GOOGLE_REDIRECT_URI
        }
        async with httpx.AsyncClient() as client:
            # Get access token
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data=token_data,
                timeout=30
            )
            if token_response.status_code != 200:
                logger.error(f"Google token exchange failed: {token_response.text}")
                return RedirectResponse(
                    url=f"{OAuthConfig.FRONTEND_URL}?auth=error&provider=google&message=Token exchange failed"
                )
            tokens = token_response.json()
            access_token = tokens.get("access_token")
            if not access_token:
                logger.error("Google OAuth: No access token received")
                return RedirectResponse(
                    url=f"{OAuthConfig.FRONTEND_URL}?auth=error&provider=google&message=No access token"
                )
            # Get user information
            user_response = await client.get(
                f"https://www.googleapis.com/oauth2/v2/userinfo?access_token={access_token}",
                timeout=30
            )
            if user_response.status_code != 200:
                logger.error(f"Google user info failed: {user_response.text}")
                return RedirectResponse(
                    url=f"{OAuthConfig.FRONTEND_URL}?auth=error&provider=google&message=Failed to get user info"
                )
            user_data = user_response.json()
            # TODO: Process user data and create/update user in database
            # For now, we'll just log the successful authentication
            logger.info(f"Google OAuth successful for user: {user_data.get('email')}")
            # Redirect to frontend with success
            return RedirectResponse(
                url=f"{OAuthConfig.FRONTEND_URL}?auth=success&provider=google&user_type={state}"
            )
    except httpx.TimeoutException:
        logger.error("Google OAuth callback: Request timeout")
        return RedirectResponse(
            url=f"{OAuthConfig.FRONTEND_URL}?auth=error&provider=google&message=Request timeout"
        )
    except Exception as e:
        logger.error(f"Google OAuth callback error: {str(e)}")
        return RedirectResponse(
            url=f"{OAuthConfig.FRONTEND_URL}?auth=error&provider=google&message=Authentication failed"
        )

@router.get("/auth/facebook/login")
async def facebook_login(request: Request, user_type: str = Query("individual")):
    try:
        if not validate_oauth_config():
            raise HTTPException(status_code=500, detail="OAuth configuration incomplete")
        # Build Facebook OAuth authorization URL
        auth_params = {
            "client_id": OAuthConfig.FACEBOOK_APP_ID,  # Fixed variable name
            "redirect_uri": OAuthConfig.FACEBOOK_REDIRECT_URI,
            "scope": "email,public_profile",
            "response_type": "code",
            "state": user_type  # Pass user type in state parameter
        }

        auth_url = f"https://www.facebook.com/v18.0/dialog/oauth?{urlencode(auth_params)}"

        logger.info(f"Facebook OAuth login initiated for user_type: {user_type}")

        return {"auth_url": auth_url, "provider": "facebook", "user_type": user_type}

    except Exception as e:
        logger.error(f"Facebook OAuth login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate Facebook login: {str(e)}")



@router.get("/auth/facebook/callback")
async def facebook_callback(
    request: Request,
    code: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    state: Optional[str] = Query("individual")
):
    """
    FIXED: Handle Facebook OAuth callback
    - Uses GET method (OAuth callbacks are always GET)
    - Uses correct Facebook environment variables
    - Proper error handling and user feedback
    """
    try:
        # Handle OAuth errors
        if error:
            logger.error(f"Facebook OAuth error: {error}")
            return RedirectResponse(
                url=f"{OAuthConfig.FRONTEND_URL}?auth=error&provider=facebook&message={error}"
            )

        # Validate authorization code
        if not code:
            logger.error("Facebook OAuth callback: No authorization code provided")
            return RedirectResponse(
                url=f"{OAuthConfig.FRONTEND_URL}?auth=error&provider=facebook&message=No authorization code"
            )

        # Exchange authorization code for access token
        token_params = {
            "client_id": OAuthConfig.FACEBOOK_APP_ID,  # Fixed variable name
            "client_secret": OAuthConfig.FACEBOOK_APP_SECRET,  # Fixed variable name
            "code": code,
            "redirect_uri": OAuthConfig.FACEBOOK_REDIRECT_URI
        }

        async with httpx.AsyncClient() as client:
            # Get access token
            token_response = await client.get(
                "https://graph.facebook.com/v18.0/oauth/access_token",
                params=token_params,
                timeout=30
            )

            if token_response.status_code != 200:
                logger.error(f"Facebook token exchange failed: {token_response.text}")
                return RedirectResponse(
                    url=f"{OAuthConfig.FRONTEND_URL}?auth=error&provider=facebook&message=Token exchange failed"
                )

            tokens = token_response.json()
            access_token = tokens.get("access_token")

            if not access_token:
                logger.error("Facebook OAuth: No access token received")
                return RedirectResponse(
                    url=f"{OAuthConfig.FRONTEND_URL}?auth=error&provider=facebook&message=No access token"
                )

            # Get user information
            user_response = await client.get(
                f"https://graph.facebook.com/me?fields=id,name,email&access_token={access_token}",
                timeout=30
            )

            if user_response.status_code != 200:
                logger.error(f"Facebook user info failed: {user_response.text}")
                return RedirectResponse(
                    url=f"{OAuthConfig.FRONTEND_URL}?auth=error&provider=facebook&message=Failed to get user info"
                )

            user_data = user_response.json()

            # TODO: Process user data and create/update user in database
            # For now, we'll just log the successful authentication
            logger.info(f"Facebook OAuth successful for user: {user_data.get('email', user_data.get('name'))}")

            # Redirect to frontend with success
            return RedirectResponse(
                url=f"{OAuthConfig.FRONTEND_URL}?auth=success&provider=facebook&user_type={state}"
            )

    except httpx.TimeoutException:
        logger.error("Facebook OAuth callback: Request timeout")
        return RedirectResponse(
            url=f"{OAuthConfig.FRONTEND_URL}?auth=error&provider=facebook&message=Request timeout"
        )
    except Exception as e:
        logger.error(f"Facebook OAuth callback error: {str(e)}")
        return RedirectResponse(
            url=f"{OAuthConfig.FRONTEND_URL}?auth=error&provider=facebook&message=Authentication failed"
        )

# ===== UTILITY ROUTES =====

@router.get("/auth/status")
async def auth_status(request: Request):
    """Check OAuth configuration status"""
    config_status = {
        "google_configured": bool(OAuthConfig.GOOGLE_CLIENT_ID and OAuthConfig.GOOGLE_CLIENT_SECRET),
        "facebook_configured": bool(OAuthConfig.FACEBOOK_APP_ID and OAuthConfig.FACEBOOK_APP_SECRET),
        "frontend_url": OAuthConfig.FRONTEND_URL,
        "backend_url": OAuthConfig.BACKEND_URL
    }
    return config_status

@router.get("/auth/test")
async def test_oauth_config(request: Request):
    """Test OAuth configuration (for debugging)"""
    if not validate_oauth_config():
        raise HTTPException(status_code=500, detail="OAuth configuration incomplete")

    return {
        "status": "OAuth configuration valid",
        "google_redirect_uri": OAuthConfig.GOOGLE_REDIRECT_URI,
        "facebook_redirect_uri": OAuthConfig.FACEBOOK_REDIRECT_URI,
        "frontend_url": OAuthConfig.FRONTEND_URL
    }
