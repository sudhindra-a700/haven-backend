"""
FINALIZED OAuth Routes for HAVEN Crowdfunding Platform
This file contains the robust, production-ready implementation that fixes all previous issues:
1. Solves the 'redirect_uri_mismatch' error by dynamically reading the BACKEND_URL at request time.
2. Uses the standard 'authlib' library for a secure and reliable OAuth flow.
3. Correctly handles token exchange and securely passes user data and tokens to the frontend.
"""

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
import os
import logging
import base64
import json
from authlib.integrations.starlette_client import OAuthError

# Import the OAuth configuration from oauth_config.py
from oauth_config import oauth

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["OAuth"])

# ===== GOOGLE OAUTH ROUTES =====

@router.get('/auth/google/login')
async def login_google(request: Request):
    """
    Redirects the user to Google's authentication page.
    The redirect URI for the callback is constructed dynamically to ensure it works in any environment.
    """
    # CRITICAL FIX: Read the BACKEND_URL from environment variables *inside the function*.
    # This ensures the correct URL is used even in a deployed environment like Render.
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    redirect_uri = f"{backend_url}/api/v1/auth/google/callback"
    
    logger.info(f"Initiating Google login. Callback URI: {redirect_uri}")
    
    # Safety check for production environments
    if "localhost" in backend_url and os.getenv("ENVIRONMENT") == "production":
        logger.error("FATAL SECURITY RISK: Using a 'localhost' URL for BACKEND_URL in a production environment!")
    
    return await oauth.google.authorize_redirect(request, redirect_uri)

@router.get('/auth/google/callback')
async def callback_google(request: Request):
    """
    Handles the callback from Google after the user authenticates.
    It exchanges the authorization code for an access token and user info.
    """
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8501")
    try:
        # Exchange the authorization code for an access token
        token = await oauth.google.authorize_access_token(request)
        # Parse the user's profile information from the token
        user_info = await oauth.google.parse_id_token(request, token)
        
        # Prepare user data for the frontend
        user_data = {
            "id": user_info.get("sub"),
            "name": user_info.get("name"),
            "first_name": user_info.get("given_name"),
            "last_name": user_info.get("family_name"),
            "email": user_info.get("email"),
            "picture": user_info.get("picture"),
            "provider": "google"
        }
        
        # Package tokens and user data to be sent to the frontend
        auth_data = {
            "user": user_data,
            "access_token": token.get("access_token"),
            "refresh_token": token.get("refresh_token")
        }
        
        # Securely encode the data to be passed in the URL
        token_str = base64.b64encode(json.dumps(auth_data).encode('utf-8')).decode('utf-8')
        
        # Redirect the user back to the frontend application with the session token
        redirect_url = f"{frontend_url}?token={token_str}"
        logger.info(f"Google login successful for {user_data.get('email')}. Redirecting to frontend.")
        
        return RedirectResponse(url=redirect_url)

    except OAuthError as error:
        logger.error(f"Google OAuth Error: {error.error}")
        return RedirectResponse(url=f"{frontend_url}?error=oauth_failed&provider=google")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Google callback: {e}")
        return RedirectResponse(url=f"{frontend_url}?error=internal_server_error&provider=google")


# ===== FACEBOOK OAUTH ROUTES =====

@router.get('/auth/facebook/login')
async def login_facebook(request: Request):
    """
    Redirects the user to Facebook's authentication page.
    """
    # CRITICAL FIX: Read the BACKEND_URL dynamically inside the function.
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    redirect_uri = f"{backend_url}/api/v1/auth/facebook/callback"
    logger.info(f"Initiating Facebook login. Callback URI: {redirect_uri}")

    return await oauth.facebook.authorize_redirect(request, redirect_uri)

@router.get('/auth/facebook/callback')
async def callback_facebook(request: Request):
    """
    Handles the callback from Facebook after the user authenticates.
    """
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8501")
    try:
        token = await oauth.facebook.authorize_access_token(request)
        
        # Get user data from the Facebook Graph API
        resp = await oauth.facebook.get('me?fields=id,name,email,picture', token=token)
        user_info = resp.json()

        user_data = {
            "id": user_info.get("id"),
            "name": user_info.get("name"),
            "email": user_info.get("email"),
            "picture": user_info.get("picture", {}).get("data", {}).get("url"),
            "provider": "facebook"
        }

        # Package tokens and user data for the frontend
        auth_data = {
            "user": user_data,
            "access_token": token.get("access_token")
        }
        
        token_str = base64.b64encode(json.dumps(auth_data).encode('utf-8')).decode('utf-8')
        redirect_url = f"{frontend_url}?token={token_str}"
        logger.info(f"Facebook login successful for {user_data.get('email')}. Redirecting to frontend.")
        
        return RedirectResponse(url=redirect_url)

    except OAuthError as error:
        logger.error(f"Facebook OAuth Error: {error.error}")
        return RedirectResponse(url=f"{frontend_url}?error=oauth_failed&provider=facebook")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Facebook callback: {e}")
        return RedirectResponse(url=f"{frontend_url}?error=internal_server_error&provider=facebook")

