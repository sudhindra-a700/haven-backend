"""
Simple OAuth Routes for HAVEN Crowdfunding Platform
Simplified OAuth integration for Google and Facebook login
"""

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional
import logging
import hashlib
import json
import os
import secrets
import urllib.parse

logger = logging.getLogger(__name__)

# Create router
oauth_router = APIRouter(tags=["oauth"])

# OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "demo-google-client-id")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "demo-google-secret")
FACEBOOK_APP_ID = os.getenv("FACEBOOK_APP_ID", "demo-facebook-app-id")
FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET", "demo-facebook-secret")

# Base URLs
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8501")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

class OAuthCallbackRequest(BaseModel):
    """OAuth callback request model"""
    code: str
    state: Optional[str] = None
    user_type: Optional[str] = "individual"  # individual or organization

def save_oauth_user(provider: str, user_data: dict, user_type: str = "individual"):
    """Save OAuth user data to simple file storage"""
    try:
        file_path = f"/tmp/haven_oauth_{provider}_users.json"
        
        # Load existing data
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                users = json.load(f)
        else:
            users = []
        
        # Check if user already exists
        for user in users:
            if user.get('email') == user_data.get('email'):
                # Update existing user
                user.update(user_data)
                user['last_login'] = user_data.get('registered_at')
                break
        else:
            # Add new user
            user_data['user_type'] = user_type
            user_data['role'] = user_type
            user_data['provider'] = provider
            user_data['user_id'] = hashlib.md5(user_data['email'].encode()).hexdigest()[:8]
            users.append(user_data)
        
        # Save back to file
        with open(file_path, 'w') as f:
            json.dump(users, f, indent=2)
        
        return user_data.get('user_id')
    except Exception as e:
        logger.error(f"Error saving OAuth user: {e}")
        raise HTTPException(status_code=500, detail="OAuth user save failed")

@oauth_router.get("/google/login")
async def google_login(user_type: str = "individual"):
    """Initiate Google OAuth login"""
    try:
        # Generate state for security
        state = secrets.token_urlsafe(32)
        
        # Store state temporarily (in production, use Redis or database)
        state_file = f"/tmp/oauth_state_{state}.json"
        with open(state_file, 'w') as f:
            json.dump({"user_type": user_type, "provider": "google"}, f)
        
        # Google OAuth URL
        redirect_uri = f"{BACKEND_URL}/api/v1/auth/google/callback"
        
        google_auth_url = (
            "https://accounts.google.com/o/oauth2/auth?"
            f"client_id={GOOGLE_CLIENT_ID}&"
            f"redirect_uri={urllib.parse.quote(redirect_uri)}&"
            "scope=openid email profile&"
            "response_type=code&"
            "access_type=offline&"
            f"state={state}"
        )
        
        logger.info(f"Google OAuth login initiated for user_type: {user_type}")
        
        return {
            "auth_url": google_auth_url,
            "provider": "google",
            "user_type": user_type,
            "state": state
        }
        
    except Exception as e:
        logger.error(f"Google OAuth initiation error: {e}")
        raise HTTPException(status_code=500, detail="OAuth initiation failed")

@oauth_router.get("/facebook/login")
async def facebook_login(user_type: str = "individual"):
    """Initiate Facebook OAuth login"""
    try:
        # Generate state for security
        state = secrets.token_urlsafe(32)
        
        # Store state temporarily
        state_file = f"/tmp/oauth_state_{state}.json"
        with open(state_file, 'w') as f:
            json.dump({"user_type": user_type, "provider": "facebook"}, f)
        
        # Facebook OAuth URL
        redirect_uri = f"{BACKEND_URL}/api/v1/auth/facebook/callback"
        
        facebook_auth_url = (
            "https://www.facebook.com/v18.0/dialog/oauth?"
            f"client_id={FACEBOOK_APP_ID}&"
            f"redirect_uri={urllib.parse.quote(redirect_uri)}&"
            "scope=email,public_profile&"
            "response_type=code&"
            f"state={state}"
        )
        
        logger.info(f"Facebook OAuth login initiated for user_type: {user_type}")
        
        return {
            "auth_url": facebook_auth_url,
            "provider": "facebook", 
            "user_type": user_type,
            "state": state
        }
        
    except Exception as e:
        logger.error(f"Facebook OAuth initiation error: {e}")
        raise HTTPException(status_code=500, detail="OAuth initiation failed")

@oauth_router.get("/google/callback")
async def google_callback(code: str, state: str):
    """Handle Google OAuth callback"""
    try:
        # Verify state
        state_file = f"/tmp/oauth_state_{state}.json"
        if not os.path.exists(state_file):
            raise HTTPException(status_code=400, detail="Invalid state parameter")
        
        with open(state_file, 'r') as f:
            state_data = json.load(f)
        
        # Clean up state file
        os.remove(state_file)
        
        # In a real implementation, you would exchange the code for tokens
        # and fetch user info from Google API
        # For demo purposes, we'll simulate this
        
        # Simulated user data (in production, fetch from Google API)
        user_data = {
            "email": "demo.google.user@gmail.com",
            "full_name": "Google Demo User",
            "provider": "google",
            "provider_id": "google_demo_123",
            "registered_at": "2025-08-07T01:35:00Z",
            "verified": True
        }
        
        user_type = state_data.get("user_type", "individual")
        user_id = save_oauth_user("google", user_data, user_type)
        
        # Redirect to frontend with success
        frontend_redirect = f"{FRONTEND_URL}?oauth_success=true&provider=google&user_type={user_type}&user_id={user_id}"
        
        logger.info(f"Google OAuth callback successful for: {user_data['email']}")
        
        return RedirectResponse(url=frontend_redirect)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Google OAuth callback error: {e}")
        # Redirect to frontend with error
        error_redirect = f"{FRONTEND_URL}?oauth_error=true&provider=google&message={urllib.parse.quote(str(e))}"
        return RedirectResponse(url=error_redirect)

@oauth_router.get("/facebook/callback")
async def facebook_callback(code: str, state: str):
    """Handle Facebook OAuth callback"""
    try:
        # Verify state
        state_file = f"/tmp/oauth_state_{state}.json"
        if not os.path.exists(state_file):
            raise HTTPException(status_code=400, detail="Invalid state parameter")
        
        with open(state_file, 'r') as f:
            state_data = json.load(f)
        
        # Clean up state file
        os.remove(state_file)
        
        # In a real implementation, you would exchange the code for tokens
        # and fetch user info from Facebook API
        # For demo purposes, we'll simulate this
        
        # Simulated user data (in production, fetch from Facebook API)
        user_data = {
            "email": "demo.facebook.user@facebook.com",
            "full_name": "Facebook Demo User",
            "provider": "facebook",
            "provider_id": "facebook_demo_456",
            "registered_at": "2025-08-07T01:35:00Z",
            "verified": True
        }
        
        user_type = state_data.get("user_type", "individual")
        user_id = save_oauth_user("facebook", user_data, user_type)
        
        # Redirect to frontend with success
        frontend_redirect = f"{FRONTEND_URL}?oauth_success=true&provider=facebook&user_type={user_type}&user_id={user_id}"
        
        logger.info(f"Facebook OAuth callback successful for: {user_data['email']}")
        
        return RedirectResponse(url=frontend_redirect)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Facebook OAuth callback error: {e}")
        # Redirect to frontend with error
        error_redirect = f"{FRONTEND_URL}?oauth_error=true&provider=facebook&message={urllib.parse.quote(str(e))}"
        return RedirectResponse(url=error_redirect)

@oauth_router.get("/status/{provider}/{user_id}")
async def get_oauth_user_status(provider: str, user_id: str):
    """Get OAuth user status"""
    try:
        file_path = f"/tmp/haven_oauth_{provider}_users.json"
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                users = json.load(f)
            for user in users:
                if user.get('user_id') == user_id:
                    return {
                        "user_id": user_id,
                        "provider": provider,
                        "email": user.get('email'),
                        "full_name": user.get('full_name'),
                        "user_type": user.get('user_type'),
                        "role": user.get('role'),
                        "verified": user.get('verified', False),
                        "registered_at": user.get('registered_at')
                    }
        
        raise HTTPException(status_code=404, detail="OAuth user not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting OAuth user status: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")

@oauth_router.get("/health")
async def oauth_health():
    """Health check for OAuth service"""
    return {
        "service": "oauth",
        "status": "healthy",
        "version": "1.0.0",
        "providers": ["google", "facebook"],
        "endpoints": [
            "/google/login",
            "/facebook/login",
            "/google/callback",
            "/facebook/callback",
            "/status/{provider}/{user_id}"
        ],
        "configuration": {
            "google_configured": bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_ID != "demo-google-client-id"),
            "facebook_configured": bool(FACEBOOK_APP_ID and FACEBOOK_APP_ID != "demo-facebook-app-id"),
            "frontend_url": FRONTEND_URL,
            "backend_url": BACKEND_URL
        }
    }


