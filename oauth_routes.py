# oauth_routes.py - Complete file with minimal changes applied
# This is your complete oauth_routes.py with the OAuth redirect fixes

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse, RedirectResponse  # ADDED HTMLResponse
from typing import Optional, Dict, Any
import os
import httpx
import json
from urllib.parse import urlencode
import jwt  # ADDED FOR JWT TOKEN GENERATION
from datetime import datetime, timedelta  # ADDED FOR JWT EXPIRATION

# Create router
router = APIRouter()

# ================================
# JWT TOKEN GENERATION (NEW FUNCTION)
# ================================

def generate_jwt_token(user_data: Dict[str, Any]) -> str:
    """Generate JWT token for authenticated user"""
    payload = {
        "user_id": user_data.get("id"),
        "email": user_data.get("email"),
        "name": user_data.get("name"),
        "provider": user_data.get("provider"),
        "user_type": user_data.get("user_type"),
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow()
    }
    
    jwt_secret = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    return jwt.encode(payload, jwt_secret, algorithm="HS256")

# ================================
# HTML PAGE GENERATION (NEW FUNCTIONS)
# ================================

def create_oauth_success_page(provider: str, jwt_token: str, user_data: Dict[str, Any]) -> str:
    """Create HTML page for successful OAuth"""
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8501")
    
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
        </style>
    </head>
    <body>
        <div class="success-container">
            <div class="success-icon">✓</div>
            <h2>Authentication Successful!</h2>
            <p>You have successfully signed in with {provider.title()}.</p>
            <p>Redirecting to application...</p>
        </div>
        
        <script>
            // Send authentication data to parent window
            const authData = {{
                success: true,
                provider: '{provider}',
                token: '{jwt_token}',
                user: {json.dumps(user_data)}
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
                    window.location.href = '{frontend_url}?auth=success&token=' + encodeURIComponent('{jwt_token}');
                }}
            }} catch (error) {{
                console.error('Error communicating with parent window:', error);
                // Fallback: redirect to frontend
                window.location.href = '{frontend_url}?auth=success&token=' + encodeURIComponent('{jwt_token}');
            }}
        </script>
    </body>
    </html>
    """

def create_oauth_error_page(provider: str, error_message: str) -> str:
    """Create HTML page for OAuth errors"""
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8501")
    
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
        </style>
    </head>
    <body>
        <div class="error-container">
            <div class="error-icon">✗</div>
            <h2>Authentication Failed</h2>
            <p>There was an error signing in with {provider.title()}.</p>
            <div>{error_message}</div>
            <button onclick="closeWindow()">Close and Retry</button>
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
                    }} else {{
                        window.location.href = '{frontend_url}?auth=error&provider={provider}&message=' + encodeURIComponent('{error_message}');
                    }}
                }} catch (error) {{
                    window.location.href = '{frontend_url}?auth=error&provider={provider}&message=' + encodeURIComponent('{error_message}');
                }}
            }}
        </script>
    </body>
    </html>
    """

# ================================
# GOOGLE OAUTH ROUTES (UPDATED)
# ================================

@router.get("/auth/google/login")
async def google_login(user_type: str = Query("individual")):
    """
    FIXED: Initiate Google OAuth login - NOW RETURNS JSON
    """
    try:
        # Build Google OAuth authorization URL
        auth_params = {
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI"),
            "scope": "openid email profile",
            "response_type": "code",
            "access_type": "offline",
            "state": user_type
        }
        
        auth_url = f"https://accounts.google.com/o/oauth2/auth?{urlencode(auth_params)}"
        
        # MAIN FIX: Return JSON instead of redirect
        return {
            "auth_url": auth_url,
            "provider": "google",
            "user_type": user_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google OAuth login error: {str(e)}")

@router.get("/auth/google/callback")
async def google_callback(
    code: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    state: Optional[str] = Query("individual")
):
    """
    FIXED: Handle Google OAuth callback with JWT token generation
    """
    try:
        if error:
            return HTMLResponse(content=create_oauth_error_page("google", error))
        
        if not code:
            return HTMLResponse(content=create_oauth_error_page("google", "No authorization code"))
        
        # Exchange code for token
        token_data = {
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI")
        }
        
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data=token_data,
                timeout=30
            )
        
        if token_response.status_code != 200:
            return HTMLResponse(content=create_oauth_error_page("google", "Token exchange failed"))
        
        tokens = token_response.json()
        access_token = tokens.get("access_token")
        
        if not access_token:
            return HTMLResponse(content=create_oauth_error_page("google", "No access token"))
        
        # Get user info
        async with httpx.AsyncClient() as client:
            user_response = await client.get(
                f"https://www.googleapis.com/oauth2/v2/userinfo?access_token={access_token}",
                timeout=30
            )
        
        if user_response.status_code != 200:
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
        jwt_token = generate_jwt_token(jwt_user_data)
        
        # Return success page with token
        return HTMLResponse(content=create_oauth_success_page("google", jwt_token, jwt_user_data))
        
    except Exception as e:
        return HTMLResponse(content=create_oauth_error_page("google", str(e)))

# ================================
# FACEBOOK OAUTH ROUTES (UPDATED)
# ================================

@router.get("/auth/facebook/login")
async def facebook_login(user_type: str = Query("individual")):
    """
    FIXED: Initiate Facebook OAuth login - NOW RETURNS JSON
    """
    try:
        # Build Facebook OAuth authorization URL
        auth_params = {
            "client_id": os.getenv("FACEBOOK_APP_ID"),
            "redirect_uri": os.getenv("FACEBOOK_REDIRECT_URI"),
            "scope": "email,public_profile",
            "response_type": "code",
            "state": user_type
        }
        
        auth_url = f"https://www.facebook.com/v18.0/dialog/oauth?{urlencode(auth_params)}"
        
        # MAIN FIX: Return JSON instead of redirect
        return {
            "auth_url": auth_url,
            "provider": "facebook",
            "user_type": user_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Facebook OAuth login error: {str(e)}")

@router.get("/auth/facebook/callback")
async def facebook_callback(
    code: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    state: Optional[str] = Query("individual")
):
    """
    FIXED: Handle Facebook OAuth callback with JWT token generation
    """
    try:
        if error:
            return HTMLResponse(content=create_oauth_error_page("facebook", error))
        
        if not code:
            return HTMLResponse(content=create_oauth_error_page("facebook", "No authorization code"))
        
        # Exchange code for token
        token_params = {
            "client_id": os.getenv("FACEBOOK_APP_ID"),
            "client_secret": os.getenv("FACEBOOK_APP_SECRET"),
            "code": code,
            "redirect_uri": os.getenv("FACEBOOK_REDIRECT_URI")
        }
        
        async with httpx.AsyncClient() as client:
            token_response = await client.get(
                f"https://graph.facebook.com/v18.0/oauth/access_token?{urlencode(token_params)}",
                timeout=30
            )
        
        if token_response.status_code != 200:
            return HTMLResponse(content=create_oauth_error_page("facebook", "Token exchange failed"))
        
        tokens = token_response.json()
        access_token = tokens.get("access_token")
        
        if not access_token:
            return HTMLResponse(content=create_oauth_error_page("facebook", "No access token"))
        
        # Get user info
        async with httpx.AsyncClient() as client:
            user_response = await client.get(
                f"https://graph.facebook.com/v18.0/me?fields=id,name,email,picture&access_token={access_token}",
                timeout=30
            )
        
        if user_response.status_code != 200:
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
        jwt_token = generate_jwt_token(jwt_user_data)
        
        # Return success page with token
        return HTMLResponse(content=create_oauth_success_page("facebook", jwt_token, jwt_user_data))
        
    except Exception as e:
        return HTMLResponse(content=create_oauth_error_page("facebook", str(e)))

# ================================
# ADDITIONAL UTILITY ROUTES
# ================================

@router.get("/auth/verify-token")
async def verify_token(token: str = Query(...)):
    """Verify JWT token validity"""
    try:
        jwt_secret = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        decoded_token = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        
        return {
            "valid": True,
            "user_data": {
                "user_id": decoded_token.get("user_id"),
                "email": decoded_token.get("email"),
                "name": decoded_token.get("name"),
                "provider": decoded_token.get("provider"),
                "user_type": decoded_token.get("user_type")
            }
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.post("/auth/refresh-token")
async def refresh_token(token: str = Query(...)):
    """Refresh JWT token"""
    try:
        jwt_secret = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        decoded_token = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        
        # Generate new token with extended expiration
        new_token = generate_jwt_token({
            "id": decoded_token.get("user_id"),
            "email": decoded_token.get("email"),
            "name": decoded_token.get("name"),
            "provider": decoded_token.get("provider"),
            "user_type": decoded_token.get("user_type")
        })
        
        return {
            "access_token": new_token,
            "token_type": "bearer"
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ================================
# HEALTH CHECK
# ================================

@router.get("/auth/health")
async def auth_health():
    """Health check for OAuth service"""
    return {
        "status": "healthy",
        "service": "oauth",
        "timestamp": datetime.utcnow().isoformat(),
        "providers": ["google", "facebook"]
    }

