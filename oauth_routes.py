"""
Complete Backend OAuth Routes for HAVEN Crowdfunding Platform
This file contains the complete OAuth implementation for the backend
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from typing import Optional, Dict, Any
import os
import httpx
import json
from urllib.parse import urlencode
import jwt
from datetime import datetime, timedelta

# Create router
router = APIRouter()

def generate_jwt_token(user_data: Dict[str, Any]) -> str:
    """Generate JWT token for authenticated user"""
    payload = {
        "user_id": user_data.get("id"),
        "email": user_data.get("email"),
        "name": user_data.get("name"),
        "provider": user_data.get("provider"),
        "user_type": user_data.get("user_type"),
        "picture": user_data.get("picture"),
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow()
    }
    
    jwt_secret = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    return jwt.encode(payload, jwt_secret, algorithm="HS256")

def create_oauth_success_page(provider: str, jwt_token: str, user_data: Dict[str, Any]) -> str:
    """Create HTML page for successful OAuth"""
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8501")
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Authentication Successful</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            .success-container {{
                text-align: center;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                padding: 3rem;
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                border: 1px solid rgba(255, 255, 255, 0.2);
                max-width: 400px;
                width: 90%;
            }}
            .success-icon {{
                font-size: 4rem;
                margin-bottom: 1rem;
                animation: bounce 2s infinite;
            }}
            .provider-name {{
                color: #4CAF50;
                font-weight: bold;
                text-transform: capitalize;
            }}
            .loading-dots {{
                display: inline-block;
                animation: dots 1.5s infinite;
            }}
            @keyframes bounce {{
                0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
                40% {{ transform: translateY(-10px); }}
                60% {{ transform: translateY(-5px); }}
            }}
            @keyframes dots {{
                0%, 20% {{ opacity: 0; }}
                50% {{ opacity: 1; }}
                100% {{ opacity: 0; }}
            }}
        </style>
    </head>
    <body>
        <div class="success-container">
            <div class="success-icon">✅</div>
            <h2>Authentication Successful!</h2>
            <p>You have successfully signed in with <span class="provider-name">{provider}</span>.</p>
            <p>Redirecting to application<span class="loading-dots">...</span></p>
        </div>
        
        <script>
            // Send authentication data to parent window
            const authData = {{
                success: true,
                provider: '{provider}',
                token: '{jwt_token}',
                user: {json.dumps(user_data)}
            }};
            
            console.log('OAuth Success - sending data to parent window:', authData);
            
            // Try multiple communication methods
            function communicateWithParent() {{
                try {{
                    // Method 1: PostMessage to opener (popup)
                    if (window.opener && !window.opener.closed) {{
                        console.log('Sending message to opener window');
                        window.opener.postMessage({{
                            type: 'OAUTH_SUCCESS',
                            data: authData
                        }}, '*');
                        
                        // Close popup after short delay
                        setTimeout(() => {{
                            window.close();
                        }}, 1000);
                        return;
                    }}
                    
                    // Method 2: PostMessage to parent (iframe)
                    if (window.parent && window.parent !== window) {{
                        console.log('Sending message to parent window');
                        window.parent.postMessage({{
                            type: 'OAUTH_SUCCESS',
                            data: authData
                        }}, '*');
                        return;
                    }}
                    
                    // Method 3: Fallback - redirect to frontend with token
                    console.log('No parent window found, redirecting to frontend');
                    const redirectUrl = '{frontend_url}?auth=success&token=' + encodeURIComponent('{jwt_token}');
                    window.location.href = redirectUrl;
                    
                }} catch (error) {{
                    console.error('Error communicating with parent window:', error);
                    // Final fallback
                    const redirectUrl = '{frontend_url}?auth=success&token=' + encodeURIComponent('{jwt_token}');
                    window.location.href = redirectUrl;
                }}
            }}
            
            // Execute communication after page loads
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', communicateWithParent);
            }} else {{
                communicateWithParent();
            }}
            
            // Also try after a short delay to ensure everything is loaded
            setTimeout(communicateWithParent, 500);
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
        <title>Authentication Failed</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                color: white;
            }}
            .error-container {{
                text-align: center;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                padding: 3rem;
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                border: 1px solid rgba(255, 255, 255, 0.2);
                max-width: 400px;
                width: 90%;
            }}
            .error-icon {{
                font-size: 4rem;
                margin-bottom: 1rem;
            }}
            .provider-name {{
                color: #ffeb3b;
                font-weight: bold;
                text-transform: capitalize;
            }}
            .error-message {{
                background: rgba(255, 255, 255, 0.1);
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
                font-family: monospace;
                font-size: 0.9rem;
            }}
            .retry-button {{
                background: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 1rem;
                transition: background-color 0.2s;
            }}
            .retry-button:hover {{
                background: #45a049;
            }}
        </style>
    </head>
    <body>
        <div class="error-container">
            <div class="error-icon">❌</div>
            <h2>Authentication Failed</h2>
            <p><span class="provider-name">{provider}</span> authentication was unsuccessful.</p>
            <div class="error-message">{error_message}</div>
            <button class="retry-button" onclick="closeWindow()">Close and Retry</button>
        </div>
        
        <script>
            function closeWindow() {{
                // Send error message to parent window
                const errorData = {{
                    success: false,
                    provider: '{provider}',
                    error: '{error_message}'
                }};
                
                try {{
                    if (window.opener && !window.opener.closed) {{
                        window.opener.postMessage({{
                            type: 'OAUTH_ERROR',
                            data: errorData
                        }}, '*');
                        window.close();
                    }} else if (window.parent && window.parent !== window) {{
                        window.parent.postMessage({{
                            type: 'OAUTH_ERROR',
                            data: errorData
                        }}, '*');
                    }} else {{
                        // Redirect to frontend with error
                        const redirectUrl = '{frontend_url}?auth=error&provider={provider}&message=' + encodeURIComponent('{error_message}');
                        window.location.href = redirectUrl;
                    }}
                }} catch (error) {{
                    console.error('Error communicating with parent window:', error);
                    // Fallback redirect
                    const redirectUrl = '{frontend_url}?auth=error&provider={provider}&message=' + encodeURIComponent('{error_message}');
                    window.location.href = redirectUrl;
                }}
            }}
            
            // Auto-close after 10 seconds
            setTimeout(closeWindow, 10000);
        </script>
    </body>
    </html>
    """

# ================================
# GOOGLE OAUTH ROUTES
# ================================

@router.get("/auth/google/login")
async def google_login(user_type: str = Query("individual")):
    """Initiate Google OAuth login - RETURNS JSON"""
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
    """Handle Google OAuth callback with JWT token generation"""
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
# FACEBOOK OAUTH ROUTES
# ================================

@router.get("/auth/facebook/login")
async def facebook_login(user_type: str = Query("individual")):
    """Initiate Facebook OAuth login - RETURNS JSON"""
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
    """Handle Facebook OAuth callback with JWT token generation"""
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
            "picture": user_data.get("picture", {}).get("data", {}).get("url") if user_data.get("picture") else None
        }
        
        # Generate JWT token
        jwt_token = generate_jwt_token(jwt_user_data)
        
        # Return success page with token
        return HTMLResponse(content=create_oauth_success_page("facebook", jwt_token, jwt_user_data))
        
    except Exception as e:
        return HTMLResponse(content=create_oauth_error_page("facebook", str(e)))

# ================================
# UTILITY ROUTES
# ================================

@router.get("/auth/health")
async def auth_health():
    """Health check for OAuth service"""
    return {
        "status": "healthy",
        "service": "oauth",
        "timestamp": datetime.utcnow().isoformat(),
        "providers": ["google", "facebook"],
        "environment": {
            "google_configured": bool(os.getenv("GOOGLE_CLIENT_ID")),
            "facebook_configured": bool(os.getenv("FACEBOOK_APP_ID")),
            "jwt_configured": bool(os.getenv("JWT_SECRET_KEY")),
            "frontend_url": os.getenv("FRONTEND_URL", "not_set")
        }
    }

@router.post("/auth/verify-token")
async def verify_token(token: str):
    """Verify JWT token validity"""
    try:
        jwt_secret = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        decoded_token = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        return {
            "valid": True,
            "user_data": decoded_token
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.post("/auth/refresh-token")
async def refresh_token(token: str):
    """Refresh expired JWT token"""
    try:
        jwt_secret = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        # Decode without verification to get user data
        decoded_token = jwt.decode(token, jwt_secret, algorithms=["HS256"], options={"verify_exp": False})
        
        # Generate new token
        new_token = generate_jwt_token(decoded_token)
        
        return {
            "token": new_token,
            "user_data": decoded_token
        }
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

