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
import requests
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

@router.get("/google/login")
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

# DEBUG VERSION - Add this to your oauth_routes.py google_callback function
# Replace the existing google_callback function with this version

@router.get("/google/callback")
async def google_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None)
):
    """Handle Google OAuth callback with detailed debugging"""
    
    print(f"🔍 DEBUG: Google callback received")
    print(f"🔍 DEBUG: code = {code[:20] if code else None}...")  # Show first 20 chars
    print(f"🔍 DEBUG: state = {state}")
    print(f"🔍 DEBUG: error = {error}")
    
    if error:
        print(f"❌ DEBUG: OAuth error from Google: {error}")
        return create_oauth_error_page("google", f"OAuth error: {error}")
    
    if not code:
        print(f"❌ DEBUG: No authorization code received")
        return create_oauth_error_page("google", "No authorization code received")
    
    try:
        # Debug environment variables
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
        
        print(f"🔍 DEBUG: GOOGLE_CLIENT_ID = {client_id[:20] if client_id else None}...")
        print(f"🔍 DEBUG: GOOGLE_CLIENT_SECRET = {'SET' if client_secret else 'NOT SET'}")
        print(f"🔍 DEBUG: GOOGLE_REDIRECT_URI = {redirect_uri}")
        
        if not all([client_id, client_secret, redirect_uri]):
            print(f"❌ DEBUG: Missing OAuth credentials")
            return create_oauth_error_page("google", "Missing OAuth credentials")
        
        # Prepare token exchange request
        token_data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }
        
        print(f"🔍 DEBUG: Making token exchange request to Google")
        
        # Make token exchange request
        response = requests.post(
            "https://oauth2.googleapis.com/token",
            data=token_data,
            headers={"Accept": "application/json"}
        )
        
        print(f"🔍 DEBUG: Token response status = {response.status_code}")
        print(f"🔍 DEBUG: Token response headers = {dict(response.headers)}")
        
        if response.status_code != 200:
            print(f"❌ DEBUG: Token exchange failed with status {response.status_code}")
            print(f"❌ DEBUG: Response text = {response.text}")
            return create_oauth_error_page("google", f"Token exchange failed: {response.text}")
        
        token_json = response.json()
        print(f"🔍 DEBUG: Token exchange successful")
        print(f"🔍 DEBUG: Token keys = {list(token_json.keys())}")
        
        access_token = token_json.get("access_token")
        if not access_token:
            print(f"❌ DEBUG: No access token in response")
            return create_oauth_error_page("google", "No access token received")
        
        # Get user info
        print(f"🔍 DEBUG: Getting user info from Google")
        user_response = requests.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        print(f"🔍 DEBUG: User info status = {user_response.status_code}")
        
        if user_response.status_code != 200:
            print(f"❌ DEBUG: Failed to get user info: {user_response.text}")
            return create_oauth_error_page("google", "Failed to get user info")
        
        user_data = user_response.json()
        print(f"🔍 DEBUG: User data keys = {list(user_data.keys())}")
        
        # Generate JWT token
        jwt_token = generate_jwt_token(user_data)
        print(f"🔍 DEBUG: JWT token generated successfully")
        
        # Return success page
        return create_oauth_success_page("google", jwt_token, user_data)
        
    except Exception as e:
        print(f"❌ DEBUG: Exception in google_callback: {str(e)}")
        print(f"❌ DEBUG: Exception type: {type(e).__name__}")
        import traceback
        print(f"❌ DEBUG: Traceback: {traceback.format_exc()}")
        return create_oauth_error_page("google", f"Internal error: {str(e)}")

# ================================
# FACEBOOK OAUTH ROUTES
# ================================

# ALSO ADD DEBUG TO FACEBOOK LOGIN ROUTE
@router.get("/facebook/login")
async def facebook_login(user_type: str = Query("individual")):
    """Initiate Facebook OAuth login with debugging"""
    
    print(f"🔍 DEBUG: Facebook login initiated")
    print(f"🔍 DEBUG: user_type = {user_type}")
    
    try:
        # Debug environment variables
        app_id = os.getenv("FACEBOOK_APP_ID")
        redirect_uri = os.getenv("FACEBOOK_REDIRECT_URI")
        
        print(f"🔍 DEBUG: FACEBOOK_APP_ID = {app_id[:20] if app_id else None}...")
        print(f"🔍 DEBUG: FACEBOOK_REDIRECT_URI = {redirect_uri}")
        
        if not app_id or not redirect_uri:
            print(f"❌ DEBUG: Missing Facebook OAuth configuration")
            raise HTTPException(status_code=500, detail="Facebook OAuth not configured")
        
        # Build Facebook OAuth authorization URL
        auth_params = {
            "client_id": app_id,
            "redirect_uri": redirect_uri,
            "scope": "email,public_profile",
            "response_type": "code",
            "state": user_type
        }
        
        auth_url = f"https://www.facebook.com/v18.0/dialog/oauth?" + urlencode(auth_params)
        
        print(f"🔍 DEBUG: Facebook OAuth URL generated")
        print(f"🔍 DEBUG: Auth URL = {auth_url[:100]}...")
        print(f"🔍 DEBUG: Scope = {auth_params['scope']}")
        
        # Return JSON instead of redirect
        return {
            "auth_url": auth_url,
            "provider": "facebook",
            "user_type": user_type
        }
        
    except Exception as e:
        print(f"❌ DEBUG: Exception in facebook_login: {str(e)}")
        import traceback
        print(f"❌ DEBUG: Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Facebook OAuth login error: {str(e)}")

# DEBUG VERSION - Add this to your oauth_routes.py facebook_callback function
# Replace the existing facebook_callback function with this version

@router.get("/facebook/callback")
async def facebook_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    error_reason: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None)
):
    """Handle Facebook OAuth callback with detailed debugging"""
    
    print(f"🔍 DEBUG: Facebook callback received")
    print(f"🔍 DEBUG: code = {code[:20] if code else None}...")  # Show first 20 chars
    print(f"🔍 DEBUG: state = {state}")
    print(f"🔍 DEBUG: error = {error}")
    print(f"🔍 DEBUG: error_reason = {error_reason}")
    print(f"🔍 DEBUG: error_description = {error_description}")
    
    if error:
        print(f"❌ DEBUG: Facebook OAuth error: {error} - {error_description}")
        return create_oauth_error_page("facebook", f"OAuth error: {error} - {error_description}")
    
    if not code:
        print(f"❌ DEBUG: No authorization code received from Facebook")
        return create_oauth_error_page("facebook", "No authorization code received")
    
    try:
        # Debug environment variables
        app_id = os.getenv("FACEBOOK_APP_ID")
        app_secret = os.getenv("FACEBOOK_APP_SECRET")
        redirect_uri = os.getenv("FACEBOOK_REDIRECT_URI")
        
        print(f"🔍 DEBUG: FACEBOOK_APP_ID = {app_id[:20] if app_id else None}...")
        print(f"🔍 DEBUG: FACEBOOK_APP_SECRET = {'SET' if app_secret else 'NOT SET'}")
        print(f"🔍 DEBUG: FACEBOOK_REDIRECT_URI = {redirect_uri}")
        
        if not all([app_id, app_secret, redirect_uri]):
            print(f"❌ DEBUG: Missing Facebook OAuth credentials")
            return create_oauth_error_page("facebook", "Missing Facebook OAuth credentials")
        
        # Step 1: Exchange code for access token
        token_url = "https://graph.facebook.com/v18.0/oauth/access_token"
        token_params = {
            "client_id": app_id,
            "client_secret": app_secret,
            "code": code,
            "redirect_uri": redirect_uri,
        }
        
        print(f"🔍 DEBUG: Making token exchange request to Facebook")
        print(f"🔍 DEBUG: Token URL = {token_url}")
        print(f"🔍 DEBUG: Redirect URI = {redirect_uri}")
        
        # Make token exchange request
        response = requests.get(token_url, params=token_params)
        
        print(f"🔍 DEBUG: Token response status = {response.status_code}")
        print(f"🔍 DEBUG: Token response headers = {dict(response.headers)}")
        print(f"🔍 DEBUG: Token response text = {response.text[:200]}...")  # First 200 chars
        
        if response.status_code != 200:
            print(f"❌ DEBUG: Facebook token exchange failed with status {response.status_code}")
            return create_oauth_error_page("facebook", f"Token exchange failed: {response.text}")
        
        # Parse token response
        try:
            token_data = response.json()
            print(f"🔍 DEBUG: Token response is JSON")
            print(f"🔍 DEBUG: Token keys = {list(token_data.keys())}")
        except:
            # Facebook sometimes returns URL-encoded response
            from urllib.parse import parse_qs
            token_data = parse_qs(response.text)
            token_data = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in token_data.items()}
            print(f"🔍 DEBUG: Token response is URL-encoded")
            print(f"🔍 DEBUG: Token keys = {list(token_data.keys())}")
        
        access_token = token_data.get("access_token")
        if not access_token:
            print(f"❌ DEBUG: No access token in Facebook response")
            print(f"❌ DEBUG: Full response = {token_data}")
            return create_oauth_error_page("facebook", "No access token received from Facebook")
        
        print(f"🔍 DEBUG: Facebook access token received: {access_token[:20]}...")
        
        # Step 2: Get user info from Facebook
        user_url = "https://graph.facebook.com/v18.0/me"
        user_params = {
            "access_token": access_token,
            "fields": "id,name,email,picture.type(large)"
        }
        
        print(f"🔍 DEBUG: Getting user info from Facebook")
        print(f"🔍 DEBUG: User URL = {user_url}")
        print(f"🔍 DEBUG: Requested fields = {user_params['fields']}")
        
        user_response = requests.get(user_url, params=user_params)
        
        print(f"🔍 DEBUG: User info status = {user_response.status_code}")
        print(f"🔍 DEBUG: User info response = {user_response.text[:200]}...")
        
        if user_response.status_code != 200:
            print(f"❌ DEBUG: Failed to get Facebook user info: {user_response.text}")
            return create_oauth_error_page("facebook", f"Failed to get user info: {user_response.text}")
        
        user_data = user_response.json()
        print(f"🔍 DEBUG: Facebook user data keys = {list(user_data.keys())}")
        print(f"🔍 DEBUG: User ID = {user_data.get('id')}")
        print(f"🔍 DEBUG: User name = {user_data.get('name')}")
        print(f"🔍 DEBUG: User email = {user_data.get('email', 'NOT PROVIDED')}")
        
        # Check if email is provided
        if not user_data.get('email'):
            print(f"⚠️ DEBUG: Facebook user didn't provide email")
            # You might want to handle this case differently
        
        # Step 3: Generate JWT token
        print(f"🔍 DEBUG: Generating JWT token for Facebook user")
        jwt_token = generate_jwt_token(user_data)
        print(f"🔍 DEBUG: JWT token generated successfully")
        
        # Return success page
        return create_oauth_success_page("facebook", jwt_token, user_data)
        
    except Exception as e:
        print(f"❌ DEBUG: Exception in facebook_callback: {str(e)}")
        print(f"❌ DEBUG: Exception type: {type(e).__name__}")
        import traceback
        print(f"❌ DEBUG: Traceback: {traceback.format_exc()}")
        return create_oauth_error_page("facebook", f"Internal error: {str(e)}")

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

