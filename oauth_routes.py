"""
OAuth Authentication Routes for HAVEN Crowdfunding Platform
Secure Google and Facebook OAuth integration with proper error handling
"""

import logging
import secrets
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from database import get_db
from models import User, UserRole
from auth_middleware import (
    create_user_tokens, PasswordManager, get_current_user,
    token_manager, session_manager
)
from oauth_config import get_oauth_config, OAuthProvider, OAuthUser
from config import get_settings

logger = logging.getLogger(__name__)

# Get configuration
settings = get_settings()
oauth_config = get_oauth_config()

# Create router
oauth_router = APIRouter()

# Pydantic models
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    phone_number: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: Dict[str, Any]

class OAuthStateData(BaseModel):
    provider: str
    redirect_url: Optional[str] = None
    timestamp: float

# In-memory state storage (use Redis in production)
oauth_states = {}

def generate_oauth_state(provider: str, redirect_url: str = None) -> str:
    """Generate secure OAuth state parameter"""
    state = secrets.token_urlsafe(32)
    oauth_states[state] = OAuthStateData(
        provider=provider,
        redirect_url=redirect_url,
        timestamp=datetime.utcnow().timestamp()
    )
    return state

def validate_oauth_state(state: str) -> Optional[OAuthStateData]:
    """Validate OAuth state parameter"""
    state_data = oauth_states.get(state)
    if not state_data:
        return None
    
    # Check if state is expired (10 minutes)
    if datetime.utcnow().timestamp() - state_data.timestamp > 600:
        oauth_states.pop(state, None)
        return None
    
    return state_data

def create_or_update_user(oauth_user: OAuthUser, db: Session) -> User:
    """Create or update user from OAuth data"""
    # Check if user exists
    user = db.query(User).filter(User.email == oauth_user.email).first()
    
    if user:
        # Update existing user
        if oauth_user.provider == "google" and not user.google_id:
            user.google_id = oauth_user.provider_id
        elif oauth_user.provider == "facebook" and not user.facebook_id:
            user.facebook_id = oauth_user.provider_id
        
        # Update profile information if not set
        if not user.profile_picture and oauth_user.picture:
            user.profile_picture = oauth_user.picture
        
        user.email_verified = True
        user.is_active = True
        
    else:
        # Create new user
        user = User(
            email=oauth_user.email,
            full_name=oauth_user.name,
            profile_picture=oauth_user.picture,
            email_verified=True,
            is_active=True,
            role=UserRole.USER
        )
        
        # Set provider-specific ID
        if oauth_user.provider == "google":
            user.google_id = oauth_user.provider_id
        elif oauth_user.provider == "facebook":
            user.facebook_id = oauth_user.provider_id
        
        db.add(user)
    
    db.commit()
    db.refresh(user)
    return user

# Traditional login/register endpoints
@oauth_router.post("/login", response_model=TokenResponse)
async def login(
    login_data: LoginRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """Traditional email/password login"""
    try:
        # Find user
        user = db.query(User).filter(User.email == login_data.email).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Verify password
        if not user.hashed_password or not PasswordManager.verify_password(
            login_data.password, user.hashed_password
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is disabled"
            )
        
        # Create tokens
        tokens = create_user_tokens(user)
        
        # Create session
        session_id = session_manager.create_session(user.id, tokens["access_token"])
        
        logger.info(f"User {user.email} logged in successfully")
        
        return TokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type=tokens["token_type"],
            expires_in=settings.jwt_expiration_hours * 3600,
            user={
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role.value,
                "is_verified": user.is_verified
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@oauth_router.post("/register", response_model=TokenResponse)
async def register(
    register_data: RegisterRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """User registration"""
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == register_data.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Validate password strength
        if not PasswordManager.validate_password_strength(register_data.password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters with uppercase, lowercase, digit, and special character"
            )
        
        # Create new user
        hashed_password = PasswordManager.hash_password(register_data.password)
        user = User(
            email=register_data.email,
            hashed_password=hashed_password,
            full_name=register_data.full_name,
            phone_number=register_data.phone_number,
            role=UserRole.USER,
            is_active=True
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # Create tokens
        tokens = create_user_tokens(user)
        
        # Create session
        session_id = session_manager.create_session(user.id, tokens["access_token"])
        
        logger.info(f"New user registered: {user.email}")
        
        return TokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type=tokens["token_type"],
            expires_in=settings.jwt_expiration_hours * 3600,
            user={
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role.value,
                "is_verified": user.is_verified
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

# Google OAuth endpoints
@oauth_router.get("/google")
async def google_login(request: Request, redirect_url: str = None):
    """Initiate Google OAuth login"""
    try:
        if not oauth_config.is_google_configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Google OAuth not configured"
            )
        
        # Generate state
        state = generate_oauth_state("google", redirect_url)
        
        # Get authorization URL
        auth_url = oauth_config.get_google_auth_url(state)
        
        logger.info("Google OAuth login initiated")
        return RedirectResponse(url=auth_url)
    
    except Exception as e:
        logger.error(f"Google OAuth initiation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth initiation failed"
        )

@oauth_router.get("/google/callback")
async def google_callback(
    request: Request,
    code: str = None,
    state: str = None,
    error: str = None,
    db: Session = Depends(get_db)
):
    """Handle Google OAuth callback"""
    try:
        # Check for OAuth errors
        if error:
            logger.warning(f"Google OAuth error: {error}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"OAuth error: {error}"
            )
        
        if not code or not state:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing authorization code or state"
            )
        
        # Validate state
        state_data = validate_oauth_state(state)
        if not state_data or state_data.provider != "google":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired state"
            )
        
        # Exchange code for token
        token_data = oauth_config.exchange_google_code_for_token(code, state)
        
        # Get user info from Google
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                oauth_config.google_userinfo_url,
                headers={"Authorization": f"Bearer {token_data['access_token']}"}
            )
            response.raise_for_status()
            user_info = response.json()
        
        # Create OAuth user object
        oauth_user = OAuthUser(
            id=user_info["id"],
            email=user_info["email"],
            name=user_info["name"],
            picture=user_info.get("picture"),
            provider="google",
            provider_id=user_info["id"]
        )
        
        # Create or update user
        user = create_or_update_user(oauth_user, db)
        
        # Create tokens
        tokens = create_user_tokens(user)
        
        # Create session
        session_id = session_manager.create_session(user.id, tokens["access_token"])
        
        # Clean up state
        oauth_states.pop(state, None)
        
        logger.info(f"Google OAuth login successful for {user.email}")
        
        # Redirect to frontend with tokens
        frontend_url = state_data.redirect_url or settings.allowed_origins[0]
        redirect_url = f"{frontend_url}/auth/callback?token={tokens['access_token']}&refresh={tokens['refresh_token']}"
        
        return RedirectResponse(url=redirect_url)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Google OAuth callback error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth callback failed"
        )

# Facebook OAuth endpoints
@oauth_router.get("/facebook")
async def facebook_login(request: Request, redirect_url: str = None):
    """Initiate Facebook OAuth login"""
    try:
        if not oauth_config.is_facebook_configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Facebook OAuth not configured"
            )
        
        # Generate state
        state = generate_oauth_state("facebook", redirect_url)
        
        # Get authorization URL
        auth_url = oauth_config.get_facebook_auth_url(state)
        
        logger.info("Facebook OAuth login initiated")
        return RedirectResponse(url=auth_url)
    
    except Exception as e:
        logger.error(f"Facebook OAuth initiation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth initiation failed"
        )

@oauth_router.get("/facebook/callback")
async def facebook_callback(
    request: Request,
    code: str = None,
    state: str = None,
    error: str = None,
    db: Session = Depends(get_db)
):
    """Handle Facebook OAuth callback"""
    try:
        # Check for OAuth errors
        if error:
            logger.warning(f"Facebook OAuth error: {error}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"OAuth error: {error}"
            )
        
        if not code or not state:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing authorization code or state"
            )
        
        # Validate state
        state_data = validate_oauth_state(state)
        if not state_data or state_data.provider != "facebook":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired state"
            )
        
        # Exchange code for token
        token_data = oauth_config.exchange_facebook_code_for_token(code, state)
        
        # Get user info from Facebook
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{oauth_config.facebook_userinfo_url}?fields=id,name,email,picture&access_token={token_data['access_token']}"
            )
            response.raise_for_status()
            user_info = response.json()
        
        # Create OAuth user object
        oauth_user = OAuthUser(
            id=user_info["id"],
            email=user_info.get("email"),
            name=user_info["name"],
            picture=user_info.get("picture", {}).get("data", {}).get("url"),
            provider="facebook",
            provider_id=user_info["id"]
        )
        
        # Create or update user
        user = create_or_update_user(oauth_user, db)
        
        # Create tokens
        tokens = create_user_tokens(user)
        
        # Create session
        session_id = session_manager.create_session(user.id, tokens["access_token"])
        
        # Clean up state
        oauth_states.pop(state, None)
        
        logger.info(f"Facebook OAuth login successful for {user.email}")
        
        # Redirect to frontend with tokens
        frontend_url = state_data.redirect_url or settings.allowed_origins[0]
        redirect_url = f"{frontend_url}/auth/callback?token={tokens['access_token']}&refresh={tokens['refresh_token']}"
        
        return RedirectResponse(url=redirect_url)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Facebook OAuth callback error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth callback failed"
        )

# Token management endpoints
@oauth_router.post("/refresh")
async def refresh_token(refresh_token: str):
    """Refresh access token"""
    try:
        new_access_token = token_manager.refresh_access_token(refresh_token)
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": settings.jwt_expiration_hours * 3600
        }
    
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

@oauth_router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """Logout user and revoke session"""
    try:
        # Revoke all user sessions
        session_manager.revoke_user_sessions(current_user.id)
        
        logger.info(f"User {current_user.email} logged out")
        
        return {"message": "Logged out successfully"}
    
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

# User info endpoint
@oauth_router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role.value,
        "is_verified": current_user.is_verified,
        "email_verified": current_user.email_verified,
        "profile_picture": current_user.profile_picture,
        "created_at": current_user.created_at.isoformat()
    }

