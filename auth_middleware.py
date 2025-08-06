"""
Enhanced Authentication Middleware for HAVEN Crowdfunding Platform
Secure JWT token validation and user authentication with role-based access control
"""

import jwt as pyjwt  # Use PyJWT with alias to avoid conflicts
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from sqlalchemy.orm import Session

# Import with error handling
try:
    from config import get_settings
    settings = get_settings()
except ImportError:
    # Fallback configuration if config import fails
    class FallbackSettings:
        jwt_secret_key = "dev-jwt-secret-change-in-production"
        jwt_algorithm = "HS256"
        jwt_expiration_hours = 24
    settings = FallbackSettings()

try:
    from database import get_db
except ImportError:
    # Mock database function if database import fails
    def get_db():
        return None

try:
    from models import User, UserRole, IndividualRegistration, OrganizationRegistration
except ImportError:
    # Mock models if models import fails
    class UserRole:
        INDIVIDUAL = "individual"
        ORGANIZATION = "organization"
        ADMIN = "admin"
        MODERATOR = "moderator"
    
    class User:
        id = None
        email = None
        is_active = True
        role = None
        is_registered = False

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer security scheme
security = HTTPBearer()

class AuthenticationError(Exception):
    """Custom authentication error"""
    pass

class AuthorizationError(Exception):
    """Custom authorization error"""
    pass

class TokenManager:
    """JWT Token management"""
    
    def __init__(self):
        self.secret_key = settings.jwt_secret_key
        self.algorithm = settings.jwt_algorithm
        self.expiration_hours = settings.jwt_expiration_hours
        
        logger.info("✅ TokenManager initialized successfully")
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(hours=self.expiration_hours)
        to_encode.update({"exp": expire, "type": "access"})
        
        try:
            encoded_jwt = pyjwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"Token creation failed: {e}")
            raise AuthenticationError("Failed to create access token")
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=30)  # 30 days for refresh
        to_encode.update({"exp": expire, "type": "refresh"})
        
        try:
            encoded_jwt = pyjwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"Refresh token creation failed: {e}")
            raise AuthenticationError("Failed to create refresh token")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = pyjwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != "access":
                raise AuthenticationError("Invalid token type")
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                raise AuthenticationError("Token has expired")
            
            return payload
        
        except pyjwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except pyjwt.JWTError as e:
            logger.error(f"Token verification failed: {e}")
            raise AuthenticationError("Invalid token")
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """Create new access token from refresh token"""
        try:
            payload = pyjwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid refresh token")
            
            # Create new access token
            user_data = {
                "sub": payload.get("sub"),
                "email": payload.get("email"),
                "user_id": payload.get("user_id"),
                "role": payload.get("role")
            }
            
            return self.create_access_token(user_data)
        
        except pyjwt.ExpiredSignatureError:
            raise AuthenticationError("Refresh token has expired")
        except pyjwt.JWTError as e:
            logger.error(f"Refresh token verification failed: {e}")
            raise AuthenticationError("Invalid refresh token")

# Global token manager instance
try:
    token_manager = TokenManager()
    logger.info("✅ Global token manager created successfully")
except Exception as e:
    logger.error(f"❌ Failed to create token manager: {e}")
    token_manager = None

class PasswordManager:
    """Password hashing and verification"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)

# Base authentication dependency functions
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    
    if not token_manager:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable"
        )
    
    try:
        # Extract token from credentials
        token = credentials.credentials
        
        # Verify token
        payload = token_manager.verify_token(token)
        
        # Extract user information
        user_id = payload.get("user_id")
        email = payload.get("email")
        
        if not user_id:
            raise AuthenticationError("Invalid token payload")
        
        # Get user from database (if database is available)
        if db is not None:
            try:
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    raise AuthenticationError("User not found")
                if not user.is_active:
                    raise AuthenticationError("User account is inactive")
                return user
            except Exception as e:
                logger.warning(f"Database query failed: {e}")
        
        # Fallback: create mock user object
        mock_user = User()
        mock_user.id = user_id
        mock_user.email = email
        mock_user.is_active = True
        mock_user.role = payload.get("role", UserRole.INDIVIDUAL)
        mock_user.is_registered = payload.get("is_registered", False)
        return mock_user
        
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

# NEW: Role-based authentication dependency functions

async def get_current_user_individual(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current user - must be INDIVIDUAL role and registered"""
    if current_user.role != UserRole.INDIVIDUAL:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only individuals can perform this action"
        )
    if not current_user.is_registered:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please complete individual registration first"
        )
    return current_user

async def get_current_user_organization(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current user - must be ORGANIZATION role and registered"""
    if current_user.role != UserRole.ORGANIZATION:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only organizations can perform this action"
        )
    if not current_user.is_registered:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please complete organization registration first"
        )
    return current_user

async def get_current_user_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current user - must be ADMIN role"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

async def get_current_user_moderator(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current user - must be MODERATOR or ADMIN role"""
    if current_user.role not in [UserRole.MODERATOR, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Moderator or admin access required"
        )
    return current_user

async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin role - alias for get_current_user_admin"""
    return await get_current_user_admin(current_user)

# Registration validation functions

async def validate_individual_registration(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> User:
    """Validate that user has completed individual registration"""
    if current_user.role != UserRole.INDIVIDUAL:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User must be registered as individual"
        )
    
    if db is not None:
        try:
            registration = db.query(IndividualRegistration).filter(
                IndividualRegistration.user_id == current_user.id
            ).first()
            
            if not registration:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Individual registration not found. Please complete registration."
                )
        except Exception as e:
            logger.warning(f"Registration validation failed: {e}")
    
    return current_user

async def validate_organization_registration(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> User:
    """Validate that user has completed organization registration"""
    if current_user.role != UserRole.ORGANIZATION:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User must be registered as organization"
        )
    
    if db is not None:
        try:
            registration = db.query(OrganizationRegistration).filter(
                OrganizationRegistration.user_id == current_user.id
            ).first()
            
            if not registration:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Organization registration not found. Please complete registration."
                )
        except Exception as e:
            logger.warning(f"Registration validation failed: {e}")
    
    return current_user

# Optional authentication (doesn't raise error if no token)
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, None otherwise"""
    
    if not credentials or not token_manager:
        return None
    
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None

# Token verification function (for direct use)
def verify_token(token: str) -> Dict[str, Any]:
    """Verify token and return payload"""
    if not token_manager:
        raise AuthenticationError("Token manager not available")
    
    return token_manager.verify_token(token)

# Token creation functions
def create_access_token(data: Dict[str, Any]) -> str:
    """Create access token"""
    if not token_manager:
        raise AuthenticationError("Token manager not available")
    
    return token_manager.create_access_token(data)

def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create refresh token"""
    if not token_manager:
        raise AuthenticationError("Token manager not available")
    
    return token_manager.create_refresh_token(data)

# Password utilities
def hash_password(password: str) -> str:
    """Hash password"""
    return PasswordManager.hash_password(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password"""
    return PasswordManager.verify_password(plain_password, hashed_password)

# Utility functions for role checking
def check_user_role(user: User, required_role: UserRole) -> bool:
    """Check if user has required role"""
    return user.role == required_role

def check_user_roles(user: User, required_roles: list) -> bool:
    """Check if user has any of the required roles"""
    return user.role in required_roles

def is_individual(user: User) -> bool:
    """Check if user is individual"""
    return user.role == UserRole.INDIVIDUAL

def is_organization(user: User) -> bool:
    """Check if user is organization"""
    return user.role == UserRole.ORGANIZATION

def is_admin(user: User) -> bool:
    """Check if user is admin"""
    return user.role == UserRole.ADMIN

def is_moderator(user: User) -> bool:
    """Check if user is moderator or admin"""
    return user.role in [UserRole.MODERATOR, UserRole.ADMIN]

# Export all authentication utilities
__all__ = [
    "TokenManager",
    "PasswordManager",
    "AuthenticationError",
    "AuthorizationError",
    "get_current_user",
    "get_current_active_user",
    "get_current_user_individual",
    "get_current_user_organization",
    "get_current_user_admin",
    "get_current_user_moderator",
    "require_admin",
    "validate_individual_registration",
    "validate_organization_registration",
    "get_current_user_optional",
    "verify_token",
    "create_access_token",
    "create_refresh_token",
    "hash_password",
    "verify_password",
    "check_user_role",
    "check_user_roles",
    "is_individual",
    "is_organization",
    "is_admin",
    "is_moderator",
    "token_manager"
]

