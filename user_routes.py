"""
User API Routes for HAVEN Crowdfunding Platform
RESTful endpoints for user management and profiles
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Request, status, Query
from pydantic import BaseModel, EmailStr, validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from auth_middleware import get_current_user, require_admin, PasswordManager
from database import get_db
from models import User, UserRole, Campaign, Donation

logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create router
user_router = APIRouter()

# Pydantic models
class UserProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    bio: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    
    @validator('full_name')
    def validate_full_name(cls, v):
        if v and len(v.strip()) < 2:
            raise ValueError("Full name must be at least 2 characters")
        return v.strip() if v else v
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        if v and not v.replace('+', '').replace('-', '').replace(' ', '').isdigit():
            raise ValueError("Invalid phone number format")
        return v
    
    @validator('bio')
    def validate_bio(cls, v):
        if v and len(v) > 500:
            raise ValueError("Bio too long (max 500 characters)")
        return v

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if not PasswordManager.validate_password_strength(v):
            raise ValueError("Password must be at least 8 characters with uppercase, lowercase, digit, and special character")
        return v

class KYCUpdateRequest(BaseModel):
    pan_number: Optional[str] = None
    aadhar_number: Optional[str] = None
    
    @validator('pan_number')
    def validate_pan(cls, v):
        if v:
            import re
            if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', v.upper()):
                raise ValueError("Invalid PAN number format")
            return v.upper()
        return v
    
    @validator('aadhar_number')
    def validate_aadhar(cls, v):
        if v:
            if not v.replace(' ', '').isdigit() or len(v.replace(' ', '')) != 12:
                raise ValueError("Invalid Aadhar number format")
        return v

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    phone_number: Optional[str]
    profile_picture: Optional[str]
    bio: Optional[str]
    date_of_birth: Optional[datetime]
    is_active: bool
    is_verified: bool
    email_verified: bool
    phone_verified: bool
    role: str
    kyc_verified: bool
    created_at: datetime
    updated_at: datetime

class UserStatsResponse(BaseModel):
    campaigns_created: int
    total_donations_made: int
    total_amount_donated: float
    total_amount_raised: float
    successful_campaigns: int

class UserListResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime

# User profile endpoints
@user_router.get("/me", response_model=UserResponse)
@limiter.limit("100/minute")
async def get_current_user_profile(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get current user profile"""
    return format_user_response(current_user)

@user_router.put("/me", response_model=UserResponse)
@limiter.limit("20/hour")
async def update_user_profile(
    request: Request,
    profile_data: UserProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user profile"""
    try:
        # Update fields
        update_data = profile_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(current_user, field, value)
        
        current_user.updated_at = datetime.now()
        db.commit()
        db.refresh(current_user)
        
        logger.info(f"User profile updated: {current_user.id}")
        
        return format_user_response(current_user)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )

@user_router.post("/me/change-password")
@limiter.limit("5/hour")
async def change_password(
    request: Request,
    password_data: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change user password"""
    try:
        # Verify current password
        if not current_user.hashed_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account uses OAuth authentication"
            )
        
        if not PasswordManager.verify_password(password_data.current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password
        current_user.hashed_password = PasswordManager.hash_password(password_data.new_password)
        current_user.updated_at = datetime.now()
        db.commit()
        
        logger.info(f"Password changed for user: {current_user.id}")
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )

@user_router.put("/me/kyc", response_model=UserResponse)
@limiter.limit("10/day")
async def update_kyc_info(
    request: Request,
    kyc_data: KYCUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update KYC information"""
    try:
        # Update KYC fields
        if kyc_data.pan_number:
            current_user.pan_number = kyc_data.pan_number
        
        if kyc_data.aadhar_number:
            current_user.aadhar_number = kyc_data.aadhar_number
        
        current_user.updated_at = datetime.now()
        db.commit()
        db.refresh(current_user)
        
        logger.info(f"KYC information updated for user: {current_user.id}")
        
        return format_user_response(current_user)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"KYC update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update KYC information"
        )

@user_router.get("/me/stats", response_model=UserStatsResponse)
@limiter.limit("50/minute")
async def get_user_stats(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user statistics"""
    try:
        # Get campaign statistics
        campaigns = db.query(Campaign).filter(Campaign.creator_id == current_user.id).all()
        campaigns_created = len(campaigns)
        successful_campaigns = len([c for c in campaigns if c.is_completed])
        total_amount_raised = sum(float(c.current_amount) for c in campaigns)
        
        # Get donation statistics
        donations = db.query(Donation).filter(Donation.donor_id == current_user.id).all()
        total_donations_made = len(donations)
        total_amount_donated = sum(float(d.amount) for d in donations)
        
        return UserStatsResponse(
            campaigns_created=campaigns_created,
            total_donations_made=total_donations_made,
            total_amount_donated=total_amount_donated,
            total_amount_raised=total_amount_raised,
            successful_campaigns=successful_campaigns
        )
        
    except Exception as e:
        logger.error(f"User stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user statistics"
        )

@user_router.get("/me/campaigns")
@limiter.limit("100/minute")
async def get_user_campaigns(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's campaigns"""
    try:
        campaigns = db.query(Campaign).filter(
            Campaign.creator_id == current_user.id
        ).offset(skip).limit(limit).all()
        
        from campaign_routes import format_campaign_response
        return [format_campaign_response(campaign) for campaign in campaigns]
        
    except Exception as e:
        logger.error(f"User campaigns error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user campaigns"
        )

@user_router.get("/me/donations")
@limiter.limit("100/minute")
async def get_user_donations(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's donations"""
    try:
        donations = db.query(Donation).filter(
            Donation.donor_id == current_user.id
        ).offset(skip).limit(limit).all()
        
        return [
            {
                "id": donation.id,
                "amount": float(donation.amount),
                "currency": donation.currency,
                "campaign": {
                    "id": donation.campaign.id,
                    "title": donation.campaign.title,
                    "creator": donation.campaign.creator.full_name
                },
                "message": donation.message,
                "is_anonymous": donation.is_anonymous,
                "created_at": donation.created_at.isoformat()
            }
            for donation in donations
        ]
        
    except Exception as e:
        logger.error(f"User donations error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user donations"
        )

# Public user endpoints
@user_router.get("/{user_id}", response_model=UserResponse)
@limiter.limit("100/minute")
async def get_user_profile(
    request: Request,
    user_id: int,
    db: Session = Depends(get_db)
):
    """Get public user profile"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Return limited public profile
        return format_user_response(user, public=True)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user profile error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )

@user_router.get("/{user_id}/campaigns")
@limiter.limit("100/minute")
async def get_user_public_campaigns(
    request: Request,
    user_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get user's public campaigns"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Only show active and completed campaigns
        campaigns = db.query(Campaign).filter(
            Campaign.creator_id == user_id,
            Campaign.status.in_(['active', 'completed'])
        ).offset(skip).limit(limit).all()
        
        from campaign_routes import format_campaign_response
        return [format_campaign_response(campaign) for campaign in campaigns]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User public campaigns error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user campaigns"
        )

# Admin endpoints
@user_router.get("/", response_model=List[UserListResponse])
@limiter.limit("50/minute")
async def get_users(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    role: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    search: Optional[str] = Query(None),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get users list (admin only)"""
    try:
        query = db.query(User)
        
        # Apply filters
        if role:
            query = query.filter(User.role == UserRole(role))
        
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                User.full_name.ilike(search_term) |
                User.email.ilike(search_term)
            )
        
        users = query.offset(skip).limit(limit).all()
        
        return [
            UserListResponse(
                id=user.id,
                email=user.email,
                full_name=user.full_name,
                role=user.role.value,
                is_active=user.is_active,
                is_verified=user.is_verified,
                created_at=user.created_at
            )
            for user in users
        ]
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Get users error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )

@user_router.put("/{user_id}/role")
@limiter.limit("20/hour")
async def update_user_role(
    request: Request,
    user_id: int,
    new_role: str,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Update user role (admin only)"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Validate role
        try:
            role_enum = UserRole(new_role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid role"
            )
        
        # Prevent self-demotion
        if user.id == current_user.id and role_enum != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot change your own admin role"
            )
        
        user.role = role_enum
        user.updated_at = datetime.now()
        db.commit()
        
        logger.info(f"User {user_id} role updated to {new_role} by admin {current_user.id}")
        
        return {"message": f"User role updated to {new_role}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update user role error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user role"
        )

@user_router.put("/{user_id}/status")
@limiter.limit("20/hour")
async def update_user_status(
    request: Request,
    user_id: int,
    is_active: bool,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Update user active status (admin only)"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Prevent self-deactivation
        if user.id == current_user.id and not is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate your own account"
            )
        
        user.is_active = is_active
        user.updated_at = datetime.now()
        db.commit()
        
        status_text = "activated" if is_active else "deactivated"
        logger.info(f"User {user_id} {status_text} by admin {current_user.id}")
        
        return {"message": f"User {status_text} successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update user status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user status"
        )

# Helper functions
def format_user_response(user: User, public: bool = False) -> UserResponse:
    """Format user for API response"""
    if public:
        # Limited public profile
        return UserResponse(
            id=user.id,
            email="",  # Hide email in public view
            full_name=user.full_name,
            phone_number=None,  # Hide phone in public view
            profile_picture=user.profile_picture,
            bio=user.bio,
            date_of_birth=None,  # Hide DOB in public view
            is_active=user.is_active,
            is_verified=user.is_verified,
            email_verified=user.email_verified,
            phone_verified=user.phone_verified,
            role=user.role.value,
            kyc_verified=user.kyc_verified,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
    else:
        # Full profile for owner/admin
        return UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            phone_number=user.phone_number,
            profile_picture=user.profile_picture,
            bio=user.bio,
            date_of_birth=user.date_of_birth,
            is_active=user.is_active,
            is_verified=user.is_verified,
            email_verified=user.email_verified,
            phone_verified=user.phone_verified,
            role=user.role.value,
            kyc_verified=user.kyc_verified,
            created_at=user.created_at,
            updated_at=user.updated_at
        )

