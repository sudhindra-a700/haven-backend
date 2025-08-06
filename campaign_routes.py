"""
Campaign API Routes for HAVEN Crowdfunding Platform - Enhanced with Role-Based Access Control
RESTful endpoints for campaign management with individual/organization role restrictions
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Request, status, Query, File, UploadFile
from pydantic import BaseModel, validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_

# Updated imports for role-based access control
from auth_middleware import (
    get_current_user, get_current_user_individual, get_current_user_organization,
    get_current_user_admin, get_current_user_moderator
)
from database import get_db
from models import (
    User, Campaign, CampaignStatus, CampaignCategory, 
    Donation, UserRole
)

logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create router
campaign_router = APIRouter()

# Pydantic models
class CampaignCreateRequest(BaseModel):
    title: str
    description: str
    short_description: Optional[str] = None
    goal_amount: float
    category: str
    organization_name: Optional[str] = None
    ngo_darpan_id: Optional[str] = None
    fcra_number: Optional[str] = None
    end_date: Optional[datetime] = None
    
    @validator('title')
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError("Title cannot be empty")
        if len(v) > 255:
            raise ValueError("Title too long (max 255 characters)")
        return v.strip()
    
    @validator('description')
    def validate_description(cls, v):
        if not v or not v.strip():
            raise ValueError("Description cannot be empty")
        if len(v) < 100:
            raise ValueError("Description too short (min 100 characters)")
        return v.strip()
    
    @validator('goal_amount')
    def validate_goal_amount(cls, v):
        if v <= 0:
            raise ValueError("Goal amount must be positive")
        if v > 100000000:  # 10 crores
            raise ValueError("Goal amount too high")
        return v
    
    @validator('category')
    def validate_category(cls, v):
        valid_categories = [cat.value for cat in CampaignCategory]
        if v not in valid_categories:
            raise ValueError(f"Invalid category. Must be one of: {valid_categories}")
        return v

class CampaignUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    short_description: Optional[str] = None
    goal_amount: Optional[float] = None
    organization_name: Optional[str] = None
    end_date: Optional[datetime] = None

class CampaignResponse(BaseModel):
    id: str
    title: str
    description: str
    short_description: Optional[str]
    goal_amount: float
    current_amount: float
    currency: str
    category: str
    status: str
    featured_image: Optional[str]
    gallery_images: Optional[List[str]]
    video_url: Optional[str]
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    organization_name: Optional[str]
    is_verified: bool
    fraud_score: Optional[float]
    view_count: int
    donor_count: int
    progress_percentage: float
    creator: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class DonationRequest(BaseModel):
    amount: float
    is_anonymous: bool = False
    message: Optional[str] = None
    dedication: Optional[str] = None
    tax_receipt_required: bool = False
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError("Donation amount must be positive")
        if v > 10000000:  # 1 crore
            raise ValueError("Donation amount too high")
        return v

# Helper function to format campaign response
def format_campaign_response(campaign: Campaign) -> Dict[str, Any]:
    """Format campaign for API response"""
    return {
        "id": campaign.id,
        "title": campaign.title,
        "description": campaign.description,
        "short_description": campaign.short_description,
        "goal_amount": campaign.goal_amount,
        "current_amount": campaign.current_amount,
        "currency": campaign.currency,
        "category": campaign.category.value if campaign.category else None,
        "status": campaign.status.value if campaign.status else None,
        "featured_image": campaign.featured_image,
        "gallery_images": campaign.gallery_images,
        "video_url": campaign.video_url,
        "start_date": campaign.start_date,
        "end_date": campaign.end_date,
        "organization_name": campaign.organization_name,
        "is_verified": campaign.is_verified,
        "fraud_score": campaign.fraud_score,
        "view_count": campaign.view_count,
        "donor_count": campaign.donor_count,
        "progress_percentage": campaign.progress_percentage,
        "creator": {
            "id": campaign.creator.id,
            "full_name": campaign.creator.full_name,
            "role": campaign.creator.role.value if campaign.creator.role else None
        } if campaign.creator else None,
        "created_at": campaign.created_at,
        "updated_at": campaign.updated_at
    }

# MODIFIED: Campaign CRUD endpoints with role-based access control

@campaign_router.post("/", response_model=Dict[str, Any])
@limiter.limit("10/hour")
async def create_campaign(
    request: Request,
    campaign_data: CampaignCreateRequest,
    current_user: User = Depends(get_current_user_organization),  # CHANGED: Only organizations can create campaigns
    db: Session = Depends(get_db)
):
    """Create a new campaign - ORGANIZATIONS ONLY"""
    try:
        # Create campaign
        campaign = Campaign(
            title=campaign_data.title,
            description=campaign_data.description,
            short_description=campaign_data.short_description,
            goal_amount=campaign_data.goal_amount,
            category=CampaignCategory(campaign_data.category),
            organization_name=campaign_data.organization_name,
            end_date=campaign_data.end_date,
            creator_id=current_user.id,
            status=CampaignStatus.DRAFT,
            start_date=datetime.utcnow()
        )
        
        # Generate slug
        import re
        slug = re.sub(r'[^a-zA-Z0-9\s]', '', campaign_data.title.lower())
        slug = re.sub(r'\s+', '-', slug)[:50]
        campaign.slug = f"{slug}-{int(datetime.utcnow().timestamp())}"
        
        db.add(campaign)
        db.commit()
        db.refresh(campaign)
        
        logger.info(f"Campaign created by organization {current_user.id}: {campaign.id}")
        
        return {
            "message": "Campaign created successfully",
            "campaign": format_campaign_response(campaign)
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Campaign creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create campaign"
        )

@campaign_router.get("/", response_model=Dict[str, Any])
async def get_campaigns(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get campaigns with filtering and pagination - PUBLIC ACCESS"""
    try:
        query = db.query(Campaign)
        
        # Apply filters
        if category:
            query = query.filter(Campaign.category == CampaignCategory(category))
        
        if status:
            query = query.filter(Campaign.status == CampaignStatus(status))
        else:
            # By default, only show active campaigns to public
            query = query.filter(Campaign.status == CampaignStatus.ACTIVE)
        
        if search:
            query = query.filter(
                or_(
                    Campaign.title.ilike(f"%{search}%"),
                    Campaign.description.ilike(f"%{search}%"),
                    Campaign.organization_name.ilike(f"%{search}%")
                )
            )
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        campaigns = query.offset(skip).limit(limit).all()
        
        return {
            "campaigns": [format_campaign_response(campaign) for campaign in campaigns],
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Get campaigns error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve campaigns"
        )

@campaign_router.get("/{campaign_id}", response_model=Dict[str, Any])
async def get_campaign(
    campaign_id: str,
    db: Session = Depends(get_db)
):
    """Get single campaign by ID - PUBLIC ACCESS"""
    try:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        
        if not campaign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign not found"
            )
        
        # Increment view count
        campaign.view_count += 1
        db.commit()
        
        return {
            "campaign": format_campaign_response(campaign)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get campaign error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve campaign"
        )

@campaign_router.put("/{campaign_id}", response_model=Dict[str, Any])
async def update_campaign(
    campaign_id: str,
    campaign_data: CampaignUpdateRequest,
    current_user: User = Depends(get_current_user_organization),  # CHANGED: Only organizations can update campaigns
    db: Session = Depends(get_db)
):
    """Update campaign - ORGANIZATIONS ONLY (creator or admin)"""
    try:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        
        if not campaign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign not found"
            )
        
        # Check permissions (creator or admin)
        if campaign.creator_id != current_user.id and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )
        
        # Update fields
        if campaign_data.title is not None:
            campaign.title = campaign_data.title
        if campaign_data.description is not None:
            campaign.description = campaign_data.description
        if campaign_data.short_description is not None:
            campaign.short_description = campaign_data.short_description
        if campaign_data.goal_amount is not None:
            campaign.goal_amount = campaign_data.goal_amount
        if campaign_data.organization_name is not None:
            campaign.organization_name = campaign_data.organization_name
        if campaign_data.end_date is not None:
            campaign.end_date = campaign_data.end_date
        
        campaign.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(campaign)
        
        logger.info(f"Campaign {campaign_id} updated by user {current_user.id}")
        
        return {
            "message": "Campaign updated successfully",
            "campaign": format_campaign_response(campaign)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update campaign error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update campaign"
        )

@campaign_router.delete("/{campaign_id}")
async def delete_campaign(
    campaign_id: str,
    current_user: User = Depends(get_current_user_organization),  # CHANGED: Only organizations can delete campaigns
    db: Session = Depends(get_db)
):
    """Delete campaign - ORGANIZATIONS ONLY (creator or admin, no donations)"""
    try:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        
        if not campaign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign not found"
            )
        
        # Check permissions (creator or admin)
        if campaign.creator_id != current_user.id and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )
        
        # Check if campaign has donations
        donation_count = db.query(Donation).filter(Donation.campaign_id == campaign_id).count()
        if donation_count > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete campaign with donations"
            )
        
        db.delete(campaign)
        db.commit()
        
        logger.info(f"Campaign {campaign_id} deleted by user {current_user.id}")
        
        return {"message": "Campaign deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete campaign error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete campaign"
        )

@campaign_router.post("/{campaign_id}/submit")
async def submit_campaign(
    campaign_id: str,
    current_user: User = Depends(get_current_user_organization),  # CHANGED: Only organizations can submit campaigns
    db: Session = Depends(get_db)
):
    """Submit campaign for review - ORGANIZATIONS ONLY"""
    try:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        
        if not campaign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign not found"
            )
        
        # Check permissions
        if campaign.creator_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )
        
        # Check if campaign is in draft status
        if campaign.status != CampaignStatus.DRAFT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only draft campaigns can be submitted"
            )
        
        # Update status
        campaign.status = CampaignStatus.PENDING
        campaign.updated_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Campaign {campaign_id} submitted for review by user {current_user.id}")
        
        return {"message": "Campaign submitted for review"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Submit campaign error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit campaign"
        )

# MODIFIED: Donation endpoints with role-based access control

@campaign_router.post("/{campaign_id}/donate")
@limiter.limit("20/hour")
async def donate_to_campaign(
    request: Request,
    campaign_id: str,
    donation_data: DonationRequest,
    current_user: User = Depends(get_current_user_individual),  # CHANGED: Only individuals can donate
    db: Session = Depends(get_db)
):
    """Make a donation to campaign - INDIVIDUALS ONLY"""
    try:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        
        if not campaign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign not found"
            )
        
        # Check if campaign accepts donations
        if campaign.status != CampaignStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Campaign is not accepting donations"
            )
        
        # Create donation record
        donation = Donation(
            amount=donation_data.amount,
            donor_id=current_user.id,
            campaign_id=campaign_id,
            is_anonymous=donation_data.is_anonymous,
            message=donation_data.message,
            dedication=donation_data.dedication,
            tax_receipt_required=donation_data.tax_receipt_required
        )
        
        db.add(donation)
        
        # Update campaign totals
        campaign.current_amount += donation_data.amount
        campaign.donor_count += 1
        campaign.progress_percentage = (campaign.current_amount / campaign.goal_amount) * 100
        campaign.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(donation)
        
        logger.info(f"Donation of {donation_data.amount} made to campaign {campaign_id} by individual {current_user.id}")
        
        return {
            "message": "Donation successful",
            "donation_id": donation.id,
            "amount": donation.amount,
            "campaign_progress": campaign.progress_percentage
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Donation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process donation"
        )

@campaign_router.get("/{campaign_id}/donations")
async def get_campaign_donations(
    campaign_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get campaign donations - PUBLIC ACCESS (respects anonymity)"""
    try:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        
        if not campaign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign not found"
            )
        
        # Get donations
        donations_query = db.query(Donation).filter(Donation.campaign_id == campaign_id)
        total = donations_query.count()
        donations = donations_query.offset(skip).limit(limit).all()
        
        # Format donations (respect anonymity)
        formatted_donations = []
        for donation in donations:
            donation_data = {
                "id": donation.id,
                "amount": donation.amount,
                "is_anonymous": donation.is_anonymous,
                "message": donation.message,
                "dedication": donation.dedication,
                "created_at": donation.created_at
            }
            
            if not donation.is_anonymous and donation.donor:
                donation_data["donor"] = {
                    "full_name": donation.donor.full_name
                }
            
            formatted_donations.append(donation_data)
        
        return {
            "donations": formatted_donations,
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get donations error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve donations"
        )

# Admin endpoints for campaign management

@campaign_router.post("/{campaign_id}/approve")
async def approve_campaign(
    campaign_id: str,
    current_user: User = Depends(get_current_user_moderator),  # Moderator or admin access
    db: Session = Depends(get_db)
):
    """Approve campaign - MODERATOR/ADMIN ONLY"""
    try:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        
        if not campaign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign not found"
            )
        
        if campaign.status != CampaignStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only pending campaigns can be approved"
            )
        
        campaign.status = CampaignStatus.ACTIVE
        campaign.updated_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Campaign {campaign_id} approved by {current_user.role.value} {current_user.id}")
        
        return {"message": "Campaign approved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Approve campaign error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to approve campaign"
        )

@campaign_router.post("/{campaign_id}/reject")
async def reject_campaign(
    campaign_id: str,
    reason: str,
    current_user: User = Depends(get_current_user_moderator),  # Moderator or admin access
    db: Session = Depends(get_db)
):
    """Reject campaign - MODERATOR/ADMIN ONLY"""
    try:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        
        if not campaign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign not found"
            )
        
        if campaign.status != CampaignStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only pending campaigns can be rejected"
            )
        
        campaign.status = CampaignStatus.CANCELLED
        campaign.updated_at = datetime.utcnow()
        # You might want to add a rejection_reason field to the Campaign model
        db.commit()
        
        logger.info(f"Campaign {campaign_id} rejected by {current_user.role.value} {current_user.id}: {reason}")
        
        return {"message": "Campaign rejected successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reject campaign error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reject campaign"
        )

# Health check endpoint
@campaign_router.get("/health")
async def campaign_health():
    """Health check for campaign service"""
    return {"status": "healthy", "service": "campaigns"}

# Export router
__all__ = ["campaign_router"]

