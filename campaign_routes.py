"""
Campaign API Routes for HAVEN Crowdfunding Platform
RESTful endpoints for campaign management
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

from auth_middleware import get_current_user, require_admin
from database import get_db
from models import (
    User, Campaign, CampaignStatus, CampaignCategory, 
    Donation, CampaignUpdate, Comment
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
    id: int
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

class CommentRequest(BaseModel):
    content: str
    parent_id: Optional[int] = None
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Comment cannot be empty")
        if len(v) > 1000:
            raise ValueError("Comment too long (max 1000 characters)")
        return v.strip()

class CampaignUpdateCreateRequest(BaseModel):
    title: str
    content: str
    
    @validator('title')
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError("Update title cannot be empty")
        return v.strip()
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Update content cannot be empty")
        return v.strip()

# Campaign CRUD endpoints
@campaign_router.post("/", response_model=CampaignResponse)
@limiter.limit("10/hour")
async def create_campaign(
    request: Request,
    campaign_data: CampaignCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new campaign"""
    try:
        # Create campaign
        campaign = Campaign(
            title=campaign_data.title,
            description=campaign_data.description,
            short_description=campaign_data.short_description,
            goal_amount=campaign_data.goal_amount,
            category=CampaignCategory(campaign_data.category),
            organization_name=campaign_data.organization_name,
            ngo_darpan_id=campaign_data.ngo_darpan_id,
            fcra_number=campaign_data.fcra_number,
            end_date=campaign_data.end_date,
            creator_id=current_user.id,
            status=CampaignStatus.DRAFT,
            start_date=datetime.now()
        )
        
        # Generate slug
        import re
        slug = re.sub(r'[^a-zA-Z0-9\s]', '', campaign_data.title.lower())
        slug = re.sub(r'\s+', '-', slug)[:50]
        campaign.slug = f"{slug}-{int(datetime.now().timestamp())}"
        
        db.add(campaign)
        db.commit()
        db.refresh(campaign)
        
        logger.info(f"Campaign created by user {current_user.id}: {campaign.id}")
        
        return format_campaign_response(campaign)
        
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

@campaign_router.get("/", response_model=List[CampaignResponse])
@limiter.limit("100/minute")
async def get_campaigns(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    sort_by: str = Query("created_at", regex="^(created_at|goal_amount|current_amount|view_count)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    db: Session = Depends(get_db)
):
    """Get campaigns with filtering and pagination"""
    try:
        query = db.query(Campaign)
        
        # Apply filters
        if category:
            query = query.filter(Campaign.category == CampaignCategory(category))
        
        if status:
            query = query.filter(Campaign.status == CampaignStatus(status))
        else:
            # Default to active campaigns for public view
            query = query.filter(Campaign.status.in_([
                CampaignStatus.ACTIVE,
                CampaignStatus.COMPLETED
            ]))
        
        if search:
            search_term = f"%{search}%"
            query = query.filter(or_(
                Campaign.title.ilike(search_term),
                Campaign.description.ilike(search_term),
                Campaign.organization_name.ilike(search_term)
            ))
        
        # Apply sorting
        if sort_order == "desc":
            query = query.order_by(getattr(Campaign, sort_by).desc())
        else:
            query = query.order_by(getattr(Campaign, sort_by).asc())
        
        # Apply pagination
        campaigns = query.offset(skip).limit(limit).all()
        
        return [format_campaign_response(campaign) for campaign in campaigns]
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Get campaigns error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve campaigns"
        )

@campaign_router.get("/{campaign_id}", response_model=CampaignResponse)
@limiter.limit("200/minute")
async def get_campaign(
    request: Request,
    campaign_id: int,
    db: Session = Depends(get_db)
):
    """Get campaign by ID"""
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
        
        return format_campaign_response(campaign)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get campaign error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve campaign"
        )

@campaign_router.put("/{campaign_id}", response_model=CampaignResponse)
@limiter.limit("20/hour")
async def update_campaign(
    request: Request,
    campaign_id: int,
    campaign_data: CampaignUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update campaign"""
    try:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        
        if not campaign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign not found"
            )
        
        # Check permissions
        if campaign.creator_id != current_user.id and current_user.role.value != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )
        
        # Update fields
        update_data = campaign_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(campaign, field, value)
        
        campaign.updated_at = datetime.now()
        db.commit()
        db.refresh(campaign)
        
        logger.info(f"Campaign {campaign_id} updated by user {current_user.id}")
        
        return format_campaign_response(campaign)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update campaign error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update campaign"
        )

@campaign_router.delete("/{campaign_id}")
@limiter.limit("10/hour")
async def delete_campaign(
    request: Request,
    campaign_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete campaign"""
    try:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        
        if not campaign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign not found"
            )
        
        # Check permissions
        if campaign.creator_id != current_user.id and current_user.role.value != "admin":
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

# Campaign status management
@campaign_router.post("/{campaign_id}/submit")
@limiter.limit("10/hour")
async def submit_campaign(
    request: Request,
    campaign_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit campaign for review"""
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
        campaign.updated_at = datetime.now()
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

# Donation endpoints
@campaign_router.post("/{campaign_id}/donate")
@limiter.limit("20/hour")
async def donate_to_campaign(
    request: Request,
    campaign_id: int,
    donation_data: DonationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Make a donation to campaign"""
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
            tax_receipt_required=donation_data.tax_receipt_required,
            donor_name=None if donation_data.is_anonymous else current_user.full_name,
            donor_email=None if donation_data.is_anonymous else current_user.email
        )
        
        db.add(donation)
        
        # Update campaign totals
        campaign.current_amount += donation_data.amount
        campaign.donor_count += 1
        campaign.updated_at = datetime.now()
        
        db.commit()
        db.refresh(donation)
        
        logger.info(f"Donation of {donation_data.amount} made to campaign {campaign_id} by user {current_user.id}")
        
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
            detail="Donation failed"
        )

# Campaign updates
@campaign_router.post("/{campaign_id}/updates")
@limiter.limit("10/day")
async def create_campaign_update(
    request: Request,
    campaign_id: int,
    update_data: CampaignUpdateCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create campaign update"""
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
        
        # Create update
        campaign_update = CampaignUpdate(
            title=update_data.title,
            content=update_data.content,
            campaign_id=campaign_id
        )
        
        db.add(campaign_update)
        db.commit()
        db.refresh(campaign_update)
        
        logger.info(f"Campaign update created for campaign {campaign_id} by user {current_user.id}")
        
        return {
            "message": "Campaign update created",
            "update_id": campaign_update.id,
            "title": campaign_update.title
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Campaign update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create campaign update"
        )

# Comments
@campaign_router.post("/{campaign_id}/comments")
@limiter.limit("30/hour")
async def create_comment(
    request: Request,
    campaign_id: int,
    comment_data: CommentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create comment on campaign"""
    try:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        
        if not campaign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign not found"
            )
        
        # Create comment
        comment = Comment(
            content=comment_data.content,
            user_id=current_user.id,
            campaign_id=campaign_id,
            parent_id=comment_data.parent_id
        )
        
        db.add(comment)
        db.commit()
        db.refresh(comment)
        
        logger.info(f"Comment created on campaign {campaign_id} by user {current_user.id}")
        
        return {
            "message": "Comment created",
            "comment_id": comment.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comment creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create comment"
        )

# Helper functions
def format_campaign_response(campaign: Campaign) -> CampaignResponse:
    """Format campaign for API response"""
    return CampaignResponse(
        id=campaign.id,
        title=campaign.title,
        description=campaign.description,
        short_description=campaign.short_description,
        goal_amount=float(campaign.goal_amount),
        current_amount=float(campaign.current_amount),
        currency=campaign.currency,
        category=campaign.category.value,
        status=campaign.status.value,
        featured_image=campaign.featured_image,
        gallery_images=campaign.gallery_images,
        video_url=campaign.video_url,
        start_date=campaign.start_date,
        end_date=campaign.end_date,
        organization_name=campaign.organization_name,
        is_verified=campaign.is_verified,
        fraud_score=float(campaign.fraud_score) if campaign.fraud_score else None,
        view_count=campaign.view_count,
        donor_count=campaign.donor_count,
        progress_percentage=campaign.progress_percentage,
        creator={
            "id": campaign.creator.id,
            "full_name": campaign.creator.full_name,
            "profile_picture": campaign.creator.profile_picture,
            "is_verified": campaign.creator.is_verified
        },
        created_at=campaign.created_at,
        updated_at=campaign.updated_at
    )

