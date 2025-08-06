"""
Enhanced Database Models for HAVEN Platform
Updated to support the expanded fraud detection database with multi-category support
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, Enum as SQLEnum, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import uuid
import json

Base = declarative_base()

# Enums
class CampaignCategory(str, Enum):
    MEDICAL = "Medical"
    EDUCATION = "Education"
    DISASTER_RELIEF = "Disaster Relief"
    ANIMAL_WELFARE = "Animal Welfare"
    ENVIRONMENT = "Environment"
    COMMUNITY_DEVELOPMENT = "Community Development"
    TECHNOLOGY = "Technology"
    SOCIAL_CAUSES = "Social Causes"
    ARTS_CULTURE = "Arts & Culture"
    SPORTS = "Sports"
    UNKNOWN = "Unknown"

class OrganizerType(str, Enum):
    INDIVIDUAL = "individual"
    ORGANIZATION = "organization"
    NGO = "ngo"
    GOVERNMENT = "government"

class Platform(str, Enum):
    KETTO = "Ketto"
    MILAAP = "Milaap"
    IMPACTGURU = "ImpactGuru"
    GIVEINDIA = "GiveIndia"
    INDIADONATES = "INDIAdonates"
    OTHER = "Other"
    UNKNOWN = "Unknown"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    PENDING = "pending"
    REJECTED = "rejected"
    UNKNOWN = "unknown"

class FraudReportStatus(str, Enum):
    RECEIVED = "received"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"

# User Management Models

class UserRole(str, Enum):
    """User roles for authentication and authorization"""
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"

# SQLAlchemy Models

class FraudDetectionEntity(Base):
    """
    Enhanced fraud detection entity model for the expanded database
    Supports all categories and comprehensive fraud detection features
    """
    __tablename__ = "fraud_detection_entities"
    
    # Primary identification
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(500), nullable=False, index=True)
    description = Column(Text)
    
    # Category information
    category = Column(SQLEnum(CampaignCategory), nullable=False, index=True)
    subcategory = Column(String(100), index=True)
    platform = Column(SQLEnum(Platform), nullable=False, index=True)
    
    # Organizer information
    organizer_name = Column(String(200), nullable=False, index=True)
    organizer_type = Column(SQLEnum(OrganizerType), nullable=False, index=True)
    beneficiary = Column(String(200))
    
    # Location information
    location_city = Column(String(100), index=True)
    location_state = Column(String(100), index=True)
    
    # Financial information
    funds_required = Column(Float, nullable=False, index=True)
    funds_raised = Column(Float, default=0.0, index=True)
    funding_percentage = Column(Float, default=0.0, index=True)
    
    # Campaign timeline
    campaign_start_date = Column(DateTime, index=True)
    campaign_age_days = Column(Integer, default=0, index=True)
    
    # Fraud detection results
    is_fraudulent = Column(Boolean, nullable=False, index=True)
    fraud_score = Column(Float, default=0.0, index=True)
    risk_level = Column(SQLEnum(RiskLevel), default=RiskLevel.MEDIUM, index=True)
    verification_status = Column(SQLEnum(VerificationStatus), default=VerificationStatus.PENDING, index=True)
    
    # Verification features
    has_government_verification = Column(Boolean, default=False, index=True)
    has_complete_documentation = Column(Boolean, default=False)
    has_clear_beneficiary = Column(Boolean, default=False)
    has_contact_info = Column(Boolean, default=True)
    has_medical_verification = Column(Boolean, default=False)
    has_regular_updates = Column(Boolean, default=False)
    has_social_media_presence = Column(Boolean, default=False)
    has_website = Column(Boolean, default=False)
    has_media_coverage = Column(Boolean, default=False)
    
    # Risk indicators
    is_new_organization = Column(Boolean, default=False, index=True)
    has_unrealistic_goal = Column(Boolean, default=False, index=True)
    has_duplicate_content = Column(Boolean, default=False, index=True)
    limited_social_proof = Column(Boolean, default=False)
    minimal_updates = Column(Boolean, default=False)
    unclear_fund_usage = Column(Boolean, default=False)
    no_previous_campaigns = Column(Boolean, default=False)
    
    # Additional features (stored as JSON)
    features = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_analyzed = Column(DateTime, index=True)
    analysis_version = Column(String(50), default="2.0_expanded")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_category_risk', 'category', 'risk_level'),
        Index('idx_platform_fraud', 'platform', 'is_fraudulent'),
        Index('idx_organizer_verification', 'organizer_type', 'verification_status'),
        Index('idx_financial_range', 'funds_required', 'funding_percentage'),
        Index('idx_temporal_analysis', 'campaign_start_date', 'campaign_age_days'),
    )

class FraudAnalysisHistory(Base):
    """
    History of fraud analysis results for tracking changes over time
    """
    __tablename__ = "fraud_analysis_history"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    entity_id = Column(String, nullable=False, index=True)
    
    # Analysis results
    fraud_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    risk_level = Column(SQLEnum(RiskLevel), nullable=False)
    
    # Analysis details
    analysis_method = Column(String(100))  # rule-based, ml-based, hybrid
    model_version = Column(String(50))
    explanation = Column(JSON)
    recommendations = Column(JSON)
    
    # Metadata
    analyzed_at = Column(DateTime, default=func.now(), index=True)
    analyzer_id = Column(String(100))  # system, user_id, etc.
    
    __table_args__ = (
        Index('idx_entity_analysis_time', 'entity_id', 'analyzed_at'),
    )

class FraudReport(Base):
    """
    User-reported fraud cases
    """
    __tablename__ = "fraud_reports"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    entity_id = Column(String, nullable=False, index=True)
    
    # Report details
    fraud_type = Column(String(100), nullable=False)
    evidence = Column(Text)
    reporter_info = Column(JSON)
    
    # Status tracking
    status = Column(SQLEnum(FraudReportStatus), default=FraudReportStatus.RECEIVED, index=True)
    investigation_notes = Column(Text)
    resolution = Column(Text)
    
    # Metadata
    reported_at = Column(DateTime, default=func.now(), index=True)
    investigated_at = Column(DateTime)
    resolved_at = Column(DateTime)

class User(Base):
    """
    User model for authentication and user management
    Supports OAuth authentication with Google and Facebook
    """
    __tablename__ = "users"
    
    # Primary identification
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    
    # Profile information
    full_name = Column(String(255), nullable=True)
    profile_picture = Column(String(500), nullable=True)
    
    # Authentication
    hashed_password = Column(String(255), nullable=True)  # Nullable for OAuth-only users
    
    # OAuth provider IDs
    google_id = Column(String(255), nullable=True, unique=True)
    facebook_id = Column(String(255), nullable=True, unique=True)
    
    # Account status
    email_verified = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Role and permissions
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    # Additional profile fields
    phone_number = Column(String(20), nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    location = Column(String(255), nullable=True)
    bio = Column(Text, nullable=True)
    
    # Privacy settings
    profile_public = Column(Boolean, default=True, nullable=False)
    email_notifications = Column(Boolean, default=True, nullable=False)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_google_id', 'google_id'),
        Index('idx_user_facebook_id', 'facebook_id'),
        Index('idx_user_created_at', 'created_at'),
        Index('idx_user_role', 'role'),
    )

# Utility functions
def dict_to_entity(data: Dict[str, Any]) -> FraudDetectionEntity:
    """Convert dictionary to FraudDetectionEntity"""
    entity = FraudDetectionEntity()
    
    # Map basic fields
    for field in ['id', 'title', 'description', 'subcategory', 'organizer_name', 
                  'beneficiary', 'location_city', 'location_state', 'funds_required',
                  'funds_raised', 'funding_percentage', 'campaign_age_days',
                  'is_fraudulent', 'fraud_score', 'analysis_version']:
        if field in data and data[field] is not None:
            setattr(entity, field, data[field])
    
    # Map enum fields
    if 'category' in data and data['category']:
        try:
            entity.category = CampaignCategory(data['category'])
        except ValueError:
            entity.category = CampaignCategory.UNKNOWN
    
    if 'platform' in data and data['platform']:
        try:
            entity.platform = Platform(data['platform'])
        except ValueError:
            entity.platform = Platform.UNKNOWN
    
    if 'organizer_type' in data and data['organizer_type']:
        try:
            entity.organizer_type = OrganizerType(data['organizer_type'])
        except ValueError:
            entity.organizer_type = OrganizerType.INDIVIDUAL
    
    if 'risk_level' in data and data['risk_level']:
        try:
            entity.risk_level = RiskLevel(data['risk_level'])
        except ValueError:
            entity.risk_level = RiskLevel.MEDIUM
    
    if 'verification_status' in data and data['verification_status']:
        try:
            entity.verification_status = VerificationStatus(data['verification_status'])
        except ValueError:
            entity.verification_status = VerificationStatus.PENDING
    
    # Map boolean fields
    boolean_fields = [
        'has_government_verification', 'has_complete_documentation', 'has_clear_beneficiary',
        'has_contact_info', 'has_medical_verification', 'has_regular_updates',
        'has_social_media_presence', 'has_website', 'has_media_coverage',
        'is_new_organization', 'has_unrealistic_goal', 'has_duplicate_content',
        'limited_social_proof', 'minimal_updates', 'unclear_fund_usage', 'no_previous_campaigns'
    ]
    
    for field in boolean_fields:
        if field in data:
            setattr(entity, field, bool(data[field]))
    
    # Map datetime fields
    if 'campaign_start_date' in data and data['campaign_start_date']:
        if isinstance(data['campaign_start_date'], str):
            try:
                entity.campaign_start_date = datetime.fromisoformat(data['campaign_start_date'].replace('Z', '+00:00'))
            except ValueError:
                pass
        elif isinstance(data['campaign_start_date'], datetime):
            entity.campaign_start_date = data['campaign_start_date']
    
    # Map features (JSON field)
    if 'features' in data and data['features']:
        if isinstance(data['features'], dict):
            entity.features = data['features']
        elif isinstance(data['features'], str):
            try:
                entity.features = json.loads(data['features'])
            except json.JSONDecodeError:
                entity.features = {}
    
    return entity

def entity_to_dict(entity: FraudDetectionEntity) -> Dict[str, Any]:
    """Convert FraudDetectionEntity to dictionary"""
    return {
        'id': entity.id,
        'title': entity.title,
        'description': entity.description,
        'category': entity.category.value if entity.category else None,
        'subcategory': entity.subcategory,
        'platform': entity.platform.value if entity.platform else None,
        'organizer_name': entity.organizer_name,
        'organizer_type': entity.organizer_type.value if entity.organizer_type else None,
        'beneficiary': entity.beneficiary,
        'location_city': entity.location_city,
        'location_state': entity.location_state,
        'funds_required': entity.funds_required,
        'funds_raised': entity.funds_raised,
        'funding_percentage': entity.funding_percentage,
        'campaign_start_date': entity.campaign_start_date.isoformat() if entity.campaign_start_date else None,
        'campaign_age_days': entity.campaign_age_days,
        'is_fraudulent': entity.is_fraudulent,
        'fraud_score': entity.fraud_score,
        'risk_level': entity.risk_level.value if entity.risk_level else None,
        'verification_status': entity.verification_status.value if entity.verification_status else None,
        'has_government_verification': entity.has_government_verification,
        'has_complete_documentation': entity.has_complete_documentation,
        'has_clear_beneficiary': entity.has_clear_beneficiary,
        'has_contact_info': entity.has_contact_info,
        'has_medical_verification': entity.has_medical_verification,
        'has_regular_updates': entity.has_regular_updates,
        'has_social_media_presence': entity.has_social_media_presence,
        'has_website': entity.has_website,
        'has_media_coverage': entity.has_media_coverage,
        'is_new_organization': entity.is_new_organization,
        'has_unrealistic_goal': entity.has_unrealistic_goal,
        'has_duplicate_content': entity.has_duplicate_content,
        'limited_social_proof': entity.limited_social_proof,
        'minimal_updates': entity.minimal_updates,
        'unclear_fund_usage': entity.unclear_fund_usage,
        'no_previous_campaigns': entity.no_previous_campaigns,
        'features': entity.features,
        'created_at': entity.created_at.isoformat() if entity.created_at else None,
        'updated_at': entity.updated_at.isoformat() if entity.updated_at else None,
        'last_analyzed': entity.last_analyzed.isoformat() if entity.last_analyzed else None,
        'analysis_version': entity.analysis_version
    }

def user_to_dict(user: User) -> Dict[str, Any]:
    """Convert User to dictionary for API responses"""
    return {
        'id': user.id,
        'email': user.email,
        'full_name': user.full_name,
        'profile_picture': user.profile_picture,
        'email_verified': user.email_verified,
        'is_active': user.is_active,
        'role': user.role.value if user.role else None,
        'created_at': user.created_at.isoformat() if user.created_at else None,
        'updated_at': user.updated_at.isoformat() if user.updated_at else None,
        'last_login': user.last_login.isoformat() if user.last_login else None,
        'phone_number': user.phone_number,
        'location': user.location,
        'bio': user.bio,
        'profile_public': user.profile_public,
        'email_notifications': user.email_notifications
    }


# Additional Enums for Campaign Management
class CampaignStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    UNDER_REVIEW = "under_review"

# Campaign Model
class Campaign(Base):
    __tablename__ = "campaigns"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    short_description = Column(String(500))
    
    # Financial details
    goal_amount = Column(Float, nullable=False)
    current_amount = Column(Float, default=0.0)
    currency = Column(String(3), default="INR")
    
    # Campaign details
    category = Column(SQLEnum(CampaignCategory), nullable=False)
    status = Column(SQLEnum(CampaignStatus), default=CampaignStatus.DRAFT)
    
    # Relationships
    creator_id = Column(String, nullable=False, index=True)
    
    # Media
    featured_image = Column(String(500))
    images = Column(JSON)  # Array of image URLs
    video_url = Column(String(500))
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    
    # Campaign settings
    allow_anonymous_donations = Column(Boolean, default=True)
    show_donor_names = Column(Boolean, default=True)
    
    # Location
    location = Column(String(200))
    country = Column(String(100))
    
    # Social proof
    donor_count = Column(Integer, default=0)
    share_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0)
    
    # Verification
    is_verified = Column(Boolean, default=False)
    verification_documents = Column(JSON)

# Donation Model
class Donation(Base):
    __tablename__ = "donations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Financial details
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="INR")
    
    # Relationships
    campaign_id = Column(String, nullable=False, index=True)
    donor_id = Column(String, nullable=True, index=True)  # Nullable for anonymous donations
    
    # Donor information (for anonymous donations)
    donor_name = Column(String(100))
    donor_email = Column(String(100))
    
    # Payment details
    payment_method = Column(String(50))
    payment_id = Column(String(100))  # External payment gateway ID
    payment_status = Column(String(20), default="pending")
    
    # Message
    message = Column(Text)
    is_anonymous = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

# Campaign Update Model
class CampaignUpdate(Base):
    __tablename__ = "campaign_updates"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Relationships
    campaign_id = Column(String, nullable=False, index=True)
    author_id = Column(String, nullable=False, index=True)
    
    # Content
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    
    # Media
    images = Column(JSON)  # Array of image URLs
    video_url = Column(String(500))
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

# Comment Model
class Comment(Base):
    __tablename__ = "comments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Relationships
    campaign_id = Column(String, nullable=False, index=True)
    author_id = Column(String, nullable=False, index=True)
    
    # Content
    content = Column(Text, nullable=False)
    
    # Moderation
    is_approved = Column(Boolean, default=True)
    is_flagged = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

# Helper functions for new models
def campaign_to_dict(campaign: Campaign) -> Dict[str, Any]:
    """Convert Campaign to dictionary for API responses"""
    return {
        'id': campaign.id,
        'title': campaign.title,
        'description': campaign.description,
        'short_description': campaign.short_description,
        'goal_amount': campaign.goal_amount,
        'current_amount': campaign.current_amount,
        'currency': campaign.currency,
        'category': campaign.category.value if campaign.category else None,
        'status': campaign.status.value if campaign.status else None,
        'creator_id': campaign.creator_id,
        'featured_image': campaign.featured_image,
        'images': campaign.images,
        'video_url': campaign.video_url,
        'created_at': campaign.created_at.isoformat() if campaign.created_at else None,
        'updated_at': campaign.updated_at.isoformat() if campaign.updated_at else None,
        'start_date': campaign.start_date.isoformat() if campaign.start_date else None,
        'end_date': campaign.end_date.isoformat() if campaign.end_date else None,
        'allow_anonymous_donations': campaign.allow_anonymous_donations,
        'show_donor_names': campaign.show_donor_names,
        'location': campaign.location,
        'country': campaign.country,
        'donor_count': campaign.donor_count,
        'share_count': campaign.share_count,
        'view_count': campaign.view_count,
        'is_verified': campaign.is_verified,
        'verification_documents': campaign.verification_documents
    }

def donation_to_dict(donation: Donation) -> Dict[str, Any]:
    """Convert Donation to dictionary for API responses"""
    return {
        'id': donation.id,
        'amount': donation.amount,
        'currency': donation.currency,
        'campaign_id': donation.campaign_id,
        'donor_id': donation.donor_id,
        'donor_name': donation.donor_name,
        'donor_email': donation.donor_email,
        'payment_method': donation.payment_method,
        'payment_id': donation.payment_id,
        'payment_status': donation.payment_status,
        'message': donation.message,
        'is_anonymous': donation.is_anonymous,
        'created_at': donation.created_at.isoformat() if donation.created_at else None,
        'updated_at': donation.updated_at.isoformat() if donation.updated_at else None
    }

