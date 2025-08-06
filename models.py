"""
Enhanced Database Models for HAVEN Platform
Updated to support separate individual and organization registration with role-based access control
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, Enum as SQLEnum, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
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

# Modified User Management Models with Role-Based Access Control

class UserRole(str, Enum):
    """User roles for authentication and authorization"""
    INDIVIDUAL = "individual"      # Can only donate
    ORGANIZATION = "organization"  # Can only create campaigns
    ADMIN = "admin"               # Full access
    MODERATOR = "moderator"       # Moderation access

class CampaignStatus(str, Enum):
    DRAFT = "draft"
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    UNDER_REVIEW = "under_review"

# New Registration Tables

class IndividualRegistration(Base):
    """Registration table for individual users who can only donate"""
    __tablename__ = "individual_registrations"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Personal Information
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone_number = Column(String(20), nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    
    # Address Information
    address_line1 = Column(String(255), nullable=True)
    address_line2 = Column(String(255), nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    postal_code = Column(String(20), nullable=True)
    country = Column(String(100), nullable=True)
    
    # Identity Verification
    id_type = Column(String(50), nullable=True)  # "aadhar", "pan", "passport", etc.
    id_number = Column(String(100), nullable=True)
    id_document_url = Column(String(500), nullable=True)
    
    # Account Status
    is_verified = Column(Boolean, default=False, nullable=False)
    verification_status = Column(SQLEnum(VerificationStatus), default=VerificationStatus.PENDING)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationship to User table
    user_id = Column(String, ForeignKey("users.id"), unique=True, nullable=True)
    user = relationship("User", back_populates="individual_registration")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_individual_email', 'email'),
        Index('idx_individual_verification', 'verification_status'),
        Index('idx_individual_created', 'created_at'),
    )

class OrganizationRegistration(Base):
    """Registration table for organizations who can only create campaigns"""
    __tablename__ = "organization_registrations"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Organization Information
    organization_name = Column(String(255), nullable=False)
    organization_type = Column(SQLEnum(OrganizerType), nullable=False)  # NGO, ORGANIZATION, GOVERNMENT
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone_number = Column(String(20), nullable=True)
    website = Column(String(255), nullable=True)
    
    # Address Information
    address_line1 = Column(String(255), nullable=False)
    address_line2 = Column(String(255), nullable=True)
    city = Column(String(100), nullable=False)
    state = Column(String(100), nullable=False)
    postal_code = Column(String(20), nullable=False)
    country = Column(String(100), nullable=False)
    
    # Legal Information
    registration_number = Column(String(100), nullable=True)
    tax_id = Column(String(100), nullable=True)
    fcra_number = Column(String(100), nullable=True)  # For NGOs
    ngo_garpan_id = Column(String(100), nullable=True)  # For NGOs
    
    # Documents
    registration_certificate_url = Column(String(500), nullable=True)
    tax_exemption_certificate_url = Column(String(500), nullable=True)
    fcra_certificate_url = Column(String(500), nullable=True)
    
    # Contact Person
    contact_person_name = Column(String(255), nullable=False)
    contact_person_designation = Column(String(100), nullable=True)
    contact_person_phone = Column(String(20), nullable=True)
    contact_person_email = Column(String(255), nullable=True)
    
    # Account Status
    is_verified = Column(Boolean, default=False, nullable=False)
    verification_status = Column(SQLEnum(VerificationStatus), default=VerificationStatus.PENDING)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationship to User table
    user_id = Column(String, ForeignKey("users.id"), unique=True, nullable=True)
    user = relationship("User", back_populates="organization_registration")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_organization_email', 'email'),
        Index('idx_organization_type', 'organization_type'),
        Index('idx_organization_verification', 'verification_status'),
        Index('idx_organization_created', 'created_at'),
    )

# Modified User Model

class User(Base):
    """
    User model for authentication and user management
    Supports OAuth authentication with Google and Facebook
    Enhanced with role-based access control and registration validation
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
    
    # Role and permissions - Modified to use new UserRole enum
    role = Column(SQLEnum(UserRole), nullable=False)  # INDIVIDUAL, ORGANIZATION, ADMIN, MODERATOR
    
    # Registration status - New fields
    is_registered = Column(Boolean, default=False, nullable=False)
    registration_completed_at = Column(DateTime, nullable=True)
    
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
    
    # Relationships to registration tables
    individual_registration = relationship("IndividualRegistration", back_populates="user", uselist=False)
    organization_registration = relationship("OrganizationRegistration", back_populates="user", uselist=False)
    
    # Existing relationships
    campaigns = relationship("Campaign", back_populates="creator")
    donations = relationship("Donation", back_populates="donor")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_google_id', 'google_id'),
        Index('idx_user_facebook_id', 'facebook_id'),
        Index('idx_user_created_at', 'created_at'),
        Index('idx_user_role', 'role'),
        Index('idx_user_registered', 'is_registered'),
    )

# Campaign Model (Updated with relationships)

class Campaign(Base):
    __tablename__ = "campaigns"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    short_description = Column(String(500), nullable=True)
    
    # Financial information
    goal_amount = Column(Float, nullable=False)
    current_amount = Column(Float, default=0.0)
    currency = Column(String(3), default="INR")
    
    # Category and status
    category = Column(SQLEnum(CampaignCategory), nullable=False)
    status = Column(SQLEnum(CampaignStatus), default=CampaignStatus.DRAFT)
    
    # Media
    featured_image = Column(String(500), nullable=True)
    gallery_images = Column(JSON, nullable=True)
    video_url = Column(String(500), nullable=True)
    
    # Timeline
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    
    # Organization details
    organization_name = Column(String(255), nullable=True)
    is_verified = Column(Boolean, default=False)
    
    # Fraud detection
    fraud_score = Column(Float, default=0.0)
    
    # Statistics
    view_count = Column(Integer, default=0)
    donor_count = Column(Integer, default=0)
    progress_percentage = Column(Float, default=0.0)
    
    # Creator relationship
    creator_id = Column(String, ForeignKey("users.id"), nullable=False)
    creator = relationship("User", back_populates="campaigns")
    
    # Donations relationship
    donations = relationship("Donation", back_populates="campaign")
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# Donation Model (Updated with relationships)

class Donation(Base):
    __tablename__ = "donations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    amount = Column(Float, nullable=False)
    is_anonymous = Column(Boolean, default=False)
    message = Column(Text, nullable=True)
    dedication = Column(String(255), nullable=True)
    tax_receipt_required = Column(Boolean, default=False)
    
    # Relationships
    donor_id = Column(String, ForeignKey("users.id"), nullable=False)
    campaign_id = Column(String, ForeignKey("campaigns.id"), nullable=False)
    
    donor = relationship("User", back_populates="donations")
    campaign = relationship("Campaign", back_populates="donations")
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())

# Keep existing fraud detection models unchanged

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

# Utility functions

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
        'is_registered': user.is_registered,
        'registration_completed_at': user.registration_completed_at.isoformat() if user.registration_completed_at else None,
        'created_at': user.created_at.isoformat() if user.created_at else None,
        'updated_at': user.updated_at.isoformat() if user.updated_at else None,
        'last_login': user.last_login.isoformat() if user.last_login else None,
        'phone_number': user.phone_number,
        'location': user.location,
        'bio': user.bio,
        'profile_public': user.profile_public,
        'email_notifications': user.email_notifications
    }

def individual_registration_to_dict(registration: IndividualRegistration) -> Dict[str, Any]:
    """Convert IndividualRegistration to dictionary for API responses"""
    return {
        'id': registration.id,
        'full_name': registration.full_name,
        'email': registration.email,
        'phone_number': registration.phone_number,
        'date_of_birth': registration.date_of_birth.isoformat() if registration.date_of_birth else None,
        'address_line1': registration.address_line1,
        'address_line2': registration.address_line2,
        'city': registration.city,
        'state': registration.state,
        'postal_code': registration.postal_code,
        'country': registration.country,
        'id_type': registration.id_type,
        'id_number': registration.id_number,
        'id_document_url': registration.id_document_url,
        'is_verified': registration.is_verified,
        'verification_status': registration.verification_status.value if registration.verification_status else None,
        'created_at': registration.created_at.isoformat() if registration.created_at else None,
        'updated_at': registration.updated_at.isoformat() if registration.updated_at else None
    }

def organization_registration_to_dict(registration: OrganizationRegistration) -> Dict[str, Any]:
    """Convert OrganizationRegistration to dictionary for API responses"""
    return {
        'id': registration.id,
        'organization_name': registration.organization_name,
        'organization_type': registration.organization_type.value if registration.organization_type else None,
        'email': registration.email,
        'phone_number': registration.phone_number,
        'website': registration.website,
        'address_line1': registration.address_line1,
        'address_line2': registration.address_line2,
        'city': registration.city,
        'state': registration.state,
        'postal_code': registration.postal_code,
        'country': registration.country,
        'registration_number': registration.registration_number,
        'tax_id': registration.tax_id,
        'fcra_number': registration.fcra_number,
        'ngo_garpan_id': registration.ngo_garpan_id,
        'registration_certificate_url': registration.registration_certificate_url,
        'tax_exemption_certificate_url': registration.tax_exemption_certificate_url,
        'fcra_certificate_url': registration.fcra_certificate_url,
        'contact_person_name': registration.contact_person_name,
        'contact_person_designation': registration.contact_person_designation,
        'contact_person_phone': registration.contact_person_phone,
        'contact_person_email': registration.contact_person_email,
        'is_verified': registration.is_verified,
        'verification_status': registration.verification_status.value if registration.verification_status else None,
        'created_at': registration.created_at.isoformat() if registration.created_at else None,
        'updated_at': registration.updated_at.isoformat() if registration.updated_at else None
    }

# Keep existing utility functions for fraud detection
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

