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
    has_complete_documentation = Column(Boolean, default=True)
    has_clear_beneficiary = Column(Boolean, default=True)
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
    investigator_id = Column(String(100))
    
    __table_args__ = (
        Index('idx_report_status_time', 'status', 'reported_at'),
    )

class FraudDetectionStats(Base):
    """
    Aggregated statistics for fraud detection performance
    """
    __tablename__ = "fraud_detection_stats"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Time period
    date = Column(DateTime, nullable=False, index=True)
    period_type = Column(String(20), nullable=False)  # daily, weekly, monthly
    
    # Overall statistics
    total_entities = Column(Integer, default=0)
    fraudulent_entities = Column(Integer, default=0)
    legitimate_entities = Column(Integer, default=0)
    fraud_rate = Column(Float, default=0.0)
    
    # Category statistics
    category_stats = Column(JSON)
    platform_stats = Column(JSON)
    risk_level_stats = Column(JSON)
    
    # Performance metrics
    avg_fraud_score = Column(Float, default=0.0)
    avg_confidence = Column(Float, default=0.0)
    processing_time_avg = Column(Float, default=0.0)
    
    # Model performance
    model_version = Column(String(50))
    accuracy_metrics = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_stats_date_period', 'date', 'period_type'),
    )

# Pydantic Models for API

class FraudEntityBase(BaseModel):
    """Base model for fraud detection entity"""
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=5000)
    category: CampaignCategory
    subcategory: Optional[str] = Field(None, max_length=100)
    platform: Platform
    organizer_name: str = Field(..., min_length=1, max_length=200)
    organizer_type: OrganizerType
    beneficiary: Optional[str] = Field(None, max_length=200)
    location_city: Optional[str] = Field(None, max_length=100)
    location_state: Optional[str] = Field(None, max_length=100)
    funds_required: float = Field(..., gt=0)
    funds_raised: Optional[float] = Field(0, ge=0)
    funding_percentage: Optional[float] = Field(0, ge=0, le=200)
    campaign_start_date: Optional[datetime] = None
    campaign_age_days: Optional[int] = Field(0, ge=0)

class FraudEntityCreate(FraudEntityBase):
    """Model for creating fraud detection entity"""
    # Verification features
    has_government_verification: Optional[bool] = False
    has_complete_documentation: Optional[bool] = True
    has_clear_beneficiary: Optional[bool] = True
    has_contact_info: Optional[bool] = True
    has_medical_verification: Optional[bool] = False
    has_regular_updates: Optional[bool] = False
    has_social_media_presence: Optional[bool] = False
    has_website: Optional[bool] = False
    has_media_coverage: Optional[bool] = False
    
    # Risk indicators
    is_new_organization: Optional[bool] = False
    has_unrealistic_goal: Optional[bool] = False
    has_duplicate_content: Optional[bool] = False
    limited_social_proof: Optional[bool] = False
    minimal_updates: Optional[bool] = False
    unclear_fund_usage: Optional[bool] = False
    no_previous_campaigns: Optional[bool] = False
    
    # Additional features
    features: Optional[Dict[str, Any]] = None

class FraudEntityResponse(FraudEntityBase):
    """Model for fraud detection entity response"""
    id: str
    is_fraudulent: bool
    fraud_score: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    verification_status: VerificationStatus
    
    # Verification features
    has_government_verification: bool
    has_complete_documentation: bool
    has_clear_beneficiary: bool
    has_contact_info: bool
    has_medical_verification: bool
    has_regular_updates: bool
    has_social_media_presence: bool
    has_website: bool
    has_media_coverage: bool
    
    # Risk indicators
    is_new_organization: bool
    has_unrealistic_goal: bool
    has_duplicate_content: bool
    limited_social_proof: bool
    minimal_updates: bool
    unclear_fund_usage: bool
    no_previous_campaigns: bool
    
    # Additional data
    features: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    last_analyzed: Optional[datetime] = None
    analysis_version: str
    
    class Config:
        from_attributes = True

class FraudAnalysisHistoryResponse(BaseModel):
    """Model for fraud analysis history response"""
    id: str
    entity_id: str
    fraud_score: float
    confidence: float
    risk_level: RiskLevel
    analysis_method: Optional[str] = None
    model_version: Optional[str] = None
    explanation: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    analyzed_at: datetime
    analyzer_id: Optional[str] = None
    
    class Config:
        from_attributes = True

class FraudReportCreate(BaseModel):
    """Model for creating fraud report"""
    entity_id: str
    fraud_type: str = Field(..., min_length=1, max_length=100)
    evidence: Optional[str] = Field(None, max_length=5000)
    reporter_info: Optional[Dict[str, Any]] = None

class FraudReportResponse(BaseModel):
    """Model for fraud report response"""
    id: str
    entity_id: str
    fraud_type: str
    evidence: Optional[str] = None
    reporter_info: Optional[Dict[str, Any]] = None
    status: FraudReportStatus
    investigation_notes: Optional[str] = None
    resolution: Optional[str] = None
    reported_at: datetime
    investigated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    investigator_id: Optional[str] = None
    
    class Config:
        from_attributes = True

class FraudStatsResponse(BaseModel):
    """Model for fraud detection statistics response"""
    id: str
    date: datetime
    period_type: str
    total_entities: int
    fraudulent_entities: int
    legitimate_entities: int
    fraud_rate: float
    category_stats: Optional[Dict[str, Any]] = None
    platform_stats: Optional[Dict[str, Any]] = None
    risk_level_stats: Optional[Dict[str, Any]] = None
    avg_fraud_score: float
    avg_confidence: float
    processing_time_avg: float
    model_version: Optional[str] = None
    accuracy_metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class DatabaseSummary(BaseModel):
    """Model for database summary"""
    total_entities: int
    fraudulent_entities: int
    legitimate_entities: int
    fraud_rate: float
    categories: Dict[str, Any]
    platforms: Dict[str, Any]
    risk_levels: Dict[str, Any]
    verification_status: Dict[str, Any]
    recent_activity: Dict[str, Any]
    model_performance: Dict[str, Any]

# Utility functions for model operations

def create_fraud_entity_from_dict(data: Dict[str, Any]) -> FraudDetectionEntity:
    """Create FraudDetectionEntity from dictionary data"""
    
    # Parse features if it's a JSON string
    features = data.get('features')
    if isinstance(features, str):
        try:
            features = json.loads(features)
        except json.JSONDecodeError:
            features = {}
    
    # Extract boolean features from features dict
    boolean_features = {}
    if isinstance(features, dict):
        boolean_feature_names = [
            'has_government_verification', 'has_complete_documentation',
            'has_clear_beneficiary', 'has_contact_info', 'has_medical_verification',
            'has_regular_updates', 'has_social_media_presence', 'has_website',
            'has_media_coverage', 'is_new_organization', 'has_unrealistic_goal',
            'has_duplicate_content', 'limited_social_proof', 'minimal_updates',
            'unclear_fund_usage', 'no_previous_campaigns'
        ]
        
        for feature_name in boolean_feature_names:
            boolean_features[feature_name] = features.get(feature_name, False)
    
    # Create entity
    entity = FraudDetectionEntity(
        id=data.get('id', str(uuid.uuid4())),
        title=data.get('title', ''),
        description=data.get('description'),
        category=CampaignCategory(data.get('category', 'Unknown')),
        subcategory=data.get('subcategory'),
        platform=Platform(data.get('platform', 'Unknown')),
        organizer_name=data.get('organizer_name', ''),
        organizer_type=OrganizerType(data.get('organizer_type', 'individual')),
        beneficiary=data.get('beneficiary'),
        location_city=data.get('location_city'),
        location_state=data.get('location_state'),
        funds_required=float(data.get('funds_required', 0)),
        funds_raised=float(data.get('funds_raised', 0)),
        funding_percentage=float(data.get('funding_percentage', 0)),
        campaign_age_days=int(data.get('campaign_age_days', 0)),
        is_fraudulent=bool(data.get('is_fraudulent', False)),
        fraud_score=float(data.get('fraud_score', 0.0)),
        risk_level=RiskLevel(data.get('risk_level', 'medium')),
        verification_status=VerificationStatus(data.get('verification_status', 'pending')),
        features=features,
        **boolean_features
    )
    
    # Parse campaign start date
    if 'campaign_start_date' in data and data['campaign_start_date']:
        try:
            if isinstance(data['campaign_start_date'], str):
                entity.campaign_start_date = datetime.strptime(data['campaign_start_date'], '%Y-%m-%d')
            elif isinstance(data['campaign_start_date'], datetime):
                entity.campaign_start_date = data['campaign_start_date']
        except ValueError:
            pass
    
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

