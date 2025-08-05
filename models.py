"""
Database Models for HAVEN Crowdfunding Platform
SQLAlchemy models with proper relationships and constraints
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, 
    Numeric, ForeignKey, Enum, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
import enum

from database import Base

# Enums
class UserRole(enum.Enum):
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"

class CampaignStatus(enum.Enum):
    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class CampaignCategory(enum.Enum):
    EDUCATION = "education"
    HEALTH = "health"
    COMMUNITY = "community"
    TECHNOLOGY = "technology"
    ENVIRONMENT = "environment"
    ARTS = "arts"
    SPORTS = "sports"
    OTHER = "other"

class TransactionType(enum.Enum):
    DONATION = "donation"
    REFUND = "refund"
    WITHDRAWAL = "withdrawal"

class TransactionStatus(enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Base model with common fields
class BaseModel(Base):
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

# User model
class User(BaseModel):
    __tablename__ = "users"
    
    # Basic information
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=True)  # Nullable for OAuth users
    full_name = Column(String(255), nullable=False)
    phone_number = Column(String(20), nullable=True)
    
    # Profile information
    profile_picture = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    phone_verified = Column(Boolean, default=False, nullable=False)
    
    # Role and permissions
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    
    # OAuth information
    google_id = Column(String(100), nullable=True, unique=True)
    facebook_id = Column(String(100), nullable=True, unique=True)
    
    # KYC information
    pan_number = Column(String(20), nullable=True)
    aadhar_number = Column(String(20), nullable=True)
    kyc_verified = Column(Boolean, default=False, nullable=False)
    kyc_documents = Column(JSONB, nullable=True)
    
    # Relationships
    campaigns = relationship("Campaign", back_populates="creator", cascade="all, delete-orphan")
    donations = relationship("Donation", back_populates="donor", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="user", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_role', 'role'),
        Index('idx_user_active', 'is_active'),
    )
    
    @validates('email')
    def validate_email(self, key, email):
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            raise ValueError("Invalid email format")
        return email.lower()
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', role='{self.role.value}')>"

# Campaign model
class Campaign(BaseModel):
    __tablename__ = "campaigns"
    
    # Basic information
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    short_description = Column(String(500), nullable=True)
    
    # Financial information
    goal_amount = Column(Numeric(12, 2), nullable=False)
    current_amount = Column(Numeric(12, 2), default=0, nullable=False)
    currency = Column(String(3), default="INR", nullable=False)
    
    # Campaign details
    category = Column(Enum(CampaignCategory), nullable=False)
    status = Column(Enum(CampaignStatus), default=CampaignStatus.DRAFT, nullable=False)
    
    # Media
    featured_image = Column(String(500), nullable=True)
    gallery_images = Column(JSONB, nullable=True)  # Array of image URLs
    video_url = Column(String(500), nullable=True)
    
    # Timeline
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    
    # Organization details
    organization_name = Column(String(255), nullable=True)
    ngo_darpan_id = Column(String(50), nullable=True)
    fcra_number = Column(String(50), nullable=True)
    
    # Verification and moderation
    is_verified = Column(Boolean, default=False, nullable=False)
    verification_documents = Column(JSONB, nullable=True)
    moderation_notes = Column(Text, nullable=True)
    fraud_score = Column(Numeric(3, 2), nullable=True)  # 0.00 to 1.00
    
    # SEO and metadata
    slug = Column(String(255), unique=True, nullable=True)
    meta_description = Column(String(500), nullable=True)
    tags = Column(JSONB, nullable=True)  # Array of tags
    
    # Statistics
    view_count = Column(Integer, default=0, nullable=False)
    share_count = Column(Integer, default=0, nullable=False)
    donor_count = Column(Integer, default=0, nullable=False)
    
    # Foreign keys
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    creator = relationship("User", back_populates="campaigns")
    donations = relationship("Donation", back_populates="campaign", cascade="all, delete-orphan")
    updates = relationship("CampaignUpdate", back_populates="campaign", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="campaign", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_campaign_status', 'status'),
        Index('idx_campaign_category', 'category'),
        Index('idx_campaign_creator', 'creator_id'),
        Index('idx_campaign_dates', 'start_date', 'end_date'),
        Index('idx_campaign_slug', 'slug'),
    )
    
    @validates('goal_amount')
    def validate_goal_amount(self, key, amount):
        if amount <= 0:
            raise ValueError("Goal amount must be positive")
        return amount
    
    @property
    def progress_percentage(self):
        if self.goal_amount > 0:
            return min((self.current_amount / self.goal_amount) * 100, 100)
        return 0
    
    @property
    def is_active(self):
        return self.status == CampaignStatus.ACTIVE
    
    @property
    def is_completed(self):
        return self.status == CampaignStatus.COMPLETED or self.current_amount >= self.goal_amount
    
    def __repr__(self):
        return f"<Campaign(id={self.id}, title='{self.title}', status='{self.status.value}')>"

# Donation model
class Donation(BaseModel):
    __tablename__ = "donations"
    
    # Basic information
    amount = Column(Numeric(12, 2), nullable=False)
    currency = Column(String(3), default="INR", nullable=False)
    
    # Donor information
    donor_name = Column(String(255), nullable=True)  # For anonymous donations
    donor_email = Column(String(255), nullable=True)
    is_anonymous = Column(Boolean, default=False, nullable=False)
    
    # Payment information
    payment_id = Column(String(100), nullable=True)
    payment_method = Column(String(50), nullable=True)
    payment_status = Column(Enum(TransactionStatus), default=TransactionStatus.PENDING, nullable=False)
    
    # Message and dedication
    message = Column(Text, nullable=True)
    dedication = Column(String(255), nullable=True)
    
    # Tax and receipts
    tax_receipt_required = Column(Boolean, default=False, nullable=False)
    tax_receipt_url = Column(String(500), nullable=True)
    
    # Foreign keys
    donor_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Nullable for guest donations
    campaign_id = Column(Integer, ForeignKey("campaigns.id"), nullable=False)
    
    # Relationships
    donor = relationship("User", back_populates="donations")
    campaign = relationship("Campaign", back_populates="donations")
    
    # Indexes
    __table_args__ = (
        Index('idx_donation_campaign', 'campaign_id'),
        Index('idx_donation_donor', 'donor_id'),
        Index('idx_donation_status', 'payment_status'),
        Index('idx_donation_amount', 'amount'),
    )
    
    @validates('amount')
    def validate_amount(self, key, amount):
        if amount <= 0:
            raise ValueError("Donation amount must be positive")
        return amount
    
    def __repr__(self):
        return f"<Donation(id={self.id}, amount={self.amount}, campaign_id={self.campaign_id})>"

# Campaign Update model
class CampaignUpdate(BaseModel):
    __tablename__ = "campaign_updates"
    
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    images = Column(JSONB, nullable=True)  # Array of image URLs
    
    # Foreign keys
    campaign_id = Column(Integer, ForeignKey("campaigns.id"), nullable=False)
    
    # Relationships
    campaign = relationship("Campaign", back_populates="updates")
    
    # Indexes
    __table_args__ = (
        Index('idx_update_campaign', 'campaign_id'),
    )

# Comment model
class Comment(BaseModel):
    __tablename__ = "comments"
    
    content = Column(Text, nullable=False)
    is_approved = Column(Boolean, default=True, nullable=False)
    
    # Foreign keys
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"), nullable=False)
    parent_id = Column(Integer, ForeignKey("comments.id"), nullable=True)  # For replies
    
    # Relationships
    user = relationship("User", back_populates="comments")
    campaign = relationship("Campaign", back_populates="comments")
    replies = relationship("Comment", backref="parent", remote_side=[id])
    
    # Indexes
    __table_args__ = (
        Index('idx_comment_campaign', 'campaign_id'),
        Index('idx_comment_user', 'user_id'),
        Index('idx_comment_approved', 'is_approved'),
    )

# Transaction model for financial tracking
class Transaction(BaseModel):
    __tablename__ = "transactions"
    
    # Transaction details
    transaction_id = Column(String(100), unique=True, nullable=False)
    amount = Column(Numeric(12, 2), nullable=False)
    currency = Column(String(3), default="INR", nullable=False)
    transaction_type = Column(Enum(TransactionType), nullable=False)
    status = Column(Enum(TransactionStatus), default=TransactionStatus.PENDING, nullable=False)
    
    # Payment gateway information
    gateway_transaction_id = Column(String(100), nullable=True)
    gateway_name = Column(String(50), nullable=True)
    gateway_response = Column(JSONB, nullable=True)
    
    # References
    donation_id = Column(Integer, ForeignKey("donations.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"), nullable=True)
    
    # Relationships
    donation = relationship("Donation")
    user = relationship("User")
    campaign = relationship("Campaign")
    
    # Indexes
    __table_args__ = (
        Index('idx_transaction_id', 'transaction_id'),
        Index('idx_transaction_status', 'status'),
        Index('idx_transaction_type', 'transaction_type'),
    )

# Fraud detection log
class FraudDetectionLog(BaseModel):
    __tablename__ = "fraud_detection_logs"
    
    # Detection details
    campaign_id = Column(Integer, ForeignKey("campaigns.id"), nullable=False)
    fraud_score = Column(Numeric(3, 2), nullable=False)
    risk_level = Column(String(20), nullable=False)  # low, medium, high
    
    # Detection results
    detection_result = Column(JSONB, nullable=True)
    model_version = Column(String(50), nullable=True)
    
    # Action taken
    action_taken = Column(String(100), nullable=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Relationships
    campaign = relationship("Campaign")
    reviewer = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_fraud_campaign', 'campaign_id'),
        Index('idx_fraud_score', 'fraud_score'),
    )

# Translation cache
class TranslationCache(BaseModel):
    __tablename__ = "translation_cache"
    
    # Translation details
    source_text_hash = Column(String(64), nullable=False)  # MD5 hash of source text
    source_language = Column(String(5), nullable=False)
    target_language = Column(String(5), nullable=False)
    translated_text = Column(Text, nullable=False)
    
    # Quality metrics
    confidence_score = Column(Numeric(3, 2), nullable=True)
    model_version = Column(String(50), nullable=True)
    
    # Usage tracking
    usage_count = Column(Integer, default=1, nullable=False)
    last_used = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        UniqueConstraint('source_text_hash', 'source_language', 'target_language'),
        Index('idx_translation_hash', 'source_text_hash'),
        Index('idx_translation_languages', 'source_language', 'target_language'),
    )

