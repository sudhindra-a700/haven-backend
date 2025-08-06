"""
Registration Routes for HAVEN Crowdfunding Platform
Handles individual and organization registration with role-based access control
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, validator, EmailStr
from typing import Optional, Dict, Any
from datetime import datetime
import logging

# Import dependencies
try:
    from database import get_db
    from models import (
        User, UserRole, IndividualRegistration, OrganizationRegistration,
        OrganizerType, VerificationStatus, user_to_dict,
        individual_registration_to_dict, organization_registration_to_dict
    )
    from auth_middleware import (
        get_current_user, create_access_token, hash_password
    )
except ImportError as e:
    logging.error(f"Import error in registration_routes: {e}")
    raise

logger = logging.getLogger(__name__)

# Create router
registration_router = APIRouter(prefix="/auth", tags=["registration"])

# Pydantic models for registration requests

class IndividualRegistrationRequest(BaseModel):
    """Request model for individual registration"""
    full_name: str = Field(..., min_length=2, max_length=255)
    email: EmailStr
    password: Optional[str] = Field(None, min_length=8)
    phone_number: Optional[str] = Field(None, max_length=20)
    date_of_birth: Optional[datetime] = None
    
    # Address Information
    address_line1: Optional[str] = Field(None, max_length=255)
    address_line2: Optional[str] = Field(None, max_length=255)
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=100)
    postal_code: Optional[str] = Field(None, max_length=20)
    country: Optional[str] = Field(None, max_length=100)
    
    # Identity Verification
    id_type: Optional[str] = Field(None, max_length=50)
    id_number: Optional[str] = Field(None, max_length=100)
    id_document_url: Optional[str] = Field(None, max_length=500)
    
    @validator('email')
    def validate_email(cls, v):
        if not v or '@' not in v:
            raise ValueError('Valid email address required')
        return v.lower()
    
    @validator('phone_number')
    def validate_phone(cls, v):
        if v and not v.replace('+', '').replace('-', '').replace(' ', '').isdigit():
            raise ValueError('Invalid phone number format')
        return v

class OrganizationRegistrationRequest(BaseModel):
    """Request model for organization registration"""
    organization_name: str = Field(..., min_length=2, max_length=255)
    organization_type: OrganizerType
    email: EmailStr
    password: Optional[str] = Field(None, min_length=8)
    phone_number: Optional[str] = Field(None, max_length=20)
    website: Optional[str] = Field(None, max_length=255)
    
    # Address Information (Required for organizations)
    address_line1: str = Field(..., max_length=255)
    address_line2: Optional[str] = Field(None, max_length=255)
    city: str = Field(..., max_length=100)
    state: str = Field(..., max_length=100)
    postal_code: str = Field(..., max_length=20)
    country: str = Field(..., max_length=100)
    
    # Legal Information
    registration_number: Optional[str] = Field(None, max_length=100)
    tax_id: Optional[str] = Field(None, max_length=100)
    fcra_number: Optional[str] = Field(None, max_length=100)
    ngo_garpan_id: Optional[str] = Field(None, max_length=100)
    
    # Documents
    registration_certificate_url: Optional[str] = Field(None, max_length=500)
    tax_exemption_certificate_url: Optional[str] = Field(None, max_length=500)
    fcra_certificate_url: Optional[str] = Field(None, max_length=500)
    
    # Contact Person (Required)
    contact_person_name: str = Field(..., max_length=255)
    contact_person_designation: Optional[str] = Field(None, max_length=100)
    contact_person_phone: Optional[str] = Field(None, max_length=20)
    contact_person_email: Optional[EmailStr] = None
    
    @validator('email')
    def validate_email(cls, v):
        if not v or '@' not in v:
            raise ValueError('Valid email address required')
        return v.lower()
    
    @validator('website')
    def validate_website(cls, v):
        if v and not (v.startswith('http://') or v.startswith('https://')):
            return f'https://{v}'
        return v

class RegistrationStatusResponse(BaseModel):
    """Response model for registration status"""
    is_registered: bool
    role: Optional[str] = None
    registration_type: Optional[str] = None
    needs_registration: bool
    available_registration_types: list = ["individual", "organization"]

# Registration endpoints

@registration_router.post("/register/individual", response_model=Dict[str, Any])
async def register_individual(
    registration_data: IndividualRegistrationRequest,
    db: Session = Depends(get_db)
):
    """Register a new individual user"""
    try:
        # Check if email already exists
        existing_user = db.query(User).filter(User.email == registration_data.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Check if individual registration already exists
        existing_registration = db.query(IndividualRegistration).filter(
            IndividualRegistration.email == registration_data.email
        ).first()
        if existing_registration:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Individual registration already exists for this email"
            )
        
        # Create individual registration
        individual_registration = IndividualRegistration(
            full_name=registration_data.full_name,
            email=registration_data.email,
            phone_number=registration_data.phone_number,
            date_of_birth=registration_data.date_of_birth,
            address_line1=registration_data.address_line1,
            address_line2=registration_data.address_line2,
            city=registration_data.city,
            state=registration_data.state,
            postal_code=registration_data.postal_code,
            country=registration_data.country,
            id_type=registration_data.id_type,
            id_number=registration_data.id_number,
            id_document_url=registration_data.id_document_url,
            verification_status=VerificationStatus.PENDING
        )
        
        # Create user account
        user = User(
            email=registration_data.email,
            full_name=registration_data.full_name,
            role=UserRole.INDIVIDUAL,
            is_registered=True,
            registration_completed_at=datetime.utcnow(),
            phone_number=registration_data.phone_number
        )
        
        # Set password if provided
        if registration_data.password:
            user.hashed_password = hash_password(registration_data.password)
        
        # Save to database
        db.add(individual_registration)
        db.add(user)
        db.commit()
        
        # Link registration to user
        individual_registration.user_id = user.id
        db.commit()
        db.refresh(user)
        db.refresh(individual_registration)
        
        # Create access token
        token_data = {
            "sub": user.email,
            "user_id": user.id,
            "email": user.email,
            "role": user.role.value,
            "is_registered": user.is_registered
        }
        access_token = create_access_token(token_data)
        
        logger.info(f"Individual registration successful for {user.email}")
        
        return {
            "message": "Individual registration successful",
            "user": user_to_dict(user),
            "registration": individual_registration_to_dict(individual_registration),
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Individual registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed. Please try again."
        )

@registration_router.post("/register/organization", response_model=Dict[str, Any])
async def register_organization(
    registration_data: OrganizationRegistrationRequest,
    db: Session = Depends(get_db)
):
    """Register a new organization user"""
    try:
        # Check if email already exists
        existing_user = db.query(User).filter(User.email == registration_data.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Check if organization registration already exists
        existing_registration = db.query(OrganizationRegistration).filter(
            OrganizationRegistration.email == registration_data.email
        ).first()
        if existing_registration:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization registration already exists for this email"
            )
        
        # Create organization registration
        organization_registration = OrganizationRegistration(
            organization_name=registration_data.organization_name,
            organization_type=registration_data.organization_type,
            email=registration_data.email,
            phone_number=registration_data.phone_number,
            website=registration_data.website,
            address_line1=registration_data.address_line1,
            address_line2=registration_data.address_line2,
            city=registration_data.city,
            state=registration_data.state,
            postal_code=registration_data.postal_code,
            country=registration_data.country,
            registration_number=registration_data.registration_number,
            tax_id=registration_data.tax_id,
            fcra_number=registration_data.fcra_number,
            ngo_garpan_id=registration_data.ngo_garpan_id,
            registration_certificate_url=registration_data.registration_certificate_url,
            tax_exemption_certificate_url=registration_data.tax_exemption_certificate_url,
            fcra_certificate_url=registration_data.fcra_certificate_url,
            contact_person_name=registration_data.contact_person_name,
            contact_person_designation=registration_data.contact_person_designation,
            contact_person_phone=registration_data.contact_person_phone,
            contact_person_email=registration_data.contact_person_email,
            verification_status=VerificationStatus.PENDING
        )
        
        # Create user account
        user = User(
            email=registration_data.email,
            full_name=registration_data.organization_name,  # Use organization name as full name
            role=UserRole.ORGANIZATION,
            is_registered=True,
            registration_completed_at=datetime.utcnow(),
            phone_number=registration_data.phone_number
        )
        
        # Set password if provided
        if registration_data.password:
            user.hashed_password = hash_password(registration_data.password)
        
        # Save to database
        db.add(organization_registration)
        db.add(user)
        db.commit()
        
        # Link registration to user
        organization_registration.user_id = user.id
        db.commit()
        db.refresh(user)
        db.refresh(organization_registration)
        
        # Create access token
        token_data = {
            "sub": user.email,
            "user_id": user.id,
            "email": user.email,
            "role": user.role.value,
            "is_registered": user.is_registered
        }
        access_token = create_access_token(token_data)
        
        logger.info(f"Organization registration successful for {user.email}")
        
        return {
            "message": "Organization registration successful",
            "user": user_to_dict(user),
            "registration": organization_registration_to_dict(organization_registration),
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Organization registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed. Please try again."
        )

@registration_router.get("/registration-status", response_model=RegistrationStatusResponse)
async def get_registration_status(
    current_user: User = Depends(get_current_user)
):
    """Get current user's registration status"""
    try:
        registration_type = None
        if current_user.role == UserRole.INDIVIDUAL:
            registration_type = "individual"
        elif current_user.role == UserRole.ORGANIZATION:
            registration_type = "organization"
        
        return RegistrationStatusResponse(
            is_registered=current_user.is_registered,
            role=current_user.role.value if current_user.role else None,
            registration_type=registration_type,
            needs_registration=not current_user.is_registered,
            available_registration_types=["individual", "organization"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get registration status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get registration status"
        )

@registration_router.get("/registration/individual", response_model=Dict[str, Any])
async def get_individual_registration(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's individual registration details"""
    try:
        if current_user.role != UserRole.INDIVIDUAL:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only individuals can access this endpoint"
            )
        
        registration = db.query(IndividualRegistration).filter(
            IndividualRegistration.user_id == current_user.id
        ).first()
        
        if not registration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Individual registration not found"
            )
        
        return {
            "user": user_to_dict(current_user),
            "registration": individual_registration_to_dict(registration)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get individual registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get registration details"
        )

@registration_router.get("/registration/organization", response_model=Dict[str, Any])
async def get_organization_registration(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's organization registration details"""
    try:
        if current_user.role != UserRole.ORGANIZATION:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )
        
        registration = db.query(OrganizationRegistration).filter(
            OrganizationRegistration.user_id == current_user.id
        ).first()
        
        if not registration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization registration not found"
            )
        
        return {
            "user": user_to_dict(current_user),
            "registration": organization_registration_to_dict(registration)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get organization registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get registration details"
        )

# Health check endpoint
@registration_router.get("/registration/health")
async def registration_health():
    """Health check for registration service"""
    return {"status": "healthy", "service": "registration"}

# Export router
__all__ = ["registration_router"]

