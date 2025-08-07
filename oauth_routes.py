"""
Simple Registration Routes for HAVEN Crowdfunding Platform
Simplified version that works without database dependencies for testing
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime
import logging
import hashlib
import json
import os

logger = logging.getLogger(__name__)

# Create router
registration_router = APIRouter(tags=["registration"])

# Pydantic models for registration requests
class IndividualRegistrationRequest(BaseModel):
    """Request model for individual registration"""
    full_name: str = Field(..., min_length=2, max_length=255)
    email: EmailStr
    password: str = Field(..., min_length=8)
    phone_number: Optional[str] = Field(None, max_length=20)
    date_of_birth: Optional[str] = None
    
    # Address Information
    address_line1: Optional[str] = Field(None, max_length=255)
    address_line2: Optional[str] = Field(None, max_length=255)
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=100)
    postal_code: Optional[str] = Field(None, max_length=20)
    country: Optional[str] = Field(None, max_length=100)

class OrganizationRegistrationRequest(BaseModel):
    """Request model for organization registration"""
    organization_name: str = Field(..., min_length=2, max_length=255)
    organization_type: str = Field(..., max_length=100)
    organization_email: EmailStr
    password: str = Field(..., min_length=8)
    phone_number: Optional[str] = Field(None, max_length=20)
    website: Optional[str] = Field(None, max_length=255)
    
    # Address Information
    address_line1: str = Field(..., max_length=255)
    address_line2: Optional[str] = Field(None, max_length=255)
    city: str = Field(..., max_length=100)
    state: str = Field(..., max_length=100)
    postal_code: str = Field(..., max_length=20)
    country: str = Field(..., max_length=100)

# Simple file-based storage for testing
def save_registration(user_type: str, data: dict):
    """Save registration data to a simple JSON file"""
    try:
        file_path = f"/tmp/haven_{user_type}_registrations.json"
        
        # Load existing data
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                registrations = json.load(f)
        else:
            registrations = []
        
        # Add new registration
        data['registered_at'] = datetime.now().isoformat()
        data['user_id'] = hashlib.md5(data['email'].encode()).hexdigest()[:8]
        registrations.append(data)
        
        # Save back to file
        with open(file_path, 'w') as f:
            json.dump(registrations, f, indent=2)
        
        return data['user_id']
    except Exception as e:
        logger.error(f"Error saving registration: {e}")
        raise HTTPException(status_code=500, detail="Registration save failed")

def check_email_exists(email: str) -> bool:
    """Check if email already exists in any registration file"""
    for user_type in ['individual', 'organization']:
        file_path = f"/tmp/haven_{user_type}_registrations.json"
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    registrations = json.load(f)
                for reg in registrations:
                    if reg.get('email') == email or reg.get('organization_email') == email:
                        return True
            except:
                continue
    return False

@registration_router.post("/register/individual")
async def register_individual(request: IndividualRegistrationRequest):
    """Register a new individual user"""
    try:
        logger.info(f"Individual registration attempt for: {request.email}")
        
        # Check if email already exists
        if check_email_exists(request.email):
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
        
        # Prepare registration data
        registration_data = {
            "full_name": request.full_name,
            "email": request.email,
            "phone_number": request.phone_number,
            "date_of_birth": request.date_of_birth,
            "address_line1": request.address_line1,
            "address_line2": request.address_line2,
            "city": request.city,
            "state": request.state,
            "postal_code": request.postal_code,
            "country": request.country,
            "user_type": "individual",
            "role": "individual"
        }
        
        # Save registration
        user_id = save_registration("individual", registration_data)
        
        logger.info(f"Individual registration successful for: {request.email}")
        
        return JSONResponse(
            status_code=201,
            content={
                "message": "Individual registration successful",
                "user_id": user_id,
                "user_type": "individual",
                "role": "individual",
                "email": request.email,
                "full_name": request.full_name,
                "permissions": [
                    "donate_to_campaigns",
                    "view_campaigns", 
                    "manage_own_profile",
                    "view_donation_history"
                ],
                "next_steps": [
                    "Complete email verification",
                    "Set up profile preferences",
                    "Browse available campaigns"
                ]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Individual registration error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Registration failed due to server error"
        )

@registration_router.post("/register/organization")
async def register_organization(request: OrganizationRegistrationRequest):
    """Register a new organization"""
    try:
        logger.info(f"Organization registration attempt for: {request.organization_email}")
        
        # Check if email already exists
        if check_email_exists(request.organization_email):
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
        
        # Prepare registration data
        registration_data = {
            "organization_name": request.organization_name,
            "organization_type": request.organization_type,
            "organization_email": request.organization_email,
            "email": request.organization_email,  # For consistency
            "phone_number": request.phone_number,
            "website": request.website,
            "address_line1": request.address_line1,
            "address_line2": request.address_line2,
            "city": request.city,
            "state": request.state,
            "postal_code": request.postal_code,
            "country": request.country,
            "user_type": "organization",
            "role": "organization"
        }
        
        # Save registration
        user_id = save_registration("organization", registration_data)
        
        logger.info(f"Organization registration successful for: {request.organization_email}")
        
        return JSONResponse(
            status_code=201,
            content={
                "message": "Organization registration successful",
                "user_id": user_id,
                "user_type": "organization",
                "role": "organization",
                "email": request.organization_email,
                "organization_name": request.organization_name,
                "permissions": [
                    "create_campaigns",
                    "manage_own_campaigns",
                    "view_campaign_analytics",
                    "manage_own_profile"
                ],
                "next_steps": [
                    "Complete email verification",
                    "Submit organization verification documents",
                    "Create your first campaign"
                ]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Organization registration error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Registration failed due to server error"
        )

@registration_router.get("/registration-status/{email}")
async def get_registration_status(email: str):
    """Get registration status for an email"""
    try:
        for user_type in ['individual', 'organization']:
            file_path = f"/tmp/haven_{user_type}_registrations.json"
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    registrations = json.load(f)
                for reg in registrations:
                    if reg.get('email') == email or reg.get('organization_email') == email:
                        return {
                            "email": email,
                            "registered": True,
                            "user_type": reg.get('user_type'),
                            "role": reg.get('role'),
                            "registered_at": reg.get('registered_at')
                        }
        
        return {
            "email": email,
            "registered": False,
            "user_type": None,
            "role": None
        }
        
    except Exception as e:
        logger.error(f"Error checking registration status: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")

@registration_router.get("/health")
async def registration_health():
    """Health check for registration service"""
    return {
        "service": "registration",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": [
            "/register/individual",
            "/register/organization", 
            "/registration-status/{email}"
        ]
    }

