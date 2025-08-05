"""
Text Simplification API Routes for HAVEN Crowdfunding Platform
RESTful endpoints for text simplification services
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Request, status
from pydantic import BaseModel, validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from simplification_service import get_simplification_service, SimplificationResult
from auth_middleware import get_current_user
from models import User

logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create router
simplification_router = APIRouter()

# Pydantic models
class SimplificationRequest(BaseModel):
    text: str
    preserve_formatting: bool = True
    max_sentence_length: int = 20
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        if len(v) > 10000:
            raise ValueError("Text too long (max 10000 characters)")
        return v.strip()
    
    @validator('max_sentence_length')
    def validate_max_sentence_length(cls, v):
        if v < 5 or v > 50:
            raise ValueError("Max sentence length must be between 5 and 50")
        return v

class BatchSimplificationRequest(BaseModel):
    texts: List[str]
    preserve_formatting: bool = True
    max_sentence_length: int = 20
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError("Texts list cannot be empty")
        if len(v) > 20:
            raise ValueError("Too many texts (max 20)")
        for text in v:
            if len(text) > 5000:
                raise ValueError("Individual text too long (max 5000 characters)")
        return v

class TermExplanationRequest(BaseModel):
    term: str
    
    @validator('term')
    def validate_term(cls, v):
        if not v or not v.strip():
            raise ValueError("Term cannot be empty")
        if len(v) > 100:
            raise ValueError("Term too long (max 100 characters)")
        return v.strip()

class SimplificationResponse(BaseModel):
    original_text: str
    simplified_text: str
    simplifications: List[Dict[str, str]]
    confidence_score: float
    processing_time: float
    method: str

class BatchSimplificationResponse(BaseModel):
    results: List[SimplificationResponse]
    total_processing_time: float
    success_count: int
    error_count: int

class TermExplanationResponse(BaseModel):
    term: str
    explanation: Optional[str]
    category: Optional[str]
    found: bool

class SuggestionsResponse(BaseModel):
    suggestions: List[Dict[str, str]]
    total_count: int

class CategoriesResponse(BaseModel):
    categories: List[str]
    total_count: int

# Simplification endpoints
@simplification_router.post("/simplify", response_model=SimplificationResponse)
@limiter.limit("30/minute")
async def simplify_text(
    request: Request,
    simplification_request: SimplificationRequest,
    current_user: User = Depends(get_current_user)
):
    """Simplify text to make it easier to understand"""
    try:
        service = get_simplification_service()
        
        result = service.simplify_text(
            text=simplification_request.text,
            preserve_formatting=simplification_request.preserve_formatting,
            max_sentence_length=simplification_request.max_sentence_length
        )
        
        logger.info(f"Text simplification completed for user {current_user.id}: {len(result.simplifications)} terms simplified")
        
        return SimplificationResponse(
            original_text=result.original_text,
            simplified_text=result.simplified_text,
            simplifications=result.simplifications,
            confidence_score=result.confidence_score,
            processing_time=result.processing_time,
            method=result.method
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Simplification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Simplification service temporarily unavailable"
        )

@simplification_router.post("/simplify/batch", response_model=BatchSimplificationResponse)
@limiter.limit("10/minute")
async def simplify_batch(
    request: Request,
    batch_request: BatchSimplificationRequest,
    current_user: User = Depends(get_current_user)
):
    """Simplify multiple texts in batch"""
    try:
        service = get_simplification_service()
        
        results = []
        error_count = 0
        total_start_time = datetime.now()
        
        for text in batch_request.texts:
            try:
                result = service.simplify_text(
                    text=text,
                    preserve_formatting=batch_request.preserve_formatting,
                    max_sentence_length=batch_request.max_sentence_length
                )
                
                results.append(SimplificationResponse(
                    original_text=result.original_text,
                    simplified_text=result.simplified_text,
                    simplifications=result.simplifications,
                    confidence_score=result.confidence_score,
                    processing_time=result.processing_time,
                    method=result.method
                ))
                
            except Exception as e:
                logger.warning(f"Batch simplification error for text: {e}")
                error_count += 1
                # Add error placeholder
                results.append(SimplificationResponse(
                    original_text=text,
                    simplified_text=text,  # Return original text
                    simplifications=[],
                    confidence_score=0.0,
                    processing_time=0.0,
                    method="error"
                ))
        
        total_processing_time = (datetime.now() - total_start_time).total_seconds()
        
        logger.info(f"Batch simplification completed for user {current_user.id}: {len(results)} texts")
        
        return BatchSimplificationResponse(
            results=results,
            total_processing_time=total_processing_time,
            success_count=len(results) - error_count,
            error_count=error_count
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Batch simplification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch simplification service temporarily unavailable"
        )

@simplification_router.post("/explain-term", response_model=TermExplanationResponse)
@limiter.limit("60/minute")
async def explain_term(
    request: Request,
    term_request: TermExplanationRequest,
    current_user: User = Depends(get_current_user)
):
    """Get explanation for a specific term"""
    try:
        service = get_simplification_service()
        
        explanation = service.get_term_explanation(term_request.term)
        found = explanation is not None
        
        # Get category if explanation found
        category = None
        if found:
            category = service._get_term_category(term_request.term.lower())
        
        logger.info(f"Term explanation requested for user {current_user.id}: {term_request.term} ({'found' if found else 'not found'})")
        
        return TermExplanationResponse(
            term=term_request.term,
            explanation=explanation,
            category=category,
            found=found
        )
        
    except Exception as e:
        logger.error(f"Term explanation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Term explanation service temporarily unavailable"
        )

@simplification_router.post("/suggestions", response_model=SuggestionsResponse)
@limiter.limit("30/minute")
async def get_simplification_suggestions(
    request: Request,
    simplification_request: SimplificationRequest,
    limit: int = 10,
    current_user: User = Depends(get_current_user)
):
    """Get simplification suggestions for text"""
    try:
        service = get_simplification_service()
        
        suggestions = service.get_simplification_suggestions(
            text=simplification_request.text,
            limit=limit
        )
        
        logger.info(f"Simplification suggestions provided for user {current_user.id}: {len(suggestions)} suggestions")
        
        return SuggestionsResponse(
            suggestions=suggestions,
            total_count=len(suggestions)
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Suggestions error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Suggestions service temporarily unavailable"
        )

@simplification_router.get("/categories", response_model=CategoriesResponse)
@limiter.limit("100/minute")
async def get_supported_categories(request: Request):
    """Get list of supported simplification categories"""
    try:
        service = get_simplification_service()
        categories = service.get_supported_categories()
        
        return CategoriesResponse(
            categories=categories,
            total_count=len(categories)
        )
        
    except Exception as e:
        logger.error(f"Get categories error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve supported categories"
        )

# Service management endpoints
@simplification_router.get("/health")
@limiter.limit("30/minute")
async def get_simplification_health(request: Request):
    """Get simplification service health status"""
    try:
        service = get_simplification_service()
        health_status = service.get_service_health()
        
        return {
            "service": "simplification",
            "status": health_status["status"],
            "details": health_status
        }
        
    except Exception as e:
        logger.error(f"Simplification health check error: {e}")
        return {
            "service": "simplification",
            "status": "unhealthy",
            "error": str(e)
        }

# Campaign-specific simplification endpoints
@simplification_router.post("/campaign/{campaign_id}/simplify")
@limiter.limit("20/minute")
async def simplify_campaign_content(
    request: Request,
    campaign_id: int,
    current_user: User = Depends(get_current_user)
):
    """Simplify campaign content (title, description)"""
    try:
        from database import get_db
        from models import Campaign
        
        # Get campaign
        db = next(get_db())
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        
        if not campaign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign not found"
            )
        
        # Check permissions (campaign creator or admin)
        if campaign.creator_id != current_user.id and current_user.role.value != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )
        
        service = get_simplification_service()
        
        # Simplify campaign content
        simplifications = {}
        
        if campaign.title:
            title_result = service.simplify_text(campaign.title)
            simplifications["title"] = {
                "original": title_result.original_text,
                "simplified": title_result.simplified_text,
                "terms_simplified": len(title_result.simplifications)
            }
        
        if campaign.description:
            desc_result = service.simplify_text(campaign.description)
            simplifications["description"] = {
                "original": desc_result.original_text,
                "simplified": desc_result.simplified_text,
                "terms_simplified": len(desc_result.simplifications)
            }
        
        if campaign.short_description:
            short_desc_result = service.simplify_text(campaign.short_description)
            simplifications["short_description"] = {
                "original": short_desc_result.original_text,
                "simplified": short_desc_result.simplified_text,
                "terms_simplified": len(short_desc_result.simplifications)
            }
        
        logger.info(f"Campaign {campaign_id} content simplified for user {current_user.id}")
        
        return {
            "campaign_id": campaign_id,
            "simplifications": simplifications,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Campaign simplification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Campaign simplification failed"
        )

# Analytics endpoints
@simplification_router.get("/analytics/usage")
@limiter.limit("10/minute")
async def get_simplification_analytics(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get simplification usage analytics (admin only)"""
    try:
        if current_user.role.value != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        # TODO: Implement analytics tracking
        # For now, return mock data
        return {
            "total_simplifications": 1250,
            "unique_users": 85,
            "most_simplified_categories": [
                {"category": "financial", "count": 450},
                {"category": "legal", "count": 320},
                {"category": "technical", "count": 280},
                {"category": "medical", "count": 200}
            ],
            "average_confidence_score": 0.78,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simplification analytics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve simplification analytics"
        )

