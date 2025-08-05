"""
Translation API Routes for HAVEN Crowdfunding Platform
RESTful endpoints for translation services
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Request, status
from pydantic import BaseModel, validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from translation_service import get_translation_service, TranslationResult
from auth_middleware import get_current_user
from models import User

logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create router
translation_router = APIRouter()

# Pydantic models
class TranslationRequest(BaseModel):
    text: str
    target_language: str
    source_language: str = "auto"
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        if len(v) > 5000:
            raise ValueError("Text too long (max 5000 characters)")
        return v.strip()
    
    @validator('target_language', 'source_language')
    def validate_language_code(cls, v):
        if v and len(v) not in [2, 4]:  # ISO 639-1 or 'auto'
            raise ValueError("Invalid language code")
        return v.lower()

class BatchTranslationRequest(BaseModel):
    texts: List[str]
    target_language: str
    source_language: str = "auto"
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError("Texts list cannot be empty")
        if len(v) > 50:
            raise ValueError("Too many texts (max 50)")
        for text in v:
            if len(text) > 1000:
                raise ValueError("Individual text too long (max 1000 characters)")
        return v

class LanguageDetectionRequest(BaseModel):
    text: str
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        if len(v) > 1000:
            raise ValueError("Text too long for detection (max 1000 characters)")
        return v.strip()

class TranslationResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str
    confidence_score: float
    method: str
    processing_time: float

class BatchTranslationResponse(BaseModel):
    translations: List[TranslationResponse]
    total_processing_time: float
    success_count: int
    error_count: int

class LanguageDetectionResponse(BaseModel):
    detected_language: str
    confidence_score: float

class SupportedLanguagesResponse(BaseModel):
    languages: Dict[str, str]
    total_count: int

# Translation endpoints
@translation_router.post("/translate", response_model=TranslationResponse)
@limiter.limit("30/minute")
async def translate_text(
    request: Request,
    translation_request: TranslationRequest,
    current_user: User = Depends(get_current_user)
):
    """Translate text to target language"""
    try:
        service = get_translation_service()
        
        result = await service.translate_text(
            text=translation_request.text,
            target_language=translation_request.target_language,
            source_language=translation_request.source_language
        )
        
        logger.info(f"Translation completed for user {current_user.id}: {translation_request.source_language} -> {translation_request.target_language}")
        
        return TranslationResponse(
            translated_text=result.translated_text,
            source_language=result.source_language,
            target_language=result.target_language,
            confidence_score=result.confidence_score,
            method=result.method,
            processing_time=result.processing_time
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Translation service temporarily unavailable"
        )

@translation_router.post("/translate/batch", response_model=BatchTranslationResponse)
@limiter.limit("10/minute")
async def translate_batch(
    request: Request,
    batch_request: BatchTranslationRequest,
    current_user: User = Depends(get_current_user)
):
    """Translate multiple texts in batch"""
    try:
        service = get_translation_service()
        
        translations = []
        error_count = 0
        total_start_time = datetime.now()
        
        for text in batch_request.texts:
            try:
                result = await service.translate_text(
                    text=text,
                    target_language=batch_request.target_language,
                    source_language=batch_request.source_language
                )
                
                translations.append(TranslationResponse(
                    translated_text=result.translated_text,
                    source_language=result.source_language,
                    target_language=result.target_language,
                    confidence_score=result.confidence_score,
                    method=result.method,
                    processing_time=result.processing_time
                ))
                
            except Exception as e:
                logger.warning(f"Batch translation error for text: {e}")
                error_count += 1
                # Add error placeholder
                translations.append(TranslationResponse(
                    translated_text=text,  # Return original text
                    source_language=batch_request.source_language,
                    target_language=batch_request.target_language,
                    confidence_score=0.0,
                    method="error",
                    processing_time=0.0
                ))
        
        total_processing_time = (datetime.now() - total_start_time).total_seconds()
        
        logger.info(f"Batch translation completed for user {current_user.id}: {len(translations)} texts")
        
        return BatchTranslationResponse(
            translations=translations,
            total_processing_time=total_processing_time,
            success_count=len(translations) - error_count,
            error_count=error_count
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Batch translation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch translation service temporarily unavailable"
        )

@translation_router.post("/detect-language", response_model=LanguageDetectionResponse)
@limiter.limit("60/minute")
async def detect_language(
    request: Request,
    detection_request: LanguageDetectionRequest,
    current_user: User = Depends(get_current_user)
):
    """Detect language of text"""
    try:
        service = get_translation_service()
        
        detected_lang = service.detect_language(detection_request.text)
        
        logger.info(f"Language detection completed for user {current_user.id}: detected {detected_lang}")
        
        return LanguageDetectionResponse(
            detected_language=detected_lang,
            confidence_score=0.8  # Default confidence
        )
        
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Language detection service temporarily unavailable"
        )

@translation_router.get("/languages", response_model=SupportedLanguagesResponse)
@limiter.limit("100/minute")
async def get_supported_languages(request: Request):
    """Get list of supported languages"""
    try:
        service = get_translation_service()
        languages = service.get_supported_languages()
        
        return SupportedLanguagesResponse(
            languages=languages,
            total_count=len(languages)
        )
        
    except Exception as e:
        logger.error(f"Get languages error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve supported languages"
        )

# Service management endpoints
@translation_router.get("/health")
@limiter.limit("30/minute")
async def get_translation_health(request: Request):
    """Get translation service health status"""
    try:
        service = get_translation_service()
        health_status = service.get_service_health()
        
        return {
            "service": "translation",
            "status": health_status["status"],
            "details": health_status
        }
        
    except Exception as e:
        logger.error(f"Translation health check error: {e}")
        return {
            "service": "translation",
            "status": "unhealthy",
            "error": str(e)
        }

@translation_router.get("/cache/stats")
@limiter.limit("10/minute")
async def get_cache_stats(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get translation cache statistics (admin only)"""
    try:
        if current_user.role.value != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        service = get_translation_service()
        cache_stats = service.get_cache_stats()
        
        return {
            "cache_statistics": cache_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve cache statistics"
        )

@translation_router.post("/cache/clear")
@limiter.limit("5/minute")
async def clear_translation_cache(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Clear translation cache (admin only)"""
    try:
        if current_user.role.value != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        service = get_translation_service()
        service.clear_cache()
        
        logger.info(f"Translation cache cleared by admin user {current_user.id}")
        
        return {
            "message": "Translation cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear translation cache"
        )

# Campaign-specific translation endpoints
@translation_router.post("/campaign/{campaign_id}/translate")
@limiter.limit("20/minute")
async def translate_campaign_content(
    request: Request,
    campaign_id: int,
    target_language: str,
    current_user: User = Depends(get_current_user)
):
    """Translate campaign content (title, description, updates)"""
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
        
        service = get_translation_service()
        
        # Translate campaign content
        translations = {}
        
        if campaign.title:
            title_result = await service.translate_text(
                text=campaign.title,
                target_language=target_language,
                source_language="auto"
            )
            translations["title"] = title_result.translated_text
        
        if campaign.description:
            desc_result = await service.translate_text(
                text=campaign.description,
                target_language=target_language,
                source_language="auto"
            )
            translations["description"] = desc_result.translated_text
        
        if campaign.short_description:
            short_desc_result = await service.translate_text(
                text=campaign.short_description,
                target_language=target_language,
                source_language="auto"
            )
            translations["short_description"] = short_desc_result.translated_text
        
        logger.info(f"Campaign {campaign_id} content translated to {target_language} for user {current_user.id}")
        
        return {
            "campaign_id": campaign_id,
            "target_language": target_language,
            "translations": translations,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Campaign translation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Campaign translation failed"
        )

# Import datetime for timestamps
from datetime import datetime

