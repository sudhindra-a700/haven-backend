"""
Fraud Detection API Routes for HAVEN Crowdfunding Platform
RESTful endpoints for fraud detection services
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Request, status
from pydantic import BaseModel, validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from fraud_detection_service import get_fraud_detection_service, FraudPrediction
from auth_middleware import get_current_user, require_admin
from database import get_db
from models import User, Campaign, FraudDetectionLog

logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create router
fraud_router = APIRouter()

# Pydantic models
class FraudAnalysisRequest(BaseModel):
    campaign_data: Dict[str, Any]
    
    @validator('campaign_data')
    def validate_campaign_data(cls, v):
        required_fields = ['title', 'description', 'goal_amount']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        return v

class FraudAnalysisResponse(BaseModel):
    fraud_score: float
    risk_level: str
    confidence: float
    explanation: str
    features_used: List[str]
    model_version: str
    analysis_timestamp: str
    recommendations: List[str]

class BulkFraudAnalysisRequest(BaseModel):
    campaign_ids: List[int]
    
    @validator('campaign_ids')
    def validate_campaign_ids(cls, v):
        if not v:
            raise ValueError("Campaign IDs list cannot be empty")
        if len(v) > 100:
            raise ValueError("Too many campaigns (max 100)")
        return v

class BulkFraudAnalysisResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_analyzed: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    processing_time: float

class FraudReportRequest(BaseModel):
    campaign_id: int
    reason: str
    evidence: Optional[str] = None
    
    @validator('reason')
    def validate_reason(cls, v):
        if not v or not v.strip():
            raise ValueError("Reason cannot be empty")
        return v.strip()

# Fraud analysis endpoints
@fraud_router.post("/analyze", response_model=FraudAnalysisResponse)
@limiter.limit("20/minute")
async def analyze_campaign_fraud(
    request: Request,
    analysis_request: FraudAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyze campaign for fraud indicators"""
    try:
        service = get_fraud_detection_service()
        
        # Add user context to campaign data
        campaign_data = analysis_request.campaign_data.copy()
        campaign_data['creator'] = {
            'id': current_user.id,
            'email_verified': current_user.email_verified,
            'phone_verified': current_user.phone_verified,
            'profile_picture': current_user.profile_picture,
            'account_age_days': (datetime.now() - current_user.created_at).days,
            'campaigns_count': db.query(Campaign).filter(Campaign.creator_id == current_user.id).count(),
            'success_rate': 0.8  # TODO: Calculate actual success rate
        }
        
        # Perform fraud analysis
        prediction = service.predict_fraud(campaign_data)
        
        # Generate recommendations
        recommendations = generate_fraud_recommendations(prediction)
        
        # Log analysis if campaign exists
        campaign_id = campaign_data.get('id')
        if campaign_id:
            log_fraud_analysis(db, campaign_id, prediction, current_user.id)
        
        logger.info(f"Fraud analysis completed for user {current_user.id}: risk level {prediction.risk_level}")
        
        return FraudAnalysisResponse(
            fraud_score=prediction.fraud_score,
            risk_level=prediction.risk_level,
            confidence=prediction.confidence,
            explanation=prediction.explanation,
            features_used=prediction.features_used,
            model_version=prediction.model_version,
            analysis_timestamp=datetime.now().isoformat(),
            recommendations=recommendations
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Fraud analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fraud analysis service temporarily unavailable"
        )

@fraud_router.post("/analyze/campaign/{campaign_id}", response_model=FraudAnalysisResponse)
@limiter.limit("15/minute")
async def analyze_existing_campaign(
    request: Request,
    campaign_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyze existing campaign for fraud"""
    try:
        # Get campaign
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        if not campaign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign not found"
            )
        
        # Check permissions (campaign creator, admin, or moderator)
        if (campaign.creator_id != current_user.id and 
            current_user.role.value not in ["admin", "moderator"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )
        
        service = get_fraud_detection_service()
        
        # Prepare campaign data
        campaign_data = {
            'id': campaign.id,
            'title': campaign.title,
            'description': campaign.description,
            'goal_amount': float(campaign.goal_amount),
            'category': campaign.category.value,
            'ngo_darpan_id': campaign.ngo_darpan_id,
            'pan_number': campaign.creator.pan_number if campaign.creator else None,
            'images': campaign.gallery_images or [],
            'video_url': campaign.video_url,
            'creator': {
                'id': campaign.creator.id,
                'email_verified': campaign.creator.email_verified,
                'phone_verified': campaign.creator.phone_verified,
                'profile_picture': campaign.creator.profile_picture,
                'account_age_days': (datetime.now() - campaign.creator.created_at).days,
                'campaigns_count': db.query(Campaign).filter(Campaign.creator_id == campaign.creator_id).count(),
                'success_rate': 0.8  # TODO: Calculate actual success rate
            }
        }
        
        # Perform fraud analysis
        prediction = service.predict_fraud(campaign_data)
        
        # Update campaign fraud score
        campaign.fraud_score = prediction.fraud_score
        db.commit()
        
        # Log analysis
        log_fraud_analysis(db, campaign_id, prediction, current_user.id)
        
        # Generate recommendations
        recommendations = generate_fraud_recommendations(prediction)
        
        logger.info(f"Existing campaign {campaign_id} analyzed: risk level {prediction.risk_level}")
        
        return FraudAnalysisResponse(
            fraud_score=prediction.fraud_score,
            risk_level=prediction.risk_level,
            confidence=prediction.confidence,
            explanation=prediction.explanation,
            features_used=prediction.features_used,
            model_version=prediction.model_version,
            analysis_timestamp=datetime.now().isoformat(),
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Campaign fraud analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Campaign fraud analysis failed"
        )

@fraud_router.post("/analyze/bulk", response_model=BulkFraudAnalysisResponse)
@limiter.limit("5/minute")
async def bulk_fraud_analysis(
    request: Request,
    bulk_request: BulkFraudAnalysisRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Bulk fraud analysis for multiple campaigns (admin only)"""
    try:
        start_time = datetime.now()
        service = get_fraud_detection_service()
        
        results = []
        risk_counts = {"high": 0, "medium": 0, "low": 0}
        
        for campaign_id in bulk_request.campaign_ids:
            try:
                # Get campaign
                campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
                if not campaign:
                    results.append({
                        "campaign_id": campaign_id,
                        "error": "Campaign not found",
                        "fraud_score": None,
                        "risk_level": None
                    })
                    continue
                
                # Prepare campaign data
                campaign_data = {
                    'id': campaign.id,
                    'title': campaign.title,
                    'description': campaign.description,
                    'goal_amount': float(campaign.goal_amount),
                    'category': campaign.category.value,
                    'creator': {
                        'id': campaign.creator.id,
                        'email_verified': campaign.creator.email_verified,
                        'phone_verified': campaign.creator.phone_verified,
                        'account_age_days': (datetime.now() - campaign.creator.created_at).days,
                        'campaigns_count': db.query(Campaign).filter(Campaign.creator_id == campaign.creator_id).count()
                    }
                }
                
                # Analyze
                prediction = service.predict_fraud(campaign_data)
                
                # Update campaign
                campaign.fraud_score = prediction.fraud_score
                
                # Log analysis
                log_fraud_analysis(db, campaign_id, prediction, current_user.id)
                
                # Count risk levels
                risk_counts[prediction.risk_level] += 1
                
                results.append({
                    "campaign_id": campaign_id,
                    "fraud_score": prediction.fraud_score,
                    "risk_level": prediction.risk_level,
                    "confidence": prediction.confidence,
                    "explanation": prediction.explanation
                })
                
            except Exception as e:
                logger.warning(f"Bulk analysis error for campaign {campaign_id}: {e}")
                results.append({
                    "campaign_id": campaign_id,
                    "error": str(e),
                    "fraud_score": None,
                    "risk_level": None
                })
        
        # Commit all updates
        db.commit()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Bulk fraud analysis completed by admin {current_user.id}: {len(results)} campaigns")
        
        return BulkFraudAnalysisResponse(
            results=results,
            total_analyzed=len(results),
            high_risk_count=risk_counts["high"],
            medium_risk_count=risk_counts["medium"],
            low_risk_count=risk_counts["low"],
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk fraud analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bulk fraud analysis failed"
        )

# Fraud reporting endpoints
@fraud_router.post("/report")
@limiter.limit("10/minute")
async def report_fraud(
    request: Request,
    fraud_report: FraudReportRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Report a campaign as potentially fraudulent"""
    try:
        # Check if campaign exists
        campaign = db.query(Campaign).filter(Campaign.id == fraud_report.campaign_id).first()
        if not campaign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign not found"
            )
        
        # TODO: Create fraud report model and save report
        # For now, just log the report
        logger.warning(f"Fraud report submitted by user {current_user.id} for campaign {fraud_report.campaign_id}: {fraud_report.reason}")
        
        return {
            "message": "Fraud report submitted successfully",
            "report_id": f"FR_{fraud_report.campaign_id}_{current_user.id}_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fraud report error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit fraud report"
        )

# Fraud statistics and monitoring
@fraud_router.get("/stats")
@limiter.limit("30/minute")
async def get_fraud_stats(
    request: Request,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get fraud detection statistics (admin only)"""
    try:
        # Get campaign fraud statistics
        total_campaigns = db.query(Campaign).count()
        high_risk_campaigns = db.query(Campaign).filter(Campaign.fraud_score >= 0.7).count()
        medium_risk_campaigns = db.query(Campaign).filter(
            Campaign.fraud_score >= 0.3,
            Campaign.fraud_score < 0.7
        ).count()
        low_risk_campaigns = db.query(Campaign).filter(Campaign.fraud_score < 0.3).count()
        
        # Get recent fraud logs
        recent_logs = db.query(FraudDetectionLog).order_by(
            FraudDetectionLog.created_at.desc()
        ).limit(10).all()
        
        return {
            "total_campaigns": total_campaigns,
            "risk_distribution": {
                "high_risk": high_risk_campaigns,
                "medium_risk": medium_risk_campaigns,
                "low_risk": low_risk_campaigns,
                "unanalyzed": total_campaigns - (high_risk_campaigns + medium_risk_campaigns + low_risk_campaigns)
            },
            "recent_analyses": [
                {
                    "campaign_id": log.campaign_id,
                    "fraud_score": float(log.fraud_score),
                    "risk_level": log.risk_level,
                    "analyzed_at": log.created_at.isoformat()
                }
                for log in recent_logs
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Fraud stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve fraud statistics"
        )

@fraud_router.get("/health")
@limiter.limit("30/minute")
async def get_fraud_detection_health(request: Request):
    """Get fraud detection service health status"""
    try:
        service = get_fraud_detection_service()
        health_status = service.get_service_health()
        
        return {
            "service": "fraud_detection",
            "status": health_status["status"],
            "details": health_status
        }
        
    except Exception as e:
        logger.error(f"Fraud detection health check error: {e}")
        return {
            "service": "fraud_detection",
            "status": "unhealthy",
            "error": str(e)
        }

# Helper functions
def generate_fraud_recommendations(prediction: FraudPrediction) -> List[str]:
    """Generate recommendations based on fraud analysis"""
    recommendations = []
    
    if prediction.risk_level == "high":
        recommendations.extend([
            "Require additional verification documents",
            "Request video call with campaign creator",
            "Implement enhanced monitoring",
            "Consider manual review before approval"
        ])
    elif prediction.risk_level == "medium":
        recommendations.extend([
            "Request additional campaign details",
            "Verify contact information",
            "Monitor campaign progress closely",
            "Consider requesting references"
        ])
    else:  # low risk
        recommendations.extend([
            "Standard verification process",
            "Regular monitoring",
            "Encourage campaign updates"
        ])
    
    return recommendations

def log_fraud_analysis(
    db: Session,
    campaign_id: int,
    prediction: FraudPrediction,
    reviewer_id: int
):
    """Log fraud analysis to database"""
    try:
        fraud_log = FraudDetectionLog(
            campaign_id=campaign_id,
            fraud_score=prediction.fraud_score,
            risk_level=prediction.risk_level,
            detection_result={
                "confidence": prediction.confidence,
                "explanation": prediction.explanation,
                "features_used": prediction.features_used
            },
            model_version=prediction.model_version,
            reviewed_by=reviewer_id
        )
        
        db.add(fraud_log)
        db.commit()
        
    except Exception as e:
        logger.error(f"Failed to log fraud analysis: {e}")
        db.rollback()

