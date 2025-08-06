"""
Smart Fraud Detection Service with Enhanced Error Handling
Handles torch compatibility issues and version conflicts gracefully
"""

import logging
import sys
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enhanced_fraud_detection_service(campaign_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Smart fraud detection service with robust fallback handling
    
    Args:
        campaign_data (Dict[str, Any]): Campaign data to analyze
    
    Returns:
        Dict[str, Any]: Fraud analysis result with metadata
    """
    
    # First, try to use torch-based fraud detection service
    try:
        # Import torch and check for compatibility
        import torch
        
        # Check if torch has required attributes
        if not hasattr(torch, 'uint64'):
            logger.warning("Torch version incompatible (missing uint64). Using fallback service.")
            raise ImportError("Torch version incompatible")
        
        # Try to import transformers
        from transformers import pipeline
        
        # Import the original fraud detection service
        from fraud_detection_service import enhanced_fraud_detection_service as torch_fraud_service
        
        logger.info("Using torch-based fraud detection service")
        return torch_fraud_service(campaign_data)
        
    except ImportError as e:
        logger.warning(f"Torch/Transformers import failed: {e}. Using fallback service.")
    except RuntimeError as e:
        if "torch" in str(e).lower() or "transformers" in str(e).lower():
            logger.warning(f"Torch/Transformers runtime error: {e}. Using fallback service.")
        else:
            raise e
    except Exception as e:
        logger.warning(f"Unexpected error with torch service: {e}. Using fallback service.")
    
    # Fallback to rule-based fraud detection service
    try:
        from fraud_detection_service_fallback import enhanced_fraud_detection_service as fallback_fraud_service
        logger.info("Using fallback rule-based fraud detection service")
        return fallback_fraud_service(campaign_data)
        
    except ImportError:
        logger.error("Fallback fraud detection service not found. Creating minimal fallback.")
        
        # Minimal fallback implementation
        title = campaign_data.get('title', '')
        description = campaign_data.get('description', '')
        goal_amount = campaign_data.get('goal_amount', 0)
        
        # Basic rule-based fraud detection
        fraud_score = 0.0
        risk_factors = []
        
        # Check for suspicious keywords
        suspicious_keywords = ['urgent', 'emergency', 'guaranteed', 'risk-free', 'limited time']
        for keyword in suspicious_keywords:
            if keyword.lower() in title.lower() or keyword.lower() in description.lower():
                fraud_score += 0.2
                risk_factors.append(f"Contains suspicious keyword: {keyword}")
        
        # Check goal amount
        if goal_amount > 100000:
            fraud_score += 0.3
            risk_factors.append("Very high goal amount")
        
        # Check description length
        if len(description) < 50:
            fraud_score += 0.2
            risk_factors.append("Very short description")
        
        # Determine risk level
        if fraud_score >= 0.7:
            risk_level = "high"
        elif fraud_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "fraud_score": min(fraud_score, 1.0),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "service_used": "minimal_fallback",
            "warning": "Advanced fraud detection unavailable. Using basic rules.",
            "success": True
        }

# Export the service function
__all__ = ['enhanced_fraud_detection_service']

