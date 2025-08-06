"""
Smart Fraud Detection Service for HAVEN Platform
Automatically uses torch-based ML or fallback based on availability
"""

import logging

logger = logging.getLogger(__name__)

try:
    # Try to import torch-based fraud detection
    import torch
    from fraud_detection_service import enhanced_fraud_detection_service
    logger.info("Using torch-based fraud detection service")
    
except ImportError:
    # Fall back to rule-based detection
    from fraud_detection_service_fallback import enhanced_fraud_detection_service
    logger.info("Using fallback rule-based fraud detection service")

# Export the service
__all__ = ['enhanced_fraud_detection_service']

