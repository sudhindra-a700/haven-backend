"""
Fallback Fraud Detection Service for HAVEN Platform
Used when torch is not available - provides basic rule-based detection
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class FallbackFraudDetector:
    """Fallback fraud detector that uses rule-based detection"""
    
    def __init__(self):
        self.initialized = True
        logger.info("Fallback fraud detector initialized (torch not available)")
    
    async def analyze_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Basic rule-based fraud analysis"""
        try:
            # Basic rule-based checks
            fraud_score = 0.0
            risk_factors = []
            
            # Check for suspicious patterns
            title = campaign_data.get('title', '').lower()
            description = campaign_data.get('description', '').lower()
            goal_amount = campaign_data.get('goal_amount', 0)
            
            # Rule 1: Extremely high goal amounts
            if goal_amount > 10000000:  # 1 crore
                fraud_score += 0.3
                risk_factors.append("extremely_high_goal")
            
            # Rule 2: Suspicious keywords
            suspicious_keywords = ['urgent', 'emergency', 'immediate', 'quick money', 'easy money']
            for keyword in suspicious_keywords:
                if keyword in title or keyword in description:
                    fraud_score += 0.2
                    risk_factors.append(f"suspicious_keyword_{keyword.replace(' ', '_')}")
            
            # Rule 3: Very short descriptions
            if len(description) < 50:
                fraud_score += 0.1
                risk_factors.append("short_description")
            
            # Determine risk level
            if fraud_score >= 0.7:
                risk_level = "HIGH"
                is_fraudulent = True
            elif fraud_score >= 0.4:
                risk_level = "MEDIUM"
                is_fraudulent = False
            else:
                risk_level = "LOW"
                is_fraudulent = False
            
            return {
                'fraud_score': min(fraud_score, 1.0),
                'risk_level': risk_level,
                'is_fraudulent': is_fraudulent,
                'risk_factors': risk_factors,
                'analysis_method': 'rule_based_fallback',
                'timestamp': datetime.utcnow().isoformat(),
                'confidence': 0.6  # Lower confidence for rule-based
            }
            
        except Exception as e:
            logger.error(f"Error in fallback fraud analysis: {e}")
            return {
                'fraud_score': 0.5,
                'risk_level': 'UNKNOWN',
                'is_fraudulent': False,
                'risk_factors': ['analysis_error'],
                'analysis_method': 'fallback_error',
                'timestamp': datetime.utcnow().isoformat(),
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the fallback detector"""
        return {
            'status': 'healthy',
            'method': 'rule_based_fallback',
            'torch_available': False,
            'timestamp': datetime.utcnow().isoformat()
        }

# Initialize fallback fraud detection service
enhanced_fraud_detection_service = FallbackFraudDetector()
