"""
Instamojo Payment Service for HAVEN Crowdfunding Platform
Handles payment processing through Instamojo API
"""

import logging
import requests
import json
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PaymentStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"

@dataclass
class PaymentRequest:
    """Payment request data"""
    amount: float
    purpose: str
    buyer_name: str
    buyer_email: str
    buyer_phone: str
    redirect_url: str
    webhook_url: str
    campaign_id: int
    user_id: int

@dataclass
class PaymentResponse:
    """Payment response data"""
    payment_id: str
    payment_url: str
    status: PaymentStatus
    amount: float
    created_at: datetime
    longurl: str

class InstamojoService:
    """
    Instamojo payment service for processing donations
    """
    
    def __init__(self, api_key: str, auth_token: str, sandbox: bool = True):
        self.api_key = api_key
        self.auth_token = auth_token
        self.sandbox = sandbox
        
        # Set base URL based on environment
        if sandbox:
            self.base_url = "https://test.instamojo.com/api/1.1/"
        else:
            self.base_url = "https://www.instamojo.com/api/1.1/"
        
        # Set headers
        self.headers = {
            'X-Api-Key': self.api_key,
            'X-Auth-Token': self.auth_token,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        logger.info(f"InstamojoService initialized (sandbox: {sandbox})")
    
    def create_payment_request(self, payment_data: PaymentRequest) -> Optional[PaymentResponse]:
        """
        Create a payment request with Instamojo
        """
        try:
            # Prepare payment request data
            data = {
                'purpose': payment_data.purpose,
                'amount': str(payment_data.amount),
                'buyer_name': payment_data.buyer_name,
                'email': payment_data.buyer_email,
                'phone': payment_data.buyer_phone,
                'redirect_url': payment_data.redirect_url,
                'webhook': payment_data.webhook_url,
                'allow_repeated_payments': 'false',
                'send_email': 'true',
                'send_sms': 'true'
            }
            
            # Make API request
            response = requests.post(
                f"{self.base_url}payment-requests/",
                data=data,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 201:
                result = response.json()
                payment_request = result['payment_request']
                
                return PaymentResponse(
                    payment_id=payment_request['id'],
                    payment_url=payment_request['longurl'],
                    status=PaymentStatus.PENDING,
                    amount=float(payment_request['amount']),
                    created_at=datetime.fromisoformat(payment_request['created_at'].replace('Z', '+00:00')),
                    longurl=payment_request['longurl']
                )
            else:
                logger.error(f"Instamojo payment request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating Instamojo payment request: {e}")
            return None
    
    def get_payment_details(self, payment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get payment details from Instamojo
        """
        try:
            response = requests.get(
                f"{self.base_url}payment-requests/{payment_id}/",
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['payment_request']
            else:
                logger.error(f"Failed to get payment details: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting payment details: {e}")
            return None
    
    def get_payment_status(self, payment_id: str) -> Optional[PaymentStatus]:
        """
        Get payment status from Instamojo
        """
        try:
            payment_details = self.get_payment_details(payment_id)
            if payment_details:
                status = payment_details.get('status', '').lower()
                
                if status == 'completed':
                    return PaymentStatus.COMPLETED
                elif status == 'pending':
                    return PaymentStatus.PENDING
                elif status in ['failed', 'cancelled']:
                    return PaymentStatus.FAILED
                else:
                    return PaymentStatus.PENDING
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting payment status: {e}")
            return None
    
    def verify_payment(self, payment_id: str, payment_request_id: str) -> bool:
        """
        Verify payment completion
        """
        try:
            response = requests.get(
                f"{self.base_url}payments/{payment_id}/",
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                payment = response.json()['payment']
                return (
                    payment['status'] == 'Credit' and 
                    payment['payment_request'] == payment_request_id
                )
            
            return False
            
        except Exception as e:
            logger.error(f"Error verifying payment: {e}")
            return False
    
    def process_webhook(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Instamojo webhook data
        """
        try:
            payment_id = webhook_data.get('payment_id')
            payment_request_id = webhook_data.get('payment_request_id')
            status = webhook_data.get('status', '').lower()
            
            # Map Instamojo status to our status
            if status == 'credit':
                payment_status = PaymentStatus.COMPLETED
            elif status == 'failed':
                payment_status = PaymentStatus.FAILED
            else:
                payment_status = PaymentStatus.PENDING
            
            return {
                'payment_id': payment_id,
                'payment_request_id': payment_request_id,
                'status': payment_status,
                'amount': float(webhook_data.get('amount', 0)),
                'buyer_name': webhook_data.get('buyer_name', ''),
                'buyer_email': webhook_data.get('buyer_email', ''),
                'buyer_phone': webhook_data.get('buyer_phone', ''),
                'currency': webhook_data.get('currency', 'INR'),
                'fees': float(webhook_data.get('fees', 0)),
                'mac': webhook_data.get('mac', ''),
                'payment_timestamp': webhook_data.get('created_at', '')
            }
            
        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
            return {}
    
    def create_refund(self, payment_id: str, refund_amount: Optional[float] = None, reason: str = "Requested by user") -> bool:
        """
        Create a refund for a payment
        """
        try:
            data = {
                'payment_id': payment_id,
                'type': 'RFD',  # Refund type
                'body': reason
            }
            
            if refund_amount:
                data['refund_amount'] = str(refund_amount)
            
            response = requests.post(
                f"{self.base_url}refunds/",
                data=data,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 201:
                logger.info(f"Refund created successfully for payment {payment_id}")
                return True
            else:
                logger.error(f"Refund creation failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating refund: {e}")
            return False
    
    def get_refund_status(self, refund_id: str) -> Optional[Dict[str, Any]]:
        """
        Get refund status
        """
        try:
            response = requests.get(
                f"{self.base_url}refunds/{refund_id}/",
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['refund']
            else:
                logger.error(f"Failed to get refund status: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting refund status: {e}")
            return None
    
    def validate_webhook_signature(self, webhook_data: Dict[str, Any], private_salt: str) -> bool:
        """
        Validate webhook signature for security
        """
        try:
            import hmac
            import hashlib
            
            # Get MAC from webhook data
            received_mac = webhook_data.get('mac', '')
            
            # Remove MAC from data for signature calculation
            data_for_mac = {k: v for k, v in webhook_data.items() if k != 'mac'}
            
            # Sort data by keys
            sorted_data = sorted(data_for_mac.items())
            
            # Create message string
            message = '|'.join([f"{k}={v}" for k, v in sorted_data])
            
            # Calculate expected MAC
            expected_mac = hmac.new(
                private_salt.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha1
            ).hexdigest()
            
            return hmac.compare_digest(received_mac, expected_mac)
            
        except Exception as e:
            logger.error(f"Error validating webhook signature: {e}")
            return False
    
    def get_service_health(self) -> Dict[str, Any]:
        """
        Check service health
        """
        try:
            # Test API connectivity
            response = requests.get(
                f"{self.base_url}payment-requests/",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code in [200, 401]:  # 401 is expected without proper auth
                return {
                    "status": "healthy",
                    "api_accessible": True,
                    "sandbox_mode": self.sandbox,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "api_accessible": False,
                    "error": f"HTTP {response.status_code}",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "api_accessible": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Global service instance
_instamojo_service = None

def get_instamojo_service(api_key: str = None, auth_token: str = None, sandbox: bool = True) -> InstamojoService:
    """Get global Instamojo service instance"""
    global _instamojo_service
    
    if _instamojo_service is None and api_key and auth_token:
        _instamojo_service = InstamojoService(api_key, auth_token, sandbox)
    
    return _instamojo_service

