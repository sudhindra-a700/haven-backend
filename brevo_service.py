"""
Brevo (formerly Sendinblue) Email Service for HAVEN Crowdfunding Platform
Handles email marketing, transactional emails, and SMS
"""

import logging
import requests
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class EmailType(Enum):
    TRANSACTIONAL = "transactional"
    MARKETING = "marketing"
    NOTIFICATION = "notification"

@dataclass
class EmailRecipient:
    """Email recipient data"""
    email: str
    name: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None

@dataclass
class EmailTemplate:
    """Email template data"""
    template_id: int
    subject: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

@dataclass
class EmailMessage:
    """Email message data"""
    to: List[EmailRecipient]
    subject: str
    html_content: Optional[str] = None
    text_content: Optional[str] = None
    sender: Optional[Dict[str, str]] = None
    reply_to: Optional[Dict[str, str]] = None
    template: Optional[EmailTemplate] = None
    tags: Optional[List[str]] = None

class BrevoService:
    """
    Brevo service for email marketing and transactional emails
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.brevo.com/v3/"
        
        self.headers = {
            'api-key': self.api_key,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        logger.info("BrevoService initialized")
    
    def send_transactional_email(self, email_message: EmailMessage) -> Optional[Dict[str, Any]]:
        """
        Send transactional email via Brevo
        """
        try:
            # Prepare email data
            email_data = {
                'to': [
                    {
                        'email': recipient.email,
                        'name': recipient.name or recipient.email
                    }
                    for recipient in email_message.to
                ],
                'subject': email_message.subject
            }
            
            # Add sender info
            if email_message.sender:
                email_data['sender'] = email_message.sender
            else:
                email_data['sender'] = {
                    'name': 'HAVEN Crowdfunding',
                    'email': 'noreply@haven.org'
                }
            
            # Add reply-to if specified
            if email_message.reply_to:
                email_data['replyTo'] = email_message.reply_to
            
            # Add content
            if email_message.template:
                email_data['templateId'] = email_message.template.template_id
                if email_message.template.params:
                    email_data['params'] = email_message.template.params
            else:
                if email_message.html_content:
                    email_data['htmlContent'] = email_message.html_content
                if email_message.text_content:
                    email_data['textContent'] = email_message.text_content
            
            # Add tags
            if email_message.tags:
                email_data['tags'] = email_message.tags
            
            # Send email
            response = requests.post(
                f"{self.base_url}smtp/email",
                headers=self.headers,
                json=email_data,
                timeout=30
            )
            
            if response.status_code == 201:
                result = response.json()
                logger.info(f"Email sent successfully: {result.get('messageId')}")
                return result
            else:
                logger.error(f"Failed to send email: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error sending transactional email: {e}")
            return None
    
    def send_campaign_notification(self, campaign_id: int, campaign_title: str, recipients: List[str]) -> bool:
        """
        Send campaign notification to subscribers
        """
        try:
            email_recipients = [EmailRecipient(email=email) for email in recipients]
            
            email_message = EmailMessage(
                to=email_recipients,
                subject=f"New Campaign: {campaign_title}",
                html_content=f"""
                <html>
                <body>
                    <h2>New Campaign Alert!</h2>
                    <p>A new campaign has been launched on HAVEN:</p>
                    <h3>{campaign_title}</h3>
                    <p>Check it out and support if you can!</p>
                    <a href="https://haven.org/campaigns/{campaign_id}" 
                       style="background-color: #667eea; color: white; padding: 10px 20px; 
                              text-decoration: none; border-radius: 5px;">
                        View Campaign
                    </a>
                    <br><br>
                    <p>Best regards,<br>The HAVEN Team</p>
                </body>
                </html>
                """,
                tags=["campaign_notification", "marketing"]
            )
            
            result = self.send_transactional_email(email_message)
            return result is not None
            
        except Exception as e:
            logger.error(f"Error sending campaign notification: {e}")
            return False
    
    def send_donation_confirmation(self, donor_email: str, donor_name: str, 
                                 campaign_title: str, amount: float, 
                                 transaction_id: str) -> bool:
        """
        Send donation confirmation email
        """
        try:
            email_message = EmailMessage(
                to=[EmailRecipient(email=donor_email, name=donor_name)],
                subject="Thank you for your donation!",
                html_content=f"""
                <html>
                <body>
                    <h2>Thank you for your generous donation!</h2>
                    <p>Dear {donor_name},</p>
                    <p>We've received your donation of ₹{amount:,.2f} for the campaign:</p>
                    <h3>{campaign_title}</h3>
                    <p><strong>Transaction ID:</strong> {transaction_id}</p>
                    <p>Your support makes a real difference in someone's life. Thank you for being part of the HAVEN community!</p>
                    <p>You will receive a tax receipt shortly.</p>
                    <br>
                    <p>With gratitude,<br>The HAVEN Team</p>
                </body>
                </html>
                """,
                tags=["donation_confirmation", "transactional"]
            )
            
            result = self.send_transactional_email(email_message)
            return result is not None
            
        except Exception as e:
            logger.error(f"Error sending donation confirmation: {e}")
            return False
    
    def send_campaign_update(self, campaign_id: int, campaign_title: str, 
                           update_title: str, update_content: str, 
                           supporters: List[str]) -> bool:
        """
        Send campaign update to supporters
        """
        try:
            email_recipients = [EmailRecipient(email=email) for email in supporters]
            
            email_message = EmailMessage(
                to=email_recipients,
                subject=f"Update: {campaign_title}",
                html_content=f"""
                <html>
                <body>
                    <h2>Campaign Update</h2>
                    <h3>{campaign_title}</h3>
                    <h4>{update_title}</h4>
                    <div>{update_content}</div>
                    <br>
                    <a href="https://haven.org/campaigns/{campaign_id}" 
                       style="background-color: #667eea; color: white; padding: 10px 20px; 
                              text-decoration: none; border-radius: 5px;">
                        View Campaign
                    </a>
                    <br><br>
                    <p>Thank you for your continued support!</p>
                    <p>The HAVEN Team</p>
                </body>
                </html>
                """,
                tags=["campaign_update", "notification"]
            )
            
            result = self.send_transactional_email(email_message)
            return result is not None
            
        except Exception as e:
            logger.error(f"Error sending campaign update: {e}")
            return False
    
    def add_contact_to_list(self, email: str, name: str, list_id: int, 
                           attributes: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add contact to Brevo mailing list
        """
        try:
            contact_data = {
                'email': email,
                'attributes': {
                    'FIRSTNAME': name.split()[0] if name else '',
                    'LASTNAME': ' '.join(name.split()[1:]) if len(name.split()) > 1 else ''
                },
                'listIds': [list_id],
                'updateEnabled': True
            }
            
            if attributes:
                contact_data['attributes'].update(attributes)
            
            response = requests.post(
                f"{self.base_url}contacts",
                headers=self.headers,
                json=contact_data,
                timeout=30
            )
            
            if response.status_code in [201, 204]:
                logger.info(f"Contact added to list: {email}")
                return True
            else:
                logger.error(f"Failed to add contact: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding contact to list: {e}")
            return False
    
    def remove_contact_from_list(self, email: str, list_id: int) -> bool:
        """
        Remove contact from Brevo mailing list
        """
        try:
            response = requests.post(
                f"{self.base_url}contacts/lists/{list_id}/contacts/remove",
                headers=self.headers,
                json={'emails': [email]},
                timeout=30
            )
            
            if response.status_code == 204:
                logger.info(f"Contact removed from list: {email}")
                return True
            else:
                logger.error(f"Failed to remove contact: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing contact from list: {e}")
            return False
    
    def create_email_template(self, template_name: str, subject: str, 
                            html_content: str, sender: Dict[str, str]) -> Optional[int]:
        """
        Create email template in Brevo
        """
        try:
            template_data = {
                'templateName': template_name,
                'subject': subject,
                'htmlContent': html_content,
                'sender': sender,
                'isActive': True
            }
            
            response = requests.post(
                f"{self.base_url}smtp/templates",
                headers=self.headers,
                json=template_data,
                timeout=30
            )
            
            if response.status_code == 201:
                result = response.json()
                template_id = result.get('id')
                logger.info(f"Email template created: {template_id}")
                return template_id
            else:
                logger.error(f"Failed to create template: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating email template: {e}")
            return None
    
    def send_sms(self, phone: str, message: str, sender: str = "HAVEN") -> bool:
        """
        Send SMS via Brevo
        """
        try:
            sms_data = {
                'recipient': phone,
                'content': message,
                'sender': sender,
                'type': 'transactional'
            }
            
            response = requests.post(
                f"{self.base_url}transactionalSMS/sms",
                headers=self.headers,
                json=sms_data,
                timeout=30
            )
            
            if response.status_code == 201:
                logger.info(f"SMS sent successfully to {phone}")
                return True
            else:
                logger.error(f"Failed to send SMS: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return False
    
    def get_email_statistics(self, start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
        """
        Get email statistics from Brevo
        """
        try:
            params = {
                'startDate': start_date,
                'endDate': end_date
            }
            
            response = requests.get(
                f"{self.base_url}smtp/statistics/events",
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get statistics: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting email statistics: {e}")
            return None
    
    def get_service_health(self) -> Dict[str, Any]:
        """
        Check Brevo service health
        """
        try:
            response = requests.get(
                f"{self.base_url}account",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                account_info = response.json()
                return {
                    "status": "healthy",
                    "api_accessible": True,
                    "account_email": account_info.get('email'),
                    "plan": account_info.get('plan', [{}])[0].get('type') if account_info.get('plan') else 'unknown',
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
_brevo_service = None

def get_brevo_service(api_key: str = None) -> BrevoService:
    """Get global Brevo service instance"""
    global _brevo_service
    
    if _brevo_service is None and api_key:
        _brevo_service = BrevoService(api_key)
    
    return _brevo_service

