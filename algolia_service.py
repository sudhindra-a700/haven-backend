"""
Algolia Search Service for HAVEN Crowdfunding Platform
Handles search indexing and querying for campaigns and users
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from algoliasearch.search_client import SearchClient
from algoliasearch.exceptions import AlgoliaException

logger = logging.getLogger(__name__)

@dataclass
class SearchableItem:
    """Base class for searchable items"""
    objectID: str
    title: str
    description: str
    category: str
    created_at: str
    updated_at: str

@dataclass
class CampaignSearchItem(SearchableItem):
    """Campaign search item"""
    creator_name: str
    creator_id: int
    goal_amount: float
    current_amount: float
    status: str
    location: str
    tags: List[str]
    image_url: Optional[str] = None
    short_description: Optional[str] = None
    organization_name: Optional[str] = None
    progress_percentage: float = 0.0
    supporters_count: int = 0
    days_remaining: Optional[int] = None

@dataclass
class UserSearchItem(SearchableItem):
    """User search item"""
    full_name: str
    email: str
    role: str
    location: Optional[str] = None
    bio: Optional[str] = None
    campaigns_created: int = 0
    total_donated: float = 0.0
    profile_image_url: Optional[str] = None

class AlgoliaService:
    """
    Algolia search service for indexing and searching campaigns and users
    """
    
    def __init__(self, app_id: str, api_key: str, admin_api_key: str):
        self.app_id = app_id
        self.api_key = api_key
        self.admin_api_key = admin_api_key
        
        # Initialize Algolia client
        self.client = SearchClient.create(app_id, admin_api_key)
        
        # Define index names
        self.campaigns_index_name = "campaigns"
        self.users_index_name = "users"
        
        # Get index references
        self.campaigns_index = self.client.init_index(self.campaigns_index_name)
        self.users_index = self.client.init_index(self.users_index_name)
        
        # Configure indices
        self._configure_indices()
        
        logger.info("AlgoliaService initialized")
    
    def _configure_indices(self):
        """Configure Algolia indices with proper settings"""
        try:
            # Configure campaigns index
            campaigns_settings = {
                'searchableAttributes': [
                    'title',
                    'description',
                    'short_description',
                    'creator_name',
                    'organization_name',
                    'category',
                    'location',
                    'tags'
                ],
                'attributesForFaceting': [
                    'category',
                    'status',
                    'location',
                    'creator_name',
                    'tags'
                ],
                'customRanking': [
                    'desc(current_amount)',
                    'desc(supporters_count)',
                    'desc(progress_percentage)'
                ],
                'ranking': [
                    'typo',
                    'geo',
                    'words',
                    'filters',
                    'proximity',
                    'attribute',
                    'exact',
                    'custom'
                ],
                'attributesToHighlight': [
                    'title',
                    'description',
                    'creator_name'
                ],
                'attributesToSnippet': [
                    'description:20'
                ],
                'hitsPerPage': 20,
                'maxValuesPerFacet': 100
            }
            
            self.campaigns_index.set_settings(campaigns_settings)
            
            # Configure users index
            users_settings = {
                'searchableAttributes': [
                    'full_name',
                    'bio',
                    'location',
                    'title'
                ],
                'attributesForFaceting': [
                    'role',
                    'location'
                ],
                'customRanking': [
                    'desc(campaigns_created)',
                    'desc(total_donated)'
                ],
                'attributesToHighlight': [
                    'full_name',
                    'bio'
                ],
                'hitsPerPage': 20
            }
            
            self.users_index.set_settings(users_settings)
            
            logger.info("Algolia indices configured successfully")
            
        except AlgoliaException as e:
            logger.error(f"Error configuring Algolia indices: {e}")
    
    def index_campaign(self, campaign_data: Dict[str, Any]) -> bool:
        """
        Index a campaign in Algolia
        """
        try:
            # Create searchable campaign item
            search_item = CampaignSearchItem(
                objectID=str(campaign_data['id']),
                title=campaign_data['title'],
                description=campaign_data['description'],
                short_description=campaign_data.get('short_description', ''),
                category=campaign_data['category'],
                creator_name=campaign_data.get('creator', {}).get('full_name', ''),
                creator_id=campaign_data.get('creator_id', 0),
                goal_amount=float(campaign_data.get('goal_amount', 0)),
                current_amount=float(campaign_data.get('current_amount', 0)),
                status=campaign_data.get('status', 'draft'),
                location=campaign_data.get('location', ''),
                organization_name=campaign_data.get('organization_name', ''),
                image_url=campaign_data.get('image_url', ''),
                tags=campaign_data.get('tags', []),
                progress_percentage=min(100, (campaign_data.get('current_amount', 0) / max(1, campaign_data.get('goal_amount', 1))) * 100),
                supporters_count=campaign_data.get('supporters_count', 0),
                days_remaining=campaign_data.get('days_remaining'),
                created_at=campaign_data.get('created_at', datetime.now().isoformat()),
                updated_at=campaign_data.get('updated_at', datetime.now().isoformat())
            )
            
            # Index the campaign
            self.campaigns_index.save_object(asdict(search_item))
            
            logger.info(f"Campaign indexed successfully: {campaign_data['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing campaign {campaign_data.get('id')}: {e}")
            return False
    
    def index_user(self, user_data: Dict[str, Any]) -> bool:
        """
        Index a user in Algolia
        """
        try:
            # Create searchable user item
            search_item = UserSearchItem(
                objectID=str(user_data['id']),
                title=user_data.get('full_name', ''),
                description=user_data.get('bio', ''),
                category='user',
                full_name=user_data['full_name'],
                email=user_data['email'],
                role=user_data.get('role', 'user'),
                location=user_data.get('location', ''),
                bio=user_data.get('bio', ''),
                campaigns_created=user_data.get('campaigns_created', 0),
                total_donated=float(user_data.get('total_donated', 0)),
                profile_image_url=user_data.get('profile_image_url', ''),
                created_at=user_data.get('created_at', datetime.now().isoformat()),
                updated_at=user_data.get('updated_at', datetime.now().isoformat())
            )
            
            # Index the user
            self.users_index.save_object(asdict(search_item))
            
            logger.info(f"User indexed successfully: {user_data['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing user {user_data.get('id')}: {e}")
            return False
    
    def search_campaigns(self, query: str, filters: Optional[Dict[str, Any]] = None, 
                        page: int = 0, hits_per_page: int = 20) -> Dict[str, Any]:
        """
        Search campaigns in Algolia
        """
        try:
            search_params = {
                'query': query,
                'page': page,
                'hitsPerPage': hits_per_page
            }
            
            # Add filters if provided
            if filters:
                filter_strings = []
                
                if 'category' in filters:
                    filter_strings.append(f"category:{filters['category']}")
                
                if 'status' in filters:
                    filter_strings.append(f"status:{filters['status']}")
                
                if 'location' in filters:
                    filter_strings.append(f"location:{filters['location']}")
                
                if 'min_amount' in filters:
                    filter_strings.append(f"current_amount >= {filters['min_amount']}")
                
                if 'max_amount' in filters:
                    filter_strings.append(f"current_amount <= {filters['max_amount']}")
                
                if filter_strings:
                    search_params['filters'] = ' AND '.join(filter_strings)
            
            # Perform search
            results = self.campaigns_index.search('', search_params)
            
            return {
                'hits': results['hits'],
                'total_hits': results['nbHits'],
                'page': results['page'],
                'total_pages': results['nbPages'],
                'hits_per_page': results['hitsPerPage'],
                'processing_time': results['processingTimeMS']
            }
            
        except Exception as e:
            logger.error(f"Error searching campaigns: {e}")
            return {
                'hits': [],
                'total_hits': 0,
                'page': 0,
                'total_pages': 0,
                'hits_per_page': hits_per_page,
                'processing_time': 0,
                'error': str(e)
            }
    
    def search_users(self, query: str, filters: Optional[Dict[str, Any]] = None,
                    page: int = 0, hits_per_page: int = 20) -> Dict[str, Any]:
        """
        Search users in Algolia
        """
        try:
            search_params = {
                'query': query,
                'page': page,
                'hitsPerPage': hits_per_page
            }
            
            # Add filters if provided
            if filters:
                filter_strings = []
                
                if 'role' in filters:
                    filter_strings.append(f"role:{filters['role']}")
                
                if 'location' in filters:
                    filter_strings.append(f"location:{filters['location']}")
                
                if filter_strings:
                    search_params['filters'] = ' AND '.join(filter_strings)
            
            # Perform search
            results = self.users_index.search('', search_params)
            
            return {
                'hits': results['hits'],
                'total_hits': results['nbHits'],
                'page': results['page'],
                'total_pages': results['nbPages'],
                'hits_per_page': results['hitsPerPage'],
                'processing_time': results['processingTimeMS']
            }
            
        except Exception as e:
            logger.error(f"Error searching users: {e}")
            return {
                'hits': [],
                'total_hits': 0,
                'page': 0,
                'total_pages': 0,
                'hits_per_page': hits_per_page,
                'processing_time': 0,
                'error': str(e)
            }
    
    def get_search_suggestions(self, query: str, index_type: str = 'campaigns') -> List[str]:
        """
        Get search suggestions based on query
        """
        try:
            index = self.campaigns_index if index_type == 'campaigns' else self.users_index
            
            results = index.search(query, {
                'hitsPerPage': 5,
                'attributesToRetrieve': ['title', 'full_name'] if index_type == 'users' else ['title']
            })
            
            suggestions = []
            for hit in results['hits']:
                if index_type == 'campaigns':
                    suggestions.append(hit.get('title', ''))
                else:
                    suggestions.append(hit.get('full_name', ''))
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting search suggestions: {e}")
            return []
    
    def delete_campaign(self, campaign_id: str) -> bool:
        """
        Delete campaign from Algolia index
        """
        try:
            self.campaigns_index.delete_object(campaign_id)
            logger.info(f"Campaign deleted from index: {campaign_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting campaign from index: {e}")
            return False
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete user from Algolia index
        """
        try:
            self.users_index.delete_object(user_id)
            logger.info(f"User deleted from index: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting user from index: {e}")
            return False
    
    def batch_index_campaigns(self, campaigns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch index multiple campaigns
        """
        try:
            search_items = []
            
            for campaign_data in campaigns:
                search_item = CampaignSearchItem(
                    objectID=str(campaign_data['id']),
                    title=campaign_data['title'],
                    description=campaign_data['description'],
                    short_description=campaign_data.get('short_description', ''),
                    category=campaign_data['category'],
                    creator_name=campaign_data.get('creator', {}).get('full_name', ''),
                    creator_id=campaign_data.get('creator_id', 0),
                    goal_amount=float(campaign_data.get('goal_amount', 0)),
                    current_amount=float(campaign_data.get('current_amount', 0)),
                    status=campaign_data.get('status', 'draft'),
                    location=campaign_data.get('location', ''),
                    organization_name=campaign_data.get('organization_name', ''),
                    image_url=campaign_data.get('image_url', ''),
                    tags=campaign_data.get('tags', []),
                    progress_percentage=min(100, (campaign_data.get('current_amount', 0) / max(1, campaign_data.get('goal_amount', 1))) * 100),
                    supporters_count=campaign_data.get('supporters_count', 0),
                    days_remaining=campaign_data.get('days_remaining'),
                    created_at=campaign_data.get('created_at', datetime.now().isoformat()),
                    updated_at=campaign_data.get('updated_at', datetime.now().isoformat())
                )
                search_items.append(asdict(search_item))
            
            # Batch index
            response = self.campaigns_index.save_objects(search_items)
            
            logger.info(f"Batch indexed {len(campaigns)} campaigns")
            return {
                'success': True,
                'indexed_count': len(campaigns),
                'task_id': response.get('taskID')
            }
            
        except Exception as e:
            logger.error(f"Error batch indexing campaigns: {e}")
            return {
                'success': False,
                'error': str(e),
                'indexed_count': 0
            }
    
    def get_popular_searches(self, index_type: str = 'campaigns') -> List[Dict[str, Any]]:
        """
        Get popular search terms (mock implementation)
        """
        try:
            # This is a mock implementation since Algolia Analytics API requires separate setup
            if index_type == 'campaigns':
                return [
                    {'query': 'education', 'count': 150},
                    {'query': 'healthcare', 'count': 120},
                    {'query': 'emergency', 'count': 100},
                    {'query': 'environment', 'count': 80},
                    {'query': 'community', 'count': 75}
                ]
            else:
                return [
                    {'query': 'fundraiser', 'count': 50},
                    {'query': 'volunteer', 'count': 40},
                    {'query': 'organizer', 'count': 30}
                ]
                
        except Exception as e:
            logger.error(f"Error getting popular searches: {e}")
            return []
    
    def get_service_health(self) -> Dict[str, Any]:
        """
        Check Algolia service health
        """
        try:
            # Test API connectivity by getting index settings
            self.campaigns_index.get_settings()
            
            return {
                "status": "healthy",
                "api_accessible": True,
                "app_id": self.app_id,
                "indices": {
                    "campaigns": self.campaigns_index_name,
                    "users": self.users_index_name
                },
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
_algolia_service = None

def get_algolia_service(app_id: str = None, api_key: str = None, admin_api_key: str = None) -> AlgoliaService:
    """Get global Algolia service instance"""
    global _algolia_service
    
    if _algolia_service is None and app_id and api_key and admin_api_key:
        _algolia_service = AlgoliaService(app_id, api_key, admin_api_key)
    
    return _algolia_service

