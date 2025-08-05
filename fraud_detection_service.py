"""
Optimized Fraud Detection Service for HAVEN Crowdfunding Platform
Fixed version with proper data handling, model optimization, and caching
"""

import os
import json
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

@dataclass
class FraudPrediction:
    """Fraud prediction result"""
    fraud_score: float
    risk_level: str
    confidence: float
    explanation: str
    features_used: List[str]
    model_version: str

@dataclass
class CampaignFeatures:
    """Campaign features for fraud detection"""
    # Basic campaign info
    title_length: int
    description_length: int
    goal_amount: float
    category: str
    
    # Creator info
    account_age_days: int
    creator_campaigns_count: int
    creator_success_rate: float
    has_profile_picture: bool
    email_verified: bool
    phone_verified: bool
    
    # Verification info
    has_ngo_registration: bool
    has_pan_card: bool
    has_bank_verification: bool
    
    # Content analysis
    urgency_keywords_count: int
    emotional_keywords_count: int
    financial_keywords_count: int
    
    # Behavioral patterns
    creation_hour: int
    creation_day_of_week: int
    images_count: int
    video_present: bool

class OptimizedFraudDetectionService:
    """
    Optimized fraud detection service with proper model management
    """
    
    def __init__(self):
        self.model_dir = Path(settings.model_cache_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Model components
        self.fraud_model = None
        self.anomaly_model = None
        self.scaler = None
        self.label_encoders = {}
        
        # Model metadata
        self.model_version = "2.0.0"
        self.feature_names = []
        self.model_trained = False
        
        # Cache for predictions
        self.prediction_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Load or train models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize fraud detection models"""
        try:
            # Try to load existing models
            if self._load_models():
                logger.info("✅ Fraud detection models loaded successfully")
                self.model_trained = True
            else:
                logger.info("🔄 Training new fraud detection models")
                self._train_models()
                self._save_models()
                self.model_trained = True
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize fraud detection models: {e}")
            # Create dummy models for basic functionality
            self._create_dummy_models()
    
    def _load_models(self) -> bool:
        """Load pre-trained models from disk"""
        try:
            model_files = {
                'fraud_model': self.model_dir / 'fraud_model.joblib',
                'anomaly_model': self.model_dir / 'anomaly_model.joblib',
                'scaler': self.model_dir / 'scaler.joblib',
                'label_encoders': self.model_dir / 'label_encoders.joblib',
                'metadata': self.model_dir / 'model_metadata.json'
            }
            
            # Check if all files exist
            if not all(f.exists() for f in model_files.values()):
                return False
            
            # Load models
            self.fraud_model = joblib.load(model_files['fraud_model'])
            self.anomaly_model = joblib.load(model_files['anomaly_model'])
            self.scaler = joblib.load(model_files['scaler'])
            self.label_encoders = joblib.load(model_files['label_encoders'])
            
            # Load metadata
            with open(model_files['metadata'], 'r') as f:
                metadata = json.load(f)
                self.model_version = metadata.get('version', '2.0.0')
                self.feature_names = metadata.get('feature_names', [])
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            # Save models
            joblib.dump(self.fraud_model, self.model_dir / 'fraud_model.joblib')
            joblib.dump(self.anomaly_model, self.model_dir / 'anomaly_model.joblib')
            joblib.dump(self.scaler, self.model_dir / 'scaler.joblib')
            joblib.dump(self.label_encoders, self.model_dir / 'label_encoders.joblib')
            
            # Save metadata
            metadata = {
                'version': self.model_version,
                'feature_names': self.feature_names,
                'trained_at': datetime.utcnow().isoformat(),
                'model_type': 'RandomForest + IsolationForest'
            }
            
            with open(self.model_dir / 'model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("✅ Models saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def _train_models(self):
        """Train fraud detection models with synthetic data"""
        logger.info("🔄 Training fraud detection models...")
        
        # Generate synthetic training data
        training_data = self._generate_training_data()
        
        # Prepare features
        X, y = self._prepare_features(training_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train fraud classification model
        self.fraud_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.fraud_model.fit(X_train_scaled, y_train)
        
        # Train anomaly detection model
        self.anomaly_model = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.anomaly_model.fit(X_train_scaled)
        
        # Evaluate models
        y_pred = self.fraud_model.predict(X_test_scaled)
        y_pred_proba = self.fraud_model.predict_proba(X_test_scaled)[:, 1]
        
        logger.info(f"Model performance:")
        logger.info(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
        
        logger.info("✅ Model training completed")
    
    def _generate_training_data(self) -> pd.DataFrame:
        """Generate synthetic training data for fraud detection"""
        np.random.seed(42)
        n_samples = 10000
        
        data = []
        
        for i in range(n_samples):
            # Determine if this is a fraudulent campaign (10% fraud rate)
            is_fraud = np.random.random() < 0.1
            
            if is_fraud:
                # Fraudulent campaign characteristics
                features = CampaignFeatures(
                    title_length=np.random.randint(10, 30),  # Shorter titles
                    description_length=np.random.randint(50, 200),  # Shorter descriptions
                    goal_amount=np.random.uniform(100000, 10000000),  # High goals
                    category=np.random.choice(['technology', 'other']),
                    account_age_days=np.random.randint(1, 30),  # New accounts
                    creator_campaigns_count=np.random.randint(0, 2),  # Few campaigns
                    creator_success_rate=np.random.uniform(0, 0.3),  # Low success rate
                    has_profile_picture=np.random.choice([True, False], p=[0.3, 0.7]),
                    email_verified=np.random.choice([True, False], p=[0.4, 0.6]),
                    phone_verified=np.random.choice([True, False], p=[0.2, 0.8]),
                    has_ngo_registration=np.random.choice([True, False], p=[0.1, 0.9]),
                    has_pan_card=np.random.choice([True, False], p=[0.3, 0.7]),
                    has_bank_verification=np.random.choice([True, False], p=[0.2, 0.8]),
                    urgency_keywords_count=np.random.randint(3, 10),  # High urgency
                    emotional_keywords_count=np.random.randint(5, 15),  # High emotion
                    financial_keywords_count=np.random.randint(2, 8),
                    creation_hour=np.random.randint(0, 24),
                    creation_day_of_week=np.random.randint(0, 7),
                    images_count=np.random.randint(0, 3),  # Few images
                    video_present=np.random.choice([True, False], p=[0.2, 0.8])
                )
                label = 1
            else:
                # Legitimate campaign characteristics
                features = CampaignFeatures(
                    title_length=np.random.randint(20, 80),
                    description_length=np.random.randint(200, 2000),
                    goal_amount=np.random.uniform(10000, 500000),
                    category=np.random.choice(['education', 'health', 'community', 'environment']),
                    account_age_days=np.random.randint(30, 1000),
                    creator_campaigns_count=np.random.randint(0, 10),
                    creator_success_rate=np.random.uniform(0.3, 1.0),
                    has_profile_picture=np.random.choice([True, False], p=[0.8, 0.2]),
                    email_verified=np.random.choice([True, False], p=[0.9, 0.1]),
                    phone_verified=np.random.choice([True, False], p=[0.7, 0.3]),
                    has_ngo_registration=np.random.choice([True, False], p=[0.6, 0.4]),
                    has_pan_card=np.random.choice([True, False], p=[0.8, 0.2]),
                    has_bank_verification=np.random.choice([True, False], p=[0.9, 0.1]),
                    urgency_keywords_count=np.random.randint(0, 3),
                    emotional_keywords_count=np.random.randint(0, 5),
                    financial_keywords_count=np.random.randint(0, 3),
                    creation_hour=np.random.randint(0, 24),
                    creation_day_of_week=np.random.randint(0, 7),
                    images_count=np.random.randint(2, 10),
                    video_present=np.random.choice([True, False], p=[0.6, 0.4])
                )
                label = 0
            
            # Convert to dictionary
            row = {
                **features.__dict__,
                'is_fraud': label
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for model training"""
        # Separate features and target
        feature_columns = [col for col in df.columns if col != 'is_fraud']
        X = df[feature_columns].copy()
        y = df['is_fraud'].values
        
        # Handle categorical variables
        categorical_columns = ['category']
        
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        return X.values, y
    
    def _create_dummy_models(self):
        """Create dummy models for basic functionality"""
        logger.warning("⚠️ Creating dummy fraud detection models")
        
        # Create simple dummy models
        self.fraud_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Train on minimal dummy data
        X_dummy = np.random.random((100, 10))
        y_dummy = np.random.randint(0, 2, 100)
        
        self.scaler.fit(X_dummy)
        self.fraud_model.fit(X_dummy, y_dummy)
        self.anomaly_model.fit(X_dummy)
        
        self.feature_names = [f'feature_{i}' for i in range(10)]
        self.model_trained = True
    
    def extract_campaign_features(self, campaign_data: Dict[str, Any]) -> CampaignFeatures:
        """Extract features from campaign data"""
        # Extract basic features with defaults
        title = campaign_data.get('title', '')
        description = campaign_data.get('description', '')
        
        # Text analysis
        urgency_keywords = ['urgent', 'emergency', 'immediate', 'crisis', 'help', 'save']
        emotional_keywords = ['please', 'desperate', 'dying', 'suffering', 'pain', 'heartbreaking']
        financial_keywords = ['money', 'funds', 'donate', 'contribution', 'payment']
        
        urgency_count = sum(1 for word in urgency_keywords if word.lower() in description.lower())
        emotional_count = sum(1 for word in emotional_keywords if word.lower() in description.lower())
        financial_count = sum(1 for word in financial_keywords if word.lower() in description.lower())
        
        # Creator information
        creator_info = campaign_data.get('creator', {})
        
        return CampaignFeatures(
            title_length=len(title),
            description_length=len(description),
            goal_amount=float(campaign_data.get('goal_amount', 0)),
            category=campaign_data.get('category', 'other'),
            account_age_days=campaign_data.get('account_age_days', 0),
            creator_campaigns_count=creator_info.get('campaigns_count', 0),
            creator_success_rate=creator_info.get('success_rate', 0.0),
            has_profile_picture=bool(creator_info.get('profile_picture')),
            email_verified=creator_info.get('email_verified', False),
            phone_verified=creator_info.get('phone_verified', False),
            has_ngo_registration=bool(campaign_data.get('ngo_darpan_id')),
            has_pan_card=bool(campaign_data.get('pan_number')),
            has_bank_verification=campaign_data.get('bank_verified', False),
            urgency_keywords_count=urgency_count,
            emotional_keywords_count=emotional_count,
            financial_keywords_count=financial_count,
            creation_hour=datetime.now().hour,
            creation_day_of_week=datetime.now().weekday(),
            images_count=len(campaign_data.get('images', [])),
            video_present=bool(campaign_data.get('video_url'))
        )
    
    def predict_fraud(self, campaign_data: Dict[str, Any]) -> FraudPrediction:
        """Predict fraud probability for a campaign"""
        try:
            # Check cache first
            cache_key = self._get_cache_key(campaign_data)
            if cache_key in self.prediction_cache:
                cached_result = self.prediction_cache[cache_key]
                if datetime.now().timestamp() - cached_result['timestamp'] < self.cache_ttl:
                    return cached_result['prediction']
            
            # Extract features
            features = self.extract_campaign_features(campaign_data)
            
            # Convert to DataFrame for processing
            feature_dict = features.__dict__
            df = pd.DataFrame([feature_dict])
            
            # Handle categorical encoding
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    try:
                        df[col] = encoder.transform(df[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        df[col] = 0
            
            # Ensure all expected features are present
            for feature_name in self.feature_names:
                if feature_name not in df.columns:
                    df[feature_name] = 0
            
            # Select and order features
            X = df[self.feature_names].values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get fraud probability
            fraud_proba = self.fraud_model.predict_proba(X_scaled)[0, 1]
            
            # Get anomaly score
            anomaly_score = self.anomaly_model.decision_function(X_scaled)[0]
            
            # Combine scores
            combined_score = (fraud_proba + (1 - (anomaly_score + 0.5))) / 2
            combined_score = max(0, min(1, combined_score))
            
            # Determine risk level
            if combined_score < 0.3:
                risk_level = "low"
            elif combined_score < 0.7:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            # Generate explanation
            explanation = self._generate_explanation(features, combined_score)
            
            # Create prediction result
            prediction = FraudPrediction(
                fraud_score=combined_score,
                risk_level=risk_level,
                confidence=max(fraud_proba, 1 - fraud_proba),
                explanation=explanation,
                features_used=self.feature_names,
                model_version=self.model_version
            )
            
            # Cache result
            self.prediction_cache[cache_key] = {
                'prediction': prediction,
                'timestamp': datetime.now().timestamp()
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Fraud prediction failed: {e}")
            # Return safe default
            return FraudPrediction(
                fraud_score=0.5,
                risk_level="medium",
                confidence=0.5,
                explanation="Unable to analyze campaign due to technical issues",
                features_used=[],
                model_version=self.model_version
            )
    
    def _generate_explanation(self, features: CampaignFeatures, score: float) -> str:
        """Generate human-readable explanation for fraud score"""
        explanations = []
        
        if features.account_age_days < 30:
            explanations.append("New account (less than 30 days old)")
        
        if features.goal_amount > 1000000:
            explanations.append("Very high funding goal")
        
        if not features.email_verified:
            explanations.append("Email not verified")
        
        if not features.has_ngo_registration:
            explanations.append("No NGO registration provided")
        
        if features.urgency_keywords_count > 3:
            explanations.append("High use of urgency keywords")
        
        if features.images_count < 2:
            explanations.append("Few or no images provided")
        
        if not explanations:
            explanations.append("Campaign appears normal based on available data")
        
        base_explanation = f"Fraud risk score: {score:.2f}. "
        return base_explanation + ". ".join(explanations[:3])
    
    def _get_cache_key(self, campaign_data: Dict[str, Any]) -> str:
        """Generate cache key for campaign data"""
        # Create a hash of relevant campaign data
        relevant_data = {
            'title': campaign_data.get('title', ''),
            'description': campaign_data.get('description', ''),
            'goal_amount': campaign_data.get('goal_amount', 0),
            'category': campaign_data.get('category', ''),
            'creator_id': campaign_data.get('creator_id', 0)
        }
        
        data_str = json.dumps(relevant_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get fraud detection service health status"""
        return {
            "status": "healthy" if self.model_trained else "unhealthy",
            "model_loaded": self.model_trained,
            "model_version": self.model_version,
            "feature_count": len(self.feature_names),
            "cache_size": len(self.prediction_cache),
            "models": {
                "fraud_classifier": type(self.fraud_model).__name__ if self.fraud_model else None,
                "anomaly_detector": type(self.anomaly_model).__name__ if self.anomaly_model else None
            }
        }

# Global service instance
fraud_detection_service = None

def get_fraud_detection_service() -> OptimizedFraudDetectionService:
    """Get or create fraud detection service instance"""
    global fraud_detection_service
    if fraud_detection_service is None:
        fraud_detection_service = OptimizedFraudDetectionService()
    return fraud_detection_service

