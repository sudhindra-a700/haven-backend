"""
Optimized Translation Service for HAVEN Crowdfunding Platform
Fixed version with proper caching, fallback mechanisms, and model optimization
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import asyncio

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

@dataclass
class TranslationResult:
    """Translation result with metadata"""
    translated_text: str
    source_language: str
    target_language: str
    confidence_score: float
    method: str  # 'model', 'cache', 'api', 'fallback'
    processing_time: float

class OptimizedTranslationService:
    """
    Optimized translation service with multiple backends and caching
    """
    
    def __init__(self):
        self.cache_dir = Path(settings.model_cache_dir) / "translations"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Translation backends
        self.model_backend = None
        self.api_backend = None
        
        # Cache management
        self.memory_cache = {}
        self.cache_ttl = 86400  # 24 hours
        self.max_cache_size = 10000
        
        # Supported languages
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'te': 'Telugu',
            'mr': 'Marathi',
            'ta': 'Tamil',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'pa': 'Punjabi',
            'or': 'Odia',
            'as': 'Assamese'
        }
        
        # Initialize backends
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize translation backends"""
        try:
            # Try to initialize model backend (optional)
            if settings.enable_translation:
                self._initialize_model_backend()
            
            # Initialize API backend (fallback)
            self._initialize_api_backend()
            
            logger.info("✅ Translation service initialized")
            
        except Exception as e:
            logger.error(f"❌ Translation service initialization failed: {e}")
            # Create minimal fallback service
            self._create_fallback_service()
    
    def _initialize_model_backend(self):
        """Initialize local model backend (optional)"""
        try:
            # Use a lightweight translation model instead of M2M100
            from transformers import pipeline
            
            # Use a smaller, faster model
            model_name = "Helsinki-NLP/opus-mt-en-hi"  # English to Hindi
            
            # Check if model is cached
            model_cache_path = self.cache_dir / "translation_model"
            
            if model_cache_path.exists():
                logger.info("📦 Loading cached translation model")
                self.model_backend = pipeline(
                    "translation",
                    model=str(model_cache_path),
                    device=-1  # Use CPU
                )
            else:
                logger.info("⬇️ Downloading translation model (one-time setup)")
                self.model_backend = pipeline(
                    "translation",
                    model=model_name,
                    device=-1  # Use CPU
                )
                # Cache the model
                self.model_backend.save_pretrained(str(model_cache_path))
            
            logger.info("✅ Model backend initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Model backend initialization failed: {e}")
            self.model_backend = None
    
    def _initialize_api_backend(self):
        """Initialize API backend for translation"""
        try:
            # Use Google Translate API as fallback
            google_api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY")
            
            if google_api_key:
                from googletrans import Translator
                self.api_backend = Translator()
                logger.info("✅ Google Translate API backend initialized")
            else:
                logger.warning("⚠️ Google Translate API key not found")
                self.api_backend = None
                
        except ImportError:
            logger.warning("⚠️ googletrans library not available")
            self.api_backend = None
        except Exception as e:
            logger.warning(f"⚠️ API backend initialization failed: {e}")
            self.api_backend = None
    
    def _create_fallback_service(self):
        """Create minimal fallback translation service"""
        logger.warning("⚠️ Creating fallback translation service")
        
        # Simple dictionary-based translations for common phrases
        self.fallback_translations = {
            ('en', 'hi'): {
                'hello': 'नमस्ते',
                'thank you': 'धन्यवाद',
                'welcome': 'स्वागत है',
                'donate': 'दान करें',
                'campaign': 'अभियान',
                'help': 'मदद',
                'support': 'समर्थन'
            },
            ('hi', 'en'): {
                'नमस्ते': 'hello',
                'धन्यवाद': 'thank you',
                'स्वागत है': 'welcome',
                'दान करें': 'donate',
                'अभियान': 'campaign',
                'मदद': 'help',
                'समर्थन': 'support'
            }
        }
    
    def _get_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key for translation"""
        content = f"{text}|{source_lang}|{target_lang}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[TranslationResult]:
        """Get translation from cache"""
        # Check memory cache first
        if cache_key in self.memory_cache:
            cached_item = self.memory_cache[cache_key]
            if datetime.now().timestamp() - cached_item['timestamp'] < self.cache_ttl:
                return cached_item['result']
            else:
                # Remove expired item
                del self.memory_cache[cache_key]
        
        # Check file cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Check if cache is still valid
                cached_time = datetime.fromisoformat(cached_data['timestamp'])
                if datetime.now() - cached_time < timedelta(seconds=self.cache_ttl):
                    result = TranslationResult(**cached_data['result'])
                    
                    # Add to memory cache
                    self.memory_cache[cache_key] = {
                        'result': result,
                        'timestamp': datetime.now().timestamp()
                    }
                    
                    return result
                else:
                    # Remove expired cache file
                    cache_file.unlink()
                    
            except Exception as e:
                logger.warning(f"Failed to read cache file: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, result: TranslationResult):
        """Save translation to cache"""
        try:
            # Save to memory cache
            self.memory_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now().timestamp()
            }
            
            # Limit memory cache size
            if len(self.memory_cache) > self.max_cache_size:
                # Remove oldest entries
                sorted_items = sorted(
                    self.memory_cache.items(),
                    key=lambda x: x[1]['timestamp']
                )
                for key, _ in sorted_items[:len(self.memory_cache) - self.max_cache_size]:
                    del self.memory_cache[key]
            
            # Save to file cache
            cache_file = self.cache_dir / f"{cache_key}.json"
            cache_data = {
                'result': {
                    'translated_text': result.translated_text,
                    'source_language': result.source_language,
                    'target_language': result.target_language,
                    'confidence_score': result.confidence_score,
                    'method': result.method,
                    'processing_time': result.processing_time
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    async def translate_text(
        self,
        text: str,
        target_language: str,
        source_language: str = 'auto'
    ) -> TranslationResult:
        """
        Translate text using available backends
        """
        start_time = datetime.now()
        
        try:
            # Validate inputs
            if not text or not text.strip():
                return TranslationResult(
                    translated_text="",
                    source_language=source_language,
                    target_language=target_language,
                    confidence_score=1.0,
                    method="validation",
                    processing_time=0.0
                )
            
            # Check if translation is needed
            if source_language == target_language:
                return TranslationResult(
                    translated_text=text,
                    source_language=source_language,
                    target_language=target_language,
                    confidence_score=1.0,
                    method="no_translation",
                    processing_time=0.0
                )
            
            # Check cache
            cache_key = self._get_cache_key(text, source_language, target_language)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                cached_result.method = "cache"
                return cached_result
            
            # Try model backend first
            if self.model_backend and source_language in ['en'] and target_language in ['hi']:
                try:
                    result = await self._translate_with_model(text, source_language, target_language)
                    if result:
                        processing_time = (datetime.now() - start_time).total_seconds()
                        result.processing_time = processing_time
                        self._save_to_cache(cache_key, result)
                        return result
                except Exception as e:
                    logger.warning(f"Model translation failed: {e}")
            
            # Try API backend
            if self.api_backend:
                try:
                    result = await self._translate_with_api(text, source_language, target_language)
                    if result:
                        processing_time = (datetime.now() - start_time).total_seconds()
                        result.processing_time = processing_time
                        self._save_to_cache(cache_key, result)
                        return result
                except Exception as e:
                    logger.warning(f"API translation failed: {e}")
            
            # Use fallback translation
            result = self._translate_with_fallback(text, source_language, target_language)
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return TranslationResult(
                translated_text=text,  # Return original text
                source_language=source_language,
                target_language=target_language,
                confidence_score=0.0,
                method="error",
                processing_time=processing_time
            )
    
    async def _translate_with_model(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Optional[TranslationResult]:
        """Translate using local model"""
        try:
            # Run model inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.model_backend(text, max_length=512)
            )
            
            translated_text = result[0]['translation_text']
            
            return TranslationResult(
                translated_text=translated_text,
                source_language=source_lang,
                target_language=target_lang,
                confidence_score=0.8,  # Model confidence
                method="model",
                processing_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"Model translation error: {e}")
            return None
    
    async def _translate_with_api(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Optional[TranslationResult]:
        """Translate using API backend"""
        try:
            # Run API call in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.api_backend.translate(
                    text,
                    src=source_lang if source_lang != 'auto' else None,
                    dest=target_lang
                )
            )
            
            return TranslationResult(
                translated_text=result.text,
                source_language=result.src,
                target_language=target_lang,
                confidence_score=0.9,  # API confidence
                method="api",
                processing_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"API translation error: {e}")
            return None
    
    def _translate_with_fallback(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> TranslationResult:
        """Fallback translation using dictionary"""
        translated_text = text
        confidence = 0.1
        
        # Check if we have fallback translations
        if hasattr(self, 'fallback_translations'):
            lang_pair = (source_lang, target_lang)
            if lang_pair in self.fallback_translations:
                translations = self.fallback_translations[lang_pair]
                
                # Simple word-by-word translation
                words = text.lower().split()
                translated_words = []
                
                for word in words:
                    if word in translations:
                        translated_words.append(translations[word])
                        confidence = 0.5  # Better confidence for known words
                    else:
                        translated_words.append(word)
                
                if translated_words:
                    translated_text = ' '.join(translated_words)
        
        return TranslationResult(
            translated_text=translated_text,
            source_language=source_lang,
            target_language=target_lang,
            confidence_score=confidence,
            method="fallback",
            processing_time=0.0
        )
    
    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        try:
            if self.api_backend:
                result = self.api_backend.detect(text)
                return result.lang
            else:
                # Simple heuristic detection
                # Check for common Hindi characters
                hindi_chars = set('अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह')
                if any(char in hindi_chars for char in text):
                    return 'hi'
                else:
                    return 'en'
                    
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'en'  # Default to English
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages"""
        return self.supported_languages.copy()
    
    def clear_cache(self):
        """Clear translation cache"""
        try:
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clear file cache
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            
            logger.info("✅ Translation cache cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        file_cache_size = len(list(self.cache_dir.glob("*.json")))
        
        return {
            "memory_cache_size": len(self.memory_cache),
            "file_cache_size": file_cache_size,
            "cache_ttl_hours": self.cache_ttl / 3600,
            "max_cache_size": self.max_cache_size
        }
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get translation service health"""
        return {
            "status": "healthy",
            "backends": {
                "model": self.model_backend is not None,
                "api": self.api_backend is not None,
                "fallback": hasattr(self, 'fallback_translations')
            },
            "supported_languages": len(self.supported_languages),
            "cache_stats": self.get_cache_stats()
        }

# Global service instance
translation_service = None

def get_translation_service() -> OptimizedTranslationService:
    """Get or create translation service instance"""
    global translation_service
    if translation_service is None:
        translation_service = OptimizedTranslationService()
    return translation_service

