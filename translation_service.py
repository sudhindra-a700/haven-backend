"""
Translation Service using IndicTrans2 trained on IndicCorp dataset
Supports only 4 languages: English, Tamil, Hindi, Telugu
"""

import os
import logging
import asyncio
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
import redis
from config import settings
import time

logger = logging.getLogger(__name__)


class IndicTrans2Service:
    """
    Translation service using IndicTrans2 trained on IndicCorp dataset
    Supports: English (en), Tamil (ta), Hindi (hi), Telugu (te)
    """

    def __init__(self):
        self.device = torch.device(settings.torch_device)
        self.use_quantization = settings.use_quantization
        self.cache_ttl = settings.translation_cache_ttl
        self.max_text_length = settings.translation_max_text_length
        self.batch_size = settings.translation_batch_size

        # Initialize models
        self.en_to_indic_model = None
        self.en_to_indic_tokenizer = None
        self.indic_to_en_model = None
        self.indic_to_en_tokenizer = None

        # Supported languages (limited to 4 as requested)
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu'
        }

        # IndicTrans2 language codes for IndicCorp dataset
        self.indic_lang_codes = {
            'hi': 'hin_Deva',  # Hindi
            'ta': 'tam_Taml',  # Tamil
            'te': 'tel_Telu',  # Telugu
            'en': 'eng_Latn'  # English
        }

        # Initialize cache
        self.cache = self._init_cache()

        # Initialize models asynchronously
        asyncio.create_task(self._initialize_models())

    def _init_cache(self) -> Optional[redis.Redis]:
        """Initialize Redis cache if available"""
        try:
            if hasattr(settings, 'redis_url') and settings.redis_url:
                return redis.from_url(settings.redis_url, decode_responses=True)
            return None
        except Exception as e:
            logger.warning(f"Redis cache not available: {e}")
            return None

    async def _initialize_models(self):
        """Initialize IndicTrans2 models"""
        try:
            logger.info("Initializing IndicTrans2 models trained on IndicCorp dataset...")

            # Load English to Indic model
            await self._load_en_to_indic_model()

            # Load Indic to English model
            await self._load_indic_to_en_model()

            logger.info("IndicTrans2 models initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing IndicTrans2 models: {e}")

    async def _load_en_to_indic_model(self):
        """Load English to Indic IndicTrans2 model"""
        try:
            model_name = "ai4bharat/indictrans2-en-indic-1B"

            logger.info(f"Loading English to Indic model: {model_name}")

            self.en_to_indic_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            self.en_to_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.use_quantization else torch.float32
            )

            if self.use_quantization and torch.cuda.is_available():
                self.en_to_indic_model = self.en_to_indic_model.half()

            self.en_to_indic_model.to(self.device)
            self.en_to_indic_model.eval()

            logger.info("English to Indic model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading English to Indic model: {e}")
            self.en_to_indic_model = None
            self.en_to_indic_tokenizer = None

    async def _load_indic_to_en_model(self):
        """Load Indic to English IndicTrans2 model"""
        try:
            model_name = "ai4bharat/indictrans2-indic-en-1B"

            logger.info(f"Loading Indic to English model: {model_name}")

            self.indic_to_en_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            self.indic_to_en_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.use_quantization else torch.float32
            )

            if self.use_quantization and torch.cuda.is_available():
                self.indic_to_en_model = self.indic_to_en_model.half()

            self.indic_to_en_model.to(self.device)
            self.indic_to_en_model.eval()

            logger.info("Indic to English model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading Indic to English model: {e}")
            self.indic_to_en_model = None
            self.indic_to_en_tokenizer = None

    def _get_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key for translation"""
        content = f"{text}:{source_lang}:{target_lang}"
        return f"indictrans2:{hashlib.md5(content.encode()).hexdigest()}"

    def _get_cached_translation(self, cache_key: str) -> Optional[str]:
        """Get cached translation"""
        if not self.cache:
            return None

        try:
            cached = self.cache.get(cache_key)
            if cached:
                data = json.loads(cached)
                if datetime.fromisoformat(data['expires']) > datetime.now():
                    return data['translation']
                else:
                    self.cache.delete(cache_key)
            return None
        except Exception as e:
            logger.warning(f"Error getting cached translation: {e}")
            return None

    def _cache_translation(self, cache_key: str, translation: str):
        """Cache translation result"""
        if not self.cache:
            return

        try:
            expires = datetime.now() + timedelta(seconds=self.cache_ttl)
            data = {
                'translation': translation,
                'expires': expires.isoformat()
            }
            self.cache.setex(cache_key, self.cache_ttl, json.dumps(data))
        except Exception as e:
            logger.warning(f"Error caching translation: {e}")

    def _preprocess_text(self, text: str, source_lang: str) -> str:
        """Preprocess text for IndicTrans2"""
        try:
            # Basic cleaning
            text = text.strip()

            # Remove extra whitespace
            text = ' '.join(text.split())

            # For Indic languages, ensure proper encoding
            if source_lang in ['hi', 'ta', 'te']:
                text = text.encode('utf-8').decode('utf-8')

            return text
        except Exception as e:
            logger.warning(f"Error preprocessing text: {e}")
            return text

    def _postprocess_text(self, text: str, target_lang: str) -> str:
        """Postprocess translated text"""
        try:
            # Basic cleaning
            text = text.strip()

            # Remove any special tokens
            text = text.replace('<pad>', '').replace('</s>', '').replace('<s>', '')
            text = text.replace('<unk>', '').replace('[UNK]', '')

            # Normalize whitespace
            text = ' '.join(text.split())

            return text
        except Exception as e:
            logger.warning(f"Error postprocessing text: {e}")
            return text

    def detect_language(self, text: str) -> str:
        """Detect language of input text (limited to supported 4 languages)"""
        try:
            # Check for Devanagari script (Hindi)
            if any('\u0900' <= char <= '\u097F' for char in text):
                return 'hi'

            # Check for Telugu script
            elif any('\u0C00' <= char <= '\u0C7F' for char in text):
                return 'te'

            # Check for Tamil script
            elif any('\u0B80' <= char <= '\u0BFF' for char in text):
                return 'ta'

            # Default to English for Latin script
            else:
                return 'en'

        except Exception as e:
            logger.warning(f"Error detecting language: {e}")
            return 'en'

    async def _translate_en_to_indic(self, text: str, target_lang: str) -> str:
        """Translate from English to Indic language using IndicTrans2"""
        if not self.en_to_indic_model or not self.en_to_indic_tokenizer:
            raise ValueError("English to Indic model not available")

        try:
            # Preprocess text
            processed_text = self._preprocess_text(text, 'en')

            # Get target language code
            tgt_lang = self.indic_lang_codes[target_lang]

            # Prepare input with language prefix
            input_text = f"<2{tgt_lang}> {processed_text}"

            # Tokenize input
            inputs = self.en_to_indic_tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)

            # Generate translation
            with torch.no_grad():
                outputs = self.en_to_indic_model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=5,
                    early_stopping=True,
                    do_sample=False,
                    repetition_penalty=1.2
                )

            # Decode output
            translation = self.en_to_indic_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Postprocess
            translation = self._postprocess_text(translation, target_lang)

            return translation

        except Exception as e:
            logger.error(f"Error in English to Indic translation: {e}")
            raise

    async def _translate_indic_to_en(self, text: str, source_lang: str) -> str:
        """Translate from Indic language to English using IndicTrans2"""
        if not self.indic_to_en_model or not self.indic_to_en_tokenizer:
            raise ValueError("Indic to English model not available")

        try:
            # Preprocess text
            processed_text = self._preprocess_text(text, source_lang)

            # Get source language code
            src_lang = self.indic_lang_codes[source_lang]

            # Prepare input with language prefix
            input_text = f"<2eng_Latn> {processed_text}"

            # Tokenize input
            inputs = self.indic_to_en_tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)

            # Generate translation
            with torch.no_grad():
                outputs = self.indic_to_en_model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=5,
                    early_stopping=True,
                    do_sample=False,
                    repetition_penalty=1.2
                )

            # Decode output
            translation = self.indic_to_en_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Postprocess
            translation = self._postprocess_text(translation, 'en')

            return translation

        except Exception as e:
            logger.error(f"Error in Indic to English translation: {e}")
            raise

    async def _translate_indic_to_indic(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate between Indic languages via English pivot"""
        try:
            # First translate to English
            english_text = await self._translate_indic_to_en(text, source_lang)

            # Then translate from English to target Indic language
            final_translation = await self._translate_en_to_indic(english_text, target_lang)

            return final_translation

        except Exception as e:
            logger.error(f"Error in Indic to Indic translation: {e}")
            raise

    async def translate(
            self,
            text: str,
            target_lang: str,
            source_lang: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate text to target language using IndicTrans2

        Args:
            text: Text to translate
            target_lang: Target language code (en, hi, ta, te)
            source_lang: Source language code (auto-detected if None)

        Returns:
            Dictionary with translation result and metadata
        """
        start_time = time.time()

        try:
            # Validate input
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")

            if len(text) > self.max_text_length:
                raise ValueError(f"Text too long. Maximum length: {self.max_text_length}")

            # Validate target language
            if target_lang not in self.supported_languages:
                raise ValueError(
                    f"Unsupported target language: {target_lang}. Supported: {list(self.supported_languages.keys())}")

            # Detect source language if not provided
            if not source_lang:
                source_lang = self.detect_language(text)

            # Validate source language
            if source_lang not in self.supported_languages:
                raise ValueError(
                    f"Unsupported source language: {source_lang}. Supported: {list(self.supported_languages.keys())}")

            # Check if translation is needed
            if source_lang == target_lang:
                return {
                    'translated_text': text,
                    'source_language': source_lang,
                    'target_language': target_lang,
                    'confidence': 1.0,
                    'processing_time': time.time() - start_time,
                    'model_used': 'no_translation_needed'
                }

            # Check cache
            cache_key = self._get_cache_key(text, source_lang, target_lang)
            cached_result = self._get_cached_translation(cache_key)

            if cached_result:
                return {
                    'translated_text': cached_result,
                    'source_language': source_lang,
                    'target_language': target_lang,
                    'confidence': 0.95,
                    'processing_time': time.time() - start_time,
                    'model_used': 'indictrans2_cached',
                    'from_cache': True
                }

            # Perform translation based on language pair
            translated_text = None
            model_used = None

            if source_lang == 'en' and target_lang in ['hi', 'ta', 'te']:
                # English to Indic
                translated_text = await self._translate_en_to_indic(text, target_lang)
                model_used = 'indictrans2_en_to_indic'

            elif source_lang in ['hi', 'ta', 'te'] and target_lang == 'en':
                # Indic to English
                translated_text = await self._translate_indic_to_en(text, source_lang)
                model_used = 'indictrans2_indic_to_en'

            elif source_lang in ['hi', 'ta', 'te'] and target_lang in ['hi', 'ta', 'te']:
                # Indic to Indic (via English pivot)
                translated_text = await self._translate_indic_to_indic(text, source_lang, target_lang)
                model_used = 'indictrans2_indic_to_indic_pivot'

            else:
                raise ValueError(f"Unsupported language pair: {source_lang} -> {target_lang}")

            if not translated_text:
                raise ValueError("Translation failed")

            # Cache result
            self._cache_translation(cache_key, translated_text)

            return {
                'translated_text': translated_text,
                'source_language': source_lang,
                'target_language': target_lang,
                'confidence': 0.92,
                'processing_time': time.time() - start_time,
                'model_used': model_used,
                'from_cache': False,
                'dataset': 'IndicCorp'
            }

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {
                'error': str(e),
                'source_language': source_lang,
                'target_language': target_lang,
                'processing_time': time.time() - start_time
            }

    async def translate_batch(
            self,
            texts: List[str],
            target_lang: str,
            source_lang: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Translate multiple texts in batch

        Args:
            texts: List of texts to translate
            target_lang: Target language code
            source_lang: Source language code (auto-detected if None)

        Returns:
            List of translation results
        """
        if not settings.batch_processing:
            # Process sequentially if batch processing is disabled
            results = []
            for text in texts:
                result = await self.translate(text, target_lang, source_lang)
                results.append(result)
            return results

        # Process in batches
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_tasks = [
                self.translate(text, target_lang, source_lang)
                for text in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle exceptions in batch results
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append({
                        'error': str(result),
                        'target_language': target_lang
                    })
                else:
                    results.append(result)

        return results

    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return self.supported_languages.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'en_to_indic_available': self.en_to_indic_model is not None,
            'indic_to_en_available': self.indic_to_en_model is not None,
            'device': str(self.device),
            'quantization_enabled': self.use_quantization,
            'cache_available': self.cache is not None,
            'supported_languages': self.get_supported_languages(),
            'dataset': 'IndicCorp',
            'model_type': 'IndicTrans2',
            'language_count': len(self.supported_languages)
        }

    def is_language_supported(self, lang_code: str) -> bool:
        """Check if language is supported"""
        return lang_code in self.supported_languages


# Global service instance
_translation_service = None


def get_translation_service() -> IndicTrans2Service:
    """Get global translation service instance"""
    global _translation_service
    if _translation_service is None:
        _translation_service = IndicTrans2Service()
    return _translation_service


# Convenience functions
async def translate_text(
        text: str,
        target_lang: str,
        source_lang: Optional[str] = None
) -> Dict[str, Any]:
    """Translate text using the global service"""
    service = get_translation_service()
    return await service.translate(text, target_lang, source_lang)


async def translate_batch(
        texts: List[str],
        target_lang: str,
        source_lang: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Translate multiple texts using the global service"""
    service = get_translation_service()
    return await service.translate_batch(texts, target_lang, source_lang)


def detect_language(text: str) -> str:
    """Detect language of text"""
    service = get_translation_service()
    return service.detect_language(text)


def get_supported_languages() -> Dict[str, str]:
    """Get supported languages"""
    service = get_translation_service()
    return service.get_supported_languages()


def is_language_supported(lang_code: str) -> bool:
    """Check if language is supported"""
    service = get_translation_service()
    return service.is_language_supported(lang_code)

