"""
Advanced Term Simplification Service using ML Models
Provides intelligent simplification of complex terms in financial, legal, technical, and medical domains
"""

import os
import logging
import asyncio
import hashlib
import json
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    BertTokenizer,
    BertForMaskedLM,
    pipeline
)
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import redis
from config import settings
import time

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except Exception as e:
    logger.warning(f"Error downloading NLTK data: {e}")


class MLTermSimplifier:
    """
    Advanced term simplification using T5 and BERT models
    Specializes in financial, legal, technical, and medical terminology
    """

    def __init__(self):
        self.device = torch.device(settings.torch_device)
        self.use_quantization = settings.use_quantization
        self.cache_ttl = settings.simplification_cache_ttl
        self.max_terms = settings.simplification_max_terms
        self.default_level = settings.simplification_default_level

        # Initialize models
        self.t5_model = None
        self.t5_tokenizer = None
        self.bert_model = None
        self.bert_tokenizer = None

        # Simplification levels
        self.simplification_levels = {
            'simple': 'Explain in very simple terms that a child could understand',
            'intermediate': 'Explain in clear, everyday language',
            'advanced': 'Provide a clear but detailed explanation'
        }

        # Domain-specific term dictionaries
        self.domain_terms = {
            'financial': {
                'compound interest': 'money that grows because you earn interest on both your original money and the interest you already earned',
                'diversification': 'spreading your investments across different types of assets to reduce risk',
                'liquidity': 'how quickly you can convert an investment into cash',
                'volatility': 'how much the price of an investment goes up and down',
                'equity': 'ownership stake in a company or property',
                'dividend': 'a payment made by companies to their shareholders',
                'portfolio': 'a collection of investments owned by a person or organization',
                'mutual fund': 'a pool of money from many investors used to buy stocks and bonds',
                'hedge fund': 'an investment fund that uses complex strategies to try to make high returns',
                'derivatives': 'financial contracts whose value depends on the price of something else',
                'collateral': 'something valuable that you promise to give up if you cannot repay a loan',
                'amortization': 'gradually paying off a debt through regular payments',
                'arbitrage': 'buying and selling the same thing in different markets to make a profit',
                'bear market': 'a period when stock prices are falling',
                'bull market': 'a period when stock prices are rising'
            },
            'legal': {
                'jurisdiction': 'the area or territory where a court has the power to make legal decisions',
                'litigation': 'the process of taking legal action in court',
                'plaintiff': 'the person who starts a lawsuit',
                'defendant': 'the person being sued or accused in court',
                'subpoena': 'a legal order requiring someone to appear in court or provide documents',
                'injunction': 'a court order that requires someone to do or stop doing something',
                'tort': 'a wrongful act that causes harm to someone, leading to legal liability',
                'statute of limitations': 'the time limit for filing a lawsuit',
                'precedent': 'a legal decision that serves as a guide for future similar cases',
                'due process': 'fair treatment through the normal judicial system',
                'habeas corpus': 'the right to challenge unlawful imprisonment',
                'affidavit': 'a written statement made under oath',
                'deposition': 'sworn testimony given outside of court',
                'discovery': 'the process of gathering evidence before a trial',
                'settlement': 'an agreement to resolve a dispute without going to trial'
            },
            'technical': {
                'algorithm': 'a set of step-by-step instructions for solving a problem',
                'bandwidth': 'the amount of data that can be transmitted over a network connection',
                'encryption': 'the process of converting information into a secret code',
                'firewall': 'a security system that controls network traffic',
                'malware': 'harmful software designed to damage or gain unauthorized access to computers',
                'cloud computing': 'using remote servers on the internet to store and process data',
                'artificial intelligence': 'computer systems that can perform tasks typically requiring human intelligence',
                'machine learning': 'a type of AI where computers learn from data without being explicitly programmed',
                'blockchain': 'a secure, distributed ledger technology that records transactions',
                'cryptocurrency': 'digital money that uses cryptography for security',
                'API': 'a set of rules that allows different software applications to communicate',
                'database': 'an organized collection of data stored electronically',
                'server': 'a computer that provides services or resources to other computers',
                'protocol': 'a set of rules for how data is transmitted over a network',
                'open source': 'software whose source code is freely available for anyone to use and modify'
            },
            'medical': {
                'hypertension': 'high blood pressure',
                'diabetes': 'a condition where blood sugar levels are too high',
                'inflammation': 'the body\'s response to injury or infection, causing redness and swelling',
                'antibiotic': 'medicine that fights bacterial infections',
                'vaccine': 'a substance that helps your body build immunity against diseases',
                'chronic': 'a condition that lasts for a long time or keeps coming back',
                'acute': 'a condition that comes on suddenly and is severe',
                'benign': 'not harmful or cancerous',
                'malignant': 'cancerous and potentially harmful',
                'prognosis': 'the likely outcome or course of a disease',
                'diagnosis': 'identifying what disease or condition a patient has',
                'symptom': 'a sign that indicates you might have a disease or condition',
                'syndrome': 'a group of symptoms that occur together',
                'pathology': 'the study of diseases and their effects on the body',
                'epidemiology': 'the study of how diseases spread in populations'
            }
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
        """Initialize T5 and BERT models for simplification"""
        try:
            logger.info("Initializing ML models for term simplification...")

            # Load T5 model for text generation
            await self._load_t5_model()

            # Load BERT model for context understanding
            await self._load_bert_model()

            logger.info("Simplification models initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing simplification models: {e}")

    async def _load_t5_model(self):
        """Load T5 model for text simplification"""
        try:
            model_name = settings.simplification_model_path or "t5-base"

            logger.info(f"Loading T5 model: {model_name}")

            self.t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.use_quantization else torch.float32
            )

            if self.use_quantization and torch.cuda.is_available():
                self.t5_model = self.t5_model.half()

            self.t5_model.to(self.device)
            self.t5_model.eval()

            logger.info("T5 model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading T5 model: {e}")
            self.t5_model = None
            self.t5_tokenizer = None

    async def _load_bert_model(self):
        """Load BERT model for context understanding"""
        try:
            model_name = "bert-base-uncased"

            logger.info(f"Loading BERT model: {model_name}")

            self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
            self.bert_model = BertForMaskedLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.use_quantization else torch.float32
            )

            if self.use_quantization and torch.cuda.is_available():
                self.bert_model = self.bert_model.half()

            self.bert_model.to(self.device)
            self.bert_model.eval()

            logger.info("BERT model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            self.bert_model = None
            self.bert_tokenizer = None

    def _get_cache_key(self, term: str, level: str, context: str) -> str:
        """Generate cache key for simplification"""
        content = f"{term}:{level}:{context}"
        return f"simplification:{hashlib.md5(content.encode()).hexdigest()}"

    def _get_cached_simplification(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached simplification"""
        if not self.cache:
            return None

        try:
            cached = self.cache.get(cache_key)
            if cached:
                data = json.loads(cached)
                if datetime.fromisoformat(data['expires']) > datetime.now():
                    return data['simplification']
                else:
                    self.cache.delete(cache_key)
            return None
        except Exception as e:
            logger.warning(f"Error getting cached simplification: {e}")
            return None

    def _cache_simplification(self, cache_key: str, simplification: Dict[str, Any]):
        """Cache simplification result"""
        if not self.cache:
            return

        try:
            expires = datetime.now() + timedelta(seconds=self.cache_ttl)
            data = {
                'simplification': simplification,
                'expires': expires.isoformat()
            }
            self.cache.setex(cache_key, self.cache_ttl, json.dumps(data))
        except Exception as e:
            logger.warning(f"Error caching simplification: {e}")

    def _detect_domain(self, term: str, context: str = "") -> str:
        """Detect the domain of a term based on context and keywords"""
        combined_text = f"{term} {context}".lower()

        # Financial keywords
        financial_keywords = ['money', 'investment', 'bank', 'loan', 'interest', 'stock', 'bond', 'fund', 'finance',
                              'economy']
        if any(keyword in combined_text for keyword in financial_keywords):
            return 'financial'

        # Legal keywords
        legal_keywords = ['court', 'law', 'legal', 'judge', 'lawsuit', 'contract', 'attorney', 'rights', 'liability']
        if any(keyword in combined_text for keyword in legal_keywords):
            return 'legal'

        # Technical keywords
        technical_keywords = ['computer', 'software', 'technology', 'data', 'network', 'system', 'digital', 'cyber']
        if any(keyword in combined_text for keyword in technical_keywords):
            return 'technical'

        # Medical keywords
        medical_keywords = ['health', 'medical', 'disease', 'treatment', 'doctor', 'patient', 'medicine', 'hospital']
        if any(keyword in combined_text for keyword in medical_keywords):
            return 'medical'

        return 'general'

    def _get_dictionary_simplification(self, term: str, domain: str) -> Optional[str]:
        """Get simplification from domain-specific dictionary"""
        term_lower = term.lower().strip()

        if domain in self.domain_terms:
            return self.domain_terms[domain].get(term_lower)

        # Check all domains if not found in detected domain
        for domain_dict in self.domain_terms.values():
            if term_lower in domain_dict:
                return domain_dict[term_lower]

        return None

    def _get_wordnet_simplification(self, term: str) -> Optional[str]:
        """Get simplification using WordNet"""
        try:
            synsets = wordnet.synsets(term)
            if synsets:
                # Get the most common definition
                definition = synsets[0].definition()
                return definition
            return None
        except Exception as e:
            logger.warning(f"Error getting WordNet definition: {e}")
            return None

    async def _generate_ml_simplification(self, term: str, context: str, level: str) -> Optional[str]:
        """Generate simplification using T5 model"""
        if not self.t5_model or not self.t5_tokenizer:
            return None

        try:
            # Create prompt for T5
            level_instruction = self.simplification_levels.get(level, self.simplification_levels['simple'])
            prompt = f"simplify: {term} in context: {context}. {level_instruction}"

            # Tokenize input
            inputs = self.t5_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)

            # Generate simplification
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    repetition_penalty=1.2
                )

            # Decode output
            simplification = self.t5_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Clean up the output
            simplification = simplification.strip()
            if simplification and len(simplification) > 10:  # Ensure meaningful output
                return simplification

            return None

        except Exception as e:
            logger.error(f"Error generating ML simplification: {e}")
            return None

    def _extract_complex_terms(self, text: str) -> List[Tuple[str, int, int]]:
        """Extract complex terms from text using NLP techniques"""
        try:
            # Tokenize and tag parts of speech
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)

            complex_terms = []

            # Look for complex terms (nouns, adjectives, compound words)
            for i, (word, pos) in enumerate(pos_tags):
                # Skip short words and common words
                if len(word) < 4 or word.lower() in ['this', 'that', 'with', 'from', 'they', 'have', 'been']:
                    continue

                # Look for nouns and adjectives that might be complex
                if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']:
                    # Check if it's a compound word or technical term
                    if len(word) > 8 or any(char.isupper() for char in word[1:]) or '-' in word:
                        start_pos = text.find(word)
                        if start_pos != -1:
                            complex_terms.append((word, start_pos, start_pos + len(word)))

            # Look for multi-word technical terms
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i + 1]}"
                if len(bigram) > 10 and any(domain in self.domain_terms for domain in self.domain_terms):
                    for domain_dict in self.domain_terms.values():
                        if bigram.lower() in domain_dict:
                            start_pos = text.find(bigram)
                            if start_pos != -1:
                                complex_terms.append((bigram, start_pos, start_pos + len(bigram)))

            return complex_terms

        except Exception as e:
            logger.error(f"Error extracting complex terms: {e}")
            return []

    async def simplify_term(
            self,
            term: str,
            context: str = "",
            level: str = None
    ) -> Dict[str, Any]:
        """
        Simplify a single term using ML models and domain knowledge

        Args:
            term: Term to simplify
            context: Context in which the term appears
            level: Simplification level (simple, intermediate, advanced)

        Returns:
            Dictionary with simplification result and metadata
        """
        start_time = time.time()

        try:
            # Validate input
            if not term or not term.strip():
                raise ValueError("Term cannot be empty")

            # Use default level if not provided
            if not level:
                level = self.default_level

            if level not in self.simplification_levels:
                raise ValueError(f"Invalid level: {level}. Must be one of {list(self.simplification_levels.keys())}")

            # Check cache
            cache_key = self._get_cache_key(term, level, context)
            cached_result = self._get_cached_simplification(cache_key)

            if cached_result:
                cached_result['from_cache'] = True
                cached_result['processing_time'] = time.time() - start_time
                return cached_result

            # Detect domain
            domain = self._detect_domain(term, context)

            # Try different simplification methods in order of preference
            simplified_text = None
            method_used = None
            confidence = 0.0

            # 1. Try domain-specific dictionary
            dict_simplification = self._get_dictionary_simplification(term, domain)
            if dict_simplification:
                simplified_text = dict_simplification
                method_used = 'domain_dictionary'
                confidence = 0.95

            # 2. Try ML-based simplification
            if not simplified_text:
                ml_simplification = await self._generate_ml_simplification(term, context, level)
                if ml_simplification:
                    simplified_text = ml_simplification
                    method_used = 'ml_model'
                    confidence = 0.85

            # 3. Try WordNet as fallback
            if not simplified_text:
                wordnet_simplification = self._get_wordnet_simplification(term)
                if wordnet_simplification:
                    simplified_text = wordnet_simplification
                    method_used = 'wordnet'
                    confidence = 0.70

            # 4. Final fallback
            if not simplified_text:
                simplified_text = f"A term used in {domain} contexts"
                method_used = 'fallback'
                confidence = 0.30

            # Create result
            result = {
                'original_term': term,
                'simplified_text': simplified_text,
                'domain': domain,
                'level': level,
                'method_used': method_used,
                'confidence': confidence,
                'context': context,
                'processing_time': time.time() - start_time,
                'from_cache': False
            }

            # Cache result
            self._cache_simplification(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Term simplification error: {e}")
            return {
                'error': str(e),
                'original_term': term,
                'processing_time': time.time() - start_time
            }

    async def simplify_text(
            self,
            text: str,
            level: str = None,
            max_terms: int = None
    ) -> Dict[str, Any]:
        """
        Simplify complex terms in a text

        Args:
            text: Text containing terms to simplify
            level: Simplification level
            max_terms: Maximum number of terms to simplify

        Returns:
            Dictionary with simplified text and term explanations
        """
        start_time = time.time()

        try:
            # Validate input
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")

            # Use defaults if not provided
            if not level:
                level = self.default_level
            if not max_terms:
                max_terms = self.max_terms

            # Extract complex terms
            complex_terms = self._extract_complex_terms(text)

            # Limit number of terms to process
            if len(complex_terms) > max_terms:
                complex_terms = complex_terms[:max_terms]

            # Simplify each term
            simplifications = {}
            simplified_text = text

            for term, start_pos, end_pos in complex_terms:
                simplification = await self.simplify_term(term, text, level)

                if 'error' not in simplification:
                    simplifications[term] = simplification

                    # Optionally replace in text (for now, we'll keep original and provide explanations)
                    # simplified_text = simplified_text.replace(term, simplification['simplified_text'])

            return {
                'original_text': text,
                'simplified_text': simplified_text,  # For now, same as original
                'simplifications': simplifications,
                'terms_processed': len(simplifications),
                'level': level,
                'processing_time': time.time() - start_time
            }

        except Exception as e:
            logger.error(f"Text simplification error: {e}")
            return {
                'error': str(e),
                'original_text': text,
                'processing_time': time.time() - start_time
            }

    async def simplify_batch(
            self,
            terms: List[str],
            context: str = "",
            level: str = None
    ) -> List[Dict[str, Any]]:
        """Simplify multiple terms in batch"""
        results = []

        for term in terms:
            result = await self.simplify_term(term, context, level)
            results.append(result)

        return results

    def get_supported_domains(self) -> List[str]:
        """Get list of supported domains"""
        return list(self.domain_terms.keys()) + ['general']

    def get_supported_levels(self) -> Dict[str, str]:
        """Get supported simplification levels"""
        return self.simplification_levels.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the simplification models"""
        return {
            't5_model_available': self.t5_model is not None,
            'bert_model_available': self.bert_model is not None,
            'device': str(self.device),
            'quantization_enabled': self.use_quantization,
            'cache_available': self.cache is not None,
            'supported_domains': self.get_supported_domains(),
            'supported_levels': list(self.simplification_levels.keys()),
            'default_level': self.default_level,
            'max_terms': self.max_terms,
            'domain_term_count': {domain: len(terms) for domain, terms in self.domain_terms.items()}
        }


# Global service instance
_simplifier = None


def get_simplifier() -> MLTermSimplifier:
    """Get global simplifier instance"""
    global _simplifier
    if _simplifier is None:
        _simplifier = MLTermSimplifier()
    return _simplifier


# Convenience functions
async def simplify_term(
        term: str,
        context: str = "",
        level: str = None
) -> Dict[str, Any]:
    """Simplify a term using the global simplifier"""
    simplifier = get_simplifier()
    return await simplifier.simplify_term(term, context, level)


async def simplify_text(
        text: str,
        level: str = None,
        max_terms: int = None
) -> Dict[str, Any]:
    """Simplify text using the global simplifier"""
    simplifier = get_simplifier()
    return await simplifier.simplify_text(text, level, max_terms)


async def simplify_batch(
        terms: List[str],
        context: str = "",
        level: str = None
) -> List[Dict[str, Any]]:
    """Simplify multiple terms using the global simplifier"""
    simplifier = get_simplifier()
    return await simplifier.simplify_batch(terms, context, level)


def get_supported_domains() -> List[str]:
    """Get supported domains"""
    simplifier = get_simplifier()
    return simplifier.get_supported_domains()


def get_supported_levels() -> Dict[str, str]:
    """Get supported simplification levels"""
    simplifier = get_simplifier()
    return simplifier.get_supported_levels()


def get_simplification_model_info() -> Dict[str, Any]:
    """Get simplification model information"""
    simplifier = get_simplifier()
    return simplifier.get_model_info()

