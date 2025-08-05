"""
Optimized Text Simplification Service for HAVEN Crowdfunding Platform
Lightweight implementation using rule-based and dictionary approaches
"""

import logging
import re
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class SimplificationResult:
    """Result of text simplification"""
    original_text: str
    simplified_text: str
    simplifications: List[Dict[str, str]]
    confidence_score: float
    processing_time: float
    method: str

class SimplificationService:
    """
    Lightweight text simplification service
    Uses rule-based approaches and predefined dictionaries
    """
    
    def __init__(self):
        self.complex_to_simple = {}
        self.financial_terms = {}
        self.legal_terms = {}
        self.technical_terms = {}
        self.medical_terms = {}
        
        # Load dictionaries
        self._load_dictionaries()
        
        # Compile regex patterns
        self._compile_patterns()
        
        logger.info("SimplificationService initialized")
    
    def _load_dictionaries(self):
        """Load simplification dictionaries"""
        try:
            # Financial terms dictionary
            self.financial_terms = {
                "equity": "ownership share",
                "dividend": "profit payment",
                "portfolio": "collection of investments",
                "liquidity": "how easily something can be sold",
                "volatility": "price changes",
                "collateral": "security for a loan",
                "amortization": "paying off debt gradually",
                "depreciation": "decrease in value",
                "appreciation": "increase in value",
                "compound interest": "interest on interest",
                "principal": "original amount of money",
                "yield": "return on investment",
                "maturity": "when investment ends",
                "diversification": "spreading investments",
                "hedge": "protection against loss",
                "leverage": "using borrowed money",
                "margin": "borrowed money for trading",
                "bull market": "rising market",
                "bear market": "falling market",
                "IPO": "first public stock sale",
                "ROI": "return on investment",
                "P/E ratio": "price to earnings ratio",
                "market cap": "total company value",
                "blue chip": "large stable company",
                "penny stock": "cheap risky stock",
                "mutual fund": "pooled investment",
                "ETF": "exchange traded fund",
                "bond": "loan to company or government",
                "stock": "company ownership share",
                "asset": "valuable item owned",
                "liability": "money owed",
                "cash flow": "money coming in and out",
                "budget": "spending plan",
                "credit score": "borrowing trustworthiness",
                "interest rate": "cost of borrowing money",
                "inflation": "rising prices",
                "recession": "economic downturn",
                "GDP": "total economic output",
                "fiscal": "government money matters",
                "monetary": "money supply matters",
                "subsidy": "government financial help",
                "tax deduction": "reduces taxable income",
                "tax credit": "reduces tax owed",
                "capital gains": "profit from selling assets",
                "capital loss": "loss from selling assets"
            }
            
            # Legal terms dictionary
            self.legal_terms = {
                "plaintiff": "person who sues",
                "defendant": "person being sued",
                "litigation": "legal case",
                "jurisdiction": "legal authority area",
                "precedent": "previous court decision",
                "statute": "written law",
                "ordinance": "local law",
                "regulation": "government rule",
                "compliance": "following rules",
                "liability": "legal responsibility",
                "negligence": "careless behavior",
                "breach": "breaking agreement",
                "contract": "legal agreement",
                "tort": "wrongful act",
                "damages": "money compensation",
                "injunction": "court order to stop",
                "subpoena": "court order to appear",
                "deposition": "sworn testimony",
                "affidavit": "sworn written statement",
                "testimony": "spoken evidence",
                "evidence": "proof in court",
                "verdict": "jury decision",
                "settlement": "agreement to end case",
                "arbitration": "private dispute resolution",
                "mediation": "helped negotiation",
                "appeal": "request higher court review",
                "probate": "will processing",
                "custody": "legal care of child",
                "alimony": "spousal support",
                "garnishment": "wage seizure",
                "lien": "legal claim on property",
                "easement": "right to use property",
                "deed": "property ownership document",
                "title": "legal ownership",
                "mortgage": "property loan",
                "foreclosure": "taking back property",
                "bankruptcy": "unable to pay debts",
                "incorporation": "forming a company",
                "LLC": "limited liability company",
                "partnership": "business with partners",
                "sole proprietorship": "one-person business"
            }
            
            # Technical terms dictionary
            self.technical_terms = {
                "algorithm": "step-by-step instructions",
                "API": "way programs talk to each other",
                "bandwidth": "data transfer speed",
                "cache": "temporary storage",
                "cloud": "internet-based computing",
                "database": "organized data storage",
                "encryption": "data protection coding",
                "firewall": "security barrier",
                "malware": "harmful software",
                "phishing": "fake email scam",
                "spam": "unwanted email",
                "virus": "harmful computer program",
                "backup": "data copy for safety",
                "browser": "internet viewing program",
                "cookie": "website tracking file",
                "download": "get file from internet",
                "upload": "send file to internet",
                "streaming": "real-time data flow",
                "server": "computer that serves data",
                "router": "network traffic director",
                "modem": "internet connection device",
                "WiFi": "wireless internet",
                "Bluetooth": "short-range wireless",
                "GPS": "location finding system",
                "smartphone": "internet-connected phone",
                "tablet": "touch screen computer",
                "laptop": "portable computer",
                "desktop": "stationary computer",
                "operating system": "main computer program",
                "software": "computer programs",
                "hardware": "physical computer parts",
                "RAM": "temporary computer memory",
                "CPU": "computer brain",
                "GPU": "graphics processor",
                "SSD": "fast storage drive",
                "USB": "universal connector",
                "HDMI": "high-quality video cable",
                "pixel": "tiny screen dot",
                "resolution": "screen sharpness",
                "refresh rate": "screen update speed"
            }
            
            # Medical terms dictionary
            self.medical_terms = {
                "diagnosis": "identifying illness",
                "prognosis": "illness outlook",
                "symptom": "sign of illness",
                "syndrome": "group of symptoms",
                "chronic": "long-lasting",
                "acute": "sudden and severe",
                "benign": "not harmful",
                "malignant": "harmful cancer",
                "biopsy": "tissue sample test",
                "CT scan": "detailed X-ray",
                "MRI": "magnetic body scan",
                "ultrasound": "sound wave imaging",
                "anesthesia": "pain blocking medicine",
                "antibiotic": "bacteria fighting medicine",
                "vaccine": "disease prevention shot",
                "immunization": "protection from disease",
                "allergy": "body overreaction",
                "inflammation": "body swelling response",
                "infection": "harmful germ invasion",
                "bacteria": "tiny living organisms",
                "virus": "tiny infectious agent",
                "pathogen": "disease-causing germ",
                "immune system": "body's defense system",
                "antibody": "infection fighting protein",
                "hormone": "body chemical messenger",
                "metabolism": "body energy processing",
                "cardiovascular": "heart and blood vessels",
                "respiratory": "breathing system",
                "neurological": "nervous system",
                "gastrointestinal": "digestive system",
                "endocrine": "hormone system",
                "musculoskeletal": "muscles and bones",
                "dermatological": "skin related",
                "ophthalmological": "eye related",
                "otolaryngological": "ear, nose, throat",
                "psychiatric": "mental health",
                "pediatric": "children's health",
                "geriatric": "elderly health",
                "oncology": "cancer treatment",
                "cardiology": "heart treatment",
                "neurology": "brain treatment"
            }
            
            # Combine all dictionaries
            self.complex_to_simple.update(self.financial_terms)
            self.complex_to_simple.update(self.legal_terms)
            self.complex_to_simple.update(self.technical_terms)
            self.complex_to_simple.update(self.medical_terms)
            
            # General complex words
            general_terms = {
                "utilize": "use",
                "facilitate": "help",
                "implement": "put in place",
                "demonstrate": "show",
                "establish": "set up",
                "maintain": "keep",
                "acquire": "get",
                "purchase": "buy",
                "commence": "start",
                "terminate": "end",
                "subsequent": "next",
                "prior": "before",
                "approximately": "about",
                "sufficient": "enough",
                "inadequate": "not enough",
                "substantial": "large",
                "minimal": "small",
                "optimal": "best",
                "maximum": "most",
                "minimum": "least",
                "alternative": "other choice",
                "equivalent": "equal",
                "identical": "same",
                "similar": "alike",
                "different": "not the same",
                "various": "different",
                "numerous": "many",
                "multiple": "many",
                "individual": "single person",
                "particular": "specific",
                "specific": "exact",
                "general": "overall",
                "comprehensive": "complete",
                "extensive": "wide-ranging",
                "significant": "important",
                "essential": "necessary",
                "fundamental": "basic",
                "primary": "main",
                "secondary": "second",
                "additional": "extra",
                "supplementary": "extra",
                "preliminary": "first step",
                "initial": "first",
                "final": "last",
                "ultimate": "final",
                "immediate": "right away",
                "temporary": "short-term",
                "permanent": "long-term",
                "continuous": "ongoing",
                "frequent": "often",
                "occasional": "sometimes",
                "rare": "uncommon",
                "common": "usual",
                "typical": "normal",
                "unusual": "not normal",
                "extraordinary": "amazing",
                "exceptional": "outstanding",
                "standard": "normal",
                "regular": "normal",
                "irregular": "not normal",
                "consistent": "steady",
                "variable": "changing",
                "stable": "steady",
                "unstable": "unsteady",
                "reliable": "trustworthy",
                "unreliable": "not trustworthy",
                "accurate": "correct",
                "inaccurate": "wrong",
                "precise": "exact",
                "approximate": "rough",
                "obvious": "clear",
                "apparent": "clear",
                "evident": "clear",
                "unclear": "confusing",
                "ambiguous": "unclear",
                "definite": "certain",
                "indefinite": "uncertain",
                "probable": "likely",
                "improbable": "unlikely",
                "possible": "might happen",
                "impossible": "cannot happen",
                "necessary": "needed",
                "unnecessary": "not needed",
                "required": "needed",
                "optional": "choice",
                "mandatory": "required",
                "voluntary": "by choice",
                "automatic": "self-working",
                "manual": "by hand",
                "mechanical": "machine-like",
                "electronic": "using electricity",
                "digital": "computer-based",
                "analog": "continuous signal",
                "virtual": "computer-simulated",
                "physical": "real world",
                "theoretical": "idea-based",
                "practical": "real-world",
                "abstract": "idea-only",
                "concrete": "real and solid",
                "tangible": "touchable",
                "intangible": "not touchable",
                "visible": "can see",
                "invisible": "cannot see",
                "audible": "can hear",
                "inaudible": "cannot hear"
            }
            
            self.complex_to_simple.update(general_terms)
            
            logger.info(f"Loaded {len(self.complex_to_simple)} simplification mappings")
            
        except Exception as e:
            logger.error(f"Error loading dictionaries: {e}")
    
    def _compile_patterns(self):
        """Compile regex patterns for text processing"""
        try:
            # Pattern for complex sentences
            self.long_sentence_pattern = re.compile(r'[.!?]+\s+')
            
            # Pattern for passive voice
            self.passive_voice_pattern = re.compile(
                r'\b(is|are|was|were|being|been)\s+\w+ed\b',
                re.IGNORECASE
            )
            
            # Pattern for complex punctuation
            self.complex_punct_pattern = re.compile(r'[;:]')
            
            # Pattern for word boundaries
            self.word_boundary_pattern = re.compile(r'\b')
            
            logger.info("Compiled regex patterns for simplification")
            
        except Exception as e:
            logger.error(f"Error compiling patterns: {e}")
    
    @lru_cache(maxsize=1000)
    def simplify_word(self, word: str) -> Tuple[str, bool]:
        """
        Simplify a single word
        Returns (simplified_word, was_simplified)
        """
        try:
            # Clean the word
            clean_word = word.lower().strip()
            
            # Check direct mapping
            if clean_word in self.complex_to_simple:
                return self.complex_to_simple[clean_word], True
            
            # Check for partial matches (for compound words)
            for complex_term, simple_term in self.complex_to_simple.items():
                if complex_term in clean_word and len(complex_term) > 3:
                    return word.replace(complex_term, simple_term), True
            
            return word, False
            
        except Exception as e:
            logger.warning(f"Error simplifying word '{word}': {e}")
            return word, False
    
    def simplify_sentence_structure(self, sentence: str) -> str:
        """Simplify sentence structure"""
        try:
            # Split long sentences
            if len(sentence.split()) > 20:
                # Try to split at conjunctions
                conjunctions = [' and ', ' but ', ' or ', ' so ', ' because ', ' although ', ' however ']
                for conj in conjunctions:
                    if conj in sentence:
                        parts = sentence.split(conj, 1)
                        if len(parts) == 2:
                            return f"{parts[0].strip()}. {parts[1].strip()}"
            
            # Replace complex punctuation
            sentence = self.complex_punct_pattern.sub('.', sentence)
            
            # Simplify passive voice (basic approach)
            sentence = re.sub(
                r'\b(is|are|was|were)\s+(\w+)ed\s+by\s+(\w+)',
                r'\3 \2s',
                sentence,
                flags=re.IGNORECASE
            )
            
            return sentence
            
        except Exception as e:
            logger.warning(f"Error simplifying sentence structure: {e}")
            return sentence
    
    def simplify_text(
        self,
        text: str,
        preserve_formatting: bool = True,
        max_sentence_length: int = 20
    ) -> SimplificationResult:
        """
        Simplify text using rule-based approach
        """
        start_time = time.time()
        
        try:
            if not text or not text.strip():
                return SimplificationResult(
                    original_text=text,
                    simplified_text=text,
                    simplifications=[],
                    confidence_score=0.0,
                    processing_time=0.0,
                    method="rule_based"
                )
            
            simplified_text = text
            simplifications = []
            
            # Split into sentences
            sentences = self.long_sentence_pattern.split(text)
            simplified_sentences = []
            
            for sentence in sentences:
                if not sentence.strip():
                    simplified_sentences.append(sentence)
                    continue
                
                # Simplify sentence structure
                simplified_sentence = self.simplify_sentence_structure(sentence)
                
                # Simplify words in the sentence
                words = simplified_sentence.split()
                simplified_words = []
                
                for word in words:
                    # Extract punctuation
                    punct = ''
                    clean_word = word
                    if word and word[-1] in '.,!?;:':
                        punct = word[-1]
                        clean_word = word[:-1]
                    
                    # Simplify the word
                    simplified_word, was_simplified = self.simplify_word(clean_word)
                    
                    if was_simplified:
                        simplifications.append({
                            'original': clean_word,
                            'simplified': simplified_word,
                            'context': sentence[:50] + '...' if len(sentence) > 50 else sentence
                        })
                    
                    simplified_words.append(simplified_word + punct)
                
                simplified_sentences.append(' '.join(simplified_words))
            
            # Rejoin sentences
            if preserve_formatting:
                simplified_text = '. '.join(simplified_sentences)
            else:
                simplified_text = ' '.join(simplified_sentences)
            
            # Clean up extra spaces and punctuation
            simplified_text = re.sub(r'\s+', ' ', simplified_text)
            simplified_text = re.sub(r'\s*\.\s*\.', '.', simplified_text)
            simplified_text = simplified_text.strip()
            
            # Calculate confidence score
            total_words = len(text.split())
            simplified_words = len(simplifications)
            confidence_score = min(1.0, simplified_words / max(1, total_words) * 2)
            
            processing_time = time.time() - start_time
            
            return SimplificationResult(
                original_text=text,
                simplified_text=simplified_text,
                simplifications=simplifications,
                confidence_score=confidence_score,
                processing_time=processing_time,
                method="rule_based"
            )
            
        except Exception as e:
            logger.error(f"Error in text simplification: {e}")
            processing_time = time.time() - start_time
            
            return SimplificationResult(
                original_text=text,
                simplified_text=text,
                simplifications=[],
                confidence_score=0.0,
                processing_time=processing_time,
                method="error"
            )
    
    def get_term_explanation(self, term: str) -> Optional[str]:
        """Get explanation for a specific term"""
        try:
            clean_term = term.lower().strip()
            return self.complex_to_simple.get(clean_term)
        except Exception as e:
            logger.warning(f"Error getting term explanation for '{term}': {e}")
            return None
    
    def get_simplification_suggestions(self, text: str, limit: int = 10) -> List[Dict[str, str]]:
        """Get simplification suggestions for text"""
        try:
            words = re.findall(r'\b\w+\b', text.lower())
            suggestions = []
            
            for word in words:
                if word in self.complex_to_simple and len(suggestions) < limit:
                    suggestions.append({
                        'original': word,
                        'simplified': self.complex_to_simple[word],
                        'category': self._get_term_category(word)
                    })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting simplification suggestions: {e}")
            return []
    
    def _get_term_category(self, term: str) -> str:
        """Get category of a term"""
        if term in self.financial_terms:
            return "financial"
        elif term in self.legal_terms:
            return "legal"
        elif term in self.technical_terms:
            return "technical"
        elif term in self.medical_terms:
            return "medical"
        else:
            return "general"
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get service health status"""
        try:
            return {
                "status": "healthy",
                "dictionaries_loaded": len(self.complex_to_simple),
                "categories": {
                    "financial": len(self.financial_terms),
                    "legal": len(self.legal_terms),
                    "technical": len(self.technical_terms),
                    "medical": len(self.medical_terms)
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_supported_categories(self) -> List[str]:
        """Get list of supported simplification categories"""
        return ["financial", "legal", "technical", "medical", "general"]
    
    def batch_simplify(self, texts: List[str]) -> List[SimplificationResult]:
        """Simplify multiple texts in batch"""
        try:
            results = []
            for text in texts:
                result = self.simplify_text(text)
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Error in batch simplification: {e}")
            return []

# Global service instance
_simplification_service = None

def get_simplification_service() -> SimplificationService:
    """Get global simplification service instance"""
    global _simplification_service
    
    if _simplification_service is None:
        _simplification_service = SimplificationService()
    
    return _simplification_service

