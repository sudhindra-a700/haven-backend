"""
Smart Translation Service with Enhanced Error Handling
Handles torch compatibility issues and version conflicts gracefully
"""

import logging
import sys
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def translation_service(text: str, target_language: str = "en") -> Dict[str, Any]:
    """
    Smart translation service with robust fallback handling
    
    Args:
        text (str): Text to translate
        target_language (str): Target language code (default: "en")
    
    Returns:
        Dict[str, Any]: Translation result with metadata
    """
    
    # First, try to use torch-based translation service
    try:
        # Import torch and check for compatibility
        import torch
        
        # Check if torch has required attributes
        if not hasattr(torch, 'uint64'):
            logger.warning("Torch version incompatible (missing uint64). Using fallback service.")
            raise ImportError("Torch version incompatible")
        
        # Try to import transformers
        from transformers import pipeline
        
        # Import the original translation service
        from translation_service import translation_service as torch_translation_service
        
        logger.info("Using torch-based translation service")
        return torch_translation_service(text, target_language)
        
    except ImportError as e:
        logger.warning(f"Torch/Transformers import failed: {e}. Using fallback service.")
    except RuntimeError as e:
        if "torch" in str(e).lower() or "transformers" in str(e).lower():
            logger.warning(f"Torch/Transformers runtime error: {e}. Using fallback service.")
        else:
            raise e
    except Exception as e:
        logger.warning(f"Unexpected error with torch service: {e}. Using fallback service.")
    
    # Fallback to rule-based translation service
    try:
        from translation_service_fallback import translation_service as fallback_translation_service
        logger.info("Using fallback rule-based translation service")
        return fallback_translation_service(text, target_language)
        
    except ImportError:
        logger.error("Fallback translation service not found. Creating minimal fallback.")
        
        # Minimal fallback implementation
        return {
            "translated_text": text,  # Return original text
            "source_language": "auto",
            "target_language": target_language,
            "confidence": 0.0,
            "service_used": "minimal_fallback",
            "warning": "Translation service unavailable. Original text returned.",
            "success": False
        }

# Export the service function
__all__ = ['translation_service']

