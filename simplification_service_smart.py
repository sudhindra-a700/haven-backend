"""
Smart Simplification Service with Enhanced Error Handling
Handles torch compatibility issues and version conflicts gracefully
"""

import logging
import sys
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simplification_service(text: str, complexity_level: str = "simple") -> Dict[str, Any]:
    """
    Smart text simplification service with robust fallback handling
    
    Args:
        text (str): Text to simplify
        complexity_level (str): Target complexity level (default: "simple")
    
    Returns:
        Dict[str, Any]: Simplification result with metadata
    """
    
    # First, try to use torch-based simplification service
    try:
        # Import torch and check for compatibility
        import torch
        
        # Check if torch has required attributes
        if not hasattr(torch, 'uint64'):
            logger.warning("Torch version incompatible (missing uint64). Using fallback service.")
            raise ImportError("Torch version incompatible")
        
        # Try to import transformers
        from transformers import pipeline
        
        # Import the original simplification service
        from simplification_service import simplification_service as torch_simplification_service
        
        logger.info("Using torch-based simplification service")
        return torch_simplification_service(text, complexity_level)
        
    except ImportError as e:
        logger.warning(f"Torch/Transformers import failed: {e}. Using fallback service.")
    except RuntimeError as e:
        if "torch" in str(e).lower() or "transformers" in str(e).lower():
            logger.warning(f"Torch/Transformers runtime error: {e}. Using fallback service.")
        else:
            raise e
    except Exception as e:
        logger.warning(f"Unexpected error with torch service: {e}. Using fallback service.")
    
    # Fallback to rule-based simplification service
    try:
        from simplification_service_fallback import simplification_service as fallback_simplification_service
        logger.info("Using fallback rule-based simplification service")
        return fallback_simplification_service(text, complexity_level)
        
    except ImportError:
        logger.error("Fallback simplification service not found. Creating minimal fallback.")
        
        # Minimal fallback implementation
        return {
            "simplified_text": text,  # Return original text
            "original_complexity": "unknown",
            "target_complexity": complexity_level,
            "simplification_score": 0.0,
            "service_used": "minimal_fallback",
            "warning": "Simplification service unavailable. Original text returned.",
            "success": False
        }

# Export the service function
__all__ = ['simplification_service']

