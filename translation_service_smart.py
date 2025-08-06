"""
Smart Translation Service for HAVEN Platform
Automatically uses torch-based ML or fallback based on availability
"""

import logging

logger = logging.getLogger(__name__)

try:
    # Try to import torch-based translation
    import torch
    from translation_service import translation_service
    logger.info("Using torch-based translation service")
    
except ImportError:
    # Fall back to basic service
    from translation_service_fallback import translation_service
    logger.info("Using fallback translation service")

# Export the service
__all__ = ['translation_service']

