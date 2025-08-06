"""
Smart Simplification Service for HAVEN Platform
Automatically uses torch-based ML or fallback based on availability
"""

import logging

logger = logging.getLogger(__name__)

try:
    # Try to import torch-based simplification
    import torch
    from simplification_service import simplification_service
    logger.info("Using torch-based simplification service")
    
except ImportError:
    # Fall back to basic service
    from simplification_service_fallback import simplification_service
    logger.info("Using fallback simplification service")

# Export the service
__all__ = ['simplification_service']

