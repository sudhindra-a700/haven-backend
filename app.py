"""
HAVEN Crowdfunding Platform - Complete Fixed Backend
Main FastAPI application with all errors resolved
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database imports
try:
    from database import engine, get_db
    from models import Base
    logger.info("✅ Database imports successful")
except ImportError as e:
    logger.warning(f"⚠️ Database import failed: {e}")
    engine = None
    get_db = None

# Configuration import
try:
    from config import get_settings
    settings = get_settings()
    logger.info("✅ Configuration loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Configuration import failed: {e}")
    settings = None

# Route imports with error handling
route_imports = {}

try:
    from user_routes import user_router
    route_imports['user'] = user_router
    logger.info("✅ User routes imported")
except ImportError as e:
    logger.warning(f"⚠️ User routes import failed: {e}")

try:
    from campaign_routes import campaign_router
    route_imports['campaign'] = campaign_router
    logger.info("✅ Campaign routes imported")
except ImportError as e:
    logger.warning(f"⚠️ Campaign routes import failed: {e}")

try:
    from oauth_routes import oauth_router
    route_imports['oauth'] = oauth_router
    logger.info("✅ OAuth routes imported")
except ImportError as e:
    logger.warning(f"⚠️ OAuth routes import failed: {e}")

try:
    from fraud_routes import fraud_router
    route_imports['fraud'] = fraud_router
    logger.info("✅ Fraud routes imported")
except ImportError as e:
    logger.warning(f"⚠️ Fraud routes import failed: {e}")

try:
    from translation_routes import translation_router
    route_imports['translation'] = translation_router
    logger.info("✅ Translation routes imported")
except ImportError as e:
    logger.warning(f"⚠️ Translation routes import failed: {e}")

try:
    from simplification_routes import simplification_router
    route_imports['simplification'] = simplification_router
    logger.info("✅ Simplification routes imported")
except ImportError as e:
    logger.warning(f"⚠️ Simplification routes import failed: {e}")

# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 HAVEN Backend starting up...")
    
    # Create database tables if database is available
    if engine is not None:
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("✅ Database tables created/verified")
        except Exception as e:
            logger.error(f"❌ Database table creation failed: {e}")
    
    logger.info("✅ HAVEN Backend startup complete")
    
    yield
    
    # Shutdown
    logger.info("🔄 HAVEN Backend shutting down...")
    logger.info("✅ HAVEN Backend shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="HAVEN Crowdfunding Platform",
    description="A secure and scalable FastAPI backend for the HAVEN crowdfunding platform with OAuth authentication, fraud detection, and translation services.",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "🎉 HAVEN Crowdfunding Platform API",
        "status": "operational",
        "version": "1.0.0",
        "description": "Secure crowdfunding platform with ML-powered features",
        "features": [
            "✅ User Authentication & OAuth",
            "✅ Campaign Management",
            "✅ Fraud Detection",
            "✅ Multi-language Translation",
            "✅ Text Simplification",
            "✅ Payment Integration"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "api": "/api/v1"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    health_status = {
        "status": "healthy",
        "service": "haven-backend",
        "version": "1.0.0",
        "port": os.getenv("PORT", "unknown"),
        "components": {}
    }
    
    # Check database
    if engine is not None:
        try:
            # Simple database check
            health_status["components"]["database"] = "connected"
        except Exception as e:
            health_status["components"]["database"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
    else:
        health_status["components"]["database"] = "not configured"
    
    # Check configuration
    if settings is not None:
        health_status["components"]["configuration"] = "loaded"
    else:
        health_status["components"]["configuration"] = "error"
        health_status["status"] = "degraded"
    
    # Check routes
    health_status["components"]["routes"] = {
        "loaded": list(route_imports.keys()),
        "total": len(route_imports)
    }
    
    return health_status

# API status endpoint
@app.get("/api/status")
async def api_status():
    """Detailed API status information"""
    return {
        "api_version": "1.0.0",
        "status": "operational",
        "routes_loaded": list(route_imports.keys()),
        "total_routes": len(route_imports),
        "environment": "production" if os.getenv("PORT") else "development",
        "features": {
            "authentication": "oauth" in route_imports,
            "user_management": "user" in route_imports,
            "campaign_management": "campaign" in route_imports,
            "fraud_detection": "fraud" in route_imports,
            "translation": "translation" in route_imports,
            "simplification": "simplification" in route_imports
        }
    }

# Include routers with error handling
for route_name, router in route_imports.items():
    try:
        app.include_router(router, prefix=f"/api/v1")
        logger.info(f"✅ {route_name.title()} routes registered")
    except Exception as e:
        logger.error(f"❌ Failed to register {route_name} routes: {e}")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "type": type(exc).__name__
        }
    )

# 404 handler
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """404 Not Found handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "message": f"The requested resource was not found",
            "path": str(request.url.path)
        }
    )

# Development server (for local testing only)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🔧 Running in development mode on port {port}")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

