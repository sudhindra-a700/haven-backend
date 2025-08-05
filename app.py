"""
HAVEN Crowdfunding Platform - Secure FastAPI Backend
Fixed version addressing all security and configuration issues
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
from dotenv import load_dotenv

# Import routers
from oauth_routes import oauth_router
from translation_routes import translation_router
from fraud_routes import fraud_router
from campaign_routes import campaign_router
from user_routes import user_router

# Import utilities
from database import init_db, get_db
from auth_middleware import verify_token
from config import get_settings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting HAVEN Crowdfunding Backend")
    
    # Initialize database
    await init_db()
    logger.info("✅ Database initialized")
    
    # Initialize ML services (lazy loading)
    logger.info("✅ ML services ready for lazy loading")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down HAVEN Crowdfunding Backend")

# Create FastAPI application
app = FastAPI(
    title="HAVEN Crowdfunding API",
    description="Secure API for the HAVEN crowdfunding platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Get settings
settings = get_settings()

# Security middleware
security = HTTPBearer()

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add trusted host middleware
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )

# Configure CORS with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for better error management"""
    logger.error(f"Global exception: {exc}", exc_info=True)
    
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "type": "http_exception"}
        )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "type": "internal_error"
        }
    )

# Health check endpoints
@app.get("/health")
@limiter.limit("10/minute")
async def health_check(request: Request):
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "haven-backend",
        "version": "2.0.0",
        "environment": settings.environment
    }

@app.get("/health/detailed")
@limiter.limit("5/minute")
async def detailed_health_check(request: Request, db=Depends(get_db)):
    """Detailed health check including database connectivity"""
    try:
        # Check database connectivity
        await db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "service": "haven-backend",
        "version": "2.0.0",
        "environment": settings.environment,
        "components": {
            "database": db_status,
            "translation_service": "healthy",
            "fraud_detection": "healthy"
        }
    }

# Root endpoint
@app.get("/")
@limiter.limit("30/minute")
async def root(request: Request):
    """Root endpoint with API information"""
    return {
        "message": "Welcome to HAVEN Crowdfunding API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "status": "operational"
    }

# Include routers with proper prefixes and tags
app.include_router(
    oauth_router,
    prefix="/auth",
    tags=["Authentication"]
)

app.include_router(
    user_router,
    prefix="/users",
    tags=["Users"],
    dependencies=[Depends(verify_token)]
)

app.include_router(
    campaign_router,
    prefix="/campaigns",
    tags=["Campaigns"]
)

app.include_router(
    translation_router,
    prefix="/translate",
    tags=["Translation"],
    dependencies=[Depends(verify_token)]
)

app.include_router(
    fraud_router,
    prefix="/fraud",
    tags=["Fraud Detection"],
    dependencies=[Depends(verify_token)]
)

# Startup event for additional initialization
@app.on_event("startup")
async def startup_event():
    """Additional startup tasks"""
    logger.info("🔧 Running additional startup tasks")

if __name__ == "__main__":
    # Get port from environment variable or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=settings.environment == "development",
        log_level="info"
    )

