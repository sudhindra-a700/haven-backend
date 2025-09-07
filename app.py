""" FIXED Main Application for HAVEN Crowdfunding Platform
This file contains the corrected FastAPI application that fixes:
1. CORS configuration issues
2. Proper route inclusion with API prefix
3. Environment variable handling
4. Uvicorn binding configuration
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import logging
from contextlib import asynccontextmanager

# Import the fixed OAuth routes
from oauth_routes import router as oauth_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting HAVEN Crowdfunding Platform API")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'Not set')}")
    logger.info(f"Frontend URL: {os.getenv('FRONTEND_URL', 'Not set')}")
    logger.info(f"Backend URL: {os.getenv('BACKEND_URL', 'Not set')}")
    yield
    logger.info("Shutting down HAVEN Crowdfunding Platform API")

# Create FastAPI application
app = FastAPI(
    title="HAVEN Crowdfunding Platform API",
    description="Secure and scalable FastAPI backend for the HAVEN crowdfunding platform",
    version="1.0.0",
    lifespan=lifespan
)

# ===== CORS CONFIGURATION =====

def get_allowed_origins():
    """Get allowed origins from environment variables"""
    allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
    
    if allowed_origins_env:
        # Split by comma and strip whitespace
        origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
    else:
        # Fallback to individual URL environment variables
        origins = []
        frontend_url = os.getenv("FRONTEND_URL")
        backend_url = os.getenv("BACKEND_URL")
        
        if frontend_url:
            origins.append(frontend_url)
        if backend_url:
            origins.append(backend_url)
        
        # Add localhost for development
        if os.getenv("ENVIRONMENT") != "production":
            origins.extend([
                "http://localhost:3000",
                "http://localhost:8501",
            ])
    
    # If no origins configured, allow all (not recommended for production)
    if not origins:
        logger.warning("No CORS origins configured, allowing all origins")
        origins = ["*"]
    
    logger.info(f"CORS allowed origins: {origins}")
    return origins

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ===== ROUTE INCLUSION =====

# Include OAuth routes (FIXED - uncommented and corrected prefix)
app.include_router(oauth_router, prefix="/api/v1/auth", tags=["OAuth Authentication"])

# ===== ROOT ROUTES =====

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HAVEN Crowdfunding Platform API",
        "version": "1.0.0",
        "status": "running",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "HAVEN Backend API is running",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "available_endpoints": {
            "root": "/",
            "health": "/health",
            "docs": "/docs",
            "oauth_google_login": "/api/v1/auth/google/login",
            "oauth_facebook_login": "/api/v1/auth/facebook/login",
            "oauth_status": "/api/v1/auth/status"
        }
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

