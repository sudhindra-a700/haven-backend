"""
FIXED Main Application for HAVEN Crowdfunding Platform
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

# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting HAVEN Crowdfunding Platform API")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
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
                "http://localhost:8000"
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

# Include OAuth routes (already has /api/v1 prefix)
app.include_router(oauth_router, tags=["OAuth Authentication"])

# Include other route modules here
# app.include_router(campaign_router, prefix="/api/v1", tags=["Campaigns"])
# app.include_router(user_router, prefix="/api/v1", tags=["Users"])
# app.include_router(fraud_router, prefix="/api/v1", tags=["Fraud Detection"])

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
        "environment": os.getenv("ENVIRONMENT", "development"),
        "oauth_enabled": os.getenv("FEATURES_OAUTH_ENABLED", "false").lower() == "true"
    }

@app.get("/api/v1/config")
async def get_config():
    """Get public configuration"""
    return {
        "oauth_enabled": os.getenv("FEATURES_OAUTH_ENABLED", "false").lower() == "true",
        "translation_enabled": os.getenv("FEATURES_TRANSLATION_ENABLED", "false").lower() == "true",
        "simplification_enabled": os.getenv("FEATURES_SIMPLIFICATION_ENABLED", "false").lower() == "true",
        "fraud_detection_enabled": os.getenv("FEATURES_FRAUD_DETECTION_ENABLED", "false").lower() == "true",
        "frontend_url": os.getenv("FRONTEND_URL"),
        "environment": os.getenv("ENVIRONMENT", "development")
    }

# ===== ERROR HANDLERS =====

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The requested path {request.url.path} was not found",
            "available_endpoints": [
                "/",
                "/health",
                "/api/v1/config",
                "/api/v1/auth/google/login",
                "/api/v1/auth/facebook/login",
                "/api/v1/auth/status"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )

# ===== MIDDLEWARE =====

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    logger.info(f"{request.method} {request.url.path} - {request.client.host if request.client else 'unknown'}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

# ===== STARTUP VALIDATION =====

@app.on_event("startup")
async def validate_environment():
    """Validate environment configuration on startup"""
    required_vars = [
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET", 
        "FACEBOOK_APP_ID",
        "FACEBOOK_APP_SECRET",
        "FRONTEND_URL"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.warning("OAuth functionality may not work properly")
    else:
        logger.info("All required environment variables are configured")

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = "0.0.0.0"  # Always bind to all interfaces for deployment
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "fixed_app:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT") != "production",
        log_level="info"
    )

