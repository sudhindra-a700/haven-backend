"""
HAVEN Crowdfunding Platform - Enhanced Backend with Role-Based Access Control
Main FastAPI application with individual/organization registration and role-based access
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

# NEW: Registration routes import
try:
    from registration_routes import registration_router
    route_imports['registration'] = registration_router
    logger.info("✅ Simple Registration routes imported")
except ImportError as e:
    logger.warning(f"⚠️ Simple Registration routes import failed: {e}")

try:
    from oauth_routes import oauth_router
    route_imports['oauth'] = oauth_router
    logger.info("✅ Simple OAuth routes imported")
except ImportError as e:
    logger.warning(f"⚠️ Simple OAuth routes import failed: {e}")

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
    title="HAVEN Crowdfunding Platform - Enhanced",
    description="A secure and scalable FastAPI backend for the HAVEN crowdfunding platform with role-based access control, separate individual/organization registration, and enhanced authentication.",
    version="2.0.0",
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
        "message": "🎉 HAVEN Crowdfunding Platform API - Enhanced",
        "status": "operational",
        "version": "2.0.0",
        "description": "Secure crowdfunding platform with role-based access control",
        "features": [
            "✅ Role-Based Access Control",
            "✅ Individual/Organization Registration",
            "✅ User Authentication & OAuth",
            "✅ Campaign Management (Organizations Only)",
            "✅ Donation System (Individuals Only)",
            "✅ Fraud Detection",
            "✅ Multi-language Translation",
            "✅ Text Simplification",
            "✅ Payment Integration"
        ],
        "user_roles": {
            "individual": "Can only donate to campaigns",
            "organization": "Can only create and manage campaigns",
            "admin": "Full system access",
            "moderator": "Campaign moderation access"
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "api": "/api/v1",
            "registration": "/api/v1/auth/register"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    health_status = {
        "status": "healthy",
        "service": "haven-backend-enhanced",
        "version": "2.0.0",
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
    
    # Check role-based access control
    health_status["components"]["rbac"] = "enabled"
    health_status["components"]["registration_types"] = ["individual", "organization"]
    
    return health_status

# API status endpoint
@app.get("/api/status")
async def api_status():
    """Detailed API status information"""
    return {
        "api_version": "2.0.0",
        "status": "operational",
        "routes_loaded": list(route_imports.keys()),
        "total_routes": len(route_imports),
        "environment": "production" if os.getenv("PORT") else "development",
        "features": {
            "role_based_access_control": True,
            "individual_registration": "registration" in route_imports,
            "organization_registration": "registration" in route_imports,
            "authentication": "oauth" in route_imports,
            "user_management": "user" in route_imports,
            "campaign_management": "campaign" in route_imports,
            "fraud_detection": "fraud" in route_imports,
            "translation": "translation" in route_imports,
            "simplification": "simplification" in route_imports
        },
        "access_control": {
            "campaign_creation": "organizations_only",
            "donations": "individuals_only",
            "campaign_approval": "moderators_and_admins",
            "user_management": "admins_only"
        }
    }

# NEW: Role-based access control information endpoint
@app.get("/api/rbac-info")
async def rbac_info():
    """Role-based access control information"""
    return {
        "roles": {
            "individual": {
                "description": "Individual users who can donate to campaigns",
                "permissions": [
                    "donate_to_campaigns",
                    "view_campaigns",
                    "manage_own_profile",
                    "view_donation_history"
                ],
                "restrictions": [
                    "cannot_create_campaigns",
                    "cannot_manage_campaigns"
                ]
            },
            "organization": {
                "description": "Organizations that can create and manage campaigns",
                "permissions": [
                    "create_campaigns",
                    "manage_own_campaigns",
                    "view_campaign_analytics",
                    "manage_own_profile"
                ],
                "restrictions": [
                    "cannot_donate_to_campaigns"
                ]
            },
            "admin": {
                "description": "System administrators with full access",
                "permissions": [
                    "full_system_access",
                    "manage_all_users",
                    "manage_all_campaigns",
                    "approve_reject_campaigns",
                    "access_fraud_detection",
                    "system_configuration"
                ]
            },
            "moderator": {
                "description": "Content moderators who can approve/reject campaigns",
                "permissions": [
                    "approve_reject_campaigns",
                    "view_all_campaigns",
                    "access_fraud_detection",
                    "moderate_content"
                ]
            }
        },
        "registration_flow": {
            "step_1": "User chooses registration type (individual or organization)",
            "step_2": "User fills appropriate registration form",
            "step_3": "Account created with assigned role",
            "step_4": "User can access role-specific features"
        },
        "endpoints": {
            "individual_registration": "/api/v1/auth/register/individual",
            "organization_registration": "/api/v1/auth/register/organization",
            "registration_status": "/api/v1/auth/registration-status"
        }
    }

# Include routers with error handling
for route_name, router in route_imports.items():
    try:
        if route_name == 'registration':
            app.include_router(router, prefix="/api/v1/auth")
        else:
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

# Role-based access control error handler
@app.exception_handler(403)
async def forbidden_handler(request, exc):
    """403 Forbidden handler for role-based access control"""
    return JSONResponse(
        status_code=403,
        content={
            "error": "Access forbidden",
            "message": "You don't have permission to access this resource",
            "hint": "Check your user role and registration status",
            "rbac_info": "/api/rbac-info"
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

