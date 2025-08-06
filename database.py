"""
Database Configuration and Connection Management
Secure database setup with connection pooling and migrations
"""

import logging
from typing import AsyncGenerator
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import asyncpg
from alembic import command
from alembic.config import Config

from config import get_settings, get_database_config

logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()
db_config = get_database_config()

# Database URL
DATABASE_URL = settings.database_url_constructed

# Default database connection pool configuration
DEFAULT_POOL_CONFIG = {
    "pool_size": 5,
    "max_overflow": 10,
    "pool_timeout": 30,
    "pool_recycle": 3600
}

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=DEFAULT_POOL_CONFIG["pool_size"],
    max_overflow=DEFAULT_POOL_CONFIG["max_overflow"],
    pool_timeout=DEFAULT_POOL_CONFIG["pool_timeout"],
    pool_recycle=DEFAULT_POOL_CONFIG["pool_recycle"],
    echo=settings.is_development,  # Log SQL queries in development
    future=True
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True
)

# Create declarative base
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()

class DatabaseManager:
    """Database connection and transaction management"""
    
    def __init__(self):
        self.engine = engine
        self.session_factory = SessionLocal
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.session_factory()
    
    async def init_database(self):
        """Initialize database and run migrations"""
        try:
            logger.info("🔄 Initializing database...")
            
            # Create all tables
            Base.metadata.create_all(bind=engine)
            
            # Run Alembic migrations if available
            try:
                alembic_cfg = Config("alembic.ini")
                command.upgrade(alembic_cfg, "head")
                logger.info("✅ Database migrations completed")
            except Exception as e:
                logger.warning(f"⚠️ Alembic migrations not available: {e}")
            
            logger.info("✅ Database initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    async def check_connection(self) -> bool:
        """Check database connection"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    def close_connections(self):
        """Close all database connections"""
        try:
            engine.dispose()
            logger.info("✅ Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

# Global database manager
db_manager = DatabaseManager()

# Dependency for getting database session
def get_db() -> AsyncGenerator[Session, None]:
    """Dependency for getting database session"""
    session = db_manager.get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()

# Database initialization function
async def init_db():
    """Initialize database"""
    await db_manager.init_database()

# Database health check
async def check_db_health() -> dict:
    """Check database health"""
    try:
        is_connected = await db_manager.check_connection()
        
        if is_connected:
            # Get additional database info
            with db_manager.get_session() as session:
                result = session.execute("""
                    SELECT 
                        version() as version,
                        current_database() as database,
                        current_user as user
                """).fetchone()
                
                return {
                    "status": "healthy",
                    "connected": True,
                    "version": result.version if result else "unknown",
                    "database": result.database if result else "unknown",
                    "user": result.user if result else "unknown"
                }
        else:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Connection failed"
            }
    
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)
        }

# Transaction context manager
class DatabaseTransaction:
    """Context manager for database transactions"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def __enter__(self):
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.session.rollback()
            logger.error(f"Transaction rolled back due to: {exc_val}")
        else:
            self.session.commit()

# Utility functions
def create_transaction(session: Session) -> DatabaseTransaction:
    """Create database transaction context"""
    return DatabaseTransaction(session)

def execute_raw_sql(query: str, params: dict = None) -> list:
    """Execute raw SQL query"""
    with db_manager.get_session() as session:
        try:
            result = session.execute(query, params or {})
            return result.fetchall()
        except Exception as e:
            logger.error(f"Raw SQL execution failed: {e}")
            raise

# Database backup and restore utilities
class DatabaseBackup:
    """Database backup and restore utilities"""
    
    @staticmethod
    def create_backup(backup_path: str):
        """Create database backup"""
        try:
            import subprocess
            
            # Extract database info from URL
            db_url_parts = DATABASE_URL.replace("postgresql://", "").split("/")
            db_name = db_url_parts[-1]
            
            # Create backup using pg_dump
            cmd = [
                "pg_dump",
                "--no-password",
                "--format=custom",
                "--file", backup_path,
                DATABASE_URL
            ]
            
            subprocess.run(cmd, check=True)
            logger.info(f"✅ Database backup created: {backup_path}")
            
        except Exception as e:
            logger.error(f"❌ Database backup failed: {e}")
            raise
    
    @staticmethod
    def restore_backup(backup_path: str):
        """Restore database from backup"""
        try:
            import subprocess
            
            # Restore using pg_restore
            cmd = [
                "pg_restore",
                "--no-password",
                "--clean",
                "--if-exists",
                "--dbname", DATABASE_URL,
                backup_path
            ]
            
            subprocess.run(cmd, check=True)
            logger.info(f"✅ Database restored from: {backup_path}")
            
        except Exception as e:
            logger.error(f"❌ Database restore failed: {e}")
            raise

# Connection pool monitoring
def get_pool_status() -> dict:
    """Get connection pool status"""
    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "invalid": pool.invalid()
    }


