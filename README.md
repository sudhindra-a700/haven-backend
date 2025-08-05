# HAVEN Crowdfunding Platform - Backend

A secure and scalable FastAPI backend for the HAVEN crowdfunding platform with OAuth authentication, fraud detection, and translation services.

## Features

- **Secure Authentication**: JWT-based authentication with OAuth support (Google, Facebook)
- **Campaign Management**: Full CRUD operations for crowdfunding campaigns
- **Fraud Detection**: AI-powered fraud detection for campaigns
- **Translation Services**: Multi-language support with text translation
- **Text Simplification**: Simplify complex terms for better understanding
- **Rate Limiting**: API rate limiting for security and performance
- **Database Integration**: PostgreSQL with SQLAlchemy ORM
- **File Upload**: Secure file upload with validation
- **Email Integration**: Email notifications and verification
- **Comprehensive API**: RESTful API with OpenAPI documentation

## Technology Stack

- **Framework**: FastAPI 0.104.1
- **Database**: PostgreSQL with SQLAlchemy 2.0
- **Authentication**: JWT with OAuth2 (Google, Facebook)
- **Caching**: Redis for session and data caching
- **ML Services**: DistillBERT for fraud detection
- **Translation**: IndicTrans@ trained under IndiCorp dataset
- **File Storage**: Local filesystem with cloud storage support
- **Deployment**: Docker with cloud platform support

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 12+
- Redis 6+ (optional, for caching)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fixed_backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Database setup**
   ```bash
   # Create database
   createdb haven_db
   
   # Run migrations
   alembic upgrade head
   ```

6. **Start the server**
   ```bash
   uvicorn app:app --reload
   ```

The API will be available at `http://localhost:8000`

## Configuration

### Environment Variables

Key environment variables (see `.env.example` for complete list):

- `DATABASE_URL`: PostgreSQL connection string
- `SECRET_KEY`: JWT secret key
- `GOOGLE_CLIENT_ID`: Google OAuth client ID
- `FACEBOOK_APP_ID`: Facebook OAuth app ID
- `ALLOWED_ORIGINS`: CORS allowed origins

### OAuth Setup

#### Google OAuth
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google+ API
4. Create OAuth 2.0 credentials
5. Add authorized redirect URIs: `http://localhost:8000/auth/google/callback`

#### Facebook OAuth
1. Go to [Facebook Developers](https://developers.facebook.com/)
2. Create a new app
3. Add Facebook Login product
4. Configure Valid OAuth Redirect URIs: `http://localhost:8000/auth/facebook/callback`

## API Documentation

### Interactive Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Main Endpoints

#### Authentication
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `GET /auth/google` - Google OAuth
- `GET /auth/facebook` - Facebook OAuth
- `POST /auth/logout` - User logout
- `POST /auth/refresh` - Refresh token

#### Campaigns
- `GET /campaigns/` - List campaigns
- `POST /campaigns/` - Create campaign
- `GET /campaigns/{id}` - Get campaign
- `PUT /campaigns/{id}` - Update campaign
- `DELETE /campaigns/{id}` - Delete campaign
- `POST /campaigns/{id}/donate` - Donate to campaign

#### Users
- `GET /users/me` - Get current user
- `PUT /users/me` - Update profile
- `GET /users/me/stats` - User statistics

#### Translation
- `POST /translate/translate` - Translate text
- `POST /translate/detect-language` - Detect language
- `GET /translate/languages` - Supported languages

#### Simplification
- `POST /simplify/simplify` - Simplify text
- `POST /simplify/explain-term` - Explain term
- `GET /simplify/categories` - Simplification categories

#### Fraud Detection
- `POST /fraud/analyze` - Analyze for fraud
- `POST /fraud/report` - Report fraud

## Database Schema

### Core Models
- **User**: User accounts and profiles
- **Campaign**: Crowdfunding campaigns
- **Donation**: Campaign donations
- **Comment**: Campaign comments
- **CampaignUpdate**: Campaign progress updates

### Enums
- **UserRole**: user, moderator, admin
- **CampaignStatus**: draft, pending, active, completed, cancelled
- **CampaignCategory**: education, healthcare, environment, etc.

## Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: Bcrypt password hashing
- **Rate Limiting**: API rate limiting per user/IP
- **CORS Protection**: Configurable CORS policies
- **Input Validation**: Pydantic model validation
- **SQL Injection Protection**: SQLAlchemy ORM protection
- **XSS Protection**: Input sanitization

## Deployment

### Docker Deployment

1. **Build image**
   ```bash
   docker build -t haven-backend .
   ```

2. **Run container**
   ```bash
   docker run -p 8000:8000 --env-file .env haven-backend
   ```

### Cloud Deployment (Render)

1. **Connect repository** to Render
2. **Set environment variables** in Render dashboard
3. **Deploy** with build command: `pip install -r requirements.txt`
4. **Start command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### Environment Variables for Production

Required environment variables for deployment:
```
DATABASE_URL=postgresql://...
SECRET_KEY=...
JWT_SECRET_KEY=...
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
FACEBOOK_APP_ID=...
FACEBOOK_APP_SECRET=...
ALLOWED_ORIGINS=https://your-frontend-domain.com
```

## Development

### Code Structure
```
fixed_backend/
├── app.py                 # Main FastAPI application
├── config.py             # Configuration management
├── database.py           # Database connection
├── models.py             # SQLAlchemy models
├── auth_middleware.py    # Authentication middleware
├── oauth_config.py       # OAuth configuration
├── oauth_routes.py       # OAuth endpoints
├── user_routes.py        # User endpoints
├── campaign_routes.py    # Campaign endpoints
├── translation_routes.py # Translation endpoints
├── simplification_routes.py # Simplification endpoints
├── fraud_routes.py       # Fraud detection endpoints
├── translation_service.py # Translation service
├── simplification_service.py # Simplification service
├── fraud_detection_service.py # Fraud detection service
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
└── .env.example         # Environment template
```

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## Monitoring and Logging

### Health Checks
- `GET /health` - Application health status
- `GET /translate/health` - Translation service health
- `GET /simplify/health` - Simplification service health
- `GET /fraud/health` - Fraud detection service health

### Logging
- Structured JSON logging
- Configurable log levels
- Request/response logging
- Error tracking with Sentry (optional)

## Performance Optimization

- **Database Connection Pooling**: Optimized connection management
- **Caching**: Redis caching for frequently accessed data
- **Async Operations**: FastAPI async support
- **Rate Limiting**: Prevent abuse and ensure fair usage
- **Pagination**: Efficient data pagination
- **Query Optimization**: Optimized database queries

## Security Best Practices

- **Environment Variables**: Sensitive data in environment variables
- **HTTPS Only**: Force HTTPS in production
- **Secure Headers**: Security headers middleware
- **Input Validation**: Comprehensive input validation
- **Error Handling**: Secure error messages
- **Audit Logging**: Track sensitive operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Email: support@haven.org
- Documentation: [API Docs](http://localhost:8000/docs)
- Issues: GitHub Issues

## Changelog

### v1.0.0
- Initial release
- Core authentication and campaign management
- OAuth integration (Google, Facebook)
- Fraud detection service
- Translation and simplification services
- Comprehensive API documentation

