"""
Logging configuration for containerized LLMWebUI application.
Provides environment-specific logging setups for development and production.
"""

import os
import sys
import logging
import logging.config
from pathlib import Path
from typing import Dict, Any


def get_logging_config(environment: str = None) -> Dict[str, Any]:
    """
    Get logging configuration based on environment.
    
    Args:
        environment: 'development', 'production', or None (auto-detect)
    
    Returns:
        Dictionary containing logging configuration
    """
    if environment is None:
        environment = os.getenv('ENVIRONMENT', 'development')
    
    # Ensure logs directory exists
    logs_dir = Path('/app/logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Base configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(name)s - %(message)s'
            },
            'json': {
                'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}',
                'datefmt': '%Y-%m-%dT%H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'stream': sys.stdout
            }
        },
        'loggers': {
            '': {  # Root logger
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'uvicorn': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'uvicorn.access': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'streamlit': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            }
        }
    }
    
    if environment == 'development':
        # Development: More verbose logging, file output for debugging
        config['handlers'].update({
            'file_debug': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': '/app/logs/debug.log',
                'maxBytes': 50 * 1024 * 1024,  # 50MB
                'backupCount': 5,
                'encoding': 'utf-8'
            },
            'file_error': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': '/app/logs/error.log',
                'maxBytes': 10 * 1024 * 1024,  # 10MB
                'backupCount': 3,
                'encoding': 'utf-8'
            }
        })
        
        # Update console handler for development
        config['handlers']['console'].update({
            'level': 'DEBUG',
            'formatter': 'detailed'
        })
        
        # Update loggers for development
        config['loggers']['']['level'] = 'DEBUG'
        config['loggers']['']['handlers'] = ['console', 'file_debug', 'file_error']
        config['loggers']['uvicorn']['level'] = 'DEBUG'
        config['loggers']['uvicorn']['handlers'] = ['console', 'file_debug']
        config['loggers']['uvicorn.access']['level'] = 'DEBUG'
        config['loggers']['uvicorn.access']['handlers'] = ['console', 'file_debug']
        
    elif environment == 'production':
        # Production: Structured JSON logging, error tracking
        config['handlers'].update({
            'file_app': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json',
                'filename': '/app/logs/application.log',
                'maxBytes': 100 * 1024 * 1024,  # 100MB
                'backupCount': 10,
                'encoding': 'utf-8'
            },
            'file_error': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json',
                'filename': '/app/logs/error.log',
                'maxBytes': 50 * 1024 * 1024,  # 50MB
                'backupCount': 5,
                'encoding': 'utf-8'
            },
            'file_access': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json',
                'filename': '/app/logs/access.log',
                'maxBytes': 100 * 1024 * 1024,  # 100MB
                'backupCount': 5,
                'encoding': 'utf-8'
            }
        })
        
        # Update console for production (JSON format)
        config['handlers']['console'].update({
            'formatter': 'json'
        })
        
        # Update loggers for production
        config['loggers']['']['handlers'] = ['console', 'file_app', 'file_error']
        config['loggers']['uvicorn']['handlers'] = ['console', 'file_app']
        config['loggers']['uvicorn.access']['handlers'] = ['file_access']
        
        # Add application-specific loggers
        config['loggers'].update({
            'app': {
                'level': 'INFO',
                'handlers': ['console', 'file_app', 'file_error'],
                'propagate': False
            },
            'app.services': {
                'level': 'INFO',
                'handlers': ['console', 'file_app'],
                'propagate': False
            },
            'app.ingestion': {
                'level': 'INFO',
                'handlers': ['console', 'file_app'],
                'propagate': False
            }
        })
    
    return config


def setup_logging(environment: str = None) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        environment: 'development', 'production', or None (auto-detect)
    """
    config = get_logging_config(environment)
    logging.config.dictConfig(config)
    
    # Log the configuration setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured for environment: {environment or os.getenv('ENVIRONMENT', 'development')}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Container-specific logging utilities
class ContainerLogFilter(logging.Filter):
    """Filter to add container-specific information to log records."""
    
    def filter(self, record):
        # Add container information
        record.container_id = os.getenv('HOSTNAME', 'unknown')
        record.environment = os.getenv('ENVIRONMENT', 'unknown')
        record.service = os.getenv('SERVICE_NAME', 'llmwebui')
        return True


def add_container_context():
    """Add container context to all loggers."""
    container_filter = ContainerLogFilter()
    
    # Add filter to root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(container_filter)


# Health check logging
def setup_health_check_logging():
    """Setup minimal logging for health checks to avoid log spam."""
    health_logger = logging.getLogger('health_check')
    health_logger.setLevel(logging.WARNING)  # Only log warnings and errors
    return health_logger


if __name__ == "__main__":
    # Test logging configuration
    import sys
    
    env = sys.argv[1] if len(sys.argv) > 1 else 'development'
    setup_logging(env)
    
    logger = get_logger(__name__)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    print(f"Logging test completed for environment: {env}")