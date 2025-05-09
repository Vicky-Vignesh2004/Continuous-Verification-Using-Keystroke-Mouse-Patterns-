import os
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Base configuration class with default settings"""
    
    # Application settings
    PROJECT_ROOT = Path(__file__).parent.parent
    SECRET_KEY = os.getenv('SECRET_KEY', os.urandom(32).hex())
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    TESTING = False
    
    # Database configuration
    DATABASE_CONFIG: Dict[str, Any] = {
        'engine': 'sqlite',
        'name': os.getenv('DB_NAME', 'keystroke_auth.db'),
        'user': os.getenv('DB_USER', ''),
        'password': os.getenv('DB_PASSWORD', ''),
        'host': os.getenv('DB_HOST', ''),
        'port': os.getenv('DB_PORT', ''),
        'pool_size': 10,
        'max_overflow': 5,
        'pool_timeout': 30,
        'pool_recycle': 3600,
        'echo': DEBUG
    }
    
    # Security settings
    SECURITY_CONFIG: Dict[str, Any] = {
        # Password policies
        'password_min_length': 12,
        'password_complexity': {
            'min_lowercase': 1,
            'min_uppercase': 1,
            'min_digits': 1,
            'min_special': 1
        },
        'password_history_size': 5,
        
        # Session security
        'session_timeout': timedelta(minutes=30),
        'session_cookie_name': 'auth_session',
        'session_cookie_secure': not DEBUG,
        'session_cookie_httponly': True,
        'session_cookie_samesite': 'Lax',
        
        # Authentication
        'max_login_attempts': 5,
        'lockout_duration': timedelta(minutes=15),
        'brute_force_protection': True,
        'brute_force_window': timedelta(minutes=5),
        'brute_force_threshold': 10,
        
        # CSRF protection
        'csrf_timeout': timedelta(hours=12),
        'csrf_cookie_name': 'auth_csrf',
        'csrf_cookie_secure': not DEBUG,
        
        # Rate limiting
        'rate_limits': {
            'login': '10 per minute',
            'signup': '5 per hour',
            'api': '100 per minute'
        },
        
        # Behavioral authentication thresholds
        'behavioral_thresholds': {
            'keystroke_confidence': 0.7,
            'mouse_confidence': 0.6,
            'composite_confidence': 0.75,
            'max_anomaly_score': -0.3
        }
    }
    
    # File storage settings
    FILE_STORAGE_CONFIG: Dict[str, Any] = {
        'base_path': 'secured_files',
        'max_file_size': 50 * 1024 * 1024,  # 50MB
        'allowed_extensions': {
            '.txt', '.pdf', '.jpg', '.jpeg', '.png',
            '.doc', '.docx', '.xls', '.xlsx', '.csv', '.json'
        },
        'scan_uploads': True,
        'quota_per_user': 1024 * 1024 * 1024  # 1GB
    }
    
    # Behavioral model settings
    MODEL_CONFIG: Dict[str, Any] = {
        'keystroke': {
            'min_samples': 50,
            'dbscan_eps': 0.5,
            'dbscan_min_samples': 5,
            'contamination': 0.05
        },
        'mouse': {
            'min_samples': 100,
            'dbscan_eps': 0.7,
            'dbscan_min_samples': 10,
            'contamination': 0.1
        },
        'retrain_interval': timedelta(days=7),
        'model_versioning': True
    }
    
    # Logging configuration
    LOGGING_CONFIG: Dict[str, Any] = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': 'auth_system.log',
                'maxBytes': 1024 * 1024 * 5,  # 5MB
                'backupCount': 5,
                'formatter': 'standard'
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': True
            },
            'security': {
                'handlers': ['file'],
                'level': 'WARNING',
                'propagate': False
            }
        }
    }
    
    @classmethod
    def init_app(cls, app):
        """Initialize configuration with Flask app"""
        app.config.from_object(cls)
        
        # Set additional Flask-specific settings
        app.config.update({
            'SQLALCHEMY_DATABASE_URI': cls.get_db_uri(),
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'PERMANENT_SESSION_LIFETIME': cls.SECURITY_CONFIG['session_timeout'],
            'SESSION_COOKIE_SECURE': cls.SECURITY_CONFIG['session_cookie_secure'],
            'SESSION_COOKIE_HTTPONLY': cls.SECURITY_CONFIG['session_cookie_httponly'],
            'SESSION_COOKIE_SAMESITE': cls.SECURITY_CONFIG['session_cookie_samesite']
        })
    
    @classmethod
    def get_db_uri(cls) -> str:
        """Generate database URI based on configuration"""
        db_config = cls.DATABASE_CONFIG
        
        if db_config['engine'] == 'sqlite':
            return f"sqlite:///{Path(cls.PROJECT_ROOT, db_config['name'])}"
        elif db_config['engine'] == 'postgresql':
            return (f"postgresql://{db_config['user']}:{db_config['password']}"
                    f"@{db_config['host']}:{db_config['port']}/{db_config['name']}")
        elif db_config['engine'] == 'mysql':
            return (f"mysql+pymysql://{db_config['user']}:{db_config['password']}"
                    f"@{db_config['host']}:{db_config['port']}/{db_config['name']}")
        else:
            raise ValueError(f"Unsupported database engine: {db_config['engine']}")

class DevelopmentConfig(Config):
    """Development-specific configuration"""
    DEBUG = True
    DATABASE_CONFIG = {
        **Config.DATABASE_CONFIG,
        'echo': True
    }
    SECURITY_CONFIG = {
        **Config.SECURITY_CONFIG,
        'session_cookie_secure': False
    }



class ProductionConfig(Config):
    """Production-specific configuration"""
    DEBUG = False
    SECRET_KEY = os.getenv('SECRET_KEY')
    
    DATABASE_CONFIG = {
        **Config.DATABASE_CONFIG,
        'engine': os.getenv('DB_ENGINE', 'postgresql'),
        'name': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
        'echo': False
    }
    
    SECURITY_CONFIG = {
        **Config.SECURITY_CONFIG,
        'password_min_length': 16,
        'session_timeout': timedelta(minutes=15),
        'brute_force_threshold': 5
    }

# Configuration selector
configs = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(config_name: str = None) -> Config:
    """Get the appropriate configuration class"""
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'default')
    return configs.get(config_name, configs['default'])