import logging
import os
from typing import Optional

def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """
    Configure logging based on WARN_LOG_LEVEL environment variable.
    Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    Defaults to INFO if not set or invalid.
    
    Args:
        name: Optional name for the logger. If None, returns root logger.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get log level from environment variable
    log_level_str = os.getenv('WARN_LOG_LEVEL', 'INFO').upper()
    
    # Map of valid log levels
    valid_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    # Set default level if invalid level specified
    log_level = valid_levels.get(log_level_str, logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Configure handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Set level
    logger.setLevel(log_level)
    
    return logger