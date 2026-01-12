"""
logging.py
----------
Logging utilities for LayerLens.

Provides structured logging with different log levels and optional file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_logger: Optional[logging.Logger] = None


def setup_logger(
    name: str = "layerlens",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up and configure the LayerLens logger.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        format_string: Optional custom format string
        
    Returns:
        Configured logger instance
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    
    # Prevent duplicate handlers
    if _logger.handlers:
        return _logger
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)
    
    return _logger


def get_logger(name: str = "layerlens") -> logging.Logger:
    """
    Get the LayerLens logger instance.
    
    If logger is not set up, creates a default one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    global _logger
    
    if _logger is None:
        return setup_logger(name=name)
    
    return logging.getLogger(name)

