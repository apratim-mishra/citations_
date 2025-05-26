"""
Logging utilities for the project.
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

def setup_logger(
    name: str = "citation_graph",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Setup and configure a logger.
    
    Args:
        name: Logger name
        log_file: Path to log file (if None, only console logging is used)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output logs to console
        log_format: Log message format
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Add handlers
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        
        # Add file handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent log propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger

def get_logger(name: str = "citation_graph") -> logging.Logger:
    """
    Get an existing logger by name or create a new one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up a new one
    if not logger.handlers:
        return setup_logger(name)
    
    return logger

class LogCapture:
    """Context manager to capture and return log messages."""
    
    def __init__(self, logger_name: str = "citation_graph", level: int = logging.INFO):
        """
        Initialize log capture.
        
        Args:
            logger_name: Name of the logger to capture
            level: Minimum logging level to capture
        """
        self.logger_name = logger_name
        self.level = level
        self.captured_logs = []
        self.old_handlers = []
        self.logger = logging.getLogger(logger_name)
    
    def __enter__(self):
        """Set up log capture when entering context."""
        # Save existing handlers
        self.old_handlers = self.logger.handlers.copy()
        
        # Remove existing handlers
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
        
        # Add custom handler to capture logs
        class LogCaptureHandler(logging.Handler):
            def __init__(self, log_list):
                super().__init__()
                self.log_list = log_list
            
            def emit(self, record):
                self.log_list.append(self.format(record))
        
        capture_handler = LogCaptureHandler(self.captured_logs)
        capture_handler.setLevel(self.level)
        capture_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(capture_handler)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original handlers when exiting context."""
        # Remove capture handler
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
        
        # Restore original handlers
        for handler in self.old_handlers:
            self.logger.addHandler(handler)
    
    def get_logs(self) -> list:
        """
        Get captured log messages.
        
        Returns:
            List of captured log messages
        """
        return self.captured_logs