"""Unit tests for config module constants."""
import pytest

from bukka.config import requirements, LOG_LEVEL
import logging


class TestConfig:
    """Test suite for the config module's constants."""

    def test_requirements_is_string(self):
        """Test that requirements constant is a non-empty string.
        
        Examples
        --------
        >>> from bukka.config import requirements
        >>> assert isinstance(requirements, str)
        >>> assert len(requirements) > 0
        """
        assert isinstance(requirements, str)
        assert len(requirements) > 0

    def test_requirements_contains_core_packages(self):
        """Test that requirements contains expected core ML packages.
        
        Examples
        --------
        >>> from bukka.config import requirements
        >>> assert 'scikit-learn' in requirements
        >>> assert 'narwhals' in requirements
        """
        # Core ML and data packages
        assert 'scikit-learn' in requirements
        assert 'narwhals' in requirements
        # 'pyarrow' is optional and may not be present
        assert 'numpy' in requirements
        
        # Visualization
        assert 'seaborn' in requirements
        
        # Notebook support
        assert 'ipykernel' in requirements
        
        # Data abstraction
        assert 'narwhals' in requirements

    def test_log_level_is_valid(self):
        """Test that LOG_LEVEL is a valid logging level.
        
        Examples
        --------
        >>> from bukka.config import LOG_LEVEL
        >>> import logging
        >>> assert LOG_LEVEL in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
        """
        valid_levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL
        ]
        assert LOG_LEVEL in valid_levels

    def test_log_level_is_debug(self):
        """Test that LOG_LEVEL is set to DEBUG as per current configuration.
        
        Examples
        --------
        >>> from bukka.config import LOG_LEVEL
        >>> import logging
        >>> assert LOG_LEVEL == logging.DEBUG
        """
        assert LOG_LEVEL == logging.DEBUG
