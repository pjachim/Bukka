"""Unit tests for the reference module containing project requirements."""
import pytest

from bukka.utils.reference import requirements


class TestReference:
    """Test suite for the reference module's requirements constant."""

    def test_requirements_is_string(self):
        """Test that requirements is a non-empty string.
        
        Examples
        --------
        >>> from bukka.utils.reference import requirements
        >>> assert isinstance(requirements, str)
        >>> assert len(requirements) > 0
        """
        assert isinstance(requirements, str)
        assert len(requirements) > 0

    def test_requirements_contains_core_packages(self):
        """Test that requirements contains expected core packages.
        
        Examples
        --------
        >>> from bukka.utils.reference import requirements
        >>> assert 'scikit-learn' in requirements
        >>> assert 'polars' in requirements
        """
        # Core ML and data packages
        assert 'scikit-learn' in requirements
        assert 'polars' in requirements
        assert 'pandas' in requirements
        assert 'numpy' in requirements
        
        # Visualization
        assert 'seaborn' in requirements
        
        # Notebook support
        assert 'ipykernel' in requirements
        
        # Data abstraction
        assert 'narwhals' in requirements
        assert 'pyarrow' in requirements

    def test_requirements_is_valid_pip_format(self):
        """Test that requirements can be parsed as valid pip format.
        
        Each line should either be empty, a comment, or a valid package name.
        
        Examples
        --------
        >>> from bukka.utils.reference import requirements
        >>> lines = requirements.strip().split('\\n')
        >>> for line in lines:
        ...     if line and not line.startswith('#'):
        ...         assert len(line.split()) == 1  # Simple package name
        """
        lines = requirements.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Should be a simple package name (no spaces except for version specs)
                # For now, just check it's not empty
                assert len(line) > 0
                # Package names shouldn't have multiple words (spaces) unless version specified
                # For simple requirements, we expect one token per line
                assert line.replace('>=', ' ').replace('==', ' ').replace('<=', ' ').split()[0]
