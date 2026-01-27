"""Categorical data preprocessing functions for the Bukka ML scaffolder.

This module provides utilities for standardizing and encoding categorical data
in machine learning pipelines.
"""

from typing import Any
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class standardize_categories(BaseEstimator, TransformerMixin):
    """Standardize inconsistent categorical values.
    
    This transformer fixes common inconsistencies in categorical data:
    - Normalizes case (converts to lowercase by default)
    - Trims whitespace
    - Can apply custom mapping for known typos/variants
    
    Parameters
    ----------
    case : str, optional
        Case normalization strategy. Options: 'lower', 'upper', 'title', None.
        Defaults to 'lower'.
    strip_whitespace : bool, optional
        Whether to strip leading/trailing whitespace. Defaults to True.
    custom_mapping : dict[str, str] | None, optional
        Dictionary mapping variant spellings to standard forms.
        Applied after case/whitespace normalization. Defaults to None.
    
    Attributes
    ----------
    case : str
        The case normalization strategy.
    strip_whitespace : bool
        Whether to strip whitespace.
    custom_mapping : dict[str, str] | None
        The custom mapping dictionary.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> data = pd.DataFrame({'category': ['Cat', 'cat ', 'CAT', 'Dog', ' dog']})
    >>> transformer = standardize_categories(case='lower', strip_whitespace=True)
    >>> transformed = transformer.fit_transform(data[['category']])
    >>> list(transformed['category'])
    ['cat', 'cat', 'cat', 'dog', 'dog']
    
    >>> # With custom mapping
    >>> data = pd.DataFrame({'color': ['red', 'RED', 'r3d', 'blue', 'BLU']})
    >>> mapping = {'r3d': 'red', 'blu': 'blue'}
    >>> transformer = standardize_categories(case='lower', custom_mapping=mapping)
    >>> transformed = transformer.fit_transform(data[['color']])
    >>> list(transformed['color'])
    ['red', 'red', 'red', 'blue', 'blue']
    """
    
    def __init__(self, case: str | None = 'lower', strip_whitespace: bool = True, 
                 custom_mapping: dict[str, str] | None = None):
        self.case = case
        self.strip_whitespace = strip_whitespace
        self.custom_mapping = custom_mapping or {}
    
    def fit(self, X: Any, y: Any = None) -> 'standardize_categories':
        """Fit the transformer (no-op for this transformer).
        
        Parameters
        ----------
        X : array-like or DataFrame
            Input data.
        y : array-like, optional
            Target values (ignored).
        
        Returns
        -------
        self
            Returns self.
        """
        return self
    
    def transform(self, X: Any) -> Any:
        """Transform categorical data by standardizing values.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Input data to transform.
        
        Returns
        -------
        array-like or DataFrame
            Transformed data with standardized categorical values.
        """
        import pandas as pd
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        # Apply transformations to each column
        for col in X_df.columns:
            # Convert to string
            X_df[col] = X_df[col].astype(str)
            
            # Strip whitespace
            if self.strip_whitespace:
                X_df[col] = X_df[col].str.strip()
            
            # Apply case normalization
            if self.case == 'lower':
                X_df[col] = X_df[col].str.lower()
            elif self.case == 'upper':
                X_df[col] = X_df[col].str.upper()
            elif self.case == 'title':
                X_df[col] = X_df[col].str.title()
            
            # Apply custom mapping
            if self.custom_mapping:
                X_df[col] = X_df[col].replace(self.custom_mapping)
        
        return X_df


class encode_categories(BaseEstimator, TransformerMixin):
    """Encode categorical values using ordinal or label encoding.
    
    This transformer converts categorical values to numerical codes.
    Supports both ordinal encoding (with user-specified order) and
    label encoding (alphabetical order).
    
    Parameters
    ----------
    encoding_type : str, optional
        Type of encoding to use. Options: 'ordinal', 'label'.
        Defaults to 'ordinal'.
    categories : dict[str, list[str]] | None, optional
        Dictionary mapping column names to ordered lists of categories.
        Only used when encoding_type='ordinal'. Defaults to None.
    
    Attributes
    ----------
    encoding_type : str
        The type of encoding.
    categories : dict[str, list[str]] | None
        The category orderings (for ordinal encoding).
    mappings_ : dict[str, dict[str, int]]
        Learned mappings from categories to integers (fitted).
    
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'size': ['small', 'medium', 'large', 'small']})
    >>> # Ordinal encoding with specified order
    >>> transformer = encode_categories(
    ...     encoding_type='ordinal',
    ...     categories={'size': ['small', 'medium', 'large']}
    ... )
    >>> transformed = transformer.fit_transform(data[['size']])
    >>> list(transformed['size'])
    [0, 1, 2, 0]
    
    >>> # Label encoding (alphabetical)
    >>> data = pd.DataFrame({'color': ['red', 'blue', 'green', 'red']})
    >>> transformer = encode_categories(encoding_type='label')
    >>> transformed = transformer.fit_transform(data[['color']])
    >>> list(transformed['color'])
    [2, 0, 1, 2]
    """
    
    def __init__(self, encoding_type: str = 'ordinal', 
                 categories: dict[str, list[str]] | None = None):
        self.encoding_type = encoding_type
        self.categories = categories or {}
        self.mappings_ = {}
    
    def fit(self, X: Any, y: Any = None) -> 'encode_categories':
        """Fit the encoder by learning category mappings.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Input data.
        y : array-like, optional
            Target values (ignored).
        
        Returns
        -------
        self
            Returns self with fitted mappings.
        """
        import pandas as pd
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X
        
        # Build mappings for each column
        for col in X_df.columns:
            if self.encoding_type == 'ordinal' and col in self.categories:
                # Use user-specified order
                cats = self.categories[col]
                self.mappings_[col] = {cat: i for i, cat in enumerate(cats)}
            else:
                # Use alphabetical order (label encoding)
                unique_cats = sorted(X_df[col].dropna().unique())
                self.mappings_[col] = {cat: i for i, cat in enumerate(unique_cats)}
        
        return self
    
    def transform(self, X: Any) -> Any:
        """Transform categorical data to numerical codes.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Input data to transform.
        
        Returns
        -------
        array-like or DataFrame
            Transformed data with numerical codes.
        """
        import pandas as pd
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        # Apply mappings
        for col in X_df.columns:
            if col in self.mappings_:
                X_df[col] = X_df[col].map(self.mappings_[col])
                # Fill unmapped values with -1
                X_df[col] = X_df[col].fillna(-1).astype(int)
        
        return X_df
