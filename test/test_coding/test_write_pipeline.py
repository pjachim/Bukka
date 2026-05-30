"""Unit tests for PipelineWriter after refactor to use TemplateBaseClass.

This test suite validates the refactored PipelineWriter which now:
- Inherits from TemplateBaseClass for template-based code generation
- Uses write_code() method instead of write()
- Separates concerns into _fetch_imports(), _parse_pipeline_steps(), _build_*() methods
- Distinguishes between transformers (column-specific) and manipulators (multi-column)
- Chains multiple transformers on the same columns into a Pipeline
- Includes hardcoded template imports in writer.imports set
"""
import importlib
from types import SimpleNamespace
from pathlib import Path
import tempfile

import pytest

from bukka.coding.write_pipeline import PipelineWriter

class TestPipelineWriter:
    def refacor_notice(self):
        assert False, 'This module is being refactored from the ground up, so tests to be added.'