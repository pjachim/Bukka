"""Unit tests for TemplateBaseClass."""
import pytest
import tempfile
from pathlib import Path

from bukka.coding.utils.template_handler import TemplateBaseClass


class TestTemplateBaseClassInitialization:
    """Test suite for TemplateBaseClass initialization."""

    def test_initialization_with_all_args(self):
        """Test TemplateBaseClass initialization with all arguments.
        
        Examples
        --------
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> template = "x = {value}"
        >>> handler = TemplateBaseClass(
        ...     template=template,
        ...     output_path="output.py",
        ...     kwargs={'value': 42},
        ...     expected_args=['value']
        ... )
        >>> assert handler.template == template
        >>> assert handler.output_path == "output.py"
        """
        template = "x = {value}"
        kwargs = {'value': 42}
        expected_args = ['value']
        
        handler = TemplateBaseClass(
            template=template,
            output_path="output.py",
            kwargs=kwargs,
            expected_args=expected_args
        )
        
        assert handler.template == template
        assert handler.output_path == "output.py"
        assert handler.kwargs == kwargs
        assert handler.expected_args == expected_args

    def test_initialization_without_expected_args(self):
        """Test initialization with expected_args=None defaults to empty list.
        
        Examples
        --------
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> handler = TemplateBaseClass(
        ...     template="test",
        ...     output_path="out.py",
        ...     kwargs={}
        ... )
        >>> assert handler.expected_args == []
        """
        handler = TemplateBaseClass(
            template="test",
            output_path="out.py",
            kwargs={}
        )
        
        assert handler.expected_args == []

    def test_initialization_with_path_object(self):
        """Test initialization accepts Path objects for output_path.
        
        Examples
        --------
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> from pathlib import Path
        >>> output_path = Path("output") / "file.py"
        >>> handler = TemplateBaseClass(
        ...     template="test",
        ...     output_path=output_path,
        ...     kwargs={}
        ... )
        >>> assert handler.output_path == output_path
        """
        output_path = Path("output") / "file.py"
        handler = TemplateBaseClass(
            template="test",
            output_path=output_path,
            kwargs={}
        )
        
        assert handler.output_path == output_path


class TestTemplateBaseClassFillTemplate:
    """Test suite for TemplateBaseClass._fill_template method."""

    def test_fill_template_simple_substitution(self):
        """Test _fill_template performs simple placeholder substitution.
        
        Examples
        --------
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> template = "Hello {name}!"
        >>> handler = TemplateBaseClass(
        ...     template=template,
        ...     output_path="out.txt",
        ...     kwargs={'name': 'World'}
        ... )
        >>> result = handler._fill_template()
        >>> assert result == "Hello World!"
        """
        template = "Hello {name}!"
        handler = TemplateBaseClass(
            template=template,
            output_path="out.txt",
            kwargs={'name': 'World'}
        )
        
        result = handler._fill_template()
        assert result == "Hello World!"

    def test_fill_template_multiple_placeholders(self):
        """Test _fill_template with multiple placeholders.
        
        Examples
        --------
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> template = "{greeting} {name}, you are {age} years old."
        >>> handler = TemplateBaseClass(
        ...     template=template,
        ...     output_path="out.txt",
        ...     kwargs={'greeting': 'Hello', 'name': 'Alice', 'age': 30}
        ... )
        >>> result = handler._fill_template()
        >>> assert result == "Hello Alice, you are 30 years old."
        """
        template = "{greeting} {name}, you are {age} years old."
        handler = TemplateBaseClass(
            template=template,
            output_path="out.txt",
            kwargs={'greeting': 'Hello', 'name': 'Alice', 'age': 30}
        )
        
        result = handler._fill_template()
        assert result == "Hello Alice, you are 30 years old."

    def test_fill_template_strips_whitespace(self):
        """Test _fill_template strips leading/trailing whitespace.
        
        Examples
        --------
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> template = "\\n\\n  {value}  \\n\\n"
        >>> handler = TemplateBaseClass(
        ...     template=template,
        ...     output_path="out.txt",
        ...     kwargs={'value': 'test'}
        ... )
        >>> result = handler._fill_template()
        >>> assert result == "test"
        """
        template = "\n\n  {value}  \n\n"
        handler = TemplateBaseClass(
            template=template,
            output_path="out.txt",
            kwargs={'value': 'test'}
        )
        
        result = handler._fill_template()
        assert result == "test"

    def test_fill_template_with_code_structure(self):
        """Test _fill_template with code-like template structure.
        
        Examples
        --------
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> template = '''
        ... class {class_name}:
        ...     def __init__(self):
        ...         self.value = {default_value}
        ... '''
        >>> handler = TemplateBaseClass(
        ...     template=template,
        ...     output_path="out.py",
        ...     kwargs={'class_name': 'MyClass', 'default_value': 42}
        ... )
        >>> result = handler._fill_template()
        >>> assert 'class MyClass:' in result
        >>> assert 'self.value = 42' in result
        """
        template = '''
class {class_name}:
    def __init__(self):
        self.value = {default_value}
'''
        handler = TemplateBaseClass(
            template=template,
            output_path="out.py",
            kwargs={'class_name': 'MyClass', 'default_value': 42}
        )
        
        result = handler._fill_template()
        assert 'class MyClass:' in result
        assert 'self.value = 42' in result


class TestTemplateBaseClassWriteCode:
    """Test suite for TemplateBaseClass.write_code method."""

    def test_write_code_creates_file(self):
        """Test write_code creates a file at the specified path.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     output_path = Path(tmp_dir) / "output.py"
        ...     handler = TemplateBaseClass(
        ...         template="x = {value}",
        ...         output_path=output_path,
        ...         kwargs={'value': 42}
        ...     )
        ...     handler.write_code()
        ...     assert output_path.exists()
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output.py"
            handler = TemplateBaseClass(
                template="x = {value}",
                output_path=output_path,
                kwargs={'value': 42}
            )
            
            handler.write_code()
            assert output_path.exists()

    def test_write_code_contains_filled_template(self):
        """Test write_code writes the filled template content.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     output_path = Path(tmp_dir) / "output.py"
        ...     handler = TemplateBaseClass(
        ...         template="result = {value}",
        ...         output_path=output_path,
        ...         kwargs={'value': 123}
        ...     )
        ...     handler.write_code()
        ...     content = output_path.read_text()
        ...     assert content == "result = 123"
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output.py"
            handler = TemplateBaseClass(
                template="result = {value}",
                output_path=output_path,
                kwargs={'value': 123}
            )
            
            handler.write_code()
            content = output_path.read_text()
            assert content == "result = 123"

    def test_write_code_overwrites_existing_file(self):
        """Test write_code overwrites existing file.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     output_path = Path(tmp_dir) / "output.py"
        ...     output_path.write_text("old content")
        ...     handler = TemplateBaseClass(
        ...         template="new = {value}",
        ...         output_path=output_path,
        ...         kwargs={'value': 456}
        ...     )
        ...     handler.write_code()
        ...     content = output_path.read_text()
        ...     assert content == "new = 456"
        ...     assert "old content" not in content
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output.py"
            output_path.write_text("old content")
            
            handler = TemplateBaseClass(
                template="new = {value}",
                output_path=output_path,
                kwargs={'value': 456}
            )
            
            handler.write_code()
            content = output_path.read_text()
            assert content == "new = 456"
            assert "old content" not in content


class TestTemplateBaseClassMakePythonStringVariableSafe:
    """Test suite for make_python_string_variable_safe method."""

    def test_safe_name_replaces_spaces(self):
        """Test that spaces are replaced with underscores.
        
        Examples
        --------
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> handler = TemplateBaseClass("", "", {})
        >>> result = handler.make_python_string_variable_safe("my variable")
        >>> assert result == "my_variable"
        """
        handler = TemplateBaseClass("", "", {})
        result = handler.make_python_string_variable_safe("my variable")
        assert result == "my_variable"

    def test_safe_name_replaces_special_characters(self):
        """Test that special characters are replaced with underscores.
        
        Examples
        --------
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> handler = TemplateBaseClass("", "", {})
        >>> result = handler.make_python_string_variable_safe("var-name!")
        >>> assert result == "var_name_"
        """
        handler = TemplateBaseClass("", "", {})
        result = handler.make_python_string_variable_safe("var-name!")
        assert result == "var_name_"

    def test_safe_name_preserves_alphanumeric_and_underscores(self):
        """Test that alphanumeric characters and underscores are preserved.
        
        Examples
        --------
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> handler = TemplateBaseClass("", "", {})
        >>> result = handler.make_python_string_variable_safe("valid_var_123")
        >>> assert result == "valid_var_123"
        """
        handler = TemplateBaseClass("", "", {})
        result = handler.make_python_string_variable_safe("valid_var_123")
        assert result == "valid_var_123"

    def test_safe_name_lowercase_option(self):
        """Test lowercase option converts to lowercase.
        
        Examples
        --------
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> handler = TemplateBaseClass("", "", {})
        >>> result = handler.make_python_string_variable_safe("MyVariable", lowercase=True)
        >>> assert result == "myvariable"
        """
        handler = TemplateBaseClass("", "", {})
        result = handler.make_python_string_variable_safe("MyVariable", lowercase=True)
        assert result == "myvariable"

    def test_safe_name_complex_string(self):
        """Test safe name generation with complex input.
        
        Examples
        --------
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> handler = TemplateBaseClass("", "", {})
        >>> result = handler.make_python_string_variable_safe("Feature #1 (test-value)")
        >>> assert result == "Feature__1__test_value_"
        """
        handler = TemplateBaseClass("", "", {})
        result = handler.make_python_string_variable_safe("Feature #1 (test-value)")
        assert result == "Feature__1__test_value_"

    def test_safe_name_with_multiple_consecutive_special_chars(self):
        """Test handling of multiple consecutive special characters.
        
        Examples
        --------
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> handler = TemplateBaseClass("", "", {})
        >>> result = handler.make_python_string_variable_safe("var---name")
        >>> assert result == "var___name"
        """
        handler = TemplateBaseClass("", "", {})
        result = handler.make_python_string_variable_safe("var---name")
        assert result == "var___name"
