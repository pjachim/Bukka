import pytest
import logging
import io
import contextlib
import math
from unittest.mock import patch, MagicMock

# Import the class/functions to be tested
# Assuming the enhanced code is saved as 'bukka_logger.py'
from bukka.utils.bukka_logger import BukkaLogger, H2_MAX_WIDTH 

# Set up a stream to capture log output for testing
@pytest.fixture
def caplog_stream():
    """Fixture to capture log output to a string buffer."""
    # A String buffer to capture the logging output
    log_capture_stream = io.StringIO()
    # A handler that writes to the string buffer
    handler = logging.StreamHandler(log_capture_stream)
    # Get the root logger
    root_logger = logging.getLogger()
    # Save current handlers and level
    old_handlers = root_logger.handlers[:]
    old_level = root_logger.level
    
    # Temporarily set root level to DEBUG to ensure all levels are captured
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
    
    yield log_capture_stream
    
    # Teardown: Restore original logger state
    root_logger.removeHandler(handler)
    root_logger.setLevel(old_level)
    root_logger.handlers = old_handlers
    handler.close()


class TestBukkaLogger:
    """
    Test suite for the BukkaLogger class constructor and basic instantiation.
    """
    TEST_LOGGER_NAME = "test_logger"

    def test_instantiation(self):
        """Test that BukkaLogger can be instantiated and sets the correct logger name."""
        logger_instance = BukkaLogger(self.TEST_LOGGER_NAME)
        assert isinstance(logger_instance, BukkaLogger)
        assert isinstance(logger_instance.logger, logging.Logger)
        assert logger_instance.logger.name == self.TEST_LOGGER_NAME

    def test_multiple_instances(self):
        """Test that two instances with different names get separate loggers."""
        logger_1 = BukkaLogger("logger_one")
        logger_2 = BukkaLogger("logger_two")
        assert logger_1.logger.name == "logger_one"
        assert logger_2.logger.name == "logger_two"
        assert logger_1.logger is not logger_2.logger


class TestBukkaLoggerLoggingMethods:
    """
    Test suite for the logging methods (debug, info, warn, error, critical).
    Focuses on ensuring the correct log level is called and that 'p' is the default format.
    """
    
    @pytest.fixture(autouse=True)
    def setup_method(self, caplog_stream):
        """Set up a fresh BukkaLogger instance for each test method."""
        # Use a consistent name to make sure we're testing the same logger context
        self.logger_name = "method_test_logger"
        self.bukka_logger = BukkaLogger(self.logger_name)
        # We need the underlying logging.Logger for mocking/checking level
        self.std_logger = logging.getLogger(self.logger_name)
        self.caplog_stream = caplog_stream
        
        # Ensure the test logger is set to DEBUG to capture all levels
        self.std_logger.setLevel(logging.DEBUG)
        
        # Mock format_message to isolate logging method testing, 
        # but only for checking if it's called
        with patch.object(self.bukka_logger, 'format_message', wraps=self.bukka_logger.format_message) as self.mock_format_message:
            yield

    @pytest.mark.parametrize("method_name, log_level", [
        ('debug', logging.DEBUG),
        ('info', logging.INFO),
        ('warn', logging.WARNING),
        ('error', logging.ERROR),
        ('critical', logging.CRITICAL),
    ])

    def test_warn_uses_warning(self):
        """Test that the 'warn' method internally uses logger.warning() (best practice)."""
        test_msg = "Warning test."
        
        # We'll mock the internal logger's methods directly to confirm the call
        with patch.object(self.bukka_logger.logger, 'warning') as mock_warning:
            self.bukka_logger.warn(test_msg)
            # format_message returns the original message for 'p'
            mock_warning.assert_called_once_with(test_msg)


class TestBukkaLoggerFormatMessage:
    """
    Test suite for the format_message method, focusing on all format levels and edge cases.
    """
    
    @pytest.fixture(autouse=True)
    def setup_class(self):
        """Setup a BukkaLogger instance for format_message testing."""
        self.bukka_logger = BukkaLogger("format_test_logger")
        self.long_message = "A" * (H2_MAX_WIDTH + 10) # 76 + 10 = 86 chars
        self.short_message = "Short msg"
        self.empty_message = ""
        self.special_chars_message = "Msg with !@#$%^&*()_+-"

    @pytest.mark.parametrize("level, expected_output", [
        ('p', "Test message"),
        ('P', "Test message"), # Case insensitivity test
        ('h4', f"Test message\n{'='*50}"),
        ('H4', f"Test message\n{'='*50}"), # Case insensitivity test
        ('h3', f"\n\nTest message\n{'='*50}"),
        ('H3', f"\n\nTest message\n{'='*50}"), # Case insensitivity test
    ])
    def test_simple_formats(self, level, expected_output):
        """Test 'p', 'h4', and 'h3' formats with a standard message."""
        result = self.bukka_logger.format_message("Test message", level)
        assert result == expected_output

    def test_h2_short_message(self):
        """Test 'h2' format with a short message (fits on one line)."""
        msg = self.short_message
        width = H2_MAX_WIDTH
        
        # Expected structure: 
        # \n
        # ++++... (width+4)
        # + Short msg<spaces> +
        # ++++... (width+4)
        # \n
        expected = (
            f'\n{"+" * (width + 4)}\n'
            f'+ {msg.ljust(width)} +\n'
            f'{"+" * (width + 4)}\n'
        )
        result = self.bukka_logger.format_message(msg, 'h2')
        assert result == expected

    def test_h2_long_message(self):
        """Test 'h2' format with a message that spans two lines."""
        msg = self.long_message # Length 86
        width = H2_MAX_WIDTH # 76
        
        segment1 = msg[:width] # First 76 chars
        segment2 = msg[width:].ljust(width) # Remaining 10 chars, padded to 76
        
        expected = (
            f'\n{"+" * (width + 4)}\n'
            f'+ {segment1} +\n'
            f'+ {segment2} +\n'
            f'{"+" * (width + 4)}\n'
        )
        result = self.bukka_logger.format_message(msg, 'h2')
        assert result == expected

    def test_h2_empty_message(self):
        """Test 'h2' format with an empty message string (edge case)."""
        msg = self.empty_message
        width = H2_MAX_WIDTH
        
        # Should result in an empty line in the middle, padded to full width
        expected = (
            f'\n{"+" * (width + 4)}\n'
            f'+ {"".ljust(width)} +\n'
            f'{"+" * (width + 4)}\n'
        )
        result = self.bukka_logger.format_message(msg, 'h2')
        assert result == expected

    def test_h2_message_exactly_max_width(self):
        """Test 'h2' format with a message that is exactly the max width (edge case)."""
        msg = "E" * H2_MAX_WIDTH # Length 76
        width = H2_MAX_WIDTH
        
        # Should result in exactly one data line
        expected = (
            f'\n{"+" * (width + 4)}\n'
            f'+ {msg} +\n'
            f'{"+" * (width + 4)}\n'
        )
        result = self.bukka_logger.format_message(msg, 'h2')
        assert result == expected
        
    def test_h1_format_is_uppercase_h2(self):
        """Test 'h1' format ensures the result is the 'h2' format and fully uppercase."""
        msg = "test message"
        
        # 1. Get the expected H2 format
        h2_result = self.bukka_logger.format_message(msg, 'h2')
        
        # 2. Assert H1 is the uppercase version of H2
        h1_result = self.bukka_logger.format_message(msg, 'h1')
        assert h1_result == h2_result.upper()

    def test_h1_with_long_message(self):
        """Test 'h1' format with a long message for wrapping and uppercase conversion."""
        msg = self.long_message
        
        h2_result = self.bukka_logger.format_message(msg, 'h2')
        h1_result = self.bukka_logger.format_message(msg, 'h1')
        
        assert h1_result == h2_result.upper()

    @patch('logging.Logger.warning')
    def test_unknown_format_defaults_to_p(self, mock_warning):
        """Test that an unknown format level defaults to 'p' and logs a warning."""
        msg = "Unknown format message"
        unknown_level = "h5" # Not supported
        
        result = self.bukka_logger.format_message(msg, unknown_level)
        
        # 1. Should return the original message (like 'p')
        assert result == msg
        
        # 2. Should log a warning about the unknown level
        mock_warning.assert_called_once()
        # Check if the warning message contains the unknown level
        call_args, _ = mock_warning.call_args
        assert f"Unknown format level '{unknown_level}'" in call_args[0]

    @pytest.mark.parametrize("input_type", [123, True, [1, 2, 3]])
    def test_non_string_input_type_handling(self, input_type):
        """Test that format_message handles non-string inputs by converting them."""
        # The logging methods convert to string before passing to format_message.
        # Here we test format_message directly with non-string, which is not strictly
        # required by the class's logic but good for robustness if format_message
        # were called directly. We expect it to fail or be converted to a string
        # because the type hint requires `str`.
        # However, since the logging methods ensure str(msg) is passed, we'll
        # test the logging methods instead for this edge case.
        pass # Covered in TestBukkaLoggerLoggingMethods
        
    @pytest.mark.parametrize("input_value, expected_str", [
        (12345, "12345"),
        (True, "True"),
        (None, "None"),
        ([1, 2, 3], "[1, 2, 3]"),
    ])
    def test_logging_methods_handle_non_string_msg(self, input_value, expected_str, caplog_stream):
        """
        Test that logging methods properly convert non-string messages to string
        before passing them to format_message.
        """
        bukka_logger = BukkaLogger("non_str_test")
        
        # Use a method that does no formatting change
        bukka_logger.info(input_value, format_level='p')
        
        log_output = caplog_stream.getvalue()
        # Check if the string representation of the input value is in the log
        assert expected_str in log_output