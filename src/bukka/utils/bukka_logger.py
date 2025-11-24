import logging
import math
from typing import Optional, Any
from bukka.config import LOG_LEVEL

# Define constants for better maintainability and readability
DEFAULT_FORMAT_LEVEL: str = 'p'
H2_MAX_WIDTH: int = 76
H4_SEPARATOR: str = "=" * 50
H3_SEPARATOR: str = "=" * 50
H2_BORDER_CHAR: str = "+"
H2_BORDER_LENGTH: int = H2_MAX_WIDTH + 4 # 76 + 4 = 80

class BukkaLogger:
    """
    A custom wrapper around Python's standard logging logger
    that adds custom message formatting capabilities based on a
    specified format level.

    The formatting levels mimic typical Markdown or HTML heading structures
    (p, h4, h3, h2, h1) for visual distinction in logs.

    Attributes
    ----------
    logger : logging.Logger
        The underlying standard library logger instance.
    """
    def __init__(self, name: str, log_level: int = LOG_LEVEL) -> None:
        """
        Initializes the BukkaLogger.

        Parameters
        ----------
        name : str
            The name to use for the underlying logging.Logger instance.
            Typically the module name (__name__).
        """
        # Get a specific logger instance with the given name
        self.logger: logging.Logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            
            # Create a simple formatter
            formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)

    def debug(self, msg: Any, format_level: str = DEFAULT_FORMAT_LEVEL) -> None:
        """
        Logs a message with level DEBUG, after formatting.

        Parameters
        ----------
        msg : Any
            The message to be logged. It will be converted to a string
            by the logging system.
        format_level : str, optional
            The formatting level to apply ('p', 'h4', 'h3', 'h2', 'h1').
            Defaults to 'p' (paragraph/plain).
        """
        # Format the message string based on the provided format_level
        formatted_msg: str = self.format_message(str(msg), format_level)
        # Log the formatted message using the standard debug method
        self.logger.debug(formatted_msg)

    def info(self, msg: Any, format_level: str = DEFAULT_FORMAT_LEVEL) -> None:
        """
        Logs a message with level INFO, after formatting.

        Parameters
        ----------
        msg : Any
            The message to be logged.
        format_level : str, optional
            The formatting level to apply. Defaults to 'p'.
        """
        formatted_msg: str = self.format_message(str(msg), format_level)
        self.logger.info(formatted_msg)

    def warn(self, msg: Any, format_level: str = DEFAULT_FORMAT_LEVEL) -> None:
        """
        Logs a message with level WARNING, after formatting.

        Parameters
        ----------
        msg : Any
            The message to be logged.
        format_level : str, optional
            The formatting level to apply. Defaults to 'p'.
        """
        # Note: The logging module recommends using .warning() instead of .warn(),
        # but .warn() is kept for backward compatibility with the original code.
        formatted_msg: str = self.format_message(str(msg), format_level)
        self.logger.warning(formatted_msg) # Using standard .warning() as per best practice

    def error(self, msg: Any, format_level: str = DEFAULT_FORMAT_LEVEL) -> None:
        """
        Logs a message with level ERROR, after formatting.

        Parameters
        ----------
        msg : Any
            The message to be logged.
        format_level : str, optional
            The formatting level to apply. Defaults to 'p'.
        """
        formatted_msg: str = self.format_message(str(msg), format_level)
        self.logger.error(formatted_msg)

    def critical(self, msg: Any, format_level: str = DEFAULT_FORMAT_LEVEL) -> None:
        """
        Logs a message with level CRITICAL, after formatting.

        Parameters
        ----------
        msg : Any
            The message to be logged.
        format_level : str, optional
            The formatting level to apply. Defaults to 'p'.
        """
        formatted_msg: str = self.format_message(str(msg), format_level)
        self.logger.critical(formatted_msg)

    def format_message(self, msg: str, format_level: str) -> str:
        """
        Applies a specific visual format to the log message based on the level.

        Parameters
        ----------
        msg : str
            The original message string to be formatted.
        format_level : str
            The format identifier ('p', 'h4', 'h3', 'h2', 'h1').
            Case-insensitive.

        Returns
        -------
        str
            The formatted message string.
        """
        # Standardize the format level to lowercase for comparison
        level: str = format_level.lower()

        # Plain paragraph format (no change)
        if level == 'p':
            return msg

        # H4: Message followed by a separator line
        elif level == 'h4':
            return f'{msg}\n{H4_SEPARATOR}'

        # H3: Two newlines before, message, then a separator line
        elif level == 'h3':
            return f'\n\n{msg}\n{H3_SEPARATOR}'

        # H2: Boxed format, wrapping message at a maximum width
        elif level == 'h2':
            # Calculate the number of lines required for wrapping
            if not msg:
                iters: int = 1
            else:
                iters: int = math.ceil(len(msg) / H2_MAX_WIDTH)

            # Start with the top border
            formatted_message: str = f'\n{H2_BORDER_CHAR * H2_BORDER_LENGTH}\n'
            
            # Iterate through the necessary lines
            for i in range(iters):
                start_index: int = i * H2_MAX_WIDTH
                end_index: int = start_index + H2_MAX_WIDTH
                # Extract the segment and pad it with spaces to the max width, 
                # then wrap with borders
                segment: str = msg[start_index:end_index]
                formatted_message += f'{H2_BORDER_CHAR} {segment.ljust(H2_MAX_WIDTH)} {H2_BORDER_CHAR}\n'

            # Add the bottom border
            formatted_message += f'{H2_BORDER_CHAR * H2_BORDER_LENGTH}\n'

            return formatted_message

        # H1: H2 format converted to uppercase
        elif level == 'h1':
            # Recursively call format_message with 'h2' level
            formatted_message_h2: str = self.format_message(msg, format_level='h2')
            # Convert the entire box to uppercase
            return formatted_message_h2.upper()
        
        # Default case for unknown format levels: treat as plain 'p'
        else:
            self.logger.warning(f"Unknown format level '{format_level}'. Defaulting to 'p'.")
            return msg