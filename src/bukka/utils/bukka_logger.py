import logging
import math


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class BukkaLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)

    def debug(self, msg, format_level='p'):
        msg = self.format_message(msg, format_level)
        self.logger.debug(msg)

    def info(self, msg, format_level='p'):
        msg = self.format_message(msg, format_level)
        self.logger.info(msg)

    def warn(self, msg, format_level='p'):
        msg = self.format_message(msg, format_level)
        self.logger.warn(msg)

    def error(self, msg, format_level='p'):
        msg = self.format_message(msg, format_level)
        self.logger.error(msg)

    def critical(self, msg, format_level='p'):
        msg = self.format_message(msg, format_level)
        self.logger.critical(msg)

    def format_message(self, msg: str, format_level: str):
        format_level=format_level.lower()
        if format_level == 'p':
            return msg
        
        elif format_level == 'h4':
            return f'{msg}\n{"="*50}'
        
        elif format_level == 'h3':
            return f'\n\n{msg}\n{"="*50}'
        
        elif format_level == 'h2':

            max_width = 76
            iters = math.ceil(len(msg) / max_width)

            formatted_message = f'\n{"+" * (max_width + 4)}\n'
            for i in range(iters):
                formatted_message += f'+ {msg[i:i + max_width].ljust(max_width)} +\n'

            formatted_message += f'{"+" * (max_width + 4)}\n'

            return formatted_message
        
        elif format_level == 'h1':
            formatted_message = self.format_message(msg, format_level='h2').upper()
            return formatted_message