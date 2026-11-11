from datetime import datetime
from pathlib import Path


class Logger:
    _instance = None
    _verbose = True
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def set_verbose(self, verbose: bool):
        self._verbose = verbose
    
    def _write(self, level: str, message: str):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted = f"[{timestamp}] [{level}] {message}"
        if self._verbose:
            print(formatted)
    
    def info(self, message: str):
        self._write('INFO', message)
    
    def warning(self, message: str):
        self._write('WARNING', message)
    
    def error(self, message: str):
        self._write('ERROR', message)
    
    def success(self, message: str):
        self._write('SUCCESS', message)
    
    def debug(self, message: str):
        self._write('DEBUG', message)


def get_logger():
    return Logger()