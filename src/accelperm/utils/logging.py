"""Logging framework with colored output and progress reporting for AccelPerm."""

import logging
import platform
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Optional


class LogLevel(Enum):
    """Log levels for AccelPerm logger."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


class PerformanceTimer:
    """Timer for tracking operation performance."""

    def __init__(self, operation_name: str) -> None:
        self.operation_name = operation_name
        self.start_time: float = 0
        self.end_time: float = 0
        self.elapsed: float = 0

    def __enter__(self) -> "PerformanceTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time


class ProgressReporter:
    """Progress reporter for long-running operations."""

    def __init__(
        self, total: int, description: str, logger: Optional["AccelPermLogger"] = None
    ) -> None:
        self.total = total
        self.description = description
        self.current = 0
        self.finished = False
        self.logger = logger

    @property
    def percentage(self) -> float:
        """Get current completion percentage."""
        if self.total == 0:
            return 100.0
        return (self.current / self.total) * 100.0

    def update(self, increment: int = 1) -> None:
        """Update progress by increment."""
        self.current += increment
        if self.current > self.total:
            self.current = self.total

        if self.logger:
            percentage = self.percentage
            message = (
                f"{self.description}: {percentage:.1f}% ({self.current}/{self.total})"
            )

            if self.logger.colored:
                # Add ANSI color codes for colored output
                message = f"\033[92m{message}\033[0m"  # Green color

            self.logger.info(message)

    def finish(self) -> None:
        """Mark progress as finished."""
        self.current = self.total
        self.finished = True

        if self.logger:
            message = f"{self.description}: Complete!"
            if self.logger.colored:
                message = f"\033[92m{message}\033[0m"  # Green color
            self.logger.info(message)

    def __enter__(self) -> "ProgressReporter":
        return self

    def __exit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None:
        if not self.finished:
            self.finish()


class AccelPermLogger:
    """Enhanced logger with colored output and progress reporting."""

    def __init__(
        self, name: str, level: LogLevel = LogLevel.INFO, colored: bool = True
    ) -> None:
        self.name = name
        self.level = level
        self.colored = colored

        # Setup underlying Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.value)

        # Add console handler if not already present
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    def info(self, message: str) -> None:
        """Log info message."""
        self._logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self._logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self._logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self._logger.error(message)

    def log_system_info(self) -> None:
        """Log system information."""
        import psutil

        python_version = sys.version.split()[0]
        platform_info = platform.platform()
        memory_gb = psutil.virtual_memory().total / (1024**3)

        system_info = f"""System Information:
Python version: {python_version}
Platform: {platform_info}
Memory: {memory_gb:.1f} GB"""

        print(system_info)

    def create_progress(self, total: int, description: str) -> ProgressReporter:
        """Create a progress reporter linked to this logger."""
        return ProgressReporter(total=total, description=description, logger=self)

    @contextmanager
    def time_operation(self, operation_name: str) -> Iterator[PerformanceTimer]:
        """Context manager for timing operations."""
        timer = PerformanceTimer(operation_name)
        with timer:
            yield timer

        elapsed_ms = timer.elapsed * 1000
        self.info(f"{operation_name} completed in {elapsed_ms:.2f} ms")


def setup_logger(
    name: str, level: LogLevel = LogLevel.INFO, colored: bool = True
) -> AccelPermLogger:
    """Set up and return an AccelPerm logger."""
    return AccelPermLogger(name=name, level=level, colored=colored)
