"""MicroMetricLogger implementation for sparse attention hub."""

import inspect
import json
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class LogEvent:
    """Log event data structure."""

    timestamp: datetime
    metric: str  # Metric identifier string
    value: Union[None, Any]
    metadata: Dict[str, Any]  # Additional context (layer, head, etc.)
    location: str  # Auto-inferred: "module.function" or "class.method"


class MicroMetricLogger:
    """Singleton logger for micro metrics with queue-based architecture."""

    _instance: Optional["MicroMetricLogger"] = None
    _initialized: bool = False

    # Class-level storage for registered metrics (works without initialization)
    _registered_metrics: Dict[str, type] = {}  # identifier -> dtype mapping

    def __new__(cls, *args: Any, **kwargs: Any) -> "MicroMetricLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        log_path: Optional[str] = None,
        flush_every: int = 1000,  # Flush every N events
        flush_interval: float = 60.0,  # Flush every N seconds
        enabled_metrics: Union[List[str], str] = None,
        max_records: Optional[int] = None,  # Maximum number of events to log
        sampling_factor: float = 1.0,  # Probability of logging each event (0.0-1.0)
    ):  # List of string identifiers to enable, or "all"
        if not self._initialized:
            self.log_path = log_path
            self.flush_every = flush_every
            self.flush_interval = flush_interval
            self.max_records = max_records
            self.sampling_factor = max(
                0.0, min(1.0, sampling_factor)
            )  # Clamp to [0.0, 1.0]

            # Internal state
            self.log_queue: deque = deque(maxlen=10000)  # Circular buffer
            self.enabled_metrics: set = set()
            self.last_flush_time = time.time()
            self._total_records_logged: int = 0  # Track total events logged

            # Enable metrics if log_path is provided
            if self.log_path is not None:
                self._ensure_log_directory()
                self.enable_metrics(enabled_metrics)

            MicroMetricLogger._initialized = True
        else:
            if self.log_path and log_path and self.log_path != log_path:
                print(
                    f"Warning: MicroMetricLogger already initialized with log_path: {self.log_path}"
                )

    # main registration function

    @classmethod
    def register_metric(cls, identifier: str, dtype: type) -> None:
        """Register a metric with its string identifier and expected data type.

        This works at class level and doesn't require initialization.
        """
        if identifier in cls._registered_metrics:
            print(f"Warning: Metric '{identifier}' is being re-registered")
        cls._registered_metrics[identifier] = dtype

    @classmethod
    def get_registered_metrics(cls) -> Dict[str, type]:
        """Get all registered metrics at class level."""
        return cls._registered_metrics.copy()

    # helper methods

    def _ensure_log_directory(self) -> None:
        """Ensure the log directory exists."""
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)

    def _get_calling_location(self) -> str:
        """Get the calling location using inspect module."""
        try:
            # Get the calling frame (skip this method and the log method)
            caller_frame = inspect.currentframe().f_back.f_back
            if caller_frame is None:
                return "unknown"

            # Get module name
            module = inspect.getmodule(caller_frame)
            module_name = module.__name__ if module else "unknown"

            # Get function/class name
            function_name = caller_frame.f_code.co_name

            # Try to get class name if it's a method
            class_name = None
            if "self" in caller_frame.f_locals:
                class_name = caller_frame.f_locals["self"].__class__.__name__

            if class_name:
                return f"{module_name}.{class_name}.{function_name}"
            else:
                return f"{module_name}.{function_name}"
        except Exception:
            return "unknown"

    def __del__(self) -> None:
        """Cleanup when logger is destroyed."""
        self.flush()  # Final flush

    # api

    def enable_metrics(self, metrics: Union[List[str], str] = None) -> None:
        """Enable logging for specific metrics.

        Args:
            metrics: List of metric identifiers to enable, or "all" for all registered metrics.
                    If None, enables no metrics (empty list).
        """
        if metrics == "all":
            self.enabled_metrics = set(self._registered_metrics.keys())
        elif isinstance(metrics, (list, set)):
            # Only enable metrics that are registered
            valid_metrics = set(metrics) & set(self._registered_metrics.keys())
            invalid_metrics = set(metrics) - set(self._registered_metrics.keys())
            if invalid_metrics:
                print(
                    f"Warning: Attempting to enable unregistered metrics: {invalid_metrics}"
                )
            self.enabled_metrics = valid_metrics
        else:
            # Default to empty set
            self.enabled_metrics = set()

    def log(self, identifier: str, value: Any, metadata: Dict[str, Any] = None) -> None:
        """Log a metric value with optional metadata. Location is auto-inferred.

        This only works if log_path is defined.
        """
        # Check if logging is configured
        if self.log_path is None:
            print(
                f"Warning: Cannot log metric '{identifier}' - log_path not defined. Use configure_logging() first."
            )
            return

        # Check if metric is enabled
        if identifier not in self.enabled_metrics:
            print(
                f"Warning: Attempting to log metric '{identifier}' which is not enabled"
            )
            return

        # Check if max_records limit has been reached
        if (
            self.max_records is not None
            and self._total_records_logged >= self.max_records
        ):
            return  # Silently ignore - we've hit the limit

        # Apply sampling factor
        if self.sampling_factor < 1.0 and random.random() > self.sampling_factor:
            return  # Event not selected for logging due to sampling

        # Create log event
        event = LogEvent(
            timestamp=datetime.now(),
            metric=identifier,
            value=value,
            metadata=metadata or {},
            location=self._get_calling_location(),
        )

        # Add to queue and increment counter
        self.log_queue.append(event)
        self._total_records_logged += 1

        # Check if we should flush
        if len(self.log_queue) >= self.flush_every:
            self.flush()

    def configure_logging(
        self,
        log_path: str,
        enabled_metrics: Union[List[str], str] = None,
        max_records: Optional[int] = None,
        sampling_factor: float = 1.0,
    ) -> None:
        """Configure logging with a log path and optionally enable metrics.

        This must be called before logging can work. Resets the total records counter.
        """
        self.log_path = log_path
        self._ensure_log_directory()
        self.enable_metrics(enabled_metrics)

        # Update limits if provided
        self.max_records = max_records
        self.sampling_factor = max(0.0, min(1.0, sampling_factor))

        # Reset the total records counter when reconfiguring
        self._total_records_logged = 0

    def flush(self) -> None:
        """Force flush the current queue to disk."""
        if not self.log_queue or self.log_path is None:
            return

        # Get current timestamp for filename
        filename = "micro_metrics.jsonl"
        filepath = os.path.join(self.log_path, filename)

        # Write events to file
        with open(filepath, "a", encoding="utf-8") as f:
            while self.log_queue:
                event = self.log_queue.popleft()
                # Convert dataclass to dict and serialize
                event_dict = asdict(event)
                # Convert datetime to ISO format string
                event_dict["timestamp"] = event_dict["timestamp"].isoformat()
                f.write(json.dumps(event_dict) + "\n")

        self.last_flush_time = time.time()

    def is_metric_enabled(self, identifier: str) -> bool:
        """Check if a specific metric is requested for logging."""
        return identifier in self.enabled_metrics

    def get_enabled_metrics(self) -> set:
        """Get currently enabled metrics."""
        return self.enabled_metrics.copy()

    def is_logging_configured(self) -> bool:
        """Check if logging is configured (log_path is set)."""
        return self.log_path is not None

    def get_total_records_logged(self) -> int:
        """Get the total number of records logged since initialization or last configure_logging call."""
        return getattr(self, "_total_records_logged", 0)

    def is_max_records_reached(self) -> bool:
        """Check if the maximum number of records has been reached."""
        if self.max_records is None:
            return False
        return self.get_total_records_logged() >= self.max_records

    def get_records_remaining(self) -> Optional[int]:
        """Get the number of records remaining before hitting max_records limit.

        Returns:
            Number of records remaining, or None if no limit is set.
        """
        if self.max_records is None:
            return None
        return max(0, self.max_records - self.get_total_records_logged())

    def get_sampling_factor(self) -> float:
        """Get the current sampling factor."""
        return getattr(self, "sampling_factor", 1.0)

    def get_max_records(self) -> Optional[int]:
        """Get the current max_records limit."""
        return self.max_records
