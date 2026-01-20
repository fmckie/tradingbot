"""JSON utilities for safe serialization of data containing NaN/Inf values.

PostgreSQL JSONB rejects NaN and Inf tokens (not valid per JSON spec).
This module provides a drop-in replacement for json.dumps() that converts
these values to null for safe database storage.
"""
import json
import math
from typing import Any

# Try importing numpy for handling numpy types
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NaN, Inf, and numpy types.

    Converts:
    - float('nan') / np.nan -> null
    - float('inf') / float('-inf') -> null
    - np.float64, np.int64, etc. -> Python native types
    """

    def default(self, obj: Any) -> Any:
        """Handle non-standard types."""
        if HAS_NUMPY:
            # Handle numpy scalar types
            if isinstance(obj, np.floating):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return self._sanitize_array(obj.tolist())
            if isinstance(obj, np.bool_):
                return bool(obj)

        return super().default(obj)

    def _sanitize_array(self, arr: list) -> list:
        """Recursively sanitize array values."""
        result = []
        for item in arr:
            if isinstance(item, list):
                result.append(self._sanitize_array(item))
            elif isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
                result.append(None)
            else:
                result.append(item)
        return result

    def encode(self, obj: Any) -> str:
        """Override encode to handle NaN/Inf in nested structures."""
        return super().encode(self._sanitize(obj))

    def _sanitize(self, obj: Any) -> Any:
        """Recursively sanitize an object, converting NaN/Inf to None."""
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj

        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._sanitize(item) for item in obj]

        if isinstance(obj, tuple):
            return tuple(self._sanitize(item) for item in obj)

        # Handle numpy types if available
        if HAS_NUMPY:
            if isinstance(obj, np.floating):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return self._sanitize(obj.tolist())

        return obj


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Drop-in replacement for json.dumps() that handles NaN/Inf values.

    Converts NaN and Inf values to null (JSON-valid) before serialization.
    Also handles numpy types by converting them to Python native types.

    Args:
        obj: Object to serialize
        **kwargs: Additional arguments passed to json.dumps()

    Returns:
        JSON string with NaN/Inf converted to null

    Example:
        >>> data = {"value": float('nan'), "normal": 42}
        >>> safe_json_dumps(data)
        '{"value": null, "normal": 42}'
    """
    # Remove 'cls' if provided to avoid conflict
    kwargs.pop('cls', None)
    return json.dumps(obj, cls=SafeJSONEncoder, **kwargs)
