"""Unit tests for JSON utilities handling NaN/Inf values."""

import json
import unittest
from typing import Any

from database.json_utils import SafeJSONEncoder, safe_json_dumps

# Try importing numpy for additional tests
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class TestSafeJSONDumps(unittest.TestCase):
    """Test cases for safe_json_dumps function."""

    def test_nan_to_null(self):
        """NaN values should be converted to null."""
        data = {"value": float("nan")}
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertIsNone(parsed["value"])

    def test_inf_to_null(self):
        """Positive infinity should be converted to null."""
        data = {"value": float("inf")}
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertIsNone(parsed["value"])

    def test_negative_inf_to_null(self):
        """Negative infinity should be converted to null."""
        data = {"value": float("-inf")}
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertIsNone(parsed["value"])

    def test_normal_float_preserved(self):
        """Normal float values should be preserved."""
        data = {"value": 3.14159}
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertAlmostEqual(parsed["value"], 3.14159, places=5)

    def test_nested_dict_with_nan(self):
        """NaN in nested dict should be converted to null."""
        data = {"level1": {"level2": {"nan_value": float("nan"), "normal": 42}}}
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertIsNone(parsed["level1"]["level2"]["nan_value"])
        self.assertEqual(parsed["level1"]["level2"]["normal"], 42)

    def test_list_with_nan(self):
        """NaN in list should be converted to null."""
        data = {"values": [1.0, float("nan"), 3.0, float("inf")]}
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertEqual(parsed["values"][0], 1.0)
        self.assertIsNone(parsed["values"][1])
        self.assertEqual(parsed["values"][2], 3.0)
        self.assertIsNone(parsed["values"][3])

    def test_mixed_types(self):
        """Mixed types including NaN should be handled correctly."""
        data = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "nan": float("nan"),
            "bool": True,
            "null": None,
            "list": [1, 2, float("nan")],
        }
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertEqual(parsed["string"], "hello")
        self.assertEqual(parsed["int"], 42)
        self.assertAlmostEqual(parsed["float"], 3.14, places=2)
        self.assertIsNone(parsed["nan"])
        self.assertTrue(parsed["bool"])
        self.assertIsNone(parsed["null"])
        self.assertIsNone(parsed["list"][2])

    def test_market_context_like_structure(self):
        """Test with structure similar to actual market context data."""
        data = {
            "GOOGL": {
                "price": 175.50,
                "rsi": float("nan"),  # NaN during ramp-up
                "macd": {
                    "value": float("nan"),
                    "signal": float("nan"),
                    "histogram": float("nan"),
                },
                "sma_20": float("nan"),
                "sma_50": 172.30,
                "volume": 1500000,
                "change_pct": 0.75,
            },
            "TSLA": {
                "price": 250.00,
                "rsi": 55.5,
                "macd": {"value": 2.5, "signal": 1.8, "histogram": 0.7},
                "sma_20": 248.0,
                "sma_50": 245.0,
                "volume": 3000000,
                "change_pct": -1.2,
            },
        }
        result = safe_json_dumps(data)
        parsed = json.loads(result)

        # GOOGL NaN values should be null
        self.assertIsNone(parsed["GOOGL"]["rsi"])
        self.assertIsNone(parsed["GOOGL"]["macd"]["value"])
        self.assertIsNone(parsed["GOOGL"]["macd"]["signal"])
        self.assertIsNone(parsed["GOOGL"]["sma_20"])

        # Normal values should be preserved
        self.assertEqual(parsed["GOOGL"]["price"], 175.50)
        self.assertEqual(parsed["TSLA"]["rsi"], 55.5)

    def test_empty_dict(self):
        """Empty dict should serialize correctly."""
        data: dict[str, Any] = {}
        result = safe_json_dumps(data)
        self.assertEqual(result, "{}")

    def test_empty_list(self):
        """Empty list should serialize correctly."""
        data: list[Any] = []
        result = safe_json_dumps(data)
        self.assertEqual(result, "[]")

    def test_output_is_valid_json(self):
        """Output should always be valid JSON."""
        data = {
            "nan": float("nan"),
            "inf": float("inf"),
            "ninf": float("-inf"),
            "nested": {"more_nan": float("nan")},
        }
        result = safe_json_dumps(data)
        # This should not raise
        parsed = json.loads(result)
        self.assertIsInstance(parsed, dict)


@unittest.skipUnless(HAS_NUMPY, "numpy not available")
class TestNumpyTypes(unittest.TestCase):
    """Test cases for numpy type handling."""

    def test_numpy_nan(self):
        """numpy.nan should be converted to null."""
        data = {"value": np.nan}
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertIsNone(parsed["value"])

    def test_numpy_inf(self):
        """numpy.inf should be converted to null."""
        data = {"value": np.inf}
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertIsNone(parsed["value"])

    def test_numpy_float64(self):
        """numpy.float64 should be converted to Python float."""
        data = {"value": np.float64(3.14159)}
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertAlmostEqual(parsed["value"], 3.14159, places=5)

    def test_numpy_int64(self):
        """numpy.int64 should be converted to Python int."""
        data = {"value": np.int64(42)}
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertEqual(parsed["value"], 42)

    def test_numpy_array(self):
        """numpy array should be converted to list."""
        data = {"values": np.array([1.0, 2.0, np.nan, 4.0])}
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertEqual(parsed["values"][0], 1.0)
        self.assertEqual(parsed["values"][1], 2.0)
        self.assertIsNone(parsed["values"][2])
        self.assertEqual(parsed["values"][3], 4.0)

    def test_numpy_bool(self):
        """numpy.bool_ should be converted to Python bool."""
        data = {"value": np.bool_(True)}
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertTrue(parsed["value"])

    def test_mixed_numpy_and_python(self):
        """Mixed numpy and Python types should work together."""
        data = {
            "py_float": 3.14,
            "np_float": np.float64(2.71),
            "py_int": 42,
            "np_int": np.int64(100),
            "py_nan": float("nan"),
            "np_nan": np.nan,
            "list": [np.float64(1.0), np.nan, 3.0],
        }
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        self.assertAlmostEqual(parsed["py_float"], 3.14, places=2)
        self.assertAlmostEqual(parsed["np_float"], 2.71, places=2)
        self.assertEqual(parsed["py_int"], 42)
        self.assertEqual(parsed["np_int"], 100)
        self.assertIsNone(parsed["py_nan"])
        self.assertIsNone(parsed["np_nan"])


class TestSafeJSONEncoder(unittest.TestCase):
    """Direct tests for SafeJSONEncoder class."""

    def test_encoder_with_json_dumps(self):
        """SafeJSONEncoder should work directly with json.dumps."""
        data = {"nan": float("nan"), "value": 42}
        result = json.dumps(data, cls=SafeJSONEncoder)
        parsed = json.loads(result)
        self.assertIsNone(parsed["nan"])
        self.assertEqual(parsed["value"], 42)


if __name__ == "__main__":
    print("=" * 60)
    print("Running JSON Utils Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
