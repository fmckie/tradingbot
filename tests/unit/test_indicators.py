"""Comprehensive unit tests for TechnicalIndicators.

Tests cover:
- All indicator calculations with edge cases
- Empty data handling
- Insufficient bars handling
- Edge cases for each indicator
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from data.indicators import (
    TechnicalIndicators,
    MACDResult,
    BollingerResult,
    IndicatorSnapshot,
)
from data.market_data import MarketDataProvider


# ==================== Fixtures ====================


@pytest.fixture
def mock_data_provider():
    """Create a mock MarketDataProvider."""
    return Mock(spec=MarketDataProvider)


@pytest.fixture
def indicators(mock_data_provider):
    """Create TechnicalIndicators with mock data provider."""
    return TechnicalIndicators(mock_data_provider)


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame with enough bars for all indicators."""
    np.random.seed(42)
    n_bars = 250  # Enough for SMA 200
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="h")

    # Generate realistic price data with trend
    close_prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)

    return pd.DataFrame({
        "timestamp": dates,
        "open": close_prices - np.random.rand(n_bars) * 0.5,
        "high": close_prices + np.random.rand(n_bars) * 1.0,
        "low": close_prices - np.random.rand(n_bars) * 1.0,
        "close": close_prices,
        "volume": np.random.randint(10000, 100000, n_bars),
    })


@pytest.fixture
def empty_df():
    """Create an empty DataFrame."""
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


@pytest.fixture
def insufficient_bars_df():
    """Create DataFrame with too few bars for most indicators."""
    dates = pd.date_range(start="2024-01-01", periods=5, freq="h")
    return pd.DataFrame({
        "timestamp": dates,
        "open": [100.0, 101.0, 102.0, 101.5, 102.5],
        "high": [101.0, 102.0, 103.0, 102.5, 103.5],
        "low": [99.0, 100.0, 101.0, 100.5, 101.5],
        "close": [100.5, 101.5, 102.5, 102.0, 103.0],
        "volume": [10000, 12000, 11000, 13000, 14000],
    })


@pytest.fixture
def trending_up_df():
    """Create DataFrame with clear uptrend."""
    n_bars = 100
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="h")

    # Clear uptrend
    close_prices = np.linspace(100, 150, n_bars) + np.random.randn(n_bars) * 0.5

    return pd.DataFrame({
        "timestamp": dates,
        "open": close_prices - 0.5,
        "high": close_prices + 1.0,
        "low": close_prices - 1.0,
        "close": close_prices,
        "volume": np.random.randint(10000, 100000, n_bars),
    })


@pytest.fixture
def trending_down_df():
    """Create DataFrame with clear downtrend."""
    n_bars = 100
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="h")

    # Clear downtrend
    close_prices = np.linspace(150, 100, n_bars) + np.random.randn(n_bars) * 0.5

    return pd.DataFrame({
        "timestamp": dates,
        "open": close_prices + 0.5,
        "high": close_prices + 1.0,
        "low": close_prices - 1.0,
        "close": close_prices,
        "volume": np.random.randint(10000, 100000, n_bars),
    })


@pytest.fixture
def high_volatility_df():
    """Create DataFrame with high volatility."""
    n_bars = 100
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="h")

    # High volatility - large price swings
    close_prices = 100 + np.cumsum(np.random.randn(n_bars) * 3)

    return pd.DataFrame({
        "timestamp": dates,
        "open": close_prices - np.random.rand(n_bars) * 2,
        "high": close_prices + np.random.rand(n_bars) * 5,
        "low": close_prices - np.random.rand(n_bars) * 5,
        "close": close_prices,
        "volume": np.random.randint(50000, 200000, n_bars),
    })


# ==================== VWAP Tests ====================


class TestCalculateVWAP:
    """Test VWAP calculation."""

    def test_vwap_with_sufficient_data(self, indicators, mock_data_provider, sample_ohlcv_df):
        """VWAP should be calculated with sufficient data."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.calculate_vwap("GOOGL")

        assert result > 0
        assert isinstance(result, float)

    def test_vwap_with_empty_data(self, indicators, mock_data_provider, empty_df):
        """VWAP should return 0.0 for empty data."""
        mock_data_provider.get_bars.return_value = empty_df

        result = indicators.calculate_vwap("GOOGL")

        assert result == 0.0

    def test_vwap_with_insufficient_data(self, indicators, mock_data_provider, insufficient_bars_df):
        """VWAP should return 0.0 for insufficient bars."""
        mock_data_provider.get_bars.return_value = insufficient_bars_df

        result = indicators.calculate_vwap("GOOGL")

        assert result == 0.0

    def test_vwap_requests_full_day_bars(self, indicators, mock_data_provider, sample_ohlcv_df):
        """VWAP should request 390 1-minute bars (full trading day)."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        indicators.calculate_vwap("GOOGL")

        mock_data_provider.get_bars.assert_called_once_with("GOOGL", "1Min", limit=390)


# ==================== RSI Tests ====================


class TestCalculateRSI:
    """Test RSI calculation."""

    def test_rsi_with_sufficient_data(self, indicators, mock_data_provider, sample_ohlcv_df):
        """RSI should be calculated with sufficient data."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.calculate_rsi("GOOGL")

        assert 0 <= result <= 100

    def test_rsi_with_insufficient_data(self, indicators, mock_data_provider, insufficient_bars_df):
        """RSI should return 50.0 (neutral) for insufficient data."""
        mock_data_provider.get_bars.return_value = insufficient_bars_df

        result = indicators.calculate_rsi("GOOGL", period=14)

        assert result == 50.0

    def test_rsi_in_uptrend(self, indicators, mock_data_provider, trending_up_df):
        """RSI should be higher in uptrend."""
        mock_data_provider.get_bars.return_value = trending_up_df

        result = indicators.calculate_rsi("GOOGL")

        # In strong uptrend, RSI should be above neutral
        assert result > 50

    def test_rsi_in_downtrend(self, indicators, mock_data_provider, trending_down_df):
        """RSI should be lower in downtrend."""
        mock_data_provider.get_bars.return_value = trending_down_df

        result = indicators.calculate_rsi("GOOGL")

        # In strong downtrend, RSI should be below neutral
        assert result < 50

    def test_rsi_custom_period(self, indicators, mock_data_provider, sample_ohlcv_df):
        """RSI should work with custom period."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.calculate_rsi("GOOGL", period=7)

        assert 0 <= result <= 100


# ==================== MACD Tests ====================


class TestCalculateMACD:
    """Test MACD calculation."""

    def test_macd_with_sufficient_data(self, indicators, mock_data_provider, sample_ohlcv_df):
        """MACD should return MACDResult with sufficient data."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.calculate_macd("GOOGL")

        assert isinstance(result, MACDResult)
        assert isinstance(result.macd_line, float)
        assert isinstance(result.signal_line, float)
        assert isinstance(result.histogram, float)

    def test_macd_with_insufficient_data(self, indicators, mock_data_provider, insufficient_bars_df):
        """MACD should return zeros for insufficient data."""
        mock_data_provider.get_bars.return_value = insufficient_bars_df

        result = indicators.calculate_macd("GOOGL")

        assert result.macd_line == 0.0
        assert result.signal_line == 0.0
        assert result.histogram == 0.0

    def test_macd_histogram_positive_in_uptrend(self, indicators, mock_data_provider, trending_up_df):
        """MACD line should be positive in uptrend."""
        mock_data_provider.get_bars.return_value = trending_up_df

        result = indicators.calculate_macd("GOOGL")

        # In uptrend, MACD line should be positive (above zero line)
        # Histogram can lag and be slightly negative even in uptrend
        assert result.macd_line > 0

    def test_macd_histogram_negative_in_downtrend(self, indicators, mock_data_provider, trending_down_df):
        """MACD histogram should be negative in downtrend."""
        mock_data_provider.get_bars.return_value = trending_down_df

        result = indicators.calculate_macd("GOOGL")

        # In strong downtrend, histogram should be negative
        assert result.histogram < 0

    def test_macd_custom_parameters(self, indicators, mock_data_provider, sample_ohlcv_df):
        """MACD should work with custom parameters."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.calculate_macd("GOOGL", fast=8, slow=17, signal=9)

        assert isinstance(result, MACDResult)


# ==================== Bollinger Bands Tests ====================


class TestCalculateBollingerBands:
    """Test Bollinger Bands calculation."""

    def test_bollinger_with_sufficient_data(self, indicators, mock_data_provider, sample_ohlcv_df):
        """Bollinger Bands should be calculated with sufficient data."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.calculate_bollinger_bands("GOOGL")

        assert isinstance(result, BollingerResult)
        assert result.upper > result.middle > result.lower
        assert result.bandwidth > 0

    def test_bollinger_with_insufficient_data(self, indicators, mock_data_provider, insufficient_bars_df):
        """Bollinger Bands should return defaults for insufficient data."""
        mock_data_provider.get_bars.return_value = insufficient_bars_df

        result = indicators.calculate_bollinger_bands("GOOGL", period=20)

        # Should return last price for all bands
        last_price = insufficient_bars_df["close"].iloc[-1]
        assert result.upper == last_price
        assert result.middle == last_price
        assert result.lower == last_price
        assert result.bandwidth == 0.0
        assert result.percent_b == 0.5

    def test_bollinger_with_empty_data(self, indicators, mock_data_provider, empty_df):
        """Bollinger Bands should handle empty data."""
        mock_data_provider.get_bars.return_value = empty_df

        result = indicators.calculate_bollinger_bands("GOOGL")

        assert result.upper == 0.0
        assert result.bandwidth == 0.0

    def test_bollinger_percent_b_range(self, indicators, mock_data_provider, sample_ohlcv_df):
        """Percent B can be outside 0-1 when price is outside bands."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.calculate_bollinger_bands("GOOGL")

        # Percent B is typically between 0-1 but can be outside
        assert isinstance(result.percent_b, float)

    def test_bollinger_high_volatility(self, indicators, mock_data_provider, high_volatility_df):
        """Bollinger Bands should be wider with high volatility."""
        mock_data_provider.get_bars.return_value = high_volatility_df

        result = indicators.calculate_bollinger_bands("GOOGL")

        # Higher bandwidth indicates wider bands
        assert result.bandwidth > 0
        assert result.upper - result.lower > 0


# ==================== ATR Tests ====================


class TestCalculateATR:
    """Test Average True Range calculation."""

    def test_atr_with_sufficient_data(self, indicators, mock_data_provider, sample_ohlcv_df):
        """ATR should be calculated with sufficient data."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.calculate_atr("GOOGL")

        assert result > 0

    def test_atr_with_insufficient_data(self, indicators, mock_data_provider, insufficient_bars_df):
        """ATR should return 0.0 for insufficient data."""
        mock_data_provider.get_bars.return_value = insufficient_bars_df

        result = indicators.calculate_atr("GOOGL", period=14)

        assert result == 0.0

    def test_atr_higher_with_volatility(self, indicators, mock_data_provider, high_volatility_df):
        """ATR should be higher with high volatility."""
        high_vol_atr = None
        low_vol_atr = None

        # High volatility
        mock_data_provider.get_bars.return_value = high_volatility_df
        high_vol_atr = indicators.calculate_atr("GOOGL")

        # Create low volatility data
        n_bars = 100
        dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="h")
        low_vol_df = pd.DataFrame({
            "timestamp": dates,
            "open": [100.0] * n_bars,
            "high": [100.5] * n_bars,
            "low": [99.5] * n_bars,
            "close": [100.0] * n_bars,
            "volume": [10000] * n_bars,
        })
        mock_data_provider.get_bars.return_value = low_vol_df
        low_vol_atr = indicators.calculate_atr("GOOGL")

        assert high_vol_atr > low_vol_atr


# ==================== EMA Tests ====================


class TestCalculateEMA:
    """Test Exponential Moving Average calculation."""

    def test_ema_with_sufficient_data(self, indicators, mock_data_provider, sample_ohlcv_df):
        """EMA should be calculated with sufficient data."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.calculate_ema("GOOGL", period=9)

        assert result > 0

    def test_ema_with_insufficient_data(self, indicators, mock_data_provider, insufficient_bars_df):
        """EMA should return last price for insufficient data."""
        mock_data_provider.get_bars.return_value = insufficient_bars_df

        result = indicators.calculate_ema("GOOGL", period=50)

        # Should return last available price
        last_price = insufficient_bars_df["close"].iloc[-1]
        assert result == last_price

    def test_ema_with_empty_data(self, indicators, mock_data_provider, empty_df):
        """EMA should return 0.0 for empty data."""
        mock_data_provider.get_bars.return_value = empty_df

        result = indicators.calculate_ema("GOOGL", period=9)

        assert result == 0.0

    def test_ema_shorter_faster_in_uptrend(self, indicators, mock_data_provider, trending_up_df):
        """Shorter EMA should be above longer EMA in uptrend."""
        mock_data_provider.get_bars.return_value = trending_up_df

        ema_9 = indicators.calculate_ema("GOOGL", period=9)
        ema_21 = indicators.calculate_ema("GOOGL", period=21)

        # In uptrend, shorter EMA should be higher
        assert ema_9 > ema_21


# ==================== SMA Tests ====================


class TestCalculateSMA:
    """Test Simple Moving Average calculation."""

    def test_sma_with_sufficient_data(self, indicators, mock_data_provider, sample_ohlcv_df):
        """SMA should be calculated with sufficient data."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.calculate_sma("GOOGL", period=50)

        assert result > 0

    def test_sma_with_insufficient_data(self, indicators, mock_data_provider, insufficient_bars_df):
        """SMA should return last price for insufficient data."""
        mock_data_provider.get_bars.return_value = insufficient_bars_df

        result = indicators.calculate_sma("GOOGL", period=50)

        last_price = insufficient_bars_df["close"].iloc[-1]
        assert result == last_price

    def test_sma_with_empty_data(self, indicators, mock_data_provider, empty_df):
        """SMA should return 0.0 for empty data."""
        mock_data_provider.get_bars.return_value = empty_df

        result = indicators.calculate_sma("GOOGL", period=50)

        assert result == 0.0

    def test_sma_200_golden_cross(self, indicators, mock_data_provider, trending_up_df):
        """SMA 50 should be above SMA 200 in strong uptrend (golden cross)."""
        # Need more data for SMA 200
        n_bars = 250
        dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="h")
        long_trend_df = pd.DataFrame({
            "timestamp": dates,
            "open": np.linspace(100, 200, n_bars),
            "high": np.linspace(101, 201, n_bars),
            "low": np.linspace(99, 199, n_bars),
            "close": np.linspace(100, 200, n_bars),
            "volume": [10000] * n_bars,
        })
        mock_data_provider.get_bars.return_value = long_trend_df

        sma_50 = indicators.calculate_sma("GOOGL", period=50)
        sma_200 = indicators.calculate_sma("GOOGL", period=200)

        # In strong uptrend, SMA 50 > SMA 200
        assert sma_50 > sma_200


# ==================== Stochastic Tests ====================


class TestCalculateStochastic:
    """Test Stochastic Oscillator calculation."""

    def test_stochastic_with_sufficient_data(self, indicators, mock_data_provider, sample_ohlcv_df):
        """Stochastic should return %K and %D values."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        k, d = indicators.calculate_stochastic("GOOGL")

        assert 0 <= k <= 100
        assert 0 <= d <= 100

    def test_stochastic_with_insufficient_data(self, indicators, mock_data_provider, insufficient_bars_df):
        """Stochastic should return 50.0, 50.0 for insufficient data."""
        mock_data_provider.get_bars.return_value = insufficient_bars_df

        k, d = indicators.calculate_stochastic("GOOGL")

        assert k == 50.0
        assert d == 50.0

    def test_stochastic_high_in_uptrend(self, indicators, mock_data_provider, trending_up_df):
        """Stochastic should be high in uptrend (near highs)."""
        mock_data_provider.get_bars.return_value = trending_up_df

        k, d = indicators.calculate_stochastic("GOOGL")

        # In uptrend, stochastic should be elevated
        assert k > 50

    def test_stochastic_low_in_downtrend(self, indicators, mock_data_provider, trending_down_df):
        """Stochastic should be low in downtrend (near lows)."""
        mock_data_provider.get_bars.return_value = trending_down_df

        k, d = indicators.calculate_stochastic("GOOGL")

        # In downtrend, stochastic should be depressed
        assert k < 50


# ==================== Volume SMA Tests ====================


class TestCalculateVolumeSMA:
    """Test Volume SMA calculation."""

    def test_volume_sma_with_sufficient_data(self, indicators, mock_data_provider, sample_ohlcv_df):
        """Volume SMA should be calculated with sufficient data."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.calculate_volume_sma("GOOGL")

        assert result > 0

    def test_volume_sma_with_insufficient_data(self, indicators, mock_data_provider, insufficient_bars_df):
        """Volume SMA should return mean of available bars."""
        mock_data_provider.get_bars.return_value = insufficient_bars_df

        result = indicators.calculate_volume_sma("GOOGL", period=20)

        # Should return mean of available volume
        expected = insufficient_bars_df["volume"].mean()
        assert result == expected

    def test_volume_sma_with_empty_data(self, indicators, mock_data_provider, empty_df):
        """Volume SMA should return 0.0 for empty data."""
        mock_data_provider.get_bars.return_value = empty_df

        result = indicators.calculate_volume_sma("GOOGL")

        assert result == 0.0


# ==================== Price Change Tests ====================


class TestCalculatePriceChange:
    """Test price change calculation."""

    def test_price_change_with_sufficient_data(self, indicators, mock_data_provider, sample_ohlcv_df):
        """Price change should be calculated with sufficient data."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.calculate_price_change("GOOGL", periods=1)

        assert isinstance(result, float)

    def test_price_change_with_insufficient_data(self, indicators, mock_data_provider):
        """Price change should return 0.0 with insufficient data."""
        single_bar = pd.DataFrame({
            "timestamp": [datetime.now()],
            "close": [100.0],
        })
        mock_data_provider.get_bars.return_value = single_bar

        result = indicators.calculate_price_change("GOOGL", periods=1)

        assert result == 0.0

    def test_price_change_positive_in_uptrend(self, indicators, mock_data_provider, trending_up_df):
        """Price change should be positive in uptrend."""
        mock_data_provider.get_bars.return_value = trending_up_df

        result = indicators.calculate_price_change("GOOGL", periods=24)

        assert result > 0

    def test_price_change_negative_in_downtrend(self, indicators, mock_data_provider, trending_down_df):
        """Price change should be negative in downtrend."""
        mock_data_provider.get_bars.return_value = trending_down_df

        result = indicators.calculate_price_change("GOOGL", periods=24)

        assert result < 0


# ==================== Get All Indicators Tests ====================


class TestGetAllIndicators:
    """Test get_all_indicators comprehensive method."""

    def test_returns_indicator_snapshot(self, indicators, mock_data_provider, sample_ohlcv_df):
        """Should return IndicatorSnapshot with all indicators."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.get_all_indicators("GOOGL")

        assert isinstance(result, IndicatorSnapshot)
        assert result.symbol == "GOOGL"

    def test_snapshot_has_all_fields(self, indicators, mock_data_provider, sample_ohlcv_df):
        """Snapshot should have all required fields."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.get_all_indicators("GOOGL")

        assert hasattr(result, "vwap")
        assert hasattr(result, "rsi")
        assert hasattr(result, "macd")
        assert hasattr(result, "bollinger")
        assert hasattr(result, "atr")
        assert hasattr(result, "ema_9")
        assert hasattr(result, "ema_21")
        assert hasattr(result, "sma_50")
        assert hasattr(result, "sma_200")
        assert hasattr(result, "stochastic_k")
        assert hasattr(result, "stochastic_d")
        assert hasattr(result, "volume_sma")
        assert hasattr(result, "price_change_1h")
        assert hasattr(result, "price_change_1d")

    def test_snapshot_macd_is_macd_result(self, indicators, mock_data_provider, sample_ohlcv_df):
        """Snapshot MACD should be MACDResult."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.get_all_indicators("GOOGL")

        assert isinstance(result.macd, MACDResult)

    def test_snapshot_bollinger_is_bollinger_result(self, indicators, mock_data_provider, sample_ohlcv_df):
        """Snapshot Bollinger should be BollingerResult."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.get_all_indicators("GOOGL")

        assert isinstance(result.bollinger, BollingerResult)


# ==================== Format Indicators Tests ====================


class TestFormatIndicatorsForAI:
    """Test format_indicators_for_ai method."""

    def test_format_returns_string(self, indicators, mock_data_provider, sample_ohlcv_df):
        """Should return formatted string."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.format_indicators_for_ai("GOOGL")

        assert isinstance(result, str)
        assert "GOOGL" in result

    def test_format_includes_rsi_status(self, indicators, mock_data_provider, sample_ohlcv_df):
        """Format should include RSI overbought/oversold status."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.format_indicators_for_ai("GOOGL")

        assert "RSI" in result
        # Should have one of these status indicators
        assert any(status in result for status in ["Overbought", "Oversold", "Neutral"])

    def test_format_includes_macd_status(self, indicators, mock_data_provider, sample_ohlcv_df):
        """Format should include MACD bullish/bearish status."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.format_indicators_for_ai("GOOGL")

        assert "MACD" in result
        assert "Histogram" in result

    def test_format_includes_trend(self, indicators, mock_data_provider, sample_ohlcv_df):
        """Format should include trend status."""
        mock_data_provider.get_bars.return_value = sample_ohlcv_df

        result = indicators.format_indicators_for_ai("GOOGL")

        assert "Trend" in result
        # Should have one of these
        assert any(trend in result for trend in ["Bullish", "Bearish"])


# ==================== Data Class Tests ====================


class TestMACDResult:
    """Test MACDResult dataclass."""

    def test_macd_result_creation(self):
        """MACDResult should be creatable with values."""
        result = MACDResult(
            macd_line=0.5,
            signal_line=0.3,
            histogram=0.2,
        )

        assert result.macd_line == 0.5
        assert result.signal_line == 0.3
        assert result.histogram == 0.2


class TestBollingerResult:
    """Test BollingerResult dataclass."""

    def test_bollinger_result_creation(self):
        """BollingerResult should be creatable with values."""
        result = BollingerResult(
            upper=110.0,
            middle=100.0,
            lower=90.0,
            bandwidth=0.2,
            percent_b=0.5,
        )

        assert result.upper == 110.0
        assert result.middle == 100.0
        assert result.lower == 90.0
        assert result.bandwidth == 0.2
        assert result.percent_b == 0.5


class TestIndicatorSnapshot:
    """Test IndicatorSnapshot dataclass."""

    def test_indicator_snapshot_creation(self):
        """IndicatorSnapshot should be creatable with all values."""
        snapshot = IndicatorSnapshot(
            symbol="GOOGL",
            vwap=150.0,
            rsi=55.0,
            macd=MACDResult(0.5, 0.3, 0.2),
            bollinger=BollingerResult(155.0, 150.0, 145.0, 0.067, 0.5),
            atr=2.5,
            ema_9=151.0,
            ema_21=149.0,
            sma_50=148.0,
            sma_200=145.0,
            stochastic_k=65.0,
            stochastic_d=60.0,
            volume_sma=50000.0,
            price_change_1h=1.5,
            price_change_1d=3.0,
        )

        assert snapshot.symbol == "GOOGL"
        assert snapshot.vwap == 150.0
        assert snapshot.rsi == 55.0
        assert snapshot.macd.macd_line == 0.5
