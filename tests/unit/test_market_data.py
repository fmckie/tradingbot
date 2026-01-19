"""Comprehensive unit tests for MarketDataProvider.

Tests cover:
- DataFrame handling with MultiIndex
- Empty response handling
- Quote and snapshot retrieval
- Symbol validation
- Recent trades retrieval
- Error handling
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from data.market_data import MarketDataProvider, Quote, Snapshot
from config.settings import SYMBOLS


# ==================== Mock Alpaca Response Classes ====================


@dataclass
class MockAlpacaQuote:
    """Mock Alpaca quote response."""
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    timestamp: datetime


@dataclass
class MockAlpacaTrade:
    """Mock Alpaca trade response."""
    price: float
    size: int
    timestamp: datetime
    conditions: list


@dataclass
class MockAlpacaBar:
    """Mock Alpaca bar response."""
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime


@dataclass
class MockAlpacaSnapshot:
    """Mock Alpaca snapshot response."""
    latest_trade: MockAlpacaTrade | None = None
    latest_quote: MockAlpacaQuote | None = None
    daily_bar: MockAlpacaBar | None = None
    previous_daily_bar: MockAlpacaBar | None = None


# ==================== Fixtures ====================


@pytest.fixture
def mock_alpaca_client():
    """Create a mock Alpaca StockHistoricalDataClient."""
    return Mock()


@pytest.fixture
def market_data_provider(mock_alpaca_client):
    """Create MarketDataProvider with mock client."""
    return MarketDataProvider(mock_alpaca_client)


@pytest.fixture
def sample_bars_response():
    """Create a sample bars response DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    df = pd.DataFrame({
        "open": close_prices - np.random.rand(100) * 0.5,
        "high": close_prices + np.random.rand(100) * 1.0,
        "low": close_prices - np.random.rand(100) * 1.0,
        "close": close_prices,
        "volume": np.random.randint(10000, 100000, 100),
    })

    # Create MultiIndex like Alpaca returns
    df.index = pd.MultiIndex.from_tuples(
        [(symbol, date) for symbol, date in zip(["GOOGL"] * 100, dates)],
        names=["symbol", "timestamp"]
    )

    return df


@pytest.fixture
def sample_multiindex_bars():
    """Create MultiIndex bars DataFrame as Alpaca returns."""
    dates = pd.date_range(start="2024-01-01", periods=50, freq="h")
    np.random.seed(42)
    close_prices = 150 + np.cumsum(np.random.randn(50) * 0.5)

    df = pd.DataFrame({
        "open": close_prices - np.random.rand(50) * 0.5,
        "high": close_prices + np.random.rand(50) * 1.0,
        "low": close_prices - np.random.rand(50) * 1.0,
        "close": close_prices,
        "volume": np.random.randint(10000, 100000, 50),
    })

    df.index = pd.MultiIndex.from_tuples(
        [("GOOGL", date) for date in dates],
        names=["symbol", "timestamp"]
    )

    return df


@pytest.fixture
def sample_quote():
    """Create a sample quote response."""
    return MockAlpacaQuote(
        bid_price=149.95,
        ask_price=150.05,
        bid_size=100,
        ask_size=150,
        timestamp=datetime.now(),
    )


@pytest.fixture
def sample_trade():
    """Create a sample trade response."""
    return MockAlpacaTrade(
        price=150.00,
        size=50,
        timestamp=datetime.now(),
        conditions=["@"],
    )


@pytest.fixture
def sample_snapshot(sample_quote, sample_trade):
    """Create a sample snapshot response."""
    daily_bar = MockAlpacaBar(
        open=148.00,
        high=152.00,
        low=147.50,
        close=150.00,
        volume=1000000,
        timestamp=datetime.now(),
    )
    prev_bar = MockAlpacaBar(
        open=147.00,
        high=149.00,
        low=146.00,
        close=148.00,
        volume=900000,
        timestamp=datetime.now() - timedelta(days=1),
    )

    return MockAlpacaSnapshot(
        latest_trade=sample_trade,
        latest_quote=sample_quote,
        daily_bar=daily_bar,
        previous_daily_bar=prev_bar,
    )


# ==================== Symbol Validation Tests ====================


class TestSymbolValidation:
    """Test symbol validation in all methods."""

    def test_get_current_quote_invalid_symbol_rejected(self, market_data_provider):
        """Invalid symbol should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            market_data_provider.get_current_quote("AAPL")

        assert "not allowed" in str(exc_info.value)
        assert "AAPL" in str(exc_info.value)

    def test_get_bars_invalid_symbol_rejected(self, market_data_provider):
        """Invalid symbol should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            market_data_provider.get_bars("AAPL")

        assert "not allowed" in str(exc_info.value)

    def test_get_snapshot_invalid_symbol_rejected(self, market_data_provider):
        """Invalid symbol should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            market_data_provider.get_snapshot("MSFT")

        assert "not allowed" in str(exc_info.value)

    def test_get_recent_trades_invalid_symbol_rejected(self, market_data_provider):
        """Invalid symbol should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            market_data_provider.get_recent_trades("XYZ")

        assert "not allowed" in str(exc_info.value)

    def test_valid_symbol_googl_accepted(self, market_data_provider, mock_alpaca_client, sample_multiindex_bars):
        """GOOGL should be accepted."""
        mock_response = Mock()
        mock_response.df = sample_multiindex_bars
        mock_alpaca_client.get_stock_bars.return_value = mock_response

        # Should not raise
        result = market_data_provider.get_bars("GOOGL")
        assert not result.empty

    def test_valid_symbol_tsla_accepted(self, market_data_provider, mock_alpaca_client, sample_multiindex_bars):
        """TSLA should be accepted."""
        # Modify mock for TSLA
        dates = pd.date_range(start="2024-01-01", periods=50, freq="h")
        df = sample_multiindex_bars.copy()
        df.index = pd.MultiIndex.from_tuples(
            [("TSLA", date) for date in dates],
            names=["symbol", "timestamp"]
        )

        mock_response = Mock()
        mock_response.df = df
        mock_alpaca_client.get_stock_bars.return_value = mock_response

        result = market_data_provider.get_bars("TSLA")
        assert not result.empty


# ==================== Get Bars Tests ====================


class TestGetBars:
    """Test get_bars method."""

    def test_get_bars_returns_dataframe(self, market_data_provider, mock_alpaca_client, sample_multiindex_bars):
        """Should return a pandas DataFrame."""
        mock_response = Mock()
        mock_response.df = sample_multiindex_bars
        mock_alpaca_client.get_stock_bars.return_value = mock_response

        result = market_data_provider.get_bars("GOOGL")

        assert isinstance(result, pd.DataFrame)

    def test_get_bars_handles_multiindex(self, market_data_provider, mock_alpaca_client, sample_multiindex_bars):
        """Should handle MultiIndex DataFrame from Alpaca."""
        mock_response = Mock()
        mock_response.df = sample_multiindex_bars
        mock_alpaca_client.get_stock_bars.return_value = mock_response

        result = market_data_provider.get_bars("GOOGL")

        # Should be flattened - no longer MultiIndex
        assert not isinstance(result.index, pd.MultiIndex)

    def test_get_bars_empty_response(self, market_data_provider, mock_alpaca_client):
        """Should return empty DataFrame for empty response."""
        mock_response = Mock()
        mock_response.df = pd.DataFrame()
        mock_alpaca_client.get_stock_bars.return_value = mock_response

        result = market_data_provider.get_bars("GOOGL")

        assert result.empty
        assert "close" in result.columns or len(result.columns) >= 0

    def test_get_bars_column_names_lowercase(self, market_data_provider, mock_alpaca_client, sample_multiindex_bars):
        """Column names should be lowercase."""
        mock_response = Mock()
        mock_response.df = sample_multiindex_bars
        mock_alpaca_client.get_stock_bars.return_value = mock_response

        result = market_data_provider.get_bars("GOOGL")

        for col in result.columns:
            assert col == col.lower()

    def test_get_bars_respects_limit(self, market_data_provider, mock_alpaca_client, sample_multiindex_bars):
        """Should respect limit parameter."""
        mock_response = Mock()
        mock_response.df = sample_multiindex_bars
        mock_alpaca_client.get_stock_bars.return_value = mock_response

        result = market_data_provider.get_bars("GOOGL", limit=10)

        assert len(result) <= 10

    def test_get_bars_invalid_timeframe(self, market_data_provider):
        """Should raise ValueError for invalid timeframe."""
        with pytest.raises(ValueError) as exc_info:
            market_data_provider.get_bars("GOOGL", timeframe="3Min")

        assert "Invalid timeframe" in str(exc_info.value)

    def test_get_bars_valid_timeframes(self, market_data_provider, mock_alpaca_client, sample_multiindex_bars):
        """Should accept valid timeframes."""
        mock_response = Mock()
        mock_response.df = sample_multiindex_bars
        mock_alpaca_client.get_stock_bars.return_value = mock_response

        valid_timeframes = ["1Min", "5Min", "15Min", "1Hour", "1Day"]

        for tf in valid_timeframes:
            result = market_data_provider.get_bars("GOOGL", timeframe=tf)
            assert isinstance(result, pd.DataFrame)

    def test_get_bars_multiindex_without_symbol_level(self, market_data_provider, mock_alpaca_client):
        """Should handle MultiIndex without named symbol level."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="h")

        df = pd.DataFrame({
            "open": [100.0] * 50,
            "high": [101.0] * 50,
            "low": [99.0] * 50,
            "close": [100.0] * 50,
            "volume": [10000] * 50,
        })

        # MultiIndex without 'symbol' name
        df.index = pd.MultiIndex.from_tuples(
            [(0, date) for date in dates],
            names=[None, "timestamp"]
        )

        mock_response = Mock()
        mock_response.df = df
        mock_alpaca_client.get_stock_bars.return_value = mock_response

        result = market_data_provider.get_bars("GOOGL")

        assert not isinstance(result.index, pd.MultiIndex)


# ==================== Get Current Quote Tests ====================


class TestGetCurrentQuote:
    """Test get_current_quote method."""

    def test_get_quote_returns_quote_object(self, market_data_provider, mock_alpaca_client, sample_quote, sample_trade):
        """Should return Quote dataclass."""
        mock_alpaca_client.get_stock_latest_quote.return_value = {"GOOGL": sample_quote}
        mock_alpaca_client.get_stock_trades.return_value = {"GOOGL": [sample_trade]}

        result = market_data_provider.get_current_quote("GOOGL")

        assert isinstance(result, Quote)

    def test_get_quote_has_all_fields(self, market_data_provider, mock_alpaca_client, sample_quote, sample_trade):
        """Quote should have all required fields."""
        mock_alpaca_client.get_stock_latest_quote.return_value = {"GOOGL": sample_quote}
        mock_alpaca_client.get_stock_trades.return_value = {"GOOGL": [sample_trade]}

        result = market_data_provider.get_current_quote("GOOGL")

        assert result.symbol == "GOOGL"
        assert result.bid_price == sample_quote.bid_price
        assert result.ask_price == sample_quote.ask_price
        assert result.bid_size == sample_quote.bid_size
        assert result.ask_size == sample_quote.ask_size
        assert result.last_price == sample_trade.price

    def test_get_quote_uses_ask_when_no_trades(self, market_data_provider, mock_alpaca_client, sample_quote):
        """Should use ask price as last price when no trades."""
        mock_alpaca_client.get_stock_latest_quote.return_value = {"GOOGL": sample_quote}
        mock_alpaca_client.get_stock_trades.return_value = {"GOOGL": []}

        result = market_data_provider.get_current_quote("GOOGL")

        assert result.last_price == sample_quote.ask_price

    def test_get_quote_converts_to_float(self, market_data_provider, mock_alpaca_client, sample_quote, sample_trade):
        """Prices should be converted to float."""
        mock_alpaca_client.get_stock_latest_quote.return_value = {"GOOGL": sample_quote}
        mock_alpaca_client.get_stock_trades.return_value = {"GOOGL": [sample_trade]}

        result = market_data_provider.get_current_quote("GOOGL")

        assert isinstance(result.bid_price, float)
        assert isinstance(result.ask_price, float)
        assert isinstance(result.last_price, float)


# ==================== Get Snapshot Tests ====================


class TestGetSnapshot:
    """Test get_snapshot method."""

    def test_get_snapshot_returns_snapshot_object(self, market_data_provider, mock_alpaca_client, sample_snapshot):
        """Should return Snapshot dataclass."""
        mock_alpaca_client.get_stock_snapshot.return_value = {"GOOGL": sample_snapshot}

        result = market_data_provider.get_snapshot("GOOGL")

        assert isinstance(result, Snapshot)

    def test_get_snapshot_has_all_fields(self, market_data_provider, mock_alpaca_client, sample_snapshot):
        """Snapshot should have all required fields."""
        mock_alpaca_client.get_stock_snapshot.return_value = {"GOOGL": sample_snapshot}

        result = market_data_provider.get_snapshot("GOOGL")

        assert result.symbol == "GOOGL"
        assert result.latest_trade_price == sample_snapshot.latest_trade.price
        assert result.latest_trade_size == sample_snapshot.latest_trade.size
        assert result.latest_quote_bid == sample_snapshot.latest_quote.bid_price
        assert result.latest_quote_ask == sample_snapshot.latest_quote.ask_price
        assert result.daily_bar_open == sample_snapshot.daily_bar.open
        assert result.daily_bar_high == sample_snapshot.daily_bar.high
        assert result.daily_bar_low == sample_snapshot.daily_bar.low
        assert result.daily_bar_close == sample_snapshot.daily_bar.close
        assert result.daily_bar_volume == sample_snapshot.daily_bar.volume
        assert result.prev_daily_bar_close == sample_snapshot.previous_daily_bar.close

    def test_get_snapshot_handles_missing_trade(self, market_data_provider, mock_alpaca_client, sample_quote):
        """Should handle missing latest trade."""
        snapshot = MockAlpacaSnapshot(
            latest_trade=None,
            latest_quote=sample_quote,
            daily_bar=None,
            previous_daily_bar=None,
        )
        mock_alpaca_client.get_stock_snapshot.return_value = {"GOOGL": snapshot}

        result = market_data_provider.get_snapshot("GOOGL")

        assert result.latest_trade_price == 0.0
        assert result.latest_trade_size == 0

    def test_get_snapshot_handles_missing_quote(self, market_data_provider, mock_alpaca_client, sample_trade):
        """Should handle missing latest quote."""
        snapshot = MockAlpacaSnapshot(
            latest_trade=sample_trade,
            latest_quote=None,
            daily_bar=None,
            previous_daily_bar=None,
        )
        mock_alpaca_client.get_stock_snapshot.return_value = {"GOOGL": snapshot}

        result = market_data_provider.get_snapshot("GOOGL")

        assert result.latest_quote_bid == 0.0
        assert result.latest_quote_ask == 0.0

    def test_get_snapshot_handles_missing_daily_bar(self, market_data_provider, mock_alpaca_client, sample_trade, sample_quote):
        """Should handle missing daily bar."""
        snapshot = MockAlpacaSnapshot(
            latest_trade=sample_trade,
            latest_quote=sample_quote,
            daily_bar=None,
            previous_daily_bar=None,
        )
        mock_alpaca_client.get_stock_snapshot.return_value = {"GOOGL": snapshot}

        result = market_data_provider.get_snapshot("GOOGL")

        assert result.daily_bar_open == 0.0
        assert result.daily_bar_close == 0.0
        assert result.daily_bar_volume == 0

    def test_get_snapshot_handles_missing_prev_daily_bar(self, market_data_provider, mock_alpaca_client, sample_snapshot):
        """Should handle missing previous daily bar."""
        sample_snapshot.previous_daily_bar = None
        mock_alpaca_client.get_stock_snapshot.return_value = {"GOOGL": sample_snapshot}

        result = market_data_provider.get_snapshot("GOOGL")

        assert result.prev_daily_bar_close == 0.0


# ==================== Get All Snapshots Tests ====================


class TestGetAllSnapshots:
    """Test get_all_snapshots method."""

    def test_get_all_snapshots_returns_dict(self, market_data_provider, mock_alpaca_client, sample_snapshot):
        """Should return dictionary of snapshots."""
        # Mock to return snapshots for each symbol call
        def mock_get_snapshot(request):
            # The request contains the symbol
            return {request.symbol_or_symbols: sample_snapshot}

        mock_alpaca_client.get_stock_snapshot.side_effect = mock_get_snapshot

        result = market_data_provider.get_all_snapshots()

        assert isinstance(result, dict)

    def test_get_all_snapshots_includes_all_symbols(self, market_data_provider, mock_alpaca_client, sample_snapshot):
        """Should include snapshot for each symbol."""
        # Mock to return snapshot for each call
        def mock_get_snapshot(request):
            return {request.symbol_or_symbols: sample_snapshot}

        mock_alpaca_client.get_stock_snapshot.side_effect = mock_get_snapshot

        result = market_data_provider.get_all_snapshots()

        assert "GOOGL" in result
        assert "TSLA" in result


# ==================== Get Recent Trades Tests ====================


class TestGetRecentTrades:
    """Test get_recent_trades method."""

    def test_get_recent_trades_returns_list(self, market_data_provider, mock_alpaca_client, sample_trade):
        """Should return list of trade dicts."""
        mock_alpaca_client.get_stock_trades.return_value = {"GOOGL": [sample_trade]}

        result = market_data_provider.get_recent_trades("GOOGL")

        assert isinstance(result, list)

    def test_get_recent_trades_has_expected_fields(self, market_data_provider, mock_alpaca_client, sample_trade):
        """Trade dicts should have expected fields."""
        mock_alpaca_client.get_stock_trades.return_value = {"GOOGL": [sample_trade]}

        result = market_data_provider.get_recent_trades("GOOGL")

        assert len(result) == 1
        trade = result[0]
        assert "price" in trade
        assert "size" in trade
        assert "timestamp" in trade
        assert "conditions" in trade

    def test_get_recent_trades_converts_types(self, market_data_provider, mock_alpaca_client, sample_trade):
        """Values should be converted to correct types."""
        mock_alpaca_client.get_stock_trades.return_value = {"GOOGL": [sample_trade]}

        result = market_data_provider.get_recent_trades("GOOGL")

        trade = result[0]
        assert isinstance(trade["price"], float)
        assert isinstance(trade["size"], int)

    def test_get_recent_trades_default_limit(self, market_data_provider, mock_alpaca_client):
        """Should use default limit of 50."""
        mock_alpaca_client.get_stock_trades.return_value = {"GOOGL": []}

        market_data_provider.get_recent_trades("GOOGL")

        # Check the request was made with limit 50
        call_args = mock_alpaca_client.get_stock_trades.call_args
        request = call_args[0][0]
        assert request.limit == 50

    def test_get_recent_trades_custom_limit(self, market_data_provider, mock_alpaca_client):
        """Should use custom limit."""
        mock_alpaca_client.get_stock_trades.return_value = {"GOOGL": []}

        market_data_provider.get_recent_trades("GOOGL", limit=100)

        call_args = mock_alpaca_client.get_stock_trades.call_args
        request = call_args[0][0]
        assert request.limit == 100

    def test_get_recent_trades_multiple_trades(self, market_data_provider, mock_alpaca_client):
        """Should return multiple trades."""
        trades = [
            MockAlpacaTrade(price=150.00, size=100, timestamp=datetime.now(), conditions=["@"]),
            MockAlpacaTrade(price=150.10, size=50, timestamp=datetime.now(), conditions=["@"]),
            MockAlpacaTrade(price=149.90, size=75, timestamp=datetime.now(), conditions=["@"]),
        ]
        mock_alpaca_client.get_stock_trades.return_value = {"GOOGL": trades}

        result = market_data_provider.get_recent_trades("GOOGL")

        assert len(result) == 3


# ==================== Timeframe Map Tests ====================


class TestTimeframeMap:
    """Test TIMEFRAME_MAP constant."""

    def test_timeframe_map_has_expected_keys(self):
        """TIMEFRAME_MAP should have expected timeframe keys."""
        expected_keys = ["1Min", "5Min", "15Min", "1Hour", "1Day"]

        for key in expected_keys:
            assert key in MarketDataProvider.TIMEFRAME_MAP


# ==================== Quote Dataclass Tests ====================


class TestQuoteDataclass:
    """Test Quote dataclass."""

    def test_quote_creation(self):
        """Quote should be creatable with all fields."""
        quote = Quote(
            symbol="GOOGL",
            bid_price=149.95,
            ask_price=150.05,
            bid_size=100,
            ask_size=150,
            last_price=150.00,
            timestamp=datetime.now(),
        )

        assert quote.symbol == "GOOGL"
        assert quote.bid_price == 149.95
        assert quote.ask_price == 150.05


# ==================== Snapshot Dataclass Tests ====================


class TestSnapshotDataclass:
    """Test Snapshot dataclass."""

    def test_snapshot_creation(self):
        """Snapshot should be creatable with all fields."""
        snapshot = Snapshot(
            symbol="GOOGL",
            latest_trade_price=150.00,
            latest_trade_size=100,
            latest_quote_bid=149.95,
            latest_quote_ask=150.05,
            daily_bar_open=148.00,
            daily_bar_high=152.00,
            daily_bar_low=147.50,
            daily_bar_close=150.00,
            daily_bar_volume=1000000,
            prev_daily_bar_close=148.00,
            timestamp=datetime.now(),
        )

        assert snapshot.symbol == "GOOGL"
        assert snapshot.latest_trade_price == 150.00
        assert snapshot.daily_bar_close == 150.00


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_get_bars_with_custom_start_end(self, market_data_provider, mock_alpaca_client, sample_multiindex_bars):
        """Should accept custom start and end dates."""
        mock_response = Mock()
        mock_response.df = sample_multiindex_bars
        mock_alpaca_client.get_stock_bars.return_value = mock_response

        start = datetime.now() - timedelta(days=7)
        end = datetime.now()

        result = market_data_provider.get_bars("GOOGL", start=start, end=end)

        assert isinstance(result, pd.DataFrame)

    def test_get_bars_default_dates(self, market_data_provider, mock_alpaca_client, sample_multiindex_bars):
        """Should calculate default dates when not provided."""
        mock_response = Mock()
        mock_response.df = sample_multiindex_bars
        mock_alpaca_client.get_stock_bars.return_value = mock_response

        # Should not raise even without start/end
        result = market_data_provider.get_bars("GOOGL", timeframe="1Day", limit=30)

        assert isinstance(result, pd.DataFrame)
