"""
Pytest configuration and shared fixtures for AI Trading Competition tests.

This module provides:
- Mock Alpaca TradingClient and StockHistoricalDataClient
- Mock Anthropic client for Claude agent testing
- Sample MarketContext objects for various market conditions
- Sample TradingDecision objects
- Portfolio state fixtures at various equity levels
- Mock database connection

Usage:
    def test_something(mock_trading_client, sample_market_context_bullish):
        # Use fixtures in your tests
        ...
"""
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any, Optional
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import pytz

# Import trading bot modules
from agents.base_agent import (
    BaseTradingAgent,
    TradingDecision,
    MarketContext,
    ActionType,
    StrategyType,
    AgentState,
)
from config.settings import RISK_LIMITS, STARTING_CAPITAL, SYMBOLS
from risk.risk_manager import RiskManager, RiskValidationResult
from execution.order_executor import OrderExecutor, ExecutionResult
from monitoring.scoreboard import Scoreboard, AgentScore
from data.market_data import Quote, Snapshot


# Timezone constant
ET = pytz.timezone("America/New_York")


# =============================================================================
# MOCK DATA CLASSES
# =============================================================================


@dataclass
class MockAccount:
    """Mock Alpaca account."""
    equity: str = "100000.00"
    cash: str = "100000.00"
    buying_power: str = "200000.00"
    portfolio_value: str = "100000.00"
    last_equity: str = "100000.00"
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    account_blocked: bool = False


@dataclass
class MockPosition:
    """Mock Alpaca position."""
    symbol: str = "GOOGL"
    qty: str = "10"
    avg_entry_price: str = "150.00"
    current_price: str = "155.00"
    market_value: str = "1550.00"
    unrealized_pl: str = "50.00"
    unrealized_plpc: str = "0.0333"
    change_today: str = "0.01"


@dataclass
class MockOrder:
    """Mock Alpaca order."""
    id: str = "order-123"
    symbol: str = "GOOGL"
    qty: str = "10"
    filled_qty: str = "10"
    side: MagicMock = field(default_factory=lambda: MagicMock(value="buy"))
    type: MagicMock = field(default_factory=lambda: MagicMock(value="market"))
    status: MagicMock = field(default_factory=lambda: MagicMock(value="filled"))
    limit_price: Optional[str] = None
    stop_price: Optional[str] = None
    filled_avg_price: str = "150.00"
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = field(default_factory=datetime.now)


@dataclass
class MockQuote:
    """Mock stock quote."""
    symbol: str = "GOOGL"
    last_price: float = 150.00
    bid_price: float = 149.95
    ask_price: float = 150.05
    bid_size: int = 100
    ask_size: int = 100
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MockSnapshot:
    """Mock market snapshot."""
    symbol: str = "GOOGL"
    latest_trade_price: float = 150.00
    latest_trade_size: int = 100
    latest_quote_bid: float = 149.95
    latest_quote_ask: float = 150.05
    daily_bar_open: float = 148.00
    daily_bar_high: float = 152.00
    daily_bar_low: float = 147.50
    daily_bar_close: float = 150.00
    daily_bar_volume: int = 1000000
    prev_daily_bar_close: float = 148.00
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# TIMEZONE AND TIME FIXTURES
# =============================================================================


@pytest.fixture
def market_time_open():
    """Return a datetime during market hours (10:30 AM ET)."""
    now = datetime.now(ET)
    return now.replace(hour=10, minute=30, second=0, microsecond=0)


@pytest.fixture
def market_time_preopen():
    """Return a datetime before market buffer period (9:35 AM ET)."""
    now = datetime.now(ET)
    return now.replace(hour=9, minute=35, second=0, microsecond=0)


@pytest.fixture
def market_time_close():
    """Return a datetime during end-of-day buffer (3:50 PM ET)."""
    now = datetime.now(ET)
    return now.replace(hour=15, minute=50, second=0, microsecond=0)


@pytest.fixture
def market_time_closed():
    """Return a datetime when market is closed (6:00 PM ET)."""
    now = datetime.now(ET)
    return now.replace(hour=18, minute=0, second=0, microsecond=0)


# =============================================================================
# MOCK TRADING CLIENT FIXTURES
# =============================================================================


@pytest.fixture
def mock_trading_client():
    """Create a mock Alpaca trading client."""
    client = MagicMock()

    # Default account with $100k
    client.get_account.return_value = MockAccount()

    # No positions by default
    client.get_all_positions.return_value = []

    # No orders by default
    client.get_orders.return_value = []

    # Order submission returns mock order
    client.submit_order.return_value = MockOrder()

    # Close position returns mock order
    client.close_position.return_value = MockOrder()

    # Cancel orders succeeds
    client.cancel_orders.return_value = None

    return client


@pytest.fixture
def mock_account_with_equity():
    """Factory fixture for creating accounts with specific equity."""
    def _create(equity: float, cash: float | None = None, last_equity: float | None = None):
        return MockAccount(
            equity=str(equity),
            cash=str(cash if cash is not None else equity),
            buying_power=str(equity * 2),
            portfolio_value=str(equity),
            last_equity=str(last_equity if last_equity is not None else equity),
        )
    return _create


@pytest.fixture
def mock_account_high_equity():
    """Mock account with high equity ($115,000 - 15% profit)."""
    return MockAccount(
        equity="115000.00",
        cash="50000.00",
        buying_power="100000.00",
        portfolio_value="115000.00",
        last_equity="112000.00",
    )


@pytest.fixture
def mock_account_low_equity():
    """Mock account with low equity ($95,500 - near daily limit)."""
    return MockAccount(
        equity="95500.00",
        cash="30000.00",
        buying_power="60000.00",
        portfolio_value="95500.00",
        last_equity="100000.00",
    )


@pytest.fixture
def mock_account_at_daily_limit():
    """Mock account that has hit the 5% daily loss limit."""
    return MockAccount(
        equity="94000.00",  # -6% from last_equity
        cash="20000.00",
        buying_power="40000.00",
        portfolio_value="94000.00",
        last_equity="100000.00",
    )


@pytest.fixture
def mock_account_starting():
    """Mock account at starting capital."""
    return MockAccount(
        equity=str(STARTING_CAPITAL),
        cash=str(STARTING_CAPITAL),
        buying_power=str(STARTING_CAPITAL * 2),
        portfolio_value=str(STARTING_CAPITAL),
        last_equity=str(STARTING_CAPITAL),
    )


# =============================================================================
# MOCK POSITION FIXTURES
# =============================================================================


@pytest.fixture
def mock_position_factory():
    """Factory fixture for creating positions."""
    def _create(
        symbol: str = "GOOGL",
        qty: int = 10,
        avg_entry: float = 150.00,
        current_price: float = 155.00
    ):
        market_value = qty * current_price
        unrealized_pl = qty * (current_price - avg_entry)
        unrealized_plpc = (current_price - avg_entry) / avg_entry

        return MockPosition(
            symbol=symbol,
            qty=str(qty),
            avg_entry_price=str(avg_entry),
            current_price=str(current_price),
            market_value=str(market_value),
            unrealized_pl=str(unrealized_pl),
            unrealized_plpc=str(unrealized_plpc),
        )
    return _create


@pytest.fixture
def mock_position_googl_profit():
    """Mock GOOGL position with unrealized profit."""
    return MockPosition(
        symbol="GOOGL",
        qty="50",
        avg_entry_price="180.00",
        current_price="185.50",
        market_value="9275.00",
        unrealized_pl="275.00",
        unrealized_plpc="0.0306",
    )


@pytest.fixture
def mock_position_googl_loss():
    """Mock GOOGL position with unrealized loss."""
    return MockPosition(
        symbol="GOOGL",
        qty="50",
        avg_entry_price="180.00",
        current_price="175.00",
        market_value="8750.00",
        unrealized_pl="-250.00",
        unrealized_plpc="-0.0278",
    )


@pytest.fixture
def mock_position_tsla_profit():
    """Mock TSLA position with unrealized profit."""
    return MockPosition(
        symbol="TSLA",
        qty="30",
        avg_entry_price="250.00",
        current_price="265.00",
        market_value="7950.00",
        unrealized_pl="450.00",
        unrealized_plpc="0.06",
    )


@pytest.fixture
def mock_position_tsla_loss():
    """Mock TSLA position with unrealized loss."""
    return MockPosition(
        symbol="TSLA",
        qty="30",
        avg_entry_price="250.00",
        current_price="240.00",
        market_value="7200.00",
        unrealized_pl="-300.00",
        unrealized_plpc="-0.04",
    )


# =============================================================================
# TRADING CLIENT WITH POSITIONS
# =============================================================================


@pytest.fixture
def mock_trading_client_with_positions(
    mock_trading_client,
    mock_account_high_equity,
    mock_position_googl_profit,
):
    """Trading client with one profitable GOOGL position."""
    mock_trading_client.get_account.return_value = mock_account_high_equity
    mock_trading_client.get_all_positions.return_value = [mock_position_googl_profit]
    return mock_trading_client


@pytest.fixture
def mock_trading_client_max_positions(
    mock_trading_client,
    mock_account_high_equity,
    mock_position_googl_profit,
    mock_position_tsla_profit,
):
    """Trading client at max positions (2 positions)."""
    mock_trading_client.get_account.return_value = mock_account_high_equity
    mock_trading_client.get_all_positions.return_value = [
        mock_position_googl_profit,
        mock_position_tsla_profit,
    ]
    return mock_trading_client


@pytest.fixture
def mock_trading_client_at_limit(
    mock_trading_client,
    mock_account_at_daily_limit,
):
    """Trading client that has hit daily loss limit."""
    mock_trading_client.get_account.return_value = mock_account_at_daily_limit
    return mock_trading_client


# =============================================================================
# MOCK DATA CLIENT FIXTURES
# =============================================================================


@pytest.fixture
def mock_data_client():
    """Create a fully mocked Alpaca StockHistoricalDataClient."""
    client = MagicMock()

    # Mock snapshot response
    client.get_stock_snapshot.return_value = {}

    # Mock bars response
    client.get_stock_bars.return_value = MagicMock(df=MagicMock())

    # Mock quote response
    client.get_stock_latest_quote.return_value = {}

    # Mock trades response
    client.get_stock_trades.return_value = {}

    return client


@pytest.fixture
def mock_snapshot_googl_bullish():
    """Mock GOOGL snapshot in bullish condition."""
    return Snapshot(
        symbol="GOOGL",
        latest_trade_price=185.50,
        latest_trade_size=100,
        latest_quote_bid=185.45,
        latest_quote_ask=185.55,
        daily_bar_open=180.00,
        daily_bar_high=186.00,
        daily_bar_low=179.50,
        daily_bar_close=185.50,
        daily_bar_volume=15000000,
        prev_daily_bar_close=179.00,
        timestamp=datetime.now(ET),
    )


@pytest.fixture
def mock_snapshot_googl_bearish():
    """Mock GOOGL snapshot in bearish condition."""
    return Snapshot(
        symbol="GOOGL",
        latest_trade_price=175.00,
        latest_trade_size=100,
        latest_quote_bid=174.95,
        latest_quote_ask=175.05,
        daily_bar_open=180.00,
        daily_bar_high=180.50,
        daily_bar_low=174.00,
        daily_bar_close=175.00,
        daily_bar_volume=20000000,
        prev_daily_bar_close=181.00,
        timestamp=datetime.now(ET),
    )


@pytest.fixture
def mock_snapshot_tsla_bullish():
    """Mock TSLA snapshot in bullish condition."""
    return Snapshot(
        symbol="TSLA",
        latest_trade_price=265.00,
        latest_trade_size=200,
        latest_quote_bid=264.90,
        latest_quote_ask=265.10,
        daily_bar_open=255.00,
        daily_bar_high=267.00,
        daily_bar_low=254.00,
        daily_bar_close=265.00,
        daily_bar_volume=50000000,
        prev_daily_bar_close=253.00,
        timestamp=datetime.now(ET),
    )


@pytest.fixture
def mock_snapshot_tsla_bearish():
    """Mock TSLA snapshot in bearish condition."""
    return Snapshot(
        symbol="TSLA",
        latest_trade_price=240.00,
        latest_trade_size=200,
        latest_quote_bid=239.90,
        latest_quote_ask=240.10,
        daily_bar_open=255.00,
        daily_bar_high=256.00,
        daily_bar_low=238.00,
        daily_bar_close=240.00,
        daily_bar_volume=60000000,
        prev_daily_bar_close=257.00,
        timestamp=datetime.now(ET),
    )


# =============================================================================
# MOCK ANTHROPIC CLIENT FIXTURES
# =============================================================================


def create_anthropic_response(text_content: str, stop_reason: str = "end_turn"):
    """Helper to create mock Anthropic response."""
    response = MagicMock()
    response.stop_reason = stop_reason

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text_content

    response.content = [text_block]
    return response


@pytest.fixture
def mock_anthropic_client():
    """Create a mocked Anthropic client for Claude agent testing."""
    client = MagicMock()

    # Default response simulating a HOLD decision
    client.messages.create.return_value = create_anthropic_response(
        "STRATEGY: Defensive\n"
        "ACTION: HOLD\n"
        "REASONING: Market conditions are uncertain."
    )

    # Attach helper for tests to easily change response
    client._create_response = create_anthropic_response

    return client


@pytest.fixture
def mock_anthropic_client_buy():
    """Anthropic client that returns a BUY decision."""
    client = MagicMock()
    client.messages.create.return_value = create_anthropic_response(
        "STRATEGY: Momentum\n"
        "ACTION: BUY\n"
        "SYMBOL: GOOGL\n"
        "QUANTITY: 25 shares\n"
        "STOP LOSS: $180.00\n"
        "TAKE PROFIT: $195.00\n"
        "REASONING: Strong upward momentum with RSI above 60."
    )
    return client


@pytest.fixture
def mock_anthropic_client_sell():
    """Anthropic client that returns a SELL decision."""
    client = MagicMock()
    client.messages.create.return_value = create_anthropic_response(
        "STRATEGY: Mean Reversion\n"
        "ACTION: SELL\n"
        "SYMBOL: TSLA\n"
        "QUANTITY: 20 shares\n"
        "STOP LOSS: $270.00\n"
        "REASONING: RSI overbought, expecting reversion to mean."
    )
    return client


@pytest.fixture
def mock_anthropic_client_with_tool_use():
    """Anthropic client that uses tools before making decision."""
    client = MagicMock()

    # First response requests tool use
    tool_use_response = MagicMock()
    tool_use_response.stop_reason = "tool_use"

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "get_stock_price"
    tool_block.input = {"symbol": "GOOGL"}
    tool_block.id = "tool_call_123"

    tool_use_response.content = [tool_block]

    # Second response is final decision
    final_response = create_anthropic_response(
        "STRATEGY: Momentum\n"
        "ACTION: BUY\n"
        "SYMBOL: GOOGL\n"
        "QUANTITY: 20 shares\n"
        "STOP LOSS: $178.00\n"
        "REASONING: After checking price, confirmed strong momentum."
    )

    client.messages.create.side_effect = [tool_use_response, final_response]
    return client


# =============================================================================
# MARKET CONTEXT FIXTURES
# =============================================================================


def create_symbol_data(
    price: float,
    daily_change_percent: float,
    rsi: float,
    macd_histogram: float,
    bollinger_percent_b: float,
    above_vwap: bool,
    trend: str,
) -> dict:
    """Helper to create symbol data dictionary."""
    return {
        "price": price,
        "bid": price - 0.05,
        "ask": price + 0.05,
        "daily_open": price * (1 - daily_change_percent / 200),
        "daily_high": price * 1.01,
        "daily_low": price * 0.99,
        "daily_close": price,
        "daily_volume": 10000000,
        "daily_change_percent": daily_change_percent,
        "prev_close": price * (1 - daily_change_percent / 100),
        "rsi": rsi,
        "macd_histogram": macd_histogram,
        "bollinger_percent_b": bollinger_percent_b,
        "atr": price * 0.02,
        "vwap": price * (0.99 if above_vwap else 1.01),
        "above_vwap": above_vwap,
        "ema_9": price * (1.01 if trend == "bullish" else 0.99),
        "ema_21": price * (0.99 if trend == "bullish" else 1.01),
        "trend": trend,
    }


def create_account_data(
    equity: float,
    cash: float,
    daily_pnl: float,
) -> dict:
    """Helper to create account data dictionary."""
    return {
        "equity": equity,
        "cash": cash,
        "buying_power": cash * 2,
        "portfolio_value": equity,
        "daily_pnl": daily_pnl,
        "daily_pnl_percent": (daily_pnl / (equity - daily_pnl)) * 100 if (equity - daily_pnl) > 0 else 0,
    }


@pytest.fixture
def sample_market_context():
    """Create a sample MarketContext for testing."""
    return MarketContext(
        timestamp=datetime.now(ET),
        symbols={
            "GOOGL": {
                "price": 150.00,
                "daily_change_percent": 1.5,
                "rsi": 55.0,
                "macd_histogram": 0.5,
                "above_vwap": True,
                "trend": "bullish",
            },
            "TSLA": {
                "price": 250.00,
                "daily_change_percent": -0.5,
                "rsi": 45.0,
                "macd_histogram": -0.3,
                "above_vwap": False,
                "trend": "bearish",
            },
        },
        account={
            "equity": 100000.00,
            "cash": 100000.00,
            "buying_power": 200000.00,
            "daily_pnl": 0.0,
            "daily_pnl_percent": 0.0,
        },
        positions=[],
        recent_trades=[],
        market_condition="bullish - moderate momentum",
    )


@pytest.fixture
def sample_market_context_bullish(market_time_open) -> MarketContext:
    """Market context with both stocks in bullish condition."""
    return MarketContext(
        timestamp=market_time_open,
        symbols={
            "GOOGL": create_symbol_data(
                price=185.50,
                daily_change_percent=2.5,
                rsi=62,
                macd_histogram=0.5,
                bollinger_percent_b=0.75,
                above_vwap=True,
                trend="bullish",
            ),
            "TSLA": create_symbol_data(
                price=265.00,
                daily_change_percent=3.2,
                rsi=65,
                macd_histogram=0.8,
                bollinger_percent_b=0.80,
                above_vwap=True,
                trend="bullish",
            ),
        },
        account=create_account_data(
            equity=105000.00,
            cash=80000.00,
            daily_pnl=500.00,
        ),
        positions=[],
        recent_trades=[],
        market_condition="bullish - both stocks trending up",
    )


@pytest.fixture
def sample_market_context_bearish(market_time_open) -> MarketContext:
    """Market context with both stocks in bearish condition."""
    return MarketContext(
        timestamp=market_time_open,
        symbols={
            "GOOGL": create_symbol_data(
                price=175.00,
                daily_change_percent=-3.0,
                rsi=35,
                macd_histogram=-0.6,
                bollinger_percent_b=0.20,
                above_vwap=False,
                trend="bearish",
            ),
            "TSLA": create_symbol_data(
                price=240.00,
                daily_change_percent=-4.5,
                rsi=30,
                macd_histogram=-1.2,
                bollinger_percent_b=0.15,
                above_vwap=False,
                trend="bearish",
            ),
        },
        account=create_account_data(
            equity=98000.00,
            cash=75000.00,
            daily_pnl=-1500.00,
        ),
        positions=[],
        recent_trades=[],
        market_condition="bearish - both stocks trending down",
    )


@pytest.fixture
def sample_market_context_sideways(market_time_open) -> MarketContext:
    """Market context with mixed/sideways conditions."""
    return MarketContext(
        timestamp=market_time_open,
        symbols={
            "GOOGL": create_symbol_data(
                price=180.00,
                daily_change_percent=0.3,
                rsi=50,
                macd_histogram=0.05,
                bollinger_percent_b=0.50,
                above_vwap=True,
                trend="bullish",
            ),
            "TSLA": create_symbol_data(
                price=252.00,
                daily_change_percent=-0.5,
                rsi=48,
                macd_histogram=-0.1,
                bollinger_percent_b=0.45,
                above_vwap=False,
                trend="bearish",
            ),
        },
        account=create_account_data(
            equity=100500.00,
            cash=90000.00,
            daily_pnl=100.00,
        ),
        positions=[],
        recent_trades=[],
        market_condition="mixed - stocks showing different trends",
    )


@pytest.fixture
def sample_market_context_with_positions(market_time_open) -> MarketContext:
    """Market context with existing positions (1 position)."""
    return MarketContext(
        timestamp=market_time_open,
        symbols={
            "GOOGL": create_symbol_data(
                price=185.50,
                daily_change_percent=2.5,
                rsi=62,
                macd_histogram=0.5,
                bollinger_percent_b=0.75,
                above_vwap=True,
                trend="bullish",
            ),
            "TSLA": create_symbol_data(
                price=265.00,
                daily_change_percent=3.2,
                rsi=65,
                macd_histogram=0.8,
                bollinger_percent_b=0.80,
                above_vwap=True,
                trend="bullish",
            ),
        },
        account=create_account_data(
            equity=108000.00,
            cash=50000.00,
            daily_pnl=800.00,
        ),
        positions=[
            {
                "symbol": "GOOGL",
                "quantity": 50,
                "avg_entry_price": 180.00,
                "current_price": 185.50,
                "market_value": 9275.00,
                "unrealized_pnl": 275.00,
                "unrealized_pnl_percent": 3.06,
            }
        ],
        recent_trades=[
            {
                "symbol": "GOOGL",
                "side": "buy",
                "quantity": 50,
                "filled_avg_price": 180.00,
                "status": "filled",
                "filled_at": (market_time_open - timedelta(hours=2)).isoformat(),
            }
        ],
        market_condition="bullish - both stocks trending up",
    )


@pytest.fixture
def sample_market_context_max_positions(market_time_open) -> MarketContext:
    """Market context at max positions (2 positions)."""
    return MarketContext(
        timestamp=market_time_open,
        symbols={
            "GOOGL": create_symbol_data(
                price=185.50,
                daily_change_percent=2.5,
                rsi=62,
                macd_histogram=0.5,
                bollinger_percent_b=0.75,
                above_vwap=True,
                trend="bullish",
            ),
            "TSLA": create_symbol_data(
                price=265.00,
                daily_change_percent=3.2,
                rsi=65,
                macd_histogram=0.8,
                bollinger_percent_b=0.80,
                above_vwap=True,
                trend="bullish",
            ),
        },
        account=create_account_data(
            equity=112000.00,
            cash=30000.00,
            daily_pnl=1200.00,
        ),
        positions=[
            {
                "symbol": "GOOGL",
                "quantity": 50,
                "avg_entry_price": 180.00,
                "current_price": 185.50,
                "market_value": 9275.00,
                "unrealized_pnl": 275.00,
                "unrealized_pnl_percent": 3.06,
            },
            {
                "symbol": "TSLA",
                "quantity": 30,
                "avg_entry_price": 250.00,
                "current_price": 265.00,
                "market_value": 7950.00,
                "unrealized_pnl": 450.00,
                "unrealized_pnl_percent": 6.0,
            },
        ],
        recent_trades=[],
        market_condition="bullish - both stocks trending up",
    )


@pytest.fixture
def sample_market_context_near_daily_limit(market_time_open) -> MarketContext:
    """Market context near the 5% daily loss limit."""
    return MarketContext(
        timestamp=market_time_open,
        symbols={
            "GOOGL": create_symbol_data(
                price=175.00,
                daily_change_percent=-3.0,
                rsi=35,
                macd_histogram=-0.6,
                bollinger_percent_b=0.20,
                above_vwap=False,
                trend="bearish",
            ),
            "TSLA": create_symbol_data(
                price=240.00,
                daily_change_percent=-4.5,
                rsi=30,
                macd_histogram=-1.2,
                bollinger_percent_b=0.15,
                above_vwap=False,
                trend="bearish",
            ),
        },
        account=create_account_data(
            equity=95500.00,  # -4.5% from 100K
            cash=95500.00,
            daily_pnl=-4500.00,
        ),
        positions=[],
        recent_trades=[],
        market_condition="bearish - both stocks trending down",
    )


@pytest.fixture
def create_market_context():
    """Factory fixture to create custom market contexts."""
    def _create(
        timestamp: Optional[datetime] = None,
        googl_trend: str = "bullish",
        tsla_trend: str = "bullish",
        equity: float = STARTING_CAPITAL,
        positions: Optional[list] = None,
    ) -> MarketContext:
        if timestamp is None:
            timestamp = datetime.now(ET).replace(hour=10, minute=30)

        googl_price = 185.50 if googl_trend == "bullish" else 175.00
        tsla_price = 265.00 if tsla_trend == "bullish" else 240.00

        market_condition = "mixed - stocks showing different trends"
        if googl_trend == tsla_trend == "bullish":
            market_condition = "bullish - both stocks trending up"
        elif googl_trend == tsla_trend == "bearish":
            market_condition = "bearish - both stocks trending down"

        return MarketContext(
            timestamp=timestamp,
            symbols={
                "GOOGL": create_symbol_data(
                    price=googl_price,
                    daily_change_percent=2.5 if googl_trend == "bullish" else -2.5,
                    rsi=62 if googl_trend == "bullish" else 38,
                    macd_histogram=0.5 if googl_trend == "bullish" else -0.5,
                    bollinger_percent_b=0.75 if googl_trend == "bullish" else 0.25,
                    above_vwap=googl_trend == "bullish",
                    trend=googl_trend,
                ),
                "TSLA": create_symbol_data(
                    price=tsla_price,
                    daily_change_percent=3.2 if tsla_trend == "bullish" else -3.2,
                    rsi=65 if tsla_trend == "bullish" else 35,
                    macd_histogram=0.8 if tsla_trend == "bullish" else -0.8,
                    bollinger_percent_b=0.80 if tsla_trend == "bullish" else 0.20,
                    above_vwap=tsla_trend == "bullish",
                    trend=tsla_trend,
                ),
            },
            account=create_account_data(
                equity=equity,
                cash=equity * 0.8,
                daily_pnl=equity - STARTING_CAPITAL,
            ),
            positions=positions or [],
            recent_trades=[],
            market_condition=market_condition,
        )

    return _create


# =============================================================================
# TRADING DECISION FIXTURES
# =============================================================================


@pytest.fixture
def sample_buy_decision():
    """Create a sample BUY decision for testing."""
    return TradingDecision(
        timestamp=datetime.now(ET),
        action=ActionType.BUY,
        symbol="GOOGL",
        quantity=10,
        order_type="market",
        stop_loss=145.00,  # ~3.3% below entry
        take_profit=160.00,
        strategy_used=StrategyType.MOMENTUM,
        reasoning="Strong upward momentum with RSI confirmation",
        confidence=0.75,
    )


@pytest.fixture
def sample_sell_decision():
    """Create a sample SELL decision for testing."""
    return TradingDecision(
        timestamp=datetime.now(ET),
        action=ActionType.SELL,
        symbol="TSLA",
        quantity=5,
        order_type="market",
        stop_loss=260.00,  # ~4% above entry
        take_profit=235.00,
        strategy_used=StrategyType.MEAN_REVERSION,
        reasoning="RSI overbought, expecting pullback",
        confidence=0.65,
    )


@pytest.fixture
def sample_hold_decision():
    """Create a sample HOLD decision for testing."""
    return TradingDecision(
        timestamp=datetime.now(ET),
        action=ActionType.HOLD,
        strategy_used=StrategyType.DEFENSIVE,
        reasoning="No clear setup, maintaining cash position",
        confidence=0.50,
    )


@pytest.fixture
def sample_close_decision():
    """Create a sample CLOSE decision for testing."""
    return TradingDecision(
        timestamp=datetime.now(ET),
        action=ActionType.CLOSE,
        symbol="GOOGL",
        strategy_used=StrategyType.DEFENSIVE,
        reasoning="Taking profits, target reached",
        confidence=0.80,
    )


@pytest.fixture
def sample_decision_invalid_no_stop(market_time_open) -> TradingDecision:
    """Sample invalid decision - missing stop loss."""
    return TradingDecision(
        timestamp=market_time_open,
        action=ActionType.BUY,
        symbol="GOOGL",
        quantity=25,
        stop_loss=None,  # Invalid - stop loss required
        strategy_used=StrategyType.MOMENTUM,
        reasoning="Test decision without stop loss.",
        confidence=0.70,
    )


@pytest.fixture
def sample_decision_invalid_symbol(market_time_open) -> TradingDecision:
    """Sample invalid decision - wrong symbol."""
    return TradingDecision(
        timestamp=market_time_open,
        action=ActionType.BUY,
        symbol="AAPL",  # Invalid - only GOOGL and TSLA allowed
        quantity=25,
        stop_loss=150.00,
        strategy_used=StrategyType.MOMENTUM,
        reasoning="Test decision with invalid symbol.",
        confidence=0.70,
    )


@pytest.fixture
def create_trading_decision():
    """Factory fixture to create custom trading decisions."""
    def _create(
        action: ActionType = ActionType.HOLD,
        symbol: Optional[str] = None,
        quantity: Optional[int] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy: StrategyType = StrategyType.DEFENSIVE,
        confidence: float = 0.5,
        timestamp: Optional[datetime] = None,
    ) -> TradingDecision:
        if timestamp is None:
            timestamp = datetime.now(ET)

        return TradingDecision(
            timestamp=timestamp,
            action=action,
            symbol=symbol,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_used=strategy,
            reasoning=f"Test decision: {action.value} {symbol or 'no symbol'}",
            confidence=confidence,
        )

    return _create


# =============================================================================
# RISK MANAGER AND ORDER EXECUTOR FIXTURES
# =============================================================================


@pytest.fixture
def risk_manager(mock_trading_client):
    """Create a RiskManager with mock client."""
    return RiskManager(mock_trading_client, "test_agent")


@pytest.fixture
def order_executor(mock_trading_client, risk_manager):
    """Create an OrderExecutor with mock client and risk manager."""
    return OrderExecutor(mock_trading_client, risk_manager, "test_agent")


@pytest.fixture
def risk_validation_passed() -> RiskValidationResult:
    """Successful risk validation result."""
    return RiskValidationResult(
        valid=True,
        message="All risk checks passed",
    )


@pytest.fixture
def risk_validation_failed_no_stop() -> RiskValidationResult:
    """Failed risk validation - missing stop loss."""
    return RiskValidationResult(
        valid=False,
        message="Stop-loss is REQUIRED for all trades",
        violations=["Stop-loss is REQUIRED for all trades"],
    )


@pytest.fixture
def risk_validation_failed_max_positions() -> RiskValidationResult:
    """Failed risk validation - max positions reached."""
    return RiskValidationResult(
        valid=False,
        message="Max positions (2) reached. Close a position first.",
        violations=["Max positions (2) reached. Close a position first."],
    )


# =============================================================================
# SCOREBOARD FIXTURE
# =============================================================================


@pytest.fixture
def scoreboard():
    """Create a fresh scoreboard."""
    board = Scoreboard()
    board.register_agent("claude")
    board.register_agent("grok")
    return board


# =============================================================================
# AGENT STATE FIXTURES
# =============================================================================


@pytest.fixture
def sample_agent_state_new() -> AgentState:
    """Fresh agent state with no history."""
    return AgentState(agent_name="claude")


@pytest.fixture
def sample_agent_state_experienced() -> AgentState:
    """Experienced agent state with trading history."""
    return AgentState(
        agent_name="claude",
        total_trades=25,
        winning_trades=15,
        losing_trades=10,
        strategies_used={
            "momentum": 10,
            "mean_reversion": 8,
            "breakout": 4,
            "defensive": 3,
        },
        last_decision_time=datetime.now(ET),
        consecutive_losses=0,
        peak_equity=115000.00,
        current_drawdown=2.5,
    )


@pytest.fixture
def sample_agent_state_losing_streak() -> AgentState:
    """Agent state on a losing streak."""
    return AgentState(
        agent_name="claude",
        total_trades=15,
        winning_trades=5,
        losing_trades=10,
        strategies_used={
            "momentum": 8,
            "mean_reversion": 4,
            "defensive": 3,
        },
        last_decision_time=datetime.now(ET),
        consecutive_losses=4,
        peak_equity=105000.00,
        current_drawdown=8.5,
    )


# =============================================================================
# MOCK DATABASE FIXTURES
# =============================================================================


@pytest.fixture
def mock_postgres_client():
    """Create a mock PostgresClient for testing without database."""
    mock = MagicMock()
    mock.fetchval = AsyncMock(return_value=1)
    mock.fetchrow = AsyncMock(return_value=None)
    mock.fetch = AsyncMock(return_value=[])
    mock.execute = AsyncMock(return_value="OK")
    return mock


# Try to import learning system types - they may not be available
try:
    from database.learning_store import (
        LearningStore,
        Episode,
        Reflection,
        Learning,
        OutcomeStatus,
    )
    LEARNING_SYSTEM_AVAILABLE = True
except ImportError:
    LEARNING_SYSTEM_AVAILABLE = False
    Episode = None  # type: ignore[misc, assignment]
    Reflection = None  # type: ignore[misc, assignment]
    Learning = None  # type: ignore[misc, assignment]
    OutcomeStatus = None  # type: ignore[misc, assignment]


if LEARNING_SYSTEM_AVAILABLE:
    @pytest.fixture
    def sample_episode():
        """Create a sample Episode for testing."""
        return Episode(
            id=1,
            agent_name="claude",
            timestamp=datetime.now(ET),
            market_regime="bullish",
            symbols_context={
                "GOOGL": {"price": 150.00, "rsi": 55.0},
                "TSLA": {"price": 250.00, "rsi": 45.0},
            },
            account_state={"equity": 100000.00, "cash": 100000.00},
            decision_made={
                "action": "BUY",
                "symbol": "GOOGL",
                "quantity": 10,
                "strategy": "momentum",
                "reasoning": "Strong momentum signal",
            },
            outcome_pnl=None,
            outcome_status=OutcomeStatus.PENDING.value,
        )

    @pytest.fixture
    def sample_episode_with_outcome():
        """Create a sample Episode with completed outcome."""
        return Episode(
            id=1,
            agent_name="claude",
            timestamp=datetime.now(ET),
            market_regime="bullish",
            symbols_context={
                "GOOGL": {"price": 150.00, "rsi": 55.0},
            },
            account_state={"equity": 100000.00},
            decision_made={
                "action": "BUY",
                "symbol": "GOOGL",
                "quantity": 10,
                "strategy": "momentum",
                "reasoning": "Strong momentum signal",
            },
            outcome_pnl=Decimal("150.00"),
            outcome_status=OutcomeStatus.WIN.value,
        )

    @pytest.fixture
    def sample_reflection():
        """Create a sample Reflection for testing."""
        return Reflection(
            id=1,
            episode_id=1,
            agent_name="claude",
            what_worked="RSI divergence correctly predicted reversal",
            what_failed="Entry timing was slightly late",
            lesson_learned="Wait for confirmation candle before entry",
            next_time_will="Use limit order at support level",
            confidence_adjustment=Decimal("0.05"),
            tags=["GOOGL", "momentum", "bullish", "rsi"],
        )

    @pytest.fixture
    def sample_learning():
        """Create a sample Learning for testing."""
        return Learning(
            id=1,
            agent_name="claude",
            category="indicator",
            pattern="RSI divergence in bullish trend",
            insight="RSI below 30 with price making higher lows signals strong buy",
            success_count=5,
            failure_count=1,
            last_validated=datetime.now(ET),
            is_active=True,
            tags=["rsi", "bullish", "divergence"],
        )

    @pytest.fixture
    def sample_learnings_list():
        """Create a list of sample learnings for testing recall."""
        return [
            Learning(
                id=1,
                agent_name="claude",
                category="indicator",
                pattern="RSI oversold bounce",
                insight="RSI below 30 often leads to bounce",
                success_count=8,
                failure_count=2,
                is_active=True,
                tags=["rsi", "bullish", "GOOGL"],
            ),
            Learning(
                id=2,
                agent_name="claude",
                category="strategy",
                pattern="Morning momentum breakout",
                insight="First hour breakouts tend to continue",
                success_count=6,
                failure_count=3,
                is_active=True,
                tags=["momentum", "breakout", "TSLA"],
            ),
            Learning(
                id=3,
                agent_name="claude",
                category="timing",
                pattern="Avoid lunch hour trades",
                insight="11:30-1:30 ET shows reduced momentum",
                success_count=4,
                failure_count=1,
                is_active=True,
                tags=["timing", "range"],
            ),
        ]


# =============================================================================
# IN-MEMORY LEARNING STORE MOCK
# =============================================================================


class InMemoryLearningStore:
    """In-memory implementation of LearningStore for testing."""

    def __init__(self):
        self.episodes: dict = {}
        self.reflections: dict = {}
        self.learnings: dict = {}
        self._next_id = 1

    def reset(self):
        """Reset all data."""
        self.episodes.clear()
        self.reflections.clear()
        self.learnings.clear()
        self._next_id = 1

    async def create_episode(self, episode) -> int:
        episode_id = self._next_id
        self._next_id += 1
        episode.id = episode_id
        episode.created_at = datetime.now(ET)
        self.episodes[episode_id] = episode
        return episode_id

    async def get_episode(self, episode_id: int):
        return self.episodes.get(episode_id)

    async def update_episode_outcome(
        self, episode_id: int, outcome_pnl: Decimal, outcome_status: str
    ) -> None:
        if episode_id in self.episodes:
            self.episodes[episode_id].outcome_pnl = outcome_pnl
            self.episodes[episode_id].outcome_status = outcome_status

    async def get_recent_episodes(self, agent_name: str, limit: int = 20) -> list:
        agent_episodes = [
            e for e in self.episodes.values() if e.agent_name == agent_name
        ]
        agent_episodes.sort(key=lambda x: x.timestamp or datetime.min, reverse=True)
        return agent_episodes[:limit]

    async def create_reflection(self, reflection) -> int:
        reflection_id = self._next_id
        self._next_id += 1
        reflection.id = reflection_id
        reflection.created_at = datetime.now(ET)
        self.reflections[reflection_id] = reflection
        return reflection_id

    async def get_reflection(self, reflection_id: int):
        return self.reflections.get(reflection_id)

    async def create_learning(self, learning) -> int:
        learning_id = self._next_id
        self._next_id += 1
        learning.id = learning_id
        learning.created_at = datetime.now(ET)
        self.learnings[learning_id] = learning
        return learning_id

    async def get_learning(self, learning_id: int):
        return self.learnings.get(learning_id)

    async def get_learnings_by_tags(
        self, agent_name: str, tags: list, limit: int = 10
    ) -> list:
        matching = []
        for learning in self.learnings.values():
            if learning.agent_name == agent_name and learning.is_active:
                if any(tag in learning.tags for tag in tags):
                    matching.append(learning)

        matching.sort(
            key=lambda x: (x.success_count - x.failure_count, x.success_count),
            reverse=True
        )
        return matching[:limit]

    async def get_top_learnings(self, agent_name: str, limit: int = 10) -> list:
        agent_learnings = [
            l for l in self.learnings.values()
            if l.agent_name == agent_name and l.is_active
        ]
        agent_learnings.sort(
            key=lambda x: (x.success_count - x.failure_count, x.success_count),
            reverse=True
        )
        return agent_learnings[:limit]

    async def increment_learning_success(self, learning_id: int) -> None:
        if learning_id in self.learnings:
            self.learnings[learning_id].success_count += 1
            self.learnings[learning_id].last_validated = datetime.now(ET)

    async def increment_learning_failure(self, learning_id: int) -> None:
        if learning_id in self.learnings:
            self.learnings[learning_id].failure_count += 1
            self.learnings[learning_id].last_validated = datetime.now(ET)

    async def find_similar_learning(
        self, agent_name: str, pattern: str, tags: list
    ):
        for learning in self.learnings.values():
            if learning.agent_name == agent_name and learning.is_active:
                if pattern.lower() in learning.pattern.lower():
                    if any(tag in learning.tags for tag in tags):
                        return learning
        return None

    async def deactivate_learning(self, learning_id: int) -> None:
        if learning_id in self.learnings:
            self.learnings[learning_id].is_active = False


@pytest.fixture
def in_memory_store():
    """Create an in-memory learning store for testing."""
    return InMemoryLearningStore()


# =============================================================================
# MOCK API RESPONSE FIXTURES
# =============================================================================


@pytest.fixture
def mock_market_data_responses():
    """Factory to create mock market data API responses."""
    def _create(symbol: str, bullish: bool = True):
        price = 185.50 if symbol == "GOOGL" else 265.00
        if not bullish:
            price *= 0.95  # 5% lower for bearish

        return {
            "symbol": symbol,
            "latestTrade": {
                "p": price,
                "s": 100,
                "t": datetime.now(ET).isoformat(),
            },
            "latestQuote": {
                "bp": price - 0.05,
                "ap": price + 0.05,
                "bs": 100,
                "as": 100,
            },
            "dailyBar": {
                "o": price * 0.99,
                "h": price * 1.01,
                "l": price * 0.98,
                "c": price,
                "v": 10000000,
            },
            "prevDailyBar": {
                "c": price * (0.975 if bullish else 1.025),
            },
        }
    return _create


@pytest.fixture
def mock_order_response():
    """Factory to create mock order submission responses."""
    def _create(
        symbol: str = "GOOGL",
        side: str = "buy",
        qty: int = 10,
        status: str = "filled",
        filled_price: float = 150.00,
    ):
        order = MockOrder(
            id=f"order-{datetime.now().timestamp()}",
            symbol=symbol,
            qty=str(qty),
            filled_qty=str(qty) if status == "filled" else "0",
            filled_avg_price=str(filled_price) if status == "filled" else "0",
        )
        order.side.value = side
        order.status.value = status
        return order
    return _create


# =============================================================================
# ASYNC TEST HELPERS
# =============================================================================


@pytest.fixture
def run_async():
    """Helper to run async functions in sync tests."""
    def _run(coro):
        loop = asyncio.get_event_loop_policy().new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    return _run
