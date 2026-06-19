"""Real-time market data provider using Alpaca API."""
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockTradesRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

from config.settings import SYMBOLS


@dataclass
class Quote:
    """Current quote data."""

    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    last_price: float
    timestamp: datetime


@dataclass
class Snapshot:
    """Full market snapshot for a symbol."""

    symbol: str
    latest_trade_price: float
    latest_trade_size: int
    latest_quote_bid: float
    latest_quote_ask: float
    daily_bar_open: float
    daily_bar_high: float
    daily_bar_low: float
    daily_bar_close: float
    daily_bar_volume: int
    prev_daily_bar_close: float
    timestamp: datetime


class MarketDataProvider:
    """Provides market data to both AI agents."""

    TIMEFRAME_MAP = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }

    def __init__(self, data_client: StockHistoricalDataClient):
        self.client = data_client

    def get_current_quote(self, symbol: str) -> Quote:
        """Get current bid/ask/last for a symbol."""
        if symbol not in SYMBOLS:
            raise ValueError(f"Symbol {symbol} not allowed. Only {SYMBOLS} permitted.")

        request = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
        quotes = self.client.get_stock_latest_quote(request)
        quote_data = quotes[symbol]

        # Get latest trade for last price
        trade_request = StockTradesRequest(
            symbol_or_symbols=symbol,
            limit=1,
            feed=DataFeed.IEX,
        )
        trades = self.client.get_stock_trades(trade_request)
        last_price = trades[symbol][0].price if trades[symbol] else quote_data.ask_price

        return Quote(
            symbol=symbol,
            bid_price=float(quote_data.bid_price),
            ask_price=float(quote_data.ask_price),
            bid_size=int(quote_data.bid_size),
            ask_size=int(quote_data.ask_size),
            last_price=float(last_price),
            timestamp=quote_data.timestamp,
        )

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Hour",
        limit: int = 100,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Get OHLCV bars for a symbol."""
        if symbol not in SYMBOLS:
            raise ValueError(f"Symbol {symbol} not allowed. Only {SYMBOLS} permitted.")

        tf = self.TIMEFRAME_MAP.get(timeframe)
        if not tf:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        # Default to last N bars if no start/end specified
        if not end:
            end = datetime.now()
        if not start:
            # Estimate start based on timeframe and limit
            if timeframe == "1Day":
                start = end - timedelta(days=limit * 2)  # Account for weekends
            elif timeframe == "1Hour":
                start = end - timedelta(hours=limit * 2)
            else:
                start = end - timedelta(minutes=limit * int(timeframe.replace("Min", "")) * 2)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit,
            feed=DataFeed.IEX,
        )

        bars = self.client.get_stock_bars(request)
        # Access the DataFrame from the response. The SDK returns a BarSet (which
        # exposes .df) on success, but may return a non-BarSet value (e.g. a str)
        # on error. Duck-type on .df rather than isinstance(BarSet) so the parser
        # is exercised under test and any error shape degrades to an empty frame.
        if not hasattr(bars, "df"):
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = bars.df

        # Handle empty DataFrame
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Handle MultiIndex: check if 'symbol' is a valid level name
        if isinstance(df.index, pd.MultiIndex):
            level_names = [name for name in df.index.names if name is not None]
            if "symbol" in level_names:
                df = df.loc[symbol].reset_index()
            else:
                # MultiIndex without named 'symbol' level - just reset
                df = df.reset_index(drop=False)
                # Remove symbol column if it exists from reset
                if "level_0" in df.columns:
                    df = df.drop(columns=["level_0"])
        else:
            df = df.reset_index()

        # Ensure standard column names
        df.columns = [c.lower() for c in df.columns]
        return df.tail(limit)

    def get_snapshot(self, symbol: str) -> Snapshot:
        """Get full market snapshot for a symbol."""
        if symbol not in SYMBOLS:
            raise ValueError(f"Symbol {symbol} not allowed. Only {SYMBOLS} permitted.")

        request = StockSnapshotRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
        snapshots = self.client.get_stock_snapshot(request)
        snap = snapshots[symbol]

        return Snapshot(
            symbol=symbol,
            latest_trade_price=float(snap.latest_trade.price) if snap.latest_trade else 0.0,
            latest_trade_size=int(snap.latest_trade.size) if snap.latest_trade else 0,
            latest_quote_bid=float(snap.latest_quote.bid_price) if snap.latest_quote else 0.0,
            latest_quote_ask=float(snap.latest_quote.ask_price) if snap.latest_quote else 0.0,
            daily_bar_open=float(snap.daily_bar.open) if snap.daily_bar else 0.0,
            daily_bar_high=float(snap.daily_bar.high) if snap.daily_bar else 0.0,
            daily_bar_low=float(snap.daily_bar.low) if snap.daily_bar else 0.0,
            daily_bar_close=float(snap.daily_bar.close) if snap.daily_bar else 0.0,
            daily_bar_volume=int(snap.daily_bar.volume) if snap.daily_bar else 0,
            prev_daily_bar_close=float(snap.previous_daily_bar.close)
            if snap.previous_daily_bar
            else 0.0,
            timestamp=snap.latest_trade.timestamp if snap.latest_trade else datetime.now(),
        )

    def get_all_snapshots(self) -> dict[str, Snapshot]:
        """Get snapshots for all allowed symbols."""
        return {symbol: self.get_snapshot(symbol) for symbol in SYMBOLS}

    def get_recent_trades(self, symbol: str, limit: int = 50) -> list[dict]:
        """Get recent trades for a symbol."""
        if symbol not in SYMBOLS:
            raise ValueError(f"Symbol {symbol} not allowed. Only {SYMBOLS} permitted.")

        request = StockTradesRequest(
            symbol_or_symbols=symbol,
            limit=limit,
            feed=DataFeed.IEX,
        )
        trades = self.client.get_stock_trades(request)

        return [
            {
                "price": float(t.price),
                "size": int(t.size),
                "timestamp": t.timestamp,
                "conditions": t.conditions,
            }
            for t in trades[symbol]
        ]
