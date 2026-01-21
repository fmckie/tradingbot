"""Alpaca API configuration for both trading accounts."""
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.live import StockDataStream

load_dotenv()


def get_claude_client() -> tuple[TradingClient, StockHistoricalDataClient]:
    """Get Alpaca clients for Claude's trading account."""
    api_key = os.getenv("CLAUDE_ALPACA_API_KEY")
    secret_key = os.getenv("CLAUDE_ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("Claude Alpaca credentials not found in environment")

    trading_client = TradingClient(api_key, secret_key, paper=True)
    data_client = StockHistoricalDataClient(api_key, secret_key)

    return trading_client, data_client


def get_grok_client() -> tuple[TradingClient, StockHistoricalDataClient]:
    """Get Alpaca clients for Grok's trading account."""
    api_key = os.getenv("GROK_ALPACA_API_KEY")
    secret_key = os.getenv("GROK_ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("Grok Alpaca credentials not found in environment")

    trading_client = TradingClient(api_key, secret_key, paper=True)
    data_client = StockHistoricalDataClient(api_key, secret_key)

    return trading_client, data_client


def get_data_stream(api_key: str, secret_key: str) -> StockDataStream:
    """Get real-time data stream client."""
    return StockDataStream(api_key, secret_key)


def get_news_client() -> NewsClient:
    """Get Alpaca news client (no API keys required)."""
    return NewsClient()
