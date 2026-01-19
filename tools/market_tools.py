"""Market analysis tools that AI agents can call."""
import json
from typing import Any
from data.market_data import MarketDataProvider
from data.indicators import TechnicalIndicators
from config.settings import SYMBOLS


# Tool schemas for Claude/Grok function calling
MARKET_TOOLS_SCHEMA = [
    {
        "name": "get_stock_price",
        "description": "Get the current price, bid, ask, and spread for GOOGL or TSLA. Use this to check current market prices before making trading decisions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol - must be GOOGL or TSLA",
                    "enum": SYMBOLS,
                }
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_price_history",
        "description": "Get historical OHLCV (Open, High, Low, Close, Volume) bars for analysis. Use different timeframes for different analysis needs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol - must be GOOGL or TSLA",
                    "enum": SYMBOLS,
                },
                "timeframe": {
                    "type": "string",
                    "description": "Bar timeframe",
                    "enum": ["1Min", "5Min", "15Min", "1Hour", "1Day"],
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of bars to retrieve (max 500)",
                    "minimum": 1,
                    "maximum": 500,
                },
            },
            "required": ["symbol", "timeframe", "limit"],
        },
    },
    {
        "name": "get_technical_indicators",
        "description": "Get technical indicators for a symbol including VWAP, RSI, MACD, Bollinger Bands, ATR, moving averages, and stochastic. Use this to analyze market conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol - must be GOOGL or TSLA",
                    "enum": SYMBOLS,
                },
                "indicators": {
                    "type": "array",
                    "description": "Which indicators to retrieve",
                    "items": {
                        "type": "string",
                        "enum": [
                            "vwap",
                            "rsi",
                            "macd",
                            "bollinger",
                            "atr",
                            "ema",
                            "sma",
                            "stochastic",
                            "all",
                        ],
                    },
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_market_snapshot",
        "description": "Get a complete market snapshot for a symbol including latest trade, quote, and daily bar data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol - must be GOOGL or TSLA",
                    "enum": SYMBOLS,
                }
            },
            "required": ["symbol"],
        },
    },
]


class MarketTools:
    """Handles market tool calls from AI agents."""

    def __init__(self, data_provider: MarketDataProvider, indicators: TechnicalIndicators):
        self.data_provider = data_provider
        self.indicators = indicators

    def execute(self, tool_name: str, parameters: dict) -> dict[str, Any]:
        """Execute a market tool and return the result."""
        handlers = {
            "get_stock_price": self._get_stock_price,
            "get_price_history": self._get_price_history,
            "get_technical_indicators": self._get_technical_indicators,
            "get_market_snapshot": self._get_market_snapshot,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return handler(parameters)
        except Exception as e:
            return {"error": str(e)}

    def _get_stock_price(self, params: dict) -> dict:
        """Get current stock price."""
        symbol = params.get("symbol")
        if symbol not in SYMBOLS:
            return {"error": f"Invalid symbol. Must be one of {SYMBOLS}"}

        quote = self.data_provider.get_current_quote(symbol)
        return {
            "symbol": quote.symbol,
            "last_price": quote.last_price,
            "bid_price": quote.bid_price,
            "ask_price": quote.ask_price,
            "spread": round(quote.ask_price - quote.bid_price, 4),
            "spread_percent": round(
                (quote.ask_price - quote.bid_price) / quote.last_price * 100, 4
            ),
            "bid_size": quote.bid_size,
            "ask_size": quote.ask_size,
            "timestamp": quote.timestamp.isoformat(),
        }

    def _get_price_history(self, params: dict) -> dict:
        """Get price history bars."""
        symbol = params.get("symbol")
        timeframe = params.get("timeframe", "1Hour")
        limit = min(params.get("limit", 100), 500)

        if symbol not in SYMBOLS:
            return {"error": f"Invalid symbol. Must be one of {SYMBOLS}"}

        df = self.data_provider.get_bars(symbol, timeframe, limit)

        # Convert to list of dicts for JSON serialization
        bars = []
        for _, row in df.iterrows():
            bars.append(
                {
                    "timestamp": row["timestamp"].isoformat()
                    if hasattr(row["timestamp"], "isoformat")
                    else str(row["timestamp"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                }
            )

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars_count": len(bars),
            "bars": bars,
        }

    def _get_technical_indicators(self, params: dict) -> dict:
        """Get technical indicators."""
        symbol = params.get("symbol")
        requested = params.get("indicators", ["all"])

        if symbol not in SYMBOLS:
            return {"error": f"Invalid symbol. Must be one of {SYMBOLS}"}

        if "all" in requested:
            # Return formatted text with all indicators
            return {
                "symbol": symbol,
                "analysis": self.indicators.format_indicators_for_ai(symbol),
            }

        # Return specific indicators
        result = {"symbol": symbol}

        if "vwap" in requested:
            result["vwap"] = self.indicators.calculate_vwap(symbol)

        if "rsi" in requested:
            result["rsi"] = self.indicators.calculate_rsi(symbol)

        if "macd" in requested:
            macd = self.indicators.calculate_macd(symbol)
            result["macd"] = {
                "macd_line": macd.macd_line,
                "signal_line": macd.signal_line,
                "histogram": macd.histogram,
            }

        if "bollinger" in requested:
            bb = self.indicators.calculate_bollinger_bands(symbol)
            result["bollinger"] = {
                "upper": bb.upper,
                "middle": bb.middle,
                "lower": bb.lower,
                "percent_b": bb.percent_b,
            }

        if "atr" in requested:
            result["atr"] = self.indicators.calculate_atr(symbol)

        if "ema" in requested:
            result["ema"] = {
                "ema_9": self.indicators.calculate_ema(symbol, 9),
                "ema_21": self.indicators.calculate_ema(symbol, 21),
            }

        if "sma" in requested:
            result["sma"] = {
                "sma_50": self.indicators.calculate_sma(symbol, 50),
                "sma_200": self.indicators.calculate_sma(symbol, 200),
            }

        if "stochastic" in requested:
            k, d = self.indicators.calculate_stochastic(symbol)
            result["stochastic"] = {"k": k, "d": d}

        return result

    def _get_market_snapshot(self, params: dict) -> dict:
        """Get full market snapshot."""
        symbol = params.get("symbol")

        if symbol not in SYMBOLS:
            return {"error": f"Invalid symbol. Must be one of {SYMBOLS}"}

        snap = self.data_provider.get_snapshot(symbol)

        return {
            "symbol": snap.symbol,
            "latest_trade": {
                "price": snap.latest_trade_price,
                "size": snap.latest_trade_size,
            },
            "latest_quote": {
                "bid": snap.latest_quote_bid,
                "ask": snap.latest_quote_ask,
            },
            "daily_bar": {
                "open": snap.daily_bar_open,
                "high": snap.daily_bar_high,
                "low": snap.daily_bar_low,
                "close": snap.daily_bar_close,
                "volume": snap.daily_bar_volume,
            },
            "previous_close": snap.prev_daily_bar_close,
            "daily_change_percent": round(
                (snap.daily_bar_close - snap.prev_daily_bar_close)
                / snap.prev_daily_bar_close
                * 100,
                2,
            )
            if snap.prev_daily_bar_close
            else 0.0,
            "timestamp": snap.timestamp.isoformat(),
        }
