"""Analysis tools for deeper market analysis."""

from typing import Any

from config.settings import SYMBOLS
from data.indicators import TechnicalIndicators
from data.market_data import MarketDataProvider

ANALYSIS_TOOLS_SCHEMA = [
    {
        "name": "get_market_context",
        "description": (
            "Get broader market context including both GOOGL and TSLA "
            "snapshots, their relative performance, and market conditions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "compare_stocks",
        "description": (
            "Compare GOOGL and TSLA on various metrics to decide which one to trade."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_support_resistance",
        "description": (
            "Calculate support and resistance levels based on recent price action."
        ),
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
        "name": "analyze_trend",
        "description": "Analyze the current trend direction and strength for a symbol.",
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


class AnalysisTools:
    """Advanced analysis tools for AI agents."""

    def __init__(
        self, data_provider: MarketDataProvider, indicators: TechnicalIndicators
    ):
        self.data_provider = data_provider
        self.indicators = indicators

    def execute(self, tool_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute an analysis tool and return the result."""
        handlers = {
            "get_market_context": self._get_market_context,
            "compare_stocks": self._compare_stocks,
            "get_support_resistance": self._get_support_resistance,
            "analyze_trend": self._analyze_trend,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return handler(parameters)
        except Exception as e:
            return {"error": str(e)}

    def _get_market_context(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get overall market context."""
        context = {}

        for symbol in SYMBOLS:
            snap = self.data_provider.get_snapshot(symbol)
            ind = self.indicators.get_all_indicators(symbol)

            daily_change = 0.0
            if snap.prev_daily_bar_close > 0:
                daily_change = (
                    (snap.daily_bar_close - snap.prev_daily_bar_close)
                    / snap.prev_daily_bar_close
                    * 100
                )

            context[symbol] = {
                "price": snap.latest_trade_price,
                "daily_change_percent": round(daily_change, 2),
                "daily_volume": snap.daily_bar_volume,
                "daily_range": round(snap.daily_bar_high - snap.daily_bar_low, 2),
                "rsi": round(ind.rsi, 1),
                "macd_histogram": round(ind.macd.histogram, 4),
                "above_vwap": snap.latest_trade_price > ind.vwap,
                "trend": "bullish" if ind.ema_9 > ind.ema_21 else "bearish",
            }

        # Relative strength
        googl_change = float(context["GOOGL"]["daily_change_percent"])
        tsla_change = float(context["TSLA"]["daily_change_percent"])

        return {
            "symbols": context,
            "relative_strength": {
                "stronger": "GOOGL" if googl_change > tsla_change else "TSLA",
                "googl_vs_tsla": round(googl_change - tsla_change, 2),
            },
            "market_condition": self._assess_market_condition(context),
        }

    def _assess_market_condition(self, context: dict[str, Any]) -> str:
        """Assess overall market condition."""
        # Both bullish
        if (
            context["GOOGL"]["trend"] == "bullish"
            and context["TSLA"]["trend"] == "bullish"
        ):
            return "bullish - both stocks trending up"

        # Both bearish
        if (
            context["GOOGL"]["trend"] == "bearish"
            and context["TSLA"]["trend"] == "bearish"
        ):
            return "bearish - both stocks trending down"

        # Mixed
        return "mixed - stocks showing different trends"

    def _compare_stocks(self, params: dict[str, Any]) -> dict[str, Any]:
        """Compare GOOGL and TSLA for trading decision."""
        comparison = {}

        for symbol in SYMBOLS:
            ind = self.indicators.get_all_indicators(symbol)
            snap = self.data_provider.get_snapshot(symbol)

            # Calculate volatility score
            atr_percent = (
                (ind.atr / snap.latest_trade_price) * 100
                if snap.latest_trade_price > 0
                else 0
            )

            # Calculate momentum score (-100 to +100)
            momentum_score: float = 0
            momentum_score += (50 - ind.rsi) * -1  # RSI contribution
            momentum_score += 50 if ind.macd.histogram > 0 else -50  # MACD contribution
            momentum_score = max(-100, min(100, momentum_score))

            # Calculate mean reversion potential
            mr_potential = ind.bollinger.percent_b  # <0 = oversold, >1 = overbought

            comparison[symbol] = {
                "current_price": snap.latest_trade_price,
                "rsi": round(ind.rsi, 1),
                "rsi_signal": "oversold"
                if ind.rsi < 30
                else "overbought"
                if ind.rsi > 70
                else "neutral",
                "macd_signal": "bullish" if ind.macd.histogram > 0 else "bearish",
                "trend": "up" if ind.ema_9 > ind.ema_21 else "down",
                "volatility_atr_percent": round(atr_percent, 2),
                "bollinger_position": round(ind.bollinger.percent_b, 2),
                "momentum_score": round(momentum_score, 0),
                "mean_reversion_opportunity": mr_potential < 0 or mr_potential > 1,
            }

        # Recommendations
        googl = comparison["GOOGL"]
        tsla = comparison["TSLA"]

        recommendations = []

        # Momentum trade opportunity
        if abs(float(googl["momentum_score"])) > abs(float(tsla["momentum_score"])):
            better_momentum = "GOOGL"
        else:
            better_momentum = "TSLA"
        recommendations.append(f"Stronger momentum: {better_momentum}")

        # Mean reversion opportunity
        if googl["mean_reversion_opportunity"]:
            recommendations.append(
                f"GOOGL at Bollinger extreme ({googl['bollinger_position']})"
            )
        if tsla["mean_reversion_opportunity"]:
            recommendations.append(
                f"TSLA at Bollinger extreme ({tsla['bollinger_position']})"
            )

        # Lower volatility (safer)
        if float(googl["volatility_atr_percent"]) < float(
            tsla["volatility_atr_percent"]
        ):
            recommendations.append("GOOGL has lower volatility (safer)")
        else:
            recommendations.append("TSLA has lower volatility (safer)")

        return {
            "comparison": comparison,
            "recommendations": recommendations,
        }

    def _get_support_resistance(self, params: dict[str, Any]) -> dict[str, Any]:
        """Calculate support and resistance levels."""
        symbol = params.get("symbol")

        if symbol not in SYMBOLS:
            return {"error": f"Invalid symbol. Must be one of {SYMBOLS}"}

        # Get recent price data
        df = self.data_provider.get_bars(symbol, "1Hour", limit=100)

        if len(df) < 20:
            return {"error": "Not enough data for support/resistance calculation"}

        # Simple pivot point calculation
        high = df["high"].max()
        low = df["low"].min()
        close = df["close"].iloc[-1]

        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)

        # Recent swing highs/lows
        recent_df = df.tail(20)
        recent_high = recent_df["high"].max()
        recent_low = recent_df["low"].min()

        # Get Bollinger bands as dynamic S/R
        bb = self.indicators.calculate_bollinger_bands(symbol)

        return {
            "symbol": symbol,
            "current_price": close,
            "pivot_point": round(pivot, 2),
            "resistance_levels": {
                "r1_pivot": round(r1, 2),
                "r2_pivot": round(r2, 2),
                "recent_high": round(recent_high, 2),
                "bollinger_upper": round(bb.upper, 2),
            },
            "support_levels": {
                "s1_pivot": round(s1, 2),
                "s2_pivot": round(s2, 2),
                "recent_low": round(recent_low, 2),
                "bollinger_lower": round(bb.lower, 2),
            },
            "price_range_20h": round(recent_high - recent_low, 2),
        }

    def _analyze_trend(self, params: dict[str, Any]) -> dict[str, Any]:
        """Analyze trend direction and strength."""
        symbol = params.get("symbol")

        if symbol not in SYMBOLS:
            return {"error": f"Invalid symbol. Must be one of {SYMBOLS}"}

        ind = self.indicators.get_all_indicators(symbol)
        df = self.data_provider.get_bars(symbol, "1Hour", limit=50)

        # Trend direction based on multiple timeframes
        short_term = "up" if ind.ema_9 > ind.ema_21 else "down"
        medium_term = "up" if df["close"].iloc[-1] > ind.sma_50 else "down"
        long_term = "up" if df["close"].iloc[-1] > ind.sma_200 else "down"

        # Trend strength (ADX would be better, but using price action)
        recent_highs = df["high"].tail(10)
        recent_lows = df["low"].tail(10)

        higher_highs = sum(
            1
            for i in range(1, len(recent_highs))
            if recent_highs.iloc[i] > recent_highs.iloc[i - 1]
        )
        higher_lows = sum(
            1
            for i in range(1, len(recent_lows))
            if recent_lows.iloc[i] > recent_lows.iloc[i - 1]
        )

        uptrend_score = higher_highs + higher_lows
        trend_strength = (
            "strong" if uptrend_score >= 14 or uptrend_score <= 4 else "weak"
        )

        # Overall assessment
        if short_term == medium_term == long_term:
            overall = f"strong {short_term}trend"
        elif short_term == medium_term:
            overall = f"{short_term}trend developing"
        else:
            overall = "mixed/ranging"

        return {
            "symbol": symbol,
            "trend_analysis": {
                "short_term_ema": short_term,
                "medium_term_sma50": medium_term,
                "long_term_sma200": long_term,
                "trend_strength": trend_strength,
                "overall": overall,
            },
            "price_action": {
                "higher_highs_count": higher_highs,
                "higher_lows_count": higher_lows,
                "uptrend_score": uptrend_score,
            },
            "key_levels": {
                "ema_9": round(ind.ema_9, 2),
                "ema_21": round(ind.ema_21, 2),
                "sma_50": round(ind.sma_50, 2),
                "sma_200": round(ind.sma_200, 2),
            },
            "recommendation": self._trend_recommendation(
                short_term, medium_term, trend_strength
            ),
        }

    def _trend_recommendation(self, short: str, medium: str, strength: str) -> str:
        """Generate trend-based recommendation."""
        if short == "up" and medium == "up" and strength == "strong":
            return "Strong uptrend - consider momentum long positions"
        elif short == "down" and medium == "down" and strength == "strong":
            return "Strong downtrend - consider short positions or stay out"
        elif short == "up" and medium == "down":
            return "Potential trend reversal up - watch for confirmation"
        elif short == "down" and medium == "up":
            return "Pullback in uptrend - potential buy opportunity"
        else:
            return "No clear trend - consider range trading or waiting"
