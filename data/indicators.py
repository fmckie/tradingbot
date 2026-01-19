"""Technical indicators calculated from market data."""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import ta
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

from .market_data import MarketDataProvider


@dataclass
class MACDResult:
    """MACD indicator values."""

    macd_line: float
    signal_line: float
    histogram: float


@dataclass
class BollingerResult:
    """Bollinger Bands values."""

    upper: float
    middle: float
    lower: float
    bandwidth: float
    percent_b: float


@dataclass
class IndicatorSnapshot:
    """All technical indicators for a symbol at a point in time."""

    symbol: str
    vwap: float
    rsi: float
    macd: MACDResult
    bollinger: BollingerResult
    atr: float
    ema_9: float
    ema_21: float
    sma_50: float
    sma_200: float
    stochastic_k: float
    stochastic_d: float
    volume_sma: float
    price_change_1h: float
    price_change_1d: float


class TechnicalIndicators:
    """Calculate technical indicators for AI analysis."""

    def __init__(self, data_provider: MarketDataProvider):
        self.data_provider = data_provider

    def calculate_vwap(self, symbol: str) -> float:
        """Calculate Volume Weighted Average Price."""
        df = self.data_provider.get_bars(symbol, "1Min", limit=390)  # Full day

        if len(df) < 10:
            return 0.0

        vwap = VolumeWeightedAveragePrice(
            high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]
        )
        return float(vwap.volume_weighted_average_price().iloc[-1])

    def calculate_rsi(self, symbol: str, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        df = self.data_provider.get_bars(symbol, "1Hour", limit=period * 3)

        if len(df) < period:
            return 50.0  # Neutral if not enough data

        rsi = RSIIndicator(close=df["close"], window=period)
        return float(rsi.rsi().iloc[-1])

    def calculate_macd(
        self, symbol: str, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> MACDResult:
        """Calculate MACD indicator."""
        df = self.data_provider.get_bars(symbol, "1Hour", limit=slow * 3)

        if len(df) < slow:
            return MACDResult(macd_line=0.0, signal_line=0.0, histogram=0.0)

        macd = MACD(close=df["close"], window_slow=slow, window_fast=fast, window_sign=signal)

        return MACDResult(
            macd_line=float(macd.macd().iloc[-1]),
            signal_line=float(macd.macd_signal().iloc[-1]),
            histogram=float(macd.macd_diff().iloc[-1]),
        )

    def calculate_bollinger_bands(self, symbol: str, period: int = 20, std: int = 2) -> BollingerResult:
        """Calculate Bollinger Bands."""
        df = self.data_provider.get_bars(symbol, "1Hour", limit=period * 3)

        if len(df) < period:
            price = df["close"].iloc[-1] if len(df) > 0 else 0.0
            return BollingerResult(
                upper=price, middle=price, lower=price, bandwidth=0.0, percent_b=0.5
            )

        bb = BollingerBands(close=df["close"], window=period, window_dev=std)

        return BollingerResult(
            upper=float(bb.bollinger_hband().iloc[-1]),
            middle=float(bb.bollinger_mavg().iloc[-1]),
            lower=float(bb.bollinger_lband().iloc[-1]),
            bandwidth=float(bb.bollinger_wband().iloc[-1]),
            percent_b=float(bb.bollinger_pband().iloc[-1]),
        )

    def calculate_atr(self, symbol: str, period: int = 14) -> float:
        """Calculate Average True Range."""
        df = self.data_provider.get_bars(symbol, "1Hour", limit=period * 3)

        if len(df) < period:
            return 0.0

        atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=period)
        return float(atr.average_true_range().iloc[-1])

    def calculate_ema(self, symbol: str, period: int) -> float:
        """Calculate Exponential Moving Average."""
        df = self.data_provider.get_bars(symbol, "1Hour", limit=period * 3)

        if len(df) < period:
            return df["close"].iloc[-1] if len(df) > 0 else 0.0

        ema = EMAIndicator(close=df["close"], window=period)
        return float(ema.ema_indicator().iloc[-1])

    def calculate_sma(self, symbol: str, period: int) -> float:
        """Calculate Simple Moving Average."""
        df = self.data_provider.get_bars(symbol, "1Hour", limit=period * 2)

        if len(df) < period:
            return df["close"].iloc[-1] if len(df) > 0 else 0.0

        sma = SMAIndicator(close=df["close"], window=period)
        return float(sma.sma_indicator().iloc[-1])

    def calculate_stochastic(
        self, symbol: str, k_period: int = 14, d_period: int = 3
    ) -> tuple[float, float]:
        """Calculate Stochastic Oscillator."""
        df = self.data_provider.get_bars(symbol, "1Hour", limit=k_period * 3)

        if len(df) < k_period:
            return 50.0, 50.0

        stoch = StochasticOscillator(
            high=df["high"], low=df["low"], close=df["close"], window=k_period, smooth_window=d_period
        )

        return float(stoch.stoch().iloc[-1]), float(stoch.stoch_signal().iloc[-1])

    def calculate_volume_sma(self, symbol: str, period: int = 20) -> float:
        """Calculate volume SMA for comparison."""
        df = self.data_provider.get_bars(symbol, "1Hour", limit=period * 2)

        if len(df) < period:
            return df["volume"].mean() if len(df) > 0 else 0.0

        return float(df["volume"].tail(period).mean())

    def calculate_price_change(self, symbol: str, periods: int) -> float:
        """Calculate price change over N periods."""
        df = self.data_provider.get_bars(symbol, "1Hour", limit=periods + 1)

        if len(df) < 2:
            return 0.0

        return float((df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100)

    def get_all_indicators(self, symbol: str) -> IndicatorSnapshot:
        """Get all technical indicators for a symbol."""
        stoch_k, stoch_d = self.calculate_stochastic(symbol)

        return IndicatorSnapshot(
            symbol=symbol,
            vwap=self.calculate_vwap(symbol),
            rsi=self.calculate_rsi(symbol),
            macd=self.calculate_macd(symbol),
            bollinger=self.calculate_bollinger_bands(symbol),
            atr=self.calculate_atr(symbol),
            ema_9=self.calculate_ema(symbol, 9),
            ema_21=self.calculate_ema(symbol, 21),
            sma_50=self.calculate_sma(symbol, 50),
            sma_200=self.calculate_sma(symbol, 200),
            stochastic_k=stoch_k,
            stochastic_d=stoch_d,
            volume_sma=self.calculate_volume_sma(symbol),
            price_change_1h=self.calculate_price_change(symbol, 1),
            price_change_1d=self.calculate_price_change(symbol, 24),
        )

    def format_indicators_for_ai(self, symbol: str) -> str:
        """Format indicators as readable text for AI consumption."""
        ind = self.get_all_indicators(symbol)

        return f"""
Technical Indicators for {symbol}:
================================
VWAP: ${ind.vwap:.2f}
RSI (14): {ind.rsi:.1f} {'(Overbought)' if ind.rsi > 70 else '(Oversold)' if ind.rsi < 30 else '(Neutral)'}

MACD:
  - MACD Line: {ind.macd.macd_line:.4f}
  - Signal Line: {ind.macd.signal_line:.4f}
  - Histogram: {ind.macd.histogram:.4f} {'(Bullish)' if ind.macd.histogram > 0 else '(Bearish)'}

Bollinger Bands (20, 2):
  - Upper: ${ind.bollinger.upper:.2f}
  - Middle: ${ind.bollinger.middle:.2f}
  - Lower: ${ind.bollinger.lower:.2f}
  - %B: {ind.bollinger.percent_b:.2f} {'(Above upper)' if ind.bollinger.percent_b > 1 else '(Below lower)' if ind.bollinger.percent_b < 0 else ''}

ATR (14): ${ind.atr:.2f}

Moving Averages:
  - EMA 9: ${ind.ema_9:.2f}
  - EMA 21: ${ind.ema_21:.2f}
  - SMA 50: ${ind.sma_50:.2f}
  - SMA 200: ${ind.sma_200:.2f}
  - Trend: {'Bullish (EMA9 > EMA21)' if ind.ema_9 > ind.ema_21 else 'Bearish (EMA9 < EMA21)'}

Stochastic (14, 3):
  - %K: {ind.stochastic_k:.1f}
  - %D: {ind.stochastic_d:.1f}

Price Change:
  - 1 Hour: {ind.price_change_1h:+.2f}%
  - 1 Day: {ind.price_change_1d:+.2f}%

Volume vs 20-period SMA: {ind.volume_sma:.0f}
"""
