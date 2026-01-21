"""News data provider using Alpaca News API with sentiment scoring."""
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from functools import lru_cache
import time

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from dotenv import load_dotenv

from config.settings import SYMBOLS

load_dotenv()


# Sentiment keyword dictionaries
BULLISH_KEYWORDS = [
    "surge", "soar", "beat", "beats", "upgrade", "upgrades", "record", "partnership",
    "outperform", "rally", "jump", "gain", "gains", "growth", "profit", "profits",
    "revenue", "exceeds", "exceeded", "strong", "bullish", "positive", "optimistic",
    "breakthrough", "innovation", "expand", "expansion", "success", "successful",
    "boost", "boosted", "rises", "rising", "buy", "overweight", "opportunity",
    "momentum", "accelerate", "accelerating", "beat expectations", "upside",
    "all-time high", "record high", "outperforms", "delivers", "delivered"
]

BEARISH_KEYWORDS = [
    "fall", "falls", "drop", "drops", "miss", "misses", "downgrade", "downgrades",
    "lawsuit", "recall", "decline", "declines", "loss", "losses", "plunge", "crash",
    "slump", "weak", "bearish", "negative", "pessimistic", "warning", "warns",
    "concern", "concerns", "risk", "risks", "cut", "cuts", "layoff", "layoffs",
    "investigation", "probe", "sell", "underweight", "underperform", "disappoints",
    "disappointing", "below expectations", "downside", "tumble", "tumbles",
    "struggle", "struggles", "delay", "delays", "setback", "fail", "fails"
]


@dataclass
class NewsArticle:
    """Represents a news article with sentiment analysis."""

    headline: str
    summary: str
    source: str
    created_at: datetime
    symbols: list[str]
    url: str
    sentiment_score: float  # -1 to +1
    sentiment_label: str  # "bullish", "bearish", "neutral"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "headline": self.headline,
            "summary": self.summary[:200] if self.summary else "",
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "symbols": self.symbols,
            "url": self.url,
            "sentiment_score": round(self.sentiment_score, 3),
            "sentiment_label": self.sentiment_label,
        }


@dataclass
class NewsSentiment:
    """Aggregated news sentiment for a symbol."""

    symbol: str
    article_count: int
    avg_sentiment: float
    sentiment_label: str  # "bullish", "bearish", "neutral"
    bullish_count: int
    bearish_count: int
    neutral_count: int
    latest_headline: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "article_count": self.article_count,
            "avg_sentiment": round(self.avg_sentiment, 3),
            "sentiment_label": self.sentiment_label,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "latest_headline": self.latest_headline,
        }


class NewsProvider:
    """
    Provides news data and sentiment analysis for trading symbols.

    Uses Alpaca News API with simple keyword-based sentiment scoring.
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables
    (uses Claude's credentials by default).
    """

    # Cache TTL in seconds (15 minutes)
    CACHE_TTL = 900

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        Initialize the news provider with Alpaca NewsClient.

        Args:
            api_key: Alpaca API key (defaults to CLAUDE_ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (defaults to CLAUDE_ALPACA_SECRET_KEY env var)
        """
        # Use provided keys or fall back to Claude's credentials
        self._api_key = api_key or os.getenv("CLAUDE_ALPACA_API_KEY")
        self._secret_key = secret_key or os.getenv("CLAUDE_ALPACA_SECRET_KEY")

        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set CLAUDE_ALPACA_API_KEY and "
                "CLAUDE_ALPACA_SECRET_KEY environment variables."
            )

        self.client = NewsClient(api_key=self._api_key, secret_key=self._secret_key)
        self._cache: dict[str, tuple[float, list[NewsArticle]]] = {}

    def _get_cache_key(self, symbols: list[str], hours_back: int, limit: int) -> str:
        """Generate cache key for news request."""
        sorted_symbols = sorted(symbols)
        return f"{','.join(sorted_symbols)}:{hours_back}:{limit}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        timestamp, _ = self._cache[cache_key]
        return (time.time() - timestamp) < self.CACHE_TTL

    def _score_sentiment(self, text: str) -> tuple[float, str]:
        """
        Score sentiment of text using keyword matching.

        Returns:
            tuple of (score, label) where score is -1 to +1
        """
        if not text:
            return 0.0, "neutral"

        text_lower = text.lower()

        # Count keyword matches
        bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0, "neutral"

        # Calculate score: positive for bullish, negative for bearish
        score = (bullish_count - bearish_count) / total

        # Determine label with thresholds
        if score > 0.15:
            label = "bullish"
        elif score < -0.15:
            label = "bearish"
        else:
            label = "neutral"

        return score, label

    def get_news(
        self,
        symbols: Optional[list[str]] = None,
        hours_back: int = 24,
        limit: int = 10
    ) -> list[NewsArticle]:
        """
        Fetch recent news articles for specified symbols.

        Args:
            symbols: List of stock symbols (defaults to SYMBOLS from config)
            hours_back: How many hours of news to fetch
            limit: Maximum number of articles per symbol

        Returns:
            List of NewsArticle objects with sentiment scores
        """
        if symbols is None:
            symbols = list(SYMBOLS)

        # Validate symbols
        valid_symbols = [s for s in symbols if s in SYMBOLS]
        if not valid_symbols:
            return []

        # Check cache
        cache_key = self._get_cache_key(valid_symbols, hours_back, limit)
        if self._is_cache_valid(cache_key):
            _, articles = self._cache[cache_key]
            return articles

        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)

        try:
            # Fetch news from Alpaca
            # Note: symbols must be a comma-separated string, not a list
            request = NewsRequest(
                symbols=",".join(valid_symbols),
                start=start_time,
                end=end_time,
                limit=limit * len(valid_symbols),  # Get enough for all symbols
            )

            news_response = self.client.get_news(request)

            # Process articles
            # NewsSet has data attribute with 'news' key containing list of News objects
            news_list = news_response.data.get('news', []) if hasattr(news_response, 'data') else []
            articles = []
            for news in news_list:
                # Combine headline and summary for sentiment analysis
                full_text = f"{news.headline or ''} {news.summary or ''}"
                score, label = self._score_sentiment(full_text)

                article = NewsArticle(
                    headline=news.headline or "",
                    summary=news.summary or "",
                    source=news.source or "Unknown",
                    created_at=news.created_at,
                    symbols=news.symbols or [],
                    url=news.url or "",
                    sentiment_score=score,
                    sentiment_label=label,
                )
                articles.append(article)

            # Sort by recency
            articles.sort(key=lambda a: a.created_at, reverse=True)

            # Limit per symbol
            symbol_counts: dict[str, int] = {s: 0 for s in valid_symbols}
            filtered_articles = []
            for article in articles:
                # Check if any symbol in article matches our symbols
                article_symbols = [s for s in article.symbols if s in valid_symbols]
                if article_symbols:
                    # Check if we've hit limit for all symbols in this article
                    all_at_limit = all(symbol_counts.get(s, 0) >= limit for s in article_symbols)
                    if not all_at_limit:
                        filtered_articles.append(article)
                        for s in article_symbols:
                            symbol_counts[s] = symbol_counts.get(s, 0) + 1

            # Cache the results
            self._cache[cache_key] = (time.time(), filtered_articles)

            return filtered_articles

        except Exception as e:
            # Return empty list on error, don't crash trading
            print(f"Warning: Failed to fetch news: {e}")
            return []

    def get_news_for_symbol(
        self,
        symbol: str,
        hours_back: int = 24,
        limit: int = 10
    ) -> list[NewsArticle]:
        """
        Fetch news for a specific symbol.

        Args:
            symbol: Stock symbol (must be in SYMBOLS)
            hours_back: How many hours of news to fetch
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects for the symbol
        """
        if symbol not in SYMBOLS:
            return []

        all_news = self.get_news([symbol], hours_back, limit)
        return [a for a in all_news if symbol in a.symbols]

    def get_sentiment_summary(self, symbol: str, hours_back: int = 24) -> NewsSentiment:
        """
        Get aggregated sentiment summary for a symbol.

        Args:
            symbol: Stock symbol (must be in SYMBOLS)
            hours_back: How many hours of news to analyze

        Returns:
            NewsSentiment object with aggregated metrics
        """
        articles = self.get_news_for_symbol(symbol, hours_back, limit=20)

        if not articles:
            return NewsSentiment(
                symbol=symbol,
                article_count=0,
                avg_sentiment=0.0,
                sentiment_label="neutral",
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                latest_headline="No recent news",
            )

        # Calculate aggregates
        bullish = sum(1 for a in articles if a.sentiment_label == "bullish")
        bearish = sum(1 for a in articles if a.sentiment_label == "bearish")
        neutral = sum(1 for a in articles if a.sentiment_label == "neutral")
        avg_score = sum(a.sentiment_score for a in articles) / len(articles)

        # Determine overall sentiment
        if avg_score > 0.1:
            overall_label = "bullish"
        elif avg_score < -0.1:
            overall_label = "bearish"
        else:
            overall_label = "neutral"

        return NewsSentiment(
            symbol=symbol,
            article_count=len(articles),
            avg_sentiment=avg_score,
            sentiment_label=overall_label,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            latest_headline=articles[0].headline if articles else "No recent news",
        )

    def get_all_sentiments(self, hours_back: int = 24) -> dict[str, NewsSentiment]:
        """
        Get sentiment summaries for all trading symbols.

        Args:
            hours_back: How many hours of news to analyze

        Returns:
            Dictionary mapping symbol to NewsSentiment
        """
        return {symbol: self.get_sentiment_summary(symbol, hours_back) for symbol in SYMBOLS}

    def format_news_for_context(
        self,
        symbols: Optional[list[str]] = None,
        hours_back: int = 24,
        max_headlines: int = 3
    ) -> str:
        """
        Format news data as a string for AI agent context.

        Args:
            symbols: List of symbols (defaults to all SYMBOLS)
            hours_back: How many hours of news to include
            max_headlines: Maximum headlines to show per symbol

        Returns:
            Formatted string for inclusion in agent prompt
        """
        if symbols is None:
            symbols = list(SYMBOLS)

        output_lines = []

        for symbol in symbols:
            sentiment = self.get_sentiment_summary(symbol, hours_back)
            news = self.get_news_for_symbol(symbol, hours_back, max_headlines)

            # Format sentiment
            sentiment_str = sentiment.sentiment_label.upper()
            if sentiment.avg_sentiment != 0:
                sentiment_str += f" ({sentiment.avg_sentiment:+.2f})"

            output_lines.append(f"\nNEWS SENTIMENT for {symbol}:")
            output_lines.append(f"  Overall: {sentiment_str} ({sentiment.article_count} articles)")

            if news:
                output_lines.append("  Recent Headlines:")
                for article in news[:max_headlines]:
                    # Truncate long headlines
                    headline = article.headline[:80] + "..." if len(article.headline) > 80 else article.headline
                    output_lines.append(f"    - [{article.sentiment_label[0].upper()}] {headline}")

        return "\n".join(output_lines)

    def clear_cache(self):
        """Clear the news cache."""
        self._cache.clear()
