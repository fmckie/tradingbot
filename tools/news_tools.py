"""News analysis tools that AI agents can call."""
from typing import Any, Optional

from data.news import NewsProvider, NewsArticle, NewsSentiment
from config.settings import SYMBOLS


# Tool schemas for Claude/Grok function calling
NEWS_TOOLS_SCHEMA = [
    {
        "name": "get_recent_news",
        "description": "Get recent news headlines for GOOGL or TSLA with sentiment analysis. Use this to understand market sentiment and recent events that may impact price.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol - must be GOOGL or TSLA",
                    "enum": list(SYMBOLS),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of headlines to retrieve (1-10)",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_news_sentiment",
        "description": "Get aggregated news sentiment summary for GOOGL or TSLA. Returns overall sentiment (bullish/bearish/neutral) based on recent news analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol - must be GOOGL or TSLA",
                    "enum": list(SYMBOLS),
                },
                "hours_back": {
                    "type": "integer",
                    "description": "How many hours of news to analyze (1-72)",
                    "minimum": 1,
                    "maximum": 72,
                    "default": 24,
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "search_news",
        "description": "Search recent news for a keyword across both GOOGL and TSLA. Useful for finding specific events or topics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Keyword to search for in headlines and summaries",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (1-10)",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                },
            },
            "required": ["keyword"],
        },
    },
]


class NewsTools:
    """Handles news tool calls from AI agents."""

    def __init__(self, news_provider: NewsProvider):
        """
        Initialize news tools.

        Args:
            news_provider: NewsProvider instance for fetching news data
        """
        self.news_provider = news_provider

    def execute(self, tool_name: str, parameters: dict) -> dict[str, Any]:
        """Execute a news tool and return the result."""
        handlers = {
            "get_recent_news": self._get_recent_news,
            "get_news_sentiment": self._get_news_sentiment,
            "search_news": self._search_news,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return handler(parameters)
        except Exception as e:
            return {"error": str(e)}

    def _get_recent_news(self, params: dict) -> dict[str, Any]:
        """Get recent news headlines for a symbol."""
        symbol = params.get("symbol")
        limit = min(params.get("limit", 5), 10)

        if symbol not in SYMBOLS:
            return {"error": f"Invalid symbol. Must be one of {list(SYMBOLS)}"}

        articles = self.news_provider.get_news_for_symbol(symbol, hours_back=24, limit=limit)

        return {
            "symbol": symbol,
            "article_count": len(articles),
            "articles": [
                {
                    "headline": a.headline,
                    "source": a.source,
                    "sentiment": a.sentiment_label,
                    "sentiment_score": round(a.sentiment_score, 2),
                    "published": a.created_at.isoformat(),
                }
                for a in articles
            ],
        }

    def _get_news_sentiment(self, params: dict) -> dict[str, Any]:
        """Get aggregated news sentiment for a symbol."""
        symbol = params.get("symbol")
        hours_back = min(params.get("hours_back", 24), 72)

        if symbol not in SYMBOLS:
            return {"error": f"Invalid symbol. Must be one of {list(SYMBOLS)}"}

        sentiment = self.news_provider.get_sentiment_summary(symbol, hours_back)

        return {
            "symbol": symbol,
            "hours_analyzed": hours_back,
            "article_count": sentiment.article_count,
            "overall_sentiment": sentiment.sentiment_label,
            "sentiment_score": round(sentiment.avg_sentiment, 3),
            "breakdown": {
                "bullish": sentiment.bullish_count,
                "bearish": sentiment.bearish_count,
                "neutral": sentiment.neutral_count,
            },
            "latest_headline": sentiment.latest_headline,
            "interpretation": self._interpret_sentiment(sentiment),
        }

    def _search_news(self, params: dict) -> dict[str, Any]:
        """Search news for a keyword."""
        keyword = params.get("keyword", "").lower().strip()
        limit = min(params.get("limit", 5), 10)

        if not keyword:
            return {"error": "Keyword is required"}

        if len(keyword) < 2:
            return {"error": "Keyword must be at least 2 characters"}

        # Get all recent news
        all_articles = self.news_provider.get_news(list(SYMBOLS), hours_back=48, limit=30)

        # Filter by keyword
        matching = []
        for article in all_articles:
            text = f"{article.headline} {article.summary}".lower()
            if keyword in text:
                matching.append(article)
            if len(matching) >= limit:
                break

        return {
            "keyword": keyword,
            "match_count": len(matching),
            "articles": [
                {
                    "headline": a.headline,
                    "symbols": a.symbols,
                    "source": a.source,
                    "sentiment": a.sentiment_label,
                    "published": a.created_at.isoformat(),
                }
                for a in matching
            ],
        }

    def _interpret_sentiment(self, sentiment: NewsSentiment) -> str:
        """Generate human-readable interpretation of sentiment."""
        if sentiment.article_count == 0:
            return "No recent news coverage - neutral signal"

        score = sentiment.avg_sentiment
        count = sentiment.article_count

        if score > 0.3:
            return f"Strongly bullish news flow ({count} articles) - positive market sentiment"
        elif score > 0.1:
            return f"Moderately bullish news ({count} articles) - slight positive bias"
        elif score < -0.3:
            return f"Strongly bearish news flow ({count} articles) - negative market sentiment"
        elif score < -0.1:
            return f"Moderately bearish news ({count} articles) - slight negative bias"
        else:
            return f"Mixed/neutral news ({count} articles) - no clear directional bias"
