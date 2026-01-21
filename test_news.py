#!/usr/bin/env python3
"""Tests for the news integration."""
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from data.news import (
    NewsProvider,
    NewsArticle,
    NewsSentiment,
    BULLISH_KEYWORDS,
    BEARISH_KEYWORDS,
)
from tools.news_tools import NewsTools
from config.settings import SYMBOLS


def test_sentiment_scoring():
    """Test the keyword-based sentiment scoring."""
    # Create a mock provider to test the _score_sentiment method without API keys
    with patch.object(NewsProvider, '__init__', lambda self, **kwargs: None):
        provider = NewsProvider()
        provider._cache = {}

        # Test bullish keywords
        score, label = provider._score_sentiment("Stock surges on record earnings beat")
        assert score > 0, f"Expected positive score for bullish text, got {score}"
        assert label == "bullish", f"Expected bullish label, got {label}"

        # Test bearish keywords
        score, label = provider._score_sentiment("Stock plunges after disappointing lawsuit news")
        assert score < 0, f"Expected negative score for bearish text, got {score}"
        assert label == "bearish", f"Expected bearish label, got {label}"

        # Test neutral text
        score, label = provider._score_sentiment("Company announces quarterly meeting scheduled")
        assert label == "neutral", f"Expected neutral label, got {label}"

        # Test empty text
        score, label = provider._score_sentiment("")
        assert score == 0, f"Expected 0 score for empty text, got {score}"
        assert label == "neutral", f"Expected neutral label for empty, got {label}"

    print("[PASS] Sentiment scoring tests passed")


def test_news_article_dataclass():
    """Test NewsArticle dataclass."""
    article = NewsArticle(
        headline="Test Headline",
        summary="Test summary content",
        source="Reuters",
        created_at=datetime.now(),
        symbols=["GOOGL"],
        url="https://example.com/news",
        sentiment_score=0.5,
        sentiment_label="bullish",
    )

    # Test to_dict
    data = article.to_dict()
    assert data["headline"] == "Test Headline"
    assert data["source"] == "Reuters"
    assert data["sentiment_score"] == 0.5
    assert data["sentiment_label"] == "bullish"
    assert "GOOGL" in data["symbols"]

    print("[PASS] NewsArticle dataclass tests passed")


def test_news_sentiment_dataclass():
    """Test NewsSentiment dataclass."""
    sentiment = NewsSentiment(
        symbol="TSLA",
        article_count=10,
        avg_sentiment=0.25,
        sentiment_label="bullish",
        bullish_count=6,
        bearish_count=2,
        neutral_count=2,
        latest_headline="Tesla stock rises on delivery numbers",
    )

    # Test to_dict
    data = sentiment.to_dict()
    assert data["symbol"] == "TSLA"
    assert data["article_count"] == 10
    assert data["avg_sentiment"] == 0.25
    assert data["bullish_count"] == 6
    assert data["bearish_count"] == 2

    print("[PASS] NewsSentiment dataclass tests passed")


def test_caching_behavior():
    """Test that caching works correctly."""
    # Create a mock provider to test caching methods without API keys
    with patch.object(NewsProvider, '__init__', lambda self, **kwargs: None):
        provider = NewsProvider()
        provider._cache = {}
        provider.CACHE_TTL = 900

        # Generate cache key
        key1 = provider._get_cache_key(["GOOGL"], 24, 10)
        key2 = provider._get_cache_key(["GOOGL"], 24, 10)
        key3 = provider._get_cache_key(["TSLA"], 24, 10)

        assert key1 == key2, "Same parameters should generate same cache key"
        assert key1 != key3, "Different symbols should generate different cache keys"

        # Test cache validity
        assert not provider._is_cache_valid(key1), "Empty cache should not be valid"

        # Add to cache
        provider._cache[key1] = (time.time(), [])
        assert provider._is_cache_valid(key1), "Fresh cache should be valid"

        # Test expired cache
        provider._cache[key1] = (time.time() - 1000, [])  # Expired
        assert not provider._is_cache_valid(key1), "Expired cache should not be valid"

    print("[PASS] Caching behavior tests passed")


def test_news_tools_schema():
    """Test that news tools schema is properly defined."""
    from tools.news_tools import NEWS_TOOLS_SCHEMA

    assert len(NEWS_TOOLS_SCHEMA) == 3, f"Expected 3 news tools, got {len(NEWS_TOOLS_SCHEMA)}"

    tool_names = [t["name"] for t in NEWS_TOOLS_SCHEMA]
    assert "get_recent_news" in tool_names
    assert "get_news_sentiment" in tool_names
    assert "search_news" in tool_names

    # Check schema structure
    for tool in NEWS_TOOLS_SCHEMA:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
        assert "properties" in tool["input_schema"]

    print("[PASS] News tools schema tests passed")


def test_news_tools_execute():
    """Test NewsTools execution with mocked provider."""
    # Create mock provider
    mock_provider = Mock(spec=NewsProvider)

    # Mock get_news_for_symbol
    mock_article = NewsArticle(
        headline="Test headline",
        summary="Test summary",
        source="Test Source",
        created_at=datetime.now(),
        symbols=["GOOGL"],
        url="https://test.com",
        sentiment_score=0.3,
        sentiment_label="bullish",
    )
    mock_provider.get_news_for_symbol.return_value = [mock_article]

    # Mock get_sentiment_summary
    mock_sentiment = NewsSentiment(
        symbol="GOOGL",
        article_count=5,
        avg_sentiment=0.2,
        sentiment_label="bullish",
        bullish_count=3,
        bearish_count=1,
        neutral_count=1,
        latest_headline="Test headline",
    )
    mock_provider.get_sentiment_summary.return_value = mock_sentiment

    # Create tools with mock
    tools = NewsTools(mock_provider)

    # Test get_recent_news
    result = tools.execute("get_recent_news", {"symbol": "GOOGL", "limit": 5})
    assert "error" not in result
    assert result["symbol"] == "GOOGL"
    assert len(result["articles"]) == 1

    # Test get_news_sentiment
    result = tools.execute("get_news_sentiment", {"symbol": "GOOGL"})
    assert "error" not in result
    assert result["overall_sentiment"] == "bullish"
    assert "interpretation" in result

    # Test invalid symbol
    result = tools.execute("get_recent_news", {"symbol": "INVALID"})
    assert "error" in result

    # Test unknown tool
    result = tools.execute("unknown_tool", {})
    assert "error" in result

    print("[PASS] NewsTools execute tests passed")


def test_format_news_for_context():
    """Test formatting news for AI context."""
    # Create a mock provider
    with patch.object(NewsProvider, '__init__', lambda self, **kwargs: None):
        provider = NewsProvider()
        provider._cache = {}

        # Mock the methods to return test data
        with patch.object(provider, 'get_sentiment_summary') as mock_sentiment, \
             patch.object(provider, 'get_news_for_symbol') as mock_news:

            mock_sentiment.return_value = NewsSentiment(
                symbol="GOOGL",
                article_count=3,
                avg_sentiment=0.25,
                sentiment_label="bullish",
                bullish_count=2,
                bearish_count=0,
                neutral_count=1,
                latest_headline="Google announces new AI features",
            )

            mock_news.return_value = [
                NewsArticle(
                    headline="Google announces new AI features",
                    summary="Summary",
                    source="Reuters",
                    created_at=datetime.now(),
                    symbols=["GOOGL"],
                    url="https://test.com",
                    sentiment_score=0.4,
                    sentiment_label="bullish",
                )
            ]

            formatted = provider.format_news_for_context(["GOOGL"], hours_back=24, max_headlines=3)

            assert "GOOGL" in formatted
            assert "BULLISH" in formatted
            assert "3 articles" in formatted

    print("[PASS] Format news for context tests passed")


def test_keyword_lists():
    """Verify keyword lists are properly defined."""
    assert len(BULLISH_KEYWORDS) > 10, "Should have many bullish keywords"
    assert len(BEARISH_KEYWORDS) > 10, "Should have many bearish keywords"

    # Check for common keywords
    assert "surge" in BULLISH_KEYWORDS
    assert "beat" in BULLISH_KEYWORDS
    assert "drop" in BEARISH_KEYWORDS
    assert "lawsuit" in BEARISH_KEYWORDS

    # No overlap between lists
    overlap = set(BULLISH_KEYWORDS) & set(BEARISH_KEYWORDS)
    assert len(overlap) == 0, f"Keywords should not overlap: {overlap}"

    print("[PASS] Keyword lists tests passed")


def test_symbol_validation():
    """Test that only valid symbols are accepted."""
    # Create a mock provider to test symbol validation without API keys
    with patch.object(NewsProvider, '__init__', lambda self, **kwargs: None):
        provider = NewsProvider()
        provider._cache = {}
        provider.client = Mock()
        provider.CACHE_TTL = 900

        # Mock the news response
        mock_response = Mock()
        mock_response.news = []
        provider.client.get_news.return_value = mock_response

        # Valid symbols should work
        result = provider.get_news(["GOOGL"], hours_back=1, limit=1)
        # Result could be empty if no news, but shouldn't raise error

        # Invalid symbols should return empty
        result = provider.get_news(["INVALID"], hours_back=1, limit=1)
        assert result == [], "Invalid symbol should return empty list"

    print("[PASS] Symbol validation tests passed")


def test_interpret_sentiment():
    """Test sentiment interpretation helper."""
    mock_provider = Mock(spec=NewsProvider)
    tools = NewsTools(mock_provider)

    # Test various sentiment levels
    strongly_bullish = NewsSentiment(
        symbol="TEST", article_count=5, avg_sentiment=0.5,
        sentiment_label="bullish", bullish_count=4, bearish_count=0,
        neutral_count=1, latest_headline="Test"
    )
    assert "bullish" in tools._interpret_sentiment(strongly_bullish).lower()

    strongly_bearish = NewsSentiment(
        symbol="TEST", article_count=5, avg_sentiment=-0.5,
        sentiment_label="bearish", bullish_count=0, bearish_count=4,
        neutral_count=1, latest_headline="Test"
    )
    assert "bearish" in tools._interpret_sentiment(strongly_bearish).lower()

    neutral = NewsSentiment(
        symbol="TEST", article_count=5, avg_sentiment=0.0,
        sentiment_label="neutral", bullish_count=2, bearish_count=2,
        neutral_count=1, latest_headline="Test"
    )
    assert "mixed" in tools._interpret_sentiment(neutral).lower() or \
           "neutral" in tools._interpret_sentiment(neutral).lower()

    no_news = NewsSentiment(
        symbol="TEST", article_count=0, avg_sentiment=0.0,
        sentiment_label="neutral", bullish_count=0, bearish_count=0,
        neutral_count=0, latest_headline="No recent news"
    )
    assert "neutral" in tools._interpret_sentiment(no_news).lower()

    print("[PASS] Interpret sentiment tests passed")


def run_all_tests():
    """Run all news integration tests."""
    print("\n" + "=" * 50)
    print("Running News Integration Tests")
    print("=" * 50 + "\n")

    tests = [
        test_sentiment_scoring,
        test_news_article_dataclass,
        test_news_sentiment_dataclass,
        test_caching_behavior,
        test_news_tools_schema,
        test_news_tools_execute,
        test_format_news_for_context,
        test_keyword_lists,
        test_symbol_validation,
        test_interpret_sentiment,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test_func.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
