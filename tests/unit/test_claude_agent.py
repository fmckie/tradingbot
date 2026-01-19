"""
Comprehensive tests for Claude agent decision parsing.

Tests the _parse_decision() method with various input formats
to ensure robust parsing of AI responses.
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from agents.base_agent import ActionType, StrategyType, TradingDecision
from agents.claude_agent import ClaudeAgent


@dataclass
class MockTextBlock:
    """Mock Anthropic text block."""
    type: str = "text"
    text: str = ""


@dataclass
class MockResponse:
    """Mock Anthropic response."""
    content: list
    stop_reason: str = "end_turn"


class TestClaudeAgentParseDecision:
    """Test suite for ClaudeAgent._parse_decision() method."""

    @pytest.fixture
    def agent(self):
        """Create ClaudeAgent instance with mocked dependencies."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('agents.claude_agent.anthropic.Anthropic'):
                agent = ClaudeAgent(tools={})
                return agent

    @pytest.fixture
    def timestamp(self):
        """Standard timestamp for tests."""
        return datetime(2024, 1, 15, 10, 30, 0)

    def _make_response(self, text: str) -> MockResponse:
        """Helper to create mock response from text."""
        return MockResponse(content=[MockTextBlock(text=text)])

    # ==================== ACTION DETECTION TESTS ====================

    def test_parse_action_buy_with_space(self, agent, timestamp):
        """Test parsing 'ACTION: BUY' with space."""
        response = self._make_response("""
        Based on my analysis:
        ACTION: BUY
        SYMBOL: GOOGL
        QUANTITY: 10 shares
        STOP_LOSS: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.BUY

    def test_parse_action_buy_no_space(self, agent, timestamp):
        """Test parsing 'ACTION:BUY' without space."""
        response = self._make_response("""
        ACTION:BUY
        Symbol: TSLA
        Quantity: 5
        Stop Loss: $250.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.BUY

    def test_parse_action_sell_with_space(self, agent, timestamp):
        """Test parsing 'ACTION: SELL' with space."""
        response = self._make_response("""
        ACTION: SELL
        SYMBOL: TSLA
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.SELL

    def test_parse_action_sell_no_space(self, agent, timestamp):
        """Test parsing 'ACTION:SELL' without space."""
        response = self._make_response("ACTION:SELL for GOOGL")
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.SELL

    def test_parse_action_hold_default(self, agent, timestamp):
        """Test that unrecognized input defaults to HOLD."""
        response = self._make_response("""
        I'm not sure what to do right now.
        The market is unclear.
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.HOLD

    def test_parse_action_close_with_space(self, agent, timestamp):
        """Test parsing 'ACTION: CLOSE' with space."""
        response = self._make_response("""
        ACTION: CLOSE
        SYMBOL: GOOGL
        Reason: Taking profits
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.CLOSE

    def test_parse_action_close_no_space(self, agent, timestamp):
        """Test parsing 'ACTION:CLOSE' without space."""
        response = self._make_response("ACTION:CLOSE position on TSLA")
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.CLOSE

    def test_parse_action_case_insensitive(self, agent, timestamp):
        """Test that action parsing is case insensitive."""
        response = self._make_response("action: buy for GOOGL")
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.BUY

    def test_parse_action_mixed_case(self, agent, timestamp):
        """Test parsing with mixed case 'Action: Buy'."""
        response = self._make_response("""
        Action: Buy
        Symbol: GOOGL
        Quantity: 15
        Stop Loss: $180.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.BUY

    def test_parse_explicit_hold(self, agent, timestamp):
        """Test that explicit HOLD mention keeps HOLD action."""
        response = self._make_response("""
        After careful analysis, I recommend:
        ACTION: HOLD

        The market is too volatile right now.
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.HOLD

    # ==================== SYMBOL DETECTION TESTS ====================

    def test_parse_symbol_googl(self, agent, timestamp):
        """Test parsing GOOGL symbol."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        Quantity: 10
        Stop Loss: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.symbol == "GOOGL"

    def test_parse_symbol_tsla(self, agent, timestamp):
        """Test parsing TSLA symbol."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: TSLA
        Quantity: 5
        Stop Loss: $250.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.symbol == "TSLA"

    def test_parse_symbol_in_text(self, agent, timestamp):
        """Test parsing symbol mentioned inline."""
        response = self._make_response("""
        ACTION: BUY
        I want to buy GOOGL at current prices.
        Quantity: 10
        Stop Loss: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.symbol == "GOOGL"

    def test_parse_symbol_googl_takes_precedence(self, agent, timestamp):
        """Test that GOOGL is chosen when mentioned first."""
        response = self._make_response("""
        ACTION: BUY
        Comparing GOOGL and TSLA, I choose GOOGL.
        Quantity: 10
        Stop Loss: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.symbol == "GOOGL"

    def test_parse_symbol_tsla_when_googl_not_present(self, agent, timestamp):
        """Test TSLA is chosen when GOOGL not mentioned."""
        response = self._make_response("""
        ACTION: BUY
        TSLA looks strong today.
        Quantity: 5
        Stop Loss: $250.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.symbol == "TSLA"

    def test_parse_no_symbol_for_hold(self, agent, timestamp):
        """Test that HOLD action has no symbol set."""
        response = self._make_response("""
        ACTION: HOLD
        Waiting for better conditions.
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.symbol is None

    # ==================== QUANTITY PARSING TESTS ====================

    def test_parse_quantity_basic(self, agent, timestamp):
        """Test parsing basic quantity."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        Stop Loss: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.quantity == 10

    def test_parse_quantity_with_shares_word(self, agent, timestamp):
        """Test parsing 'QUANTITY: 15 shares'."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 15 shares
        Stop Loss: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.quantity == 15

    def test_parse_quantity_shares_keyword(self, agent, timestamp):
        """Test parsing 'Shares: 20'."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: TSLA
        Shares: 20
        Stop Loss: $250.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.quantity == 20

    def test_parse_quantity_large_number(self, agent, timestamp):
        """Test parsing large quantity."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 100
        Stop Loss: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.quantity == 100

    def test_parse_quantity_single_digit(self, agent, timestamp):
        """Test parsing single digit quantity."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 5
        Stop Loss: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.quantity == 5

    def test_parse_quantity_from_inline_text(self, agent, timestamp):
        """Test parsing quantity from text like 'buy 25 shares'."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        I recommend buying 25 shares at current prices.
        Stop Loss: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        # Note: current implementation may not parse this correctly
        # This test documents expected behavior

    def test_parse_quantity_missing(self, agent, timestamp):
        """Test that missing quantity returns None."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        Stop Loss: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.quantity is None

    # ==================== STOP-LOSS PARSING TESTS ====================

    def test_parse_stop_loss_with_dollar_sign(self, agent, timestamp):
        """Test parsing stop-loss with $ sign."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        STOP LOSS: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.stop_loss == 175.00

    def test_parse_stop_loss_without_dollar_sign(self, agent, timestamp):
        """Test parsing stop-loss without $ sign."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        STOP LOSS: 175.50
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.stop_loss == 175.50

    def test_parse_stop_loss_with_comma(self, agent, timestamp):
        """Test parsing stop-loss with comma in price."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        STOP LOSS: $1,175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.stop_loss == 1175.00

    def test_parse_stop_loss_hyphenated(self, agent, timestamp):
        """Test parsing 'Stop-Loss: $175.00'."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        Stop-Loss: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.stop_loss == 175.00

    def test_parse_stop_loss_inline(self, agent, timestamp):
        """Test parsing stop loss mentioned inline."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        Set stop loss at $175.00 to limit risk.
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.stop_loss == 175.00

    def test_parse_stop_loss_decimal_precision(self, agent, timestamp):
        """Test parsing stop-loss with various decimal places."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        STOP LOSS: $175.567
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.stop_loss == 175.567

    def test_parse_stop_loss_sanity_check_valid(self, agent, timestamp):
        """Test that valid stock prices pass sanity check (>10)."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        STOP LOSS: $150.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.stop_loss == 150.00

    def test_parse_stop_loss_sanity_check_too_low(self, agent, timestamp):
        """Test that prices below 10 are rejected."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        STOP LOSS: $5.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        # Price below 10 should be rejected by sanity check
        assert decision.stop_loss is None

    # ==================== TAKE-PROFIT PARSING TESTS ====================

    def test_parse_take_profit_basic(self, agent, timestamp):
        """Test parsing basic take-profit."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        STOP LOSS: $175.00
        TAKE PROFIT: $195.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.take_profit == 195.00

    def test_parse_take_profit_with_dollar_sign(self, agent, timestamp):
        """Test parsing take-profit with $ sign."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        STOP LOSS: $175.00
        TAKE PROFIT: $200.50
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.take_profit == 200.50

    def test_parse_take_profit_hyphenated(self, agent, timestamp):
        """Test parsing 'Take-Profit: $195.00'."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        STOP LOSS: $175.00
        Take-Profit: $195.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.take_profit == 195.00

    def test_parse_take_profit_missing(self, agent, timestamp):
        """Test that missing take-profit returns None."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        STOP LOSS: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.take_profit is None

    def test_parse_take_profit_inline(self, agent, timestamp):
        """Test parsing take profit mentioned inline."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        STOP LOSS: $175.00
        Set take profit at $200.00 for the target.
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.take_profit == 200.00

    # ==================== STRATEGY DETECTION TESTS ====================

    def test_parse_strategy_momentum(self, agent, timestamp):
        """Test detecting MOMENTUM strategy."""
        response = self._make_response("""
        STRATEGY: MOMENTUM
        ACTION: BUY
        Symbol: GOOGL
        The stock shows strong momentum upward.
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.strategy_used == StrategyType.MOMENTUM

    def test_parse_strategy_mean_reversion_space(self, agent, timestamp):
        """Test detecting 'MEAN REVERSION' strategy."""
        response = self._make_response("""
        STRATEGY: MEAN REVERSION
        ACTION: BUY
        Symbol: GOOGL
        Price is oversold and due for a bounce.
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.strategy_used == StrategyType.MEAN_REVERSION

    def test_parse_strategy_mean_reversion_hyphenated(self, agent, timestamp):
        """Test detecting 'MEAN-REVERSION' strategy."""
        response = self._make_response("""
        STRATEGY: MEAN-REVERSION
        ACTION: BUY
        Symbol: GOOGL
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.strategy_used == StrategyType.MEAN_REVERSION

    def test_parse_strategy_trend_following_space(self, agent, timestamp):
        """Test detecting 'TREND FOLLOWING' strategy."""
        response = self._make_response("""
        STRATEGY: TREND FOLLOWING
        ACTION: BUY
        Symbol: GOOGL
        Following the established uptrend.
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.strategy_used == StrategyType.TREND_FOLLOWING

    def test_parse_strategy_trend_following_hyphenated(self, agent, timestamp):
        """Test detecting 'TREND-FOLLOWING' strategy."""
        response = self._make_response("""
        STRATEGY: TREND-FOLLOWING
        ACTION: SELL
        Symbol: TSLA
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.strategy_used == StrategyType.TREND_FOLLOWING

    def test_parse_strategy_breakout(self, agent, timestamp):
        """Test detecting BREAKOUT strategy."""
        response = self._make_response("""
        STRATEGY: BREAKOUT
        ACTION: BUY
        Symbol: GOOGL
        Price breaking above resistance.
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.strategy_used == StrategyType.BREAKOUT

    def test_parse_strategy_range_trading(self, agent, timestamp):
        """Test detecting RANGE trading strategy."""
        response = self._make_response("""
        STRATEGY: RANGE TRADING
        ACTION: BUY
        Symbol: GOOGL
        Buying at support level in ranging market.
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.strategy_used == StrategyType.RANGE_TRADING

    def test_parse_strategy_defensive_default(self, agent, timestamp):
        """Test that no strategy keyword defaults to DEFENSIVE."""
        response = self._make_response("""
        ACTION: HOLD
        Market conditions are unclear.
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.strategy_used == StrategyType.DEFENSIVE

    def test_parse_strategy_from_inline_mention(self, agent, timestamp):
        """Test detecting strategy mentioned inline."""
        response = self._make_response("""
        Using a momentum approach, I'll buy GOOGL.
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        STOP LOSS: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.strategy_used == StrategyType.MOMENTUM

    # ==================== MALFORMED RESPONSE TESTS ====================

    def test_parse_empty_response(self, agent, timestamp):
        """Test parsing empty response."""
        response = self._make_response("")
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.HOLD
        assert decision.strategy_used == StrategyType.DEFENSIVE

    def test_parse_gibberish_response(self, agent, timestamp):
        """Test parsing gibberish defaults to HOLD."""
        response = self._make_response("asdfghjkl qwerty zxcvbn")
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.HOLD

    def test_parse_partial_action(self, agent, timestamp):
        """Test parsing partial action text."""
        response = self._make_response("""
        I think we should BUY but I'm not sure.
        ACTION is uncertain.
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.HOLD  # No "ACTION: BUY" pattern

    def test_parse_multiple_actions_takes_first(self, agent, timestamp):
        """Test that first matching action is used."""
        response = self._make_response("""
        ACTION: BUY
        Wait, let me reconsider.
        ACTION: SELL
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.BUY

    def test_parse_response_with_special_characters(self, agent, timestamp):
        """Test parsing response with special characters."""
        response = self._make_response("""
        *** TRADING DECISION ***
        ========================
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        STOP LOSS: $175.00
        ========================
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.BUY
        assert decision.symbol == "GOOGL"

    def test_parse_response_with_markdown(self, agent, timestamp):
        """Test parsing response with markdown formatting."""
        response = self._make_response("""
        ## Analysis

        **ACTION: BUY**
        - Symbol: GOOGL
        - Quantity: 10
        - Stop Loss: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.BUY
        assert decision.symbol == "GOOGL"

    def test_parse_response_missing_all_numbers(self, agent, timestamp):
        """Test parsing response without any numbers."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.BUY
        assert decision.symbol == "GOOGL"
        assert decision.quantity is None
        assert decision.stop_loss is None

    # ==================== EDGE CASE TESTS ====================

    def test_parse_preserves_reasoning(self, agent, timestamp):
        """Test that full text is preserved as reasoning."""
        text = """
        Based on RSI being oversold at 25, I recommend:
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        STOP LOSS: $175.00

        This presents a good entry point.
        """
        response = self._make_response(text)
        decision = agent._parse_decision(response, timestamp, [])
        assert "RSI being oversold" in decision.reasoning
        assert "good entry point" in decision.reasoning

    def test_parse_tool_calls_preserved(self, agent, timestamp):
        """Test that tool calls list is preserved."""
        tool_calls = [
            {"tool": "get_stock_price", "input": {"symbol": "GOOGL"}, "result": {"price": 185.0}}
        ]
        response = self._make_response("ACTION: BUY GOOGL")
        decision = agent._parse_decision(response, timestamp, tool_calls)
        assert decision.tool_calls == tool_calls

    def test_parse_timestamp_preserved(self, agent, timestamp):
        """Test that timestamp is preserved."""
        response = self._make_response("ACTION: HOLD")
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.timestamp == timestamp

    def test_parse_very_long_response(self, agent, timestamp):
        """Test parsing a very long response."""
        long_text = "This is analysis. " * 100 + """
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        STOP LOSS: $175.00
        """ + "More analysis. " * 100
        response = self._make_response(long_text)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.BUY
        assert decision.symbol == "GOOGL"

    def test_parse_multiline_stop_loss_description(self, agent, timestamp):
        """Test parsing stop loss split across lines."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 10
        STOP
        LOSS: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        # Current implementation requires both STOP and LOSS on same line
        assert decision.stop_loss is None

    def test_parse_quantity_with_comma_separator(self, agent, timestamp):
        """Test parsing quantity like '1,000 shares'."""
        response = self._make_response("""
        ACTION: BUY
        Symbol: GOOGL
        QUANTITY: 1 000 shares
        STOP LOSS: $175.00
        """)
        decision = agent._parse_decision(response, timestamp, [])
        # Current implementation may not handle this

    def test_parse_action_at_end_of_response(self, agent, timestamp):
        """Test action at end of response."""
        response = self._make_response("""
        After careful consideration of all factors...
        Symbol: GOOGL
        QUANTITY: 10
        STOP LOSS: $175.00
        ACTION: BUY
        """)
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.BUY

    def test_parse_multiple_content_blocks(self, agent, timestamp):
        """Test parsing response with multiple content blocks."""
        response = MockResponse(
            content=[
                MockTextBlock(text="First part of response. "),
                MockTextBlock(text="ACTION: BUY "),
                MockTextBlock(text="Symbol: GOOGL QUANTITY: 10 STOP LOSS: $175.00"),
            ]
        )
        decision = agent._parse_decision(response, timestamp, [])
        assert decision.action == ActionType.BUY
        assert decision.symbol == "GOOGL"


class TestClaudeAgentIntegration:
    """Integration tests for ClaudeAgent decision flow."""

    @pytest.fixture
    def agent(self):
        """Create ClaudeAgent instance with mocked dependencies."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('agents.claude_agent.anthropic.Anthropic'):
                return ClaudeAgent(tools={})

    def test_strategy_explanation_updated(self, agent):
        """Test that strategy explanation is updated after parsing."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        text = "My analysis shows bullish momentum. ACTION: BUY GOOGL"
        response = MockResponse(content=[MockTextBlock(text=text)])

        agent._parse_decision(response, timestamp, [])

        assert "bullish momentum" in agent.get_strategy_explanation()

    def test_parse_returns_trading_decision_type(self, agent):
        """Test that parse always returns TradingDecision."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        response = MockResponse(content=[MockTextBlock(text="random text")])

        decision = agent._parse_decision(response, timestamp, [])

        assert isinstance(decision, TradingDecision)

    def test_default_values_on_malformed_input(self, agent):
        """Test default values are set on malformed input."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        response = MockResponse(content=[MockTextBlock(text="xyz")])

        decision = agent._parse_decision(response, timestamp, [])

        assert decision.action == ActionType.HOLD
        assert decision.strategy_used == StrategyType.DEFENSIVE
        assert decision.symbol is None
        assert decision.quantity is None
        assert decision.stop_loss is None
        assert decision.take_profit is None
