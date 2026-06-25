# AI Trading Bot

AI trading competition pitting Claude Sonnet 4.6 vs Grok 4.3 on GOOGL and TSLA. Runs hourly trading decisions during market hours with integrated risk management, performance tracking, and a learning system.

## Project Structure

```
tradingbot/
├── agents/              # AI agent implementations
│   ├── base_agent.py   # Base class, MarketContext, TradingDecision dataclasses
│   ├── claude_agent.py # Claude Sonnet 4.6 trading agent
│   └── grok_agent.py   # Grok API trading agent
├── config/              # Configuration
│   ├── settings.py     # Risk limits, trading hours, symbols
│   └── alpaca_config.py # API clients (Claude, Grok, Alpaca)
├── data/                # Market data pipeline
│   ├── market_data.py  # OHLCV data fetching
│   └── indicators.py   # Technical indicators (RSI, MACD, etc.)
├── database/            # Learning system persistence
│   ├── postgres_client.py # PostgreSQL connection
│   ├── learning_store.py  # Trade learnings storage
│   └── schema.sql      # Database schema
├── execution/           # Trade execution
│   └── order_executor.py # Order placement via Alpaca
├── risk/                # Risk management
│   └── risk_manager.py # Hard position limits enforcement
├── monitoring/          # Logging and tracking
│   ├── logger.py       # Trade event logging
│   ├── scoreboard.py   # Performance metrics (console)
│   ├── dashboard.py    # Read-only web monitoring dashboard (stdlib http.server)
│   └── dashboard.html  # nof1-style live dashboard page
├── reports/             # Competition reports
├── tools/               # AI-exposed tools
│   ├── market_tools.py
│   ├── trading_tools.py
│   └── analysis_tools.py
├── main.py              # Competition orchestrator (hourly decisions)
├── modal_app.py         # Modal cloud deployment
└── test_*.py            # Test files
```

## Organization Rules

**Keep code organized and modularized:**
- Agent implementations → `/agents`, one agent per file
- Configuration → `/config`, settings vs API clients separated
- Data pipeline → `/data`, fetching vs indicators separated
- Database operations → `/database`, client vs store logic separated
- Each module has single responsibility

**Key constraints:**
- Risk limits enforced in `/risk/risk_manager.py` (2% max risk, 50% max exposure)
- Max 1 position per symbol, 2 total positions
- Trading hours: Market hours only (Eastern Time)

## Code Quality

After editing ANY Python file, run:

```bash
python -m py_compile <file>  # Syntax check
python test_setup.py         # Verify configuration
```

For full system validation:
```bash
python test_learning_system.py  # Database integration
python test_simulation.py       # Trading simulation
```

**Before committing:** Ensure all test files pass without errors.

## Running the Bot

Local execution:
```bash
python main.py
```

Modal cloud deployment:
```bash
modal run modal_app.py
```

## Monitoring

Console scoreboard prints automatically during a run. For a live web view, run
the read-only dashboard (stdlib only, opens SQLite with `mode=ro` so it never
blocks `main.py`):
```bash
python -m monitoring.dashboard            # http://127.0.0.1:8787
python -m monitoring.dashboard --once     # print JSON snapshot and exit
```
Serves `GET /` (page) and `GET /api/state` (JSON from `TradeLogger` data +
`RISK_LIMITS`); `?theme=dark|light`.

## Environment Setup

Copy `.env.template` to `.env` and configure:
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` - Alpaca paper trading
- `ANTHROPIC_API_KEY` - Claude API
- `XAI_API_KEY` - Grok API
- `DATABASE_URL` - PostgreSQL (Neon) for learning system
