# AI Trading Bot Competition

An automated trading competition pitting **Claude Sonnet 4.6** against **Grok 4.3** on GOOGL and TSLA stocks. Each AI agent makes independent hourly trading decisions during market hours, with integrated risk management, performance tracking, and a learning system.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TRADING COMPETITION                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐              ┌─────────────┐              │
│   │   CLAUDE    │              │    GROK     │              │
│   │ Sonnet 4.6  │              │     4.3     │              │
│   └──────┬──────┘              └──────┬──────┘              │
│          │                            │                      │
│          ▼                            ▼                      │
│   ┌─────────────┐              ┌─────────────┐              │
│   │   Alpaca    │              │   Alpaca    │              │
│   │  Account 1  │              │  Account 2  │              │
│   │  $100,000   │              │  $100,000   │              │
│   └─────────────┘              └─────────────┘              │
│                                                              │
│   Symbols: GOOGL, TSLA    │    Frequency: Hourly            │
│   Risk Limit: 2%/trade    │    Max Exposure: 50%            │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Dual AI Agents**: Claude Sonnet 4.6 and Grok 4.3 make independent trading decisions
- **Paper Trading**: Uses Alpaca paper trading accounts (no real money)
- **Risk Management**: Hard-coded limits enforced by the system
- **Technical Analysis**: RSI, MACD, Bollinger Bands, EMA, VWAP, ATR
- **Learning System**: PostgreSQL-backed memory for trade reflections
- **Real-time Scoreboard**: Track performance, win rates, and P&L
- **Cloud Deployment**: Modal-ready for scheduled execution

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/fmckie/tradingbot.git
cd tradingbot
pip install -r requirements.txt
```

#### Reproducible install (lockfile)

`requirements.txt` / `requirements-dev.txt` use `>=` floors and stay the
human-edited source of truth. For a byte-for-byte reproducible environment, the
repo also ships a committed, fully-pinned lockfile with hashes,
`requirements.lock`, covering both production and dev/test dependencies:

```bash
pip install --require-hashes -r requirements.lock
```

The lockfile is generated with [pip-tools](https://pip-tools.readthedocs.io/).
Regenerate it whenever you change the floors in the requirements files:

```bash
pip install pip-tools
pip-compile --generate-hashes --allow-unsafe \
  --output-file=requirements.lock requirements.txt requirements-dev.txt
```

### 2. Configure Environment

```bash
cp .env.template .env
```

Edit `.env` with your API keys:

```env
# Alpaca Paper Trading (create two separate accounts)
CLAUDE_ALPACA_API_KEY=your_claude_alpaca_key
CLAUDE_ALPACA_SECRET_KEY=your_claude_alpaca_secret
GROK_ALPACA_API_KEY=your_grok_alpaca_key
GROK_ALPACA_SECRET_KEY=your_grok_alpaca_secret

# AI APIs
ANTHROPIC_API_KEY=your_anthropic_key
XAI_API_KEY=your_xai_key

# Learning System (optional - Neon PostgreSQL)
DATABASE_URL=postgres://user:pass@host/db
```

### 3. Verify Setup

```bash
python test_setup.py
```

### 4. Run

```bash
# Local continuous mode
python main.py

# Single cycle (for testing)
python main.py --single-cycle --skip-market-check

# Cloud deployment (Modal)
modal run modal_app.py
```

## Architecture

```
tradingbot/
├── agents/                 # AI agent implementations
│   ├── base_agent.py      # Base class, dataclasses
│   ├── claude_agent.py    # Claude Sonnet 4.6 agent
│   └── grok_agent.py      # Grok API agent
├── config/                 # Configuration
│   ├── settings.py        # Risk limits, symbols, hours
│   └── alpaca_config.py   # API client setup
├── data/                   # Market data pipeline
│   ├── market_data.py     # OHLCV fetching via Alpaca
│   └── indicators.py      # Technical indicators
├── database/               # Learning system
│   ├── postgres_client.py # PostgreSQL connection
│   ├── learning_store.py  # Episode/Reflection storage
│   └── schema.sql         # Database schema
├── execution/              # Trade execution
│   └── order_executor.py  # Order placement
├── risk/                   # Risk management
│   └── risk_manager.py    # Hard limit enforcement
├── monitoring/             # Logging and tracking
│   ├── logger.py          # Trade event logging
│   ├── scoreboard.py      # Performance metrics (console)
│   ├── dashboard.py       # Read-only web monitoring dashboard (stdlib)
│   └── dashboard.html     # nof1-style live dashboard page
├── tools/                  # AI-exposed tools
│   ├── market_tools.py    # Price/quote tools
│   ├── trading_tools.py   # Order management
│   └── analysis_tools.py  # Technical analysis
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── main.py                 # Competition orchestrator
└── modal_app.py            # Modal cloud deployment
```

## Risk Management

The system enforces hard limits that AI agents cannot override:

| Limit | Value | Description |
|-------|-------|-------------|
| Max Risk/Trade | 2% | Maximum loss per trade |
| Max Exposure | 50% | Maximum capital deployed |
| Max Positions | 2 | One per symbol (GOOGL, TSLA) |
| Daily Loss Limit | 5% | Trading halts if exceeded |
| Stop-Loss | Required | 0.5% - 5% from entry |

## Trading Hours

- **Market**: 9:30 AM - 4:00 PM Eastern
- **Buffer**: No trading in first/last 15 minutes
- **Frequency**: Hourly decisions

## Learning System

The bot includes an optional PostgreSQL-backed learning system:

```
Episode → Outcome → Reflection → Learning
   │         │          │           │
   │         │          │           └── Distilled patterns
   │         │          └── What worked/failed analysis
   │         └── Win/Loss/Breakeven
   └── Market context + Decision made
```

Enable by setting `DATABASE_URL` in `.env`. Uses Neon serverless PostgreSQL.

## Testing

```bash
# Run all tests
pytest tests/ -q

# With coverage
pytest tests/ --cov=agents --cov=risk --cov=execution --cov-report=term-missing

# Specific module
pytest tests/unit/test_risk_manager.py -v
```

## Command Line Options

```bash
python main.py [options]

Options:
  --single-cycle        Run one trading cycle and exit
  --skip-market-check   Ignore market hours (for testing)
```

## Performance Tracking

The scoreboard tracks:
- Total P&L and daily P&L
- Win rate and trade count
- Max drawdown
- Strategy usage
- Position values

View real-time updates in the console during operation.

## Web Monitoring Dashboard

A zero-dependency web dashboard renders the same competition telemetry as a
live, nof1-style page (monospace cards, risk-control panel, decision/order log,
KPI strip). It opens the SQLite log **read-only**, so it is safe to run
alongside a live `python main.py` session without blocking writes.

```bash
# Serve at http://127.0.0.1:8787 (polls itself every few seconds)
python -m monitoring.dashboard

# Custom host/port/database
python -m monitoring.dashboard --host 0.0.0.0 --port 9000 --db trading_competition.sqlite

# Print the JSON snapshot and exit (no server) — handy for scripting
python -m monitoring.dashboard --once
```

Themes: append `?theme=dark` (default) or `?theme=light` to the URL. Routes:
`GET /` (page), `GET /api/state` (JSON snapshot), `GET /healthz`.

## API Requirements

### Alpaca
- Paper trading accounts (free)
- Separate accounts for each agent recommended
- IEX data feed (included with paper trading)

### Anthropic
- Claude Sonnet 4.6 access
- API key with sufficient credits

### xAI
- Grok API access
- API key with sufficient credits

## License

MIT

## Disclaimer

This is a paper trading competition for educational purposes. No real money is involved. Past performance does not guarantee future results. This is not financial advice.
