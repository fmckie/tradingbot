-- Trading Bot Learning System PostgreSQL Schema
-- Designed for Neon (serverless PostgreSQL)

-- 1. Episodes: Each decision cycle
CREATE TABLE IF NOT EXISTS episodes (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    market_regime VARCHAR(50),        -- bullish/bearish/mixed/volatile
    symbols_context JSONB,            -- snapshot of GOOGL/TSLA data
    account_state JSONB,              -- equity, cash, positions
    decision_made JSONB,              -- action, symbol, qty, strategy
    outcome_pnl DECIMAL(12,2),        -- realized P&L from this decision
    outcome_status VARCHAR(20),       -- win/loss/hold/pending
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Reflections: Post-outcome analysis
CREATE TABLE IF NOT EXISTS reflections (
    id SERIAL PRIMARY KEY,
    episode_id INT REFERENCES episodes(id) ON DELETE CASCADE,
    agent_name VARCHAR(50) NOT NULL,
    what_worked TEXT,                 -- "RSI oversold signal was accurate"
    what_failed TEXT,                 -- "Entered too early before confirmation"
    lesson_learned TEXT,              -- "Wait for RSI to cross back above 30"
    next_time_will TEXT,              -- "Use RSI + MACD confirmation together"
    confidence_adjustment DECIMAL(3,2), -- +0.1 or -0.2 etc
    tags TEXT[],                      -- ['RSI', 'TSLA', 'MOMENTUM', 'OVERSOLD']
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Learnings: Distilled rules/patterns
CREATE TABLE IF NOT EXISTS learnings (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(50) NOT NULL,
    category VARCHAR(50),             -- strategy/indicator/timing/risk
    pattern TEXT NOT NULL,            -- "When RSI < 25 and MACD crossing up"
    insight TEXT NOT NULL,            -- "High probability reversal, but wait for confirmation"
    success_count INT DEFAULT 0,
    failure_count INT DEFAULT 0,
    last_validated TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    tags TEXT[],                      -- Tag-based search (no embeddings - cost efficient)
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Competition scoreboard (denormalized for quick queries)
CREATE TABLE IF NOT EXISTS competition_scores (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    starting_equity DECIMAL(12,2),
    ending_equity DECIMAL(12,2),
    daily_pnl DECIMAL(12,2),
    trades_count INT,
    wins INT,
    losses INT,
    strategies_used JSONB,
    top_learning_id INT REFERENCES learnings(id),
    UNIQUE(agent_name, date)
);

-- Indexes for fast retrieval
CREATE INDEX IF NOT EXISTS idx_episodes_agent_time ON episodes(agent_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_regime ON episodes(market_regime);
CREATE INDEX IF NOT EXISTS idx_episodes_outcome ON episodes(outcome_status);
CREATE INDEX IF NOT EXISTS idx_reflections_episode ON reflections(episode_id);
CREATE INDEX IF NOT EXISTS idx_reflections_agent ON reflections(agent_name);
CREATE INDEX IF NOT EXISTS idx_reflections_tags ON reflections USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_learnings_agent_active ON learnings(agent_name, is_active);
CREATE INDEX IF NOT EXISTS idx_learnings_category ON learnings(category);
CREATE INDEX IF NOT EXISTS idx_learnings_tags ON learnings USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_scores_agent_date ON competition_scores(agent_name, date DESC);
