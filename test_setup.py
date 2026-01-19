#!/usr/bin/env python3
"""Quick test to verify the trading bot setup."""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("AI TRADING COMPETITION - SETUP TEST")
print("=" * 60)

# Check environment variables
required_vars = {
    "CLAUDE_ALPACA_API_KEY": "Alpaca (Claude account)",
    "CLAUDE_ALPACA_SECRET_KEY": "Alpaca (Claude account)",
    "GROK_ALPACA_API_KEY": "Alpaca (Grok account)",
    "GROK_ALPACA_SECRET_KEY": "Alpaca (Grok account)",
    "ANTHROPIC_API_KEY": "Anthropic (Claude AI)",
    "XAI_API_KEY": "xAI (Grok AI)",
}

print("\n1. CHECKING ENVIRONMENT VARIABLES")
print("-" * 40)
missing = []
for var, desc in required_vars.items():
    value = os.getenv(var)
    if value:
        # Mask the key for security
        masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
        print(f"  ✓ {var}: {masked}")
    else:
        print(f"  ✗ {var}: MISSING ({desc})")
        missing.append(var)

if missing:
    print(f"\n  ⚠ Missing {len(missing)} required variables!")
    print("  Copy .env.template to .env and fill in your keys.")
else:
    print("\n  ✓ All environment variables configured!")

# Test imports
print("\n2. TESTING MODULE IMPORTS")
print("-" * 40)

imports = [
    ("config.alpaca_config", "Alpaca configuration"),
    ("config.settings", "Risk settings"),
    ("data.market_data", "Market data provider"),
    ("data.indicators", "Technical indicators"),
    ("tools.market_tools", "Market tools"),
    ("tools.trading_tools", "Trading tools"),
    ("agents.base_agent", "Base agent"),
    ("agents.claude_agent", "Claude agent"),
    ("agents.grok_agent", "Grok agent"),
    ("risk.risk_manager", "Risk manager"),
    ("execution.order_executor", "Order executor"),
    ("monitoring.scoreboard", "Scoreboard"),
    ("monitoring.logger", "Trade logger"),
]

import_errors = []
for module, desc in imports:
    try:
        __import__(module)
        print(f"  ✓ {module}")
    except Exception as e:
        print(f"  ✗ {module}: {e}")
        import_errors.append((module, str(e)))

if import_errors:
    print(f"\n  ⚠ {len(import_errors)} import errors!")
else:
    print("\n  ✓ All modules imported successfully!")

# Test Alpaca connection if keys are present
print("\n3. TESTING ALPACA CONNECTION")
print("-" * 40)

if not missing or all(v not in missing for v in ["CLAUDE_ALPACA_API_KEY", "CLAUDE_ALPACA_SECRET_KEY"]):
    try:
        from config.alpaca_config import get_claude_client
        trading_client, data_client = get_claude_client()
        account = trading_client.get_account()
        print(f"  ✓ Claude account connected!")
        print(f"    Equity: ${float(account.equity):,.2f}")
        print(f"    Cash: ${float(account.cash):,.2f}")
        print(f"    Buying Power: ${float(account.buying_power):,.2f}")
    except Exception as e:
        print(f"  ✗ Claude account failed: {e}")
else:
    print("  ⊘ Skipped (missing Alpaca keys)")

if not missing or all(v not in missing for v in ["GROK_ALPACA_API_KEY", "GROK_ALPACA_SECRET_KEY"]):
    try:
        from config.alpaca_config import get_grok_client
        trading_client, data_client = get_grok_client()
        account = trading_client.get_account()
        print(f"  ✓ Grok account connected!")
        print(f"    Equity: ${float(account.equity):,.2f}")
        print(f"    Cash: ${float(account.cash):,.2f}")
        print(f"    Buying Power: ${float(account.buying_power):,.2f}")
    except Exception as e:
        print(f"  ✗ Grok account failed: {e}")
else:
    print("  ⊘ Skipped (missing Alpaca keys)")

# Summary
print("\n" + "=" * 60)
if not missing and not import_errors:
    print("✓ ALL TESTS PASSED - Ready to run competition!")
    print("\nStart with: python main.py")
else:
    print("⚠ SETUP INCOMPLETE")
    if missing:
        print(f"  - Configure {len(missing)} missing environment variables")
    if import_errors:
        print(f"  - Fix {len(import_errors)} import errors")
print("=" * 60)
