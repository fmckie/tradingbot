#!/usr/bin/env python3
"""
Diagnostic script for Grok (xAI) API connectivity.

Run this to identify why GrokAgent may be failing silently.

Usage:
    python test_grok_api.py
"""

import asyncio
import os
import json
from dotenv import load_dotenv
import httpx

load_dotenv()


class GrokAPIDiagnostics:
    """Comprehensive diagnostics for xAI Grok API."""

    def __init__(self):
        self.base_url = "https://api.x.ai/v1"
        self.api_key = os.getenv("XAI_API_KEY")
        self.results = []

    def log(self, status: str, message: str, details: str = ""):
        """Log a diagnostic result."""
        icon = {"PASS": "\u2713", "FAIL": "\u2717", "WARN": "!", "INFO": "-"}[status]
        self.results.append((status, message))
        print(f"  [{icon}] {message}")
        if details:
            # Print full details indented
            for line in details.split("\n"):
                print(f"      {line}")

    def check_api_key_exists(self) -> bool:
        """Check if XAI_API_KEY environment variable exists."""
        print("\n1. Checking API Key Environment Variable...")

        if not self.api_key:
            self.log("FAIL", "XAI_API_KEY not found in environment")
            self.log("INFO", "Set it in .env file: XAI_API_KEY=xai-...")
            return False

        self.log("PASS", f"XAI_API_KEY found (length: {len(self.api_key)})")
        return True

    def check_api_key_format(self) -> bool:
        """Validate API key format."""
        print("\n2. Validating API Key Format...")

        if not self.api_key:
            self.log("FAIL", "No API key to validate")
            return False

        # Check prefix
        if self.api_key.startswith("xai-"):
            self.log("PASS", "API key has expected 'xai-' prefix")
        else:
            self.log("WARN", f"API key prefix: '{self.api_key[:4]}...' (expected 'xai-')")
            self.log("INFO", "Key may still work, but format is unusual")

        # Check length (typical xAI keys are ~50+ chars)
        if len(self.api_key) < 20:
            self.log("WARN", f"API key seems short ({len(self.api_key)} chars)")
            return False
        else:
            self.log("PASS", f"API key length looks reasonable ({len(self.api_key)} chars)")

        return True

    async def test_basic_connectivity(self) -> bool:
        """Test basic connection to xAI API."""
        print("\n3. Testing Basic API Connectivity...")

        if not self.api_key:
            self.log("FAIL", "Cannot test - no API key")
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=30.0,
                )

                if response.status_code == 200:
                    self.log("PASS", "Successfully connected to xAI API")
                    models = response.json()
                    if "data" in models:
                        model_ids = [m.get("id", "unknown") for m in models.get("data", [])]
                        self.log("INFO", f"Available models: {model_ids}")
                    return True
                elif response.status_code == 401:
                    self.log("FAIL", "401 Unauthorized - API key is invalid or expired")
                    self.log("INFO", f"Response: {response.text[:500]}")
                    return False
                elif response.status_code == 403:
                    self.log("FAIL", "403 Forbidden - API key lacks permissions")
                    self.log("INFO", f"Response: {response.text[:500]}")
                    return False
                else:
                    self.log("FAIL", f"Unexpected status: {response.status_code}")
                    self.log("INFO", f"Response: {response.text[:500]}")
                    return False

        except httpx.TimeoutException:
            self.log("FAIL", "Connection timed out after 30s")
            return False
        except httpx.ConnectError as e:
            self.log("FAIL", f"Connection failed: {e}")
            return False
        except Exception as e:
            self.log("FAIL", f"Unexpected error: {type(e).__name__}: {e}")
            return False

    async def test_chat_completion(self, model: str = "grok-4") -> bool:
        """Test chat completion endpoint."""
        print(f"\n4. Testing Chat Completion with model '{model}'...")

        if not self.api_key:
            self.log("FAIL", "Cannot test - no API key")
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "user", "content": "Say 'hello' and nothing else."}
                        ],
                        "max_tokens": 10,
                    },
                    timeout=60.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    self.log("PASS", f"Chat completion successful with {model}")
                    self.log("INFO", f"Response: '{content}'")
                    return True
                elif response.status_code == 400:
                    error = response.json()
                    error_msg = error.get("error", {}).get("message", response.text[:500])
                    self.log("FAIL", f"400 Bad Request - likely invalid model name")
                    self.log("INFO", f"Error: {error_msg}")
                    return False
                elif response.status_code == 401:
                    self.log("FAIL", "401 Unauthorized")
                    self.log("INFO", f"Response: {response.text[:500]}")
                    return False
                elif response.status_code == 429:
                    self.log("WARN", "429 Rate Limited - API key works but hit rate limit")
                    self.log("INFO", f"Response: {response.text[:500]}")
                    return True  # Key works, just rate limited
                else:
                    self.log("FAIL", f"Status {response.status_code}")
                    self.log("INFO", f"Full response: {response.text}")
                    return False

        except httpx.TimeoutException:
            self.log("FAIL", "Request timed out after 60s")
            return False
        except Exception as e:
            self.log("FAIL", f"Error: {type(e).__name__}: {e}")
            return False

    async def test_tool_calling(self, model: str = "grok-4") -> bool:
        """Test tool/function calling capability."""
        print(f"\n5. Testing Tool Calling with model '{model}'...")

        if not self.api_key:
            self.log("FAIL", "Cannot test - no API key")
            return False

        # Simple test tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "description": "Get the current price of a stock",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock ticker symbol"
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            }
        ]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "user", "content": "What is the current price of GOOGL? Use the get_stock_price tool."}
                        ],
                        "tools": tools,
                        "tool_choice": "auto",
                        "max_tokens": 100,
                    },
                    timeout=60.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    message = result.get("choices", [{}])[0].get("message", {})
                    tool_calls = message.get("tool_calls", [])

                    if tool_calls:
                        self.log("PASS", f"Tool calling works with {model}")
                        for tc in tool_calls:
                            self.log("INFO", f"Tool called: {tc.get('function', {}).get('name')}")
                            self.log("INFO", f"Arguments: {tc.get('function', {}).get('arguments')}")
                        return True
                    else:
                        self.log("WARN", "API responded but didn't use the tool")
                        self.log("INFO", f"Response content: {message.get('content', 'empty')[:200]}")
                        return True  # API works, just didn't use tool
                else:
                    self.log("FAIL", f"Status {response.status_code}")
                    self.log("INFO", f"Response: {response.text[:500]}")
                    return False

        except Exception as e:
            self.log("FAIL", f"Error: {type(e).__name__}: {e}")
            return False

    async def test_alternative_models(self):
        """Try alternative model names if primary fails."""
        print("\n6. Testing Alternative Model Names...")

        models_to_try = ["grok-3", "grok-2-latest", "grok-2", "grok-beta"]

        for model in models_to_try:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": "Hi"}],
                            "max_tokens": 5,
                        },
                        timeout=30.0,
                    )

                    if response.status_code == 200:
                        self.log("PASS", f"Model '{model}' works!")
                    elif response.status_code == 429:
                        self.log("WARN", f"Model '{model}' works but rate limited")
                    else:
                        self.log("INFO", f"Model '{model}' returned {response.status_code}")

            except Exception as e:
                self.log("INFO", f"Model '{model}' failed: {e}")

    async def run_all_diagnostics(self):
        """Run complete diagnostic suite."""
        print("=" * 60)
        print("        GROK API DIAGNOSTIC SUITE")
        print("=" * 60)
        print(f"Base URL: {self.base_url}")

        # Step 1: Check API key exists
        if not self.check_api_key_exists():
            print("\n" + "=" * 60)
            print("DIAGNOSIS: Missing API key")
            print("ACTION: Add XAI_API_KEY to your .env file")
            print("=" * 60)
            return

        # Step 2: Validate key format
        self.check_api_key_format()

        # Step 3: Test connectivity
        connected = await self.test_basic_connectivity()

        if not connected:
            print("\n" + "=" * 60)
            print("DIAGNOSIS: Cannot connect to xAI API")
            print("POSSIBLE CAUSES:")
            print("  - Invalid or expired API key")
            print("  - Network issues")
            print("  - xAI API service down")
            print("ACTION: Verify your API key at https://console.x.ai")
            print("=" * 60)
            return

        # Step 4: Test chat completion with grok-4
        chat_works = await self.test_chat_completion("grok-4")

        if not chat_works:
            # Try alternative models
            await self.test_alternative_models()

        # Step 5: Test tool calling (only if chat works)
        if chat_works:
            await self.test_tool_calling("grok-4")

        # Summary
        print("\n" + "=" * 60)
        print("        DIAGNOSTIC SUMMARY")
        print("=" * 60)

        passes = sum(1 for status, _ in self.results if status == "PASS")
        fails = sum(1 for status, _ in self.results if status == "FAIL")
        warns = sum(1 for status, _ in self.results if status == "WARN")

        print(f"  Passed: {passes}")
        print(f"  Failed: {fails}")
        print(f"  Warnings: {warns}")

        if fails == 0:
            print("\n  STATUS: Grok API appears to be working correctly!")
            print("  If GrokAgent still fails, check:")
            print("    - Tool schemas match what Grok expects")
            print("    - Response parsing in _parse_decision()")
            print("    - Timeout issues under load")
        else:
            print("\n  STATUS: Issues detected - see failures above")

        print("=" * 60)


async def main():
    diagnostics = GrokAPIDiagnostics()
    await diagnostics.run_all_diagnostics()


if __name__ == "__main__":
    asyncio.run(main())
