# =============================================================================
# llm_client.py — Unified LLM caller for both Groq and OpenAI
# Every LLM call in the project goes through call_llm().
# Switching providers is a one-line change in config.py.
#
# CLIENTS ARE CREATED ON FIRST USE, NOT AT IMPORT.
# Both SDKs raise immediately if their API key is absent, so constructing the
# clients at module level made a missing key an IMPORT-time failure. Because
# this module is imported by every agent, every system and the judge, that meant
# a checkout holding only one provider's key could not run anything at all —
# including tests that make no LLM call, such as the D5 prompt-invariant test.
# Deferring construction moves the failure to the point of use, where it is
# actionable, and keeps everything that does not call an LLM runnable with no
# keys at all.
# =============================================================================

import os
import time
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from src.core.config import DEV_MODEL

load_dotenv()

# Created on first use by the accessors below, then cached for the process.
# Never construct these at import time (see module docstring).
_groq_client = None
_openai_client = None


def _get_groq_client() -> Groq:
    """Return the shared Groq client, constructing it on first use."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set, but a Groq model was requested. "
                "Either add GROQ_API_KEY to your .env file, or select an "
                "OpenAI model via DEV_MODEL / EVAL_MODEL in src/core/config.py."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _get_openai_client() -> OpenAI:
    """Return the shared OpenAI client, constructing it on first use."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set, but an OpenAI model was requested. "
                "Add OPENAI_API_KEY to your .env file."
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _is_groq_model(model: str) -> bool:
    """Detect whether a model name belongs to Groq or OpenAI."""
    groq_prefixes = ("llama", "mixtral", "gemma", "whisper", "openai/gpt-oss", "qwen")
    return any(model.lower().startswith(prefix) for prefix in groq_prefixes)

def call_llm(
    prompt: str,
    model: str = DEV_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.0
) -> dict:
    """
    Call an LLM and return the response text plus token counts.

    Args:
        prompt:     The prompt string to send
        model:      Model name (from config.py — DEV_MODEL, EVAL_MODEL, or JUDGE_MODEL)
        max_tokens: Maximum tokens in the response
        temperature: 0.0 = deterministic (best for structured YES/NO outputs)

    Returns:
        dict with keys:
            - text:          the model's response as a string
            - input_tokens:  number of input tokens used
            - output_tokens: number of output tokens used

    Raises:
        Exception if all 3 retry attempts fail
    """
    max_retries = 3
    wait_times = [1, 2, 4]  # exponential backoff in seconds

    # Resolved BEFORE the retry loop. A missing API key is a configuration
    # error, not a transient one — retrying it three times with backoff would
    # waste seven seconds and bury the message under two misleading
    # "Retrying..." lines. The retry loop is for network and rate-limit
    # failures only.
    client = _get_groq_client() if _is_groq_model(model) else _get_openai_client()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )

            return {
                "text": response.choices[0].message.content.strip(),
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_times[attempt]}s...")
                time.sleep(wait_times[attempt])
            else:
                raise Exception(
                    f"LLM call failed after {max_retries} attempts: {e}"
                )