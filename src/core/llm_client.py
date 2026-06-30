# =============================================================================
# llm_client.py — Unified LLM caller for both Groq and OpenAI
# Every LLM call in the project goes through call_llm().
# Switching providers is a one-line change in config.py.
# =============================================================================

import os
import time
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from src.core.config import DEV_MODEL

load_dotenv()

# Initialise clients once at module level
_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    for attempt in range(max_retries):
        try:
            if _is_groq_model(model):
                response = _groq_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            else:
                response = _openai_client.chat.completions.create(
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