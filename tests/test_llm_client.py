# =============================================================================
# test_llm_client.py — LLM client construction and configuration errors
#
# These tests make NO API calls and require NO API keys. That is the point:
# the clients used to be constructed at import time, which meant every module
# importing llm_client (every agent, every system, the judge) required BOTH
# providers' keys before it would even load. Deferring construction is what
# makes this file runnable at all.
#
# Run with: python -m pytest tests/test_llm_client.py -v
# =============================================================================

import os
import subprocess
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import llm_client


@pytest.fixture(autouse=True)
def _reset_client_cache():
    """
    Clear the cached clients around each test.

    The accessors memoise, so a test that populates the cache would otherwise
    hide a missing key from the next test.
    """
    llm_client._groq_client = None
    llm_client._openai_client = None
    yield
    llm_client._groq_client = None
    llm_client._openai_client = None


# --- THE REGRESSION TEST -----------------------------------------------------
# Importing llm_client must not construct either client. Both SDKs raise if
# their key is absent, so eager construction turned a missing key into an
# import-time failure across the whole project.
def test_importing_llm_client_does_not_require_api_keys():
    """
    Import llm_client in a subprocess with both API keys stripped from the
    environment. A subprocess is required because the module is already
    imported in this process, and because load_dotenv() would otherwise
    repopulate the keys from .env.

    If this fails, someone has reintroduced module-level client construction.
    """
    env = {k: v for k, v in os.environ.items()
           if k not in ("GROQ_API_KEY", "OPENAI_API_KEY")}
    # Neutralise .env: load_dotenv() reads from the working directory upwards,
    # so run somewhere it cannot find the project's file.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = repo_root

    result = subprocess.run(
        [sys.executable, "-c", "from src.core import llm_client; print('OK')"],
        capture_output=True, text=True, env=env, cwd="/",
    )

    assert result.returncode == 0, (
        "Importing llm_client failed without API keys. Clients are being "
        "constructed at import time again; construct them on first use "
        f"instead.\n\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


# --- Configuration errors are clear and immediate ----------------------------

def test_missing_groq_key_raises_a_named_error(monkeypatch):
    """A missing key must name the variable, not surface as an SDK error."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm_client._get_groq_client()


def test_missing_openai_key_raises_a_named_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm_client._get_openai_client()


def test_missing_key_is_not_retried(monkeypatch):
    """
    A missing key is a configuration error, not a transient one. It must
    propagate immediately rather than being swallowed by the retry loop, which
    would spend seven seconds on backoff and print two misleading "Retrying..."
    lines before failing.
    """
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm_client.call_llm("hello", model="llama-3.1-8b-instant")


# --- Provider routing (pure string logic, no network) ------------------------

@pytest.mark.parametrize("model", [
    "llama-3.1-8b-instant", "mixtral-8x7b-32768",
    "gemma2-9b-it", "qwen-2.5-32b", "openai/gpt-oss-120b",
])
def test_groq_models_route_to_groq(model):
    assert llm_client._is_groq_model(model) is True


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4o", "o3-mini"])
def test_openai_models_route_to_openai(model):
    assert llm_client._is_groq_model(model) is False