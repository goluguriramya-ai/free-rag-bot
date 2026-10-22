"""
llm.py - Any-model support for the RAG bot.

Providers:
- ollama     : local, free, no API key  -> http://localhost:11434/api/chat
- groq       : free tier API            -> OpenAI-compatible
- gemini     : free tier API            -> Google generateContent
- openrouter : free models available    -> OpenAI-compatible

Add a new model by editing PROVIDERS below - no other code changes needed.
Add a new provider by adding an entry + a branch in chat().
"""

from __future__ import annotations

import os
import requests

PROVIDERS: dict = {
    "ollama": {
        "label": "Ollama (Local, Free)",
        "requires_key": False,
        "base_url": "http://localhost:11434",
        "models": [
            "llama3.1",
            "llama3.1:8b",
            "mistral",
            "gemma2",
            "phi3",
        ],
    },
    "groq": {
        "label": "Groq (Free Tier)",
        "requires_key": True,
        "env_var": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "models": [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "qwen/qwen3-32b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ],
    },
    "gemini": {
        "label": "Google Gemini (Free Tier)",
        "requires_key": True,
        "env_var": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "models": [
            "gemini-3.8-flash",
            "gemini-3.5-flash-lite",
        ],
    },
    "openrouter": {
        "label": "OpenRouter (Free Models)",
        "requires_key": True,
        "env_var": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "models": [
            "meta-llama/llama-3.1-8b-instruct:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemma-2-9b-it:free",
            "mistralai/mistral-7b-instruct:free",
        ],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_api_key(provider: str, api_key: str | None) -> str | None:
    if api_key and api_key.strip():
        return api_key.strip()
    env_var = PROVIDERS.get(provider, {}).get("env_var")
    if env_var:
        return os.getenv(env_var) or None
    return None


def _openai_compatible_chat(url: str, model: str, prompt: str, api_key: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # OpenRouter recommends these, harmless elsewhere if ignored by caller;
    # only add for openrouter URL
    if "openrouter" in url:
        headers["HTTP-Referer"] = "http://localhost:8501"
        headers["X-Title"] = "RAG Bot - Free Tools"

    resp = requests.post(
        url,
        headers=headers,
        json={"model": model, "messages": messages, "temperature": 0.3},
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _ollama_chat(model: str, prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to Ollama at http://localhost:11434. "
            "Is Ollama running? Install from https://ollama.com, run `ollama serve`, "
            f"and pull the model first e.g. `ollama pull {model}`."
        )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama error {resp.status_code}: {resp.text[:500]}. "
            f"Tip: run `ollama pull {model}` first."
        )
    data = resp.json()
    return (data.get("message", {}).get("content") or "").strip()


def _gemini_chat(model: str, prompt: str, api_key: str, system: str = "") -> str:
    url = PROVIDERS["gemini"]["base_url"].format(model=model)
    full_prompt = prompt
    if system:
        full_prompt = f"{system}\n\n{prompt}"

    resp = requests.post(
        url,
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": full_prompt}]}]},
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Gemini response: {data}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chat(provider: str, model: str, prompt: str, api_key: str | None = None, system: str = "") -> str:
    """
    Send a prompt to any supported provider/model and return the text reply.

    Args:
        provider: one of PROVIDERS keys - "ollama", "groq", "gemini", "openrouter"
        model: model name from PROVIDERS[provider]["models"] (or any valid model id)
        prompt: user prompt (for RAG: context + question)
        api_key: optional; if omitted, read from the provider's env var
        system: optional system instruction
    """
    provider = (provider or "").lower().strip()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}")

    config = PROVIDERS[provider]

    if provider == "ollama":
        return _ollama_chat(model, prompt, system=system)

    # All other providers need a key
    key = _resolve_api_key(provider, api_key)
    if config.get("requires_key") and not key:
        env_var = config.get("env_var", "API_KEY")
        raise RuntimeError(
            f"Missing API key for {config['label']}. "
            f"Paste it in the sidebar, or set {env_var}=... in a .env file. "
            f"See README.md for how to get a free key."
        )

    if provider == "gemini":
        return _gemini_chat(model, prompt, api_key=key, system=system)

    # groq + openrouter = OpenAI-compatible
    return _openai_compatible_chat(config["base_url"], model, prompt, api_key=key, system=system)
