"""Minimal DeepSeek chat client (stdlib only), shared by extract_cards.py and scoop.py.

Key: $DEEPSEEK_API_KEY, else the macOS Keychain item 'deepseek-api-key', stored once with
    security add-generic-password -a "$USER" -s deepseek-api-key -w
(it prompts for the key, so it never lands in shell history or in any file here).
Model: $DEEPSEEK_MODEL, default "deepseek-chat".
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

URL = "https://api.deepseek.com/chat/completions"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
KEYCHAIN = "deepseek-api-key"


def api_key(required=True):
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        try:
            key = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN, "-w"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            key = ""
    if not key and required:
        sys.exit("No DeepSeek key. Export DEEPSEEK_API_KEY, or store it once in the macOS Keychain:\n"
                 f"  security add-generic-password -a \"$USER\" -s {KEYCHAIN} -w")
    return key


class DeepSeekError(RuntimeError):
    pass


def chat_json(key, system, user, max_tokens=2000, temperature=0.0, retries=4):
    """One JSON-mode chat call. Returns (parsed_object, meta) where meta has model,
    usage and finish_reason. Retries rate limits and server errors with backoff."""
    body = {"model": MODEL, "temperature": temperature, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(URL, data=data, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.load(r)
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 ** attempt * 3)
                continue
            raise DeepSeekError(f"HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries:
                time.sleep(2 ** attempt * 3)
                continue
            raise DeepSeekError(str(e)) from None
    choice = resp["choices"][0]
    meta = {"model": resp.get("model", MODEL), "usage": resp.get("usage", {}), "finish_reason": choice.get("finish_reason")}
    if choice.get("finish_reason") == "length":
        raise DeepSeekError("output hit max_tokens")
    content = choice["message"].get("content") or ""
    try:
        return json.loads(content), meta
    except json.JSONDecodeError:
        raise DeepSeekError(f"not JSON: {content[:200]}") from None
