from dotenv import load_dotenv
import os
import functools
from google import genai

load_dotenv()


def _load_api_keys_from_file(filepath: str = ".env.gemini", varname: str = "GEMINI_API_KEY"):
    keys = []
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith(varname + "="):
                        val = line.split("=", 1)[1].strip()
                        if val.startswith(("'", '"')) and val.endswith(("'", '"')):
                            val = val[1:-1]
                        if val:
                            keys.append(val)
    except Exception:
        pass

    # fallback to environment variable if file didn't contain keys
    if not keys:
        env_key = os.getenv(varname) or os.getenv("GEMINI_API_KEY")
        if env_key:
            keys.append(env_key)

    return keys


API_KEYS = _load_api_keys_from_file()
if not API_KEYS:
    raise RuntimeError("No Gemini API keys found in .env.gemini or environment variables.")

_CURRENT_INDEX = 0
_client = genai.Client(api_key=API_KEYS[_CURRENT_INDEX])


def get_client():
    return _client


def _mask_key(key: str) -> str:
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return key[:2] + "..." + key[-2:]
    return key[:4] + "..." + key[-4:]


def rotate_client():
    """
    Rotate to the next API key, recreate client, and log the masked key.
    """
    global _CURRENT_INDEX, _client
    _CURRENT_INDEX = (_CURRENT_INDEX + 1) % len(API_KEYS)
    _client = genai.Client(api_key=API_KEYS[_CURRENT_INDEX])
    print(f"[gemini_key_manager] Switched API key -> index={_CURRENT_INDEX}, key={_mask_key(API_KEYS[_CURRENT_INDEX])}")
    return _client


def _is_rate_limit_error(exc: Exception) -> bool:
    try:
        code = getattr(exc, "code", None) or getattr(exc, "status", None)
        if code == 429 or str(code) == "429":
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "quota" in msg:
        return True
    return False


def rotate_on_rate_limit(func):
    """
    Decorator that retries the wrapped function, rotating API keys on detected rate-limit (429) errors.
    It cycles through all loaded keys once; non-rate-limit exceptions are re-raised immediately.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_exc = None
        attempts = len(API_KEYS)
        for _ in range(attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if _is_rate_limit_error(e):
                    rotate_client()
                    continue
                raise
        # exhausted keys
        raise last_exc
    return wrapper