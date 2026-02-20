from __future__ import annotations

import urllib.error
import urllib.request


def _prefixed(text: str) -> str:
    value = text.strip()
    if value.startswith("TinyLORA"):
        return value
    return f"TinyLORA | {value}"


def notify_tinylora(
    message: str,
    topic_url: str = "",
    title: str | None = None,
    priority: str = "default",
    timeout_seconds: int = 5,
) -> tuple[bool, str]:
    """
    Send TinyLORA-prefixed ntfy notification.
    Returns (success, detail).
    """
    body = _prefixed(message)
    headers = {
        "Title": _prefixed(title if title else "Status"),
        "Priority": priority,
    }
    if not topic_url.strip():
        return True, "notification skipped (empty topic_url)"

    request = urllib.request.Request(
        topic_url,
        data=body.encode("utf-8"),
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status_code = getattr(response, "status", 200)
            if 200 <= status_code < 300:
                return True, f"ok:{status_code}"
            return False, f"http:{status_code}"
    except urllib.error.URLError as exc:
        return False, f"url_error:{exc}"
    except Exception as exc:  # pragma: no cover
        return False, f"error:{exc}"
