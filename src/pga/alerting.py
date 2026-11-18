"""Outgoing notification: POST JSON `{"text": ...}` to `PGA_ALERT_WEBHOOK` (Slack format).

Without this variable, nothing is sent. An unreachable webhook must never interrupt a decision
already made: the failure is reported on stderr, without the URL, which is a secret.
"""
from __future__ import annotations

import os
import sys

import requests


def notify(text: str, timeout: float = 10) -> bool:
    url = os.environ.get("PGA_ALERT_WEBHOOK")
    if not url:
        return False
    try:
        requests.post(url, json={"text": text}, timeout=timeout).raise_for_status()
    except requests.RequestException as exc:
        print(f"  ! notification not sent: {type(exc).__name__}", file=sys.stderr)
        return False
    return True
