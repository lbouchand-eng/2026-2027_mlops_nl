"""Notification sortante : POST JSON `{"text": ...}` vers `PGA_ALERT_WEBHOOK` (format Slack).

Sans cette variable, rien n'est envoyé. Un webhook injoignable ne doit jamais interrompre une
décision déjà prise : l'échec est signalé sur stderr, sans l'URL, qui est un secret.
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
        print(f"  ! notification non envoyée : {type(exc).__name__}", file=sys.stderr)
        return False
    return True
