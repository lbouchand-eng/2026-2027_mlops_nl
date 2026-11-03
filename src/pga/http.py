"""Client HTTP minimal : throttle, retries exponentiels, sauvegarde du JSON brut.

Les réponses brutes sont écrites telles quelles sur disque (versionnées par DVC) :
si le fichier existe déjà, on le relit au lieu de refaire la requête, ce qui rend
chaque étape idempotente et rejouable hors-ligne.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

HEADERS = {"User-Agent": "Mozilla/5.0 (albert-school-mlops-pga-project)"}
_session = requests.Session()
_session.headers.update(HEADERS)
_last_call = 0.0


class TransientHTTPError(Exception):
    pass


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, (TransientHTTPError, requests.ConnectionError, requests.Timeout))


@retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _fetch(url: str, params: dict | None, timeout: float, interval: float) -> dict:
    global _last_call
    wait = interval - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()
    r = _session.get(url, params=params, timeout=timeout)
    if r.status_code == 429 or r.status_code >= 500:
        raise TransientHTTPError(f"{r.status_code} {r.url}")
    r.raise_for_status()
    return r.json()


def get_json(
    url: str,
    cache_path: Path,
    params: dict | None = None,
    *,
    timeout: float = 30,
    interval: float = 1.0,
    refresh: bool = False,
) -> dict:
    """Renvoie le JSON de `url`, en le lisant depuis `cache_path` s'il existe déjà."""
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())
    data = _fetch(url, params, timeout, interval)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(cache_path)  # écriture atomique : pas de JSON tronqué si on coupe le script
    return data
