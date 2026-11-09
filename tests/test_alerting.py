"""Tests de la notification sortante : envoi, échec silencieux, messages du monitoring et du déclencheur."""
import pytest
import requests

from pga import alerting as A
from pga import monitor as M
from pga import trigger as TR

URL = "https://hooks.example/secret-token"


class Sent:
    def __init__(self):
        self.calls = []

    def __call__(self, url, json, timeout):
        self.calls.append((url, json))
        return type("Ok", (), {"raise_for_status": lambda self: None})()


@pytest.fixture
def post(monkeypatch):
    sent = Sent()
    monkeypatch.setattr(A.requests, "post", sent)
    monkeypatch.setenv("PGA_ALERT_WEBHOOK", URL)
    return sent


def test_no_webhook_sends_nothing(post, monkeypatch):
    monkeypatch.delenv("PGA_ALERT_WEBHOOK")
    assert A.notify("x") is False and post.calls == []


def test_notify_posts_the_text(post):
    assert A.notify("bonjour") is True
    assert post.calls == [(URL, {"text": "bonjour"})]


def test_failure_does_not_raise_or_leak_the_url(monkeypatch, capsys):
    def boom(url, json, timeout):
        raise requests.ConnectionError(f"impossible de joindre {url}")

    monkeypatch.setattr(A.requests, "post", boom)
    monkeypatch.setenv("PGA_ALERT_WEBHOOK", URL)
    assert A.notify("x") is False
    assert "secret-token" not in capsys.readouterr().err


@pytest.mark.parametrize("status, expected", [("alert", 1), ("warning", 0), ("ok", 0)])
def test_monitoring_notifies_alerts_only(post, status, expected):
    M.notify_alert({"status": status, "reasons": ["[alert] biais"], "model": {"version": "3"},
                    "last_event": {"event_name": "Masters"}})
    assert len(post.calls) == expected
    if expected:
        assert "v3" in post.calls[0][1]["text"] and "[alert] biais" in post.calls[0][1]["text"]


def decision(action: str) -> TR.Decision:
    return TR.Decision(action, "motif", "2024-03-11", "4", "alert", ["alert"])


@pytest.mark.parametrize("action, sent", [("none", 0), ("retrain", 1), ("data_incident", 1), ("rollback", 1)])
def test_trigger_notifies_every_action_but_none(post, monkeypatch, tmp_path, action, sent):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'm.db'}")
    monkeypatch.setattr(TR, "run_trigger", lambda *a, **k: (decision(action), {"executed": False}))
    assert TR.main(["--dry-run"]) == TR.CODES[action]
    assert len(post.calls) == sent
    if sent:
        assert action.upper() in post.calls[0][1]["text"] and "simulation" in post.calls[0][1]["text"]
