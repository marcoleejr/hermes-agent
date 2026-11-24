"""A provider stream that goes silent while the host sleeps, end to end over a real socket.

Incident: a laptop slept mid-turn again and again. Every wake logged ``Stream stale for 1007s
(threshold 240s)``, which read as a stall detector firing 13 minutes late, and the turns ended as
``Operation interrupted: waiting for model response`` without a fallback. The detector was on
time: the process had been frozen, and the kill landed on the first poll after wake. What ended
the turns was the session turn lease: its first renewal after wake found the row past its wall
clock expiry and stopped the turn without asking the store whether anyone had taken the row.

Wire level: the real anthropic SDK against a loopback server that sends SSE pings and no events.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import hermes_state
import run_agent
from agent import chat_completion_helpers as helpers
from agent import chat_completion_stream_monitor as monitor
from agent import turn_facade_lease
from agent.turn_facade_lease import LEASE_TTL_SECONDS, DurableTurnLease
from tests.fakes.fake_llm_provider import FakeLLMServer, Text

pytest.importorskip("anthropic")

MODEL = "claude-test-1"
ANSWER = "answer after wake"
SLEEP_S = 1000.0  # far past LEASE_TTL_SECONDS and every stale budget below


def _anthropic_events(text):
    usage = {"input_tokens": 3, "output_tokens": 2}
    yield "message_start", {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": usage}}
    yield "content_block_start", {"type": "content_block_start", "index": 0,
                                  "content_block": {"type": "text", "text": ""}}
    yield "content_block_delta", {"type": "content_block_delta", "index": 0,
                                  "delta": {"type": "text_delta", "text": text}}
    yield "content_block_stop", {"type": "content_block_stop", "index": 0}
    yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": usage}
    yield "message_stop", {"type": "message_stop"}


class _PingWire:
    """``/v1/messages``: the first ``silent`` requests stream pings and nothing else; later ones
    answer. ``on_silent(n)`` runs once request ``n`` is streaming; ``before_answer()`` before an answer."""

    def __init__(self, silent, *, on_silent=None, before_answer=None):
        self.requests, self.pings = [], 0  # (monotonic time of each request)
        self.release = threading.Event()
        wire = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length", 0)))
                self.close_connection = True
                if not self.path.split("?")[0].endswith("/v1/messages"):
                    self.send_response(404)
                    self.send_header("content-length", "0")
                    self.end_headers()
                    return
                wire.requests.append(time.monotonic())
                n = len(wire.requests)
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("connection", "close")
                self.end_headers()
                try:
                    if n <= silent:
                        self._ping_until_dropped(n)
                        return
                    if before_answer is not None:
                        before_answer()
                    for name, payload in _anthropic_events(ANSWER):
                        self.wfile.write(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode())
                    self.wfile.flush()
                except OSError:
                    pass  # the client aborted the request (expected)

            def _ping_until_dropped(self, n):
                started = False
                while not wire.release.wait(timeout=0.05):
                    self.wfile.write(b'event: ping\ndata: {"type": "ping"}\n\n')
                    self.wfile.flush()
                    wire.pings += 1
                    if not started and on_silent is not None:
                        started = True
                        on_silent(n)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.05), daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


def _agent(wire, **kwargs):
    defaults = dict(
        api_key="test-key", base_url=wire.base_url, provider="anthropic", model=MODEL,
        api_mode="anthropic_messages", platform="cron", quiet_mode=True, skip_context_files=True,
        skip_memory=True, enabled_toolsets=[], max_iterations=2)
    defaults.update(kwargs)
    return run_agent.AIAgent(**defaults)


@pytest.fixture()
def host_clock(monkeypatch):
    """``time.time`` that a sleep moves forward while ``time.monotonic`` stays put: the macOS and
    Linux host sleep. The loopback wire stands in for a remote provider, which does not sleep."""
    real, clock = time.time, SimpleNamespace(offset=0.0)
    clock.sleep = lambda seconds: setattr(clock, "offset", clock.offset + seconds)
    monkeypatch.setattr(time, "time", lambda: real() + clock.offset)
    monkeypatch.setattr(monitor, "is_loopback_host", lambda _host: False)
    return clock


def test_sse_pings_do_not_hold_off_the_stale_kill(monkeypatch):
    """Pings keep the socket (and httpx's read timeout) alive but are not output: the kill must
    land on the stale budget, then the retry gets a fresh request."""
    threshold = 1.0
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", str(threshold))
    with _PingWire(silent=1) as wire:
        response = _agent(wire)._interruptible_streaming_api_call(
            {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]})

    assert wire.pings >= 5, "the scenario needs a live, pinging stream"
    assert len(wire.requests) == 2
    assert wire.requests[1] - wire.requests[0] <= threshold + 10.0
    assert response.content[0].text == ANSWER


def test_breaker_hands_a_pinging_provider_over_to_the_fallback(monkeypatch):
    """A provider that only ever pings trips the cross-turn breaker; the same turn then finishes
    on ``fallback_providers`` instead of failing."""
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.6")
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "2")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")
    with _PingWire(silent=99) as wire, FakeLLMServer(lambda _req: Text("from fallback"),
                                                     aux=lambda _req: Text("from fallback")) as backup:
        agent = _agent(wire, fallback_model=[{
            "provider": "custom", "model": "backup-model", "base_url": backup.base_url, "api_key": "k"}])
        result = agent.run_conversation("hi")

    assert result["final_response"] == "from fallback", result
    assert len(wire.requests) == 2, "the breaker must refuse the dead provider a third attempt"
    assert agent.model == "backup-model"


def test_breaker_counts_the_killed_attempt_not_the_one_that_replaced_it(monkeypatch):
    """The abort wakes the worker, which can open its next attempt before the kill is counted.
    Counting that one instead hides its own later kill, and the breaker trips a window late."""
    with _PingWire(silent=0) as wire:
        agent = _agent(wire)
        call = helpers._StreamingCall(agent, {"model": MODEL, "messages": []}, None)
        call._stream_stale_timeout = 1.0
        killed = call._start_stream_attempt()
        monkeypatch.setattr(call.clients, "close_once", lambda _reason: call._start_stream_attempt())
        call._kill_stale_stream(9.0)

    assert call._stale_counted_attempts == {killed}


def test_turn_killed_by_a_host_sleep_reconnects_and_finishes(tmp_path, monkeypatch, caplog, host_clock):
    """The incident: the host sleeps past the lease TTL while a request waits. The stream is cut on
    the first poll after wake, the retry answers, and the turn ends with that answer: the lease
    renewal after wake must not interrupt a turn whose row nobody took."""
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "240")
    woke = {}
    renewed = threading.Event()
    real_tick = DurableTurnLease.refresh_tick

    def _tick(lease):
        outcome = real_tick(lease)
        if woke:
            renewed.set()
        return outcome

    monkeypatch.setattr(DurableTurnLease, "refresh_tick", _tick)

    def _sleep_host(_n):
        host_clock.sleep(SLEEP_S)
        woke["at"] = time.monotonic()

    db = hermes_state.SessionDB(tmp_path / "state.db")
    with _PingWire(silent=1, on_silent=_sleep_host, before_answer=lambda: renewed.wait(20)) as wire:
        agent = _agent(wire, session_db=db, session_id="slept")
        agent._session_turn_lease_refresh_interval = 0.2
        with caplog.at_level("INFO"):
            result = agent.run_conversation("hi")
    db.close()

    assert renewed.is_set(), "no lease renewal ran after the wake"
    assert result["final_response"] == ANSWER, result
    assert not result.get("interrupted") and len(wire.requests) == 2
    assert wire.requests[1] - woke["at"] <= 10.0, "the dead stream outlived the wake"
    assert any("Computer slept for" in r.getMessage() for r in caplog.records), (
        "a sleep during a remote request must be named next to the stale kill it explains")


@pytest.fixture()
def slept_lease(tmp_path, host_clock):
    db = hermes_state.SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="test")
    assert db.try_acquire_session_turn_lease("s1", "h", ttl_seconds=LEASE_TTL_SECONDS)
    interrupts = []
    agent = SimpleNamespace(session_id="s1", interrupt=lambda msg, **_kw: interrupts.append(msg))
    lease = DurableTurnLease(agent, db, "s1", "h", expires_at=db.session_turn_lease_expires_at("s1", "h"))
    lease.turn_active = True
    host_clock.sleep(SLEEP_S)
    yield db, lease, interrupts
    db.close()


def test_renewal_after_a_sleep_keeps_an_untaken_row(slept_lease):
    db, lease, interrupts = slept_lease
    assert lease.refresh_tick() is None
    assert interrupts == [] and lease.interrupt_message is None
    assert db.session_turn_lease_expires_at("s1", "h") > time.time() + LEASE_TTL_SECONDS / 2
    assert lease._authority_deadline <= db.session_turn_lease_expires_at("s1", "h")


def test_renewal_after_a_sleep_still_stops_when_the_row_was_taken(slept_lease):
    db, lease, interrupts = slept_lease
    assert db.try_acquire_session_turn_lease("s1", "successor", ttl_seconds=LEASE_TTL_SECONDS)
    assert lease.refresh_tick() is False
    assert len(interrupts) == 1
    assert db.session_turn_lease_expires_at("s1", "successor") is not None


def test_renewal_after_a_sleep_stops_when_the_store_stays_locked(slept_lease, tmp_path, monkeypatch):
    """Ownership cannot be confirmed, and the row is already reclaimable: stop, do not keep trying."""
    import sqlite3

    monkeypatch.setattr(turn_facade_lease, "_REFRESH_WRITE_PATIENCE_S", 0.2)
    _db, lease, interrupts = slept_lease
    blocker = sqlite3.connect(tmp_path / "state.db", timeout=0)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        assert lease.refresh_tick() is False
    finally:
        blocker.rollback()
        blocker.close()
    assert interrupts == ["Session turn lease could not be refreshed; stopping to protect the transcript."]
