"""AC-L4.1-L4.4: ``Runner.load``, local execution, fallback policies, refresh and stop.

Offline: the HTTP transport, clock and model provider are injected fakes.
"""

import asyncio
import hashlib
import http.client
import http.server
import importlib
import inspect
import json
import shutil
import stat
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

GLINER = "fastino/gliner2.5-multi-v1"
SCHEMA = {
    "type": "object",
    "properties": {"employee_id": {"type": "string"}},
    "required": ["employee_id"],
}
KEY = "sk-secret-key"
NOW = 1_800_000_000.0


@pytest.fixture(scope="module")
def rt(tmp_path_factory):
    """Import an isolated copy so dependency doubles cannot leak between modules."""
    root = tmp_path_factory.mktemp("rt")
    pkg = root / "seldon_runtime"
    shutil.copytree(Path(__file__).resolve().parents[1] / "seldon_runtime", pkg)

    def ours():
        return [n for n in sys.modules if n == "seldon_runtime" or n.startswith("seldon_runtime.")]

    saved = {n: sys.modules.pop(n) for n in ours()}
    sys.path.insert(0, str(root))
    try:
        yield importlib.import_module("seldon_runtime.runner")
    finally:
        sys.path.remove(str(root))
        for n in ours():
            del sys.modules[n]
        sys.modules.update(saved)


@pytest.fixture(autouse=True)
def clock(rt, monkeypatch):
    monkeypatch.setattr(rt, "_now", lambda: NOW)


def run(coro):
    return asyncio.run(coro)


RUNNERS = []


@pytest.fixture(autouse=True)
def close_runners():
    yield
    while RUNNERS:
        run(RUNNERS.pop().aclose())


def make_bundle(rt, **overrides):
    from seldon_runtime.contracts import FunctionSpec, Plan, digest

    spec = FunctionSpec.model_validate(
        {
            "calls": [
                {
                    "id": "extract",
                    "model": "reference",
                    "messages": [
                        {"role": "system", "content": "Return employee_id."},
                        {"role": "user", "content": "{{document}}"},
                    ],
                }
            ],
            "inputs": {"document": "Ada works at Acme"},
            "bindings": [
                {
                    "call": "extract",
                    "message": 1,
                    "placeholder": "{{document}}",
                    "source": "input#/document",
                }
            ],
            "output_schema": SCHEMA,
        }
    )
    plan = Plan.model_validate(
        {
            "version": "function-plan-v3",
            "fallback": True,
            "steps": [
                {
                    "id": "a",
                    "op": "extract",
                    "model": GLINER,
                    "call": "extract",
                    "config": {"schema": SCHEMA},
                }
            ],
        }
    )
    body = {
        "version": "function-bundle-v1",
        "function_id": "fn-1",
        "revision_id": "rev-1",
        "plan_digest": plan.digest,
        "contract_digest": spec.digest,
        "optimizer_contract_version": None,
        "spec": spec.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "local_models": {},
        "runtime_min_version": "0.1.0",
        "conformance_ref": None,
        "expires_at": (datetime.fromtimestamp(NOW, UTC) + timedelta(hours=24)).isoformat(),
        **overrides,
    }
    return body, digest({k: v for k, v in body.items() if k != "expires_at"})


class Http:
    """Fake transport: ``routes`` maps (method, url suffix) to a response or exception."""

    def __init__(self, bundle_response, post=None):
        self.bundle_response = bundle_response
        self.post = post
        self.calls = []

    async def __call__(self, method, url, headers, body):
        self.calls.append((method, url, headers, body))
        if method == "GET":
            result = self.bundle_response
        else:
            result = self.post or (500, {}, b"")
        if isinstance(result, Exception):
            raise result
        return result

    @property
    def posts(self):
        return [c for c in self.calls if c[0] == "POST"]


def ok(bundle_etag):
    body, etag = bundle_etag
    return 200, {"etag": f'"{etag}"'}, json.dumps(body).encode()


async def local_ok(model, payload):
    assert "primitive" in payload
    return {"employee_id": "E1"}


def failing(code):
    async def provider(model, payload):
        raise ValueError(code)

    return provider


def completion(content):
    message = {"content": json.dumps(content)}
    return 200, {}, json.dumps({"choices": [{"message": message}]}).encode()


def load(rt, tmp_path, http, **kw):
    kw.setdefault("provider", local_ok)
    kw.setdefault("telemetry", False)
    kw.setdefault("allow_unverified", True)  # gate behaviour is covered by the conformance tests
    runner = run(rt.Runner.load("fn-1", KEY, cache_dir=tmp_path, http=http, **kw))
    RUNNERS.append(runner)
    return runner


def env_dir(tmp_path, base_url="https://api.seldon-ai.com"):
    return tmp_path / "bundles" / hashlib.sha256(base_url.encode()).hexdigest()[:16]


def cache_file(tmp_path):
    return env_dir(tmp_path) / "fn-1.json"


def conflict(reason):
    body = {"error": "local_ineligible", "message": "m", "correlation_id": "c"}
    return 409, {}, json.dumps({**body, "details": {"reason": reason}}).encode()


def test_load_200_caches_0600_and_calls(rt, tmp_path):
    runner = load(rt, tmp_path, Http(ok(make_bundle(rt))))
    mode = stat.S_IMODE(cache_file(tmp_path).stat().st_mode)
    assert mode == 0o600
    assert KEY not in cache_file(tmp_path).read_text()
    assert run(runner({"document": "x"})) == {
        "output": {"employee_id": "E1"},
        "fallback_used": False,
        "fallback_reason": None,
    }


def test_second_load_sends_if_none_match_and_304_serves_cache(rt, tmp_path):
    bundle = make_bundle(rt)
    load(rt, tmp_path, Http(ok(bundle)))
    http = Http((304, {}, b""))
    runner = load(rt, tmp_path, http)
    assert http.calls[0][2]["If-None-Match"] == f'"{bundle[1]}"'
    assert http.calls[0][2]["Authorization"] == f"Bearer {KEY}"
    assert runner._expires_at == NOW + 86400
    assert run(runner({"document": "x"}))["output"] == {"employee_id": "E1"}


def tampered(rt, how):
    body, etag = make_bundle(rt)
    if how == "etag":
        etag = "0" * 64
    elif how == "plan":
        body["plan"]["max_items"] = 7
    elif how == "spec":
        body["spec"]["inputs"] = {"document": "other"}
    elif how == "version":
        body["version"] = "function-bundle-v2"
    return body, etag


@pytest.mark.parametrize("how", ["etag", "plan", "spec", "version"])
def test_validation_failures_are_bundle_invalid_and_not_cached(rt, tmp_path, how):
    with pytest.raises(rt.LocalExecutionError) as exc:
        load(rt, tmp_path, Http(ok(tampered(rt, how))))
    assert str(exc.value) == exc.value.reason == "bundle_invalid"
    assert not cache_file(tmp_path).exists()


def test_newer_runtime_min_version_requires_upgrade(rt, tmp_path):
    with pytest.raises(rt.LocalExecutionError, match="runtime_upgrade_required"):
        load(rt, tmp_path, Http(ok(make_bundle(rt, runtime_min_version="99.0.0"))))
    assert not cache_file(tmp_path).exists()


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (conflict("disabled"), "local_ineligible_disabled"),
        (conflict("weird"), "local_ineligible"),
        ((409, {}, b'{"reason":"disabled"}'), "local_ineligible"),
        ((409, {}, b"not json"), "local_ineligible"),
        ((409, {}, b'{"details":[1]}'), "local_ineligible"),
        ((401, {}, b""), "unauthorized"),
        ((404, {}, b""), "function_not_found"),
        ((500, {}, b""), "bundle_unavailable"),
        (OSError("down"), "bundle_unavailable"),
    ],
)
def test_status_mapping(rt, tmp_path, response, code):
    with pytest.raises(rt.LocalExecutionError) as exc:
        load(rt, tmp_path, Http(response))
    assert exc.value.reason == code


def test_network_error_serves_valid_cache_until_expiry(rt, tmp_path, monkeypatch):
    load(rt, tmp_path, Http(ok(make_bundle(rt))))
    runner = load(rt, tmp_path, Http(OSError("down")))
    assert run(runner({"document": "x"}))["output"] == {"employee_id": "E1"}
    monkeypatch.setattr(rt, "_now", lambda: NOW + 2 * 86400)
    with pytest.raises(rt.LocalExecutionError, match="bundle_unavailable"):
        load(rt, tmp_path, Http(OSError("down")))


def test_parity_with_shared_execute(rt, tmp_path, monkeypatch):
    from seldon_runtime import executor

    assert inspect.signature(executor.execute).parameters["timeout"].default == 55
    seen = {}
    real = rt.execute

    async def spy(*args, **kwargs):
        seen.update(kwargs)
        return await real(*args, **kwargs)

    monkeypatch.setattr(rt, "execute", spy)
    runner = load(rt, tmp_path, Http(ok(make_bundle(rt))))
    run(runner({"document": "x"}))
    assert "timeout" not in seen and seen["serving"] is True


def test_oversized_inputs_rejected_by_executor(rt, tmp_path):
    runner = load(rt, tmp_path, Http(ok(make_bundle(rt))))
    with pytest.raises(rt.LocalExecutionError) as exc:
        run(runner({"document": "x" * 262144}))
    assert exc.value.reason == "invalid_function_inputs"


def test_slow_provider_hits_executor_deadline(rt, tmp_path, monkeypatch):
    real = rt.execute

    async def short(*args, **kwargs):
        return await real(*args, timeout=0.05, **kwargs)

    async def slow(model, payload):
        await asyncio.sleep(5)

    monkeypatch.setattr(rt, "execute", short)
    runner = load(rt, tmp_path, Http(ok(make_bundle(rt))), provider=slow)
    with pytest.raises(rt.LocalExecutionError) as exc:
        run(runner({"document": "x"}))
    assert exc.value.reason == "execution_timeout"


def test_fallback_seldon(rt, tmp_path):
    http = Http(ok(make_bundle(rt)), post=completion({"employee_id": "E2"}))
    runner = load(rt, tmp_path, http, provider=failing("provider_unavailable"))
    result = run(runner({"document": "x"}))
    assert result == {
        "output": {"employee_id": "E2"},
        "fallback_used": True,
        "fallback_reason": "provider_unavailable",
    }
    ((_, url, headers, body),) = http.posts
    assert url == "https://api.seldon-ai.com/v1/chat/completions"
    assert headers["Authorization"] == f"Bearer {KEY}"
    sent = json.loads(body)
    assert sent["model"] == "reference" and "messages" in sent
    assert "seldon_output_token_policy" not in sent


def test_fallback_endpoint(rt, tmp_path):
    http = Http(ok(make_bundle(rt)), post=completion({"employee_id": "E2"}))
    fallback = {"endpoint": "https://llm.example/v1/", "api_key": "other-key"}
    runner = load(rt, tmp_path, http, provider=failing("provider_unavailable"), fallback=fallback)
    assert run(runner({"document": "x"}))["fallback_used"] is True
    ((_, url, headers, _),) = http.posts
    assert url == "https://llm.example/v1/chat/completions"
    assert headers["Authorization"] == "Bearer other-key"


@pytest.mark.parametrize("valid_name", [True, False])
def test_tool_output_fallback(rt, tmp_path, valid_name):
    from seldon_runtime.contracts import FunctionSpec, digest
    from seldon_runtime.tool_wire import completion_choice

    body, _ = make_bundle(rt)
    call = body["spec"]["calls"][0]
    call.update(
        tools=[{"type": "function", "function": {"name": "result", "parameters": SCHEMA}}],
        tool_choice="required",
        parallel_tool_calls=False,
    )
    body["spec"]["wire_mode"] = "messages"
    body["contract_digest"] = FunctionSpec.model_validate(body["spec"]).digest
    etag = digest({k: v for k, v in body.items() if k != "expires_at"})
    choice = completion_choice(call, {"employee_id": "E2"}, call_id="call_1")
    if not valid_name:
        choice["message"]["tool_calls"][0]["function"]["name"] = "other"
    post = (200, {}, json.dumps({"choices": [choice]}).encode())
    http = Http(ok((body, etag)), post=post)
    runner = load(rt, tmp_path, http, provider=failing("provider_unavailable"))
    if valid_name:
        assert run(runner({"document": "x"})) == {
            "output": {"employee_id": "E2"},
            "fallback_used": True,
            "fallback_reason": "provider_unavailable",
        }
    else:
        with pytest.raises(rt.LocalExecutionError, match="invalid_provider_output"):
            run(runner({"document": "x"}))
    sent = json.loads(http.posts[0][3])
    assert sent["tools"] == call["tools"]
    assert sent["tool_choice"] == "required"
    assert sent["parallel_tool_calls"] is False


def test_fallback_raise_never_posts(rt, tmp_path):
    http = Http(ok(make_bundle(rt)))
    runner = load(rt, tmp_path, http, provider=failing("provider_unavailable"), fallback="raise")
    with pytest.raises(rt.LocalExecutionError) as exc:
        run(runner({"document": "x"}))
    assert exc.value.reason == "provider_unavailable"
    assert http.posts == []


def test_ineligible_failure_does_not_fall_back(rt, tmp_path):
    http = Http(ok(make_bundle(rt)), post=completion({"employee_id": "E2"}))
    runner = load(rt, tmp_path, http, provider=failing("call_limit"))
    with pytest.raises(rt.LocalExecutionError) as exc:
        run(runner({"document": "x"}))
    assert exc.value.reason == "call_limit"
    assert http.posts == []


@pytest.mark.parametrize("fallback", ["nope", {"endpoint": "x"}, None, 3])
def test_invalid_fallback(rt, tmp_path, fallback):
    http = Http(ok(make_bundle(rt)))
    with pytest.raises(ValueError, match="invalid_fallback"):
        load(rt, tmp_path, http, fallback=fallback)
    assert http.calls == []


def test_cache_write_failure_is_not_fatal_and_leaves_no_temp(rt, tmp_path, monkeypatch):
    def boom(*_):
        raise OSError("/secret/path")

    monkeypatch.setattr(rt.os, "replace", boom)
    runner = load(rt, tmp_path, Http(ok(make_bundle(rt))))
    assert run(runner({"document": "x"}))["output"] == {"employee_id": "E1"}
    assert [p for p in (tmp_path / "bundles").rglob("*") if p.is_file()] == []


class Switch:
    """Mutable fake transport: replace ``.response`` between refreshes."""

    def __init__(self, response, post=None):
        self.response, self.post, self.calls = response, post, []

    async def __call__(self, method, url, headers, body):
        self.calls.append((method, url, headers, body))
        result = self.response if method == "GET" else self.post or (500, {}, b"")
        if isinstance(result, Exception):
            raise result
        return result

    @property
    def gets(self):
        return [c for c in self.calls if c[0] == "GET"]

    @property
    def posts(self):
        return [c for c in self.calls if c[0] == "POST"]


def set_clock(monkeypatch, rt, value):
    monkeypatch.setattr(rt, "_now", lambda: value)


def test_refresh_304_extends_expiry_and_rewrites_cache(rt, tmp_path, monkeypatch):
    http = Switch(ok(make_bundle(rt)))
    runner = load(rt, tmp_path, http)
    set_clock(monkeypatch, rt, NOW + 3600)
    http.response = (304, {}, b"")
    run(runner.refresh())
    assert runner._expires_at == NOW + 3600 + 86400
    assert json.loads(cache_file(tmp_path).read_text())["expires_at"] == NOW + 3600 + 86400
    assert http.gets[-1][2]["If-None-Match"].strip('"') == runner._etag


def test_refresh_200_swaps_bundle_and_etag(rt, tmp_path):
    http = Switch(ok(make_bundle(rt)))
    runner = load(rt, tmp_path, http)
    new = make_bundle(rt, revision_id="rev-2")
    http.response = ok(new)
    run(runner.refresh())
    assert runner._bundle["revision_id"] == "rev-2"
    assert runner._plan is not None and runner._spec is not None
    assert json.loads(cache_file(tmp_path).read_text())["etag"] == new[1]
    http.response = (304, {}, b"")
    run(runner.refresh())
    assert http.gets[-1][2]["If-None-Match"] == f'"{new[1]}"'


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (conflict("disabled"), "local_ineligible_disabled"),
        ((404, {}, b""), "function_not_found"),
        ((401, {}, b""), "unauthorized"),
    ],
)
def test_definitive_answers_stop_serving_without_fallback(rt, tmp_path, response, code):
    http = Switch(ok(make_bundle(rt)), post=completion({"employee_id": "E2"}))
    runner = load(rt, tmp_path, http)
    http.response = response
    run(runner.refresh())
    before = len(http.calls)
    with pytest.raises(rt.LocalExecutionError) as exc:
        run(runner({"document": "x"}))
    assert exc.value.reason == code
    assert len(http.calls) == before and http.posts == []
    assert not cache_file(tmp_path).exists()


def test_reenable_after_stop(rt, tmp_path):
    http = Switch(ok(make_bundle(rt)))
    runner = load(rt, tmp_path, http)
    http.response = conflict("disabled")
    run(runner.refresh())
    http.response = ok(make_bundle(rt, revision_id="rev-2"))
    run(runner.refresh())
    assert run(runner({"document": "x"}))["output"] == {"employee_id": "E1"}
    assert cache_file(tmp_path).exists()


@pytest.mark.parametrize("down", [OSError("down"), (503, {}, b"")])
def test_unreachable_serves_until_expiry_then_seldon_fallback(rt, tmp_path, monkeypatch, down):
    http = Switch(ok(make_bundle(rt)), post=completion({"employee_id": "E2"}))
    runner = load(rt, tmp_path, http, provider=failing("must_not_be_called"))
    http.response = down
    run(runner.refresh())
    set_clock(monkeypatch, rt, NOW + 86400 - 1)
    with pytest.raises(rt.LocalExecutionError) as exc:  # still local: provider fails, no expiry
        run(runner({"document": "x"}))
    assert exc.value.reason != "bundle_expired"
    set_clock(monkeypatch, rt, NOW + 86400)
    result = run(runner({"document": "x"}))
    assert result == {
        "output": {"employee_id": "E2"},
        "fallback_used": True,
        "fallback_reason": "bundle_expired",
    }
    ((_, url, _, _),) = http.posts
    assert url == "https://api.seldon-ai.com/v1/chat/completions"


def test_unreachable_serves_locally_before_expiry(rt, tmp_path, monkeypatch):
    http = Switch(ok(make_bundle(rt)))
    runner = load(rt, tmp_path, http)
    http.response = OSError("down")
    run(runner.refresh())
    set_clock(monkeypatch, rt, NOW + 86400 - 1)
    assert run(runner({"document": "x"}))["fallback_used"] is False


def test_expired_with_raise_policy(rt, tmp_path, monkeypatch):
    http = Switch(ok(make_bundle(rt)))
    runner = load(rt, tmp_path, http, fallback="raise")
    set_clock(monkeypatch, rt, NOW + 86400)
    with pytest.raises(rt.LocalExecutionError) as exc:
        run(runner({"document": "x"}))
    assert exc.value.reason == "bundle_expired" and http.posts == []


def test_invalid_refresh_keeps_serving_and_cache(rt, tmp_path):
    http = Switch(ok(make_bundle(rt)))
    runner = load(rt, tmp_path, http)
    cached = cache_file(tmp_path).read_text()
    expires = runner._expires_at
    http.response = ok(tampered(rt, "plan"))
    run(runner.refresh())
    assert cache_file(tmp_path).read_text() == cached and runner._expires_at == expires
    assert run(runner({"document": "x"}))["output"] == {"employee_id": "E1"}


def test_expiry_after_a_refused_refresh_reports_the_refusal(rt, tmp_path, monkeypatch):
    http = Switch(ok(make_bundle(rt)))
    runner = load(rt, tmp_path, http, fallback="raise")
    http.response = ok(tampered(rt, "plan"))
    run(runner.refresh())
    set_clock(monkeypatch, rt, NOW + 86400)
    with pytest.raises(rt.LocalExecutionError) as exc:
        run(runner({"document": "x"}))
    assert exc.value.reason == "bundle_invalid"

    http.response = OSError("down")  # unreachable is not a refusal: the reason is kept
    run(runner.refresh())
    assert runner._refused == "bundle_invalid"
    http.response = ok(make_bundle(rt))
    run(runner.refresh())
    assert runner._refused is None
    with pytest.raises(rt.LocalExecutionError, match="bundle_expired"):
        run(runner({"document": "x"}))  # the fixture bundle's own expiry has passed


def test_restarted_refresh_task_checks_at_once_when_overdue(rt, tmp_path, monkeypatch):
    http = Switch(ok(make_bundle(rt)))
    runner = load(rt, tmp_path, http)  # its loop is gone, so no check ever ran
    http.response = (409, {}, json.dumps({"details": {"reason": "disabled"}}).encode())
    set_clock(monkeypatch, rt, NOW + rt.REFRESH_SECONDS)

    async def call():
        await runner({"document": "x"})  # served; the check runs behind it
        for _ in range(10):
            await asyncio.sleep(0)
        await runner.aclose()

    run(call())
    assert len(http.gets) == 2
    with pytest.raises(rt.LocalExecutionError, match="local_ineligible_disabled"):
        run(runner({"document": "x"}))


def test_refresh_loop_sleeps_900_then_refreshes_and_aclose_ends_it(rt, tmp_path, monkeypatch):
    assert rt.REFRESH_SECONDS == 900
    real_sleep, sleeps = asyncio.sleep, []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 1:
            await asyncio.Event().wait()  # park the loop after the first tick
        await real_sleep(0)

    async def scenario():
        http = Switch(ok(make_bundle(rt)))
        runner = await rt.Runner.load(
            "fn-1",
            KEY,
            cache_dir=tmp_path,
            http=http,
            provider=local_ok,
            allow_unverified=True,
        )
        http.response = (304, {}, b"")
        monkeypatch.setattr(rt.asyncio, "sleep", fake_sleep)
        for _ in range(10):
            await real_sleep(0)
        assert sleeps[:1] == [900] and len(http.gets) == 2
        task = runner._task
        await runner.aclose()
        assert task.done()

    run(scenario())


def test_call_never_fetches_bundle(rt, tmp_path):
    http = Switch(ok(make_bundle(rt)))
    runner = load(rt, tmp_path, http)
    for _ in range(3):
        run(runner({"document": "x"}))
    assert len(http.gets) == 1


def test_cache_is_keyed_by_base_url(rt, tmp_path):
    load(rt, tmp_path, Http(ok(make_bundle(rt))))
    assert cache_file(tmp_path).exists()
    http = Http((503, {}, b""))
    with pytest.raises(rt.LocalExecutionError, match="bundle_unavailable"):
        load(rt, tmp_path, http, base_url="https://staging.example")
    assert "If-None-Match" not in http.calls[0][2]


def test_http_exception_is_a_transport_failure(rt, tmp_path, monkeypatch):
    def broken(*_a, **_k):
        raise http.client.IncompleteRead(b"x")

    monkeypatch.setattr(rt.urllib.request.OpenerDirector, "open", broken)
    with pytest.raises(rt.LocalExecutionError, match="bundle_unavailable"):
        run(rt.Runner.load("fn-1", KEY, cache_dir=tmp_path))


class _Handler(http.server.BaseHTTPRequestHandler):
    seen = []

    def do_GET(self):
        _Handler.seen.append((self.path, self.headers.get("Authorization")))
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/target")
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header("X-Custom", "V")
            self.end_headers()
            self.wfile.write(b"hello")

    def log_message(self, *_):
        pass


def test_real_transport_round_trip_and_no_redirects(rt):
    _Handler.seen = []
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, headers, data = run(
            rt._urllib_http("GET", f"{base}/ok", {"Authorization": "B"}, None)
        )
        assert (status, headers["x-custom"], data) == (200, "V", b"hello")
        status, _, _ = run(rt._urllib_http("GET", f"{base}/redirect", {"Authorization": "B"}, None))
        assert status == 302
        assert [path for path, _ in _Handler.seen] == ["/ok", "/redirect"]
    finally:
        server.shutdown()
        server.server_close()


def test_refresh_task_restarts_on_the_calling_loop(rt, tmp_path):
    runner = load(rt, tmp_path, Http(ok(make_bundle(rt))))
    assert runner._task.done()  # its loop (asyncio.run in load) is gone

    async def call():
        await runner({"document": "x"})
        assert not runner._task.done() and runner._task.get_loop() is asyncio.get_running_loop()
        await runner.aclose()
        assert runner._task.done()

    run(call())
    run(runner.aclose())  # stored task belongs to a closed loop: still safe


def test_in_flight_call_keeps_its_provider_across_a_swap(rt, tmp_path):
    release = {}

    def pinned(name):
        async def provider(model, payload):
            await release["event"].wait()
            return {"employee_id": name}

        return provider

    async def scenario():
        release["event"] = asyncio.Event()
        http = Switch(ok(make_bundle(rt)))
        runner = await rt.Runner.load(
            "fn-1",
            KEY,
            cache_dir=tmp_path,
            http=http,
            provider=pinned("old"),
            allow_unverified=True,
        )
        runner._injected = False  # let the swap rebuild the provider
        new = make_bundle(rt, revision_id="rev-2")
        http.response = ok(new)
        task = asyncio.create_task(runner({"document": "x"}))
        for _ in range(5):
            await asyncio.sleep(0)
        rt.LocalModels = lambda _models: type("L", (), {"provider": pinned("new")})()
        await runner.refresh()
        assert runner._bundle["revision_id"] == "rev-2"
        release["event"].set()
        out = (await task)["output"]
        await runner.aclose()
        return out

    original = rt.LocalModels
    try:
        assert run(scenario()) == {"employee_id": "old"}
    finally:
        rt.LocalModels = original


# ---- Task 5b: verify() and the 0.95 agreement gate (AC-L5.2) ----

REF = "/v1/seldon/functions/fn-1/conformance?revision=rev-1"


def good_digest(rt):
    from seldon_runtime.contracts import digest

    return digest({"employee_id": "E1"})


def conf_body(rt, bad=0, n=20, **over):
    items = [
        {"input": {"document": f"secret-doc-{i}"}, "output_digest": good_digest(rt)}
        for i in range(n)
    ]
    for item in items[:bad]:
        item["output_digest"] = "0" * 64
    body = {
        "version": "function-conformance-v1",
        "function_id": "fn-1",
        "revision_id": "rev-1",
        "plan_digest": make_bundle(rt)[0]["plan_digest"],
        "items": items,
        **over,
    }
    return 200, {}, json.dumps(body).encode()


def gated(rt, **overrides):
    return make_bundle(rt, conformance_ref=REF, **overrides)


class Conf(Switch):
    """Bundle on GET /bundle, conformance answer on GET <conformance_ref>."""

    def __init__(self, bundle, conf):
        super().__init__(ok(bundle))
        self.conf = conf

    async def __call__(self, method, url, headers, body):
        if "/conformance" not in url:
            return await super().__call__(method, url, headers, body)
        self.calls.append((method, url, headers, body))
        if isinstance(self.conf, Exception):
            raise self.conf
        return self.conf

    @property
    def conf_calls(self):
        return [c for c in self.calls if "/conformance" in c[1]]


def gate_load(rt, tmp_path, http, **kw):
    kw["allow_unverified"] = False
    return load(rt, tmp_path, http, **kw)


@pytest.mark.parametrize(
    ("bad", "agreement", "passed"), [(0, 1.0, True), (1, 0.95, True), (2, 0.9, False)]
)
def test_verify_agreement_threshold(rt, tmp_path, bad, agreement, passed):
    http = Conf(gated(rt), conf_body(rt, bad=bad))
    runner = load(rt, tmp_path, http)
    result = run(runner.verify())
    assert (result["agreement"], result["samples"], result["passed"]) == (
        agreement,
        20,
        passed,
    )
    assert http.conf_calls[0][2]["Authorization"] == f"Bearer {KEY}"


def test_execution_exception_is_a_disagreement_and_latency_is_nearest_rank(
    rt, tmp_path, monkeypatch
):
    http = Conf(gated(rt), conf_body(rt, n=4))
    runner = load(rt, tmp_path, http)
    out = [("ok", 10), ("boom", 0), ("ok", 40), ("ok", 20)]

    async def fake(plan, spec, inputs, provider, **kw):
        kind, ms = out.pop(0)
        assert plan.fallback is False
        if kind == "boom":
            raise RuntimeError("secret-doc")
        return type("E", (), {"output": {"employee_id": "E1"}, "elapsed_ms": ms})()

    monkeypatch.setattr(rt, "execute", fake)
    result = run(runner.verify())
    assert result == {
        "agreement": 0.75,
        "samples": 4,
        "p50_ms": 20,
        "p95_ms": 40,
        "passed": False,
    }


def test_verify_never_uses_the_fallback(rt, tmp_path):
    http = Conf(gated(rt), conf_body(rt, n=3))
    runner = load(rt, tmp_path, http, provider=failing("provider_unavailable"))
    with pytest.raises(rt.LocalExecutionError, match="^provider_unavailable$"):
        run(runner.verify())
    assert http.posts == []


@pytest.mark.parametrize(
    "ref",
    [
        None,
        "https://evil.example" + REF,
        "/v1/other",
        REF + "&x=1",
        "/v1/seldon/functions/a/b",
    ],
)
def test_bad_conformance_ref_makes_no_request(rt, tmp_path, ref):
    http = Conf(make_bundle(rt, conformance_ref=ref), conf_body(rt))
    runner = load(rt, tmp_path, http)
    with pytest.raises(rt.LocalExecutionError, match="^conformance_unavailable$"):
        run(runner.verify())
    assert http.conf_calls == []


@pytest.mark.parametrize(
    ("conf", "code"),
    [
        ((404, {}, b"{}"), "conformance_unavailable"),
        ((500, {}, b""), "conformance_unavailable"),
        (OSError("down"), "conformance_unavailable"),
        ((200, {}, b"not json"), "conformance_unavailable"),
        ((401, {}, b""), "unauthorized"),
        ((403, {}, b""), "unauthorized"),
    ],
)
def test_conformance_fetch_failures(rt, tmp_path, conf, code):
    runner = load(rt, tmp_path, Conf(gated(rt), conf))
    with pytest.raises(rt.LocalExecutionError, match=f"^{code}$"):
        run(runner.verify())


@pytest.mark.parametrize(
    "over",
    [
        {"version": "v2"},
        {"plan_digest": "f" * 64},
        {"items": []},
        {"items": [{"input": 1}]},
    ],
)
def test_conformance_body_validation(rt, tmp_path, over):
    runner = load(rt, tmp_path, Conf(gated(rt), conf_body(rt, **over)))
    with pytest.raises(rt.LocalExecutionError, match="^conformance_unavailable$"):
        run(runner.verify())


def test_gate_failing_set_refuses_load_and_leaves_no_task(rt, tmp_path):
    http = Conf(gated(rt), conf_body(rt, bad=5))

    async def scenario():
        with pytest.raises(rt.LocalExecutionError, match="^conformance_failed$"):
            await rt.Runner.load(
                "fn-1",
                KEY,
                cache_dir=tmp_path,
                http=http,
                provider=local_ok,
                allow_unverified=False,
            )
        return [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    assert run(scenario()) == []
    assert not cache_file(tmp_path).exists()


def test_allow_unverified_never_requests_conformance(rt, tmp_path):
    http = Conf(gated(rt), conf_body(rt, bad=20))
    runner = load(rt, tmp_path, http)  # helper default is allow_unverified=True
    assert http.conf_calls == []
    assert run(runner({"document": "x"}))["output"] == {"employee_id": "E1"}
    assert run(runner.verify())["passed"] is False  # explicit verify still works


def test_result_cache_hit_and_failing_result_cached(rt, tmp_path):
    http = Conf(gated(rt), conf_body(rt))
    gate_load(rt, tmp_path, http)
    assert len(http.conf_calls) == 1
    gate_load(rt, tmp_path, http)
    assert len(http.conf_calls) == 1
    (result_file,) = (tmp_path / "verify").rglob("*.json")
    assert oct(result_file.stat().st_mode & 0o777) == "0o600"
    assert "secret-doc" not in result_file.read_text() and KEY not in result_file.read_text()

    failing_http = Conf(
        gated(rt, local_models={GLINER: {"hf_revision": "r9"}}), conf_body(rt, bad=20)
    )
    for _ in range(2):
        with pytest.raises(rt.LocalExecutionError, match="^conformance_failed$"):
            gate_load(rt, tmp_path, failing_http)
    assert len(failing_http.conf_calls) == 1


def test_result_cache_key_changes_replay(rt, tmp_path, monkeypatch):
    http = Conf(gated(rt), conf_body(rt))
    gate_load(rt, tmp_path, http)
    # hf_revision change (different local_models entry, so also a different etag)
    models = {GLINER: {"hf_revision": "r2"}}
    http2 = Conf(gated(rt, local_models=models), conf_body(rt))
    gate_load(rt, tmp_path, http2, provider=local_ok)
    assert len(http2.conf_calls) == 1
    # runtime version change
    monkeypatch.setattr(rt, "__version__", "9.9.9")
    gate_load(rt, tmp_path, http)
    assert len(http.conf_calls) == 2


def test_plan_digest_change_replays(rt, tmp_path):
    http = Conf(gated(rt), conf_body(rt))
    gate_load(rt, tmp_path, http)
    other = gated(rt)
    other[0]["plan"]["steps"][0]["config"] = {"schema": {**SCHEMA, "title": "t"}}
    from seldon_runtime.contracts import Plan, digest

    body = other[0]
    body["plan_digest"] = Plan.model_validate(body["plan"]).digest
    http2 = Conf(
        (body, digest({k: v for k, v in body.items() if k != "expires_at"})),
        conf_body(rt, plan_digest=body["plan_digest"]),
    )
    gate_load(rt, tmp_path, http2)
    assert len(http2.conf_calls) == 1


def test_unwritable_result_cache_still_works(rt, tmp_path, monkeypatch):
    def boom(*_):
        raise OSError("/secret/path")

    monkeypatch.setattr(rt.os, "replace", boom)
    runner = gate_load(rt, tmp_path, Conf(gated(rt), conf_body(rt)))
    assert run(runner({"document": "x"}))["output"] == {"employee_id": "E1"}


def test_refresh_to_new_revision_is_gated_before_the_swap(rt, tmp_path):
    http = Conf(gated(rt), conf_body(rt))
    runner = gate_load(rt, tmp_path, http)
    cached, expires = cache_file(tmp_path).read_text(), runner._expires_at

    new = gated(rt, revision_id="rev-2")
    http.response, http.conf = ok(new), conf_body(rt, bad=20)
    http.conf_calls.clear()
    run(runner.refresh())  # plan_digest is unchanged, so the passing result is reused
    assert runner._bundle["revision_id"] == "rev-2"

    models = {GLINER: {"hf_revision": "r3"}}
    bad = gated(rt, revision_id="rev-3", local_models=models)
    http.response = ok(bad)
    cached, expires = cache_file(tmp_path).read_text(), runner._expires_at
    run(runner.refresh())
    assert runner._bundle["revision_id"] == "rev-2" and runner._expires_at == expires
    assert cache_file(tmp_path).read_text() == cached and runner._stopped is None
    assert run(runner({"document": "x"}))["output"] == {"employee_id": "E1"}

    good = gated(rt, revision_id="rev-4", local_models={GLINER: {"hf_revision": "r4"}})
    http.response, http.conf = ok(good), conf_body(rt)
    run(runner.refresh())
    assert runner._bundle["revision_id"] == "rev-4"


def test_refresh_conformance_401_stops_serving(rt, tmp_path):
    http = Conf(gated(rt), conf_body(rt))
    runner = gate_load(rt, tmp_path, http)
    http.response = ok(gated(rt, revision_id="rev-2", local_models={GLINER: {"hf_revision": "z"}}))
    http.conf = (401, {}, b"")
    run(runner.refresh())
    with pytest.raises(rt.LocalExecutionError, match="^unauthorized$"):
        run(runner({"document": "x"}))


def test_offline_start_needs_a_cached_result(rt, tmp_path):
    http = Conf(gated(rt), conf_body(rt))
    gate_load(rt, tmp_path, http)
    down = Conf(gated(rt), OSError("down"))
    down.response = OSError("down")
    runner = gate_load(rt, tmp_path, down)  # bundle cache + result cache
    assert down.conf_calls == []
    assert run(runner({"document": "x"}))["output"] == {"employee_id": "E1"}

    for f in (tmp_path / "verify").rglob("*.json"):
        f.unlink()
    with pytest.raises(rt.LocalExecutionError, match="^conformance_unavailable$"):
        gate_load(rt, tmp_path, down)


def flaky(raises):
    """Provider whose first ``raises`` calls fail (environment trouble), then succeed."""
    state = {"n": 0}

    async def provider(model, payload):
        state["n"] += 1
        if state["n"] <= raises:
            raise ValueError("provider_unavailable")
        return {"employee_id": "E1"}

    return provider


def test_all_items_raising_raises_the_reason_and_caches_nothing(rt, tmp_path):
    http = Conf(gated(rt), conf_body(rt))
    with pytest.raises(rt.LocalExecutionError, match="^provider_unavailable$"):
        gate_load(rt, tmp_path, http, provider=failing("provider_unavailable"))
    assert list((tmp_path / "verify").rglob("*.json")) == []
    gate_load(rt, tmp_path, http)  # environment fixed: replays and passes
    assert len(http.conf_calls) == 2


def test_partial_raises_below_threshold_is_not_cached(rt, tmp_path):
    http = Conf(gated(rt), conf_body(rt))
    with pytest.raises(rt.LocalExecutionError, match="^conformance_failed$"):
        gate_load(rt, tmp_path, http, provider=flaky(5))
    assert list((tmp_path / "verify").rglob("*.json")) == []
    gate_load(rt, tmp_path, http)
    assert len(http.conf_calls) == 2


def test_partial_raises_still_passing_is_cached(rt, tmp_path):
    http = Conf(gated(rt), conf_body(rt, n=40))
    gate_load(rt, tmp_path, http, provider=flaky(2))  # 38/40 = 0.95
    assert len(list((tmp_path / "verify").rglob("*.json"))) == 1
    gate_load(rt, tmp_path, http)
    assert len(http.conf_calls) == 1


def test_verify_all_raising_raises_and_partial_failure_is_not_cached(rt, tmp_path):
    runner = load(
        rt,
        tmp_path,
        Conf(gated(rt), conf_body(rt)),
        provider=failing("model_identity_changed"),
    )
    with pytest.raises(rt.LocalExecutionError, match="^model_identity_changed$"):
        run(runner.verify())
    runner = load(rt, tmp_path, Conf(gated(rt), conf_body(rt)), provider=flaky(5))
    assert run(runner.verify())["passed"] is False
    assert list((tmp_path / "verify").rglob("*.json")) == []


@pytest.mark.parametrize("junk", ["not json", "[]", '{"agreement": 1.0}', '{"passed": "yes"}'])
def test_corrupt_result_file_is_a_miss(rt, tmp_path, junk):
    http = Conf(gated(rt), conf_body(rt))
    gate_load(rt, tmp_path, http)
    (result_file,) = (tmp_path / "verify").rglob("*.json")
    result_file.write_text(junk)
    gate_load(rt, tmp_path, http)
    assert len(http.conf_calls) == 2


def test_whole_model_entry_is_in_the_result_key(rt, tmp_path):
    entry = {"hf_revision": "r1", "threshold": 0.5}
    http = Conf(gated(rt, local_models={GLINER: entry}), conf_body(rt))
    gate_load(rt, tmp_path, http)
    http2 = Conf(gated(rt, local_models={GLINER: {**entry, "threshold": 0.6}}), conf_body(rt))
    gate_load(rt, tmp_path, http2)
    assert len(http2.conf_calls) == 1


def test_malformed_local_models_never_raises_from_refresh(rt, tmp_path):
    http = Conf(gated(rt), conf_body(rt))
    runner = gate_load(rt, tmp_path, http)
    body = gated(rt, revision_id="rev-2")
    del body[0]["local_models"]  # unsigned field removal: re-sign
    from seldon_runtime.contracts import digest

    http.response = ok((body[0], digest({k: v for k, v in body[0].items() if k != "expires_at"})))
    run(runner.refresh())
    assert runner._bundle["revision_id"] == "rev-1"


def test_telemetry_batches_counts_without_payloads_and_optout(rt, tmp_path):
    async def scenario(enabled):
        from uuid import uuid4

        http = Http(ok(make_bundle(rt, revision_id=str(uuid4()))), post=(200, {}, b"{}"))
        runner = await rt.Runner.load(
            "fn-1",
            KEY,
            cache_dir=tmp_path,
            http=http,
            provider=local_ok,
            allow_unverified=True,
            telemetry=enabled,
        )
        await runner({"document": "SENTINEL PRIVATE TEXT"})
        assert not http.posts
        await runner.flush_telemetry()
        await runner.aclose()
        return http.posts

    reports = run(scenario(True))
    usage = [json.loads(call[3]) for call in reports if call[1].endswith("/local-usage")]
    assert usage and usage[0]["requests"] == 1
    assert sum(usage[0]["latency_histogram"]) == 1
    assert "SENTINEL" not in repr(reports) and "employee_id" not in repr(reports)
    assert run(scenario(False)) == []


def test_local_status_is_read_only_and_reports_actual_revision(rt, tmp_path):
    http = Http(ok(make_bundle(rt)))
    runner = load(rt, tmp_path, http)
    before = len(http.calls)
    status = runner.status()
    assert status["revision_id"] == "rev-1"
    assert status["verification"] is None
    assert status["state"] == "serving"
    assert status["reason"] is None
    assert len(http.calls) == before


def test_telemetry_retries_same_batch_then_drops_without_raising(rt, tmp_path):
    async def scenario():
        from uuid import uuid4

        http = Http(ok(make_bundle(rt, revision_id=str(uuid4()))), post=(503, {}, b"{}"))
        runner = await rt.Runner.load(
            "fn-1",
            KEY,
            cache_dir=tmp_path,
            http=http,
            provider=local_ok,
            allow_unverified=True,
            telemetry=True,
        )
        await runner({"document": "Ada"})
        for _ in range(4):
            await runner.flush_telemetry()
        await runner.aclose()
        return [json.loads(call[3]) for call in http.posts if call[1].endswith("/local-usage")]

    reports = run(scenario())
    assert len(reports) == 3
    assert reports[0] == reports[1] == reports[2]


def test_public_status_exposes_verification_and_refused_refresh(rt, tmp_path):
    http = Conf(gated(rt), conf_body(rt))
    runner = gate_load(rt, tmp_path, http)
    before = len(http.calls)
    assert runner.status()["verification"]["passed"] is True
    assert len(http.calls) == before
    http.response = ok(gated(rt, revision_id="rev-2", runtime_min_version="999.0.0"))
    run(runner.refresh())
    status = runner.status()
    assert status["revision_id"] == "rev-1"
    assert status["state"] == "refresh_refused"
    assert status["reason"] == "runtime_upgrade_required"


def test_flush_bounds_network_work_and_defers_remaining_batches(rt, tmp_path):
    async def scenario():
        from uuid import uuid4

        http = Http(ok(make_bundle(rt, revision_id=str(uuid4()))), post=(200, {}, b"{}"))
        runner = await rt.Runner.load(
            "fn-1",
            KEY,
            cache_dir=tmp_path,
            http=http,
            provider=local_ok,
            allow_unverified=True,
            telemetry=True,
        )
        for _ in range(7):
            await runner({"document": "Ada"})
            http.bundle_response = ok(make_bundle(rt, revision_id=str(uuid4())))
            await runner.refresh()
        await runner.flush_telemetry()
        assert len([c for c in http.posts if c[1].endswith("/local-usage")]) == 4
        await runner.aclose()
        reports = [json.loads(c[3]) for c in http.posts if c[1].endswith("/local-usage")]
        assert len(reports) == 7 and sum(r["requests"] for r in reports) == 7

    run(scenario())
