"""Customer-facing ``Runner``: load an activated Function bundle and run it locally."""

import asyncio
import hashlib
import http.client
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from . import __version__
from .contracts import FunctionSpec, Plan, Step, digest
from .executor import execute, failure_reason
from .local import LocalModels
from .local_telemetry import bounded_reason
from .telemetry import Telemetry
from .tool_wire import decode_output

DEFAULT_BASE_URL = "https://api.seldon-ai.com"
BUNDLE_VERSION = "function-bundle-v1"
INELIGIBLE = {
    "not_activated",
    "hosted_only_model",
    "plan_version",
    "reference_bound",
    "disabled",
}
GRACE_SECONDS = 86400
REFRESH_SECONDS = 900
AGREEMENT_MIN = 0.95
CONFORMANCE_VERSION = "function-conformance-v1"
CONFORMANCE_REF = re.compile(
    r"/v1/seldon/functions/[A-Za-z0-9_-]{1,80}/conformance\?revision=[A-Za-z0-9_-]{1,80}"
)
_RESULT_KEYS = {"agreement", "samples", "p50_ms", "p95_ms", "passed"}


class _Cached(NamedTuple):
    etag: str
    expires: float
    bundle: dict
    plan: Plan
    spec: FunctionSpec


class LocalExecutionError(Exception):
    """``str(exc)`` is a bounded lowercase code; ``.reason`` holds the same code."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _now():
    return time.time()


def _release(version):
    match = re.match(r"\d+(?:\.\d+)*", version)
    parts = tuple(int(p) for p in match.group().split(".")) if match else (0,)
    return parts + (0,) * (4 - len(parts))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None  # urllib would forward Authorization to the redirect target


_OPENER = urllib.request.build_opener(_NoRedirect)


async def _urllib_http(method, url, headers, body):
    def call():
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            timeout = 10 if url.endswith(("/local-usage", "/local-status")) else 60
            with _OPENER.open(request, timeout=timeout) as response:
                return response.status, dict(response.headers.items()), response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers.items()), exc.read()
        except http.client.HTTPException:
            raise OSError("transport_error") from None

    status, headers, data = await asyncio.to_thread(call)
    return status, {k.lower(): v for k, v in headers.items()}, data


def _atomic_write(path, data):
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent)
        with os.fdopen(fd, "w") as handle:  # mkstemp creates the file 0o600
            json.dump(data, handle)
        os.replace(tmp, path)
    except OSError:
        # ponytail: caches are only optimisations; without them offline grace/verify reuse is lost.
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _signed(body):
    return digest({k: v for k, v in body.items() if k != "expires_at"})


def _validate(body, etag):
    try:
        if body["version"] != BUNDLE_VERSION or _signed(body) != etag:
            raise ValueError
        plan = Plan.model_validate(body["plan"])
        spec = FunctionSpec.model_validate(body["spec"])
        if plan.digest != body["plan_digest"] or spec.digest != body["contract_digest"]:
            raise ValueError
        expires = datetime.fromisoformat(body["expires_at"]).timestamp()
        minimum = _release(body["runtime_min_version"])
    except Exception:
        raise LocalExecutionError("bundle_invalid") from None
    if minimum > _release(__version__):
        raise LocalExecutionError("runtime_upgrade_required")
    return expires, plan, spec


class Runner:
    def __init__(
        self,
        function_id,
        api_key,
        fallback,
        base_url,
        cache_file,
        http,
        provider,
        allow_unverified,
        telemetry=True,
    ):
        self._function_id = function_id
        self._api_key = api_key
        self._fallback = fallback
        self._base_url = base_url.rstrip("/")
        self._cache_file = cache_file
        self._http = http
        self._provider = provider
        self._injected = provider is not None
        self._allow_unverified = allow_unverified
        self._bundle = self._plan = self._spec = self._etag = self._expires_at = None
        self._stopped = self._task = None
        self._refused = None  # why the last refresh declined a newer bundle
        self._checked_at = 0.0
        self._verification = None
        self._telemetry = Telemetry(self, telemetry)

    @classmethod
    async def load(
        cls,
        function_id,
        api_key,
        *,
        fallback="seldon",
        base_url=DEFAULT_BASE_URL,
        cache_dir=None,
        http=None,
        provider=None,
        allow_unverified=False,
        telemetry=True,
    ):
        if fallback not in ("seldon", "raise") and not (
            isinstance(fallback, dict)
            and set(fallback) == {"endpoint", "api_key"}
            and all(isinstance(v, str) and v for v in fallback.values())
        ):
            raise ValueError("invalid_fallback")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", str(function_id)):
            raise LocalExecutionError("function_not_found")
        root = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "seldon"
        env = hashlib.sha256(base_url.rstrip("/").encode()).hexdigest()[:16]  # per environment
        runner = cls(
            function_id,
            api_key,
            fallback,
            base_url,
            root / "bundles" / env / f"{function_id}.json",
            http or _urllib_http,
            provider,
            allow_unverified,
            telemetry,
        )
        await runner._fetch(runner._read_cache())
        runner._checked_at = _now()
        runner._task = asyncio.create_task(runner._refresh_loop())
        runner._telemetry.start()
        return runner

    async def _refresh_loop(self):
        while True:
            # a task restarted on a new loop checks at once when a check is overdue
            await asyncio.sleep(max(0.0, self._checked_at + REFRESH_SECONDS - _now()))
            self._checked_at = _now()
            try:
                await self.refresh()
            except Exception:  # ponytail: one failed check must not end revalidation
                pass

    async def aclose(self):
        if self._task:
            try:
                self._task.cancel()
                await self._task
            except (
                asyncio.CancelledError,
                RuntimeError,
            ):  # RuntimeError: its loop is closed
                pass
        await self._telemetry.close()

    async def flush_telemetry(self):
        """Await one bounded best-effort flush; never changes execution or funding."""
        await self._telemetry.flush()

    def status(self):
        """Local observation only: no model, network or cache reads are performed."""
        reason = self._stopped or self._refused
        state = "stopped" if self._stopped else "refresh_refused" if self._refused else "serving"
        if not self._stopped and _now() >= self._expires_at:
            state, reason = "expired", reason or "bundle_expired"
        return {
            "reporting_id": self._telemetry.reporting_id,
            "revision_id": self._bundle["revision_id"],
            "runtime_version": __version__,
            "model_versions": {
                k: v["hf_revision"] for k, v in self._bundle["local_models"].items()
            },
            "verification": dict(self._verification) if self._verification else None,
            "state": state,
            "reason": bounded_reason(reason) if reason else None,
        }

    async def refresh(self):
        """One conditional revalidation; never raises, never touches the request path."""
        current = _Cached(self._etag, self._expires_at, self._bundle, self._plan, self._spec)
        try:
            await self._fetch(current)
        except LocalExecutionError as exc:
            if exc.reason.startswith("local_ineligible") or exc.reason in (
                "function_not_found",
                "unauthorized",
            ):
                self._stopped = exc.reason
                try:  # a restart must not serve a revoked bundle from disk
                    self._cache_file.unlink()
                except OSError:
                    pass
            elif exc.reason != "bundle_unavailable":  # Seldon answered; the bundle was refused
                self._refused = exc.reason

    async def _install(self, etag, expires, bundle, plan, spec):
        provider = self._provider
        if bundle is not self._bundle:  # the same object was gated when it was installed
            try:
                models = bundle["local_models"]
                if not self._injected:
                    provider = LocalModels(models).provider
            except Exception:  # a malformed bundle must not raise out of refresh()
                raise LocalExecutionError("bundle_invalid") from None
            if not self._allow_unverified:
                result = self._read_result(bundle)
                if result is None:
                    result, cacheable = await self._replay(bundle, plan, spec, provider)
                    if cacheable:
                        self._write_result(bundle, result)
                if not result["passed"]:
                    raise LocalExecutionError("conformance_failed")
        self._set(etag, expires, bundle, plan, spec, provider)

    def _set(self, etag, expires, bundle, plan, spec, provider):
        self._provider = provider
        # one synchronous swap: a call sees the old or the new bundle, never a mix
        self._etag, self._expires_at, self._bundle = etag, expires, bundle
        self._plan, self._spec = plan, spec
        self._verification = self._read_result(bundle)

    def _read_cache(self):
        try:
            cached = json.loads(self._cache_file.read_text())
            etag, expires, bundle = (
                cached["etag"],
                float(cached["expires_at"]),
                cached["bundle"],
            )
            _, plan, spec = _validate(bundle, etag)
            return _Cached(etag, expires, bundle, plan, spec)
        except Exception:
            return None

    def _write_cache(self):
        _atomic_write(
            self._cache_file,
            {
                "etag": self._etag,
                "expires_at": self._expires_at,
                "bundle": self._bundle,
            },
        )

    def _result_file(self, bundle):
        models = bundle["local_models"]  # whole entries: a changed pin or threshold re-verifies
        key = digest({"plan": bundle["plan_digest"], "models": models, "runtime": __version__})
        # <cache root>/bundles/<env>/<fn>.json -> <cache root>/verify/<env>/<key>.json
        return self._cache_file.parents[2] / "verify" / self._cache_file.parent.name / f"{key}.json"

    def _read_result(self, bundle):
        try:
            result = json.loads(self._result_file(bundle).read_text())
            ok = isinstance(result, dict) and set(result) == _RESULT_KEYS
            return result if ok and isinstance(result["passed"], bool) else None
        except Exception:
            return None

    def _write_result(self, bundle, result):
        _atomic_write(self._result_file(bundle), result)

    async def verify(self):
        """Replay the conformance set against the current bundle; always re-measures."""
        bundle = self._bundle
        result, cacheable = await self._replay(bundle, self._plan, self._spec, self._provider)
        if self._bundle is bundle:
            self._verification = result
        if cacheable:
            self._write_result(bundle, result)
        return result

    async def _conformance(self, bundle):
        ref = bundle.get("conformance_ref")
        if not isinstance(ref, str) or not CONFORMANCE_REF.fullmatch(ref):
            raise LocalExecutionError("conformance_unavailable")  # never send the key elsewhere
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            status, _, data = await self._http("GET", f"{self._base_url}{ref}", headers, None)
        except OSError:
            status = None
        if status in (401, 403):
            raise LocalExecutionError("unauthorized")
        try:
            body = json.loads(data) if status == 200 else None
            items = body["items"]
            if (
                body["version"] != CONFORMANCE_VERSION
                or body["plan_digest"] != bundle["plan_digest"]
            ):
                raise ValueError
            if (
                not isinstance(items, list)
                or not items
                or not all(
                    isinstance(i, dict)
                    and isinstance(i.get("input"), dict)
                    and isinstance(i.get("output_digest"), str)
                    for i in items
                )
            ):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            raise LocalExecutionError("conformance_unavailable") from None
        return items

    async def _replay(self, bundle, plan, spec, provider):
        items = await self._conformance(bundle)
        plan = plan.model_copy(update={"fallback": False})  # measure the local program only

        async def local_only(model, payload):  # the fallback route is never used here
            if "primitive" not in payload:
                raise ValueError("provider_unavailable")
            return await provider(model, payload)

        agreed, times, reason = 0, [], None
        for item in items:  # ponytail: sequential; batch if verify time bites (max 200 items)
            try:
                result = await execute(
                    plan,
                    spec,
                    item["input"],
                    local_only,
                    contract=bundle["optimizer_contract_version"],
                    serving=True,
                )
            except Exception as exc:
                reason = reason or failure_reason(exc)
                continue
            times.append(result.elapsed_ms)
            agreed += digest(result.output) == item["output_digest"]
        if not times:  # nothing ran: an environment problem, not a measurement
            raise LocalExecutionError(reason)
        times.sort()

        def pct(p):  # nearest rank
            return int(times[math.ceil(p * len(times) / 100) - 1]) if times else None

        agreement = agreed / len(items)
        result = {
            "agreement": agreement,
            "samples": len(items),
            "p50_ms": pct(50),
            "p95_ms": pct(95),
            "passed": agreement >= AGREEMENT_MIN,
        }
        # a failing result is cached only if every item ran: partial raises may be environmental
        return result, result["passed"] or len(times) == len(items)

    async def _fetch(self, cached):
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if cached:
            headers["If-None-Match"] = f'"{cached.etag}"'
        url = f"{self._base_url}/v1/seldon/functions/{self._function_id}/bundle"
        try:
            status, response, data = await self._http("GET", url, headers, None)
        except OSError:
            status = None
        if status == 200:
            try:
                body = json.loads(data)
                etag = response.get("etag", "").removeprefix("W/").strip('"')
                expires, plan, spec = _validate(body, etag)
            except (ValueError, AttributeError):
                raise LocalExecutionError("bundle_invalid") from None
            await self._install(etag, expires, body, plan, spec)
        elif status == 304 and cached:
            await self._install(*cached._replace(expires=_now() + GRACE_SECONDS))
        elif status == 409:
            try:
                reason = (json.loads(data).get("details") or {}).get("reason")
            except (ValueError, AttributeError):
                reason = None
            raise LocalExecutionError(
                f"local_ineligible_{reason}" if reason in INELIGIBLE else "local_ineligible"
            )
        elif status in (401, 403):
            raise LocalExecutionError("unauthorized")
        elif status == 404:
            raise LocalExecutionError("function_not_found")
        elif cached and _now() < cached.expires:  # offline grace
            await self._install(*cached)
            return
        else:
            raise LocalExecutionError("bundle_unavailable")
        self._stopped = self._refused = None
        self._write_cache()

    async def _post(self, url, key, model, payload):
        body = {"model": model, **payload}
        body.pop("seldon_output_token_policy", None)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            status, _, data = await self._http("POST", url, headers, json.dumps(body).encode())
        except OSError:
            raise ValueError("provider_unavailable") from None
        if status != 200:
            raise ValueError("provider_unavailable")
        try:
            raw = json.loads(data)
            choice = raw["choices"][0]
            message = choice["message"]
            refused = choice.get("finish_reason") == "content_filter" or message.get("refusal")
            output = None if refused else decode_output(raw, payload)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise ValueError("invalid_provider_output") from None
        if refused:
            raise ValueError("provider_refusal")
        return output

    async def _route(self, provider, model, payload):
        if "primitive" in payload:
            return await provider(model, payload)
        if isinstance(self._fallback, dict):
            endpoint, key = (
                self._fallback["endpoint"].rstrip("/"),
                self._fallback["api_key"],
            )
        elif self._fallback == "seldon":
            endpoint, key = f"{self._base_url}/v1", self._api_key
        else:
            raise ValueError("provider_unavailable")
        return await self._post(f"{endpoint}/chat/completions", key, model, payload)

    async def __call__(self, inputs):
        started, bundle = time.monotonic(), self._bundle
        try:
            result = await self._call(inputs)
        except Exception:
            self._telemetry.record(bundle, (time.monotonic() - started) * 1000, failed=True)
            raise
        self._telemetry.record(
            bundle,
            (time.monotonic() - started) * 1000,
            fallback=result["fallback_reason"],
        )
        return result

    async def _call(self, inputs):
        if self._stopped:  # definitive answer from Seldon: no fallback policy
            raise LocalExecutionError(self._stopped)
        if self._task is None or self._task.done():
            # ponytail: the loop that ran load() may be gone; no bundle request is made here
            self._task = asyncio.create_task(self._refresh_loop())
        plan, spec, bundle, provider = (
            self._plan,
            self._spec,
            self._bundle,
            self._provider,
        )
        expired = None
        if _now() >= self._expires_at:  # Seldon unreachable past the grace window
            if self._fallback == "raise" or not plan.fallback:
                raise LocalExecutionError(self._refused or "bundle_expired")
            plan = Plan(
                steps=[Step(id=c.id, op="llm", model=c.model, call=c.id) for c in spec.calls]
            )
            expired = self._refused or "bundle_expired"
        elif self._fallback == "raise":
            plan = plan.model_copy(update={"fallback": False})
        try:
            result = await execute(
                plan,
                spec,
                inputs,
                lambda model, payload: self._route(provider, model, payload),
                contract=bundle["optimizer_contract_version"],
                serving=True,
            )
        except Exception as exc:
            raise LocalExecutionError(failure_reason(exc)) from None
        return {
            "output": result.output,
            "fallback_used": bool(expired) or result.fallback_used,
            "fallback_reason": expired or result.fallback_reason,
        }
