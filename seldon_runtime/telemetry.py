"""Best-effort bounded aggregate reports, outside the inference path."""

import asyncio
import json
import logging
import time
from uuid import uuid4

from . import __version__
from .local_telemetry import (
    LATENCY_BOUNDS_MS,
    LocalStatusReport,
    LocalUsageBatch,
    bounded_reason,
)

logger = logging.getLogger(__name__)
REPORT_SECONDS = 60
MAX_PENDING = 32
MAX_SEND = 4


class Telemetry:
    def __init__(self, runner, enabled):
        self.runner = runner
        self.enabled = enabled
        self.counts = {}
        self.pending = []
        self.dropped = 0
        self.task = None
        self.reporting_id = str(uuid4())
        self.flushing = False

    def start(self):
        if self.enabled and (self.task is None or self.task.done()):
            self.task = asyncio.create_task(self.loop())

    def record(self, bundle, elapsed_ms, *, fallback=None, failed=False):
        if not self.enabled:
            return
        self.start()
        revision = bundle["revision_id"]
        if revision not in self.counts:
            if len(self.counts) >= MAX_PENDING:
                self.dropped += 1
                return  # bounded best-effort telemetry, never billing authority
            self.counts[revision] = {
                "batch_id": str(uuid4()),
                "revision_id": revision,
                "runtime_version": __version__,
                "model_versions": {k: v["hf_revision"] for k, v in bundle["local_models"].items()},
                "requests": 0,
                "failures": 0,
                "fallbacks": {},
                "latency_histogram": [0] * 6,
            }
        row = self.counts[revision]
        if row["requests"] >= 1_000_000:
            self.dropped += 1
            return
        row["requests"] += 1
        row["failures"] += bool(failed)
        row["latency_histogram"][sum(elapsed_ms > b for b in LATENCY_BOUNDS_MS)] += 1
        if fallback:
            reason = bounded_reason(fallback)
            row["fallbacks"][reason] = row["fallbacks"].get(reason, 0) + 1

    async def loop(self):
        while True:
            await asyncio.sleep(REPORT_SECONDS)
            await self.flush()

    async def send(self, path, report):
        runner = self.runner
        headers = {
            "Authorization": f"Bearer {runner._api_key}",
            "Content-Type": "application/json",
        }
        started = time.monotonic()
        context = {"entity_id": runner._function_id, "correlation_id": self.reporting_id}
        logger.info(
            "function_local_report_started",
            extra={
                **context,
                "operation": "function_local_report_started",
                "endpoint": path,
                "method": "POST",
            },
        )
        try:
            status, _, response = await asyncio.wait_for(
                runner._http(
                    "POST",
                    f"{runner._base_url}/v1/seldon/functions/{runner._function_id}/{path}",
                    headers,
                    json.dumps(report).encode(),
                ),
                timeout=10,
            )
            if status in (200, 201, 202, 204):
                logger.info(
                    "function_local_report_completed",
                    extra={
                        **context,
                        "operation": "function_local_report_completed",
                        "duration_ms": round((time.monotonic() - started) * 1000),
                        "response_bytes": len(response),
                    },
                )
                return True
            logger.warning(
                "function_local_report_rejected",
                extra={
                    "operation": "function_local_report_rejected",
                    "entity_id": runner._function_id,
                    "correlation_id": self.reporting_id,
                    "status": status,
                },
            )
            return False
        except Exception as exc:
            logger.warning(
                "function_local_report_unavailable",
                extra={
                    "operation": "function_local_report_unavailable",
                    "entity_id": runner._function_id,
                    "correlation_id": self.reporting_id,
                    "error_type": type(exc).__name__,
                },
            )
            return False

    async def flush(self):
        if not self.enabled or self.flushing:
            return
        self.flushing = True
        try:
            for row in self.counts.values():
                if len(self.pending) >= MAX_PENDING:
                    self.dropped += row["requests"]
                else:
                    self.pending.append([row, 0])
            self.counts.clear()
            # Bound a manual/background flush to five HTTP budgets including status.
            for entry in list(self.pending)[:MAX_SEND]:
                row, attempts = entry
                try:
                    body = LocalUsageBatch.model_validate(row).model_dump(mode="json")
                except ValueError:
                    self.dropped += row["requests"]
                    self.pending.remove(entry)
                    continue
                delivered = await self.send("local-usage", body)
                entry[1] = attempts + 1
                if delivered or entry[1] >= 3:
                    if not delivered:
                        self.dropped += row["requests"]
                    self.pending.remove(entry)
            try:
                status = LocalStatusReport.model_validate(self.runner.status()).model_dump(
                    mode="json"
                )
            except ValueError:
                return
            await self.send("local-status", status)
        finally:
            self.flushing = False
            if self.dropped:
                logger.warning(
                    "function_local_report_dropped",
                    extra={
                        "operation": "function_local_report_dropped",
                        "entity_id": self.runner._function_id,
                        "correlation_id": self.reporting_id,
                        "count": self.dropped,
                    },
                )
                self.dropped = 0

    async def close(self):
        if self.task:
            try:
                self.task.cancel()
                await self.task
            except (asyncio.CancelledError, RuntimeError):
                pass
        try:
            await asyncio.wait_for(self.flush(), timeout=10)
        except TimeoutError:
            logger.warning(
                "function_local_report_shutdown_timeout",
                extra={
                    "operation": "function_local_report_shutdown_timeout",
                    "entity_id": self.runner._function_id,
                    "correlation_id": self.reporting_id,
                },
            )
