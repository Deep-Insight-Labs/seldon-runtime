# Seldon Runtime

`seldon-runtime` is the standalone Python interpreter and local execution client
for eligible Seldon Functions. The package name is **`seldon-runtime`**, and the
Python import is `seldon_runtime`. This project is separate from SeldonIO.

A single generic runtime executes authorized Function bundles on your hardware.
Optimization and hosted comparison happen on Seldon. The runtime contains no
optimizer, billing service, dashboard or server infrastructure. Its first local
model adapter uses pinned GLiNER2 open weights for extraction and classification.

The first preview, [\`0.1.1.dev0\`](https://pypi.org/project/seldon-runtime/0.1.1.dev0/),
is available on PyPI. Local execution requires a Seldon deployment with bundle
delivery enabled for your workspace.

## Install

Python 3.11 or later is required. The base interpreter uses Pydantic, jsonschema
and tiktoken. Local GLiNER inference additionally needs PyTorch and the model extra.
The commands below pin the published preview.

On Linux, install CPU PyTorch explicitly to avoid downloading CUDA dependencies:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'seldon-runtime[gliner]==0.1.1.dev0'
```

On macOS:

```sh
python -m pip install torch
python -m pip install 'seldon-runtime[gliner]==0.1.1.dev0'
```

In a uv environment, use `uv pip install` for the same commands. In a Linux uv
project, persist the dedicated torch index:

```sh
uv add torch --index pytorch=https://download.pytorch.org/whl/cpu
uv add 'seldon-runtime[gliner]==0.1.1.dev0'
```

The model weights and tokenizer files download on first use. Cache the pinned
snapshot in advance for restricted networks. Token-chunked plans also need the
tiktoken BPE cache (`TIKTOKEN_CACHE_DIR`). Model weights are not bundled in the wheel.

## Run a Function locally

Only activated, locally eligible v3/v4 revisions can be loaded. Hosted-only models,
reference-bound Functions and v5 plans are ineligible. Authorization uses your
workspace API key; it does not grant management access.

```python
import asyncio
import os

from seldon_runtime.runner import Runner


async def main():
    runner = await Runner.load(
        "FUNCTION_UUID",
        os.environ["SELDON_API_KEY"],
        fallback="raise",
        telemetry=False,
    )
    try:
        result = await runner({"document": "Ada works at Acme"})
        print(result["output"])
        print(runner.status())
    finally:
        await runner.aclose()


asyncio.run(main())
```

Use the input names from your Function contract. Each result also reports
`fallback_used` and `fallback_reason`. `LocalExecutionError.reason` is a bounded
machine-readable code; errors do not expose credentials or internal paths.

`fallback="raise"` prevents hosted inference fallback. `fallback="seldon"` sends
failed calls to Seldon and incurs ordinary hosted usage charges. An explicit
`{"endpoint": "...", "api_key": "..."}` uses your own OpenAI-compatible endpoint.
A failed initial verification raises before installing the bundle.

The [Seldon Python SDK](https://github.com/Deep-Insight-Labs/seldon-python) provides
optional typed declarations and `seldon function pull`, `verify` and `run` commands
using this runtime. It is not required for direct `Runner` use.

## Verification and refresh

`Runner.load()` checks at least 95% exact agreement against the revision's held-out
conformance set before installing a new revision/model/runtime combination.
`await runner.verify()` reruns up to 200 local inputs and reports agreement,
sample count and p50/p95 latency. This measures agreement on that set, not business
correctness or coverage of all inputs. `allow_unverified=True` explicitly skips
verification and is not parity evidence.

Bundles are cached without API keys, revalidated at load and every 15 minutes on
a long-lived event loop. A failed refresh keeps the last validated revision for
at most 24 hours from its last successful check; expiry applies your fallback
policy. A disabled, deleted or unauthorized Function stops serving when observed.
Short-lived loops may not finish background refresh, so the 24-hour expiry is the
upper bound. Local execution still needs these control-plane calls.

`runner.status()` reads current local state without network or inference. Optional
reporting sends bounded counts, latency histograms and status outside inference;
it never sends inputs, outputs or payload hashes. `telemetry=False` disables both
usage and status reporting. Reports are best effort and are not billing authority.
Always close the runner to stop background tasks and attempt a bounded flush.

## Development

```sh
python -m pip install -e '.[dev]'
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m build
python -m twine check dist/*
python scripts/verify_artifacts.py dist/*
```

Default tests replace model weights and HTTP at explicit runtime seams. They
exercise execution, output normalization, authorization refusals, conformance,
refresh, cache handling, fallback and reporting. Synthetic frozen plan/spec
fixtures check serialization and digest stability. No private repository or
server package is required. Run `pytest -m local_model` only after installing the
GLiNER extra; it downloads or reuses the pinned model weights.

`tests/installed_local_probe.py` verifies a real installed GLiNER runtime in a
server-free environment. Base-wheel installation alone does not qualify native
model dependencies or performance. Existing model probes cover macOS ARM64 and
Linux ARM64 CPU; other hardware and Windows are not claimed as qualified.

## Releases

The publishing workflow is `.github/workflows/runtime-publish.yml` in
`Deep-Insight-Labs/seldon-runtime`. Register project `seldon-runtime` as a trusted
publisher on TestPyPI with environment `testpypi`, and on PyPI with environment
`pypi`. A `runtime-v<version>` tag must match `pyproject.toml` and `__version__`.

Publication is disabled unless `SELDON_RUNTIME_TESTPYPI_PUBLISH_ENABLED` and
`SELDON_RUNTIME_PUBLISH_ENABLED` are enabled separately. The workflow checks the
standalone project, builds wheel and sdist, publishes to TestPyPI, downloads and
installs that exact wheel with normal dependencies from PyPI, then publishes the
original checked artifacts to PyPI. A manual branch dispatch only builds artifacts.
No stored registry token is required. Registry versions are immutable.

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Optional GLiNER2 code and
downloaded weights retain their own upstream licenses. The release is scoped to
this repository; it does not license the separately operated Seldon service.
