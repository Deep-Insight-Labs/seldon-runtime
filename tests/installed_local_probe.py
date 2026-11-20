"""Run with ``python -I`` after installing the wheel and its gliner extra.

No server package or source-path injection; synthetic text and pinned real weights.
Use HF_HUB_OFFLINE=1 when the exact pinned snapshot is already cached.
"""

import asyncio
import importlib.util
import json
import platform
from importlib.metadata import version

import torch

from seldon_runtime.local import LocalModels

assert importlib.util.find_spec("seldon") is None, "server package leaked into runtime installation"


async def main():
    rows = await LocalModels().provider(
        "fastino/gliner2.5-multi-v1",
        {
            "messages": [{"role": "user", "content": "ignored for spans"}],
            "primitive": {
                "kind": "extract",
                "source": "Ada Lovelace met Bob Smith in London.",
                "config": {
                    "mode": "spans",
                    "labels": {"person": "a named person", "city": "a city"},
                },
            },
        },
    )
    assert {(row["label"], row["text"], row["start"], row["end"]) for row in rows} == {
        ("person", "Ada Lovelace", 0, 12),
        ("person", "Bob Smith", 17, 26),
        ("city", "London", 30, 36),
    }
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "runtime": version("seldon-runtime"),
                "gliner2": version("gliner2"),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "observations": len(rows),
                "passed": True,
            }
        )
    )


asyncio.run(main())
