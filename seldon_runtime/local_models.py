"""Pinned local open-weight models. Pure data: staged into the wheel and read by the bundle."""

RUNTIME_MIN_VERSION = "0.1.0.dev0"

LOCAL_MODELS = {
    "fastino/gliner2.5-multi-v1": {
        "loader": "gliner2",
        "hf_repo": "fastino/gliner2.5-multi-v1",
        "hf_revision": "7b005274e82da37dcf1f5fd9d6a2ef413c6a004c",
        # sha256 of every file loaded from the revision; any other bytes are refused.
        "files": {
            "config.json": "3e4a96bf01da1094180f2d8998411c1a9fd6e2ae4b5b3e4bc6950318cee95bb7",
            "encoder_config/config.json": (
                "fa4f9ef2903b5369ab172333aae4574e6a476511d7465845cf59f8360ee18716"
            ),
            "model.safetensors": "c1ff4ec0bc00031c15530b8f3c33d3677f27949e6a0cb52e1247a6224b6c5395",
            "tokenizer.json": "c62446df87ae18ec98b133f8f84fc449a07cc89bbf8ef192a4cb5f9c53777a7a",
            "tokenizer_config.json": (
                "0bf3ea0873234bd9bfdd3853c440395009ac6365a925b91654daed5396d655e1"
            ),
        },
        "threshold": 0.5,
        # The hosted catalog's limits (offering context_length, card max_output_tokens).
        "context_length": 4096,
        "max_output_tokens": 4096,
    }
}
