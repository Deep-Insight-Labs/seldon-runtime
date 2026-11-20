"""Native record-schema translation and typed native contract failure codes."""


def structured_schema(schema):
    """Translate only representable record schemas; never flatten relationships."""
    if schema.get("type") != "object" or not schema.get("properties"):
        raise ValueError("unsupported_native_schema")
    structures = {}
    for name, collection in schema["properties"].items():
        record = collection.get("items", {})
        if (
            collection.get("type") != "array"
            or record.get("type") != "object"
            or not record.get("properties")
        ):
            raise ValueError("unsupported_native_schema")
        fields = []
        for field, value in record["properties"].items():
            if value.get("type") not in ("string", ["string", "null"], ["null", "string"]):
                raise ValueError("unsupported_native_schema")
            fields.append(
                {"name": field, "dtype": "str", "description": value.get("description", field)}
            )
        structures[name] = fields
    return {"structures": structures}


# Typed, deterministic native contract failures (W6). A malformed response stays
# ``invalid_provider_output``; none of these is a provider outage or a retry signal.
SCHEMA_MISMATCH = "native_output_schema_mismatch"
INVALID_GROUNDED_SPAN = "invalid_grounded_span"
UNGROUNDED_QUOTATION = "ungrounded_quotation"
NATIVE_CONTRACT_ERRORS = frozenset({SCHEMA_MISMATCH, INVALID_GROUNDED_SPAN, UNGROUNDED_QUOTATION})
