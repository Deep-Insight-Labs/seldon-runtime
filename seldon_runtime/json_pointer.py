"""RFC 6901 pointer resolution over decoded JSON documents."""

_MISSING = object()


def pointer_value(document, pointer, *, default=_MISSING):
    """Resolve an RFC 6901 pointer; ``default`` replaces a missing path."""
    value = document
    if pointer in {"", None}:
        return value
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif (
            isinstance(value, list)
            and token.isdigit()
            and str(int(token)) == token
            and int(token) < len(value)
        ):
            value = value[int(token)]
        else:
            if default is _MISSING:
                raise ValueError("prediction_path_missing")
            return default
    return value
