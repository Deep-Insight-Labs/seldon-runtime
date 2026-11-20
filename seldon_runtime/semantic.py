"""Pure bounded collection kernels. Entity identity is never inferred by position.

Offsets are half-open Unicode codepoints in the original source. Retrieval is
lexical candidate generation, not semantic truth; omission is explicit evidence.
"""

import re
import unicodedata
from decimal import Decimal

from .contracts import canonical, digest

OPERATORS = {
    "normalize_text": (1, {"casefold"}),
    "chunk_text": (1, {"size", "overlap", "unit"}),
    "retrieve_candidates": (2, {"top_k"}),
    "exact_links": (2, {"unique"}),
    "resolve_links": (2, {"unique"}),
    "assemble_records": (2, {"fields"}),
    "align_spans": (2, set()),
    "project_records": (1, {"fields"}),
    "merge_spans": (2, set()),
    "merge_records": (1, {"fields", "sums"}),
    "merge_labels": (1, {"rule", "order"}),
}
MERGE_RULES = frozenset({"all_agree", "any_true", "max_ordinal", "union"})

NEW_OPERATORS = {"align_spans", "project_records"}
# Advertised only with the chunking construction. Legacy registries stay unchanged.
CHUNK_OPERATORS = {"merge_spans", "merge_records", "merge_labels"}


UNALIGNED_MODES = frozenset({"fail", "drop"})
# Quotations the kernel cannot place. In ``drop`` mode only these omit a record;
# every other validation error still raises.
UNPLACEABLE = ("ungrounded_span", "ambiguous_span")


def align_spans(text, records, *, unaligned="fail", context_policy="strict", diagnostics=None):
    """Ground exact quotations without correcting provider offsets or guessing repeats.

    ``unaligned="drop"`` (span-recipes-v2) omits a quotation that is absent or still
    ambiguous after its prefix/suffix, and grounds the rest unchanged. ``diagnostics``
    (a dict) then receives counts only, never quotation text, since quotes can be PII.
    """
    if (
        not isinstance(unaligned, str)
        or unaligned not in UNALIGNED_MODES
        or not isinstance(context_policy, str)
        or context_policy not in {"strict", "hint"}
    ):
        raise ValueError("invalid_kernel_configuration")
    if not isinstance(text, str) or len(text.encode()) > 262144:
        raise ValueError("invalid_text")
    if not isinstance(records, list) or len(records) > 32:
        raise ValueError("span_limit")
    grounded = []
    dropped = dict.fromkeys(UNPLACEABLE, 0)
    repaired = 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid_span")
        quote, label = record.get("text"), record.get("label")
        prefix, suffix = record.get("prefix", ""), record.get("suffix", "")
        if not isinstance(quote, str) or not quote or not isinstance(label, str) or not label:
            raise ValueError("invalid_span")
        if not isinstance(prefix, str) or not isinstance(suffix, str):
            raise ValueError("invalid_span_context")
        if "start" in record or "end" in record:
            normalized_span = normalize_spans(text, [record])[0]
            start, end = normalized_span["start"], normalized_span["end"]
            if not text[:start].endswith(prefix) or not text[end:].startswith(suffix):
                raise ValueError("invalid_span_context")
            grounded.append(normalized_span)
            continue
        matches, start = [], text.find(quote)
        while start >= 0:
            end = start + len(quote)
            if text[:start].endswith(prefix) and text[end:].startswith(suffix):
                matches.append((start, end))
                if len(matches) > 1:
                    break
            start = text.find(quote, start + 1)
        reason = (
            "ambiguous_span" if len(matches) > 1 else "ungrounded_span" if not matches else None
        )
        if reason == "ungrounded_span" and context_policy == "hint":
            first = text.find(quote)
            if first >= 0 and text.find(quote, first + 1) < 0:
                matches = [(first, first + len(quote))]
                reason = None
                repaired += 1
        if reason is not None:
            if unaligned == "fail":
                raise ValueError(reason)
            dropped[reason] += 1
            continue
        start, end = matches[0]
        grounded.append({"text": quote, "label": label, "start": start, "end": end})
    if (unaligned == "drop" or context_policy == "hint") and diagnostics is not None:
        diagnostics.update(
            quoted_records=len(records),
            dropped_quotations=sum(dropped.values()),
            dropped_reasons=dropped,
        )
        if context_policy == "hint":
            diagnostics["repaired_contexts"] = repaired
    return normalize_spans(text, grounded)


def project_records(records, fields):
    """Explicit JSON-pointer projections only; missing values are never filled."""
    from .executor import pointer

    if not isinstance(records, list) or len(records) > 32:
        raise ValueError("record_limit")
    result = []
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("invalid_record")
        try:
            result.append({name: pointer(row, path) for name, path in fields.items()})
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("missing_projection_field") from exc
    return result


def bounded_rows(rows, limit, code):
    if (
        not isinstance(rows, list)
        or len(rows) > limit
        or any(not isinstance(r, dict) for r in rows)
    ):
        raise ValueError(code)
    ids = [r.get("id") for r in rows]
    if any(not isinstance(i, str) or not i or len(i) > 128 for i in ids) or len(set(ids)) != len(
        ids
    ):
        raise ValueError("invalid_identity")
    return {r["id"]: r for r in rows}


def normalized(text):
    if not isinstance(text, str):
        raise ValueError("invalid_text")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def normalize_spans(text, spans, *, unit="codepoint", document_id=None):
    if not isinstance(text, str) or not isinstance(spans, list) or len(spans) > 32:
        raise ValueError("span_limit")
    if unit not in {"codepoint", "utf8"}:
        raise ValueError("unsupported_offset_unit")
    doc = document_id or digest(text)
    byte_map, offset = {0: 0}, 0
    for i, character in enumerate(text):
        offset += len(character.encode())
        byte_map[offset] = i + 1
    output = {}
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("invalid_span")
        start, end = span.get("start"), span.get("end")
        if type(start) is not int or type(end) is not int:
            raise ValueError("invalid_span")
        if unit == "utf8":
            start, end = byte_map.get(start), byte_map.get(end)
        if (
            start is None
            or end is None
            or not 0 <= start < end <= len(text)
            or text[start:end] != span.get("text")
        ):
            raise ValueError("invalid_span")
        label = span.get("label", "entity")
        if not isinstance(label, str) or len(label) > 128:
            raise ValueError("invalid_span")
        identity = digest([doc, start, end, label])
        item = {
            "id": identity,
            "document_id": doc,
            "start": start,
            "end": end,
            "text": text[start:end],
            "label": label,
        }
        if "confidence" in span:
            score = span["confidence"]
            if type(score) not in {int, float} or not 0 <= score <= 1:
                raise ValueError("invalid_span")
            item["confidence"] = score
        if identity in output and output[identity] != item:
            raise ValueError("conflicting_span")
        output[identity] = item
    return sorted(output.values(), key=lambda v: (v["start"], v["end"], v["label"]))


def candidates(mentions, catalog, *, top_k=3):
    bounded_rows(mentions, 32, "mention_limit")
    bounded_rows(catalog, 500, "catalog_limit")
    if type(top_k) is not int or top_k not in {1, 3, 5}:
        raise ValueError("invalid_top_k")
    catalog = sorted(catalog, key=lambda r: r["id"])
    catalog_id = digest(catalog)
    pairs, omitted = [], 0
    for mention in mentions:
        needle = normalized(mention.get("text"))
        tokens = set(re.findall(r"\w+", needle))
        matches = []
        for row in catalog:
            aliases = row.get("aliases", [])
            if not isinstance(aliases, list) or len(aliases) > 32:
                raise ValueError("invalid_aliases")
            names = [normalized(row.get("name")), *[normalized(a) for a in aliases]]
            score = max(
                (
                    1.0
                    if needle and needle == name
                    else len(tokens & set(re.findall(r"\w+", name)))
                    / max(1, len(tokens | set(re.findall(r"\w+", name))))
                )
                for name in names
            )
            if score > 0:
                matches.append((score, row["id"]))
        matches.sort(key=lambda v: (-v[0], v[1]))
        omitted += max(0, len(matches) - top_k)
        for score, candidate in matches[:top_k]:
            pairs.append(
                {
                    "id": digest([catalog_id, mention["id"], candidate]),
                    "mention_id": mention["id"],
                    "candidate_id": candidate,
                    "retrieval_score": score,
                }
            )
    return {
        "catalog_id": catalog_id,
        "pairs": pairs,
        "omitted_pairs": omitted,
        "mention_ids": [r["id"] for r in mentions],
    }


def resolve_links(retrieval, decisions, *, unique=False):
    pairs = bounded_rows(retrieval["pairs"], 160, "pair_limit")
    mentions = retrieval.get("mention_ids")
    if (
        not isinstance(mentions, list)
        or len(mentions) > 32
        or any(not isinstance(m, str) or not m or len(m) > 128 for m in mentions)
        or len(set(mentions)) != len(mentions)
    ):
        raise ValueError("invalid_identity")
    identities = set()
    for pair in pairs.values():
        identity = (pair.get("mention_id"), pair.get("candidate_id"))
        if (
            not all(isinstance(v, str) and 0 < len(v) <= 128 for v in identity)
            or identity[0] not in mentions
            or identity in identities
        ):
            raise ValueError("invalid_identity")
        identities.add(identity)
    judged = bounded_rows(decisions, 160, "pair_limit")
    if pairs.keys() != judged.keys() or any(type(d.get("match")) is not bool for d in decisions):
        raise ValueError("incomplete_pair_decisions")
    matches = {m: [] for m in retrieval["mention_ids"]}
    for identity, decision in judged.items():
        if decision["match"]:
            pair = pairs[identity]
            matches[pair["mention_id"]].append(pair["candidate_id"])
    chosen = [v[0] for v in matches.values() if len(v) == 1]
    output = []
    for mention, values in matches.items():
        ambiguous = len(values) > 1 or (unique and values and chosen.count(values[0]) > 1)
        output.append(
            {
                "mention_id": mention,
                "candidate_id": values[0] if len(values) == 1 and not ambiguous else None,
                "status": "ambiguous" if ambiguous else "matched" if values else "no_match",
            }
        )
    return output


def retrieval_evidence(trace, expected):
    """Teacher-labelled recall is separate from decisions on the retrieved pairs.

    This diagnostic is never a qualification score or a planning feature. An
    absent reference or omitted intermediate has no observed recall value.
    """
    if not isinstance(expected, dict):
        return None
    targets = {
        (item["mention_id"], item["candidate_id"])
        for values in expected.values()
        if isinstance(values, list)
        for item in values
        if isinstance(item, dict)
        and isinstance(item.get("mention_id"), str)
        and isinstance(item.get("candidate_id"), str)
    }
    for step in trace:
        if step["operator"] != "retrieve_candidates":
            continue
        value = (step.get("output") or {}).get("value")
        if not isinstance(value, dict):
            return {"state": "unavailable"}
        pairs = {(p["mention_id"], p["candidate_id"]) for p in value["pairs"]}
        return {
            "state": "observed",
            "positive_targets": len(targets),
            "retrieved_targets": len(targets & pairs),
            "missing_targets": [
                {"mention_id": m, "candidate_id": c} for m, c in sorted(targets - pairs)
            ],
            "scope": "reference_labelled_targets",
        }
    return None


def assemble_records(mentions, relations, fields):
    index = bounded_rows(mentions, 32, "mention_limit")
    bounded_rows(relations, 32, "record_limit")
    if (
        not isinstance(fields, list)
        or not 1 <= len(fields) <= 32
        or any(not isinstance(f, str) or not f for f in fields)
        or len(set(fields)) != len(fields)
    ):
        raise ValueError("invalid_fields")
    output = []
    for relation in relations:
        record = {}
        for field in fields:
            identity = relation.get(field)
            if identity is not None and identity not in index:
                raise ValueError("unknown_mention")
            record[field] = index[identity]["text"] if identity is not None else None
        output.append(record)
    return output


def validate_config(op, inputs, config):
    from jsonschema import Draft202012Validator

    from .components import semantic_contract

    arity, keys = OPERATORS[op]
    if op == "align_spans" and isinstance(config, dict) and "context_policy" in config:
        if not isinstance(config["context_policy"], str) or config["context_policy"] not in {
            "strict",
            "hint",
        }:
            raise ValueError("invalid_kernel_configuration")
        config = {k: v for k, v in config.items() if k != "context_policy"}
    if op == "align_spans" and isinstance(config, dict) and "unaligned" in config:
        # span-recipes-v2. Kept out of OPERATORS so the advertised component registry
        # (and every frozen proposal envelope) stays byte-identical.
        if not isinstance(config["unaligned"], str) or config["unaligned"] not in UNALIGNED_MODES:
            raise ValueError("invalid_kernel_configuration")
        config = {k: v for k, v in config.items() if k != "unaligned"}
    if len(inputs) != arity or set(config) - keys:
        raise ValueError("invalid_kernel_configuration")
    if not Draft202012Validator(semantic_contract(op)["config_schema"]).is_valid(config):
        raise ValueError("invalid_kernel_configuration")
    if op == "project_records":
        from .contracts import reference

        for path in config["fields"].values():
            reference("input#" + path)
    if op == "retrieve_candidates" and (
        type(config.get("top_k", 3)) is not int or config.get("top_k", 3) not in {1, 3, 5}
    ):
        raise ValueError("invalid_top_k")
    if op == "chunk_text":
        size, overlap = config.get("size", 4096), config.get("overlap", 128)
        unit = config.get("unit", "chars")
        if (
            type(size) is not int
            or type(overlap) is not int
            or not 1 <= size <= 16384
            or not 0 <= overlap < size
            or unit not in {"chars", "tokens"}
        ):
            raise ValueError("invalid_chunk_configuration")
    if op == "merge_labels":
        rule = config.get("rule", "all_agree")
        order = config.get("order")
        if rule not in MERGE_RULES or (
            rule == "max_ordinal" and (not isinstance(order, list) or len(order) < 2)
        ):
            raise ValueError("invalid_kernel_configuration")
    if op == "merge_records":
        fields = config.get("fields")
        if not isinstance(fields, dict) or not fields:
            raise ValueError("invalid_kernel_configuration")
        if any(policy not in {"array", "scalar", "ignore"} for policy in fields.values()):
            raise ValueError("invalid_kernel_configuration")
    for key in ("casefold", "unique"):
        if key in config and type(config[key]) is not bool:
            raise ValueError("invalid_kernel_configuration")


def _chunk_count(length, size, overlap):
    if length <= 0:
        return 1
    return max(1, (max(0, length - size) + size - overlap - 1) // (size - overlap) + 1)


def _char_chunk_spans(text, size, overlap):
    count = _chunk_count(len(text), size, overlap)
    if count > 32:
        raise ValueError("chunk_limit")
    return [
        (start, min(start + size, len(text)))
        for start in [i * (size - overlap) for i in range(count)]
    ]


def _token_pieces(text):
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    pieces, cursor = [], 0
    for token in enc.encode(text):
        piece = enc.decode([token])
        if not text.startswith(piece, cursor):
            pieces.append(text[cursor:])
            return pieces
        pieces.append(piece)
        cursor += len(piece)
    if cursor != len(text):
        pieces.append(text[cursor:])
    return pieces


def _snap_end(text, start, end, floor):
    window = text[start:end]
    best = None
    for mark in ("\n\n", "\n", ". ", "? ", "! "):
        found = window.rfind(mark)
        if found < 0:
            continue
        cut = start + found + len(mark)
        if cut >= floor and (best is None or cut > best):
            best = cut
    return best or end


def _token_chunk_spans(text, size, overlap, pieces=None, limit=32):
    """``pieces`` may be precomputed by callers that try many sizes on one text;
    ``limit=None`` lets context fit inspect chunk sizes beyond the kernel cap."""
    pieces = _token_pieces(text) if pieces is None else pieces
    count = _chunk_count(len(pieces), size, overlap)
    if limit is not None and count > limit:
        raise ValueError("chunk_limit")
    offsets = [0]
    for piece in pieces:
        offsets.append(offsets[-1] + len(piece))
    spans = []
    step = size - overlap
    for index in range(count):
        token_start = index * step
        token_end = min(token_start + size, len(pieces))
        start = offsets[token_start]
        end = offsets[token_end]
        if index < count - 1:
            next_start = offsets[min(token_start + step, len(pieces))]
            end = _snap_end(text, start, end, next_start)
        spans.append((start, end if end > start or not text else start))
    if spans:
        spans[-1] = (spans[-1][0], len(text))
    elif not text:
        spans = [(0, 0)]
    return spans


def merge_spans(text, chunks):
    """Shift chunk-local spans onto the original text and drop overlap duplicates."""
    if not isinstance(text, str) or not isinstance(chunks, list) or len(chunks) > 32:
        reason = "chunk_limit" if isinstance(chunks, list) and len(chunks) > 32 else "invalid_text"
        raise ValueError(reason)
    shifted = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("invalid_span")
        base = chunk.get("start", 0)
        width = chunk.get("end", base) - base
        for span in chunk.get("value") or []:
            if not isinstance(span, dict):
                raise ValueError("invalid_span")
            start, end = span.get("start"), span.get("end")
            if type(start) is not int or type(end) is not int:
                raise ValueError("invalid_span")
            shifted.append(
                {
                    **span,
                    "start": start + base,
                    "end": end + base,
                    "_edge": min(start, width - end),
                }
            )
    kept = []
    for span in shifted:
        replaced = False
        for index, prior in enumerate(kept):
            overlaps = (
                prior.get("label") == span.get("label")
                and span["start"] < prior["end"]
                and prior["start"] < span["end"]
            )
            if not overlaps:
                continue
            if span["_edge"] > prior["_edge"]:
                kept[index] = span
            replaced = True
            break
        if not replaced:
            kept.append(span)
    cleaned = [{k: v for k, v in span.items() if k != "_edge"} for span in kept]
    return normalize_spans(text, cleaned)


def merge_records(chunks, fields, sums=None):
    """Merge per-chunk records. Scalars must agree. Totals are recomputed."""
    if not isinstance(chunks, list) or len(chunks) > 32:
        raise ValueError("chunk_limit" if isinstance(chunks, list) else "invalid_input_shape")
    merged, arrays = {}, {name: [] for name, policy in fields.items() if policy == "array"}
    for chunk in chunks:
        value = chunk.get("value") if isinstance(chunk, dict) else None
        if not isinstance(value, dict):
            raise ValueError("invalid_input_shape")
        for name, policy in fields.items():
            if policy == "ignore" or name not in value or value[name] is None:
                continue
            item = value[name]
            if policy == "array":
                if not isinstance(item, list):
                    raise ValueError("invalid_input_shape")
                arrays[name].extend(item)
            elif name in merged and merged[name] != item:
                raise ValueError("merge_conflict")
            else:
                merged[name] = item
    for name, items in arrays.items():
        seen, unique = set(), []
        for item in items:
            key = canonical(item)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        merged[name] = unique
    for name, path in (sums or {}).items():
        collection, _, field = str(path).partition("/")
        total = Decimal(0)
        for item in merged.get(collection, []):
            if isinstance(item, dict) and item.get(field) not in {None, ""}:
                total += Decimal(str(item[field]))
        merged[name] = str(total)
    return merged


def merge_labels(chunks, rule="all_agree", order=None):
    """Combine per-chunk labels. Disagreement is a conflict unless the rule allows it."""
    if not isinstance(chunks, list) or not chunks or len(chunks) > 32:
        too_many = isinstance(chunks, list) and len(chunks) > 32
        raise ValueError("chunk_limit" if too_many else "invalid_input_shape")
    values = []
    for chunk in chunks:
        value = chunk.get("value") if isinstance(chunk, dict) else None
        if not isinstance(value, dict):
            raise ValueError("invalid_input_shape")
        values.append(value)
    rank = {label: index for index, label in enumerate(order or [])}
    merged = {}
    for key in set().union(*(value.keys() for value in values)):
        present = [value[key] for value in values if key in value and value[key] is not None]
        if not present:
            merged[key] = None
        elif rule == "any_true" and any(item is True for item in present):
            merged[key] = True
        elif rule == "any_true" and all(item is False for item in present):
            merged[key] = False
        elif rule == "max_ordinal":
            if any(item not in rank for item in present):
                raise ValueError("merge_conflict")
            merged[key] = max(present, key=rank.__getitem__)
        elif rule == "union":
            union = []
            for item in present:
                for row in item if isinstance(item, list) else [item]:
                    if row not in union:
                        union.append(row)
            merged[key] = union
        elif any(item != present[0] for item in present):
            raise ValueError("merge_conflict")
        else:
            merged[key] = present[0]
    return merged


def kernel(op, args, config, diagnostics=None):
    validate_config(op, args, config)
    from jsonschema import Draft202012Validator

    from .components import semantic_contract

    if not Draft202012Validator(semantic_contract(op)["input_schema"]).is_valid(args):
        raise ValueError("invalid_input_shape")
    if op == "align_spans":
        return align_spans(
            *args,
            unaligned=config.get("unaligned", "fail"),
            context_policy=config.get("context_policy", "strict"),
            diagnostics=diagnostics,
        )
    if op == "project_records":
        return project_records(args[0], config["fields"])
    if op == "exact_links":
        mentions, catalog = args
        bounded_rows(mentions, 32, "mention_limit")
        bounded_rows(catalog, 500, "catalog_limit")
        pairs = []
        for mention in mentions:
            needle = normalized(mention.get("text"))
            for row in sorted(catalog, key=lambda r: r["id"]):
                aliases = row.get("aliases", [])
                if not isinstance(aliases, list) or len(aliases) > 32:
                    raise ValueError("invalid_aliases")
                if needle and needle in [normalized(row.get("name")), *map(normalized, aliases)]:
                    pairs.append(
                        {
                            "id": digest([mention["id"], row["id"]]),
                            "mention_id": mention["id"],
                            "candidate_id": row["id"],
                        }
                    )
        return resolve_links(
            {"pairs": pairs, "mention_ids": [m["id"] for m in mentions]},
            [{"id": p["id"], "match": True} for p in pairs],
            **config,
        )
    if op == "retrieve_candidates":
        return candidates(*args, **config)
    if op == "resolve_links":
        return resolve_links(*args, **config)
    if op == "assemble_records":
        return assemble_records(*args, config.get("fields"))
    if op == "merge_spans":
        return merge_spans(*args)
    if op == "merge_records":
        return merge_records(args[0], config.get("fields"), config.get("sums"))
    if op == "merge_labels":
        return merge_labels(args[0], config.get("rule", "all_agree"), config.get("order"))
    text = args[0]
    if not isinstance(text, str):
        raise ValueError("invalid_text")
    if op == "normalize_text":
        # A per-output-codepoint source map makes expansion explicit. Never use
        # normalized offsets as if they referred to the original document.
        pieces = [c.casefold() if config.get("casefold", True) else c for c in text]
        return {
            "text": "".join(pieces),
            "source_map": [i for i, p in enumerate(pieces) for _ in p],
            "document_id": digest(text),
        }
    size, overlap = config.get("size", 4096), config.get("overlap", 128)
    spans = (
        _token_chunk_spans(text, size, overlap)
        if config.get("unit") == "tokens"
        else _char_chunk_spans(text, size, overlap)
    )
    return [
        {
            "id": digest([text, start, end]),
            "document_id": digest(text),
            "start": start,
            "end": end,
            "text": text[start:end],
        }
        for start, end in spans
    ]
