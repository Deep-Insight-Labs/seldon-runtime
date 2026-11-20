"""Bounded literal-anchor templates; no regex backtracking or expression evaluation."""

import re
from functools import lru_cache
from string import Formatter
from typing import NamedTuple


class TemplateError(ValueError):
    def __init__(self, reason, segment=0, *, repeated=False):
        super().__init__(reason)
        self.segment = segment
        self.repeated = repeated


class Template(NamedTuple):
    literals: tuple[str, ...]
    names: tuple[str, ...]
    placeholders: tuple[str, ...]

    def render(self, values):
        return "".join(
            literal + (values[self.names[index]] if index < len(self.names) else "")
            for index, literal in enumerate(self.literals)
        )

    def match(self, text):
        literals, names, _ = self
        if not names:
            if text != literals[0]:
                raise TemplateError("template_mismatch")
            return {}
        start, end = len(literals[0]), len(text) - len(literals[-1])
        if not text.startswith(literals[0]) or not text.endswith(literals[-1]) or end < start:
            raise TemplateError("template_mismatch")
        first, pos = [], start
        for index in range(len(names)):
            boundary = end if index == len(names) - 1 else text.find(literals[index + 1], pos, end)
            if boundary < pos:
                raise TemplateError("template_mismatch", index + 1)
            first.append(text[pos:boundary])
            pos = boundary + len(literals[index + 1])
        last, pos = [], end
        for index in range(len(names) - 1, -1, -1):
            boundary = start if index == 0 else text.rfind(literals[index], start, pos)
            if boundary < start:
                raise TemplateError("template_mismatch", index)
            last.append(text[boundary + (len(literals[index]) if index else 0) : pos])
            pos = boundary
        if first != last[::-1]:
            raise TemplateError("ambiguous_template_match")
        values = {}
        for name, value in zip(names, first):
            if name in values and values[name] != value:
                raise TemplateError("template_mismatch", repeated=True)
            values[name] = value
        return values


@lru_cache(maxsize=1024)
def parse_template(text: str, syntax: str) -> Template:
    literals, names, placeholders = [], [], []
    pending = ""
    try:
        if syntax == "python_format":
            for token in re.findall(r"\{\{|\}\}|\{[^{}]*\}|[{}]", text):
                if token not in {"{{", "}}"} and not re.fullmatch(
                    r"\{[a-zA-Z_][a-zA-Z0-9_]{0,63}\}", token
                ):
                    raise ValueError
            parts = Formatter().parse(text)
        elif syntax == "jinja_variables":
            parts = []
            end = 0
            for match in re.finditer(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]{0,63})\s*\}\}", text):
                parts.append((text[end : match.start()], match[1], "", None, match[0]))
                end = match.end()
            parts.append((text[end:], None, "", None, None))
        else:
            raise ValueError
        for part in parts:
            literal, name, format_spec, conversion = part[:4]
            if syntax == "jinja_variables" and any(
                token in literal for token in ("{{", "}}", "{%", "{#")
            ):
                raise ValueError
            pending += literal
            if name is None:
                continue
            if format_spec or conversion or not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]{0,63}", name):
                raise ValueError
            if names and not pending:
                raise TemplateError("adjacent_placeholders")
            literals.append(pending)
            names.append(name)
            placeholders.append("{" + name + "}" if syntax == "python_format" else part[4])
            pending = ""
        literals.append(pending)
    except TemplateError:
        raise
    except (ValueError, TypeError):
        raise TemplateError("template_syntax_unsupported") from None
    return Template(tuple(literals), tuple(names), tuple(placeholders))


@lru_cache(maxsize=256)
def compile_templates(spec_digest: str, syntax: str, messages: tuple[str, ...]):
    """The digest makes the cache follow immutable contract identity."""
    return tuple(parse_template(text, syntax) for text in messages)
