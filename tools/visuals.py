"""
title: Claude-like Inline Visuals
author: Classic298 (original inline-visualizer-v2), Marios Adamidis (adapted)
author_url: https://github.com/Classic298/open-webui-plugins/tree/main/inline-visualizer-v2
version: 1.0.0
description: |
    Renders interactive charts, diagrams, and visualizations inline in the chat
    message stream. Uses OWUI's embeds event emitter to display self-contained
    HTML directly in the conversation — no side panel, no artifacts.
    Supports: Chart.js, Plotly, Mermaid, raw HTML, tables, metric cards, dashboards.
requirements: pydantic
"""

import json
import html
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Module-level security setting, updated per-call from Valves
_security_level = "strict"


# ═══════════════════════════════════════════════════════════════════════════════
#  Module-level helpers (invisible to OWUI tool scanner)
# ═══════════════════════════════════════════════════════════════════════════════


async def _emit(emitter: Optional[Any], msg: str, done: bool = False) -> None:
    """Emit a status message."""
    if emitter:
        try:
            await emitter(
                {"type": "status", "data": {"description": msg, "done": done}}
            )
        except Exception:
            pass


async def _emit_html(emitter: Optional[Any], html_content: str) -> None:
    """Emit HTML content as an inline embed in the chat."""
    if emitter:
        try:
            await emitter({"type": "embeds", "data": {"embeds": [html_content]}})
        except Exception as e:
            logger.error(f"Failed to emit HTML embed: {e}")


def _escape_json_for_js(data: Any) -> str:
    """Safely serialize data for embedding in a <script> tag."""
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


# ── Robust JSON parser ────────────────────────────────────────────────────────
# Gemini frequently sends malformed JSON in tool args. This handles:
# 1. Normal JSON (pass-through)
# 2. JSON with literal \n in strings that should be actual newlines
# 3. JSON wrapped in markdown code fences ```json ... ```
# 4. Trailing commas before } or ]
# 5. Single quotes instead of double quotes


def _parse_spec_json(raw: str) -> Dict[str, Any]:
    """Parse spec_json with multiple fallback strategies for Gemini quirks."""
    if not raw or not raw.strip():
        raise ValueError("Empty spec_json")

    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fix trailing commas  ,} or ,]
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 3: replace single quotes with double quotes (crude but handles simple cases)
    if "'" in cleaned and '"' not in cleaned:
        sq_fixed = cleaned.replace("'", '"')
        try:
            return json.loads(sq_fixed)
        except json.JSONDecodeError:
            pass

    # If all attempts fail, raise with the original error
    return json.loads(text)  # will raise the original JSONDecodeError


# ── Mermaid code normalizer ───────────────────────────────────────────────────
# Gemini mangles multi-line strings in JSON tool call arguments:
# - Strips \n entirely → "graph TDA[Start] --> B[End]"
# - Sends literal \\n → "graph TD\\nA[Start] --> B[End]"
# - Sends ; as separator → "graph TD;A[Start] --> B[End]"
# - Sends code in wrong key → arbitrary field names instead of "code"

_MERMAID_KNOWN_KEYS = (
    "code",
    "definition",
    "diagram",
    "mermaid",
    "mermaid_code",
    "content",
    "spec",
    "data",
    "chart",
    "graph",
    "flow",
    "sequence",
    "value",
    "payload",
    "body",
    "source",
    "text",
)

_MERMAID_DECLARATIONS = (
    "graph",
    "flowchart",
    "sequencediagram",
    "classdiagram",
    "statediagram",
    "erdiagram",
    "journey",
    "gantt",
    "pie",
    "gitgraph",
    "mindmap",
    "timeline",
    "requirementdiagram",
    "quadrantchart",
    "xychart-beta",
    "block-beta",
    "packet-beta",
    "architecture-beta",
)

_MERMAID_INDICATORS = (
    "-->",
    "---",
    "-.->",
    "==>",
    "->>",
    "-->>",
    "-x",
    "--x",
    "subgraph",
    "participant ",
    "actor ",
    "sequenceDiagram",
    "classDiagram",
    "erDiagram",
    "gantt",
    "pie title",
    "gitgraph",
    "flowchart",
    "graph TD",
    "graph LR",
    "graph TB",
    "graph RL",
)

_MERMAID_SPECIAL_CHARS = set('/\\#{}<>"')


def _iter_string_values(value: Any, path: str = "root"):
    """Yield every non-empty string inside nested Mermaid specs."""
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield path, text
        return

    if isinstance(value, dict):
        for key, subvalue in value.items():
            yield from _iter_string_values(subvalue, f"{path}.{key}")
        return

    if isinstance(value, list):
        for idx, item in enumerate(value):
            yield from _iter_string_values(item, f"{path}[{idx}]")


def _looks_like_mermaid(text: str) -> bool:
    """Heuristic detection for Mermaid snippets, even when Gemini invents keys."""
    sample = text.strip()
    if not sample:
        return False

    lowered = sample.lower()
    first_line = lowered.splitlines()[0].strip()
    if any(first_line.startswith(prefix) for prefix in _MERMAID_DECLARATIONS):
        return True

    return any(indicator.lower() in lowered for indicator in _MERMAID_INDICATORS)


def _extract_mermaid_candidate(spec: Dict[str, Any]) -> str:
    """Find Mermaid code even when Gemini hides it under arbitrary keys."""
    candidates = []

    for key in _MERMAID_KNOWN_KEYS:
        if key not in spec:
            continue
        for path, text in _iter_string_values(spec[key], f"root.{key}"):
            score = 200 if _looks_like_mermaid(text) else 120
            score += 50 if path.count(".") <= 2 else 0
            score += min(len(text), 400) // 20
            candidates.append((score, path, text))

    for path, text in _iter_string_values(spec):
        if path in {"root.type", "root.title", "root.subtitle", "root.min_height"}:
            continue
        if _looks_like_mermaid(text):
            score = 100
            if any(f".{key}" in path for key in _MERMAID_KNOWN_KEYS):
                score += 40
            score += min(len(text), 400) // 20
            candidates.append((score, path, text))

    if candidates:
        _, path, text = max(candidates, key=lambda item: item[0])
        logger.info(f"[Mermaid] Selected Mermaid code from {path}")
        return text

    if str(spec.get("type", "")).lower() == "mermaid":
        fallbacks = [
            (len(text), path, text)
            for path, text in _iter_string_values(spec)
            if path not in {"root.type", "root.title", "root.subtitle"}
        ]
        if fallbacks:
            _, path, text = max(fallbacks, key=lambda item: item[0])
            logger.info(f"[Mermaid] Falling back to longest string from {path}")
            return text

    return ""


def _normalize_semicolon_lines(code: str) -> str:
    """Convert top-level semicolons to line breaks without touching labels."""
    out = []
    depth_square = depth_round = depth_curly = 0
    in_single = in_double = False
    escape = False

    for char in code:
        if escape:
            out.append(char)
            escape = False
            continue

        if char == "\\":
            out.append(char)
            escape = True
            continue

        if char == "'" and not in_double:
            in_single = not in_single
            out.append(char)
            continue

        if char == '"' and not in_single:
            in_double = not in_double
            out.append(char)
            continue

        if not in_single and not in_double:
            if char == "[":
                depth_square += 1
            elif char == "]" and depth_square > 0:
                depth_square -= 1
            elif char == "(":
                depth_round += 1
            elif char == ")" and depth_round > 0:
                depth_round -= 1
            elif char == "{":
                depth_curly += 1
            elif char == "}" and depth_curly > 0:
                depth_curly -= 1
            elif (
                char == ";"
                and depth_square == 0
                and depth_round == 0
                and depth_curly == 0
            ):
                out.append("\n")
                continue

        out.append(char)

    return "".join(out)


def _infer_mermaid_declaration(code: str) -> str:
    """Guess the Mermaid diagram type when Gemini drops the declaration line."""
    lowered = code.lower()
    if "participant " in lowered or "->>" in code or "-->>" in code or "-x" in code:
        return "sequenceDiagram"
    if "class " in lowered or "<|--" in code or "*--" in code:
        return "classDiagram"
    if "||--" in code or "o{" in code or "}o" in code:
        return "erDiagram"
    if "section " in lowered or re.search(r":\s*\d{4}-\d{2}-\d{2}", code):
        return "gantt"
    if "-->" in code or "subgraph" in lowered:
        return "flowchart TD"
    return ""


def _insert_missing_mermaid_newlines(code: str) -> str:
    """Repair diagrams that arrive as one long line."""
    compact = re.sub(r"[ \t]+", " ", code.strip())
    if not compact or "\n" in compact:
        return compact

    first_token = compact.split(" ", 1)[0]
    lowered = first_token.lower()

    if lowered in _MERMAID_DECLARATIONS:
        if " " in compact:
            first_line, rest = compact.split(" ", 1)
            compact = f"{first_line}\n{rest.strip()}"
    else:
        declaration = _infer_mermaid_declaration(compact)
        if declaration:
            compact = f"{declaration}\n{compact}"

    substitutions = [
        (r"(\])\s+([A-Za-z_])", r"\1\n\2"),
        (r"(\})\s+([A-Za-z_])", r"\1\n\2"),
        (r"(\))\s+([A-Za-z_])", r"\1\n\2"),
        (
            r"\s+(?=(?:subgraph|end|classDef|class|style|linkStyle|click|Note\b|note\b|section\b|title\b|dateFormat\b|axisFormat\b|participant\b|actor\b|activate\b|deactivate\b|alt\b|else\b|opt\b|loop\b|par\b|and\b|rect\b|critical\b|option\b))",
            "\n",
        ),
        (
            r"(?<=[A-Za-z0-9_}\]\)])\s+(?=[A-Za-z_][A-Za-z0-9_]*\s*(?:--?>|--?>>|->>|-x|--x|<--|<->|<\|--|\|\|--|o\{|}\|))",
            "\n",
        ),
        (
            r"(?<=[A-Za-z0-9_}\]\)])\s+(?=[A-Z][A-Z0-9_]+\s+\|\|--)",
            "\n",
        ),
        (
            r"(?<=\S)\s+(?=[A-Za-z][^:\n]{0,60}:\s*(?:\d{4}-\d{2}-\d{2}|after\s+\S+|\d+[dhmsw]))",
            "\n",
        ),
    ]

    for pattern, replacement in substitutions:
        compact = re.sub(pattern, replacement, compact)

    if compact.lower().startswith("gantt\n"):
        compact = re.sub(
            r"\nsection\s*\n([^\n:]+?)(?=\n[A-Za-z][^:\n]{0,60}:)",
            r"\nsection \1",
            compact,
        )

    return compact


def _sanitize_mermaid_label_text(text: str) -> str:
    """Keep label text parser-friendly without changing semantics much."""
    cleaned = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .replace("\u00a0", " ")
    )
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _quote_mermaid_text(text: str) -> str:
    """Quote Mermaid message text when special characters make parsing fragile."""
    stripped = _sanitize_mermaid_label_text(text)
    if not stripped:
        return stripped
    if (stripped.startswith('"') and stripped.endswith('"')) or (
        stripped.startswith("'") and stripped.endswith("'")
    ):
        return stripped
    escaped = stripped.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _sanitize_sequence_line(line: str) -> str:
    """Quote risky sequence-diagram message payloads but keep arrows intact."""
    stripped = line.strip()
    if not stripped or stripped.startswith("%%") or ":" not in line:
        return line

    if any(
        token in line
        for token in ("->>", "-->>", "->", "-->", "-x", "--x", "<<->>", "<--", "<->")
    ):
        prefix, message = line.split(":", 1)
        payload = message.strip()
        if payload and any(char in _MERMAID_SPECIAL_CHARS for char in payload):
            return f"{prefix}: {_quote_mermaid_text(payload)}"
        return line

    if stripped.lower().startswith("note "):
        prefix, message = line.split(":", 1)
        payload = message.strip()
        if payload and any(char in _MERMAID_SPECIAL_CHARS for char in payload):
            return f"{prefix}: {_quote_mermaid_text(payload)}"

    return line


def _sanitize_flowchart_line(line: str) -> str:
    """Normalize fragile label content inside Mermaid flowchart delimiters."""
    sanitized = line

    def _replace(match: re.Match) -> str:
        open_delim, content, close_delim = match.groups()
        return f"{open_delim}{_sanitize_mermaid_label_text(content)}{close_delim}"

    for pattern in (
        r"(\[)([^\]]*)(\])",
        r"(\()([^)]*)(\))",
        r"(\{)([^}]*)(\})",
        r"(\|)([^|]+)(\|)",
    ):
        sanitized = re.sub(pattern, _replace, sanitized)

    return sanitized


def _sanitize_mermaid_code(code: str) -> str:
    """Apply light-touch Mermaid sanitation without rewriting valid diagrams."""
    if not code.strip():
        return ""

    normalized = code.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = normalized.split("\n")
    header = lines[0].strip()
    header_lower = header.lower()
    body = lines[1:]

    cleaned_lines = [header]
    for line in body:
        updated = line.rstrip()
        if header_lower.startswith("sequencediagram"):
            updated = _sanitize_sequence_line(updated)
        elif header_lower.startswith(("graph", "flowchart", "mindmap", "journey")):
            updated = _sanitize_flowchart_line(updated)
        else:
            updated = (
                _sanitize_mermaid_label_text(updated) if updated.strip() else updated
            )
        cleaned_lines.append(updated)

    return "\n".join(cleaned_lines).strip()


def _normalize_mermaid_code(spec: Dict[str, Any]) -> str:
    """Extract and normalize Mermaid code from hostile Gemini payloads."""
    code = _extract_mermaid_candidate(spec)
    if not code.strip():
        return ""

    text = code.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:mermaid)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    text = text.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    text = _normalize_semicolon_lines(text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    if "\n" not in text:
        text = _insert_missing_mermaid_newlines(text)

    first_line = text.strip().split("\n", 1)[0].strip().lower()
    if not any(first_line.startswith(prefix) for prefix in _MERMAID_DECLARATIONS):
        declaration = _infer_mermaid_declaration(text)
        if declaration:
            text = f"{declaration}\n{text.strip()}"

    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line.strip() or line == "").strip()


# ── Color palettes ────────────────────────────────────────────────────────────

_PALETTE = [
    "#E8956A",  # warm orange (Claude-like)
    "#5DCAA5",  # teal
    "#85B7EB",  # blue
    "#AFA9EC",  # purple
    "#ED93B1",  # pink
    "#EF9F27",  # amber
    "#97C459",  # green
    "#F0997B",  # coral
    "#6ea8fe",  # sky
    "#da77f2",  # violet
]


def _get_colors(n: int) -> List[str]:
    """Return n colors from the palette, cycling if needed."""
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]


# ── Common style / theme infrastructure ───────────────────────────────────────

_COMMON_STYLE = """
<style>
  /* ── Theme: self-contained light/dark ── */
  /* Primary detection via parent class (OWUI sets .dark on <html>),
     fallback to prefers-color-scheme for isolated rendering. */
  :root {
    --iv-bg: transparent;
    --iv-card: #f5f5f5;
    --iv-card-hover: #ebebeb;
    --iv-text: #1a1a1a;
    --iv-muted: #666666;
    --iv-hint: #999999;
    --iv-border: rgba(0,0,0,0.1);
    --iv-border-light: rgba(0,0,0,0.06);
    --iv-pie-border: var(--bg-primary, Canvas);
    --iv-font-sans: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    --iv-font-mono: 'SF Mono', Menlo, Consolas, monospace;
    --iv-radius-md: 8px;
    --iv-radius-lg: 12px;
    /* ── Color ramp variables (light) ── */
    --ramp-purple-fill:#EEEDFE; --ramp-purple-stroke:#534AB7; --ramp-purple-th:#3C3489; --ramp-purple-ts:#534AB7;
    --ramp-teal-fill:#E1F5EE;   --ramp-teal-stroke:#0F6E56;   --ramp-teal-th:#085041;   --ramp-teal-ts:#0F6E56;
    --ramp-coral-fill:#FAECE7;  --ramp-coral-stroke:#993C1D;  --ramp-coral-th:#712B13;  --ramp-coral-ts:#993C1D;
    --ramp-pink-fill:#FBEAF0;   --ramp-pink-stroke:#993556;   --ramp-pink-th:#72243E;   --ramp-pink-ts:#993556;
    --ramp-gray-fill:#F1EFE8;   --ramp-gray-stroke:#5F5E5A;   --ramp-gray-th:#444441;   --ramp-gray-ts:#5F5E5A;
    --ramp-blue-fill:#E6F1FB;   --ramp-blue-stroke:#185FA5;   --ramp-blue-th:#0C447C;   --ramp-blue-ts:#185FA5;
    --ramp-green-fill:#EAF3DE;  --ramp-green-stroke:#3B6D11;  --ramp-green-th:#27500A;  --ramp-green-ts:#3B6D11;
    --ramp-amber-fill:#FAEEDA;  --ramp-amber-stroke:#854F0B;  --ramp-amber-th:#633806;  --ramp-amber-ts:#854F0B;
    --ramp-red-fill:#FCEBEB;    --ramp-red-stroke:#A32D2D;    --ramp-red-th:#791F1F;    --ramp-red-ts:#A32D2D;
  }
  :root[data-theme="dark"] {
    --iv-card: #2a2a2a;
    --iv-card-hover: #333333;
    --iv-text: #e0e0e0;
    --iv-muted: #a0a0a0;
    --iv-hint: #707070;
    --iv-border: rgba(255,255,255,0.12);
    --iv-border-light: rgba(255,255,255,0.06);
    --iv-pie-border: var(--bg-primary, Canvas);
    --ramp-purple-fill:#3C3489; --ramp-purple-stroke:#AFA9EC; --ramp-purple-th:#CECBF6; --ramp-purple-ts:#AFA9EC;
    --ramp-teal-fill:#085041;   --ramp-teal-stroke:#5DCAA5;   --ramp-teal-th:#9FE1CB;   --ramp-teal-ts:#5DCAA5;
    --ramp-coral-fill:#712B13;  --ramp-coral-stroke:#F0997B;  --ramp-coral-th:#F5C4B3;  --ramp-coral-ts:#F0997B;
    --ramp-pink-fill:#72243E;   --ramp-pink-stroke:#ED93B1;   --ramp-pink-th:#F4C0D1;   --ramp-pink-ts:#ED93B1;
    --ramp-gray-fill:#444441;   --ramp-gray-stroke:#B4B2A9;   --ramp-gray-th:#D3D1C7;   --ramp-gray-ts:#B4B2A9;
    --ramp-blue-fill:#0C447C;   --ramp-blue-stroke:#85B7EB;   --ramp-blue-th:#B5D4F4;   --ramp-blue-ts:#85B7EB;
    --ramp-green-fill:#27500A;  --ramp-green-stroke:#97C459;  --ramp-green-th:#C0DD97;  --ramp-green-ts:#97C459;
    --ramp-amber-fill:#633806;  --ramp-amber-stroke:#EF9F27;  --ramp-amber-th:#FAC775;  --ramp-amber-ts:#EF9F27;
    --ramp-red-fill:#791F1F;    --ramp-red-stroke:#F09595;    --ramp-red-th:#F7C1C1;    --ramp-red-ts:#F09595;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --iv-card: #2a2a2a;
      --iv-card-hover: #333333;
      --iv-text: #e0e0e0;
      --iv-muted: #a0a0a0;
      --iv-hint: #707070;
      --iv-border: rgba(255,255,255,0.12);
      --iv-border-light: rgba(255,255,255,0.06);
      --iv-pie-border: var(--bg-primary, Canvas);
      --ramp-purple-fill:#3C3489; --ramp-purple-stroke:#AFA9EC; --ramp-purple-th:#CECBF6; --ramp-purple-ts:#AFA9EC;
      --ramp-teal-fill:#085041;   --ramp-teal-stroke:#5DCAA5;   --ramp-teal-th:#9FE1CB;   --ramp-teal-ts:#5DCAA5;
      --ramp-coral-fill:#712B13;  --ramp-coral-stroke:#F0997B;  --ramp-coral-th:#F5C4B3;  --ramp-coral-ts:#F0997B;
      --ramp-pink-fill:#72243E;   --ramp-pink-stroke:#ED93B1;   --ramp-pink-th:#F4C0D1;   --ramp-pink-ts:#ED93B1;
      --ramp-gray-fill:#444441;   --ramp-gray-stroke:#B4B2A9;   --ramp-gray-th:#D3D1C7;   --ramp-gray-ts:#B4B2A9;
      --ramp-blue-fill:#0C447C;   --ramp-blue-stroke:#85B7EB;   --ramp-blue-th:#B5D4F4;   --ramp-blue-ts:#85B7EB;
      --ramp-green-fill:#27500A;  --ramp-green-stroke:#97C459;  --ramp-green-th:#C0DD97;  --ramp-green-ts:#97C459;
      --ramp-amber-fill:#633806;  --ramp-amber-stroke:#EF9F27;  --ramp-amber-th:#FAC775;  --ramp-amber-ts:#EF9F27;
      --ramp-red-fill:#791F1F;    --ramp-red-stroke:#F09595;    --ramp-red-th:#F7C1C1;    --ramp-red-ts:#F09595;
    }
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body {
    font-family: var(--iv-font-sans);
    background: var(--iv-bg);
    color: var(--iv-text);
    overflow: hidden;
    width: 100%;
    max-width: 100%;
    line-height: 1.5;
  }
  body { padding: 8px 0; }
  svg { overflow: visible; }
  svg text { fill: var(--iv-text); }
  h1 { font-size: 22px; font-weight: 500; color: var(--iv-text); margin-bottom: 12px; }
  h2 { font-size: 18px; font-weight: 500; color: var(--iv-text); margin-bottom: 8px; }
  h3 { font-size: 16px; font-weight: 500; color: var(--iv-text); margin-bottom: 6px; }
  p  { font-size: 14px; color: var(--iv-muted); margin-bottom: 8px; }
  /* ── Pre-styled interactive elements ── */
  button {
    background: transparent; border: 0.5px solid var(--iv-border);
    border-radius: var(--iv-radius-md); padding: 6px 14px; font-size: 13px;
    color: var(--iv-text); cursor: pointer; font-family: var(--iv-font-sans);
    transition: all 0.15s;
  }
  button:hover { background: var(--iv-card); }
  button.active { background: var(--iv-card); border-color: var(--iv-muted); }
  input[type="range"] {
    -webkit-appearance: none; width: 100%; height: 4px;
    background: var(--iv-border); border-radius: 2px; outline: none;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%;
    background: var(--iv-card); border: 0.5px solid var(--iv-muted); cursor: pointer;
  }
  select {
    background: var(--iv-card); border: 0.5px solid var(--iv-border);
    border-radius: var(--iv-radius-md); padding: 6px 10px; font-size: 13px;
    color: var(--iv-text); font-family: var(--iv-font-sans);
  }
  code {
    font-family: var(--iv-font-mono); font-size: 13px; background: var(--iv-card);
    padding: 2px 6px; border-radius: 4px;
  }
  /* ── SVG utility classes ── */
  .t  { font: 400 14px/1.4 var(--iv-font-sans); fill: var(--iv-text); }
  .ts { font: 400 12px/1.4 var(--iv-font-sans); fill: var(--iv-muted); }
  .th { font: 500 14px/1.4 var(--iv-font-sans); fill: var(--iv-text); }
  .box    { fill: var(--iv-card); stroke: var(--iv-border); stroke-width: 0.5; }
  .node   { cursor: pointer; }
  .node:hover { opacity: 0.85; }
  .arr    { stroke: var(--iv-muted); stroke-width: 1.5; fill: none; }
  .leader { stroke: var(--iv-hint); stroke-width: 0.5; stroke-dasharray: 3 2; fill: none; }
  /* ── Color ramp selectors (fill/stroke on child shapes, text via .th/.ts) ── */
  .c-purple>rect,.c-purple>circle,.c-purple>ellipse{fill:var(--ramp-purple-fill);stroke:var(--ramp-purple-stroke);stroke-width:.5}
  .c-purple>.th{fill:var(--ramp-purple-th)!important} .c-purple>.ts{fill:var(--ramp-purple-ts)!important}
  .c-teal>rect,.c-teal>circle,.c-teal>ellipse{fill:var(--ramp-teal-fill);stroke:var(--ramp-teal-stroke);stroke-width:.5}
  .c-teal>.th{fill:var(--ramp-teal-th)!important} .c-teal>.ts{fill:var(--ramp-teal-ts)!important}
  .c-coral>rect,.c-coral>circle,.c-coral>ellipse{fill:var(--ramp-coral-fill);stroke:var(--ramp-coral-stroke);stroke-width:.5}
  .c-coral>.th{fill:var(--ramp-coral-th)!important} .c-coral>.ts{fill:var(--ramp-coral-ts)!important}
  .c-pink>rect,.c-pink>circle,.c-pink>ellipse{fill:var(--ramp-pink-fill);stroke:var(--ramp-pink-stroke);stroke-width:.5}
  .c-pink>.th{fill:var(--ramp-pink-th)!important} .c-pink>.ts{fill:var(--ramp-pink-ts)!important}
  .c-gray>rect,.c-gray>circle,.c-gray>ellipse{fill:var(--ramp-gray-fill);stroke:var(--ramp-gray-stroke);stroke-width:.5}
  .c-gray>.th{fill:var(--ramp-gray-th)!important} .c-gray>.ts{fill:var(--ramp-gray-ts)!important}
  .c-blue>rect,.c-blue>circle,.c-blue>ellipse{fill:var(--ramp-blue-fill);stroke:var(--ramp-blue-stroke);stroke-width:.5}
  .c-blue>.th{fill:var(--ramp-blue-th)!important} .c-blue>.ts{fill:var(--ramp-blue-ts)!important}
  .c-green>rect,.c-green>circle,.c-green>ellipse{fill:var(--ramp-green-fill);stroke:var(--ramp-green-stroke);stroke-width:.5}
  .c-green>.th{fill:var(--ramp-green-th)!important} .c-green>.ts{fill:var(--ramp-green-ts)!important}
  .c-amber>rect,.c-amber>circle,.c-amber>ellipse{fill:var(--ramp-amber-fill);stroke:var(--ramp-amber-stroke);stroke-width:.5}
  .c-amber>.th{fill:var(--ramp-amber-th)!important} .c-amber>.ts{fill:var(--ramp-amber-ts)!important}
  .c-red>rect,.c-red>circle,.c-red>ellipse{fill:var(--ramp-red-fill);stroke:var(--ramp-red-stroke);stroke-width:.5}
  .c-red>.th{fill:var(--ramp-red-th)!important} .c-red>.ts{fill:var(--ramp-red-ts)!important}
  .iv-container {
    width: 100%;
    max-width: 100%;
    overflow: hidden;
  }
  .iv-title {
    font-size: 15px;
    font-weight: 500;
    margin-bottom: 4px;
    color: var(--iv-text);
  }
  .iv-subtitle {
    font-size: 13px;
    color: var(--iv-muted);
    margin-bottom: 16px;
  }
  canvas { display: block; max-width: 100%; }
  .iv-footer {
    font-size: 11px;
    color: var(--iv-hint);
    margin-top: 12px;
    text-align: right;
  }
  /* Metric cards */
  .iv-metrics {
    display: grid;
    gap: 12px;
    margin-bottom: 16px;
  }
  .iv-metric {
    background: var(--iv-card);
    border-radius: 8px;
    padding: 14px 16px;
  }
  .iv-metric-label {
    font-size: 13px;
    color: var(--iv-muted);
    margin-bottom: 4px;
  }
  .iv-metric-value {
    font-size: 24px;
    font-weight: 500;
    color: var(--iv-text);
  }
  .iv-metric-sub {
    font-size: 13px;
    font-weight: 400;
    color: var(--iv-muted);
  }
  /* Custom legend */
  .iv-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin-bottom: 10px;
    font-size: 12px;
    color: var(--iv-muted);
  }
  .iv-legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
  }
  .iv-legend-swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  /* Interactive list rows */
  .iv-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    border-radius: 6px;
    background: var(--iv-card);
    cursor: pointer;
    font-size: 13px;
    transition: background 0.15s;
  }
  .iv-row:hover {
    background: var(--iv-card-hover);
  }
  .iv-row-dot {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .iv-row-label {
    flex: 1;
    color: var(--iv-text);
  }
  .iv-row-meta {
    font-size: 11px;
    color: var(--iv-muted);
  }
  /* Tables */
  .iv-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  .iv-table th {
    text-align: left;
    font-weight: 500;
    font-size: 12px;
    color: var(--iv-muted);
    padding: 8px 12px;
    border-bottom: 1px solid var(--iv-border);
    white-space: nowrap;
  }
  .iv-table td {
    padding: 8px 12px;
    border-bottom: 0.5px solid var(--iv-border-light);
    color: var(--iv-text);
  }
  .iv-table tr:last-child td { border-bottom: none; }
  .iv-table tr:hover td {
    background: var(--iv-card);
  }
  .iv-table .num { text-align: right; font-variant-numeric: tabular-nums; }
</style>
"""

_THEME_DETECTION_SCRIPT = """
<script>
(function() {
  function detectTheme(root) {
    return root.classList.contains('dark')
      || root.getAttribute('data-theme') === 'dark'
      || getComputedStyle(root).colorScheme === 'dark';
  }
  function applyTheme(isDark) {
    var theme = isDark ? 'dark' : 'light';
    if (document.documentElement.getAttribute('data-theme') === theme) return;
    document.documentElement.setAttribute('data-theme', theme);
  }
  try {
    var p = parent.document.documentElement;
    applyTheme(detectTheme(p));
    new MutationObserver(function() { applyTheme(detectTheme(p)); })
      .observe(p, { attributes: true, attributeFilter: ['class', 'data-theme', 'style'] });
  } catch(e) {
    var mq = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
    if (mq) {
      applyTheme(mq.matches);
      mq.addEventListener('change', function(e) { applyTheme(e.matches); });
    }
  }
})();
</script>
<script>
  // Theme JS vars for Chart.js (canvas can't use CSS vars) — separate script ensures IIFE completes first
  var _ivDark = document.documentElement.getAttribute('data-theme') === 'dark'
    || window.matchMedia('(prefers-color-scheme: dark)').matches;
  var _ivText = _ivDark ? '#c2c0b6' : '#3d3d3a';
  var _ivTextMuted = _ivDark ? '#888780' : '#73726c';
  var _ivGrid = _ivDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
  var _ivBgPrimary = getComputedStyle(document.documentElement).getPropertyValue('--bg-primary').trim() || (_ivDark ? '#2c2c2a' : 'Canvas');
  var _ivTooltipBg = _ivBgPrimary;
  var _ivTooltipBorder = _ivDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
</script>
"""

_BODY_SCRIPTS = """
<script>
// ── Height reporting (SVG overflow aware) ──
function reportHeight() {
  var b = document.body;
  var svgOverflow = 0;
  document.querySelectorAll('svg[viewBox]').forEach(function(svg) {
    try {
      var bbox = svg.getBBox();
      var vb = svg.viewBox.baseVal;
      if (vb && vb.width > 0 && vb.height > 0) {
        var overflow = bbox.y + bbox.height - (vb.y + vb.height);
        if (overflow > 0) {
          var scale = svg.getBoundingClientRect().width / vb.width;
          svgOverflow += Math.ceil(overflow * scale);
        }
      }
    } catch(e) {}
  });
  b.style.height = '0';
  var h = b.scrollHeight + svgOverflow;
  b.style.height = '';
  parent.postMessage({ type: 'iframe:height', height: h }, '*');
  // Legacy compat
  parent.postMessage({ type: 'resize', height: h + 16 }, '*');
}
window.addEventListener('load', reportHeight);
window.addEventListener('resize', reportHeight);
new ResizeObserver(reportHeight).observe(document.body);
document.addEventListener('toggle', function() {
  setTimeout(reportHeight, 50);
}, true);
setTimeout(reportHeight, 500);
setTimeout(reportHeight, 1500);

// ── sendPrompt bridge ──
function sendPrompt(text) {
  try {
    parent.postMessage({ type: 'input:prompt:submit', text: text }, '*');
  } catch(e) {}
}

// ── openLink bridge ──
function openLink(url) {
  try { parent.window.open(url, '_blank'); }
  catch(e) { window.open(url, '_blank'); }
}
</script>
"""

_STRICT_SECURITY_SCRIPT = """
<script>
(function() {
  function stripParams(rawUrl) {
    try { var u = new URL(rawUrl, location.href); u.search = ''; return u.toString(); }
    catch(e) { return rawUrl; }
  }
  var _origOpenLink = window.openLink;
  window.openLink = function(url) { _origOpenLink(stripParams(url)); };
  var _origOpen = window.open;
  window.open = function(url) {
    arguments[0] = stripParams(url);
    return _origOpen.apply(this, arguments);
  };
  function sanitizeLinks(root) {
    (root.querySelectorAll ? root : document).querySelectorAll('a[href]').forEach(function(a) {
      a.href = stripParams(a.href);
    });
  }
  sanitizeLinks(document);
  new MutationObserver(function(muts) {
    muts.forEach(function(m) {
      m.addedNodes.forEach(function(n) { if (n.nodeType === 1) sanitizeLinks(n); });
    });
  }).observe(document.body, { childList: true, subtree: true });
})();
</script>
"""


# ── CSP generation ────────────────────────────────────────────────────────────

_KNOWN_CDNS = (
    "https://cdnjs.cloudflare.com"
    " https://cdn.jsdelivr.net"
    " https://unpkg.com"
    " https://esm.sh"
    " https://d3js.org"
    " https://cdn.plot.ly"
)


def _build_csp_tag(level: str) -> str:
    """Return a <meta> CSP tag for the given security level, or empty string."""
    if level == "none":
        return ""
    if level == "strict":
        return (
            '<meta http-equiv="Content-Security-Policy" content="'
            f"default-src 'self'; "
            f"script-src 'unsafe-inline' {_KNOWN_CDNS}; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'none'; "
            "form-action 'none'; "
            "img-src 'self' data: blob:; "
            "font-src 'self' data:; "
            "media-src 'self'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            '">'
        )
    # balanced: block outbound connections & forms, allow external images
    return (
        '<meta http-equiv="Content-Security-Policy" content="'
        f"default-src 'self'; "
        f"script-src 'unsafe-inline' {_KNOWN_CDNS}; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'none'; "
        "form-action 'none'; "
        "img-src * data: blob:; "
        "font-src 'self' data:; "
        "media-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        '">'
    )


# ── HTML document builder ─────────────────────────────────────────────────────


def _wrap_html_doc(
    body: str, extra_head: str = "", min_height: int = 300, security_level: str = ""
) -> str:
    """Wrap body content in a full HTML document with common styles."""
    level = security_level or _security_level
    csp_tag = _build_csp_tag(level)
    strict_script = _STRICT_SECURITY_SCRIPT if level == "strict" else ""
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"{csp_tag}"
        f"{_COMMON_STYLE}{_THEME_DETECTION_SCRIPT}{extra_head}</head>"
        f"<body>{body}{_BODY_SCRIPTS}{strict_script}</body></html>"
    )


# ── Custom legend builder ─────────────────────────────────────────────────────


def _build_legend_html(
    datasets: List[Dict], labels: List[str] = None, is_radial: bool = False
) -> str:
    """Build a custom HTML legend with colored squares."""
    items = []
    if is_radial and labels:
        colors = _get_colors(len(labels))
        for i, label in enumerate(labels):
            color = colors[i]
            items.append(
                f'<span class="iv-legend-item">'
                f'<span class="iv-legend-swatch" style="background:{color}"></span>'
                f"{html.escape(str(label))}</span>"
            )
    else:
        for i, ds in enumerate(datasets):
            label = ds.get("label", f"Dataset {i+1}")
            color = ds.get("color", _get_colors(max(i + 1, 1))[i])
            items.append(
                f'<span class="iv-legend-item">'
                f'<span class="iv-legend-swatch" style="background:{color}"></span>'
                f"{html.escape(str(label))}</span>"
            )
    if not items:
        return ""
    return f'<div class="iv-legend">{"".join(items)}</div>'


# ── Metric cards builder ──────────────────────────────────────────────────────


def _build_metrics_html(metrics: List[Dict]) -> str:
    """Build metric cards row. Each metric: {label, value, sub?}."""
    if not metrics:
        return ""
    n = min(len(metrics), 4)
    cols = f"repeat({n}, minmax(0, 1fr))"
    cards = []
    for m in metrics[:4]:
        label = html.escape(str(m.get("label", "")))
        value = html.escape(str(m.get("value", "")))
        sub = m.get("sub", "")
        sub_html = (
            f'<span class="iv-metric-sub"> {html.escape(str(sub))}</span>'
            if sub
            else ""
        )
        cards.append(
            f'<div class="iv-metric">'
            f'<div class="iv-metric-label">{label}</div>'
            f'<div class="iv-metric-value">{value}{sub_html}</div>'
            f"</div>"
        )
    return f'<div class="iv-metrics" style="grid-template-columns:{cols}">{"".join(cards)}</div>'


# ── Chart.js builder ──────────────────────────────────────────────────────────


def _build_chartjs_html(spec: Dict[str, Any]) -> str:
    chart_type = spec.get("type", spec.get("chart_type", "bar"))
    title = spec.get("title", "")
    subtitle = spec.get("subtitle", "")
    labels = spec.get("labels", [])
    datasets_raw = spec.get("datasets", [])
    x_label = spec.get("x_label", "")
    y_label = spec.get("y_label", "")
    x_unit = spec.get("x_unit", "")
    y_unit = spec.get("y_unit", "")
    stacked = spec.get("stacked", False)
    log_x = spec.get("log_x", False)
    log_y = spec.get("log_y", False)
    horizontal = spec.get("horizontal", False)
    min_height = spec.get("min_height", 320)
    metrics = spec.get("metrics", [])
    show_legend = spec.get("show_legend", None)
    index_axis = spec.get("index_axis", "y" if horizontal else "x")

    if x_unit and x_label:
        x_label = f"{x_label} ({x_unit})"
    if y_unit and y_label:
        y_label = f"{y_label} ({y_unit})"

    n_datasets = len(datasets_raw)
    colors = _get_colors(n_datasets)

    datasets = []
    for i, ds in enumerate(datasets_raw):
        color = ds.get("color", colors[i])
        entry = {
            "label": ds.get("label", f"Dataset {i+1}"),
            "data": ds.get("data", []),
            "borderColor": color,
            "backgroundColor": color + "33",
            "borderWidth": 2,
            "pointRadius": 3 if chart_type in ("scatter", "bubble") else 2,
            "tension": 0.35 if chart_type == "line" else 0,
            "fill": ds.get("fill", chart_type == "line" and n_datasets == 1),
            "borderRadius": 6 if chart_type == "bar" else 0,
            "borderSkipped": False,
        }
        if chart_type in ("pie", "doughnut", "polarArea"):
            pie_colors = _get_colors(len(ds.get("data", [])))
            entry["backgroundColor"] = [c + "cc" for c in pie_colors]
            entry["borderColor"] = "var(--iv-pie-border)"
            entry["borderWidth"] = 3
        datasets.append(entry)

    is_radial = chart_type in ("pie", "doughnut", "polarArea", "radar")
    x_type = (
        "logarithmic" if log_x else "linear" if chart_type == "scatter" else "category"
    )
    y_type = "logarithmic" if log_y else "linear"

    # For horizontal bars (indexAxis: "y"), Chart.js swaps which axis is
    # category vs value. We must swap our explicit types to match,
    # otherwise the forced types conflict with indexAxis and bars vanish.
    if horizontal:
        x_type, y_type = y_type, x_type
        # Also swap axis labels if only one was provided
        if x_label and not y_label:
            x_label, y_label = "", x_label
        elif y_label and not x_label:
            x_label, y_label = y_label, ""

    if show_legend is None:
        needs_legend = (n_datasets > 1 or is_radial) and n_datasets <= 10
    else:
        needs_legend = show_legend

    config = {
        "type": chart_type,
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "indexAxis": index_axis,
            "animation": {"duration": 500, "easing": "easeOutQuart"},
            "plugins": {
                "legend": {"display": False},
                "tooltip": {
                    "backgroundColor": "_ivTooltipBg",
                    "titleColor": "_ivText",
                    "bodyColor": "_ivTextMuted",
                    "borderColor": "_ivTooltipBorder",
                    "borderWidth": 1,
                    "cornerRadius": 8,
                    "padding": 10,
                },
            },
        },
    }

    if not is_radial:
        config["options"]["scales"] = {
            "x": {
                "type": x_type,
                "title": {
                    "display": bool(x_label),
                    "text": x_label,
                    "color": "_ivTextMuted",
                    "font": {"size": 12},
                },
                "ticks": {"color": "_ivTextMuted", "font": {"size": 11}},
                "grid": {"color": "_ivGrid"},
                "stacked": stacked,
            },
            "y": {
                "type": y_type,
                "title": {
                    "display": bool(y_label),
                    "text": y_label,
                    "color": "_ivTextMuted",
                    "font": {"size": 12},
                },
                "ticks": {"color": "_ivTextMuted", "font": {"size": 11}},
                "grid": {"color": "_ivGrid"},
                "stacked": stacked,
            },
        }
    else:
        if chart_type == "doughnut":
            config["options"]["cutout"] = "62%"

    config_json = _escape_json_for_js(config)

    config_json = config_json.replace('"_ivTooltipBg"', "_ivTooltipBg")
    config_json = config_json.replace('"_ivText"', "_ivText")
    config_json = config_json.replace('"_ivTextMuted"', "_ivTextMuted")
    config_json = config_json.replace('"_ivTooltipBorder"', "_ivTooltipBorder")
    config_json = config_json.replace('"_ivGrid"', "_ivGrid")
    config_json = config_json.replace(
        '"var(--iv-pie-border)"', '_ivBgPrimary'
    )

    title_html = f'<div class="iv-title">{html.escape(title)}</div>' if title else ""
    subtitle_html = (
        f'<div class="iv-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    )
    metrics_html = _build_metrics_html(metrics)
    legend_html = (
        _build_legend_html(datasets_raw, labels if is_radial else None, is_radial)
        if needs_legend
        else ""
    )

    chart_height = min_height - 60
    if horizontal:
        n_bars = (
            len(labels)
            if labels
            else len(datasets_raw[0].get("data", [])) if datasets_raw else 5
        )
        chart_height = max(chart_height, n_bars * 40 + 60)

    body = f"""
<div class="iv-container">
  {title_html}
  {subtitle_html}
  {metrics_html}
  {legend_html}
  <div style="position:relative; height:{chart_height}px;">
    <canvas id="chart"></canvas>
  </div>
</div>
"""

    extra_head = (
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1'
        '/chart.umd.js"></script>'
    )

    script = f"""
<script>
  const cfg = {config_json};
  new Chart(document.getElementById('chart'), cfg);
</script>
"""
    return _wrap_html_doc(body + script, extra_head=extra_head, min_height=min_height)


# ── Plotly builder ────────────────────────────────────────────────────────────


def _build_plotly_html(spec: Dict[str, Any]) -> str:
    title = spec.get("title", "")
    traces = spec.get("traces", [])
    layout_overrides = spec.get("layout", {})
    min_height = spec.get("min_height", 420)
    metrics = spec.get("metrics", [])

    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "system-ui, -apple-system, sans-serif", "size": 13},
        "margin": {"l": 56, "r": 24, "t": 36, "b": 48},
        "autosize": True,
        "height": min_height - 60,
    }
    layout.update(layout_overrides)

    traces_json = _escape_json_for_js(traces)
    layout_json = _escape_json_for_js(layout)

    title_html = f'<div class="iv-title">{html.escape(title)}</div>' if title else ""
    metrics_html = _build_metrics_html(metrics)

    body = f"""
<div class="iv-container">
  {title_html}
  {metrics_html}
  <div id="plotly-chart" style="width:100%;"></div>
</div>
<script>
  // Apply dark/light theme to Plotly layout
  const _pLayout = {layout_json};
  _pLayout.font.color = _ivText;
  if (_pLayout.xaxis) _pLayout.xaxis.gridcolor = _ivGrid;
  if (_pLayout.yaxis) _pLayout.yaxis.gridcolor = _ivGrid;
  // Auto-set axis colors for any xaxis/yaxis keys
  for (const k of Object.keys(_pLayout)) {{
    if (k.startsWith('xaxis') || k.startsWith('yaxis')) {{
      _pLayout[k] = _pLayout[k] || {{}};
      _pLayout[k].gridcolor = _pLayout[k].gridcolor || _ivGrid;
      _pLayout[k].linecolor = _pLayout[k].linecolor || _ivGrid;
      if (_pLayout[k].title) {{
        if (typeof _pLayout[k].title === 'string') {{
          _pLayout[k].title = {{ text: _pLayout[k].title, font: {{ color: _ivTextMuted }} }};
        }} else {{
          _pLayout[k].title.font = _pLayout[k].title.font || {{}};
          _pLayout[k].title.font.color = _ivTextMuted;
        }}
      }}
      _pLayout[k].tickfont = _pLayout[k].tickfont || {{}};
      _pLayout[k].tickfont.color = _ivTextMuted;
    }}
  }}
  Plotly.newPlot('plotly-chart', {traces_json}, _pLayout,
    {{responsive: true, displayModeBar: false}});
</script>
"""

    extra_head = (
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.1'
        '/plotly-basic.min.js"></script>'
    )
    return _wrap_html_doc(body, extra_head=extra_head, min_height=min_height)


# ── Mermaid builder ───────────────────────────────────────────────────────────


def _build_mermaid_html(spec: Dict[str, Any]) -> str:
    code = _sanitize_mermaid_code(_normalize_mermaid_code(spec))
    title = spec.get("title", "")
    min_height = spec.get("min_height", 300)

    if not code:
        spec_keys = list(spec.keys())
        return f"<p>Error: empty mermaid code. Spec keys received: {spec_keys}</p>"

    # Pass the raw code to JS — let the browser-side handle retry/repair
    # This avoids double-escaping issues from Python html.escape + JS template literals
    import base64

    code_b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")

    title_html = f'<div class="iv-title">{html.escape(title)}</div>' if title else ""

    body = f"""
<div class="iv-container">
  {title_html}
  <div id="mermaid-output"></div>
</div>
<script type="module">
  import mermaid from 'https://esm.sh/mermaid@11/dist/mermaid.esm.min.mjs';
  const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  mermaid.initialize({{
    startOnLoad: false,
    theme: 'base',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    themeVariables: {{
      darkMode: dark,
      fontSize: '13px',
      lineColor: dark ? '#9c9a92' : '#73726c',
      textColor: dark ? '#c2c0b6' : '#3d3d3a',
      primaryColor: dark ? '#3C3489' : '#EEEDFE',
      primaryBorderColor: dark ? '#AFA9EC' : '#534AB7',
      primaryTextColor: dark ? '#CECBF6' : '#26215C',
      secondaryColor: dark ? '#085041' : '#E1F5EE',
      tertiaryColor: dark ? '#2C2C2A' : '#F1EFE8',
    }},
  }});

  const rawCode = atob('{code_b64}');
  const out = document.getElementById('mermaid-output');

  // Repair functions for sequence diagrams with stripped newlines
  function repairSequence(code) {{
    // Insert newlines before known sequence keywords
    let fixed = code;
    // Before participant/actor declarations
    fixed = fixed.replace(/\\s+(participant |actor )/g, '\\n$1');
    // Before arrows: X->>Y, X-->>Y, X-xY, X--xY
    // The key pattern: word chars then arrow then word chars
    fixed = fixed.replace(/([\\w\\s]+?)\\s*(->>|-->>|-x|--x|->|-->)\\s*/g, '\\n$1$2');
    // Before Note/note
    fixed = fixed.replace(/\\s+(Note |note )/g, '\\n$1');
    // Before loop/alt/opt/par/rect/critical/end
    fixed = fixed.replace(/\\s+(loop |alt |else |opt |par |rect |critical |end\\b)/g, '\\n$1');
    // Before activate/deactivate
    fixed = fixed.replace(/\\s+(activate |deactivate )/g, '\\n$1');
    // Clean: remove leading newline, dedupe newlines
    fixed = fixed.replace(/^\\n+/, '').replace(/\\n{{3,}}/g, '\\n\\n');
    // Ensure first line is sequenceDiagram
    if (!fixed.trim().toLowerCase().startsWith('sequencediagram')) {{
      fixed = 'sequenceDiagram\\n' + fixed;
    }}
    return fixed;
  }}

  function repairFlowchart(code) {{
    let fixed = code;
    // Insert newline before node IDs that follow a closing bracket
    fixed = fixed.replace(/(\\])\\s+([A-Za-z_])/g, '$1\\n$2');
    fixed = fixed.replace(/(\\}})\\s+([A-Za-z_])/g, '$1\\n$2');
    fixed = fixed.replace(/(\\))\\s+([A-Za-z_])/g, '$1\\n$2');
    // Before subgraph/end/classDef/style
    fixed = fixed.replace(/\\s+(subgraph |end\\b|classDef |style |linkStyle )/g, '\\n$1');
    if (!fixed.trim().toLowerCase().startsWith('flowchart') && !fixed.trim().toLowerCase().startsWith('graph')) {{
      fixed = 'flowchart TD\\n' + fixed;
    }}
    return fixed;
  }}

  function repairGeneric(code) {{
    // Try sequence repair first (most common failure case)
    if (code.includes('->>') || code.includes('-->>') || code.toLowerCase().includes('participant')) {{
      return repairSequence(code);
    }}
    if (code.includes('-->') || code.toLowerCase().includes('subgraph')) {{
      return repairFlowchart(code);
    }}
    return code;
  }}

  // Build a simple HTML fallback for when Mermaid parsing fails completely
  function htmlFallback(code) {{
    const lines = code.split('\\n').filter(l => l.trim());
    // Try to render as a simple step list
    const steps = lines
      .filter(l => l.includes('->>') || l.includes('-->>') || l.includes('-->') || l.includes('-x'))
      .map(l => {{
        const clean = l.replace(/^\\s+/, '');
        return '<div style="padding:6px 12px;margin:4px 0;background:var(--iv-card);border-radius:6px;font-size:13px;color:var(--iv-text);font-family:monospace;white-space:pre-wrap">' + clean.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</div>';
      }});
    if (steps.length === 0) {{
      // Just show the raw code
      return '<pre style="padding:12px;background:var(--iv-card);border-radius:8px;font-size:12px;color:var(--iv-text);overflow-x:auto;white-space:pre-wrap">' + code.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</pre>';
    }}
    return '<div style="display:flex;flex-direction:column;gap:2px">' + steps.join('') + '</div>' +
      '<div style="font-size:11px;color:var(--iv-hint);margin-top:8px">Mermaid rendering failed — showing diagram as text</div>';
  }}

  // Progressive retry: raw → repair → fallback
  async function tryRender(code, id) {{
    try {{
      const {{ svg }} = await mermaid.render(id, code);
      return svg;
    }} catch(e) {{
      return null;
    }}
  }}

  let svg = await tryRender(rawCode, 'mermaid-svg-1');
  if (!svg) {{
    // Attempt repair
    const repaired = repairGeneric(rawCode);
    svg = await tryRender(repaired, 'mermaid-svg-2');
  }}
  if (svg) {{
    out.innerHTML = svg;
  }} else {{
    // All Mermaid attempts failed — render HTML fallback
    out.innerHTML = htmlFallback(rawCode);
  }}
</script>
"""
    return _wrap_html_doc(body, min_height=min_height)


# ── Table builder ─────────────────────────────────────────────────────────────


def _build_table_html(spec: Dict[str, Any]) -> str:
    title = spec.get("title", "")
    subtitle = spec.get("subtitle", "")
    headers = spec.get("headers", [])
    rows = spec.get("rows", [])
    sortable = spec.get("sortable", True)
    col_types = spec.get("col_types", [])
    metrics = spec.get("metrics", [])

    title_html = f'<div class="iv-title">{html.escape(title)}</div>' if title else ""
    subtitle_html = (
        f'<div class="iv-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    )
    metrics_html = _build_metrics_html(metrics)

    th_cells = []
    for i, h in enumerate(headers):
        is_num = i < len(col_types) and col_types[i] == "num"
        cls = ' class="num"' if is_num else ""
        sort_attr = (
            f' onclick="ivSort({i})" style="cursor:pointer;user-select:none"'
            if sortable
            else ""
        )
        th_cells.append(f"<th{cls}{sort_attr}>{html.escape(str(h))}</th>")

    tr_rows = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            is_num = i < len(col_types) and col_types[i] == "num"
            cls = ' class="num"' if is_num else ""
            cells.append(f"<td{cls}>{html.escape(str(cell))}</td>")
        tr_rows.append(f"<tr>{''.join(cells)}</tr>")

    sort_script = ""
    if sortable:
        sort_script = """
<script>
  let _ivSortDir = {};
  function ivSort(col) {
    _ivSortDir[col] = !_ivSortDir[col];
    const tbody = document.querySelector('.iv-table tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {
      let va = a.cells[col].textContent.trim();
      let vb = b.cells[col].textContent.trim();
      const na = parseFloat(va.replace(/[^0-9.\\-]/g, ''));
      const nb = parseFloat(vb.replace(/[^0-9.\\-]/g, ''));
      if (!isNaN(na) && !isNaN(nb)) { va = na; vb = nb; }
      if (va < vb) return _ivSortDir[col] ? -1 : 1;
      if (va > vb) return _ivSortDir[col] ? 1 : -1;
      return 0;
    });
    rows.forEach(r => tbody.appendChild(r));
  }
</script>
"""

    body = f"""
<div class="iv-container">
  {title_html}
  {subtitle_html}
  {metrics_html}
  <table class="iv-table">
    <thead><tr>{"".join(th_cells)}</tr></thead>
    <tbody>{"".join(tr_rows)}</tbody>
  </table>
  {sort_script}
</div>
"""
    return _wrap_html_doc(body)


# ── Metric cards (standalone) ─────────────────────────────────────────────────


def _build_metrics_only_html(spec: Dict[str, Any]) -> str:
    title = spec.get("title", "")
    metrics = spec.get("metrics", [])
    title_html = f'<div class="iv-title">{html.escape(title)}</div>' if title else ""
    metrics_html = _build_metrics_html(metrics)

    body = f"""
<div class="iv-container">
  {title_html}
  {metrics_html}
</div>
"""
    return _wrap_html_doc(body, min_height=80)


# ── Visual router ─────────────────────────────────────────────────────────────


# ── Spec normalizer — handles Gemini sending raw Chart.js config ──────────────
# Gemini frequently sends the native Chart.js format:
#   {"type": "bar", "data": {"labels": [...], "datasets": [...]}, "options": {...}}
# instead of our flat format:
#   {"type": "bar", "labels": [...], "datasets": [...]}
# This normalizer unwraps the nested format so our builders work.


def _normalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap Chart.js native config format into our flat spec format."""
    # If spec has "data" dict with "labels" or "datasets", unwrap them to root
    data_block = spec.get("data")
    if isinstance(data_block, dict):
        if "labels" in data_block and "labels" not in spec:
            spec["labels"] = data_block["labels"]
        if "datasets" in data_block and "datasets" not in spec:
            spec["datasets"] = data_block["datasets"]

    # Extract useful bits from "options" if Gemini sent them
    opts = spec.get("options")
    if isinstance(opts, dict):
        # indexAxis: "y" means horizontal bar
        if opts.get("indexAxis") == "y" and not spec.get("horizontal"):
            spec["horizontal"] = True

        # Extract axis titles from options.scales
        scales = opts.get("scales", {})
        for axis_key in ("x", "y"):
            axis = scales.get(axis_key, {})
            title_obj = axis.get("title", {})
            if isinstance(title_obj, dict) and title_obj.get("text"):
                label_key = f"{axis_key}_label"
                if not spec.get(label_key):
                    spec[label_key] = title_obj["text"]

        # Extract chart title from options.plugins.title
        plugins = opts.get("plugins", {})
        title_obj = plugins.get("title", {})
        if isinstance(title_obj, dict) and title_obj.get("text"):
            if not spec.get("title"):
                spec["title"] = title_obj["text"]

    return spec


def _build_visual(spec: Dict[str, Any]) -> str:
    spec = _normalize_spec(spec)
    vis_type = spec.get("type", spec.get("chart_type", "bar")).lower()

    if vis_type == "mermaid":
        return _build_mermaid_html(spec)

    if vis_type == "table":
        return _build_table_html(spec)

    if vis_type in ("metrics", "metric_cards", "kpi"):
        return _build_metrics_only_html(spec)

    if vis_type in ("plotly", "heatmap", "surface", "contour", "3d"):
        if vis_type in ("heatmap", "surface", "contour"):
            for trace in spec.get("traces", []):
                if "type" not in trace:
                    trace["type"] = vis_type
        return _build_plotly_html(spec)

    # Default: Chart.js
    if vis_type in (
        "line",
        "bar",
        "scatter",
        "pie",
        "doughnut",
        "radar",
        "polarArea",
        "bubble",
        "horizontal_bar",
    ):
        if vis_type == "horizontal_bar":
            spec["type"] = "bar"
            spec["horizontal"] = True
        return _build_chartjs_html(spec)

    return _build_chartjs_html(spec)


# ═══════════════════════════════════════════════════════════════════════════════
#  OWUI Tools class — ONLY public methods here
# ═══════════════════════════════════════════════════════════════════════════════


class Tools:
    class Valves(BaseModel):
        default_min_height: int = Field(
            default=320,
            description="Default minimum height (px) for inline visuals",
        )
        security_level: Literal["strict", "balanced", "none"] = Field(
            default="strict",
            description="CSP level. Strict: blocks outbound requests+images. Balanced: allows external images. None: no restrictions.",
        )
        debug: bool = Field(
            default=False,
            description="Log visual specs to stdout for debugging",
        )

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True,
            description="Show 'Rendering visual...' status messages",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def render_inline_visual(
        self,
        spec_json: str = "",
        __event_emitter__: Optional[Any] = None,
    ) -> str:
        """
        Render an interactive visualization inline in the chat.

        :param spec_json: JSON string describing the visual. Must include "type". Supported types: line, bar, scatter, pie, doughnut, radar, polarArea, bubble, horizontal_bar, plotly, heatmap, surface, contour, mermaid, table, metrics. Optional keys: metrics (array of {label,value,sub} for summary cards above charts), show_legend (bool), horizontal (bool for bar charts).
        :return: Brief text description of what was rendered.
        """
        await _emit(__event_emitter__, "Rendering visual…")

        global _security_level
        _security_level = self.valves.security_level

        try:
            spec = _parse_spec_json(spec_json)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            await _emit(__event_emitter__, f"Invalid JSON: {e}", done=True)
            return f"Error: could not parse spec_json — {e}"

        if not spec.get("min_height"):
            spec["min_height"] = self.valves.default_min_height

        if self.valves.debug:
            logger.info(f"[InlineVisuals] spec: {json.dumps(spec, indent=2)}")

        try:
            html_doc = _build_visual(spec)
        except Exception as e:
            await _emit(__event_emitter__, f"Build error: {e}", done=True)
            return f"Error building visual: {e}"

        await _emit_html(__event_emitter__, html_doc)
        await _emit(__event_emitter__, "Visual rendered", done=True)

        vis_type = spec.get("type", spec.get("chart_type", "visual"))
        title = spec.get("title", "")
        n_datasets = len(spec.get("datasets", spec.get("traces", [])))
        summary = f"[Inline {vis_type} visual"
        if title:
            summary += f": {title}"
        if n_datasets:
            summary += f" ({n_datasets} dataset{'s' if n_datasets > 1 else ''})"
        summary += " rendered in chat]"
        return summary

    async def render_inline_html(
        self,
        html_content: str = "",
        title: str = "",
        min_height: int = 300,
        __event_emitter__: Optional[Any] = None,
    ) -> str:
        """
        Render raw HTML inline in the chat for custom interactive content.

        :param html_content: HTML body content (not a full document — will be wrapped with theme-aware styles). Can include <script> and <style> tags. For CDN libraries, include them as <script src="...">. Theme JS vars: _ivDark (bool), _ivText, _ivTextMuted, _ivGrid. CSS vars: --iv-text, --iv-muted, --iv-hint, --iv-card, --iv-border (auto light/dark). SVG utility classes: .t (14px text), .ts (12px muted), .th (14px bold), .box (neutral rect), .node (clickable), .arr (arrow line), .leader (dashed guide). Color ramp classes on <g>: .c-teal, .c-purple, .c-coral, .c-blue, .c-amber, .c-green, .c-pink, .c-gray, .c-red. Bridges: sendPrompt(text) sends a message to the chat, openLink(url) opens a URL in a new tab.
        :param title: Optional title displayed above the content.
        :param min_height: Minimum height in pixels (default 300).
        :return: Brief text description of what was rendered.
        """
        await _emit(__event_emitter__, "Rendering HTML…")

        global _security_level
        _security_level = self.valves.security_level

        if not html_content.strip():
            await _emit(__event_emitter__, "Empty content", done=True)
            return "Error: html_content was empty."

        # If user passed a full document, use as-is
        if re.search(r"<(!DOCTYPE|html)\b", html_content, re.IGNORECASE):
            full_html = html_content
        else:
            title_html = (
                f'<div class="iv-title">{html.escape(title)}</div>' if title else ""
            )
            body = f'<div class="iv-container">{title_html}{html_content}</div>'
            full_html = _wrap_html_doc(body, min_height=min_height)

        await _emit_html(__event_emitter__, full_html)
        await _emit(__event_emitter__, "HTML rendered", done=True)

        summary = "[Inline HTML"
        if title:
            summary += f": {title}"
        summary += " rendered in chat]"
        return summary
