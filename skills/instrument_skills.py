"""
title: FORTHought Instrument Skills
description: Tool gateway for instrument analysis model profiles (Gemini VLM).
    SEM microscopy (FFT periodicity + particle sizing), XRD phase identification
    & purity analysis (v2: parse, identify, search, Origin export),
    OriginLab data engine (inspect, get_data, fit, graph, audit — standalone
    'origin.*' namespace), PL experimental planning (laser/filter/optics
    recommendation, material emission lookup, substrate enhancement, SHG/THG
    planning, strain analysis, PLE planning, imaging, valley polarization),
    document creation, and chat-scoped image discovery.
    Attach Web Search separately in model settings.
author: Marios Adamidis (FORTHought Lab)
version: 1.0.0
required_open_webui_version: 0.6.6
"""

# ═══════════════════════════════════════════════════════════════════════════
#  Imports
# ═══════════════════════════════════════════════════════════════════════════

import os, re, json, time, asyncio, logging, ipaddress
from typing import Optional, Any, Dict, List
from pathlib import Path
from urllib.parse import urlparse

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from pydantic import BaseModel, Field

log = logging.getLogger("forthought.skills.instrument")

# ═══════════════════════════════════════════════════════════════════════════
#  MCP Streamable-HTTP micro-client  (shared singleton)
# ═══════════════════════════════════════════════════════════════════════════

_PROTOCOL_VERSION = "2025-03-26"


class _MCPClient:
    """Tiny MCP Streamable-HTTP client with session management and SSE support."""

    def __init__(self) -> None:
        self._sessions: Dict[str, str] = {}
        self._init_ts: Dict[str, float] = {}
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        if httpx is None:
            raise RuntimeError("httpx is not installed")
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(300, connect=15),
                follow_redirects=True,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    async def _ensure_session(self, url: str, auth_token: Optional[str] = None) -> str:
        now = time.monotonic()
        if url in self._sessions and (now - self._init_ts.get(url, 0)) < 600:
            return self._sessions[url]

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": _PROTOCOL_VERSION,
        }
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "FORTHought-Instrument-Skills",
                    "version": "3.2.1",
                },
            },
        }
        client = self._get_client()
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()

        sid = resp.headers.get("mcp-session-id", "")
        if sid:
            self._sessions[url] = sid
            self._init_ts[url] = now

        notify = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        nh = {**headers}
        if sid:
            nh["MCP-Session-Id"] = sid
        try:
            await client.post(url, json=notify, headers=nh)
        except Exception:
            pass

        return sid

    async def call(
        self,
        url: str,
        tool_name: str,
        arguments: dict,
        timeout: int = 120,
        auth_token: Optional[str] = None,
        retries: int = 2,
    ) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                sid = await self._ensure_session(url, auth_token)
                headers: Dict[str, str] = {
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "MCP-Protocol-Version": _PROTOCOL_VERSION,
                }
                if auth_token:
                    headers["Authorization"] = f"Bearer {auth_token}"
                if sid:
                    headers["MCP-Session-Id"] = sid

                payload = {
                    "jsonrpc": "2.0",
                    "id": int(time.time() * 1000) % 999999,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                }
                client = self._get_client()
                resp = await client.post(
                    url, json=payload, headers=headers, timeout=timeout
                )
                resp.raise_for_status()

                ct = resp.headers.get("content-type", "")

                # SSE streaming response
                if "text/event-stream" in ct:
                    return self._extract_sse_result(resp.text)

                # Normal JSON response
                data = resp.json()
                if "error" in data:
                    err_msg = data["error"]
                    if isinstance(err_msg, dict):
                        err_msg = err_msg.get("message", str(err_msg))
                    return json.dumps({"error": str(err_msg)})

                result = data.get("result", data)
                content = result.get("content") if isinstance(result, dict) else None
                if content and isinstance(content, list):
                    texts = [
                        c.get("text", "") for c in content if c.get("type") == "text"
                    ]
                    return "\n".join(texts) if texts else json.dumps(result)
                return json.dumps(result, default=str)

            except Exception as e:
                last_err = e
                log.warning(
                    f"[MCP] attempt {attempt+1} failed for {tool_name}@{url}: {e}"
                )
                if attempt < retries:
                    self._sessions.pop(url, None)
                    await asyncio.sleep(1.5 * (attempt + 1))

        raise last_err or RuntimeError(f"MCP call failed for {tool_name}")

    def _extract_sse_result(self, text: str) -> str:
        last_data = ""
        for line in text.split("\n"):
            if line.startswith("data: "):
                last_data = line[6:]
        if not last_data:
            return json.dumps({"error": "Empty SSE stream"})
        try:
            msg = json.loads(last_data)
            result = msg.get("result", msg)
            content = result.get("content") if isinstance(result, dict) else None
            if content and isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return "\n".join(texts) if texts else json.dumps(result)
            return json.dumps(result, default=str)
        except (json.JSONDecodeError, TypeError):
            return last_data

    async def list_tools(
        self, url: str, auth_token: Optional[str] = None
    ) -> List[dict]:
        sid = await self._ensure_session(url, auth_token)
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": _PROTOCOL_VERSION,
        }
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        if sid:
            headers["MCP-Session-Id"] = sid

        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000) % 999999,
            "method": "tools/list",
            "params": {},
        }
        client = self._get_client()
        resp = await client.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", {}).get("tools", [])


_mcp = _MCPClient()


# ═══════════════════════════════════════════════════════════════════════════
#  URL safety
# ═══════════════════════════════════════════════════════════════════════════


def _is_url_safe(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname or ""
        try:
            ip = ipaddress.ip_address(host)
            return ip.is_global
        except ValueError:
            pass
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  Image file helpers  (for get_image_path)
# ═══════════════════════════════════════════════════════════════════════════

_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".svg",
}


def _is_image_filename(name: str) -> bool:
    return Path(name).suffix.lower() in _IMAGE_EXTS if name else False


def _extract_image_file_ids(
    files: Optional[List[dict]] = None,
    messages: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Scan current-message attachments and chat history for image file_ids.
    Returns list of {'file_id': str, 'name': str, 'source': str} dicts,
    most-recent-first (current message files first, then reverse-chrono messages).
    """
    results: List[dict] = []
    seen: set = set()

    # --- 1. Current message attachments (__files__) — highest priority ---
    for f in files or []:
        fid = f.get("id") or (f.get("file", {}) or {}).get("id")
        fname = f.get("name") or (f.get("file", {}) or {}).get("filename") or ""
        ftype = f.get("type", "")
        if not fid or fid in seen:
            continue
        if ftype == "image" or _is_image_filename(fname):
            results.append(
                {"file_id": str(fid), "name": fname, "source": "current_message"}
            )
            seen.add(fid)

    # --- 2. Chat history (__messages__) — reverse order for recency ---
    for msg in reversed(messages or []):
        for f in msg.get("files") or []:
            fid = f.get("id") or (f.get("file", {}) or {}).get("id")
            fname = f.get("name") or (f.get("file", {}) or {}).get("filename") or ""
            ftype = f.get("type", "")
            if not fid or fid in seen:
                continue
            if ftype == "image" or _is_image_filename(fname):
                results.append(
                    {"file_id": str(fid), "name": fname, "source": "chat_history"}
                )
                seen.add(fid)

    return results


def _resolve_file_id_to_path(file_id: str, uploads_dir: str) -> Optional[str]:
    """
    Resolve an OWUI file_id to its filesystem path.
    Supports:
      A) uploads_dir/<file_id>_<original_filename>  (flat)
      B) uploads_dir/<file_id>/<somefile>            (subdirectory)
    """
    base = Path(uploads_dir)
    if not base.exists():
        return None
    for p in base.glob(f"{file_id}_*"):
        if p.is_file():
            return str(p)
    id_dir = base / file_id
    if id_dir.is_dir():
        for p in sorted(id_dir.iterdir()):
            if p.is_file():
                return str(p)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Namespace alias: spec.* → origin.*  (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════

# Old spec.* names map to new origin.* canonical names.
# run() resolves these before registry lookup.
_NAMESPACE_ALIASES = {
    "spec.inspect": "origin.inspect",
    "spec.get_data": "origin.get_data",
    "spec.merge": "origin.merge",
    "spec.fit_peak": "origin.fit_peak",
    "spec.batch_fit": "origin.batch_fit",
    "spec.create_graph": "origin.create_graph",
    "spec.save_project": "origin.save_project",
    "spec.export_csv": "origin.export_csv",
    "spec.import_csv": "origin.import_csv",
    "spec.multi_fit": "origin.multi_fit",
    "spec.batch_multi_fit": "origin.batch_multi_fit",
    "spec.trend_plot": "origin.trend_plot",
    "spec.normalize": "origin.normalize",
    "spec.smooth": "origin.smooth",
    "spec.find_peaks": "origin.find_peaks",
    # Also handle bare opj_ names models sometimes hallucinate
    "opj_inspect": "origin.inspect",
    "opj_get_data": "origin.get_data",
    "opj_fit_peak": "origin.fit_peak",
    "opj_batch_fit": "origin.batch_fit",
    "opj_create_graph": "origin.create_graph",
    "opj_save_project": "origin.save_project",
    "opj_export_csv": "origin.export_csv",
    "opj_import_csv": "origin.import_csv",
    "opj_multi_fit": "origin.multi_fit",
    "opj_batch_multi_fit": "origin.batch_multi_fit",
    "opj_merge_power_series": "origin.merge",
    "opj_trend_plot": "origin.trend_plot",
    "opj_normalize": "origin.normalize",
    "opj_smooth": "origin.smooth",
    "opj_find_peaks": "origin.find_peaks",
}


def _resolve_tool_alias(name: str) -> str:
    """Resolve legacy spec.* and opj_* names to origin.* canonical form."""
    if name in _NAMESPACE_ALIASES:
        return _NAMESPACE_ALIASES[name]
    # Handle spec.opj_* pattern (some models add the opj_ prefix)
    if name.startswith("spec.opj_"):
        tail = name[9:]  # strip "spec.opj_"
        candidate = f"origin.{tail}"
        if candidate in _NAMESPACE_ALIASES.values():
            return candidate
    # Handle opj_* not explicitly listed
    if name.startswith("opj_"):
        candidate = f"origin.{name[4:]}"
        # Check if it resolves to a known canonical name
        return candidate  # Will fail at registry lookup if truly unknown
    return name


# ═══════════════════════════════════════════════════════════════════════════
#  Registry — Instrument profile: micro, xrd, origin, pl, files
# ═══════════════════════════════════════════════════════════════════════════


def _build_registry(valves: Any) -> Dict[str, dict]:
    t = valves.default_timeout
    return {
        # -- microscopy --
        "micro.sem_fft": {
            "url": valves.micro_url,
            "mcp": "sem_analyze_fft",
            "timeout": t,
        },
        # -- xrd (v2: 5 tools) --
        "xrd.analyze": {
            "url": valves.xrd_url,
            "mcp": "analyze_xrd",
            "timeout": t,
        },
        "xrd.identify": {
            "url": valves.xrd_url,
            "mcp": "identify_xrd",
            "timeout": t,
        },
        "xrd.parse": {
            "url": valves.xrd_url,
            "mcp": "parse_xrd",
            "timeout": t,
        },
        "xrd.search_ref": {
            "url": valves.xrd_url,
            "mcp": "search_xrd_ref",
            "timeout": t,
        },
        "xrd.export_origin": {
            "url": valves.xrd_url,
            "mcp": "export_xrd_origin",
            "timeout": t,
        },
        # -- origin (standalone OriginLab engine, was spec.*) --
        "origin.inspect": {
            "url": valves.origin_url,
            "mcp": "opj_inspect",
            "timeout": valves.origin_timeout,
        },
        "origin.get_data": {
            "url": valves.origin_url,
            "mcp": "opj_get_data",
            "timeout": valves.origin_timeout,
        },
        "origin.merge": {
            "url": valves.origin_url,
            "mcp": "opj_merge_power_series",
            "timeout": valves.origin_timeout,
        },
        "origin.fit_peak": {
            "url": valves.origin_url,
            "mcp": "opj_fit_peak",
            "timeout": valves.origin_timeout,
        },
        "origin.batch_fit": {
            "url": valves.origin_url,
            "mcp": "opj_batch_fit",
            "timeout": valves.origin_timeout,
        },
        "origin.multi_fit": {
            "url": valves.origin_url,
            "mcp": "opj_multi_fit",
            "timeout": valves.origin_timeout,
        },
        "origin.batch_multi_fit": {
            "url": valves.origin_url,
            "mcp": "opj_batch_multi_fit",
            "timeout": valves.origin_timeout,
        },
        "origin.create_graph": {
            "url": valves.origin_url,
            "mcp": "opj_create_graph",
            "timeout": valves.origin_timeout,
        },
        "origin.save_project": {
            "url": valves.origin_url,
            "mcp": "opj_save_project",
            "timeout": valves.origin_timeout,
        },
        "origin.export_csv": {
            "url": valves.origin_url,
            "mcp": "opj_export_csv",
            "timeout": valves.origin_timeout,
        },
        "origin.import_csv": {
            "url": valves.origin_url,
            "mcp": "opj_import_csv",
            "timeout": valves.origin_timeout,
        },
        "origin.trend_plot": {
            "url": valves.origin_url,
            "mcp": "opj_trend_plot",
            "timeout": valves.origin_timeout + 30,  # runs batch_multi_fit internally
        },
        "origin.normalize": {
            "url": valves.origin_url,
            "mcp": "opj_normalize",
            "timeout": valves.origin_timeout,
        },
        "origin.smooth": {
            "url": valves.origin_url,
            "mcp": "opj_smooth",
            "timeout": valves.origin_timeout,
        },
        "origin.find_peaks": {
            "url": valves.origin_url,
            "mcp": "opj_find_peaks",
            "timeout": valves.origin_timeout,
        },
        # -- files (for saving results) --
        "doc.create": {
            "url": valves.files_url,
            "mcp": "create_file",
            "timeout": t,
            "auth": "files",
        },
        "doc.read": {
            "url": valves.files_url,
            "mcp": "full_context_document",
            "timeout": t,
            "auth": "files",
        },
        "doc.edit": {
            "url": valves.files_url,
            "mcp": "edit_document",
            "timeout": t,
            "auth": "files",
        },
        # -- PL experimental planning (v3: 11 tools) --
        "pl.recommend": {
            "url": valves.pl_url,
            "mcp": "pl_recommend",
            "timeout": t,
        },
        "pl.material_lookup": {
            "url": valves.pl_url,
            "mcp": "pl_material_lookup",
            "timeout": t,
        },
        "pl.filter_search": {
            "url": valves.pl_url,
            "mcp": "pl_filter_search",
            "timeout": t,
        },
        "pl.check_setup": {
            "url": valves.pl_url,
            "mcp": "pl_check_setup",
            "timeout": t,
        },
        "pl.spectrum_sketch": {
            "url": valves.pl_url,
            "mcp": "pl_spectrum_sketch",
            "timeout": t,
        },
        # -- PL v3 additions: 2D materials / nanostructure tools --
        "pl.substrate_enhancement": {
            "url": valves.pl_url,
            "mcp": "pl_substrate_enhancement",
            "timeout": t,
        },
        "pl.nonlinear_plan": {
            "url": valves.pl_url,
            "mcp": "pl_nonlinear_plan",
            "timeout": t,
        },
        "pl.strain_from_shift": {
            "url": valves.pl_url,
            "mcp": "pl_strain_from_shift",
            "timeout": t,
        },
        "pl.ple_plan": {
            "url": valves.pl_url,
            "mcp": "pl_ple_plan",
            "timeout": t,
        },
        "pl.imaging_plan": {
            "url": valves.pl_url,
            "mcp": "pl_imaging_plan",
            "timeout": t,
        },
        "pl.valley_polarization": {
            "url": valves.pl_url,
            "mcp": "pl_valley_polarization",
            "timeout": t,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers (module-level -- invisible to OWUI)
# ═══════════════════════════════════════════════════════════════════════════


async def _emit(emitter: Optional[Any], msg: str, done: bool = False) -> None:
    if emitter:
        try:
            await emitter(
                {"type": "status", "data": {"description": msg, "done": done}}
            )
        except Exception:
            pass


def _get_user_token(user: Optional[dict] = None) -> Optional[str]:
    return user.get("token") if user else None


def _auth_for(valves: Any, entry: dict, user: Optional[dict] = None) -> Optional[str]:
    if entry.get("auth") == "files":
        return _get_user_token(user) or valves.files_admin_token or None
    return None


def _json_safe(result: str) -> str:
    """Ensure result is valid JSON for OWUI citation parser compatibility."""
    try:
        json.loads(result)
        return result
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"output": result})


async def _call_mcp(valves, url, tool_name, args, timeout, auth=None) -> str:
    try:
        result = await _mcp.call(
            url,
            tool_name,
            args,
            timeout=timeout,
            auth_token=auth,
            retries=valves.max_retries,
        )
        return _json_safe(result)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


# ═══════════════════════════════════════════════════════════════════════════
#  kwargs → args rescue logic
# ═══════════════════════════════════════════════════════════════════════════
#
# Models (especially Gemini) frequently put tool arguments in the wrong
# parameter.  Instead of run(tool=X, args={file_id: Y}) they send:
#   - run(tool=X, kwargs='{"file_id": "Y"}')         ← JSON string in kwargs
#   - run(tool=X, file_id="Y")                       ← bare kwarg
#   - run(args='["file_id_value"]', kwargs='{}')      ← args as JSON list
#   - run(tool=X, kwargs='{},tool:')                  ← garbage
#
# All of these end up with args={} and the real arguments lost in **kwargs.
# This function rescues them.

_OWUI_PRIVATE = frozenset(
    {
        "__user__",
        "__event_emitter__",
        "__files__",
        "__messages__",
        "__metadata__",
        "__request__",
        "__id__",
    }
)


def _rescue_kwargs_into_args(args: dict, kwargs: dict) -> dict:
    """Merge model-misplaced arguments from **kwargs into args dict."""
    extra = {k: v for k, v in kwargs.items() if k not in _OWUI_PRIVATE}
    if not extra:
        return args

    # Case 1: model sent a "kwargs" key containing a JSON string of real args
    kw_val = extra.pop("kwargs", None)
    if isinstance(kw_val, str):
        kw_val = kw_val.strip()
        if kw_val and kw_val not in ("{}", "{},tool:"):
            try:
                parsed = json.loads(kw_val)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        if k not in args:
                            args[k] = v
            except (json.JSONDecodeError, TypeError):
                pass

    # Case 2: model sent an "args" key as a stringified dict in kwargs
    args_val = extra.pop("args", None)
    if isinstance(args_val, str):
        args_val = args_val.strip()
        try:
            parsed = json.loads(args_val)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if k not in args:
                        args[k] = v
        except (json.JSONDecodeError, TypeError):
            pass

    # Case 3: model sent tool-argument keys directly as kwargs
    #   e.g. run(tool="origin.inspect", file_id="abc-123")
    for k, v in extra.items():
        if k not in args and k != "tool":
            args[k] = v

    return args


# ═══════════════════════════════════════════════════════════════════════════
#  Tools class
# ═══════════════════════════════════════════════════════════════════════════


class Tools:
    class Valves(BaseModel):
        micro_url: str = Field(
            default="http://mcp-micro:9006/mcp", description="Microscopy MCP server"
        )
        xrd_url: str = Field(
            default="http://mcp-xrd:9008/mcp",
            description="XRD analysis MCP server (v2, port 9008)",
        )
        origin_url: str = Field(
            default=os.getenv("ORIGIN_MCP_URL", "http://localhost:12009/mcp"),
            description="OriginLab MCP server (Tailscale)",
        )
        files_url: str = Field(
            default="http://mcp-files:9004/mcp", description="Files MCP server"
        )
        pl_url: str = Field(
            default="http://mcp-pl:9010/mcp",
            description="PL experimental planning MCP server",
        )
        files_admin_token: str = Field(
            default="", description="Fallback JWT for files server"
        )
        default_timeout: int = Field(default=120)
        origin_timeout: int = Field(default=300)
        uploads_dir: str = Field(
            default="/app/backend/data/uploads", description="OWUI uploads directory"
        )
        max_retries: int = Field(default=2)
        debug: bool = Field(default=False)

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show tool progress status messages"
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    # -- run: generic tool gateway --

    async def run(
        self,
        tool: Optional[str] = None,
        args: Optional[dict] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """
        Execute any backend tool by name. Use find() to discover available tools and their parameters.
        :param tool: Tool name (e.g. "micro.sem_fft", "xrd.identify", "origin.inspect", "doc.create"). Use find() to list them.
        :param args: Arguments dict matching the tool's parameters.
        """
        if not tool:
            return json.dumps(
                {
                    "error": "Missing required parameter 'tool'. Use find() to list available tools."
                }
            )

        # --- Normalize args ---
        args = args or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                return json.dumps(
                    {"error": f"Invalid args for '{tool}': expected a dict."}
                )
        if isinstance(args, list):
            # Some models send args as a list — not usable as dict, ignore
            args = {}
        if not isinstance(args, dict):
            try:
                args = dict(args)
            except (TypeError, ValueError):
                return json.dumps(
                    {
                        "error": f"Invalid args for '{tool}': expected a dict, got {type(args).__name__}."
                    }
                )

        # --- Rescue misplaced arguments from **kwargs into args ---
        args = _rescue_kwargs_into_args(args, kwargs)

        # --- Resolve namespace aliases (spec.* → origin.*, opj_* → origin.*) ---
        tool = _resolve_tool_alias(tool)

        registry = _build_registry(self.valves)
        if tool not in registry:
            candidates = [k for k in registry if tool.split(".")[-1] in k]
            hint = f" Did you mean: {', '.join(candidates[:5])}?" if candidates else ""
            return json.dumps(
                {"error": f"Unknown tool '{tool}'.{hint} Use find() to list tools."}
            )
        entry = registry[tool]
        log.info(f"[run] {tool} args={json.dumps(args, default=str)[:500]}")
        await _emit(__event_emitter__, f"⚙️ {tool}...")
        result = await _call_mcp(
            self.valves,
            entry["url"],
            entry["mcp"],
            args,
            entry.get("timeout", self.valves.default_timeout),
            _auth_for(self.valves, entry, __user__),
        )
        await _emit(__event_emitter__, f"✅ {tool} done", done=True)
        return result

    # -- find: tool discovery --

    async def find(
        self,
        query: Optional[str] = None,
        server: Optional[str] = None,
        __event_emitter__: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """
        Discover available tools. Returns names, parameters, descriptions.
        :param query: Filter by keyword (e.g. "sem", "peak", "opj", "xrd", "origin"). Leave empty for all.
        :param server: Filter by server -- micro, xrd, origin, files, pl. Leave empty for all.
        """
        await _emit(__event_emitter__, "🔎 Discovering tools...")
        registry = _build_registry(self.valves)

        # Server-alias map: keyword → registry prefix
        _SA = {
            "microscop": "micro.",
            "sem": "micro.",
            "fft": "micro.",
            "xrd": "xrd.",
            "diffract": "xrd.",
            "crystal": "xrd.",
            "origin": "origin.",
            "originlab": "origin.",
            "opj": "origin.",
            "spectro": "origin.",
            "raman": "origin.",
            "spec": "origin.",  # backward compat
            "files": "doc.",
            "file": "doc.",
            "document": "doc.",
            "pl": "pl.",
            "photoluminescence": "pl.",
            "laser": "pl.",
            "filter": "pl.",
            "substrate": "pl.",
            "fresnel": "pl.",
            "shg": "pl.",
            "thg": "pl.",
            "nonlinear": "pl.",
            "strain": "pl.",
            "ple": "pl.",
            "valley": "pl.",
            "polarization": "pl.",
        }

        # Keyword-alias map: keyword → prefix for matching tool names
        _KA = {
            "sem": "micro.",
            "fft": "micro.",
            "microscop": "micro.",
            "period": "micro.",
            "xrd": "xrd.",
            "diffract": "xrd.",
            "crystal": "xrd.",
            "phase": "xrd.",
            "purity": "xrd.",
            "brml": "xrd.",
            "bruker": "xrd.",
            "raw": "xrd.",
            "identify": "xrd.",
            "parse": "xrd.",
            "origin_export": "xrd.",
            "search_ref": "xrd.",
            "origin": "origin.",
            "originlab": "origin.",
            "spectro": "origin.",
            "raman": "origin.",
            "peak": "origin.",
            "fit": "origin.",
            "opj": "origin.",
            "graph": "origin.",
            "merge": "origin.",
            "import_csv": "origin.",
            "import_origin": "origin.",
            "inspect": "origin.",
            "audit": "origin.",
            "batch": "origin.",
            "multi_fit": "origin.",
            "multi": "origin.",
            "decompos": "origin.",
            "trion": "origin.",
            "exciton": "origin.",
            "export": "origin.",
            "edit": "doc.edit",
            "create": "doc.create",
            "read": "doc.read",
            "pl": "pl.",
            "photoluminescence": "pl.",
            "luminescence": "pl.",
            "laser": "pl.",
            "filter": "pl.",
            "notch": "pl.",
            "longpass": "pl.",
            "long_pass": "pl.",
            "dichroic": "pl.",
            "emission": "pl.",
            "bandgap": "pl.",
            "band_gap": "pl.",
            "optics": "pl.",
            "recommend": "pl.",
            "setup": "pl.",
            "substrate": "pl.",
            "enhancement": "pl.",
            "fresnel": "pl.",
            "sio2": "pl.",
            "shg": "pl.",
            "thg": "pl.",
            "nonlinear": "pl.",
            "harmonic": "pl.",
            "strain": "pl.",
            "shift": "pl.",
            "ple": "pl.",
            "excitation": "pl.",
            "imaging": "pl.",
            "widefield": "pl.",
            "confocal": "pl.",
            "hyperspectral": "pl.",
            "valley": "pl.",
            "polarization": "pl.",
            "nanoantenna": "pl.",
            "mie": "pl.",
            "pillar": "pl.",
            "auger": "pl.",
        }

        servers: Dict[str, List[str]] = {}
        for name, entry in registry.items():
            if server:
                sl = server.lower().strip()
                prefix = _SA.get(sl, sl)
                if not name.startswith(prefix) and not name.startswith(sl):
                    continue
            if query:
                ql = query.lower()
                nl = name.lower()
                words = ql.split()
                matched = any(
                    w in nl or (_KA.get(w, "") and _KA[w] in nl) for w in words
                )
                if not matched and ql not in nl:
                    continue
            servers.setdefault(entry["url"], []).append(name)

        lines: List[str] = []
        seen: set = set()
        for url, tool_names in servers.items():
            if url in seen:
                continue
            seen.add(url)
            try:
                raw_tools = await _mcp.list_tools(
                    url, _auth_for(self.valves, registry[tool_names[0]], None)
                )
            except Exception as e:
                lines.append(f"\n❌ {url}: {e}")
                continue
            mcp_to_short = {registry[sn]["mcp"]: sn for sn in tool_names}
            label = tool_names[0].split(".")[0].upper()
            lines.append(f"\n=== {label} ({url}) ===")
            for t in raw_tools:
                mcp_name = t.get("name", "?")
                short = mcp_to_short.get(mcp_name, f"(unmapped:{mcp_name})")
                desc = (t.get("description") or "")[:120]
                schema = t.get("inputSchema", {}).get("properties", {})
                required = set(t.get("inputSchema", {}).get("required", []))
                params = ", ".join(
                    f"{k}{'*' if k in required else ''}: {v.get('type','?')}"
                    for k, v in schema.items()
                )
                lines.append(f"  {short}({params})")
                if desc:
                    lines.append(f"    -> {desc}")
        await _emit(__event_emitter__, "✅ Discovery complete", done=True)
        result_text = (
            "\n".join(lines) if lines else "No tools found matching your query."
        )
        return json.dumps({"type": "tool_discovery", "content": result_text})

    # -- health --

    async def health(self, __event_emitter__: Optional[Any] = None, **kwargs) -> str:
        """Check connectivity to all backend MCP servers."""
        await _emit(__event_emitter__, "🏥 Checking servers...")
        registry = _build_registry(self.valves)
        urls = {}
        for name, entry in registry.items():
            urls[name.split(".")[0]] = entry["url"]
        lines: List[str] = []
        for label, url in sorted(urls.items()):
            try:
                t0 = time.monotonic()
                await _mcp.list_tools(url)
                ms = int((time.monotonic() - t0) * 1000)
                lines.append(f"  ✅ {label:12s}  {ms:>5d}ms  {url}")
            except Exception as e:
                lines.append(f"  ❌ {label:12s}  ERROR   {url}  ({e})")
        await _emit(__event_emitter__, "✅ Health check done", done=True)
        result_text = "Server Health:\n" + "\n".join(lines)
        return json.dumps({"type": "health_check", "content": result_text})

    # -- get_image_path --

    async def get_image_path(
        self,
        file_id: Optional[str] = None,
        __files__: Optional[List[dict]] = None,
        __messages__: Optional[List[dict]] = None,
        __event_emitter__: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """
        Get the file_id and filesystem path of an image from the current chat.
        If file_id is provided, resolves that specific file.
        Otherwise returns the most recently uploaded image in this conversation.
        :param file_id: Optional OWUI file ID to resolve directly.
        """
        if file_id:
            path = _resolve_file_id_to_path(str(file_id), self.valves.uploads_dir)
            if path:
                name = Path(path).name
                await _emit(__event_emitter__, f"🖼️ Found: {name}", done=True)
                return json.dumps({"file_id": str(file_id), "path": path, "name": name})
            await _emit(
                __event_emitter__,
                f"🖼️ file_id={file_id} (no local path, MCP will use HTTP)",
                done=True,
            )
            return json.dumps({"file_id": str(file_id), "path": None, "name": None})

        images = _extract_image_file_ids(__files__, __messages__)
        if not images:
            return json.dumps(
                {
                    "error": "No images found in this conversation. Please upload an image in your message."
                }
            )

        img = images[0]
        fid = img["file_id"]
        fname = img["name"]

        path = _resolve_file_id_to_path(fid, self.valves.uploads_dir)
        if path:
            await _emit(
                __event_emitter__,
                f"🖼️ Found: {fname or Path(path).name} ({img['source']})",
                done=True,
            )
            return json.dumps(
                {"file_id": fid, "path": path, "name": fname or Path(path).name}
            )

        await _emit(
            __event_emitter__,
            f"🖼️ {fname} — file_id={fid} ({img['source']})",
            done=True,
        )
        return json.dumps({"file_id": fid, "path": None, "name": fname})

    # -- create_document (for saving analysis results) --

    async def create_document(
        self,
        format: str,
        filename: str,
        content: list,
        title: Optional[str] = None,
        persistent: Optional[bool] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """
        Create a document to save analysis results (DOCX, PDF, CSV, etc.).
        :param format: File format.
        :param filename: Output filename.
        :param content: Content blocks list or raw string for plain formats.
        :param title: Optional document title.
        :param persistent: Keep file for later download (default true).
        """
        persistent = persistent if persistent is not None else True
        await _emit(__event_emitter__, f"📝 Creating {format.upper()}: {filename}...")
        args: Dict[str, Any] = {
            "format": format,
            "filename": filename,
            "content": content,
            "persistent": persistent,
        }
        if title:
            args["title"] = title
        result = await _call_mcp(
            self.valves,
            self.valves.files_url,
            "create_file",
            args,
            self.valves.default_timeout,
            _get_user_token(__user__) or self.valves.files_admin_token or None,
        )
        await _emit(__event_emitter__, f"✅ {filename} created", done=True)
        return result

    # -- read_document --

    async def read_document(
        self,
        file_id: str,
        file_name: str,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """
        Read the structure of an uploaded document. Call before editing.
        :param file_id: The OWUI file ID from upload.
        :param file_name: Filename with extension.
        """
        await _emit(__event_emitter__, f"📖 Reading: {file_name}...")
        result = await _call_mcp(
            self.valves,
            self.valves.files_url,
            "full_context_document",
            {"file_id": file_id, "file_name": file_name},
            self.valves.default_timeout,
            _get_user_token(__user__) or self.valves.files_admin_token or None,
        )
        await _emit(__event_emitter__, "✅ Document read complete", done=True)
        return result

    # -- fetch_url --

    async def fetch_url(
        self,
        url: Optional[str] = None,
        __event_emitter__: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """
        Fetch and extract text content from a web URL.
        :param url: The URL to fetch (must be http/https, no internal IPs).
        """
        if not url:
            return json.dumps({"error": "Missing required parameter 'url'."})
        if not _is_url_safe(url):
            return json.dumps({"error": f"URL blocked by security policy: {url}"})
        await _emit(__event_emitter__, f"🌐 Fetching: {url[:80]}...")
        try:
            client = _mcp._get_client()
            resp = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                timeout=45.0,
                follow_redirects=True,
            )
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "text/html" in ct:
                text = resp.text
                text = re.sub(
                    r"<script[^>]*>.*?</script>",
                    "",
                    text,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                text = re.sub(
                    r"<style[^>]*>.*?</style>",
                    "",
                    text,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) > 30000:
                    text = text[:30000] + "\n\n[... truncated]"
                result = text
            else:
                result = resp.text[:30000]
            await _emit(__event_emitter__, "✅ Fetched", done=True)
            return _json_safe(result)
        except Exception as e:
            return json.dumps({"error": f"Failed to fetch {url}: {e}"})
