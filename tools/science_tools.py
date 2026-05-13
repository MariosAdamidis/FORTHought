"""
title: FORTHought Science Tools
description: Instrument and experimental data gateway for lab models.
    Microscopy (SEM FFT), XRD (phase ID, purity, 5 tools),
    OriginLab data engine (22 tools), PL experimental planning (11 tools),
    vibrational spectroscopy (FT-IR/ATR + Raman, 4 tools).
    Chat-scoped image discovery for analyzing uploaded microscopy/spectral images.
    Lab models get this tool — it handles all instrument operations.
author: Marios Adamidis
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

log = logging.getLogger("forthought.tools.science")

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
                limits=httpx.Limits(max_connections=30, max_keepalive_connections=15),
                timeout=httpx.Timeout(300.0, connect=15.0),
                follow_redirects=True,
            )
        return self._client

    async def _init_session(self, url: str, headers: Dict[str, str]) -> str:
        client = self._get_client()
        resp = await client.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "forthought-core", "version": "1.0.0"},
                },
            },
            headers=headers,
            timeout=30.0,
        )
        resp.raise_for_status()
        sid = resp.headers.get("mcp-session-id") or ""
        nh = {**headers}
        if sid:
            nh["mcp-session-id"] = sid
        try:
            await client.post(
                url,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                headers=nh,
                timeout=10.0,
            )
        except Exception:
            pass
        self._sessions[url] = sid
        self._init_ts[url] = time.monotonic()
        return sid

    @staticmethod
    def _parse_response(resp: Any) -> dict:
        ct = resp.headers.get("content-type", "")
        if "text/event-stream" in ct:
            last_data = None
            for line in resp.text.split("\n"):
                if line.startswith("data: "):
                    try:
                        last_data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
            if last_data:
                return last_data
            raise ValueError("No valid JSON in SSE stream")
        return resp.json()

    @staticmethod
    def _extract_text(result: dict) -> str:
        content = result.get("content", [])
        if isinstance(content, str):
            return content
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    v = block.get("text", "")
                    parts.append(
                        v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    )
                elif btype == "image":
                    parts.append(f"[image: {block.get('mimeType', 'unknown')}]")
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
        return "\n".join(parts) if parts else json.dumps(result, ensure_ascii=False)

    async def call(
        self,
        url: str,
        tool_name: str,
        arguments: dict,
        timeout: float = 120.0,
        auth_token: Optional[str] = None,
        retries: int = 2,
    ) -> str:
        client = self._get_client()
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if auth_token:
            headers["Authorization"] = (
                auth_token
                if auth_token.lower().startswith("bearer ")
                else f"Bearer {auth_token}"
            )
        for attempt in range(retries + 1):
            try:
                if url not in self._sessions:
                    await self._init_session(url, headers)
                sid = self._sessions.get(url, "")
                ch = {**headers}
                if sid:
                    ch["mcp-session-id"] = sid
                resp = await client.post(
                    url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": tool_name, "arguments": arguments},
                    },
                    headers=ch,
                    timeout=timeout,
                )
                if resp.status_code in (404, 410):
                    self._sessions.pop(url, None)
                    if attempt < retries:
                        continue
                    resp.raise_for_status()
                resp.raise_for_status()
                data = self._parse_response(resp)
                if "error" in data:
                    msg = data["error"].get("message", str(data["error"]))
                    if any(
                        w in msg.lower()
                        for w in ("session", "expired", "invalid session")
                    ):
                        self._sessions.pop(url, None)
                        if attempt < retries:
                            continue
                    raise RuntimeError(f"MCP error from {tool_name}: {msg}")
                result = data.get("result", data)
                if result.get("isError"):
                    raise RuntimeError(
                        f"Tool {tool_name} error: {self._extract_text(result)}"
                    )
                return self._extract_text(result)
            except (RuntimeError, ValueError):
                raise
            except Exception as e:
                self._sessions.pop(url, None)
                if attempt < retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"{type(e).__name__} calling {tool_name}@{url}: {e}"
                ) from e

    async def list_tools(
        self, url: str, auth_token: Optional[str] = None
    ) -> List[dict]:
        client = self._get_client()
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if auth_token:
            headers["Authorization"] = (
                auth_token
                if auth_token.lower().startswith("bearer ")
                else f"Bearer {auth_token}"
            )
        if url not in self._sessions:
            await self._init_session(url, headers)
        sid = self._sessions.get(url, "")
        ch = {**headers}
        if sid:
            ch["mcp-session-id"] = sid
        resp = await client.post(
            url,
            json={"jsonrpc": "2.0", "id": 99, "method": "tools/list", "params": {}},
            headers=ch,
            timeout=30.0,
        )
        data = self._parse_response(resp)
        return data.get("result", {}).get("tools", [])


_mcp = _MCPClient()

# ═══════════════════════════════════════════════════════════════════════════
#  SSRF Guard
# ═══════════════════════════════════════════════════════════════════════════

_BLOCKED_NETS = [
    ipaddress.ip_network(n)
    for n in [
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    ]
]
_ALLOWED_INTERNAL = {"127.0.0.1"}  # Set to your internal IP


def _is_url_safe(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname or ""
        if host in _ALLOWED_INTERNAL:
            return True
        try:
            return not any(ipaddress.ip_address(host) in net for net in _BLOCKED_NETS)
        except ValueError:
            return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  Image helpers (chat-scoped image discovery)
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
    # v11.0 new tools
    "opj_batch_merge": "origin.batch_merge",
    "opj_subtract": "origin.subtract",
    "opj_despike": "origin.despike",
    "opj_sheet_math": "origin.sheet_math",
    "opj_power_fit": "origin.power_fit",
    "opj_baseline": "origin.baseline",
    "spec.batch_merge": "origin.batch_merge",
    "spec.subtract": "origin.subtract",
    "spec.despike": "origin.despike",
    "spec.sheet_math": "origin.sheet_math",
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
#  Registry — Instrument profile: micro, xrd, origin, pl, vibrational, files

# ═══════════════════════════════════════════════════════════════════════════
#  Registry — Science profile: microscopy, XRD, OriginLab, PL, vibrational
#  (NO document/paper/presentation servers — those are in core_tools/research_tools)
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
        # v11.0 new tools
        "origin.batch_merge": {
            "url": valves.origin_url,
            "mcp": "opj_batch_merge",
            "timeout": valves.origin_timeout + 60,  # processes multiple folders
        },
        "origin.subtract": {
            "url": valves.origin_url,
            "mcp": "opj_subtract",
            "timeout": valves.origin_timeout,
        },
        "origin.despike": {
            "url": valves.origin_url,
            "mcp": "opj_despike",
            "timeout": valves.origin_timeout,
        },
        "origin.sheet_math": {
            "url": valves.origin_url,
            "mcp": "opj_sheet_math",
            "timeout": valves.origin_timeout,
        },
        # v12.0 new tools
        "origin.power_fit": {
            "url": valves.origin_url,
            "mcp": "opj_power_fit",
            "timeout": valves.origin_timeout,
        },
        "origin.baseline": {
            "url": valves.origin_url,
            "mcp": "opj_baseline",
            "timeout": valves.origin_timeout,
        },
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
        # -- vibrational spectroscopy (FT-IR/ATR + Raman, v1) --
        "vibrational.identify": {
            "url": valves.vibrational_url,
            "mcp": "identify",
            "timeout": t,
        },
        "vibrational.compare": {
            "url": valves.vibrational_url,
            "mcp": "compare",
            "timeout": t,
        },
        "vibrational.band_search": {
            "url": valves.vibrational_url,
            "mcp": "band_search",
            "timeout": t,
        },
        "vibrational.material_reference": {
            "url": valves.vibrational_url,
            "mcp": "material_reference",
            "timeout": t,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers (module-level — invisible to OWUI)
# ═══════════════════════════════════════════════════════════════════════════


async def _emit(emitter: Optional[Any], msg: str, done: bool = False) -> None:
    if emitter:
        try:
            await emitter(
                {"type": "status", "data": {"description": msg, "done": done}}
            )
        except Exception:
            pass


async def _emit_origin_markdown(emitter: Optional[Any], result: str) -> None:
    """Emit Origin MCP plot markdown as a chat message event.

    Origin MCP returns embed_markdown in JSON; without this helper, image
    delivery to the chat depended on Gemini copying the markdown verbatim
    into its final answer (S324 audit, fix deployed S326).
    """
    if not emitter:
        return
    try:
        parsed = json.loads(result)
        if not isinstance(parsed, dict):
            return
        md = parsed.get("embed_markdown")
        if not isinstance(md, str) or "![" not in md:
            return
        await emitter({"type": "message", "data": {"content": md}})
    except Exception:
        pass


def _mint_user_jwt(user: Optional[dict] = None) -> Optional[str]:
    if not user or not user.get("id"):
        return None
    try:
        from open_webui.utils.auth import create_token

        return create_token(data={"id": user["id"]})
    except Exception:
        return None


def _auth_for(valves: Any, entry: dict, user: Optional[dict] = None) -> Optional[str]:
    if entry.get("auth") == "files":
        return _mint_user_jwt(user) or valves.files_admin_api_key or None
    return None


def _json_safe(result: str) -> str:
    try:
        json.loads(result)
        return result
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"output": result})


async def _call_mcp(
    valves, url, tool_name, args, timeout, auth=None, max_chars: int = 8000
) -> str:
    try:
        result = await _mcp.call(
            url,
            tool_name,
            args,
            timeout=timeout,
            auth_token=auth,
            retries=valves.max_retries,
        )
        result = _json_safe(result)
        if len(result) > max_chars:
            try:
                parsed = json.loads(result)
                return json.dumps(
                    {
                        "truncated": True,
                        "original_chars": len(result),
                        "showing_chars": max_chars,
                        "note": f"Response too large ({len(result)} chars). Use more specific queries or compact=true.",
                        "data": json.dumps(parsed, ensure_ascii=False)[
                            : max_chars - 200
                        ],
                    },
                    ensure_ascii=False,
                )
            except (json.JSONDecodeError, TypeError):
                return json.dumps(
                    {
                        "truncated": True,
                        "original_chars": len(result),
                        "data": result[: max_chars - 200],
                    }
                )
        return result
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


# ═══════════════════════════════════════════════════════════════════════════
#  Tools class
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
#  Tools class
# ═══════════════════════════════════════════════════════════════════════════


class Tools:
    class Valves(BaseModel):
        micro_url: str = Field(
            default="http://localhost:9006/mcp", description="Microscopy MCP server"
        )
        xrd_url: str = Field(
            default="http://localhost:9008/mcp",
            description="XRD analysis MCP server (v2, port 9008)",
        )
        origin_url: str = Field(
            default="http://localhost:12009/mcp",
            description="OriginLab MCP server endpoint",
        )
        pl_url: str = Field(
            default="http://localhost:9010/mcp",
            description="PL experimental planning MCP server",
        )
        vibrational_url: str = Field(
            default="http://localhost:9012/mcp",
            description="Vibrational spectroscopy (FT-IR/ATR + Raman) MCP server",
        )
        uploads_dir: str = Field(
            default="/app/backend/data/uploads", description="OWUI uploads directory"
        )
        default_timeout: int = Field(default=120)
        origin_timeout: int = Field(default=300)
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
        :param tool: Tool name (e.g. "xrd.identify", "pl.recommend", "origin.fit_peak"). Use find() to list them.
        :param args: Arguments dict matching the tool's parameters.
        """
        if not tool:
            return json.dumps(
                {
                    "error": "Missing required parameter 'tool'. Use find() to list available tools."
                }
            )
        args = args or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                return json.dumps(
                    {"error": f"Invalid args for '{tool}': expected a dict."}
                )
        if not isinstance(args, dict):
            try:
                args = dict(args)
            except (TypeError, ValueError):
                return json.dumps(
                    {
                        "error": f"Invalid args for '{tool}': expected a dict, got {type(args).__name__}."
                    }
                )
        registry = _build_registry(self.valves)
        if tool not in registry:
            candidates = [k for k in registry if tool.split(".")[-1] in k]
            hint = f" Did you mean: {', '.join(candidates[:5])}?" if candidates else ""
            return json.dumps(
                {"error": f"Unknown tool '{tool}'.{hint} Use find() to list tools."}
            )
        entry = registry[tool]
        await _emit(__event_emitter__, f"⚙️ {tool}...")
        _TIGHT = {"batch_details", "details", "books", "book_download"}
        cap = 4000 if entry["mcp"] in _TIGHT else 8000
        result = await _call_mcp(
            self.valves,
            entry["url"],
            entry["mcp"],
            args,
            entry.get("timeout", self.valves.default_timeout),
            max_chars=cap,
        )
        if tool.startswith("origin."):
            await _emit_origin_markdown(__event_emitter__, result)
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
        :param query: Filter by keyword (e.g. "sem", "peak", "opj", "xrd", "origin", "pl", "raman"). Leave empty for all.
        :param server: Filter by server -- micro, xrd, origin, pl, vibrational. Leave empty for all.
        """
        await _emit(__event_emitter__, "🔎 Discovering tools...")
        registry = _build_registry(self.valves)
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
            "spec": "origin.",
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
            "vibrational": "vibrational.",
            "raman": "vibrational.",
            "ftir": "vibrational.",
            "atr": "vibrational.",
            "ir": "vibrational.",
        }
        _KA = {
            "sem": "micro.",
            "fft": "micro.",
            "microscop": "micro.",
            "xrd": "xrd.",
            "phase": "xrd.",
            "diffract": "xrd.",
            "origin": "origin.",
            "opj": "origin.",
            "peak": "origin.",
            "fit": "origin.",
            "normalize": "origin.",
            "smooth": "origin.",
            "csv": "origin.",
            "graph": "origin.",
            "plot": "origin.",
            "trend": "origin.",
            "merge": "origin.",
            "baseline": "origin.",
            "subtract": "origin.",
            "pl": "pl.",
            "photoluminescence": "pl.",
            "laser": "pl.",
            "filter": "pl.",
            "substrate": "pl.",
            "enhancement": "pl.",
            "recommend": "pl.",
            "setup": "pl.",
            "spectrum": "pl.",
            "raman": "vibrational.",
            "ftir": "vibrational.",
            "atr": "vibrational.",
            "vibrational": "vibrational.",
            "band": "vibrational.",
            "identify": "vibrational.",
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

        tools_found: List[dict] = []
        seen: set = set()
        for url, tool_names in servers.items():
            if url in seen:
                continue
            seen.add(url)
            try:
                raw_tools = await _mcp.list_tools(url)
            except Exception as e:
                tools_found.append({"server": url, "error": str(e)[:200]})
                continue
            mcp_to_short = {registry[sn]["mcp"]: sn for sn in tool_names}
            for t in raw_tools:
                mcp_name = t.get("name", "?")
                short = mcp_to_short.get(mcp_name, f"(unmapped:{mcp_name})")
                desc = (t.get("description") or "")[:120]
                schema = t.get("inputSchema", {}).get("properties", {})
                required = set(t.get("inputSchema", {}).get("required", []))
                params = {
                    k: {"type": v.get("type", "?"), "required": k in required}
                    for k, v in schema.items()
                }
                tools_found.append(
                    {
                        "name": short,
                        "description": desc,
                        "parameters": params,
                    }
                )
        await _emit(__event_emitter__, "✅ Discovery complete", done=True)
        if not tools_found:
            return json.dumps(
                {
                    "type": "tool_discovery",
                    "tools": [],
                    "message": "No tools found matching your query.",
                }
            )
        return json.dumps({"type": "tool_discovery", "tools": tools_found})

    # -- health --

    async def health(self, __event_emitter__: Optional[Any] = None, **kwargs) -> str:
        """Check connectivity to all backend MCP servers."""
        await _emit(__event_emitter__, "🏥 Checking servers...")
        registry = _build_registry(self.valves)
        urls = {}
        for name, entry in registry.items():
            urls[name.split(".")[0]] = entry["url"]
        results: List[dict] = []
        for label, url in sorted(urls.items()):
            try:
                t0 = time.monotonic()
                await _mcp.list_tools(url)
                ms = int((time.monotonic() - t0) * 1000)
                results.append({"name": label, "status": "ok", "ms": ms})
            except Exception as e:
                results.append(
                    {"name": label, "status": "error", "error": str(e)[:200]}
                )
        await _emit(__event_emitter__, "✅ Health check done", done=True)
        return json.dumps({"type": "health_check", "servers": results})

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
