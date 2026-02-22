"""
title: FORTHought Coder Skills
description: Tool gateway for coder model profiles. Library documentation via Context7,
    document creation and editing, presentations, and web fetch. Attach Web Search and
    Jupyter Code Tool separately in model settings.
author: Marios Adamidis (FORTHought Lab)
version: 1.0.0
required_open_webui_version: 0.6.6
"""

# ═══════════════════════════════════════════════════════════════════════════
#  Imports
# ═══════════════════════════════════════════════════════════════════════════

import os, re, json, time, asyncio, logging, ipaddress
from typing import Optional, Any, Dict, List
from urllib.parse import urlparse

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from pydantic import BaseModel, Field

log = logging.getLogger("forthought.skills.coder")

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
                    "clientInfo": {"name": "forthought-skills", "version": "1.0.0"},
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
_ALLOWED_INTERNAL = set(os.getenv("ALLOWED_INTERNAL_IPS", "").split(",")) if os.getenv("ALLOWED_INTERNAL_IPS") else set()


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
#  Registry — Coder profile: files, context7, presenton
# ═══════════════════════════════════════════════════════════════════════════


def _build_registry(valves: Any) -> Dict[str, dict]:
    t = valves.default_timeout
    return {
        # -- files --
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
        "doc.review": {
            "url": valves.files_url,
            "mcp": "review_document",
            "timeout": t,
            "auth": "files",
        },
        "doc.archive": {
            "url": valves.files_url,
            "mcp": "generate_and_archive",
            "timeout": t,
            "auth": "files",
        },
        # -- library docs (context7) --
        "lib.resolve": {
            "url": valves.context7_url,
            "mcp": "resolve-library-uri",
            "timeout": 60,
        },
        "lib.docs": {
            "url": valves.context7_url,
            "mcp": "search-library-docs",
            "timeout": 60,
        },
        # -- presenton --
        "pptx.generate": {
            "url": valves.presenton_url,
            "mcp": "generate_presentation",
            "timeout": t,
        },
        "pptx.templates": {
            "url": valves.presenton_url,
            "mcp": "templates_list",
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
    """Ensure result is valid JSON for OWUI citation parser compatibility.
    The OR pipe's get_citation_source_from_tool_result does json.loads()
    on every tool return. Non-JSON strings crash it and dump raw text to chat.
    """
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
#  Tools class
# ═══════════════════════════════════════════════════════════════════════════


class Tools:
    class Valves(BaseModel):
        files_url: str = Field(
            default="http://mcp-files:9004/mcp", description="Files MCP server"
        )
        presenton_url: str = Field(
            default="http://presenton:80/mcp", description="Presenton PPTX MCP server"
        )
        context7_url: str = Field(
            default="https://context7.liam.sh/mcp",
            description="Context7 library docs (Streamable HTTP)",
        )
        files_admin_token: str = Field(
            default="", description="Fallback JWT for files server"
        )
        default_timeout: int = Field(default=120)
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
        :param tool: Tool name (e.g. "doc.create", "doc.edit", "lib.resolve"). Use find() to list them.
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
        :param query: Filter by keyword (e.g. "edit", "library", "pptx"). Leave empty for all.
        :param server: Filter by server -- files, lib, pptx. Leave empty for all.
        """
        await _emit(__event_emitter__, "🔎 Discovering tools...")
        registry = _build_registry(self.valves)
        _SA = {
            "files": "doc.",
            "file": "doc.",
            "document": "doc.",
            "docs": "doc.",
            "library": "lib.",
            "context7": "lib.",
            "c7": "lib.",
            "presentation": "pptx.",
            "slide": "pptx.",
            "presenton": "pptx.",
        }
        _KA = {
            "edit": "doc.edit",
            "replace": "doc.edit",
            "modify": "doc.edit",
            "read": "doc.read",
            "create": "doc.create",
            "write": "doc.create",
            "review": "doc.review",
            "archive": "doc.archive",
            "document": "doc.",
            "docx": "doc.",
            "xlsx": "doc.",
            "pdf": "doc.",
            "library": "lib.",
            "docs": "lib.",
            "documentation": "lib.",
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
        # Wrap in JSON for OWUI citation parser compatibility
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
        # Wrap in JSON for OWUI citation parser compatibility
        return json.dumps({"type": "health_check", "content": result_text})

    # -- lib_lookup --

    async def lib_lookup(
        self,
        library_name: str,
        topic: Optional[str] = None,
        tokens: Optional[int] = None,
        __event_emitter__: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """
        Look up library/framework documentation via Context7. Resolves automatically then fetches docs.
        :param library_name: Package name (e.g. "react", "fastapi", "numpy").
        :param topic: Focus topic (e.g. "hooks", "routing"). Optional.
        :param tokens: Max doc tokens to retrieve (default 5000).
        """
        tokens = tokens or 5000
        ctx7_url = self.valves.context7_url
        await _emit(__event_emitter__, f"📚 Resolving {library_name}...")
        resolve_result = await _call_mcp(
            self.valves,
            ctx7_url,
            "resolve-library-uri",
            {"libraryName": library_name},
            60,
        )

        lib_uri = None
        if isinstance(resolve_result, str):
            try:
                data = json.loads(resolve_result)
                if isinstance(data, dict):
                    lib_uri = (
                        data.get("resourceURI")
                        or data.get("libraryID")
                        or data.get("uri")
                    )
                elif isinstance(data, list) and data:
                    lib_uri = data[0].get("resourceURI") or data[0].get("libraryID")
            except (json.JSONDecodeError, TypeError, IndexError):
                pass
            if not lib_uri:
                uris = re.findall(r"context7://libraries/\S+", resolve_result)
                if uris:
                    name_lower = library_name.lower().replace("-", "").replace("_", "")
                    exact = partial = None
                    for u in uris:
                        slug = (
                            u.split("/")[-1].lower().replace("-", "").replace("_", "")
                        )
                        if slug == name_lower:
                            exact = u
                            break
                        if not partial and name_lower in slug:
                            partial = u
                    lib_uri = exact or partial or uris[0]

        if not lib_uri:
            return _json_safe(
                f"Resolve result (extract the resourceURI and call run('lib.docs', {{...}}) manually):\n{resolve_result}"
            )

        await _emit(__event_emitter__, f"📖 Fetching docs for {lib_uri}...")
        args: Dict[str, Any] = {"resourceURI": lib_uri, "tokens": tokens}
        if topic:
            args["topic"] = topic
        result = await _call_mcp(self.valves, ctx7_url, "search-library-docs", args, 60)
        await _emit(__event_emitter__, "✅ Docs retrieved", done=True)
        return result

    # -- create_document --

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
        Create a document (DOCX, PDF, XLSX, CSV, PPTX, HTML, TXT, MD, PY, JSON, etc.).
        :param format: File format.
        :param filename: Output filename (e.g. "solver.py").
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
        Read the full structure of a document (DOCX, XLSX, PPTX). Call this BEFORE editing.
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
