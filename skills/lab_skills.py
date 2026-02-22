"""
title: FORTHought Lab Skills
description: Tool gateway for lab model profiles. Academic paper search (8 sources with author ID resolution) and download,
    document creation and editing, presentations, and web fetch. Attach Web Search,
    Jupyter Code Tool, and Chemistry DB separately in model settings.
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

log = logging.getLogger("forthought.skills.lab")

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
#  Registry — Lab profile: papers, files, presenton
# ═══════════════════════════════════════════════════════════════════════════


def _build_registry(valves: Any) -> Dict[str, dict]:
    t = valves.default_timeout
    return {
        # -- papers --
        "papers.search": {
            "url": valves.papers_url,
            "mcp": "search",
            "timeout": valves.papers_timeout,
        },
        "papers.details": {
            "url": valves.papers_url,
            "mcp": "details",
            "timeout": valves.papers_timeout,
        },
        "papers.download": {
            "url": valves.papers_url,
            "mcp": "download",
            "timeout": valves.papers_timeout,
        },
        "papers.batch_details": {
            "url": valves.papers_url,
            "mcp": "batch_details",
            "timeout": valves.papers_timeout,
        },
        "papers.batch_download": {
            "url": valves.papers_url,
            "mcp": "batch_download",
            "timeout": valves.papers_timeout,
        },
        "papers.resolve": {
            "url": valves.papers_url,
            "mcp": "resolve_references",
            "timeout": valves.papers_timeout,
        },
        "papers.books": {
            "url": valves.papers_url,
            "mcp": "books",
            "timeout": valves.papers_timeout,
        },
        "papers.book_download": {
            "url": valves.papers_url,
            "mcp": "book_download",
            "timeout": valves.papers_timeout,
        },
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


async def _call_mcp(
    valves, url, tool_name, args, timeout, auth=None, max_chars: int = 8000
) -> str:
    """Call an MCP tool and return JSON-safe result, truncated if needed.

    Args:
        max_chars: Maximum response length. If exceeded, returns a clean
                   truncation JSON instead of raw-chopping the response.
                   Default 8000. Paper detail tools should use 4000.
    """
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
        # Truncation safety net — prevent oversized responses from blowing context
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


class Tools:
    class Valves(BaseModel):
        papers_url: str = Field(
            default="http://mcp-papers:9005/mcp", description="Papers MCP server"
        )
        files_url: str = Field(
            default="http://mcp-files:9004/mcp", description="Files MCP server"
        )
        presenton_url: str = Field(
            default="http://presenton:80/mcp", description="Presenton PPTX MCP server"
        )
        files_admin_token: str = Field(
            default="", description="Fallback JWT for files server"
        )
        default_timeout: int = Field(default=120)
        papers_timeout: int = Field(default=180)
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
        :param tool: Tool name (e.g. "papers.search", "doc.create", "pptx.generate"). Use find() to list them.
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
        # Tight cap for large-payload paper tools
        _TIGHT = {"batch_details", "details", "books", "book_download"}
        cap = 4000 if entry["mcp"] in _TIGHT else 8000
        result = await _call_mcp(
            self.valves,
            entry["url"],
            entry["mcp"],
            args,
            entry.get("timeout", self.valves.default_timeout),
            _auth_for(self.valves, entry, __user__),
            max_chars=cap,
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
        :param query: Filter by keyword (e.g. "paper", "docx", "pptx"). Leave empty for all.
        :param server: Filter by server -- papers, files, pptx. Leave empty for all.
        """
        await _emit(__event_emitter__, "🔎 Discovering tools...")
        registry = _build_registry(self.valves)
        _SA = {
            "files": "doc.",
            "file": "doc.",
            "document": "doc.",
            "docs": "doc.",
            "paper": "papers.",
            "lit": "papers.",
            "academic": "papers.",
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
            "paper": "papers.",
            "literature": "papers.",
            "arxiv": "papers.",
            "search": "papers.search",
            "download": "papers.download",
            "resolve": "papers.resolve",
            "reference": "papers.resolve",
            "cite": "papers.resolve",
            "citation": "papers.resolve",
            "doi": "papers.resolve",
            "book": "papers.books",
            "textbook": "papers.books",
            "presentation": "pptx.",
            "pptx": "pptx.",
            "slides": "pptx.",
            "generate": "pptx.generate",
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
                raw_tools = await _mcp.list_tools(
                    url, _auth_for(self.valves, registry[tool_names[0]], None)
                )
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

    # -- search_papers --

    async def search_papers(
        self,
        query: Optional[str] = None,
        max_results: Optional[int] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        sort: Optional[str] = None,
        mode: Optional[str] = None,
        author: Optional[str] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """
        Search academic papers across arXiv, OpenAlex, NASA ADS, Semantic Scholar, PubMed, CrossRef, OpenAIRE, and ORCID.
        When an author name is provided, the server resolves it to canonical IDs (OpenAlex, S2, ORCID)
        then uses ID-based endpoints for exhaustive coverage — finds 5-10x more papers than name search.
        :param query: Search terms, boolean (AND/OR/NOT), or natural-language description. Optional if author is set.
        :param max_results: Maximum results to return (default 15, max 100). Server over-fetches and ranks.
        :param year_min: Only papers published on or after this year (e.g. 2022).
        :param year_max: Only papers published on or before this year.
        :param sort: Ranking — "relevance" (default), "date" (newest first), "citations".
        :param mode: Strategy — "auto" (default), "semantic" (AI embeddings), "keyword" (boolean/exact).
        :param author: Author name to filter by (e.g. "Emmanuel Stratakis"). Uses ID-based resolution for exhaustive coverage. For comprehensive author results, use max_results=50.
        """
        if not query and not author:
            return json.dumps({"error": "Provide at least 'query' or 'author'."})
        label = f"🔍 Searching: {query[:60] if query else f'author={author}'}"
        await _emit(__event_emitter__, label + "...")
        args = {"query": query or "", "max_results": max_results or 15}
        if year_min is not None:
            args["year_min"] = year_min
        if year_max is not None:
            args["year_max"] = year_max
        if sort is not None:
            args["sort"] = sort
        if mode is not None:
            args["mode"] = mode
        if author is not None:
            args["author"] = author
        result = await _call_mcp(
            self.valves,
            self.valves.papers_url,
            "search",
            args,
            self.valves.papers_timeout,
        )
        await _emit(__event_emitter__, "✅ Search complete", done=True)
        return result

    # -- download_papers --

    async def download_papers(
        self,
        papers: Optional[list] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """
        Download paper PDFs. Any number of papers supported — auto-batched and zipped.
        Tries arXiv, Unpaywall, then Sci-Hub.
        :param papers: List of dicts with "doi" and/or "arxiv_id". Example: [{"doi": "10.1038/..."}, {"arxiv_id": "2301.12345"}]
        """
        # Models sometimes pass papers as a JSON string instead of a list
        if isinstance(papers, str):
            try:
                papers = json.loads(papers)
            except (json.JSONDecodeError, TypeError):
                return json.dumps(
                    {"error": "papers must be a list of dicts, got unparseable string."}
                )
        if not papers:
            return json.dumps(
                {
                    "error": "papers list is empty -- provide at least one {'doi': '...'} or {'arxiv_id': '...'}"
                }
            )
        count = len(papers)
        if count == 1:
            p = papers[0]
            await _emit(__event_emitter__, "⬇️ Downloading paper...")
            args: Dict[str, Any] = {}
            if p.get("doi"):
                args["doi"] = p["doi"]
            if p.get("arxiv_id"):
                args["arxiv_id"] = p["arxiv_id"]
            if not args:
                return json.dumps(
                    {"error": "Each paper needs a 'doi' or 'arxiv_id' key"}
                )
            result = await _call_mcp(
                self.valves,
                self.valves.papers_url,
                "download",
                args,
                self.valves.papers_timeout,
            )
            await _emit(__event_emitter__, "✅ Download complete", done=True)
            return result
        else:
            await _emit(__event_emitter__, f"⬇️ Downloading {count} papers...")
            result = await _call_mcp(
                self.valves,
                self.valves.papers_url,
                "batch_download",
                {"papers": papers},
                self.valves.papers_timeout,
            )
            await _emit(__event_emitter__, f"✅ {count} downloads complete", done=True)
            return result

    # -- resolve_references --

    async def resolve_references(
        self,
        references: Optional[list] = None,
        text: Optional[str] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """
        Resolve raw citation strings or DOIs to structured paper identifiers via CrossRef.
        Call this when the user provides a reference list, bibliography, or citation strings
        that need to be identified before downloading. Always use this tool directly --
        do NOT route through run().
        :param references: List of citation strings or DOIs. Example: ["Bonse et al., J Laser Appl 2012...", "10.1103/PhysRevB.73.035439"]
        :param text: Alternative: paste all references as a single string, one per line. The tool will split them automatically.
        """
        # ── Normalize input: ensure references is a list of strings ──
        # Models (especially Gemini) often pass the entire text block as
        # `references` (a string) instead of a list. Detect and fix.
        if isinstance(references, str):
            text = references  # redirect to text-splitting path
            references = None
        if not references and text:
            # Split on newlines, strip numbering prefixes like "1.", "1)", "[1]", "- "
            lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
            references = [
                re.sub(r"^\s*[\[\(]?\d+[\]\)\.:\-]\s*", "", l).strip() for l in lines
            ]
            references = [r for r in references if len(r) > 5]  # drop junk lines
        if not references:
            return json.dumps(
                {
                    "error": "No references provided. Pass either 'references' (list) or 'text' (one reference per line)."
                }
            )
        # Papers v5.1 handles up to 50 refs in one call with its own
        # semaphore-based concurrency — no need to batch in skills.
        count = len(references)
        await _emit(
            __event_emitter__,
            f"🔍 Resolving {count} reference{'s' if count != 1 else ''}...",
        )
        result = await _call_mcp(
            self.valves,
            self.valves.papers_url,
            "resolve_references",
            {"references": references},
            self.valves.papers_timeout,
        )
        await _emit(__event_emitter__, "✅ Resolution complete", done=True)
        return result

    # -- create_document --

    async def create_document(
        self,
        format: Optional[str] = None,
        filename: Optional[str] = None,
        content: Optional[list] = None,
        title: Optional[str] = None,
        persistent: Optional[bool] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """
        Create a document (DOCX, PDF, XLSX, CSV, PPTX, HTML, TXT, MD, PY, JSON, etc.).
        :param format: File format (docx, pdf, xlsx, csv, pptx, html, txt, md, py, json).
        :param filename: Output filename (e.g. "report.docx").
        :param content: Content blocks list or raw string for plain formats.
        :param title: Optional document title.
        :param persistent: Keep file for later download (default true).
        """
        missing = [
            p
            for p, v in [
                ("format", format),
                ("filename", filename),
                ("content", content),
            ]
            if not v
        ]
        if missing:
            return json.dumps(
                {
                    "error": f"Missing required parameters: {', '.join(missing)}. All three (format, filename, content) are required."
                }
            )
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
        file_id: Optional[str] = None,
        file_name: Optional[str] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """
        Read the full structure of a document (DOCX, XLSX, PPTX). Call this BEFORE editing.
        :param file_id: The OWUI file ID from upload.
        :param file_name: Filename with extension (e.g. "report.docx").
        """
        if not file_id or not file_name:
            missing = [
                p for p, v in [("file_id", file_id), ("file_name", file_name)] if not v
            ]
            return json.dumps(
                {"error": f"Missing required parameters: {', '.join(missing)}"}
            )
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
