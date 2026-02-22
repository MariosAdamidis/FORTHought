"""
title: Lab Search (LangSearch)
author: Marios Adamidis (FORTHought Lab)
description: Multi-query web search with quality filtering, reranking, and auto-fetch. Returns fewer, richer results instead of a noisy dump.
version: 1.0.0
"""

import re
import json
import logging
import asyncio
import aiohttp
from datetime import date
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field

from open_webui.env import BASE_DIR

# ═══════════════════════════════════════════════════════════════════════════
#  Setup
# ═══════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_API_SEM = asyncio.Semaphore(3)

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


@dataclass
class Document:
    page_content: str
    metadata: Dict
    relevance_score: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════
#  Internal helpers (module-level — OWUI doesn't expose these)
# ═══════════════════════════════════════════════════════════════════════════


async def _emit_status(emitter, msg, done):
    if emitter:
        await emitter({"type": "status", "data": {"description": msg, "done": done}})


async def _emit_message(emitter, content):
    if emitter:
        await emitter({"type": "message", "data": {"content": content}})


# ── LangSearch API ────────────────────────────────────────────────────────


async def _query_langsearch(valves, query: str, count: int = 0) -> List[Document]:
    url = "https://api.langsearch.com/v1/web-search"
    headers = {
        "Authorization": f"Bearer {valves.langsearch_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "freshness": valves.freshness,
        "summary": valves.enable_summary,
        "count": count if count > 0 else valves.count,
    }
    timeout = aiohttp.ClientTimeout(total=30)

    async with _API_SEM:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"LangSearch {response.status}: {text[:200]}")
                        return []
                    data = await response.json()
                    web_pages = data.get("webPages", {}) or data.get("data", {}).get(
                        "webPages", {}
                    )
                    items = web_pages.get("value", [])
                    docs = []
                    for item in items:
                        raw_content = item.get("summary") or item.get("snippet", "")
                        docs.append(
                            Document(
                                page_content=raw_content,
                                metadata={
                                    "source": item.get("url"),
                                    "site_name": item.get("siteName", "Web"),
                                    "title": item.get("name"),
                                },
                            )
                        )
                    return docs
            except asyncio.TimeoutError:
                logger.error(f"LangSearch timeout: {query[:80]}")
                return []
            except Exception as e:
                logger.error(f"LangSearch error: {e}")
                return []


# ── Reranker ──────────────────────────────────────────────────────────────


async def _rerank_docs(valves, query: str, docs: List[Document]) -> List[Document]:
    """Rerank via external BGE reranker. Falls back to original order on failure."""
    if not docs or not valves.reranker_url:
        return docs

    doc_texts = [" ".join(d.page_content.split())[:2000] for d in docs]
    payload = {
        "model": valves.reranker_model,
        "query": query,
        "documents": doc_texts,
        "top_n": len(docs),
    }
    headers = {"Content-Type": "application/json"}
    if valves.reranker_api_key:
        headers["Authorization"] = f"Bearer {valves.reranker_api_key}"

    timeout = aiohttp.ClientTimeout(total=15)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                valves.reranker_url, headers=headers, json=payload
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        f"Reranker {resp.status}: {(await resp.text())[:200]}"
                    )
                    return docs
                data = await resp.json()
                results = data.get("results", [])
                if not results:
                    return docs

                for item in results:
                    idx = item.get("index", -1)
                    score = item.get("relevance_score", 0.0)
                    if 0 <= idx < len(docs):
                        docs[idx].relevance_score = score

                scored = [d for d in docs if d.relevance_score is not None]
                scored.sort(key=lambda d: d.relevance_score, reverse=True)

                if valves.min_relevance_score > 0:
                    filtered = [
                        d
                        for d in scored
                        if d.relevance_score >= valves.min_relevance_score
                    ]
                    if not filtered and scored:
                        filtered = scored[:3]
                    scored = filtered

                return scored if scored else docs
    except Exception as e:
        logger.warning(f"Reranker failed (unranked): {e}")
        return docs


# ── Quality Filters ───────────────────────────────────────────────────────


def _latin_ratio(text: str) -> float:
    """Return the fraction of alphabetic chars that are Latin-script."""
    if not text:
        return 0.0
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    latin = sum(1 for c in alpha_chars if c < "\u0250")  # Basic Latin + Extended
    return latin / len(alpha_chars)


def _query_is_latin(query: str) -> bool:
    """Check if the query is predominantly Latin-script (English, etc.)."""
    return _latin_ratio(query) > 0.8


def _quality_filter(
    docs: List[Document], query: str, min_content_len: int = 100
) -> List[Document]:
    """
    Remove low-quality results:
    1. Empty or missing title
    2. Content shorter than min_content_len
    3. Non-Latin content when the query is Latin-script
    """
    query_latin = _query_is_latin(query)
    filtered = []

    for doc in docs:
        title = (doc.metadata.get("title") or "").strip()
        content = " ".join(doc.page_content.split())

        # Gate 1: Must have a title
        if not title:
            logger.debug(f"Quality filter: dropped result with empty title")
            continue

        # Gate 2: Must have substantial content
        if len(content) < min_content_len:
            logger.debug(
                f"Quality filter: dropped '{title[:50]}' — content too short ({len(content)} chars)"
            )
            continue

        # Gate 3: Language mismatch — skip non-Latin results for Latin queries
        if query_latin and _latin_ratio(content) < 0.5:
            logger.debug(
                f"Quality filter: dropped '{title[:50]}' — non-Latin content for Latin query"
            )
            continue

        filtered.append(doc)

    return filtered


def _dedup_docs(docs: List[Document]) -> List[Document]:
    """Deduplicate by URL, then by content similarity (>60% overlap in first 200 chars)."""
    seen_urls = set()
    seen_content = []
    unique = []

    for doc in docs:
        url = doc.metadata.get("source", "")

        # URL dedup
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)

        # Content similarity dedup (compare first 200 chars of cleaned content)
        content_prefix = " ".join(doc.page_content.split())[:200].lower()
        is_dup = False
        for existing in seen_content:
            if not content_prefix or not existing:
                continue
            # Simple overlap: shared chars / max length
            shorter = min(len(content_prefix), len(existing))
            if shorter < 50:
                continue
            matches = sum(1 for a, b in zip(content_prefix, existing) if a == b)
            if matches / shorter > 0.6:
                is_dup = True
                break

        if is_dup:
            logger.debug(
                f"Content dedup: dropped '{doc.metadata.get('title', '')[:50]}'"
            )
            continue

        seen_content.append(content_prefix)
        unique.append(doc)

    return unique


# ── Page Fetcher ──────────────────────────────────────────────────────────


async def _fetch_page_text(url: str, max_chars: int = 8000) -> str:
    """Fetch URL and extract text. Browser-like UA to avoid 403s."""
    timeout = aiohttp.ClientTimeout(total=20)
    headers = {"User-Agent": _BROWSER_UA}
    try:
        async with _API_SEM:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    url, headers=headers, allow_redirects=True
                ) as resp:
                    if resp.status != 200:
                        return ""
                    ct = resp.headers.get("content-type", "")
                    text = await resp.text()
                    if "text/html" in ct:
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
                    return text[:max_chars] if text else ""
    except Exception as e:
        logger.warning(f"Auto-fetch failed for {url[:80]}: {e}")
        return ""


# ── Query Parser ──────────────────────────────────────────────────────────


def _parse_queries(query: str, queries: str) -> List[str]:
    result = []
    if queries and queries.strip():
        q = queries.strip()
        if q.startswith("["):
            try:
                parsed = json.loads(q)
                if isinstance(parsed, list):
                    result.extend(
                        str(item).strip() for item in parsed if str(item).strip()
                    )
            except json.JSONDecodeError:
                pass
        if not result:
            result.extend(part.strip() for part in q.split(";") if part.strip())
    if not result and query and query.strip():
        result.append(query.strip())
    if query and query.strip() and result and query.strip() not in result:
        result.insert(0, query.strip())
    return result[:3]


# ── Quota ─────────────────────────────────────────────────────────────────


def _quota_file_path() -> Path:
    base = Path(BASE_DIR)
    path = base / "data" / "langsearch_quota.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _check_and_increment_quota(valves, count: int = 1) -> Tuple[bool, int, int]:
    limit = valves.daily_limit
    today = date.today().isoformat()
    path = _quota_file_path()
    data = {"date": today, "used": 0}
    if path.exists():
        try:
            content = json.loads(path.read_text())
            if content.get("date") == today:
                data = content
        except Exception:
            pass
    if data["used"] + count > limit:
        return False, data["used"], limit
    data["used"] += count
    try:
        path.write_text(json.dumps(data))
    except Exception:
        pass
    return True, data["used"], limit


# ── Formatters ────────────────────────────────────────────────────────────


def _error_json(msg: str) -> str:
    """Return error as valid JSON so OWUI middleware's json.loads() won't crash."""
    return json.dumps([{"error": msg, "_meta": {"total_results": 0}}])


def _format_results_rag(
    valves, docs: List[Document], fetched: Dict[str, str] = None
) -> str:
    fetched = fetched or {}
    results = []
    for i, doc in enumerate(docs):
        clean_content = " ".join(doc.page_content.split())
        limit = valves.max_chars_per_result
        if len(clean_content) > limit:
            clean_content = clean_content[:limit] + "... [Truncated]"

        entry = {
            "ref": i + 1,
            "title": doc.metadata.get("title", ""),
            "url": doc.metadata.get("source", ""),
            "content": clean_content,
        }
        if doc.relevance_score is not None:
            entry["relevance"] = round(doc.relevance_score, 4)

        page_url = doc.metadata.get("source", "")
        if page_url in fetched and fetched[page_url]:
            entry["full_page"] = fetched[page_url]

        results.append(entry)

    n = len(results)
    reranked = any(d.relevance_score is not None for d in docs)
    hint = (
        f"Returned {n} quality-filtered results"
        + (" (reranked by relevance)" if reranked else "")
        + ". Synthesize your answer from these. "
        "Do NOT call search_web again unless NONE of these results address the query."
    )
    results.append({"_meta": {"total_results": n, "hint": hint}})
    return json.dumps(results, ensure_ascii=False)


def _katex_escape_str(string: str) -> str:
    return (
        string.replace("\n", "\\n")
        .replace("\\[", "{[}")
        .replace("\\]", "{]}")
        .replace("\r", "")
    )


def _format_results_display(docs: List[Document]) -> str:
    output = ["### Search Results"]
    for i, doc in enumerate(docs):
        clean_content = " ".join(doc.page_content.split())[:300]
        score_tag = ""
        if doc.relevance_score is not None:
            score_tag = f" (rel: {doc.relevance_score:.2f})"
        block = (
            f"Ref [{i+1}]{score_tag}: {doc.metadata.get('title', '')}\n"
            f"URL: {doc.metadata.get('source', '')}\n"
            f"Data: {clean_content}...\n"
        )
        output.append(block)
    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════
#  OWUI Tool class — ONLY public methods here
# ═══════════════════════════════════════════════════════════════════════════


class Tools:
    class Valves(BaseModel):
        # ── LangSearch ────────────────────────────────────────────────
        langsearch_api_key: str = Field(
            default="", description="API key from https://langsearch.com/dashboard"
        )
        freshness: str = Field(
            default="noLimit",
            description="Time filter: 'noLimit', 'oneDay', 'oneWeek', 'oneMonth', 'oneYear'",
        )
        count: int = Field(
            default=10,
            description="Results per query from LangSearch (max 10)",
            ge=1,
            le=10,
        )
        enable_summary: bool = Field(
            default=True, description="Use AI summaries (richer context per result)"
        )
        daily_limit: int = Field(default=1000, description="Max API calls per day")
        max_chars_per_result: int = Field(
            default=1500, description="Max snippet length per result"
        )

        # ── Reranker ──────────────────────────────────────────────────
        reranker_enabled: bool = Field(
            default=True,
            description="Score and filter results using BGE reranker",
        )
        reranker_url: str = Field(
            default=os.getenv("RERANKER_URL", "http://localhost:8040/api/v1/reranking"),
            description="Reranker API endpoint",
        )
        reranker_model: str = Field(
            default="bge-reranker-v2-m3-GGUF",
            description="Reranker model name",
        )
        reranker_api_key: str = Field(
            default="", description="Reranker API key (leave blank if none)"
        )
        min_relevance_score: float = Field(
            default=1.0,
            description="Min BGE reranker score to keep a result (raw logit, typical range 0-5+). Higher = stricter.",
            ge=0.0,
        )

        # ── Quality Filters ───────────────────────────────────────────
        quality_filter_enabled: bool = Field(
            default=True,
            description="Drop results with empty titles, short content, or wrong language",
        )
        min_content_length: int = Field(
            default=100,
            description="Min chars of content to keep a result",
            ge=0,
        )
        max_final_results: int = Field(
            default=5,
            description="Max results to return after all filtering (quality > quantity)",
            ge=1,
            le=10,
        )

        # ── Auto-fetch ────────────────────────────────────────────────
        auto_fetch_top: int = Field(
            default=2,
            description="Auto-fetch full page text for the top N results after filtering (0 = disabled)",
            ge=0,
            le=5,
        )
        max_chars_per_fetch: int = Field(
            default=8000, description="Max chars when auto-fetching a page"
        )

        # ── UI ────────────────────────────────────────────────────────
        include_citations: bool = Field(
            default=True, description="Show clickable citations in the UI"
        )
        keep_results_in_context: bool = Field(
            default=True, description="Inject search data into the chat context"
        )

    def __init__(self):
        self.valves = self.Valves()

    async def search_web(
        self,
        query: str = "",
        __user__: dict = None,
        __event_emitter__=None,
        queries: str = "",
        max_results: int = 0,
        fetch_top: int = -1,
        **kwargs,
    ) -> str:
        """
        Search the web. Returns quality-filtered results with title, URL, content, and optionally full page text.

        SINGLE QUERY: Set `query` to a short keyword phrase (2-5 words).
        MULTI-QUERY: Set `queries` to a JSON array or semicolons, e.g. '["perovskite stability", "halide degradation"]'. Up to 3 queries run concurrently.

        TIPS: Short keyword queries work best. Do NOT use quotes, site: operators, or boolean syntax.
        IMPORTANT: Results are reranked and quality-filtered. Synthesize your answer from these — do NOT call search_web again unless NONE of these results address the question.

        :param query: Primary search query — short keywords (e.g. "perovskite stability 2025").
        :param queries: Optional additional queries as JSON array or semicolon-separated. Up to 3 total.
        :param max_results: Override max results to return (0 = use valve default).
        :param fetch_top: Override auto-fetch count (-1 = use valve default, 0 = disable).
        """
        # ── 0. Coerce None → defaults (LLMs may pass null for optional params) ──
        query = query if query is not None else ""
        queries = queries if queries is not None else ""
        max_results = max_results if max_results is not None else 0
        fetch_top = fetch_top if fetch_top is not None else -1

        # ── 1. Validation ─────────────────────────────────────────────
        if not self.valves.langsearch_api_key:
            msg = "Error: LangSearch API Key is missing."
            await _emit_status(__event_emitter__, msg, True)
            return _error_json(msg)

        # ── 2. Parse queries ──────────────────────────────────────────
        all_queries = _parse_queries(query, queries)
        if not all_queries:
            return _error_json("No search query provided.")

        num_queries = len(all_queries)
        per_query_count = self.valves.count
        if num_queries > 1:
            per_query_count = min(per_query_count, 5)

        # Resolve fetch_top: -1 = use valve, 0+ = override
        effective_fetch_top = self.valves.auto_fetch_top if fetch_top < 0 else fetch_top
        effective_fetch_top = max(0, min(effective_fetch_top, 5))

        # Resolve max results
        effective_max = (
            max_results if max_results > 0 else self.valves.max_final_results
        )

        # ── 3. Quota check ────────────────────────────────────────────
        allowed, used, limit = _check_and_increment_quota(
            self.valves, count=num_queries
        )
        if not allowed:
            return _error_json(f"Daily search limit reached ({used}/{limit}).")

        # ── 4. Search (concurrent) ────────────────────────────────────
        await _emit_status(
            __event_emitter__,
            f"Searching ({num_queries} {'queries' if num_queries > 1 else 'query'})...",
            False,
        )
        tasks = [
            _query_langsearch(self.valves, q, count=per_query_count)
            for q in all_queries
        ]
        try:
            results_per_query = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Search gather failed: {e}")
            await _emit_status(__event_emitter__, "Search failed.", True)
            return _error_json(str(e))

        all_docs = []
        for result in results_per_query:
            if isinstance(result, list):
                all_docs.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"One query failed: {result}")

        if not all_docs:
            await _emit_status(__event_emitter__, "No results found.", True)
            return _error_json("No relevant results found.")

        # ── 5. Deduplication ──────────────────────────────────────────
        all_docs = _dedup_docs(all_docs)

        # ── 6. Quality filter ─────────────────────────────────────────
        pre_filter_count = len(all_docs)
        if self.valves.quality_filter_enabled:
            all_docs = _quality_filter(
                all_docs, all_queries[0], self.valves.min_content_length
            )
            dropped = pre_filter_count - len(all_docs)
            if dropped:
                logger.info(
                    f"Quality filter: dropped {dropped}/{pre_filter_count} results"
                )

        if not all_docs:
            await _emit_status(__event_emitter__, "No quality results found.", True)
            return _error_json("No relevant results found after quality filtering.")

        # ── 7. Rerank ─────────────────────────────────────────────────
        if self.valves.reranker_enabled and self.valves.reranker_url:
            await _emit_status(__event_emitter__, "Reranking...", False)
            all_docs = await _rerank_docs(self.valves, all_queries[0], all_docs)

        # ── 8. Cap results ────────────────────────────────────────────
        all_docs = all_docs[:effective_max]

        # ── 9. Auto-fetch survivors ───────────────────────────────────
        fetched_pages: Dict[str, str] = {}
        if effective_fetch_top > 0 and all_docs:
            fetch_count = min(effective_fetch_top, len(all_docs))
            fetch_urls = [
                doc.metadata.get("source", "")
                for doc in all_docs[:fetch_count]
                if doc.metadata.get("source")
            ]
            if fetch_urls:
                await _emit_status(
                    __event_emitter__, f"Fetching {len(fetch_urls)} page(s)...", False
                )
                fetch_tasks = [
                    _fetch_page_text(u, self.valves.max_chars_per_fetch)
                    for u in fetch_urls
                ]
                fetch_results = await asyncio.gather(
                    *fetch_tasks, return_exceptions=True
                )
                for url, text in zip(fetch_urls, fetch_results):
                    if isinstance(text, str) and text:
                        fetched_pages[url] = text

        # ── 10. Citations ─────────────────────────────────────────────
        if self.valves.include_citations and __event_emitter__:
            for doc in all_docs:
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "document": [doc.page_content],
                            "metadata": [doc.metadata],
                            "source": {"name": doc.metadata.get("site_name") or "Web"},
                        },
                    }
                )

        # ── 11. Format for LLM ────────────────────────────────────────
        formatted_results = _format_results_rag(self.valves, all_docs, fetched_pages)

        # ── 12. Visual injection ──────────────────────────────────────
        if self.valves.keep_results_in_context:
            display_text = _format_results_display(all_docs)
            escaped = _katex_escape_str(display_text)
            await _emit_message(__event_emitter__, f"\\[ % {escaped}\n \\] ")

        await _emit_status(__event_emitter__, "Done.", True)
        return formatted_results
