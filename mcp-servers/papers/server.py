#!/usr/bin/env python3
"""
Academic Download MCP Server - Optimized for LLM Tool Pipelines
Author: Marios Adamidis (FORTHought Lab)
High-performance academic search, download, and reference resolution.
Designed for minimal context consumption in multi-tool chains.

Sources: arXiv, OpenAlex (keyword + semantic + author-ID), NASA ADS,
         Semantic Scholar (search + author-ID), PubMed, CrossRef, OpenAIRE, ORCID
Download: arXiv -> OpenAlex Content -> Unpaywall -> Elsevier API -> Direct Publisher -> PMC -> Sci-Hub

v5.7 changes:
  - Author identity resolution: resolve name → OA Author ID + S2 Author ID + ORCID
    before querying, then use ID-based endpoints for exhaustive author coverage
  - OpenAlex author-ID works: filter=authorships.author.id:{id} with cursor pagination
    (replaces display_name.search for author queries — 135 vs ~20 results)
  - Semantic Scholar author papers: /author/{id}/papers endpoint with offset pagination
    (replaces name-prepend-to-query hack — 344 vs ~10 results)
  - ORCID works: reads self-curated publication list from ORCID record (DOI-based)
  - OpenAIRE: 8th source, 324M+ EU/repository products, ORCID + name search
  - Resolution cache: author IDs cached for session to avoid re-resolving
  - Graceful fallback: if resolution fails, falls back to v5.6 name-based search

v5.6 changes:
  - Over-fetch + rank: author searches cast wide net (3x max_results per source),
    dedup merges duplicates, composite scoring ranks results, returns top N with pool summary
  - CrossRef promoted to primary source for author searches (was fallback-only)
  - CrossRef: native query.author= filter, year range filtering, published-online fallback
  - Source agreement tracking: papers found by multiple sources ranked higher
  - Composite scoring: relevance (word overlap) + recency + citations + source agreement
  - Pool summary: model sees full picture (raw count, unique count, per-source breakdown)
  - max_results cap raised from 20 to 50

v5.5b fix:
  - Semantic mode is now ADDITIVE: all 5 keyword sources always fire,
    OA /find/works adds embedding results on top when endpoint goes live.
  - Prevents auto-mode from silently dropping OpenAlex + PubMed.

v5.5 changes (backward-compatible search() gains optional params):
  - OpenAlex semantic search: /find/works endpoint (AI embedding similarity)
  - OpenAlex boolean search: AND/OR/NOT, exact phrases, field-targeted
  - OpenAlex content download: content.openalex.org PDFs as download source
  - OpenAlex batch DOI lookup: pipe-separated batch resolution (up to 50)
  - NASA ADS integration: physics/astrophysics/instrumentation optimized search
  - Semantic Scholar integration: 200M+ papers, CS/STEM relevance ranking (API key auth)
  - search() tool: new optional params year_min, year_max, sort, mode
  - Improved dedup: Jaccard word-similarity (threshold 0.85) instead of prefix
  - CrossRef promoted to fallback-only (fires when primary sources < 3 results)

v5.3 changes:
  - Cookie jar auth, Elsevier API, cookie health monitoring

v5.2 changes:
  - Direct publisher download via institutional IP (HEAL-Link / campus network)
  - 17 publisher PDF URL patterns, smart APS detection, 4xx fail-fast

Tools:
  - search:             Search papers across sources (keyword or semantic mode)
  - details:            Get paper abstract/metadata
  - download:           Download single paper PDF
  - batch_details:      Get details for multiple papers (compact by default)
  - batch_download:     Download multiple papers, zip result, report failures
  - resolve_references: Resolve raw citation strings to DOIs via CrossRef
  - books:              Search Library Genesis / Anna's Archive
  - book_download:      Download book from LibGen / Anna's Archive
"""

import asyncio
import json
import os
import random
import re
import math
import sys
import unicodedata
import zipfile
from datetime import datetime
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from typing import Optional
from urllib.parse import quote, unquote, urlparse

import httpx
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

# =========================================================================
# CONFIGURATION
# =========================================================================

DOWNLOAD_DIR = Path(os.getenv("FILE_EXPORT_DIR", "/data/files")) / "papers"
PUBLIC_BASE_URL = os.getenv("FILE_EXPORT_BASE_URL", "http://localhost:8084/files")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
MAILTO = os.getenv("OPENALEX_MAILTO", "your-email@example.com")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")

# Elsevier API (for ScienceDirect articles)
ELSEVIER_API_KEY = os.getenv("ELSEVIER_API_KEY", "")
ELSEVIER_INSTTOKEN = os.getenv("ELSEVIER_INSTTOKEN", "")

# NASA ADS API (for physics/astrophysics/instrumentation)
ADS_API_KEY = os.getenv("ADS_API_KEY", "")

# Semantic Scholar API (broad STEM, CS-strong)
S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# ORCID Public API (author identity resolution + works)
ORCID_CLIENT_ID = os.getenv("ORCID_CLIENT_ID", "")
ORCID_CLIENT_SECRET = os.getenv("ORCID_CLIENT_SECRET", "")
_orcid_token: Optional[str] = None  # Populated lazily on first use

# Cookie jar for institutional auth (Netscape cookies.txt format)
COOKIE_FILE = os.getenv("COOKIE_FILE", "")

# Timeouts (seconds)
SEARCH_TIMEOUT = 20.0
DETAIL_TIMEOUT = 15.0
DOWNLOAD_TIMEOUT = 90.0

# Concurrency limits
BATCH_SEMAPHORE = 5       # Max concurrent batch operations (resolve, details)
DOWNLOAD_SEMAPHORE = 8    # Max concurrent downloads (IO-bound, can be higher)
SEARCH_MAX_DEFAULT = 8    # Default search results cap

# Ensure download directory exists
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================================
# COOKIE JAR
# =========================================================================


def _load_cookie_jar() -> Optional[MozillaCookieJar]:
    """Load Netscape cookies.txt and return a MozillaCookieJar, or None."""
    cookie_path = COOKIE_FILE
    if not cookie_path:
        return None
    p = Path(cookie_path)
    if not p.exists():
        _log(f"Cookie file not found: {cookie_path}")
        return None
    try:
        jar = MozillaCookieJar(str(p))
        jar.load(ignore_discard=True, ignore_expires=True)
        # Stats
        total = len(jar)
        now = datetime.now().timestamp()
        expired = sum(1 for c in jar if c.expires and c.expires < now)
        domains = set(c.domain.lstrip(".") for c in jar)
        _log(f"Loaded {total} cookies from {p.name} ({len(domains)} domains, {expired} expired)")
        # Warn about key publisher cookies
        publisher_domains = {
            "journals.aps.org": "APS",
            "pubs.aip.org": "AIP",
            "iopscience.iop.org": "IOP",
            "onlinelibrary.wiley.com": "Wiley",
            "link.springer.com": "Springer",
            "www.sciencedirect.com": "Elsevier",
        }
        for domain, name in publisher_domains.items():
            domain_cookies = [c for c in jar if domain in c.domain]
            if not domain_cookies:
                _log(f"  WARNING: No cookies for {name} ({domain})")
            else:
                fresh = [c for c in domain_cookies if not c.expires or c.expires > now]
                if len(fresh) < len(domain_cookies) / 2:
                    _log(f"  WARNING: Most {name} cookies expired -- consider re-exporting")
        # Check file age
        file_age_days = (now - p.stat().st_mtime) / 86400
        if file_age_days > 7:
            _log(f"  WARNING: Cookie file is {file_age_days:.0f} days old -- session cookies likely expired")
        return jar
    except Exception as e:
        _log(f"Failed to load cookie file: {e}")
        return None


def _cookiejar_to_httpx(jar: MozillaCookieJar) -> httpx.Cookies:
    """Convert a MozillaCookieJar to httpx.Cookies for use with the client."""
    cookies = httpx.Cookies()
    now = datetime.now().timestamp()
    for cookie in jar:
        # Skip expired cookies
        if cookie.expires and cookie.expires < now:
            continue
        try:
            cookies.set(
                cookie.name,
                cookie.value or "",
                domain=cookie.domain,
                path=cookie.path,
            )
        except Exception:
            continue
    return cookies


# =========================================================================
# GLOBAL HTTP CLIENT (Connection Pooling)
# =========================================================================

_client: Optional[httpx.AsyncClient] = None
_cookies: Optional[httpx.Cookies] = None


def _init_cookies():
    """Initialize cookies from cookie file (called once at startup)."""
    global _cookies
    if COOKIE_FILE:
        jar = _load_cookie_jar()
        if jar:
            _cookies = _cookiejar_to_httpx(jar)
            _log(f"Loaded {len(_cookies)} active cookies into HTTP client")
        else:
            _cookies = None
    else:
        _log("No COOKIE_FILE configured -- institutional cookie auth disabled")
        _cookies = None


async def get_client() -> httpx.AsyncClient:
    """Get or create the shared HTTP client with connection pooling and cookies."""
    global _client, _cookies
    if _client is None or _client.is_closed:
        # Initialize cookies on first client creation
        if _cookies is None and COOKIE_FILE:
            _init_cookies()
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            cookies=_cookies,
        )
    return _client

# =========================================================================
# MCP SERVER
# =========================================================================

mcp = FastMCP(
    "academic",
    host=os.getenv("MCP_HTTP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_HTTP_PORT", "9005")),
)


def _log(msg: str):
    print(f"[academic] {msg}", file=sys.stderr, flush=True)


def _sanitize(s: str, max_len: int = 80) -> str:
    return re.sub(r"[^\w\-]", "_", s)[:max_len].strip("_")


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _pub_url(filename: str) -> str:
    return f"{PUBLIC_BASE_URL}/papers/{quote(filename)}"


def _clean_doi(doi: str) -> str:
    if not doi:
        return ""
    doi = (
        doi.replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("https://dx.doi.org/", "")
        .replace("http://dx.doi.org/", "")
        .replace("doi:", "")
        .replace("DOI:", "")
        .replace("DOI ", "")
        .strip()
        .rstrip(".")
    )
    # Handle URL-encoded DOIs (e.g., 10.1002%2Fjcc.1087)
    if "%" in doi:
        doi = unquote(doi)
    # Strip any trailing punctuation or whitespace
    doi = re.sub(r'[\s,;]+$', '', doi)
    return doi


def _clean_arxiv(arxiv_id: str) -> str:
    if not arxiv_id:
        return ""
    return arxiv_id.replace("arXiv:", "").replace("arxiv:", "").split("v")[0].strip()


# =========================================================================
# COMPACT HELPERS -- minimize context consumption
# =========================================================================


def _compact_paper(paper: dict) -> dict:
    """Reduce paper metadata for context-constrained models.
    First-sentence abstract, max 3 authors, strip nulls."""
    out = {k: v for k, v in paper.items() if v is not None}
    if out.get("abstract"):
        # Take first sentence (split on ". " to avoid splitting on "Dr." etc.)
        first = out["abstract"].split(". ")[0].rstrip(".")
        out["abstract"] = first + "."
    authors = out.get("authors", [])
    if len(authors) > 3:
        out["authors"] = authors[:3] + [f"+{len(authors) - 3} more"]
    return out


def _slim_search_result(paper: dict) -> dict:
    """Produce minimal search result: title, first_author, year, identifiers, download hint.
    This is what the model sees when presenting results to the user."""
    authors = paper.get("authors", [])
    out = {"title": paper.get("title", ""), "year": paper.get("year")}
    if authors:
        out["first_author"] = authors[0] if isinstance(authors[0], str) else str(authors[0])
    if paper.get("doi"):
        out["doi"] = paper["doi"]
    if paper.get("arxiv_id"):
        out["arxiv"] = paper["arxiv_id"]
    if paper.get("pmid"):
        out["pmid"] = paper["pmid"]
    if paper.get("bibcode"):
        out["bibcode"] = paper["bibcode"]
    if paper.get("citations"):
        out["cites"] = paper["citations"]
    if paper.get("topic"):
        out["topic"] = paper["topic"]
    if paper.get("is_oa"):
        out["oa"] = True
    if paper.get("similarity"):
        out["sim"] = paper["similarity"]
    # Source agreement — papers found by multiple sources are higher quality
    sc = paper.get("_source_count", 1)
    if sc > 1:
        out["sources_found"] = sc
    # Download hint -- tells model which identifier to use
    out["dl"] = "arxiv" if paper.get("arxiv_id") else ("doi" if paper.get("doi") else None)
    return {k: v for k, v in out.items() if v is not None}


def _oa_params(**extra) -> dict:
    """Build OpenAlex query params with polite pool and optional API key."""
    params = {"mailto": MAILTO}
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    params.update(extra)
    return params


def _ads_headers() -> dict:
    """Build NASA ADS auth headers."""
    return {"Authorization": f"Bearer {ADS_API_KEY}"} if ADS_API_KEY else {}


# =========================================================================
# ORCID TOKEN
# =========================================================================

async def _get_orcid_token(client: httpx.AsyncClient) -> Optional[str]:
    """Get or refresh ORCID read-public token (client_credentials grant).
    Token lasts ~20 years so effectively permanent."""
    global _orcid_token
    if _orcid_token:
        return _orcid_token
    if not ORCID_CLIENT_ID or not ORCID_CLIENT_SECRET:
        return None
    try:
        r = await client.post(
            "https://orcid.org/oauth/token",
            data={
                "client_id": ORCID_CLIENT_ID,
                "client_secret": ORCID_CLIENT_SECRET,
                "scope": "/read-public",
                "grant_type": "client_credentials",
            },
            headers={"Accept": "application/json"},
            timeout=15.0,
        )
        if r.status_code == 200:
            _orcid_token = r.json().get("access_token")
            _log(f"ORCID token acquired (expires in {r.json().get('expires_in', '?')}s)")
            return _orcid_token
        _log(f"ORCID token error: HTTP {r.status_code}")
    except Exception as e:
        _log(f"ORCID token error: {e}")
    return None


# =========================================================================
# AUTHOR IDENTITY RESOLUTION
# =========================================================================

# Cache: name_lower -> {oa_id, s2_id, orcid, display_name}
_author_cache: dict[str, dict] = {}


async def _resolve_author(
    client: httpx.AsyncClient, name: str
) -> dict:
    """Resolve an author name to canonical IDs across platforms.

    Returns dict with keys: oa_id, s2_id, orcid, display_name
    Any key may be None if resolution failed for that platform.
    Results are cached for the session.
    """
    cache_key = name.lower().strip()
    if cache_key in _author_cache:
        _log(f"  author resolve: cache hit for '{name}'")
        return _author_cache[cache_key]

    _log(f"  author resolve: resolving '{name}'...")
    result = {"oa_id": None, "s2_id": None, "orcid": None, "display_name": name}

    # --- OpenAlex author search (most reliable, has ORCID) ---
    try:
        r = await client.get(
            "https://api.openalex.org/authors",
            params=_oa_params(search=name),
            timeout=SEARCH_TIMEOUT,
        )
        if r.status_code == 200:
            authors = r.json().get("results", [])
            if authors:
                top = authors[0]
                result["oa_id"] = top.get("id", "").replace("https://openalex.org/", "")
                result["display_name"] = top.get("display_name", name)
                if top.get("orcid"):
                    result["orcid"] = top["orcid"].replace("https://orcid.org/", "")
                _log(f"    OA: {result['oa_id']} ({top.get('works_count', '?')} works)"
                     + (f", ORCID={result['orcid']}" if result['orcid'] else ""))
    except Exception as e:
        _log(f"    OA author resolve error: {e}")

    # --- Semantic Scholar author search ---
    try:
        headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
        r = await client.get(
            "https://api.semanticscholar.org/graph/v1/author/search",
            params={"query": name, "limit": 5, "fields": "name,paperCount,hIndex"},
            headers=headers,
            timeout=SEARCH_TIMEOUT,
        )
        if r.status_code == 429:
            await asyncio.sleep(1.5)
            r = await client.get(
                "https://api.semanticscholar.org/graph/v1/author/search",
                params={"query": name, "limit": 5, "fields": "name,paperCount,hIndex"},
                headers=headers,
                timeout=SEARCH_TIMEOUT,
            )
        if r.status_code == 200:
            s2_authors = r.json().get("data", [])
            if s2_authors:
                # Pick the author with the most papers (disambiguation heuristic)
                best = max(s2_authors, key=lambda a: a.get("paperCount", 0))
                result["s2_id"] = best.get("authorId")
                _log(f"    S2: {result['s2_id']} ({best.get('paperCount', '?')} papers, "
                     f"h={best.get('hIndex', '?')})")
    except Exception as e:
        _log(f"    S2 author resolve error: {e}")

    # --- ORCID search (if not already found via OpenAlex) ---
    if not result["orcid"]:
        try:
            token = await _get_orcid_token(client)
            if token:
                # Split name into parts for structured search
                parts = name.strip().split()
                if len(parts) >= 2:
                    given = parts[0]
                    family = parts[-1]
                    q = f"family-name:{family}+AND+given-names:{given}"
                else:
                    q = f"family-name:{name}"
                r = await client.get(
                    f"https://pub.orcid.org/v3.0/expanded-search/?q={q}&rows=3",
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {token}",
                    },
                    timeout=SEARCH_TIMEOUT,
                )
                if r.status_code == 200:
                    orcid_results = r.json().get("expanded-result", [])
                    if orcid_results:
                        result["orcid"] = orcid_results[0].get("orcid-id")
                        _log(f"    ORCID: {result['orcid']}")
        except Exception as e:
            _log(f"    ORCID resolve error: {e}")

    _author_cache[cache_key] = result
    resolved = sum(1 for k in ("oa_id", "s2_id", "orcid") if result[k])
    _log(f"  author resolve: {resolved}/3 IDs found for '{name}'")
    return result


# =========================================================================
# DEDUP HELPERS -- Jaccard word-set similarity
# =========================================================================


def _normalize_title(title: str) -> str:
    """Lowercase, strip accents, remove punctuation, collapse whitespace."""
    t = unicodedata.normalize("NFKD", title.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _titles_match(a: str, b: str, threshold: float = 0.85) -> bool:
    """Jaccard word-set similarity above threshold."""
    wa = set(_normalize_title(a).split())
    wb = set(_normalize_title(b).split())
    if not wa or not wb:
        return False
    inter = len(wa & wb)
    union = len(wa | wb)
    return (inter / union) >= threshold if union else False


# =========================================================================
# SEARCH SOURCES
# =========================================================================


async def _arxiv_search(client: httpx.AsyncClient, query: str, n: int, author: str = None) -> list:
    """Search arXiv."""
    try:
        # Clean query for arXiv API compatibility
        q = (query or "").replace('\\"', '"')  # unescape model-generated quotes

        # Build search query parts
        parts = []

        # Author filter using arXiv au: field
        if author:
            # Clean author name for arXiv (remove dots, use last name)
            au_clean = author.replace('\\"', '').replace('"', '').strip()
            parts.append(f'au:"{au_clean}"')

        # Topic/keyword query
        _FIELD_PREFIXES = ("au:", "ti:", "abs:", "cat:", "author:", "title:")
        if q:
            if any(q.lower().startswith(p) or f" {p}" in q.lower() for p in _FIELD_PREFIXES):
                # Map common prefixes to arXiv syntax
                parts.append(q.replace("author:", "au:").replace("title:", "ti:"))
            else:
                parts.append(f"all:{q}")

        search_q = " AND ".join(parts) if parts else "all:*"

        r = await client.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": search_q, "max_results": n, "sortBy": "relevance"},
            timeout=SEARCH_TIMEOUT,
        )
        papers = []
        soup = BeautifulSoup(r.text, "xml")
        for entry in soup.find_all("entry"):
            arxiv_url = entry.find("id").text.strip()
            arxiv_id = arxiv_url.split("/abs/")[-1].split("v")[0]
            doi_tag = entry.find("arxiv:doi")
            authors = [a.find("name").text for a in entry.find_all("author")]
            pub = entry.find("published")
            year = int(pub.text[:4]) if pub else None
            papers.append({
                "title": entry.find("title").text.strip().replace("\n", " "),
                "authors": authors,
                "year": year,
                "arxiv_id": arxiv_id,
                "doi": doi_tag.text.strip() if doi_tag else None,
                "source": "arxiv",
            })
        return papers
    except Exception as e:
        _log(f"arXiv search error: {e}")
        return []


async def _oa_keyword_search(
    client: httpx.AsyncClient,
    query: str,
    n: int,
    year_min: int = None,
    year_max: int = None,
    sort_mode: str = "relevance",
    author: str = None,
) -> list:
    """OpenAlex keyword/boolean search with year filtering, sort, and field selection.

    Supports boolean operators (AND, OR, NOT) and exact phrases ("...") in query.
    Uses filter-based search for year ranges and field-targeted queries.
    When author is provided, uses native authorships filter for accurate results.
    """
    try:
        params = {}
        filters = []

        # Author filtering via native OpenAlex authorships filter
        if author:
            filters.append(f"authorships.author.display_name.search:{author}")

        # Detect field-targeted queries (e.g. "title:ablation")
        has_field = query and any(
            query.lower().startswith(p)
            for p in ("title:", "abstract:", "fulltext:", "title_and_abstract:")
        )

        if has_field:
            # Field-targeted: put in filter syntax
            filters.append(f"default.search:{query}")
        elif query:
            # Regular search (supports boolean AND/OR/NOT natively)
            params["search"] = query

        # Year filtering via OpenAlex filter syntax
        if year_min and year_max:
            filters.append(f"publication_year:{year_min}-{year_max}")
        elif year_min:
            filters.append(f"publication_year:>{year_min - 1}")
        elif year_max:
            filters.append(f"publication_year:<{year_max + 1}")

        if filters:
            params["filter"] = ",".join(filters)

        # Sort control — relevance_score only valid when 'search' param is present
        has_search = "search" in params
        if sort_mode == "date":
            if has_search:
                params["sort"] = "publication_year:desc,relevance_score:desc"
            else:
                params["sort"] = "publication_year:desc"
        elif sort_mode == "citations":
            params["sort"] = "cited_by_count:desc"
        elif not has_search:
            # No search query (e.g. author-only): default to newest first
            params["sort"] = "publication_year:desc"

        params["per_page"] = min(n, 200)  # OA API max is 200
        params["select"] = (
            "id,doi,title,publication_year,cited_by_count,authorships,"
            "open_access,primary_topic,best_oa_location"
        )

        r = await client.get(
            "https://api.openalex.org/works",
            params=_oa_params(**params),
            timeout=SEARCH_TIMEOUT,
        )
        return _parse_oa_results(r.json().get("results", []))
    except Exception as e:
        _log(f"OpenAlex keyword search error: {e}")
        return []


async def _oa_semantic_search(
    client: httpx.AsyncClient,
    query: str,
    n: int,
    year_min: int = None,
    year_max: int = None,
) -> list:
    """OpenAlex AI semantic search via /find/works endpoint.

    Uses vector embeddings to find conceptually related works regardless of
    exact terminology. Requires API key. Costs 1000 credits/query.
    Finds papers even when they use completely different words for the same concept.
    """
    if not OPENALEX_API_KEY:
        _log("OpenAlex semantic search requires API key, falling back to keyword")
        return []

    try:
        params = {
            "query": query,
            "count": min(n, 100),
            "api_key": OPENALEX_API_KEY,
        }
        # Semantic search supports filter narrowing
        filters = []
        if year_min and year_max:
            filters.append(f"publication_year:{year_min}-{year_max}")
        elif year_min:
            filters.append(f"publication_year:>{year_min - 1}")
        elif year_max:
            filters.append(f"publication_year:<{year_max + 1}")

        if filters:
            params["filter"] = ",".join(filters)

        r = await client.get(
            "https://api.openalex.org/find/works",
            params=params,
            timeout=SEARCH_TIMEOUT + 10,  # semantic search can be slower
        )
        data = r.json()
        results = data.get("results", [])
        papers = _parse_oa_results(results)
        # Attach similarity scores from semantic ranking
        for i, p in enumerate(papers):
            if i < len(results) and results[i].get("relevance_score"):
                p["similarity"] = round(results[i]["relevance_score"], 3)
        return papers
    except Exception as e:
        _log(f"OpenAlex semantic search error: {e}")
        return []


def _parse_oa_results(results: list) -> list:
    """Parse OpenAlex work objects into standard paper format."""
    papers = []
    for w in results:
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        authors = []
        for a in w.get("authorships") or []:
            if not a:
                continue
            name = (a.get("author") or {}).get("display_name")
            if name:
                authors.append(name)

        # Extract arXiv ID from OpenAlex locations
        arxiv_id = None
        oa_id = (w.get("id") or "").replace("https://openalex.org/", "")
        best_loc = w.get("best_oa_location") or {}
        landing = best_loc.get("landing_page_url", "") or ""
        if "arxiv.org" in landing:
            match = re.search(r"(\d{4}\.\d{4,5})", landing)
            if match:
                arxiv_id = match.group(1)

        oa_info = w.get("open_access") or {}
        topic = w.get("primary_topic") or {}

        paper = {
            "title": w.get("title"),
            "authors": authors,
            "year": w.get("publication_year"),
            "citations": w.get("cited_by_count", 0),
            "doi": doi or None,
            "source": "openalex",
        }
        if arxiv_id:
            paper["arxiv_id"] = arxiv_id
        if oa_id:
            paper["openalex_id"] = oa_id
        if oa_info.get("is_oa"):
            paper["is_oa"] = True
        if topic.get("display_name"):
            paper["topic"] = topic["display_name"]

        papers.append(paper)
    return papers


async def _ads_search(
    client: httpx.AsyncClient,
    query: str,
    n: int,
    year_min: int = None,
    year_max: int = None,
    sort_mode: str = "relevance",
    author: str = None,
) -> list:
    """Search NASA ADS (astrophysics, physics, instrumentation, materials science)."""
    if not ADS_API_KEY:
        _log("NASA ADS search requires API key, skipping")
        return []

    try:
        # Build ADS query parts
        parts = []
        if author:
            # ADS uses author:"Last, First" or author:"Name"
            au = author.replace('\\"', '').replace('"', '').strip()
            parts.append(f'author:"{au}"')
        if query:
            parts.append(f"({query})")

        q = " ".join(parts) if parts else "*"

        if year_min or year_max:
            yr_start = year_min or 1900
            yr_end = year_max or 2100
            q = f"({q}) year:{yr_start}-{yr_end}"

        sort_map = {
            "relevance": "score desc",
            "date": "date desc",
            "citations": "citation_count desc",
        }

        r = await client.get(
            "https://api.adsabs.harvard.edu/v1/search/query",
            params={
                "q": q,
                "rows": n,
                "fl": "title,author,year,doi,identifier,citation_count,bibcode",
                "sort": sort_map.get(sort_mode, "score desc"),
            },
            headers=_ads_headers(),
            timeout=SEARCH_TIMEOUT,
        )
        papers = []
        for doc in r.json().get("response", {}).get("docs", []):
            arxiv_id = None
            for ident in doc.get("identifier", []):
                match = re.search(r"(\d{4}\.\d{4,5})", ident)
                if match:
                    arxiv_id = match.group(1)
                    break

            doi_list = doc.get("doi", [])
            doi = doi_list[0] if doi_list else None

            papers.append({
                "title": (doc.get("title") or [""])[0],
                "authors": doc.get("author", [])[:10],
                "year": doc.get("year"),
                "citations": doc.get("citation_count", 0),
                "doi": doi,
                "arxiv_id": arxiv_id,
                "bibcode": doc.get("bibcode"),
                "source": "ads",
            })
        return papers
    except Exception as e:
        _log(f"NASA ADS search error: {e}")
        return []


async def _s2_search(
    client: httpx.AsyncClient,
    query: str,
    n: int,
    year_min: int = None,
    year_max: int = None,
    author: str = None,
) -> list:
    """Search Semantic Scholar (200M+ papers, strong CS/STEM relevance ranking).
    Note: S2 has no native author filter — author name is prepended to query.
    """
    try:
        headers = {}
        if S2_API_KEY:
            headers["x-api-key"] = S2_API_KEY

        # S2 doesn't have native author filtering — combine into query
        search_q = ""
        if author:
            au = author.replace('\\"', '').replace('"', '').strip()
            search_q = au
        if query:
            search_q = f"{search_q} {query}".strip() if search_q else query
        if not search_q:
            return []

        params = {
            "query": search_q,
            "limit": min(n, 100),
            "fields": "title,authors,year,externalIds,citationCount",
        }
        # S2 year filter format: "2020-2024", "2020-", "-2024"
        if year_min and year_max:
            params["year"] = f"{year_min}-{year_max}"
        elif year_min:
            params["year"] = f"{year_min}-"
        elif year_max:
            params["year"] = f"-{year_max}"

        r = await client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            headers=headers,
            timeout=SEARCH_TIMEOUT,
        )
        # S2 has 1 req/sec limit — retry once on 429
        if r.status_code == 429:
            await asyncio.sleep(1.5)
            r = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
                headers=headers,
                timeout=SEARCH_TIMEOUT,
            )
        papers = []
        for item in r.json().get("data") or []:
            ext = item.get("externalIds") or {}
            authors = [a.get("name", "") for a in (item.get("authors") or [])[:10]]
            papers.append({
                "title": item.get("title"),
                "authors": authors,
                "year": item.get("year"),
                "citations": item.get("citationCount", 0),
                "doi": ext.get("DOI"),
                "arxiv_id": ext.get("ArXiv"),
                "pmid": ext.get("PubMed"),
                "source": "s2",
            })
        return papers
    except Exception as e:
        _log(f"Semantic Scholar search error: {e}")
        return []


async def _pm_search(client: httpx.AsyncClient, query: str, n: int, author: str = None) -> list:
    """Search PubMed."""
    try:
        # Build PubMed query with native author field
        parts = []
        if author:
            au = author.replace('\\"', '').replace('"', '').strip()
            parts.append(f'"{au}"[Author]')
        if query:
            parts.append(f"({query})")
        term = " AND ".join(parts) if parts else query or ""
        if not term:
            return []

        sr = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": term, "retmax": n, "retmode": "json"},
            timeout=SEARCH_TIMEOUT,
        )
        pmids = sr.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []
        sm = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
            timeout=SEARCH_TIMEOUT,
        )
        data = sm.json().get("result", {})
        papers = []
        for pid in pmids:
            p = data.get(pid, {})
            if not isinstance(p, dict):
                continue
            authors = [a.get("name", "") for a in p.get("authors", [])]
            year = None
            for date_field in ("pubdate", "epubdate", "sortpubdate"):
                d = p.get(date_field, "")
                m = re.search(r"(\d{4})", d)
                if m:
                    year = int(m.group(1))
                    break
            doi = None
            for art_id in p.get("articleids", []):
                if art_id.get("idtype") == "doi":
                    doi = art_id["value"]
                    break
            papers.append({
                "title": p.get("title", ""),
                "authors": authors,
                "year": year,
                "doi": doi,
                "pmid": pid,
                "source": "pubmed",
            })
        return papers
    except Exception as e:
        _log(f"PubMed search error: {e}")
        return []


async def _crossref_search(
    client: httpx.AsyncClient,
    query: str,
    n: int,
    year_min: int = None,
    year_max: int = None,
    author: str = None,
) -> list:
    """CrossRef search — 150M+ DOIs, near-complete for published works.
    Supports native author filtering and date ranges."""
    try:
        params = {
            "rows": min(n, 100),
            "select": "DOI,title,author,published-print,published-online,is-referenced-by-count",
            "mailto": MAILTO,
        }
        if query:
            params["query"] = query
        if author:
            params["query.author"] = author
        if not query and not author:
            return []

        # Year filtering via CrossRef filter syntax
        filters = []
        if year_min:
            filters.append(f"from-pub-date:{year_min}")
        if year_max:
            filters.append(f"until-pub-date:{year_max}")
        if filters:
            params["filter"] = ",".join(filters)

        # Sort
        params["sort"] = "published"
        params["order"] = "desc"

        r = await client.get(
            "https://api.crossref.org/works",
            params=params,
            timeout=SEARCH_TIMEOUT + 5,  # CrossRef can be slow
        )
        papers = []
        for item in r.json().get("message", {}).get("items", []):
            title = item.get("title", [""])[0] if isinstance(item.get("title"), list) else ""
            authors = []
            for a in (item.get("author") or [])[:10]:
                name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                if name:
                    authors.append(name)
            year = None
            # Try published-print first, then published-online
            for date_field in ("published-print", "published-online"):
                dp = (item.get(date_field) or {}).get("date-parts", [[]])
                if dp and dp[0]:
                    year = dp[0][0]
                    break
            papers.append({
                "title": title,
                "authors": authors,
                "year": year,
                "citations": item.get("is-referenced-by-count", 0),
                "doi": item.get("DOI"),
                "source": "crossref",
            })
        return papers
    except Exception as e:
        _log(f"CrossRef search error: {e}")
        return []


# =========================================================================
# ID-BASED AUTHOR SEARCH SOURCES
# =========================================================================


async def _oa_author_id_works(
    client: httpx.AsyncClient,
    oa_author_id: str,
    n: int,
    year_min: int = None,
    year_max: int = None,
) -> list:
    """Fetch works by OpenAlex Author ID with cursor pagination.
    This is far more complete than display_name.search for prolific authors.
    """
    try:
        filters = [f"authorships.author.id:{oa_author_id}"]
        if year_min and year_max:
            filters.append(f"publication_year:{year_min}-{year_max}")
        elif year_min:
            filters.append(f"publication_year:>{year_min - 1}")
        elif year_max:
            filters.append(f"publication_year:<{year_max + 1}")

        all_papers = []
        cursor = "*"
        per_page = min(n, 200)
        pages_fetched = 0
        max_pages = max(1, (n + per_page - 1) // per_page)  # Ceiling division

        while cursor and pages_fetched < max_pages:
            params = _oa_params(
                filter=",".join(filters),
                per_page=per_page,
                cursor=cursor,
                sort="publication_year:desc",
                select="id,doi,title,publication_year,cited_by_count,authorships,"
                       "open_access,primary_topic,best_oa_location",
            )
            r = await client.get(
                "https://api.openalex.org/works",
                params=params,
                timeout=SEARCH_TIMEOUT + 5,
            )
            if r.status_code != 200:
                _log(f"OA author-ID works: HTTP {r.status_code}")
                break
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            all_papers.extend(_parse_oa_results(results))
            # Mark source as author-ID-based
            for p in all_papers[-len(results):]:
                p["source"] = "openalex"
            cursor = data.get("meta", {}).get("next_cursor")
            pages_fetched += 1

        _log(f"OA author-ID works: {len(all_papers)} papers from {pages_fetched} pages")
        return all_papers[:n]
    except Exception as e:
        _log(f"OA author-ID works error: {e}")
        return []


async def _s2_author_papers(
    client: httpx.AsyncClient,
    s2_author_id: str,
    n: int,
    year_min: int = None,
    year_max: int = None,
) -> list:
    """Fetch papers by Semantic Scholar Author ID with offset pagination.
    Replaces the name-prepend-to-query approach with the proper author endpoint.
    """
    try:
        headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
        all_papers = []
        offset = 0
        batch = min(100, max(n, 100))  # Always fetch 100 per page for efficiency
        # When year filtering is active, scan more papers since many will be filtered
        scan_limit = n * 5 if (year_min or year_max) else n * 2

        while offset < scan_limit:
            r = await client.get(
                f"https://api.semanticscholar.org/graph/v1/author/{s2_author_id}/papers",
                params={
                    "fields": "title,authors,year,externalIds,citationCount",
                    "limit": batch,
                    "offset": offset,
                },
                headers=headers,
                timeout=SEARCH_TIMEOUT,
            )
            if r.status_code == 429:
                await asyncio.sleep(1.5)
                r = await client.get(
                    f"https://api.semanticscholar.org/graph/v1/author/{s2_author_id}/papers",
                    params={
                        "fields": "title,authors,year,externalIds,citationCount",
                        "limit": batch,
                        "offset": offset,
                    },
                    headers=headers,
                    timeout=SEARCH_TIMEOUT,
                )
            if r.status_code != 200:
                _log(f"S2 author papers: HTTP {r.status_code} at offset {offset}")
                break

            data = r.json()
            papers = data.get("data", [])
            if not papers:
                break

            for item in papers:
                ext = item.get("externalIds") or {}
                year = item.get("year")
                # Year filtering (S2 author endpoint has no native year filter)
                if year_min and year and year < year_min:
                    continue
                if year_max and year and year > year_max:
                    continue
                authors = [a.get("name", "") for a in (item.get("authors") or [])[:10]]
                all_papers.append({
                    "title": item.get("title"),
                    "authors": authors,
                    "year": year,
                    "citations": item.get("citationCount", 0),
                    "doi": ext.get("DOI"),
                    "arxiv_id": ext.get("ArXiv"),
                    "pmid": ext.get("PubMed"),
                    "source": "s2",
                })

            offset += len(papers)
            # S2's 'next' field is the offset for the next page
            next_val = data.get("next")
            if next_val is None:
                break  # No more pages
            # Early exit if we have enough year-matched papers
            if len(all_papers) >= n:
                break
            await asyncio.sleep(0.5)  # Rate limit courtesy

        _log(f"S2 author papers: {len(all_papers)} papers (offset {offset})")
        return all_papers[:n]
    except Exception as e:
        _log(f"S2 author papers error: {e}")
        return []


async def _orcid_works(
    client: httpx.AsyncClient,
    orcid_id: str,
    year_min: int = None,
    year_max: int = None,
) -> list:
    """Read works from an ORCID record. Returns DOI-identified papers.
    ORCID records are self-curated so coverage varies, but DOIs are reliable."""
    try:
        token = await _get_orcid_token(client)
        if not token:
            return []

        r = await client.get(
            f"https://pub.orcid.org/v3.0/{orcid_id}/works",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
            },
            timeout=SEARCH_TIMEOUT + 5,
        )
        if r.status_code != 200:
            _log(f"ORCID works: HTTP {r.status_code}")
            return []

        papers = []
        for group in r.json().get("group", []):
            summaries = group.get("work-summary", [])
            if not summaries:
                continue
            s = summaries[0]

            title_obj = s.get("title", {}).get("title", {})
            title = title_obj.get("value", "") if isinstance(title_obj, dict) else ""

            # Extract year
            pub_date = s.get("publication-date") or {}
            year = None
            if pub_date.get("year"):
                try:
                    year = int(pub_date["year"]["value"])
                except (ValueError, TypeError, KeyError):
                    pass

            # Year filter
            if year_min and year and year < year_min:
                continue
            if year_max and year and year > year_max:
                continue

            # Extract DOI
            ext_ids = s.get("external-ids", {}).get("external-id", [])
            doi = None
            for eid in ext_ids:
                if eid.get("external-id-type") == "doi":
                    doi = eid.get("external-id-value")
                    break

            if title:
                paper = {
                    "title": title,
                    "authors": [],  # ORCID summary doesn't include co-authors
                    "year": year,
                    "doi": doi,
                    "source": "orcid",
                }
                papers.append(paper)

        _log(f"ORCID works: {len(papers)} papers from {orcid_id}")
        return papers
    except Exception as e:
        _log(f"ORCID works error: {e}")
        return []


async def _openaire_search(
    client: httpx.AsyncClient,
    query: str,
    n: int,
    year_min: int = None,
    author: str = None,
    orcid: str = None,
) -> list:
    """Search OpenAIRE Research Graph (324M+ products, strong EU/repository coverage)."""
    try:
        params = {"format": "json", "size": min(n, 50)}  # OpenAIRE max page is 50

        # OpenAIRE name search is broader than ORCID search (73 vs 49 for Stratakis)
        # Use author name as primary, ORCID only if no name provided
        if author:
            params["author"] = author
        elif orcid:
            params["orcid"] = orcid
        if query:
            params["keywords"] = query
        if year_min:
            params["fromDateAccepted"] = f"{year_min}-01-01"

        if not params.get("orcid") and not params.get("author") and not params.get("keywords"):
            return []

        params["sortBy"] = "resultdateofacceptance,descending"

        r = await client.get(
            "https://api.openaire.eu/search/researchProducts",
            params=params,
            timeout=SEARCH_TIMEOUT + 10,  # OpenAIRE can be slow
        )
        if r.status_code != 200:
            _log(f"OpenAIRE: HTTP {r.status_code}")
            return []

        data = r.json()
        resp = data.get("response", {})
        results_obj = resp.get("results")
        if not results_obj:
            _log(f"OpenAIRE: no results")
            return []
        results = results_obj.get("result", [])
        if not results:
            return []

        papers = []
        for item in results:
            meta = item.get("metadata", {}).get("oaf:entity", {}).get("oaf:result", {})
            if not meta:
                continue

            # Title
            title = meta.get("title", {})
            if isinstance(title, list):
                title = title[0]
            if isinstance(title, dict):
                title = title.get("$", "")
            if not title:
                continue

            # Year
            date = meta.get("dateofacceptance", {})
            if isinstance(date, dict):
                date = date.get("$", "")
            year = None
            if date:
                m = re.search(r"(\d{4})", str(date))
                if m:
                    year = int(m.group(1))

            # DOI
            pids = meta.get("pid", [])
            if isinstance(pids, dict):
                pids = [pids]
            doi = None
            for pid in pids:
                if isinstance(pid, dict) and pid.get("@classid") == "doi":
                    doi = pid.get("$", "")
                    break

            # Authors
            authors_raw = meta.get("creator", [])
            if isinstance(authors_raw, dict):
                authors_raw = [authors_raw]
            authors = []
            for a in authors_raw[:10]:
                if isinstance(a, dict):
                    authors.append(a.get("$", ""))
                elif isinstance(a, str):
                    authors.append(a)

            papers.append({
                "title": str(title).strip(),
                "authors": authors,
                "year": year,
                "doi": doi,
                "source": "openaire",
            })

        _log(f"OpenAIRE: {len(papers)} papers")
        return papers
    except Exception as e:
        _log(f"OpenAIRE search error: {e}")
        return []


# =========================================================================
# OPENALEX BATCH DOI RESOLUTION
# =========================================================================


async def _oa_batch_doi_lookup(client: httpx.AsyncClient, dois: list[str]) -> list:
    """Batch-resolve up to 50 DOIs via OpenAlex pipe operator."""
    if not dois:
        return []
    try:
        chunks = [dois[i:i + 50] for i in range(0, len(dois), 50)]
        all_results = []
        for chunk in chunks:
            doi_filter = "|".join(f"https://doi.org/{d}" for d in chunk)
            r = await client.get(
                "https://api.openalex.org/works",
                params=_oa_params(
                    filter=f"doi:{doi_filter}",
                    per_page=50,
                    select="id,doi,title,publication_year,cited_by_count,authorships,"
                           "open_access,best_oa_location",
                ),
                timeout=SEARCH_TIMEOUT,
            )
            all_results.extend(_parse_oa_results(r.json().get("results", [])))
        return all_results
    except Exception as e:
        _log(f"OpenAlex batch DOI lookup error: {e}")
        return []


# =========================================================================
# INTERNAL DETAIL FUNCTIONS (Multi-Source Fallback)
# =========================================================================


async def _arxiv_details(client: httpx.AsyncClient, arxiv_id: str) -> Optional[dict]:
    try:
        r = await client.get(
            f"https://export.arxiv.org/api/query?id_list={arxiv_id}",
            timeout=DETAIL_TIMEOUT,
        )
        soup = BeautifulSoup(r.text, "xml")
        e = soup.find("entry")
        if e and e.title and "Error" not in e.title.text:
            return {
                "title": e.title.text.strip().replace("\n", " "),
                "authors": [a.find("name").text for a in e.find_all("author")],
                "year": e.published.text[:4] if e.published else None,
                "abstract": e.summary.text.strip() if e.summary else None,
                "arxiv_id": arxiv_id,
                "pdf": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "source": "arxiv",
            }
    except Exception as e:
        _log(f"arxiv details error: {e}")
    return None


async def _oa_details(client: httpx.AsyncClient, doi: str) -> Optional[dict]:
    """Get paper details from OpenAlex with field selection."""
    try:
        r = await client.get(
            f"https://api.openalex.org/works/doi:{doi}",
            params=_oa_params(
                select="id,doi,title,publication_year,cited_by_count,authorships,abstract_inverted_index,primary_location,open_access",
            ),
            timeout=DETAIL_TIMEOUT,
        )
        if r.status_code == 200:
            w = r.json()
            if w.get("title"):
                # Reconstruct abstract from inverted index
                abstract = None
                if w.get("abstract_inverted_index"):
                    inv = w["abstract_inverted_index"]
                    words = [(pos, word) for word, positions in inv.items() for pos in positions]
                    words.sort(key=lambda x: x[0])
                    abstract = " ".join(w for _, w in words)

                authors = []
                for a in w.get("authorships") or []:
                    if not a:
                        continue
                    name = (a.get("author") or {}).get("display_name")
                    if name:
                        authors.append(name)

                return {
                    "title": w.get("title"),
                    "authors": authors,
                    "year": w.get("publication_year"),
                    "citations": w.get("cited_by_count"),
                    "abstract": abstract,
                    "doi": doi,
                    "venue": (
                        (w.get("primary_location") or {}).get("source", {}).get("display_name")
                        if w.get("primary_location")
                        else None
                    ),
                    "open_access": (w.get("open_access") or {}).get("is_oa"),
                    "source": "openalex",
                }
    except Exception as e:
        _log(f"OpenAlex details error: {e}")
    return None


async def _crossref_details(client: httpx.AsyncClient, doi: str) -> Optional[dict]:
    try:
        r = await client.get(f"https://api.crossref.org/works/{doi}", timeout=DETAIL_TIMEOUT)
        if r.status_code == 200:
            w = r.json().get("message", {})
            if w.get("title"):
                authors = []
                for a in w.get("author", []):
                    name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                    if name:
                        authors.append(name)

                year = None
                for df in ["published-print", "published-online", "issued", "created"]:
                    parts = w.get(df, {}).get("date-parts", [[]])[0]
                    if parts and parts[0]:
                        year = parts[0]
                        break

                abstract = w.get("abstract", "")
                if abstract:
                    abstract = re.sub(r"<[^>]+>", "", abstract).strip()

                return {
                    "title": w["title"][0] if isinstance(w.get("title"), list) else w.get("title"),
                    "authors": authors,
                    "year": year,
                    "citations": w.get("is-referenced-by-count"),
                    "abstract": abstract or None,
                    "doi": doi,
                    "venue": w.get("container-title", [None])[0] if w.get("container-title") else None,
                    "source": "crossref",
                }
    except Exception as e:
        _log(f"CrossRef details error: {e}")
    return None


async def _pm_details(client: httpx.AsyncClient, pmid: str) -> Optional[dict]:
    try:
        r = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": pmid, "retmode": "xml"},
            timeout=DETAIL_TIMEOUT,
        )
        soup = BeautifulSoup(r.text, "xml")
        art = soup.find("PubmedArticle")
        if art:
            ab = art.find("Abstract")
            abstract = " ".join(t.text for t in ab.find_all("AbstractText")) if ab else None
            found_doi = next(
                (i.text for i in art.find_all("ArticleId") if i.get("IdType") == "doi"), None
            )
            title = art.find("ArticleTitle")
            journal = art.find("Journal")
            return {
                "title": title.text if title else None,
                "authors": [
                    a.find("LastName").text
                    for a in art.find_all("Author")
                    if a.find("LastName")
                ],
                "year": art.find("Year").text if art.find("Year") else None,
                "abstract": abstract,
                "pmid": pmid,
                "doi": found_doi,
                "venue": journal.find("Title").text if journal and journal.find("Title") else None,
                "source": "pubmed",
            }
    except Exception as e:
        _log(f"PubMed details error: {e}")
    return None


async def _get_paper_details(
    client: httpx.AsyncClient,
    doi: str = None,
    arxiv_id: str = None,
    pmid: str = None,
) -> dict:
    """Get paper details with optimized multi-source fallback.
    Priority: arXiv -> OpenAlex -> CrossRef -> PubMed
    """
    doi = _clean_doi(doi) if doi else None
    arxiv_id = _clean_arxiv(arxiv_id) if arxiv_id else None

    if arxiv_id:
        result = await _arxiv_details(client, arxiv_id)
        if result:
            return result

    if doi:
        # Try OpenAlex and CrossRef concurrently
        oa_res, cr_res = await asyncio.gather(
            _oa_details(client, doi),
            _crossref_details(client, doi),
            return_exceptions=True,
        )
        for result in (oa_res, cr_res):
            if isinstance(result, dict) and result.get("title"):
                return result

    if pmid:
        result = await _pm_details(client, pmid)
        if result:
            return result

    return {"error": "Paper not found", "tried": ["arxiv", "openalex", "crossref", "pubmed"]}



# =====================================================================
# PUBLISHER DIRECT-PDF RULES (institutional IP auth via HEAL-Link)
# =====================================================================
# DOI prefix -> (publisher_name, pdf_url_template)
# Template uses {doi} placeholder. Tried in order; first %PDF wins.
# These work when the server runs on an institutional network whose IP
# is registered with the publisher (e.g., FORTH via HEAL-Link).
# =====================================================================

PUBLISHER_PDF_RULES: dict[str, tuple[str, list[str]]] = {
    # Elsevier / ScienceDirect -- no template; PII must come from DOI resolution
    # Handled entirely in Strategy 2 (resolve DOI -> find /pdfft link)
    "10.1016": ("elsevier", []),
    # Springer Nature
    "10.1007": ("springer", [
        "https://link.springer.com/content/pdf/{doi}.pdf",
    ]),
    # Nature
    "10.1038": ("nature", [
        "https://www.nature.com/articles/{suffix}.pdf",
    ]),
    # Wiley
    "10.1002": ("wiley", [
        "https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}",
    ]),
    # ACS
    "10.1021": ("acs", [
        "https://pubs.acs.org/doi/pdf/{doi}",
    ]),
    # APS (Physical Review) -- uses _aps_pdf_url() for smart journal detection
    "10.1103": ("aps", []),   # templates generated dynamically below
    # IOP Science
    "10.1088": ("iop", [
        "https://iopscience.iop.org/article/{doi}/pdf",
    ]),
    # Taylor & Francis
    "10.1080": ("taylor_francis", [
        "https://www.tandfonline.com/doi/pdf/{doi}",
    ]),
    # RSC (Royal Society of Chemistry)
    "10.1039": ("rsc", []),   # complex URL structure, use Strategy 2
    # IEEE
    "10.1109": ("ieee", []),  # needs arnumber from DOI resolution
    # Optica / OSA
    "10.1364": ("optica", [
        "https://opg.optica.org/directpdfaccess/{doi}",
    ]),
    # AAAS / Science
    "10.1126": ("science", [
        "https://www.science.org/doi/pdf/{doi}",
    ]),
    # Oxford University Press
    "10.1093": ("oup", []),   # complex URL structure, use Strategy 2
    # SAGE
    "10.1177": ("sage", [
        "https://journals.sagepub.com/doi/pdf/{doi}",
    ]),
    # AIP (American Institute of Physics) -- uses _aip_pdf_url() for journal detection
    "10.1063": ("aip", []),   # templates generated dynamically below
    # PNAS
    "10.1073": ("pnas", [
        "https://www.pnas.org/doi/pdf/{doi}",
    ]),
    # Cambridge University Press
    "10.1017": ("cambridge", []),  # complex URL, use Strategy 2
    # MDPI (usually OA, but good to have)
    "10.3390": ("mdpi", [
        "https://www.mdpi.com/{doi_suffix}/pdf",
    ]),
}

# APS journal abbreviation -> URL slug mapping
_APS_JOURNAL_MAP = {
    "PhysRevLett": "prl",
    "PhysRevB": "prb",
    "PhysRevA": "pra",
    "PhysRevE": "pre",
    "PhysRevX": "prx",
    "PhysRevD": "prd",
    "PhysRevC": "prc",
    "PhysRevResearch": "prresearch",
    "PhysRevMaterials": "prmaterials",
    "PhysRevApplied": "prapplied",
    "PhysRevAccelBeams": "prab",
    "PhysRevFluids": "prfluids",
    "PhysRevPhysEducRes": "prper",
    "RevModPhys": "rmp",
}


def _aps_pdf_urls(doi: str) -> list[str]:
    """Generate APS PDF URL(s) from DOI, using the journal name embedded in it.
    e.g. 10.1103/PhysRevLett.133.150401 -> journals.aps.org/prl/pdf/...
    """
    suffix = doi.split("/", 1)[1] if "/" in doi else ""
    # Extract journal abbreviation from DOI suffix (first dot-separated token)
    journal_key = suffix.split(".")[0] if suffix else ""

    if journal_key in _APS_JOURNAL_MAP:
        slug = _APS_JOURNAL_MAP[journal_key]
        return [f"https://journals.aps.org/{slug}/pdf/{doi}"]
    # Unknown APS journal -- fall through to Strategy 2
    return []


def _aip_pdf_urls(doi: str) -> list[str]:
    """Generate AIP PDF URL. AIP's Silverchair platform uses a standard path."""
    # AIP Advances, J. Appl. Phys., Appl. Phys. Lett., Rev. Sci. Instrum., etc.
    # The correct journal slug can't be reliably determined from DOI alone,
    # so we use the journal-agnostic DOI-based URL that AIP supports
    return [f"https://pubs.aip.org/aip/adv/article-pdf/doi/{doi}"]

# PubMed Central -- free full-text for biomedical papers
PMC_API_BASE = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"


def _get_doi_prefix(doi: str) -> str:
    """Extract DOI prefix (e.g., '10.1038' from '10.1038/s41586-024-07891-2')."""
    parts = doi.split("/", 1)
    return parts[0] if len(parts) >= 2 else ""


async def _try_pmc_download(
    client: httpx.AsyncClient, doi: str, fpath: Path
) -> bool:
    """Try downloading from PubMed Central (free full-text archive)."""
    try:
        # Convert DOI to PMCID using new API endpoint
        r = await client.get(
            PMC_API_BASE,
            params={"ids": doi, "format": "json", "tool": "forthought", "email": MAILTO},
            timeout=10.0,
            follow_redirects=True,
        )
        if r.status_code != 200:
            return False
        records = r.json().get("records", [])
        if not records or "pmcid" not in records[0]:
            return False
        pmcid = records[0]["pmcid"]
        # Download PDF from PMC
        pdf_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"
        return await _download_file(client, pdf_url, fpath)
    except Exception as e:
        _log(f"PMC download error: {e}")
        return False


async def _try_elsevier_api(
    client: httpx.AsyncClient, doi: str, fpath: Path
) -> bool:
    """Try downloading Elsevier article via ScienceDirect API with API key.

    Uses the Article Retrieval API endpoint with httpAccept=application/pdf.
    Requires ELSEVIER_API_KEY; ELSEVIER_INSTTOKEN improves access scope.
    Falls back to /pdfft with API key as query param if API endpoint fails.
    """
    if not ELSEVIER_API_KEY:
        return False
    if not doi.startswith("10.1016/"):
        return False

    headers = {
        "X-ELS-APIKey": ELSEVIER_API_KEY,
        "Accept": "application/pdf",
    }
    if ELSEVIER_INSTTOKEN:
        headers["X-ELS-Insttoken"] = ELSEVIER_INSTTOKEN

    # Strategy 1: Direct API endpoint (returns PDF if entitled)
    try:
        api_url = f"https://api.elsevier.com/content/article/doi/{doi}"
        _log(f"Elsevier API: trying {doi}")
        r = await client.get(api_url, headers=headers, timeout=30.0)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            fpath.write_bytes(r.content)
            _log(f"Elsevier API: downloaded {len(r.content)} bytes -> {fpath.name}")
            return True
        elif r.status_code == 200:
            # Got XML/JSON instead of PDF -- API doesn't support PDF for this article
            _log(f"Elsevier API: got non-PDF response ({r.headers.get('content-type', '?')})")
        else:
            _log(f"Elsevier API: HTTP {r.status_code}")
    except Exception as e:
        _log(f"Elsevier API error: {e}")

    # Strategy 2: Resolve DOI to get PII, then hit /pdfft with API key
    try:
        r = await client.get(f"https://doi.org/{doi}", timeout=15.0, follow_redirects=True)
        if r.status_code != 200:
            return False
        pii_match = re.search(r'/pii/([A-Z0-9]+)', str(r.url))
        if pii_match:
            pii = pii_match.group(1)
            pdfft_url = (
                f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft"
                f"?APIKey={ELSEVIER_API_KEY}"
            )
            _log(f"Elsevier pdfft+key: PII={pii}")
            if await _download_file(client, pdfft_url, fpath):
                return True
    except Exception as e:
        _log(f"Elsevier pdfft+key error: {e}")

    return False


async def _try_direct_publisher(
    client: httpx.AsyncClient, doi: str, fpath: Path
) -> Optional[str]:
    """Try downloading PDF directly from publisher using institutional IP auth.
    Returns the publisher name on success, None on failure.

    Strategy 1: Match DOI prefix to known publisher PDF URL patterns
    Strategy 2: Resolve DOI -> follow redirects -> parse HTML for PDF link
    """
    prefix = _get_doi_prefix(doi)
    suffix = doi.split("/", 1)[1] if "/" in doi else ""

    # --- Strategy 1: Known publisher PDF URL patterns ---
    if prefix in PUBLISHER_PDF_RULES:
        pub_name, url_templates = PUBLISHER_PDF_RULES[prefix]

        # Dynamic URL generators for publishers with complex journal routing
        if prefix == "10.1103":       # APS
            url_templates_resolved = _aps_pdf_urls(doi)
        elif prefix == "10.1063":     # AIP
            url_templates_resolved = _aip_pdf_urls(doi)
        else:
            url_templates_resolved = []
            for template in url_templates:
                try:
                    url = template.format(
                        doi=doi,
                        suffix=suffix,
                        doi_suffix=suffix,
                    )
                    # Skip templates with empty placeholders
                    if "//" not in url.split("://", 1)[-1]:
                        url_templates_resolved.append(url)
                except (KeyError, IndexError):
                    continue

        if url_templates_resolved:
            _log(f"Direct publisher: trying {pub_name} for {doi}")
            for url in url_templates_resolved:
                if await _download_file(client, url, fpath):
                    return pub_name
            # All templates failed for this publisher -- fall to Strategy 2

    # --- Strategy 2: Resolve DOI and parse HTML for PDF links ---
    try:
        _log(f"Direct publisher: resolving DOI {doi} for PDF link")
        r = await client.get(
            f"https://doi.org/{doi}",
            timeout=20.0,
            follow_redirects=True,
        )
        if r.status_code != 200:
            return None

        final_url = str(r.url)
        html = r.text

        # --- Elsevier special handling: extract PII and try /pdfft ---
        if "sciencedirect.com" in final_url or "linkinghub.elsevier.com" in final_url:
            pii_match = re.search(r'/pii/([A-Z0-9]+)', final_url)
            if pii_match:
                pii = pii_match.group(1)
                pdfft_url = f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft"
                _log(f"  Elsevier PII={pii}, trying pdfft")
                if await _download_file(client, pdfft_url, fpath):
                    return "elsevier"

        # Build PDF candidate URLs from the resolved page
        pdf_candidates = []

        # Pattern: /abs/ -> /pdf/ or /full/ -> /pdf/
        for old, new in [("/abs/", "/pdf/"), ("/full/", "/pdf/"), ("/abstract/", "/pdf/")]:
            if old in final_url:
                pdf_candidates.append(final_url.replace(old, new))

        # Pattern: /doi/X -> /doi/pdf/X (Wiley, SAGE, T&F style)
        if "/doi/" in final_url and "/doi/pdf/" not in final_url:
            pdf_candidates.append(final_url.replace("/doi/", "/doi/pdf/", 1))

        # Pattern: append .pdf to URL
        if not final_url.endswith(".pdf"):
            pdf_candidates.append(final_url.rstrip("/") + ".pdf")

        # Pattern: /doi/reader/ -> /doi/pdf/
        for old_p in ["/doi/reader/", "/doi/epdf/", "/doi/epub/"]:
            if old_p in final_url:
                pdf_candidates.append(final_url.replace(old_p, "/doi/pdf/"))

        # Parse HTML meta tags for PDF URL (most reliable)
        from bs4 import BeautifulSoup as _BS
        soup = _BS(html, "html.parser")

        # <meta name="citation_pdf_url" content="...">
        meta_pdf = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if meta_pdf and meta_pdf.get("content"):
            pdf_candidates.insert(0, meta_pdf["content"])  # highest priority

        # <a> tags with PDF in href
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = (a.get_text(strip=True) or "").lower()
            if ("pdf" in text or href.endswith(".pdf")) and "doi" not in text:
                if href.startswith("/"):
                    parsed = urlparse(final_url)
                    href = f"{parsed.scheme}://{parsed.netloc}{href}"
                pdf_candidates.append(href)

        # Deduplicate while preserving order
        seen = set()
        unique_candidates = []
        for c in pdf_candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        # Try each candidate (capped)
        for pdf_url in unique_candidates[:5]:
            try:
                if await _download_file(client, pdf_url, fpath):
                    domain = urlparse(final_url).netloc
                    pub = domain.replace("www.", "").split(".")[0]
                    return f"direct:{pub}"
            except Exception:
                continue

    except Exception as e:
        _log(f"Direct publisher DOI resolve error: {e}")

    return None


# =========================================================================
# DOWNLOAD FUNCTIONS
# =========================================================================


async def _download_file(
    client: httpx.AsyncClient, url: str, path: Path, expected_type: str = "pdf"
) -> bool:
    """Download file with retry (only for transient errors) and content validation.

    4xx errors (403 Forbidden, 404 Not Found, etc.) are permanent -- never retried.
    Only 5xx / timeouts / connection errors get exponential backoff.
    """
    max_attempts = 3
    base_delay = 1.0
    for attempt in range(max_attempts):
        try:
            r = await client.get(url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True)

            # --- Permanent failures: don't retry ---
            if 400 <= r.status_code < 500:
                _log(f"HTTP {r.status_code} for {url} -- not retrying")
                return False
            r.raise_for_status()       # raises on 5xx (will retry)

            data = r.content

            # Minimum size check (real PDFs/ebooks > 50KB usually)
            if len(data) < 50000:
                lower_head = data[:1000].lower()
                if b"<!doctype" in lower_head or b"<html" in lower_head:
                    _log("Downloaded content is HTML, not a file")
                    return False        # HTML paywall page -- permanent
                if len(data) < 10000:
                    return False

            # Check for access-denied / CAPTCHA pages disguised as large HTML
            lower_head = data[:2000].lower()
            if any(marker in lower_head for marker in [
                b"access denied", b"captcha", b"please verify",
                b"institutional login", b"sign in to continue",
                b"subscription required", b"purchase this article",
                b"perfdrive", b"botmanager",  # Radware bot detection (IOP etc.)
            ]):
                _log("Detected access-denied/paywall page")
                return False            # paywall -- permanent

            # Magic byte validation
            is_valid = False
            start = data[:20]
            if expected_type == "pdf" or path.suffix.lower() == ".pdf":
                is_valid = start.startswith(b"%PDF")
                if is_valid and b"%%EOF" not in data[-1024:]:
                    _log("Warning: PDF missing %%EOF trailer (possibly truncated)")
            elif path.suffix.lower() == ".epub":
                is_valid = start.startswith(b"PK")
            elif path.suffix.lower() == ".djvu":
                is_valid = start.startswith(b"AT&TFORM")
            elif path.suffix.lower() in (".mobi", ".azw3"):
                is_valid = b"BOOKMOBI" in data[:100]
            else:
                is_valid = not (b"<!doctype" in lower_head or b"<html" in lower_head)

            if not is_valid:
                _log(f"Content validation failed for {path.suffix}")
                return False            # wrong content type -- permanent

            path.write_bytes(data)
            _log(f"Downloaded {len(data)} bytes -> {path.name}")
            return True

        except httpx.HTTPStatusError:
            # 5xx already raised -- retry with backoff
            _log(f"Server error (attempt {attempt + 1}/{max_attempts}) for {url}")
        except Exception as e:
            _log(f"Download error (attempt {attempt + 1}/{max_attempts}): {e}")

        if path.exists():
            path.unlink()
        if attempt < max_attempts - 1:
            delay = min(base_delay * (2 ** attempt), 30.0)
            delay *= (0.5 + random.random())  # jitter
            await asyncio.sleep(delay)
    return False


async def _download_paper(
    client: httpx.AsyncClient, arxiv_id: str = None, doi: str = None
) -> dict:
    """Download paper PDF with multi-source fallback.
    Returns dict with ok, url/filename, source, or error with fallback URL."""
    arxiv_id = _clean_arxiv(arxiv_id) if arxiv_id else None
    doi = _clean_doi(doi) if doi else None

    # Try arXiv (fastest, most reliable)
    if arxiv_id:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        fname = f"arxiv_{_sanitize(arxiv_id)}_{_ts()}.pdf"
        if await _download_file(client, url, DOWNLOAD_DIR / fname):
            return {"ok": True, "url": _pub_url(fname), "filename": fname, "source": "arxiv"}

    if doi:
        fname = f"doi_{_sanitize(doi)}_{_ts()}.pdf"
        fpath = DOWNLOAD_DIR / fname

        # Try OpenAlex Content API (60M cached OA PDFs)
        if OPENALEX_API_KEY:
            try:
                # First check if content exists via a quick metadata lookup
                oa_r = await client.get(
                    "https://api.openalex.org/works",
                    params=_oa_params(
                        filter=f"doi:https://doi.org/{doi}",
                        select="id,open_access",
                        per_page=1,
                    ),
                    timeout=10.0,
                )
                oa_results = oa_r.json().get("results", [])
                if oa_results:
                    oa_id = oa_results[0].get("id", "").replace("https://openalex.org/", "")
                    if oa_id:
                        content_url = (
                            f"https://content.openalex.org/works/{oa_id}.pdf"
                            f"?api_key={OPENALEX_API_KEY}"
                        )
                        if await _download_file(client, content_url, fpath):
                            return {
                                "ok": True, "url": _pub_url(fname),
                                "filename": fname, "source": "openalex_content",
                            }
            except Exception as e:
                _log(f"OpenAlex content download error: {e}")

        # Try Unpaywall (legal open access)
        try:
            r = await client.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": MAILTO},
                timeout=15.0,
            )
            if r.status_code == 200:
                pdf_url = (r.json().get("best_oa_location") or {}).get("url_for_pdf")
                if pdf_url and await _download_file(client, pdf_url, fpath):
                    return {"ok": True, "url": _pub_url(fname), "filename": fname, "source": "unpaywall"}
        except Exception as e:
            _log(f"Unpaywall error: {e}")

        # Try Elsevier API (dedicated API key auth, only for 10.1016/* DOIs)
        if await _try_elsevier_api(client, doi, fpath):
            return {"ok": True, "url": _pub_url(fname), "filename": fname, "source": "elsevier_api"}

        # Try Direct Publisher (institutional cookies + IP auth via HEAL-Link)
        pub_name = await _try_direct_publisher(client, doi, fpath)
        if pub_name:
            return {"ok": True, "url": _pub_url(fname), "filename": fname, "source": pub_name}

        # Try PubMed Central (free full-text for biomedical papers)
        if await _try_pmc_download(client, doi, fpath):
            return {"ok": True, "url": _pub_url(fname), "filename": fname, "source": "pmc"}

        # Try Sci-Hub mirrors (skip dead domains)
        for mirror in ["https://sci-hub.st", "https://sci-hub.ru"]:
            try:
                r = await client.get(f"{mirror}/{doi}", timeout=30.0)
                if r.status_code != 200:
                    continue
                soup = BeautifulSoup(r.text, "html.parser")
                pdf_url = None

                for tag in soup.find_all(["embed", "iframe", "object"]):
                    src = tag.get("src", "") or tag.get("data", "")
                    if ".pdf" in src or "/pdf" in src:
                        pdf_url = src
                        break

                if not pdf_url:
                    button = soup.find("button", onclick=True)
                    if button:
                        match = re.search(
                            r"location\.href\s*=\s*['\"]([^'\"]+)['\"]",
                            button.get("onclick", ""),
                        )
                        if match:
                            pdf_url = match.group(1)

                if pdf_url:
                    if pdf_url.startswith("//"):
                        pdf_url = "https:" + pdf_url
                    elif pdf_url.startswith("/"):
                        pdf_url = f"{mirror}{pdf_url}"
                    if await _download_file(client, pdf_url, fpath):
                        return {
                            "ok": True,
                            "url": _pub_url(fname),
                            "filename": fname,
                            "source": mirror.split("//")[1],
                        }
            except Exception as e:
                _log(f"Sci-Hub {mirror} error: {e}")
                continue

        return {
            "ok": False,
            "doi": doi,
            "fallback_url": f"https://doi.org/{doi}",
            "error": "Could not download -- paywalled or unavailable",
        }

    return {"ok": False, "error": "No arxiv_id or doi provided"}


# =========================================================================
# ZIP HELPER
# =========================================================================


def _zip_files(file_paths: list[Path], zip_name: str) -> Optional[Path]:
    """Zip a list of files into a single archive. Returns path or None."""
    if not file_paths:
        return None
    zip_path = DOWNLOAD_DIR / zip_name
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for fp in file_paths:
                if fp.exists():
                    zf.write(fp, fp.name)
        _log(f"Created zip: {zip_name} ({len(file_paths)} files)")
        return zip_path
    except Exception as e:
        _log(f"Zip error: {e}")
        return None


# =========================================================================
# REFERENCE RESOLUTION (CrossRef query.bibliographic)
# =========================================================================


async def _resolve_one_ref(
    client: httpx.AsyncClient, ref_text: str, sem: asyncio.Semaphore
) -> dict:
    """Resolve a raw citation string to a DOI using CrossRef."""
    async with sem:
        try:
            r = await client.get(
                "https://api.crossref.org/works",
                params={
                    "query.bibliographic": ref_text,
                    "rows": 1,
                    "select": "DOI,title,author,published-print,score",
                    "mailto": MAILTO,
                },
                timeout=DETAIL_TIMEOUT,
            )
            if r.status_code == 200:
                items = r.json().get("message", {}).get("items", [])
                if items:
                    item = items[0]
                    score = item.get("score", 0)
                    doi = item.get("DOI", "")
                    title = item.get("title", [""])[0] if isinstance(item.get("title"), list) else ""
                    authors = []
                    for a in item.get("author", [])[:3]:
                        name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                        if name:
                            authors.append(name)

                    # Confidence based on CrossRef relevance score
                    # >100 is very high confidence, >50 is decent, <20 is dubious
                    confidence = "high" if score > 100 else ("medium" if score > 40 else "low")

                    return {
                        "input": ref_text[:120],
                        "doi": doi,
                        "title": title,
                        "first_author": authors[0] if authors else None,
                        "confidence": confidence,
                        "score": round(score, 1),
                    }
        except Exception as e:
            _log(f"CrossRef resolve error: {e}")

        return {"input": ref_text[:120], "doi": None, "error": "Could not resolve"}


# =========================================================================
# SCORING & RANKING
# =========================================================================


def _score_paper(paper: dict, query: str = "", current_year: int = 2026) -> float:
    """Composite score for ranking deduplicated papers.

    Weights shift based on whether a topic query is present:
      With topic:  relevance=0.35, recency=0.20, citations=0.20, source_count=0.25
      Author-only: relevance=0.00, recency=0.40, citations=0.30, source_count=0.30
    """
    has_query = bool(query and query.strip())

    # --- Relevance: word overlap between query and title ---
    relevance = 0.0
    if has_query:
        q_words = set(re.findall(r'\w{3,}', query.lower()))
        t_words = set(re.findall(r'\w{3,}', (paper.get("title") or "").lower()))
        if q_words and t_words:
            relevance = len(q_words & t_words) / len(q_words)

    # --- Recency: linear 0→1, current_year=1.0, 10 years ago=0.0 ---
    year = int(paper.get("year") or 0)
    if year:
        age = max(current_year - year, 0)
        recency = max(0.0, 1.0 - age / 10.0)
    else:
        recency = 0.0

    # --- Citations: log-scaled ---
    cites = paper.get("citations") or 0
    citation_score = min(1.0, math.log(1 + cites) / 6.0)  # ~400 cites = 1.0

    # --- Source agreement: how many sources found this paper ---
    source_count = paper.get("_source_count", 1)
    source_score = min(1.0, (source_count - 1) / 2.0)  # 1 source=0, 3+=1.0

    # --- Weighted composite ---
    if has_query:
        return (0.35 * relevance + 0.20 * recency
                + 0.20 * citation_score + 0.25 * source_score)
    else:
        return (0.40 * recency + 0.30 * citation_score + 0.30 * source_score)


def _pool_summary(all_papers: list, unique: list, sources_queried: list,
                  returned_count: int, year_min: int = None, year_max: int = None) -> dict:
    """Generate a statistical summary of the full search pool."""
    years = [int(p["year"]) for p in unique if p.get("year")]
    source_counts = {}
    for p in unique:
        src = p.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    summary = {
        "raw_results": len(all_papers),
        "unique_after_dedup": len(unique),
        "returned": returned_count,
        "sources": sources_queried,
        "per_source": source_counts,
    }
    if years:
        summary["year_range"] = f"{min(years)}-{max(years)}"
    if year_min or year_max:
        summary["year_filter"] = (
            f"{year_min or '...'}-{year_max or '...'}"
        )
    if len(unique) > returned_count:
        summary["note"] = (
            f"Showing top {returned_count} of {len(unique)} unique papers. "
            f"Ask for more with max_results or narrow with year/topic filters."
        )
    return summary


# =========================================================================
# MCP TOOLS
# =========================================================================


@mcp.tool()
async def search(
    query: str,
    max_results: int = SEARCH_MAX_DEFAULT,
    year_min: int = None,
    year_max: int = None,
    sort: str = "relevance",
    mode: str = "auto",
    author: str = None,
) -> str:
    """
    Search academic papers across arXiv, OpenAlex, NASA ADS, Semantic Scholar,
    PubMed, CrossRef, OpenAIRE, and ORCID.
    Returns a lean list: title, first_author, year, identifiers, download hint.
    Use details() for abstracts. Use download() with the arxiv/doi identifiers.

    When an author name is provided, the server first resolves it to canonical IDs
    (OpenAlex Author ID, S2 Author ID, ORCID), then uses ID-based endpoints for
    exhaustive coverage. This finds 5-10x more papers than name-string search.

    Args:
        query: Search terms, boolean expressions, or natural language descriptions.
               Supports: AND, OR, NOT (uppercase), "exact phrases" in quotes.
               For author search, use the 'author' parameter instead of embedding in query.
               Examples: "femtosecond laser ablation silicon"
                         "XRD OR X-ray diffraction AND thin film"
        max_results: Max papers to return (default 8, max 100). Server over-fetches
                    from all sources, deduplicates, ranks, then returns the top N.
                    For author searches, try max_results=30-50 for comprehensive results.
        year_min: Filter papers published on or after this year (e.g. 2020)
        year_max: Filter papers published on or before this year (e.g. 2024)
        sort: Ranking mode - "relevance" (default), "date" (newest first), "citations"
        mode: Search strategy - "auto" (default), "semantic", "keyword"
              auto: uses semantic when query is natural language, keyword otherwise
              semantic: AI embedding similarity (finds related concepts, not just keywords)
              keyword: traditional text matching with boolean support
        author: Author name to filter by (e.g. "Emmanuel Stratakis", "E. Stratakis").
                Uses native author-filtering per source for accurate results.
                Can be combined with query for topic+author search, or used alone.

    Returns: JSON with papers array, pool_summary, and metadata
    """
    max_results = min(max_results, 100)

    # Extract author from query if embedded as prefix (model often does this)
    if not author:
        # Match patterns: author:"Name" or author:'Name' or author:Name
        m = re.match(
            r'''^\s*author\s*:\s*(?:"([^"]+)"|'([^']+)'|(\S+))\s*(.*)$''',
            query, re.IGNORECASE,
        )
        if m:
            author = m.group(1) or m.group(2) or m.group(3)
            query = m.group(4).strip()
            # Clean leftover boolean operators at start
            query = re.sub(r'^\s*(AND|OR)\s+', '', query).strip()

    _log(f"search: {query or '(author-only)'} (max={max_results}, mode={mode}, sort={sort}"
         f"{f', author={author}' if author else ''})")
    client = await get_client()

    # --- Author identity resolution (v5.7) ---
    # Resolve once, then use ID-based endpoints for exhaustive coverage
    author_ids = None
    if author:
        author_ids = await _resolve_author(client, author)

    # --- Over-fetch strategy ---
    # When author is set, we want completeness: fetch more per source, include CrossRef.
    # For topic-only searches, keep limits lean for speed.
    if author:
        fetch_n = max(max_results * 3, 50)  # overfetch 3x or minimum 50
    else:
        fetch_n = max(max_results * 2, 20)

    # Determine search mode
    if mode == "auto":
        if not query:
            mode = "keyword"
            if sort == "relevance" and author:
                sort = "date"
        else:
            has_boolean = any(f" {op} " in query for op in ("AND", "OR", "NOT"))
            has_quotes = '"' in query
            is_long = len(query.split()) >= 6
            mode = "semantic" if (is_long and not has_boolean and not has_quotes) else "keyword"
        _log(f"  auto-detected mode: {mode}")

    sources_queried = []
    coros = []

    # --- ID-based author queries (v5.7, preferred when IDs available) ---
    use_id_path = author and author_ids and (author_ids.get("oa_id") or author_ids.get("s2_id"))

    if use_id_path:
        _log(f"  using ID-based author path")

        # OpenAlex by Author ID (cursor-paginated, most complete)
        if author_ids.get("oa_id"):
            coros.append(
                _oa_author_id_works(client, author_ids["oa_id"], fetch_n, year_min, year_max)
            )
            sources_queried.append("openalex_id")
        else:
            # Fallback to name-based OA search
            coros.append(
                _oa_keyword_search(client, query, fetch_n, year_min, year_max, sort, author=author)
            )
            sources_queried.append("openalex")

        # Semantic Scholar by Author ID + name-based fallback
        # S2 often fragments authors into multiple IDs, so we run BOTH:
        # - Author ID endpoint for the main corpus (highest paper count)
        # - Name-based search to catch papers under split profiles
        if author_ids.get("s2_id"):
            coros.append(
                _s2_author_papers(client, author_ids["s2_id"], fetch_n, year_min, year_max)
            )
            sources_queried.append("s2_id")
        # Always also run name-based S2 search (catches split profiles + recent papers)
        coros.append(
            _s2_search(client, query, min(fetch_n // 2, 100), year_min, year_max, author=author)
        )
        sources_queried.append("s2")

        # ORCID works (supplementary DOI source)
        if author_ids.get("orcid"):
            coros.append(
                _orcid_works(client, author_ids["orcid"], year_min, year_max)
            )
            sources_queried.append("orcid")

        # OpenAIRE (strong for EU/FORTH, uses ORCID or name)
        coros.append(
            _openaire_search(
                client, query, fetch_n // 2, year_min,
                author=author, orcid=author_ids.get("orcid"),
            )
        )
        sources_queried.append("openaire")

        # Still run arXiv, ADS, PubMed, CrossRef with name-based (they don't have ID endpoints)
        coros.append(_arxiv_search(client, query, min(fetch_n, 50), author=author))
        sources_queried.append("arxiv")
        coros.append(_ads_search(client, query, fetch_n // 2, year_min, year_max, sort, author=author))
        sources_queried.append("ads")
        coros.append(_pm_search(client, query, fetch_n // 2, author=author))
        sources_queried.append("pubmed")
        coros.append(
            _crossref_search(client, query, fetch_n, year_min, year_max, author=author)
        )
        sources_queried.append("crossref")

    else:
        # --- Standard name/keyword path (v5.6 behavior) ---
        coros = [
            _oa_keyword_search(client, query, fetch_n, year_min, year_max, sort, author=author),
            _arxiv_search(client, query, min(fetch_n, 50), author=author),
            _ads_search(client, query, fetch_n // 2, year_min, year_max, sort, author=author),
            _s2_search(client, query, min(fetch_n // 2, 100), year_min, year_max, author=author),
            _pm_search(client, query, fetch_n // 2, author=author),
        ]
        sources_queried = ["openalex", "arxiv", "ads", "s2", "pubmed"]

        # CrossRef: always primary for author searches, fallback otherwise
        if author:
            coros.append(
                _crossref_search(client, query, fetch_n, year_min, year_max, author=author),
            )
            sources_queried.append("crossref")

        # OpenAIRE for author searches even without ID resolution
        if author:
            coros.append(
                _openaire_search(client, query, fetch_n // 2, year_min, author=author)
            )
            sources_queried.append("openaire")

    if mode == "semantic" and OPENALEX_API_KEY:
        coros.append(
            _oa_semantic_search(client, query, max_results, year_min, year_max),
        )
        sources_queried.append("openalex_semantic")

    _log(f"  fetching up to {fetch_n}/source from {len(coros)} sources")
    results = await asyncio.gather(*coros, return_exceptions=True)

    all_papers = []
    for r in results:
        if isinstance(r, list):
            all_papers.extend(r)

    # CrossRef fallback for non-author searches with few results
    if not author and len(all_papers) < 3:
        try:
            cr = await _crossref_search(client, query, max_results)
            all_papers.extend(cr)
            sources_queried.append("crossref")
        except Exception:
            pass

    total_raw = len(all_papers)

    # --- Year filtering (for sources without native support) ---
    if year_min or year_max:
        filtered = []
        for p in all_papers:
            yr = p.get("year")
            if yr is None:
                filtered.append(p)
                continue
            yr = int(yr)
            if year_min and yr < year_min:
                continue
            if year_max and yr > year_max:
                continue
            filtered.append(p)
        all_papers = filtered

    # --- Dedup with source tracking ---
    # Track how many sources found each paper (quality signal for ranking)
    unique = []
    seen_doi = {}     # doi_key -> index in unique
    seen_arxiv = {}   # arxiv_id -> index in unique
    for p in all_papers:
        if not p.get("title"):
            continue

        merged = False

        # DOI dedup
        doi = p.get("doi")
        if doi:
            doi_key = doi.lower().strip()
            if doi_key in seen_doi:
                idx = seen_doi[doi_key]
                existing = unique[idx]
                existing["_source_count"] = existing.get("_source_count", 1) + 1
                existing["_sources"] = existing.get("_sources", set())
                existing["_sources"].add(p.get("source", ""))
                # Merge missing fields
                if not existing.get("arxiv_id") and p.get("arxiv_id"):
                    existing["arxiv_id"] = p["arxiv_id"]
                if not existing.get("citations") and p.get("citations"):
                    existing["citations"] = p["citations"]
                if not existing.get("topic") and p.get("topic"):
                    existing["topic"] = p["topic"]
                if not existing.get("bibcode") and p.get("bibcode"):
                    existing["bibcode"] = p["bibcode"]
                if not existing.get("pmid") and p.get("pmid"):
                    existing["pmid"] = p["pmid"]
                merged = True
            else:
                seen_doi[doi_key] = len(unique)

        # arXiv ID dedup
        arxiv = p.get("arxiv_id")
        if arxiv and not merged:
            if arxiv in seen_arxiv:
                idx = seen_arxiv[arxiv]
                existing = unique[idx]
                existing["_source_count"] = existing.get("_source_count", 1) + 1
                existing["_sources"] = existing.get("_sources", set())
                existing["_sources"].add(p.get("source", ""))
                if not existing.get("doi") and doi:
                    existing["doi"] = doi
                merged = True
            else:
                seen_arxiv[arxiv] = len(unique)

        # Fuzzy title dedup (Jaccard)
        if not merged:
            title = p.get("title", "")
            for i, existing in enumerate(unique):
                if _titles_match(title, existing.get("title", "")):
                    existing["_source_count"] = existing.get("_source_count", 1) + 1
                    existing["_sources"] = existing.get("_sources", set())
                    existing["_sources"].add(p.get("source", ""))
                    if not existing.get("doi") and doi:
                        existing["doi"] = doi
                    if not existing.get("arxiv_id") and arxiv:
                        existing["arxiv_id"] = arxiv
                    merged = True
                    break

        if not merged:
            p["_source_count"] = 1
            p["_sources"] = {p.get("source", "")}
            unique.append(p)

    _log(f"  pool: {total_raw} raw -> {len(all_papers)} after year filter -> {len(unique)} unique")

    # --- Ranking ---
    current_year = datetime.now().year
    if sort == "date":
        unique.sort(key=lambda p: int(p.get("year") or 0), reverse=True)
    elif sort == "citations":
        unique.sort(key=lambda p: p.get("citations") or 0, reverse=True)
    else:
        # Composite scoring
        for p in unique:
            p["_score"] = _score_paper(p, query, current_year)
        unique.sort(key=lambda p: p["_score"], reverse=True)

    # --- Pool summary (full picture for the model) ---
    returned_count = min(max_results, len(unique))
    summary = _pool_summary(all_papers, unique, sources_queried,
                            returned_count, year_min, year_max)

    # Add author resolution info to summary (v5.7)
    if author_ids:
        resolution = {}
        if author_ids.get("oa_id"):
            resolution["openalex_id"] = author_ids["oa_id"]
        if author_ids.get("s2_id"):
            resolution["s2_id"] = author_ids["s2_id"]
        if author_ids.get("orcid"):
            resolution["orcid"] = author_ids["orcid"]
        if resolution:
            summary["author_resolution"] = resolution

    # --- Slim output ---
    papers = [_slim_search_result(p) for p in unique[:returned_count]]

    # Clean internal fields
    for p in unique:
        p.pop("_source_count", None)
        p.pop("_sources", None)
        p.pop("_score", None)

    return json.dumps({
        "papers": papers,
        "pool_summary": summary,
    })


@mcp.tool()
async def details(
    doi: str = None, arxiv_id: str = None, pmid: str = None, identifier: str = None, compact: bool = False
) -> str:
    """
    Get full paper details including abstract from best available source.
    Only call this when the user needs abstracts -- search() is sufficient for listing.

    Args:
        doi: DOI (e.g. "10.1038/nature14539")
        arxiv_id: arXiv ID (e.g. "1706.03762")
        pmid: PubMed ID
        identifier: Alternative -- pass any identifier string (auto-detected)
        compact: If true, truncate abstract to first sentence, cap authors at 3

    Returns: JSON with title, authors, year, abstract, citations, venue
    """
    # Auto-detect identifier type if not explicitly provided
    if not doi and not arxiv_id and not pmid and identifier:
        identifier = identifier.strip()
        cleaned = _clean_doi(identifier)
        if cleaned.startswith("10."):
            doi = cleaned
        elif re.match(r"^\d{4}\.\d{4,5}$", identifier) or "arxiv" in identifier.lower():
            arxiv_id = _clean_arxiv(identifier)
        elif identifier.isdigit():
            pmid = identifier
        elif "/" in cleaned:
            doi = cleaned
    _log(f"details: doi={doi} arxiv={arxiv_id} pmid={pmid}")
    client = await get_client()
    result = await _get_paper_details(client, doi, arxiv_id, pmid)
    if compact:
        result = _compact_paper(result)
    return json.dumps(result)


@mcp.tool()
async def download(arxiv_id: str = None, doi: str = None, identifier: str = None) -> str:
    """
    Download a single paper PDF. Tries arXiv -> Unpaywall -> Elsevier API -> Direct Publisher (cookies+IP) -> PMC -> Sci-Hub.
    On institutional networks (HEAL-Link), direct publisher access succeeds for most paywalled content.
    Prefer arxiv_id when available

    Args:
        arxiv_id: arXiv ID (preferred)
        doi: DOI identifier
        identifier: Alternative -- pass any identifier string (auto-detected as DOI or arXiv)

    Returns: JSON with ok, url (download link), or error with fallback_url for manual access
    """
    # Auto-detect identifier type if doi/arxiv_id not explicitly provided
    if not doi and not arxiv_id and identifier:
        identifier = identifier.strip()
        cleaned = _clean_doi(identifier)
        if cleaned.startswith("10."):
            doi = cleaned
        elif re.match(r"^\d{4}\.\d{4,5}$", identifier) or "arxiv" in identifier.lower():
            arxiv_id = _clean_arxiv(identifier)
        elif "/" in cleaned:
            doi = cleaned
        else:
            arxiv_id = identifier
    _log(f"download: arxiv={arxiv_id} doi={doi}")
    client = await get_client()
    result = await _download_paper(client, arxiv_id, doi)
    # Strip filename from single download response (model doesn't need it)
    result.pop("filename", None)
    return json.dumps(result)


@mcp.tool()
async def batch_details(identifiers: list[dict], compact: bool = True) -> str:
    """
    Get details for multiple papers. Compact mode is ON by default to save context.

    Args:
        identifiers: List of dicts: [{"doi": "..."}, {"arxiv_id": "..."}, {"pmid": "..."}]
                     Also accepts: [{"identifier": "10.1000/xyz"}], or mixed key names.
        compact: Truncate abstracts, cap authors (default true)

    Returns: JSON array of paper details (or error per paper)
    """
    # ---- Normalize input: accept various key names ----
    normalized = []
    for item in identifiers:
        if isinstance(item, str):
            item = {"identifier": item}
        if not isinstance(item, dict):
            continue
        if item.get("doi") or item.get("arxiv_id") or item.get("pmid"):
            normalized.append(item)
            continue
        raw = (
            item.get("identifier")
            or item.get("id")
            or item.get("DOI")
            or item.get("paper_id")
            or ""
        )
        if not raw:
            for v in item.values():
                if isinstance(v, str) and v.strip():
                    raw = v.strip()
                    break
        raw = raw.strip()
        if not raw:
            continue
        cleaned = _clean_doi(raw)
        if cleaned.startswith("10."):
            normalized.append({"doi": cleaned})
        elif re.match(r"^\d{4}\.\d{4,5}$", raw) or "arxiv" in raw.lower():
            normalized.append({"arxiv_id": _clean_arxiv(raw)})
        else:
            normalized.append({"doi": cleaned} if "/" in cleaned else {"arxiv_id": raw})
    identifiers = normalized

    _log(f"batch_details: {len(identifiers)} papers, compact={compact}")
    client = await get_client()
    sem = asyncio.Semaphore(BATCH_SEMAPHORE)

    async def get_one(ident: dict) -> dict:
        async with sem:
            try:
                return await _get_paper_details(
                    client,
                    doi=ident.get("doi"),
                    arxiv_id=ident.get("arxiv_id"),
                    pmid=ident.get("pmid"),
                )
            except Exception as e:
                return {"error": str(e), "identifier": ident}

    results = await asyncio.gather(
        *[get_one(i) for i in identifiers], return_exceptions=True
    )

    output = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            output.append({"error": str(result), "identifier": identifiers[i]})
        elif compact and isinstance(result, dict):
            output.append(_compact_paper(result))
        else:
            output.append(result)

    return json.dumps({"results": output})


@mcp.tool()
async def batch_download(papers: list[dict]) -> str:
    """
    Download any number of papers and package them in a single ZIP file.
    Handles concurrency, deduplication, and cleanup automatically.

    Args:
        papers: List of dicts: [{"arxiv_id": "1706.03762"}, {"doi": "10.1000/xyz"}]
                Also accepts: [{"identifier": "10.1000/xyz"}], ["10.1000/xyz"], or mixed.
                No hard limit -- concurrency is managed internally via semaphore.

    Returns: JSON with zip_url (single download), downloaded count, and failed list with fallback URLs
    """
    if not papers:
        return json.dumps({"error": "Empty papers list"})

    # ---- Normalize input: accept strings, dicts with various key names ----
    normalized = []
    for item in papers:
        if isinstance(item, str):
            item = {"identifier": item}
        if not isinstance(item, dict):
            continue
        # Already has doi or arxiv_id -- keep as-is
        if item.get("doi") or item.get("arxiv_id"):
            normalized.append(item)
            continue
        # Try common alternative keys
        raw = (
            item.get("identifier")
            or item.get("id")
            or item.get("DOI")
            or item.get("paper_id")
            or ""
        )
        if not raw:
            # Last resort: grab first string value from the dict
            for v in item.values():
                if isinstance(v, str) and v.strip():
                    raw = v.strip()
                    break
        raw = raw.strip()
        if not raw:
            continue
        # Classify as DOI or arXiv ID
        cleaned = _clean_doi(raw)
        if cleaned.startswith("10."):
            normalized.append({"doi": cleaned})
        elif re.match(r"^\d{4}\.\d{4,5}$", raw) or "arxiv" in raw.lower():
            normalized.append({"arxiv_id": _clean_arxiv(raw)})
        else:
            # Might be a DOI without the 10. prefix visible, or something else
            normalized.append({"doi": cleaned} if "/" in cleaned else {"arxiv_id": raw})
    papers = normalized
    if not papers:
        return json.dumps({"error": "No valid identifiers found in input"})

    # Soft cap for sanity -- 50 papers is ~2-4 min with semaphore=8
    papers = papers[:50]

    # Deduplicate by DOI/arxiv_id to avoid downloading the same paper twice
    seen = set()
    unique_papers = []
    for p in papers:
        key = p.get("doi") or p.get("arxiv_id") or ""
        if key and key not in seen:
            seen.add(key)
            unique_papers.append(p)
        elif not key:
            unique_papers.append(p)  # keep entries without identifiers
    papers = unique_papers

    total = len(papers)
    _log(f"batch_download: {total} papers (deduped)")
    client = await get_client()
    sem = asyncio.Semaphore(DOWNLOAD_SEMAPHORE)

    completed = 0

    async def dl_one(idx: int, paper: dict) -> dict:
        nonlocal completed
        async with sem:
            result = await _download_paper(
                client,
                arxiv_id=paper.get("arxiv_id"),
                doi=paper.get("doi"),
            )
            completed += 1
            if completed % 5 == 0 or completed == total:
                _log(f"  progress: {completed}/{total} downloads complete")
            return result

    results = await asyncio.gather(
        *[dl_one(i, p) for i, p in enumerate(papers)],
        return_exceptions=True,
    )

    succeeded = []
    failed = []
    downloaded_files = []

    for i, result in enumerate(results):
        identifier = papers[i].get("doi") or papers[i].get("arxiv_id") or "unknown"
        if isinstance(result, Exception):
            failed.append({
                "identifier": identifier,
                "fallback_url": f"https://doi.org/{papers[i]['doi']}" if papers[i].get("doi") else None,
                "error": str(result),
            })
        elif result.get("ok"):
            succeeded.append({
                "identifier": identifier,
                "source": result.get("source"),
            })
            fname = result.get("filename")
            if fname:
                fpath = DOWNLOAD_DIR / fname
                if fpath.exists():
                    downloaded_files.append(fpath)
        else:
            failed.append({
                "identifier": identifier,
                "fallback_url": result.get("fallback_url"),
                "error": result.get("error", "Download failed"),
            })

    # Package ALL successes into a single ZIP
    response = {
        "total_requested": total,
        "downloaded": len(succeeded),
        "failed_count": len(failed),
    }

    if downloaded_files:
        zip_name = f"papers_batch_{_ts()}.zip"
        zip_path = _zip_files(downloaded_files, zip_name)
        if zip_path:
            response["zip_url"] = _pub_url(zip_name)
            # Clean up individual PDFs after successful zip
            for fp in downloaded_files:
                try:
                    fp.unlink()
                except OSError:
                    pass
        else:
            # Fallback: return individual URLs if zip failed
            response["individual_urls"] = [
                _pub_url(fp.name) for fp in downloaded_files
            ]

    if failed:
        response["failed"] = [
            {k: v for k, v in f.items() if v is not None}
            for f in failed
        ]

    _log(f"batch_download complete: {len(succeeded)}/{total} downloaded, zip={'yes' if response.get('zip_url') else 'no'}")
    return json.dumps(response)


@mcp.tool()
async def resolve_references(references: list[str]) -> str:
    """
    Resolve raw citation strings to DOIs using CrossRef bibliographic matching.
    Use this when a user pastes a reference list without DOIs.
    Feed the resolved DOIs into batch_download().

    Args:
        references: List of citation strings, e.g.:
            ["Smith et al., Nature 2024, Perovskite degradation...",
             "10.1038/s41586-024-07472-z",
             "Zhang, J. Phys. Chem. Lett. 2023, 14, 5680"]

    Returns: JSON with resolved (doi + confidence) and unresolved lists
    """
    _log(f"resolve_references: {len(references)} refs")
    client = await get_client()
    sem = asyncio.Semaphore(BATCH_SEMAPHORE)

    # Pre-process: if a reference is already a DOI, just pass it through
    resolved = []
    to_resolve = []
    for ref in references[:50]:  # Soft cap
        ref = ref.strip()
        if not ref:
            continue
        # Check if it's already a DOI
        clean = _clean_doi(ref)
        if re.match(r"^10\.\d{4,}/", clean):
            resolved.append({
                "input": ref[:120],
                "doi": clean,
                "confidence": "exact",
            })
        else:
            to_resolve.append(ref)

    # Resolve the rest via CrossRef
    if to_resolve:
        tasks = [_resolve_one_ref(client, ref, sem) for ref in to_resolve]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                resolved.append({"input": "unknown", "doi": None, "error": str(r)})
            elif r.get("doi"):
                resolved.append(r)
            else:
                resolved.append(r)

    # Split into resolved and unresolved for clean model consumption
    good = [r for r in resolved if r.get("doi")]
    bad = [r for r in resolved if not r.get("doi")]

    return json.dumps({
        "resolved": good,
        "unresolved": bad,
        "ready_for_download": [{"doi": r["doi"]} for r in good],
    })


# =========================================================================
# BOOK TOOLS (Anna's Archive / LibGen)
# =========================================================================


@mcp.tool()
async def books(query: str, max_results: int = 8) -> str:
    """
    Search for books via Anna's Archive and LibGen mirrors.

    Args:
        query: Book title, author, or ISBN
        max_results: Max results (default 8)

    Returns: JSON with books array: id, title, author, year, format, size
    """
    _log(f"books: {query}")
    client = await get_client()

    # === Try Anna's Archive first ===
    try:
        r = await client.get(
            "https://annas-archive.org/search",
            params={"q": query, "content": "book_any", "sort": ""},
            timeout=20.0,
        )
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            books_list = []

            for item in soup.select("a[href*='/md5/']")[: max_results * 2]:
                href = item.get("href", "")
                md5_match = re.search(r"/md5/([a-fA-F0-9]+)", href)
                if not md5_match:
                    continue

                text_content = item.get_text(separator="\n", strip=True)
                lines = [ln.strip() for ln in text_content.split("\n") if ln.strip()]
                if not lines:
                    continue

                title = lines[0]
                author, year, fmt, size = "", "", "pdf", ""
                for line in lines[1:]:
                    ll = line.lower()
                    for f in ("pdf", "epub", "mobi", "djvu", "azw3"):
                        if f in ll:
                            fmt = f
                    sm = re.search(r"([\d.]+\s*[kmg]b)", ll)
                    if sm:
                        size = sm.group(1).upper()
                    ym = re.search(r"\b(19|20)\d{2}\b", line)
                    if ym:
                        year = ym.group(0)
                    if (
                        not author
                        and line != title
                        and not any(x in ll for x in ("pdf", "epub", "mb", "kb"))
                    ):
                        author = line

                books_list.append(
                    {
                        "id": md5_match.group(1),
                        "title": title[:200],
                        "author": author[:100] or "Unknown",
                        "year": year,
                        "size": size,
                        "fmt": fmt,
                    }
                )
                if len(books_list) >= max_results:
                    break

            if books_list:
                return json.dumps({"books": books_list})
    except Exception as e:
        _log(f"Anna's Archive error: {e}")

    # === Fallback: LibGen mirrors ===
    for mirror in [
        "https://libgen.is",
        "https://libgen.rs",
        "https://libgen.st",
        "https://libgen.li",
    ]:
        try:
            r = await client.get(
                f"{mirror}/search.php",
                params={
                    "req": query,
                    "lg_topic": "libgen",
                    "open": 0,
                    "view": "simple",
                    "res": max_results,
                    "column": "def",
                },
                timeout=15.0,
            )
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            table = soup.find("table", class_="c")
            if not table:
                continue

            books_list = []
            for row in table.find_all("tr")[1 : max_results + 1]:
                cols = row.find_all("td")
                if len(cols) < 10:
                    continue
                link = cols[2].find("a", href=True)
                md5 = None
                if link and "md5=" in link.get("href", ""):
                    m = re.search(r"md5=([A-Fa-f0-9]+)", link["href"])
                    if m:
                        md5 = m.group(1)
                books_list.append(
                    {
                        "id": md5 or cols[0].text.strip(),
                        "title": link.text.strip() if link else cols[2].text.strip(),
                        "author": cols[1].text.strip(),
                        "year": cols[4].text.strip(),
                        "size": cols[7].text.strip(),
                        "fmt": cols[8].text.strip().lower(),
                    }
                )

            if books_list:
                return json.dumps({"books": books_list})
        except Exception as e:
            _log(f"LibGen {mirror} error: {e}")
            continue

    return json.dumps({"error": "All book sources unavailable"})


@mcp.tool()
async def book_download(book_id: str, title: str = None) -> str:
    """
    Download book by MD5 hash ID from Anna's Archive / LibGen mirrors.

    Args:
        book_id: Book MD5 hash (from books search results)
        title: Optional title for filename

    Returns: JSON with ok, url, format, or error with suggestion
    """
    _log(f"book_download: {book_id}")
    client = await get_client()
    book_id = book_id.lower().strip()
    file_ext = ".pdf"

    # === Anna's Archive ===
    try:
        r = await client.get(f"https://annas-archive.org/md5/{book_id}", timeout=20.0)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            page_text = soup.get_text().lower()

            for fmt in (".epub", ".mobi", ".djvu", ".azw3", ".cbr", ".cbz"):
                if fmt[1:] in page_text:
                    file_ext = fmt
                    break

            download_urls = []
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                text = (a.get_text(strip=True) or "").lower()
                if "library.lol" in href:
                    download_urls.insert(0, ("library.lol", href))
                elif "cloudflare" in href.lower() or "ipfs" in href.lower():
                    download_urls.insert(0, ("cloudflare/ipfs", href))
                elif href.startswith("http") and (
                    "download" in text or "get" in text or "fast" in text
                ):
                    download_urls.append(("other", href))

            for source_name, dl_url in download_urls[:8]:
                try:
                    if "library.lol" in dl_url or "libgen" in dl_url:
                        r2 = await client.get(dl_url, timeout=25.0)
                        if r2.status_code == 200:
                            soup2 = BeautifulSoup(r2.text, "html.parser")
                            actual_link = None
                            for a2 in soup2.find_all("a", href=True):
                                a2_text = (a2.get_text(strip=True) or "").lower()
                                a2_href = a2.get("href", "")
                                if "get" in a2_text or "download" in a2_text:
                                    actual_link = a2_href
                                    break
                                if "cloudflare" in a2_href.lower() or "/get.php" in a2_href:
                                    actual_link = a2_href
                                    break
                            if actual_link:
                                if actual_link.startswith("//"):
                                    actual_link = "https:" + actual_link
                                elif actual_link.startswith("/"):
                                    parsed = urlparse(str(r2.url))
                                    actual_link = f"{parsed.scheme}://{parsed.netloc}{actual_link}"
                                dl_url = actual_link

                    fname = f"book_{_sanitize(title or book_id[:16])}_{_ts()}{file_ext}"
                    fpath = DOWNLOAD_DIR / fname
                    if await _download_file(client, dl_url, fpath, expected_type=file_ext[1:]):
                        return json.dumps(
                            {"ok": True, "url": _pub_url(fname), "format": file_ext[1:]}
                        )
                except Exception as e:
                    _log(f"Download from {source_name} failed: {e}")
                    continue
    except Exception as e:
        _log(f"Anna's Archive error: {e}")

    # === Fallback: Library.lol directly ===
    for path_type in ("main", "fiction"):
        try:
            mirror_url = f"https://library.lol/{path_type}/{book_id}"
            r = await client.get(mirror_url, timeout=25.0)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            download_link = None
            for a in soup.find_all("a", href=True):
                text = (a.get_text(strip=True) or "").lower()
                href = a.get("href", "")
                if "get" in text:
                    download_link = href
                    break
                if "cloudflare" in href.lower():
                    download_link = href
                    break

            if download_link:
                if download_link.startswith("//"):
                    download_link = "https:" + download_link
                for ext in (".epub", ".mobi", ".djvu", ".azw3", ".pdf"):
                    if ext in download_link.lower():
                        file_ext = ext
                        break
                fname = f"book_{_sanitize(title or book_id[:16])}_{_ts()}{file_ext}"
                fpath = DOWNLOAD_DIR / fname
                if await _download_file(client, download_link, fpath, expected_type=file_ext[1:]):
                    return json.dumps(
                        {"ok": True, "url": _pub_url(fname), "format": file_ext[1:]}
                    )
        except Exception as e:
            _log(f"Library.lol {path_type} error: {e}")
            continue

    return json.dumps({
        "ok": False,
        "error": "Download failed from all sources",
        "suggestion": f"Try manually: https://annas-archive.org/md5/{book_id}",
    })


# =========================================================================
# SERVER STARTUP
# =========================================================================

if __name__ == "__main__":
    _mode = os.getenv("MODE", "stdio").lower()
    _log(f"Starting Academic MCP Server v5.7 (mode={_mode})")
    _log(f"  Sources: arXiv, OpenAlex (keyword+ID), "
         f"{'ADS, ' if ADS_API_KEY else ''}{'S2 (search+ID), ' if S2_API_KEY else ''}PubMed, CrossRef"
         f"{', +OA semantic (additive)' if OPENALEX_API_KEY else ''}"
         f", OpenAIRE"
         f"{', ORCID' if ORCID_CLIENT_ID else ''}")
    _log(f"  Author identity resolution: "
         f"OA={'yes' if OPENALEX_API_KEY or MAILTO else 'no'}, "
         f"S2={'yes' if S2_API_KEY else 'no'}, "
         f"ORCID={'yes' if ORCID_CLIENT_ID else 'no'}")
    _log(f"  Over-fetch + rank pipeline active (author: 3x, topic: 2x)")
    _log(f"  OA content download: {'enabled' if OPENALEX_API_KEY else 'disabled (no API key)'}")
    if _mode == "http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")