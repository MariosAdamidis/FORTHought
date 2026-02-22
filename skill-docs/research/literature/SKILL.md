# Literature Search & Download — Skill v7

You have access to a powerful academic search and download system via MCP tools.
Use this skill whenever the user asks about papers, references, or literature.

## Tools

### search_papers() — Find Papers
```
search_papers(query, max_results=15, year_min=None, year_max=None, sort="relevance", mode="auto", author=None)
```
All parameters are passed directly — use them, don't try to work around them.

**Sources:** arXiv, OpenAlex, NASA ADS, Semantic Scholar, PubMed, CrossRef, OpenAIRE, ORCID

**How search works internally:** The server over-fetches from all sources (3× for author searches, 2× for topics), deduplicates, applies composite scoring (relevance + recency + citations + source agreement), then returns the top N. You always get the best papers from a much larger pool.

**Author identity resolution (v5.7):** When `author` is set, the server first resolves the name to canonical IDs:
- **OpenAlex Author ID** → cursor-paginated works (most complete, 135+ papers for prolific authors)
- **Semantic Scholar Author ID** → /author/{id}/papers endpoint (full corpus)
- **ORCID** → self-curated publication list with DOIs

Then fires ID-based queries in parallel alongside name-based queries (arXiv, ADS, PubMed, CrossRef, OpenAIRE). This finds **5-10× more papers** than name-string search alone.

**Pool summary:** Every search returns `pool_summary` alongside results — use it to inform the user:
- `raw_results` / `unique_after_dedup` / `returned` — how wide the net was
- `per_source` — which databases contributed (helps diagnose gaps)
- `author_resolution` — shows which IDs were resolved (OA, S2, ORCID)
- `note` — tells user when more results are available

Tell the user about the pool: "Found 187 unique papers across 8 sources, showing the top 30 ranked by relevance. Author resolved via OpenAlex (A5089770490), S2 (4549479), and ORCID."

**Author search (author param):**
- `author="Emmanuel Stratakis"` — triggers ID resolution for exhaustive coverage
- Can combine with `query` for topic+author: `search_papers(query="laser ablation", author="Stratakis")`
- Author-only (no query): returns all papers by that author, sorted by date
- For comprehensive author coverage, use `max_results=50` (server casts wide net across 8+ sources)

**Modes (mode param):**
- `"auto"` (default): always searches all keyword sources; additionally attempts AI semantic matching for natural language queries (≥6 words)
- `"semantic"`: forces AI semantic matching alongside keyword sources — finds conceptually related papers even with different terminology
- `"keyword"`: keyword sources only, no semantic. Use for exact terms, author names, boolean precision

> All modes always query: arXiv, OpenAlex, ADS, Semantic Scholar, PubMed, CrossRef, OpenAIRE (+ORCID for author searches). Semantic is additive — it never replaces keyword sources.

**Boolean support (keyword mode):**
- `AND`, `OR`, `NOT` must be UPPERCASE
- Exact phrases: use double quotes `"pulsed laser deposition"`
- Example: `(femtosecond AND laser) NOT biological`

**Year filtering:**
- "recent papers" → `year_min=2022, sort="date"`
- "papers from 2015-2020" → `year_min=2015, year_max=2020`
- "classic/seminal papers" → `sort="citations"`

**Sort options:**
- `"relevance"` (default): composite score (word overlap + recency + citations + source agreement)
- `"date"`: newest first
- `"citations"`: most cited first

### download_papers() — Get PDFs
Pass a list of paper dicts with `arxiv_id` or `doi`. Prefer `arxiv_id` (fastest).
Download cascade: arXiv → OpenAlex Content → Unpaywall → Elsevier API → Publisher Direct → PMC → Sci-Hub

### run("papers.details", {"arxiv_id": ...}) — Get Abstracts
Only call after search. Pass arxiv_id or doi from search results.

### run("papers.resolve_references", {"citations": [...]}) — Citation Strings → DOIs
Pass raw citation strings, returns DOIs with confidence scores.

### run("papers.books", {"query": ...}) / run("papers.book_download", ...) — Textbooks
Search and download books via Library Genesis.

## Decision Patterns

| User says | search_papers() call |
|-----------|---------------------|
| "find papers about X" | `search_papers(query="X")` |
| "10 recent papers on X" | `search_papers(query="X", max_results=10, year_min=2023, sort="date")` |
| "papers by Stratakis" | `search_papers(author="Emmanuel Stratakis", max_results=50)` |
| "Stratakis papers on laser ablation" | `search_papers(query="laser ablation", author="Emmanuel Stratakis")` |
| "all papers by [name] since 2022" | `search_papers(author="[name]", year_min=2022, max_results=50, sort="date")` |
| "recent papers by [name] on [topic]" | `search_papers(query="[topic]", author="[name]", year_min=2023, sort="date")` |
| "papers similar to [description]" | `search_papers(query="[description]", mode="semantic")` |
| "seminal/classic papers on X" | `search_papers(query="X", sort="citations")` |
| "XRD papers on thin films 2020-2024" | `search_papers(query="XRD thin film", year_min=2020, year_max=2024)` |
| Exact phrase needed | `search_papers(query='"pulsed laser deposition" AND silicon', mode="keyword")` |

## Rules
1. **One search, then present.** Call search_papers ONCE per user request, then present results. Do NOT re-search because results seem imperfect — multi-source search already queries 8 databases with over-fetching and author ID resolution. If the user wants refinement, they will ask.
2. **Report the pool.** Always mention pool_summary in your response so the user knows how comprehensive the search was and whether more results are available. If author_resolution is present, mention which IDs were resolved.
3. **Don't auto-download** — present search results first, let user choose.
4. **Use the author parameter** for author searches — never embed `author:"Name"` in the query string. The `author` param triggers ID-based resolution which is far more exhaustive.
5. **Prefer arxiv_id** for downloads (fastest, most reliable source).
6. **Always pass year params** when user mentions recency or date ranges — the server filters natively.
7. **Compact output** — search returns lean data. Use details() only when user wants abstracts.
8. **Never fabricate** — every DOI, arXiv ID, and citation must come from tool results.
