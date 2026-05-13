"""
title: Citation Metadata Enricher
author: Marios Adamidis
description: Extracts bibliographic metadata (title, authors, DOI, year) from uploaded 
             PDF files and injects a citation reference table into the system prompt.
             Fixes the "filename-only citation" bug where RAG/full-context sources lack
             bibliographic information. Works on both initial upload messages and 
             follow-up messages by scanning conversation history for file references.
version: 1.0.0
license: MIT
"""

import re
import json
import sqlite3
import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# ── Regex patterns ──────────────────────────────────────────────────────
DOI_RE = re.compile(r'\b(10\.\d{4,}/[^\s,;\]}>]+)')
YEAR_RE = re.compile(r'\b(20[0-3]\d)\b')
# Files JSON block from uploadedfilename filter
FILES_JSON_RE = re.compile(
    r'\[\[FILES_JSON\]\]\s*(\{.*?\})\s*\[\[/FILES_JSON\]\]', re.DOTALL
)
MARKER = "[[CITATION_REFERENCES]]"

# Title junk: skip lines that are just numbers, page markers, labels
TITLE_SKIP_RE = re.compile(
    r'^(\d[\d():,;\s]*$'            # just numbers and punctuation
    r'|<!--.*-->$'                   # HTML comments  
    r'|https?://'                    # URLs
    r'|arXiv:'                       # arXiv IDs
    r'|doi:'                         # DOI prefixes
    r'|ARTICLE$'                     # section labels
    r'|PERSPECTIVE$'
    r'|REVIEW$'
    r'|LETTER$'
    r'|ORIGINAL PAPER$'
    r'|Open Access$'
    r'|Received|Accepted|Published'  # date lines
    r')',
    re.IGNORECASE
)


class Filter:
    class Valves(BaseModel):
        enabled: bool = Field(
            default=True, description="Enable/disable citation enrichment"
        )
        db_path: str = Field(
            default="/app/backend/data/webui.db",
            description="Path to OWUI SQLite database",
        )
        content_scan_chars: int = Field(
            default=2000,
            description="Characters of content to scan for metadata",
        )
        debug: bool = Field(
            default=False, description="Print debug info to server logs"
        )

    def __init__(self):
        self.valves = self.Valves()
        self._cache: Dict[str, Dict[str, str]] = {}

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _msg_text(msg: dict) -> str:
        """Safely extract text from a message, handling both str and list content."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return ""

    @staticmethod
    def _to_lines(text: str) -> List[str]:
        """Split content into lines, handling both real newlines and literal \\n."""
        # Replace literal \n sequences with actual newlines, then split
        normalized = text.replace("\\n", "\n")
        return [ln.strip() for ln in normalized.splitlines() if ln.strip()]

    # ── File discovery ──────────────────────────────────────────────────

    def _collect_pdf_ids(self, body: dict) -> List[Dict[str, str]]:
        """
        Collect PDF file references from all available sources:
        1. body["files"] and body["metadata"]["files"] (current message)
        2. msg["files"] for all messages in history
        3. [[FILES_JSON]] blocks from the uploadedfilename filter
        """
        seen: set = set()
        results: List[Dict[str, str]] = []

        def _add(fid: str, fname: str):
            if fid and fid not in seen and fname.lower().endswith(".pdf"):
                seen.add(fid)
                results.append({"id": fid, "name": fname})

        def _scan_file_list(file_list):
            if not isinstance(file_list, list):
                return
            for f in file_list:
                if not isinstance(f, dict):
                    continue
                fid = f.get("id")
                fname = f.get("name") or f.get("filename") or ""
                if not fid and isinstance(f.get("file"), dict):
                    inner = f["file"]
                    fid = inner.get("id")
                    fname = fname or inner.get("filename") or ""
                _add(fid, fname)

        # Current message files
        _scan_file_list(body.get("files"))
        _scan_file_list(body.get("metadata", {}).get("files"))

        # All messages in history
        for msg in body.get("messages", []):
            if isinstance(msg, dict):
                _scan_file_list(msg.get("files"))

        # [[FILES_JSON]] blocks from uploadedfilename filter
        for msg in body.get("messages", []):
            if not isinstance(msg, dict):
                continue
            content = self._msg_text(msg)
            if not content:
                continue
            match = FILES_JSON_RE.search(content)
            if match:
                try:
                    fj = json.loads(match.group(1))
                    for f in fj.get("files", []):
                        _add(f.get("id", ""), f.get("name", ""))
                except (json.JSONDecodeError, AttributeError):
                    pass

        return results

    # ── Metadata extraction ─────────────────────────────────────────────

    def _get_metadata(self, file_id: str, filename: str) -> Dict[str, str]:
        """Get bibliographic metadata for a PDF, with caching."""
        if file_id in self._cache:
            return self._cache[file_id]

        result = {"filename": filename}

        try:
            conn = sqlite3.connect(self.valves.db_path, timeout=3)
            cur = conn.cursor()
            cur.execute(
                "SELECT filename, path, data FROM file WHERE id = ?",
                (file_id,),
            )
            row = cur.fetchone()
            conn.close()

            if not row:
                self._cache[file_id] = result
                return result

            db_filename, file_path, data_json = row
            result["filename"] = db_filename or filename

            # Strategy 1: Parse extracted text content (most reliable)
            if data_json:
                try:
                    data = json.loads(data_json)
                    content = data.get("content", "")
                    if content:
                        parsed = self._parse_content(content)
                        result.update(parsed)
                except (json.JSONDecodeError, KeyError):
                    pass

            # Strategy 2: pypdf metadata as fallback
            if file_path and os.path.exists(file_path):
                try:
                    from pypdf import PdfReader

                    reader = PdfReader(file_path)
                    meta = reader.metadata
                    if meta:
                        if meta.title and "title" not in result:
                            t = str(meta.title).strip()
                            if len(t) > 5:
                                result["title"] = t[:200]
                        if meta.author and "authors" not in result:
                            a = str(meta.author).strip()
                            if len(a) > 3:
                                result["authors"] = a[:300]
                except Exception:
                    pass

        except Exception as e:
            if self.valves.debug:
                print(f"[Citation Enricher] Error for {file_id}: {e}")

        self._cache[file_id] = result
        return result

    def _parse_content(self, content: str) -> Dict[str, str]:
        """Extract title, authors, year, DOI from document text."""
        result: Dict[str, str] = {}
        preview = content[: self.valves.content_scan_chars]
        lines = self._to_lines(preview)
        if not lines:
            return result

        # ── Title ───────────────────────────────────────────────────────
        # First substantial line that isn't junk
        title_idx = -1
        for i, line in enumerate(lines[:10]):
            clean = line.lstrip("#").strip()
            if len(clean) < 15:
                continue
            if TITLE_SKIP_RE.match(clean):
                continue
            result["title"] = clean[:250]
            title_idx = i
            break

        # ── Authors ─────────────────────────────────────────────────────
        # Look at lines after the title for comma-separated names
        if title_idx >= 0:
            for line in lines[title_idx + 1 : title_idx + 8]:
                # Skip short lines and non-name lines
                if len(line) < 20:
                    continue
                # Skip lines that look like abstracts or section headers
                if line.lower().startswith(
                    ("abstract", "keyword", "##", "introduction", "doi", "http")
                ):
                    break
                # Author heuristic: has commas and at least 2 capitalized words
                caps = re.findall(r'\b[A-Z][a-z\u00e0-\u00ff]+\b', line)
                if len(caps) >= 2 and "," in line:
                    # Clean superscript numbers between names
                    cleaned = re.sub(r'\s*\d+\s*,', ',', line)
                    cleaned = re.sub(r'\s*\d+\s*$', '', cleaned)
                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                    # Remove trailing markers like *, †, ‡
                    cleaned = re.sub(r'[*†‡§∥⊥]+\s*$', '', cleaned).strip()
                    if len(cleaned) > 10:
                        result["authors"] = cleaned[:350]
                    break

        # ── DOI ─────────────────────────────────────────────────────────
        doi_match = DOI_RE.search(content[:5000])
        if doi_match:
            doi = doi_match.group(1).rstrip(".,;)")
            result["doi"] = doi

        # ── Year (from DOI or near title) ───────────────────────────────
        # Prefer year from DOI URL or nearby the title
        if "doi" in result:
            doi_year = YEAR_RE.search(result["doi"])
            if doi_year:
                result["year"] = doi_year.group(1)
        if "year" not in result:
            # Look in first 1500 chars
            year_match = YEAR_RE.search(content[:1500])
            if year_match:
                result["year"] = year_match.group(1)

        return result

    # ── Inlet ───────────────────────────────────────────────────────────

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        if not self.valves.enabled:
            return body

        if self.valves.debug:
            file_count = len(body.get("files") or [])
            meta_file_count = len(body.get("metadata", {}).get("files") or [])
            msg_count = len(body.get("messages", []))
            print(f"[Citation Enricher] inlet called: {file_count} body.files, {meta_file_count} meta.files, {msg_count} messages")

        # Skip if we already injected
        for msg in body.get("messages", []):
            if MARKER in self._msg_text(msg):
                return body

        pdf_files = self._collect_pdf_ids(body)
        if self.valves.debug:
            print(f"[Citation Enricher] Found {len(pdf_files)} PDFs: {[f['name'][:30] for f in pdf_files]}")
        if not pdf_files:
            return body

        # Extract metadata
        citations = []
        for f in pdf_files:
            meta = self._get_metadata(f["id"], f["name"])
            if any(k in meta for k in ("title", "authors", "doi", "year")):
                citations.append(meta)

        if not citations:
            return body

        # Build injection block
        lines = [
            MARKER,
            "[System: Citation Reference Data]",
            "Bibliographic metadata extracted from uploaded PDFs.",
            "When citing these sources, use proper academic citations",
            "with the information below instead of just the filename.",
            "",
        ]

        for c in citations:
            parts = [f'File: "{c.get("filename", "unknown")}"']
            if c.get("title"):
                parts.append(f'  Title: {c["title"]}')
            if c.get("authors"):
                parts.append(f'  Authors: {c["authors"]}')
            if c.get("year"):
                parts.append(f'  Year: {c["year"]}')
            if c.get("doi"):
                parts.append(f'  DOI: {c["doi"]}')
            lines.extend(parts)
            lines.append("")

        lines.append("[[/CITATION_REFERENCES]]")
        injection = "\n".join(lines)

        if self.valves.debug:
            print(f"[Citation Enricher] Injecting {len(citations)} citation(s)")
            for c in citations:
                print(
                    f"  {c.get('filename','?')[:40]}: "
                    f"{c.get('title','no title')[:50]}"
                )

        # Insert after existing system messages
        messages = body.get("messages", [])
        insert_at = 0
        while (
            insert_at < len(messages)
            and messages[insert_at].get("role") == "system"
        ):
            insert_at += 1
        messages.insert(insert_at, {"role": "system", "content": injection})
        body["messages"] = messages

        return body
