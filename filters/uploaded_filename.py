"""
title: Files Metadata (IDs for tools)
author: Marios Adamidis (FORTHought Lab)
version: 1.0.0
"""

import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        enabled: bool = Field(default=True, description="Enable/disable filter")

    def __init__(self):
        self.valves = self.Valves()

    def _extract_files(self, body: dict) -> List[Dict[str, str]]:
        files = body.get("files") or []
        out: List[Dict[str, str]] = []
        for f in files:
            # Handle both OWUI shapes:
            # 1) {"id": "...", "name": "..."}
            # 2) {"file": {"id": "...", "filename": "..."}}
            fid = None
            fname = None
            if isinstance(f, dict) and "id" in f and "name" in f:
                fid = f.get("id")
                fname = f.get("name")
            elif isinstance(f, dict) and "file" in f:
                inner = f.get("file") or {}
                fid = inner.get("id") or inner.get("hash") or inner.get("uid")
                fname = inner.get("filename") or inner.get("name")
            if fid and fname:
                out.append({"id": fid, "name": fname})
        return out

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if not self.valves.enabled:
            return body

        flist = self._extract_files(body)
        if not flist:
            return body

        marker = "[[FILES_JSON]]"
        # Avoid duplicating the block
        for m in body.get("messages", []):
            if marker in (m.get("content") or ""):
                return body

        payload = {"files": flist}
        block = f"{marker}\n{json.dumps(payload)}\n[[/FILES_JSON]]"

        if body.get("messages") and body["messages"][0].get("role") == "system":
            body["messages"][0]["content"] += "\n\n" + block
        else:
            body.setdefault("messages", []).insert(
                0, {"role": "system", "content": block}
            )

        return body
