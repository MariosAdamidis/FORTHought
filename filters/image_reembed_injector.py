"""
title: Image Re-Embed Injector (Passive)
author: Marios Adamidis
description: Injects uploaded image URLs into system context and post-processes model
             output to fix bare-UUID image references into working /api/v1/files paths.
version: 1.0.0
changelog: v1.4.0 - Added outlet() post-processing that rewrites bare-UUID image markdown
           into full /api/v1/files/{id}/content paths. This fixes re-embed for ALL models
           regardless of whether they follow the injection prompt. Belt-and-suspenders.
           v1.3.1 - Added explicit anti-hallucination instruction.
           v1.3.0 - Rewrote injection prompt for cleaner URL exposure.
"""

import os
import glob
import re
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field


UUID_RE = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)

# Matches ![any alt text](bare-uuid) or ![any alt text](bare-uuid/content)
# Captures: group(1)=alt text, group(2)=the UUID, group(3)=optional trailing path
_BARE_UUID_IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]"  # ![alt text]
    r"\("  # (
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"  # bare UUID
    r"(/content)?"  # optional /content suffix
    r"\)"  # )
)

# Matches ![alt text](URL) where URL is an external host the model hallucinated
# (e.g. oaiusercontent.com, openai.com file URLs, googleusercontent, etc.)
_HALLUCINATED_URL_IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]"  # ![alt text]
    r"\("  # (
    r"https?://(?:files\.oaiusercontent\.com|openaifiles|.*?googleusercontent\.com)/[^\s)]*"  # hallucinated URL
    r"\)"  # )
)


class Filter:
    class Valves(BaseModel):
        max_images: int = Field(
            default=5, ge=1, le=20, description="Max images to expose to the model."
        )
        use_disk_fallback: bool = Field(
            default=True,
            description="If no images found in message payload, scan uploads dir for newest image.",
        )
        uploads_dir: str = Field(
            default="/app/backend/data/uploads",
            description="Open WebUI uploads directory inside container.",
        )
        fix_hallucinated_urls: bool = Field(
            default=True,
            description="Replace hallucinated external image URLs (oaiusercontent, etc.) with the latest uploaded image.",
        )
        debug: bool = Field(
            default=False, description="Print debug info to server logs."
        )

    def __init__(self):
        self.valves = self.Valves()
        # Track known file IDs from inlet so outlet can validate UUIDs
        self._known_file_ids: set = set()

    def _is_image_name(self, name: str) -> bool:
        if not name or "." not in name:
            return False
        ext = name.lower().rsplit(".", 1)[-1]
        return ext in {"png", "jpg", "jpeg", "gif", "webp", "bmp", "svg", "tiff"}

    def _extract_uuid_from_url(self, url: str) -> Optional[str]:
        if not url:
            return None
        m = UUID_RE.search(url)
        return m.group(1) if m else None

    def _extract_images_from_message(
        self, msg: Dict[str, Any]
    ) -> List[Tuple[str, str, str]]:
        """
        Returns list of (file_id, display_name, url).
        Supports:
          - msg["files"] (older/alternate payloads)
          - msg["images"] (some pipelines emit this)
          - msg["content"] as list with {"type":"image_url", "image_url":{"url": ...}}
        """
        found: List[Tuple[str, str, str]] = []

        # 1) msg["files"]
        files = msg.get("files")
        if isinstance(files, list):
            for f in files:
                if not isinstance(f, dict):
                    continue
                fid = f.get("id")
                name = f.get("name") or f.get("filename") or "image"
                if fid and self._is_image_name(name):
                    url = f"/api/v1/files/{fid}/content"
                    found.append((fid, name, url))

        # 2) msg["images"]
        images = msg.get("images")
        if isinstance(images, list):
            for img in images:
                if not isinstance(img, dict):
                    continue
                url = img.get("url") or img.get("image_url")
                if not isinstance(url, str):
                    continue
                fid = self._extract_uuid_from_url(url)
                if fid:
                    name = img.get("name") or f"image_{fid[:8]}"
                    found.append((fid, name, f"/api/v1/files/{fid}/content"))

        # 3) OpenAI-style multimodal content blocks
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "image_url":
                    continue
                image_url = item.get("image_url")
                url = None
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                elif isinstance(image_url, str):
                    url = image_url

                if isinstance(url, str) and "/api/v1/files/" in url:
                    fid = self._extract_uuid_from_url(url)
                    if fid:
                        name = f"uploaded_image_{fid[:8]}"
                        found.append((fid, name, f"/api/v1/files/{fid}/content"))

        return found

    def _get_latest_image_from_disk(self) -> Optional[Tuple[str, str, str]]:
        """
        Scans uploads dir for newest image. Filenames look like:
          <uuid>_<originalname>.png or <uuid>__<something>.jpg
        """
        try:
            pattern = os.path.join(self.valves.uploads_dir, "*")
            files = glob.glob(pattern)
            if not files:
                return None
            files.sort(key=os.path.getmtime, reverse=True)
            for fpath in files:
                base = os.path.basename(fpath)
                if not self._is_image_name(base):
                    continue
                m = UUID_RE.match(base)
                if not m:
                    continue
                fid = m.group(1)
                rest = base[len(fid) :].lstrip("_") or base
                url = f"/api/v1/files/{fid}/content"
                return (fid, rest, url)
        except Exception as e:
            if self.valves.debug:
                print(f"[Image Injector] Disk scan error: {e}")
        return None

    # ── INLET: inject image context into system prompt ───────────────────
    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """Pre-process: inject image URLs into system context."""
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return body

        # Collect images from all messages
        imgs: List[Tuple[str, str, str]] = []
        for msg in messages:
            if isinstance(msg, dict):
                imgs.extend(self._extract_images_from_message(msg))

        # De-dup by file id, keep last occurrence (most recent)
        dedup: Dict[str, Tuple[str, str, str]] = {}
        for fid, name, url in imgs:
            dedup[fid] = (fid, name, url)
        imgs = list(dedup.values())

        # Disk fallback if nothing found in payload
        if not imgs and self.valves.use_disk_fallback:
            latest = self._get_latest_image_from_disk()
            if latest:
                imgs = [latest]

        # Limit
        imgs = imgs[-self.valves.max_images :]

        if self.valves.debug:
            print(f"[Image Injector] Found {len(imgs)} image(s): {imgs}")

        if not imgs:
            return body

        # Track known file IDs so outlet can validate bare UUIDs
        self._known_file_ids = {fid for fid, _, _ in imgs}

        # ── Build injection prompt ───────────────────────────────────────
        lines = [
            "[System: Uploaded Images]",
            "The user has uploaded the following images.",
            "You can see and analyze them. You do not need to re-display them in every response.",
            "When you reference an uploaded image, you MUST use the EXACT full URL path shown below.",
            "NEVER fabricate, guess, or use any other URL. NEVER use oaiusercontent.com or any external hosting URL.",
            "The ONLY valid image URLs are the ones listed here.",
            "",
        ]

        for _fid, name, url in imgs:
            lines.append(f"Image: {name}")
            lines.append(f"  URL: {url}")
            lines.append("")

        # Explicit copy-paste directive
        lines.append(
            "When the user asks you to show, display, re-embed, or include any uploaded image,"
        )
        lines.append(
            "respond with EXACTLY this markdown (copy it verbatim, do not modify the URL):"
        )
        lines.append("")
        for _fid, name, url in imgs:
            lines.append(f"![{name}]({url})")
        lines.append("")

        injection = "\n".join(lines)

        # Insert as a system message after any existing system messages
        insert_at = 0
        while insert_at < len(messages) and messages[insert_at].get("role") == "system":
            insert_at += 1
        messages.insert(insert_at, {"role": "system", "content": injection})

        body["messages"] = messages
        return body

    # ── OUTLET: fix model output post-hoc ────────────────────────────────
    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Post-process: rewrite bare-UUID image references in the model's
        response to full /api/v1/files/{id}/content paths.

        This is the reliable fix — it doesn't matter if the model ignored
        our injection prompt. We catch and fix the output regardless.

        Handles:
          - ![alt](bare-uuid)           → ![alt](/api/v1/files/bare-uuid/content)
          - ![alt](bare-uuid/content)   → ![alt](/api/v1/files/bare-uuid/content)
          - ![alt](hallucinated-url)    → ![alt](/api/v1/files/latest-id/content)
        """
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return body

        # Only process the last assistant message
        last_msg = messages[-1]
        if not isinstance(last_msg, dict) or last_msg.get("role") != "assistant":
            return body

        content = last_msg.get("content")
        if not isinstance(content, str) or not content:
            return body

        original = content
        modified = False

        # ── Fix 1: bare UUID image references ────────────────────────────
        def _fix_bare_uuid(match):
            alt = match.group(1)
            uuid = match.group(2)
            return f"![{alt}](/api/v1/files/{uuid}/content)"

        new_content = _BARE_UUID_IMAGE_RE.sub(_fix_bare_uuid, content)
        if new_content != content:
            content = new_content
            modified = True

        # ── Fix 2: hallucinated external URLs ────────────────────────────
        # Replace with the most recent known uploaded image
        if self.valves.fix_hallucinated_urls and self._known_file_ids:
            latest_fid = list(self._known_file_ids)[-1]

            def _fix_hallucinated(match):
                alt = match.group(1)
                return f"![{alt}](/api/v1/files/{latest_fid}/content)"

            new_content = _HALLUCINATED_URL_IMAGE_RE.sub(_fix_hallucinated, content)
            if new_content != content:
                content = new_content
                modified = True

        if modified:
            if self.valves.debug:
                print(f"[Image Injector] Outlet rewrote image URLs:")
                print(f"  Before: {original[:300]}")
                print(f"  After:  {content[:300]}")
            last_msg["content"] = content

        return body
