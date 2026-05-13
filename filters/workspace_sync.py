"""
title: Workspace Sync
author: Marios Adamidis
description: Auto-syncs Jupyter output files to the requesting user's Open Terminal sidebar. Runs as an always-on filter — invisible to models. Enable permanently in Admin → Functions.
version: 1.0.0
required_open_webui_version: 0.8.0
"""

import os
import shutil
import logging
import threading
from typing import List, Optional

from pydantic import BaseModel, Field

log = logging.getLogger("forthought.workspace_sync")

# ═══════════════════════════════════════════════════════════════════════════
#  Global state — tracks which files have been seen across all outlets
# ═══════════════════════════════════════════════════════════════════════════

_lock = threading.Lock()
_known_files: Optional[set] = None


def _get_known_files(files_dir: str) -> set:
    """Lazy-init the known files set on first call."""
    global _known_files
    if _known_files is None:
        with _lock:
            if _known_files is None:
                if os.path.isdir(files_dir):
                    _known_files = set(
                        f
                        for f in os.listdir(files_dir)
                        if os.path.isfile(os.path.join(files_dir, f))
                        and not f.startswith(".")
                    )
                else:
                    _known_files = set()
    return _known_files


def _sync_new_files(files_dir: str, workspace_dir: str, username: str) -> List[str]:
    """Copy genuinely new files from files_dir to workspace/{username}/."""
    global _known_files

    if not username or not os.path.isdir(files_dir):
        return []

    known = _get_known_files(files_dir)

    current = set(
        f
        for f in os.listdir(files_dir)
        if os.path.isfile(os.path.join(files_dir, f)) and not f.startswith(".")
    )

    new_files = current - known

    # Update known set immediately (thread-safe)
    with _lock:
        _known_files = current

    if not new_files:
        return []

    user_dir = os.path.join(workspace_dir, username)
    os.makedirs(user_dir, exist_ok=True)

    # Fix permissions so OT containers can read
    try:
        os.chmod(user_dir, 0o777)
    except OSError:
        pass

    copied_files: List[str] = []
    for fname in sorted(new_files):
        src = os.path.join(files_dir, fname)
        dst = os.path.join(user_dir, fname)
        if not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
                copied_files.append(fname)
            except Exception as e:
                log.warning(f"[SYNC] Copy failed: {fname} → {e}")

    if copied_files:
        log.info(f"[SYNC] {len(copied_files)} file(s) → {username}")

    return copied_files


# ═══════════════════════════════════════════════════════════════════════════
#  OWUI Filter
# ═══════════════════════════════════════════════════════════════════════════


class Filter:

    class Valves(BaseModel):
        FILES_DIR: str = Field(
            default="/data/files",
            description="Path to Jupyter output files (must be mounted into OWUI container).",
        )
        WORKSPACE_DIR: str = Field(
            default="/app/backend/data/workspace",
            description="Path to per-user workspace directories.",
        )
        ENABLED: bool = Field(
            default=True,
            description="Enable/disable workspace sync.",
        )

    def __init__(self):
        self.valves = self.Valves()

    def inlet(self, body: dict, **kwargs) -> dict:
        """Pass-through — no pre-processing needed."""
        return body

    async def outlet(
        self, body: dict, __user__: dict = None, __event_emitter__=None, **kwargs
    ) -> dict:
        """After every response, sync any new Jupyter outputs to this user's sidebar."""
        if not self.valves.ENABLED:
            return body

        if not __user__ or not isinstance(__user__, dict):
            return body

        username = (__user__.get("name") or "").strip()
        if not username:
            return body

        try:
            copied_files = _sync_new_files(
                self.valves.FILES_DIR,
                self.valves.WORKSPACE_DIR,
                username,
            )
            if copied_files and __event_emitter__:
                count = len(copied_files)
                noun = "file" if count == 1 else "files"
                preview = ", ".join(copied_files[:3])
                if count > 3:
                    preview += f", +{count - 3} more"
                detail = f" ({preview})" if preview else ""
                msg = f"Workspace synced: {count} {noun} added to Open Terminal sidebar{detail}."
                await __event_emitter__(
                    {"type": "notification", "data": {"type": "success", "content": msg}}
                )
                await __event_emitter__(
                    {"type": "status", "data": {"description": msg, "done": True}}
                )
        except Exception as e:
            log.warning(f"[SYNC] Error: {e}")

        return body
