"""
title: Image→Text Context (Vision Inlet + Status, v1.4.1)
author: inMorphis (original); Marios Adamidis / FORTHought Lab (adaptation)
version: 1.0.0
required_open_webui_version: 0.6.4
description: Extracts text from images using an OpenAI-compatible vision model, injects context (user + pinned system), and strips image_url parts every turn so text-only models never see multimodal content. Shows spec-correct status with optional debug traces.
"""

import os, re, json, asyncio
from typing import Any, Dict, List, Optional, Callable, Awaitable
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────
# Your exact prompts (verbatim)
# ─────────────────────────────────────────────────────────────
USER_VISION_PROMPT = (
    "You act solely as a vision-to-text assistant transcribing every word you see, every small detail needs to be "
    "covered. You respond only with detail—no affirmations, no queues. If the picture only depicts words, repeat them "
    "word for word. If you see words + pictures, describe both. If something is blurry, unclear, or ambiguous, say so. "
    "If there is a specific character/item that has a name, name it, don't just describe it.\n"
)

JSON_SCHEMA_INSTRUCTION = (
    "Return ONLY valid JSON with the following keys:\n"
    "- transcript: the full detailed transcription/description following the user's instruction above\n"
    "- ui_hints: list of short strings naming visible UI elements, labels, or fields\n"
    "- text_blocks: list of strings for any distinct text/code/paragraph blocks\n"
    "- uncertainties: list of strings for anything blurry, unclear, or ambiguous\n"
    "No commentary outside JSON. No markdown. JSON only.\n"
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _last_user_index(messages: List[Dict[str, Any]]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def _message_has_images(msg: Dict[str, Any]) -> bool:
    c = msg.get("content")
    if isinstance(c, list):
        for p in c:
            if isinstance(p, dict) and p.get("type") == "image_url":
                return True
    # Extras some builds may use
    if isinstance(msg.get("images"), list) and msg["images"]:
        return True
    if isinstance(msg.get("files"), list):
        for f in msg["files"]:
            if isinstance(f, dict) and f.get("media_type", "").startswith("image/"):
                return True
    return False


def _conversation_has_images(messages: List[Dict[str, Any]]) -> bool:
    return any(_message_has_images(m) for m in messages or [])


def _last_user_has_images(messages: List[Dict[str, Any]]) -> bool:
    i = _last_user_index(messages)
    return i >= 0 and _message_has_images(messages[i])


def _collect_last_user_images_and_text(
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    i = _last_user_index(messages)
    if i < 0:
        return {"images": [], "user_text": ""}
    imgs, txt = [], ""
    msg = messages[i]
    c = msg.get("content")
    if isinstance(c, list):
        for p in c:
            if isinstance(p, dict) and p.get("type") == "image_url":
                u = (p.get("image_url") or {}).get("url")
                if u:
                    imgs.append(u)
            elif isinstance(p, dict) and p.get("type") == "text":
                txt += (p.get("text") or "") + "\n"
    elif isinstance(c, str):
        txt += c + "\n"
    # extras
    for u in msg.get("images") or []:
        if isinstance(u, str):
            imgs.append(u)
    for f in msg.get("files") or []:
        if (
            isinstance(f, dict)
            and f.get("url")
            and f.get("media_type", "").startswith("image/")
        ):
            imgs.append(f["url"])
    # unique order-preserving
    imgs = list(dict.fromkeys(imgs))
    return {"images": imgs, "user_text": txt.strip()}


def _already_injected_for_last_user(
    messages: List[Dict[str, Any]], header: str
) -> bool:
    i = _last_user_index(messages)
    if i <= 0:
        return False
    prev = messages[i - 1]
    cont = prev.get("content", "")
    return isinstance(cont, str) and (
        cont.strip().startswith(f"## {header}")
        or cont.strip().startswith(f"### {header}")
    )


def _mk_vision_messages(
    user_text: str, images: List[str], json_mode: bool = True
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": USER_VISION_PROMPT}]
    if json_mode:
        content.append({"type": "text", "text": JSON_SCHEMA_INSTRUCTION})
    if user_text:
        content.append({"type": "text", "text": "USER NOTE: " + user_text})
    for url in images:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [{"role": "user", "content": content}]


def _safe_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {
        "transcript": s[:100000],
        "ui_hints": [],
        "text_blocks": [],
        "uncertainties": [],
    }


def _to_markdown(j: Dict[str, Any]) -> str:
    parts = []
    tx = (j.get("transcript") or "").strip()
    if tx:
        parts.append("**Transcript (detailed):**\n" + tx)
    ui = j.get("ui_hints") or []
    if ui:
        parts.append("**UI hints:**\n- " + "\n- ".join(ui[:20]))
    tb = j.get("text_blocks") or []
    if tb:
        parts.append("**Text blocks:**\n- " + "\n- ".join(tb[:20]))
    un = j.get("uncertainties") or []
    if un:
        parts.append("**Uncertainties:**\n- " + "\n- ".join(un[:20]))
    return "### Screenshot Context\n" + "\n\n".join(parts) if parts else ""


async def _post_chat_completions(
    base: str, key: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    import urllib.request

    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {key}"} if key else {}),
        },
        method="POST",
    )

    def run():
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode())

    return await asyncio.to_thread(run)


def _strip_images_inplace(
    messages: List[Dict[str, Any]], scope: str, last_user_index: int
):
    """Remove image_url parts while preserving any text within the list."""

    def strip_msg(m):
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = "\n".join(
                [
                    p.get("text", "")
                    for p in c
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
            ).strip()

    if scope == "last" and last_user_index >= 0:
        strip_msg(messages[last_user_index])
        return
    for m in messages:
        strip_msg(m)


# ─────────────────────────────────────────────────────────────
# Filter
# ─────────────────────────────────────────────────────────────
class Filter:
    class Valves(BaseModel):
        # OpenAI-compatible endpoint
        VISION_API_BASE: str = Field(
            default_factory=lambda: os.getenv("VISION_API_BASE", "")
        )
        VISION_API_KEY: str = Field(
            default_factory=lambda: os.getenv("VISION_API_KEY", "")
        )
        VISION_MODEL: str = Field(
            default_factory=lambda: os.getenv("VISION_MODEL", "gemma3-12b-vision")
        )

        # Behavior
        OUTPUT_MODE: str = Field(
            default_factory=lambda: os.getenv("OUTPUT_MODE", "JSON")
        )  # JSON or RAW
        STRICT_JSON: bool = Field(default=True)
        STRIP_IMAGES: bool = Field(default=True)
        STRIP_SCOPE: str = Field(default="all", description="all | last")
        CLEAN_ON_EVERY_TURN: bool = Field(default=True)

        # If current model is text-only, force full-history strip
        TEXT_ONLY_MODELS: str = Field(
            default="qwen2.5-coder",
            description="Comma-separated substrings of text-only model ids.",
        )
        FORCE_STRIP_ON_TEXT_ONLY: bool = Field(default=True)

        PREPEND_AS: str = Field(default="user")  # inject right before request as user
        PIN_CONTEXT_AT_TOP: bool = Field(
            default=True
        )  # also pin a copy at top as system
        CONTEXT_HEADER: str = Field(default="Screenshot Context")
        SHOW_STATUS: bool = Field(default=True)
        ONLY_FOR_MODELS: str = Field(default="")  # optional guard

        DEBUG: bool = Field(default=False)

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __request__: Any,
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        v = self.valves

        async def emit_status(desc: str, done: bool = False):
            if not v.SHOW_STATUS:
                return
            try:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": str(desc),
                            "done": bool(done),
                            "hidden": False,
                        },
                    }
                )
            except Exception:
                try:
                    await __event_emitter__(
                        {
                            "type": "notification",
                            "data": {
                                "type": "info" if not done else "success",
                                "content": str(desc),
                            },
                        }
                    )
                except Exception:
                    pass

        async def debug(msg: str):
            if v.DEBUG:
                await emit_status(f"[ImageCtx DEBUG] {msg}", done=False)

        # Model scoping
        model_id = (
            (__model__ or {}).get("id", "") if isinstance(__model__, dict) else ""
        )
        if v.ONLY_FOR_MODELS and (v.ONLY_FOR_MODELS.lower() not in model_id.lower()):
            await debug("Skipped: ONLY_FOR_MODELS mismatch")
            return body

        messages: List[Dict[str, Any]] = body.get("messages") or []
        last_idx = _last_user_index(messages)

        # 1) DETECT (before any cleaning)
        has_new_images = _last_user_has_images(messages)
        already_injected = _already_injected_for_last_user(messages, v.CONTEXT_HEADER)
        did_vision = False

        # 2) VISION (only if new images)
        if has_new_images and not already_injected and v.VISION_API_BASE:
            gathered = _collect_last_user_images_and_text(messages)
            imgs, user_text = gathered["images"], gathered["user_text"]
            await debug(f"Found {len(imgs)} image(s) on last user turn")
            if imgs:
                await emit_status(
                    f"Analyzing image(s) with {v.VISION_MODEL}…", done=False
                )
                json_mode = (
                    v.OUTPUT_MODE or "JSON"
                ).upper() == "JSON" and v.STRICT_JSON
                vision_messages = _mk_vision_messages(
                    user_text, imgs, json_mode=json_mode
                )
                payload = {
                    "model": v.VISION_MODEL,
                    "messages": vision_messages,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                }
                try:
                    resp = await _post_chat_completions(
                        v.VISION_API_BASE, v.VISION_API_KEY, payload
                    )
                    content = resp["choices"][0]["message"]["content"]
                    await emit_status("Image analysis complete ✓", done=True)
                    # build context
                    if (v.OUTPUT_MODE or "JSON").upper() == "RAW" or not v.STRICT_JSON:
                        ctx_md = f"## {v.CONTEXT_HEADER}\n\n{content.strip()}"
                    else:
                        data = _safe_json(content)
                        ctx_md = f"## {v.CONTEXT_HEADER}\n\n{_to_markdown(data)}"
                    # inject: user + pinned system
                    injected_user = {
                        "role": (
                            v.PREPEND_AS
                            if v.PREPEND_AS in ("user", "system")
                            else "user"
                        ),
                        "content": ctx_md,
                    }
                    insert_at = max(_last_user_index(messages), 0)
                    messages.insert(insert_at, injected_user)
                    if v.PIN_CONTEXT_AT_TOP:
                        messages.insert(0, {"role": "system", "content": ctx_md})
                    did_vision = True
                    await debug("Context injected (user + pinned system)")
                except Exception as e:
                    await emit_status(
                        f"Image analysis failed: {type(e).__name__}", done=True
                    )
                    await debug(f"Vision call error: {type(e).__name__}")

        # 3) CLEAN (after vision, or on turns with no new images)
        if (
            v.CLEAN_ON_EVERY_TURN
            and v.STRIP_IMAGES
            and _conversation_has_images(messages)
        ):
            effective_scope = (v.STRIP_SCOPE or "all").lower()
            if v.FORCE_STRIP_ON_TEXT_ONLY and v.TEXT_ONLY_MODELS:
                for token in [
                    t.strip().lower()
                    for t in v.TEXT_ONLY_MODELS.split(",")
                    if t.strip()
                ]:
                    if token and token in model_id.lower():
                        effective_scope = "all"
                        break
            await debug(f"Cleaning history; scope={effective_scope}")
            # last index may have shifted due to injected message
            last_idx = _last_user_index(messages)
            _strip_images_inplace(messages, effective_scope, last_idx)

        body["messages"] = messages
        if not did_vision and not has_new_images:
            await debug("No new images on last user turn; skipping vision")
        return body
