"""
title: Google Gemini Pipeline (v4.0 - Token Optimized)
author: owndev, olivier-lacroix (original); Marios Adamidis / FORTHought Lab (v4.0 adaptation)
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
version: 1.0.0
required_open_webui_version: 0.6.26
license: Apache License 2.0
requirements: google-genai, httpx, aiofiles, pillow, cryptography
description: |
  Gemini pipeline optimized for Gemini 3 Flash Preview on Vertex AI free-tier.
  Focused on minimizing TPM (tokens-per-minute) usage during tool loops.

changelog_v4:
  - FIX: _retry_with_backoff now catches 429/RESOURCE_EXHAUSTED (was ServerError-only)
  - FIX: Proper exponential backoff with jitter (was fixed 2^n, no jitter)
  - NEW: Tool-round thinking reduction (minimal thinking during tool decisions)
  - NEW: History trimming before tool loop (configurable max messages)
  - NEW: Streaming after tool loop — final response streams instead of blocking
  - REMOVED: All image generation code (unused, ~300 lines)
  - IMPROVED: Budget tracking uses actual response token counts from API metadata
  - IMPROVED: Budget-break summary uses trimmed context, not full bloated history
  - KEPT: All v3.0 safety (truncation, budget, status emitter)
  - KEPT: All v2.9 fixes (registry merge, case insensitive, dot notation, MCP)
"""

import os
import re
import time
import asyncio
import base64
import hashlib
import logging
import io
import uuid
import json
import inspect
import random
import aiofiles
import httpx
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError, APIError
from typing import List, Union, Optional, Dict, Any, Tuple, AsyncIterator, Callable
from pydantic_core import core_schema
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
from open_webui.env import SRC_LOG_LEVELS
from fastapi import Request
from open_webui.models.users import UserModel, Users

# ═══════════════════════════════════════════════════════════════════════════════
# ENCRYPTED STRING TYPE
# ═══════════════════════════════════════════════════════════════════════════════


class EncryptedStr(str):
    """A string type that automatically handles encryption and decryption."""

    @classmethod
    def _get_encryption_key(cls) -> Optional[bytes]:
        secret = os.getenv("WEBUI_SECRET_KEY")
        if not secret:
            return None
        hashed_key = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(hashed_key)

    @classmethod
    def encrypt(cls, value: str) -> str:
        if not value or value.startswith("encrypted:"):
            return value
        key = cls._get_encryption_key()
        if not key:
            return value
        f = Fernet(key)
        encrypted = f.encrypt(value.encode())
        return f"encrypted:{encrypted.decode()}"

    @classmethod
    def decrypt(cls, value: str) -> str:
        if not value or not value.startswith("encrypted:"):
            return value
        key = cls._get_encryption_key()
        if not key:
            return value[len("encrypted:") :]
        try:
            encrypted_part = value[len("encrypted:") :]
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_part.encode())
            return decrypted.decode()
        except (InvalidToken, Exception):
            return value

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.chain_schema(
                    [
                        core_schema.str_schema(),
                        core_schema.no_info_plain_validator_function(
                            lambda value: cls(cls.encrypt(value) if value else value)
                        ),
                    ]
                ),
            ],
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: str(instance)
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


class Pipe:
    """
    Google Gemini Pipeline v4.0 — Token-Optimized for Gemini 3 Flash / Vertex AI.
    """

    class Valves(BaseModel):
        # ── API Configuration ──────────────────────────────────────────────
        BASE_URL: str = Field(
            default=os.getenv(
                "GOOGLE_GENAI_BASE_URL", "https://generativelanguage.googleapis.com/"
            ),
            description="Base URL for the Google Generative AI API.",
        )
        GOOGLE_API_KEY: EncryptedStr = Field(
            default=os.getenv("GOOGLE_API_KEY", ""),
            description="API key for Google Generative AI.",
        )
        API_VERSION: str = Field(
            default=os.getenv("GOOGLE_API_VERSION", "v1alpha"),
            description="API version for Google Generative AI.",
        )

        # ── Vertex AI ──────────────────────────────────────────────────────
        USE_VERTEX_AI: bool = Field(
            default=os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true",
            description="Whether to use Google Cloud Vertex AI.",
        )
        VERTEX_PROJECT: str | None = Field(
            default=os.getenv("GOOGLE_CLOUD_PROJECT"),
            description="The Google Cloud project ID.",
        )
        VERTEX_LOCATION: str = Field(
            default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            description="The Google Cloud region.",
        )
        VERTEX_AI_RAG_STORE: str | None = Field(
            default=os.getenv("GOOGLE_VERTEX_AI_RAG_STORE"),
            description="Vertex AI RAG Store path.",
        )

        # ── MCP Tool Execution ─────────────────────────────────────────────
        MCP_BASE_URL: str = Field(
            default=os.getenv("MCP_BASE_URL", "http://localhost:12008/metamcp"),
            description="Base URL for MetaMCP server.",
        )
        MCP_TOOL_PREFIX: str = Field(
            default=os.getenv("MCP_TOOL_PREFIX", "Skills_"),
            description="Prefix used in MCP tool names.",
        )
        MCP_TIMEOUT: int = Field(
            default=int(os.getenv("MCP_TIMEOUT", "60")),
            description="Timeout in seconds for MCP HTTP calls.",
        )

        # ── Generation Settings ────────────────────────────────────────────
        STREAMING_ENABLED: bool = Field(
            default=os.getenv("GOOGLE_STREAMING_ENABLED", "true").lower() == "true",
            description="Enable streaming responses.",
        )
        INCLUDE_THOUGHTS: bool = Field(
            default=os.getenv("GOOGLE_INCLUDE_THOUGHTS", "true").lower() == "true",
            description="Enable Gemini thoughts outputs.",
        )
        THINKING_BUDGET: int = Field(
            default=int(os.getenv("GOOGLE_THINKING_BUDGET", "-1")),
            description="Thinking budget for Gemini 2.5 models.",
        )
        THINKING_LEVEL: str = Field(
            default=os.getenv("GOOGLE_THINKING_LEVEL", ""),
            description="Thinking level for Gemini 3 models ('minimal','low','medium','high').",
        )
        USE_PERMISSIVE_SAFETY: bool = Field(
            default=os.getenv("GOOGLE_USE_PERMISSIVE_SAFETY", "false").lower()
            == "true",
            description="Use permissive safety settings.",
        )
        MODEL_CACHE_TTL: int = Field(
            default=int(os.getenv("GOOGLE_MODEL_CACHE_TTL", "600")),
            description="Time in seconds to cache the model list.",
        )

        # ── Retry & Backoff (v4.0 — now handles 429) ──────────────────────
        RETRY_COUNT: int = Field(
            default=int(os.getenv("GOOGLE_RETRY_COUNT", "4")),
            description="Number of retry attempts for API errors (including 429).",
        )
        RETRY_BASE_DELAY: float = Field(
            default=float(os.getenv("GOOGLE_RETRY_BASE_DELAY", "2.0")),
            description="Base delay in seconds for exponential backoff.",
        )
        RETRY_MAX_DELAY: float = Field(
            default=float(os.getenv("GOOGLE_RETRY_MAX_DELAY", "30.0")),
            description="Maximum delay in seconds between retries.",
        )

        # ── Context Safety (v3.0+) ────────────────────────────────────────
        TOOL_RESPONSE_MAX_CHARS: int = Field(
            default=int(os.getenv("GOOGLE_TOOL_RESPONSE_MAX_CHARS", "8000")),
            description="Max chars per individual tool response before truncation.",
        )
        TOOL_LOOP_BUDGET_CHARS: int = Field(
            default=int(os.getenv("GOOGLE_TOOL_LOOP_BUDGET_CHARS", "400000")),
            description="Max total chars of tool responses accumulated in one tool loop.",
        )
        HISTORY_TOOL_RESULT_MAX_CHARS: int = Field(
            default=int(os.getenv("GOOGLE_HISTORY_TOOL_RESULT_MAX_CHARS", "4000")),
            description="Max chars for historical tool results in conversation context.",
        )

        # ── v4.0: Tool Loop Optimization ───────────────────────────────────
        TOOL_LOOP_THINKING_LEVEL: str = Field(
            default=os.getenv("GOOGLE_TOOL_LOOP_THINKING_LEVEL", "minimal"),
            description="Thinking level during tool-decision rounds. 'minimal' saves ~80%% output tokens.",
        )
        TOOL_LOOP_MAX_HISTORY: int = Field(
            default=int(os.getenv("GOOGLE_TOOL_LOOP_MAX_HISTORY", "10")),
            description="Max conversation messages to keep when entering tool loop. Older messages trimmed.",
        )

        # ── System & Image Input ───────────────────────────────────────────
        DEFAULT_SYSTEM_PROMPT: str = Field(
            default=os.getenv("GOOGLE_DEFAULT_SYSTEM_PROMPT", ""),
            description="Default system prompt applied to all chats.",
        )
        ENABLE_FORWARD_USER_INFO_HEADERS: bool = Field(
            default=os.getenv(
                "GOOGLE_ENABLE_FORWARD_USER_INFO_HEADERS", "false"
            ).lower()
            == "true",
            description="Whether to forward user information headers.",
        )
        IMAGE_MAX_SIZE_MB: float = Field(
            default=float(os.getenv("GOOGLE_IMAGE_MAX_SIZE_MB", "15.0")),
            description="Maximum image size in MB before compression.",
        )
        IMAGE_MAX_DIMENSION: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_MAX_DIMENSION", "2048")),
            description="Maximum width or height in pixels before resizing.",
        )
        IMAGE_COMPRESSION_QUALITY: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_COMPRESSION_QUALITY", "85")),
            description="JPEG compression quality.",
        )
        IMAGE_ENABLE_OPTIMIZATION: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_ENABLE_OPTIMIZATION", "true").lower()
            == "true",
            description="Enable intelligent image optimization.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name: str = "Google Gemini: "
        self.log = logging.getLogger("google_ai.pipe")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self._model_cache: Optional[List[Dict[str, str]]] = None
        self._model_cache_time: float = 0
        self.user: Optional[UserModel] = None

    # ══════════════════════════════════════════════════════════════════════════
    # MCP TOOL EXECUTION
    # ══════════════════════════════════════════════════════════════════════════

    def _parse_mcp_tool_name(self, tool_name: str) -> Tuple[Optional[str], str]:
        """Parse MCP tool name into (namespace, actual_tool_name)."""
        prefix = self.valves.MCP_TOOL_PREFIX
        # Format 1: Skills_<namespace>__<tool>
        if tool_name.startswith(prefix):
            without_prefix = tool_name[len(prefix) :]
            if "__" in without_prefix:
                ns, tool = without_prefix.split("__", 1)
                return (ns, tool)
        # Format 2: <namespace>.<tool>
        if "." in tool_name and not tool_name.startswith("."):
            ns, tool = tool_name.split(".", 1)
            return (ns, tool)
        return (None, tool_name)

    async def _execute_mcp_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an MCP tool via HTTP JSON-RPC."""
        namespace, actual_tool = self._parse_mcp_tool_name(tool_name)
        if not namespace:
            return {"error": f"Cannot determine MCP namespace from: {tool_name}"}

        mcp_url = f"{self.valves.MCP_BASE_URL}/{namespace}/mcp"
        self.log.info(f"[MCP] {actual_tool} @ {namespace} -> {mcp_url}")

        mcp_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": actual_tool, "arguments": args},
            "id": str(uuid.uuid4()),
        }

        try:
            async with httpx.AsyncClient(
                timeout=float(self.valves.MCP_TIMEOUT)
            ) as client:
                response = await client.post(
                    mcp_url,
                    json=mcp_request,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    },
                )
                if response.status_code != 200:
                    return {
                        "error": f"MCP HTTP {response.status_code}: {response.text[:500]}"
                    }

                mcp_response = response.json()

                if "result" in mcp_response:
                    result = mcp_response["result"]
                    if isinstance(result, dict) and "content" in result:
                        content = result["content"]
                        if isinstance(content, list):
                            texts = [
                                item.get("text", "")
                                for item in content
                                if isinstance(item, dict) and item.get("type") == "text"
                            ]
                            if texts:
                                return {"output": "\n".join(texts)}
                    return (
                        result if isinstance(result, dict) else {"output": str(result)}
                    )
                elif "error" in mcp_response:
                    error = mcp_response["error"]
                    msg = (
                        error.get("message", str(error))
                        if isinstance(error, dict)
                        else str(error)
                    )
                    return {"error": f"MCP error: {msg}"}
                else:
                    return {"output": str(mcp_response)}

        except httpx.TimeoutException:
            return {"error": f"MCP timeout after {self.valves.MCP_TIMEOUT}s"}
        except httpx.ConnectError as e:
            return {"error": f"Cannot connect to MCP at {mcp_url}: {e}"}
        except Exception as e:
            self.log.error(f"[MCP] Unexpected error: {e}", exc_info=True)
            return {"error": f"MCP execution failed: {e}"}

    # ══════════════════════════════════════════════════════════════════════════
    # CONTEXT SAFETY HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _truncate_tool_response(self, result: dict, max_chars: int) -> dict:
        """Truncate a tool response dict to prevent context blowup."""
        if not isinstance(result, dict):
            s = str(result)
            if len(s) > max_chars:
                return {"output": s[:max_chars] + f"\n[...truncated, {len(s)} total]"}
            return result

        try:
            result_str = json.dumps(result, default=str)
        except Exception:
            result_str = str(result)

        if len(result_str) <= max_chars:
            return result

        original_len = len(result_str)
        truncated = {}
        budget_per_key = max(max_chars // max(len(result), 1), 500)

        for k, v in result.items():
            if isinstance(v, str) and len(v) > budget_per_key:
                truncated[k] = v[:budget_per_key] + f"\n[...truncated, {len(v)} total]"
            elif isinstance(v, list) and len(str(v)) > budget_per_key:
                kept = []
                running = 0
                for item in v:
                    item_str = (
                        json.dumps(item, default=str)
                        if not isinstance(item, str)
                        else item
                    )
                    if running + len(item_str) > budget_per_key and kept:
                        break
                    kept.append(item)
                    running += len(item_str)
                truncated[k] = kept
                if len(kept) < len(v):
                    truncated[f"_{k}_note"] = f"Showing {len(kept)}/{len(v)} items"
            elif (
                isinstance(v, dict) and len(json.dumps(v, default=str)) > budget_per_key
            ):
                truncated[k] = str(v)[:budget_per_key] + "...[truncated]"
            else:
                truncated[k] = v

        self.log.info(f"[TRUNCATE] {original_len} -> ~{max_chars} chars")
        return truncated

    def _estimate_contents_chars(self, gemini_contents: list) -> int:
        """Rough char estimate of Gemini contents for budget tracking."""
        total = 0
        for item in gemini_contents:
            for part in getattr(item, "parts", None) or []:
                if hasattr(part, "text") and part.text:
                    total += len(part.text)
                elif hasattr(part, "function_call") and part.function_call:
                    total += len(str(part.function_call))
                elif hasattr(part, "function_response") and part.function_response:
                    total += len(str(part.function_response))
        return total

    # ══════════════════════════════════════════════════════════════════════════
    # TOOL SCHEMA HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _sanitize_schema_for_gemini(
        self, schema: Any, depth: int = 0
    ) -> Dict[str, Any]:
        """Convert JSON Schema into a Gemini-compatible subset."""
        if not isinstance(schema, dict):
            return {"type": "string"}

        stype = schema.get("type")

        if stype == "object" or ("properties" in schema and not stype):
            props = schema.get("properties", {})
            new_props = {
                name: self._sanitize_schema_for_gemini(sub, depth + 1)
                for name, sub in props.items()
            }
            result: Dict[str, Any] = {"type": "object", "properties": new_props}
            if isinstance(schema.get("required"), list):
                result["required"] = [r for r in schema["required"] if r in new_props]
            return result

        if stype == "array":
            return {
                "type": "array",
                "items": self._sanitize_schema_for_gemini(
                    schema.get("items", {}), depth + 1
                ),
            }

        if stype in ("string", "number", "integer", "boolean"):
            res: Dict[str, Any] = {"type": stype}
            if "enum" in schema and isinstance(schema["enum"], list):
                res["enum"] = schema["enum"]
            if "description" in schema and isinstance(schema["description"], str):
                res["description"] = schema["description"]
            return res

        return {"type": "string"}

    def _build_function_declarations(
        self, tools: dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build Gemini function_declarations from Open WebUI tools dict."""
        decls: List[Dict[str, Any]] = []
        for name, tool_def in (tools or {}).items():
            if name.startswith("_"):
                continue

            spec = (
                tool_def.get("definition")
                or tool_def.get("schema")
                or tool_def.get("json_schema")
                or tool_def.get("tool")
                or tool_def.get("spec")
                or {}
            )
            fn_spec = spec.get("function", spec) if isinstance(spec, dict) else {}
            fn_name = fn_spec.get("name") or name
            description = fn_spec.get("description", "")
            raw_params = fn_spec.get("parameters") or {}
            if not isinstance(raw_params, dict):
                raw_params = {}

            sanitized = self._sanitize_schema_for_gemini(raw_params)
            if sanitized.get("type") != "object":
                sanitized = {"type": "object", "properties": {}}

            decls.append(
                {"name": fn_name, "description": description, "parameters": sanitized}
            )
        return decls

    def _format_tools_to_string(self, tools: dict[str, Any]) -> str:
        """Format tools as text for {{TOOLS}} injection."""
        parts = []
        for name, tool_def in (tools or {}).items():
            if name.startswith("_"):
                continue
            spec = (
                tool_def.get("definition")
                or tool_def.get("schema")
                or tool_def.get("json_schema")
                or tool_def.get("tool")
                or tool_def.get("spec")
            )
            if spec and isinstance(spec, dict):
                fn_spec = spec.get("function", spec)
                t_name = fn_spec.get("name", name)
                desc = fn_spec.get("description", "No description.")
                params = json.dumps(fn_spec.get("parameters", {}), indent=2)
                parts.append(
                    f"Tool: {t_name}\nDescription: {desc}\nParameters: {params}"
                )
                continue
            func = tool_def.get("callable")
            if func:
                doc = inspect.getdoc(func) or "No description."
                sig = str(inspect.signature(func))
                parts.append(f"Tool: {name}\nDescription: {doc}\nParameters: {sig}")
        return "\n\n".join(parts)

    # ══════════════════════════════════════════════════════════════════════════
    # SYSTEM PROMPT & IMAGE INPUT HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _combine_system_prompts(
        self, user_system_prompt: Optional[str]
    ) -> Optional[str]:
        default = self.valves.DEFAULT_SYSTEM_PROMPT.strip()
        user = user_system_prompt.strip() if user_system_prompt else ""
        if default and user:
            return f"{default}\n\n{user}"
        return default or user or None

    def _optimize_image_for_api(self, image_data: str) -> str:
        """Compress/resize an image if it exceeds size limits."""
        if not self.valves.IMAGE_ENABLE_OPTIMIZATION:
            return image_data
        try:
            if image_data.startswith("data:"):
                header, encoded = image_data.split(",", 1)
            else:
                encoded = image_data
                header = "data:image/png;base64"

            image_bytes = base64.b64decode(encoded)
            if len(image_bytes) / (1024 * 1024) < self.valves.IMAGE_MAX_SIZE_MB:
                return image_data

            with Image.open(io.BytesIO(image_bytes)) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.thumbnail((self.valves.IMAGE_MAX_DIMENSION,) * 2)
                buf = io.BytesIO()
                img.save(
                    buf, format="JPEG", quality=self.valves.IMAGE_COMPRESSION_QUALITY
                )
                new_b64 = base64.b64encode(buf.getvalue()).decode()
                return f"data:image/jpeg;base64,{new_b64}"
        except Exception:
            return image_data

    # ══════════════════════════════════════════════════════════════════════════
    # CLIENT & MODEL MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════════

    def _get_client(self) -> genai.Client:
        self._validate_api_key()
        if self.valves.USE_VERTEX_AI:
            return genai.Client(
                vertexai=True,
                project=self.valves.VERTEX_PROJECT,
                location=self.valves.VERTEX_LOCATION,
            )

        headers = {}
        if self.valves.ENABLE_FORWARD_USER_INFO_HEADERS and self.user:
            sanitize = lambda v: (
                re.sub(r"[\x00-\x1F\x7F]", "", str(v)).strip()[:255] if v else ""
            )
            attrs = {
                "X-OpenWebUI-User-Name": getattr(self.user, "name", None),
                "X-OpenWebUI-User-Id": getattr(self.user, "id", None),
                "X-OpenWebUI-User-Email": getattr(self.user, "email", None),
                "X-OpenWebUI-User-Role": getattr(self.user, "role", None),
            }
            headers = {k: sanitize(v) for k, v in attrs.items() if v}

        return genai.Client(
            api_key=EncryptedStr.decrypt(self.valves.GOOGLE_API_KEY),
            http_options=types.HttpOptions(
                api_version=self.valves.API_VERSION,
                base_url=self.valves.BASE_URL,
                headers=headers,
            ),
        )

    def _validate_api_key(self) -> None:
        if self.valves.USE_VERTEX_AI:
            if not self.valves.VERTEX_PROJECT:
                raise ValueError("VERTEX_PROJECT is not set.")
        elif not self.valves.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set.")

    def strip_prefix(self, model_name: str) -> str:
        return re.sub(r"^(?:.*/|[^.]*\.)", "", model_name)

    def get_google_models(self, force_refresh: bool = False) -> List[Dict[str, str]]:
        now = time.time()
        if (
            not force_refresh
            and self._model_cache is not None
            and (now - self._model_cache_time) < self.valves.MODEL_CACHE_TTL
        ):
            return self._model_cache

        try:
            client = self._get_client()
            models = client.models.list()
            available = []
            for model in models:
                actions = model.supported_actions
                if actions is None or "generateContent" in actions:
                    mid = self.strip_prefix(model.name)
                    available.append({"id": mid, "name": model.display_name or mid})

            # Ensure key models are always listed
            existing_ids = {m["id"] for m in available}
            for mid in [
                "gemini-flash-latest",
                "gemini-2.5-pro",
                "gemini-3-flash-preview",
                "gemini-3-pro-preview",
            ]:
                if mid not in existing_ids:
                    available.append({"id": mid, "name": mid})

            self._model_cache = available
            self._model_cache_time = now
            return self._model_cache
        except Exception as e:
            self.log.exception(f"Could not fetch models: {e}")
            return [{"id": "error", "name": f"Error: {e}"}]

    def _is_gemini3(self, model_id: str) -> bool:
        return "gemini-3-" in model_id.lower()

    def _check_thinking_support(self, model_id: str) -> bool:
        # Image-gen models don't support thinking, but we removed those
        return True

    def _validate_thinking_level(self, level: str) -> Optional[str]:
        if not level:
            return None
        normalized = level.strip().lower()
        return (
            normalized if normalized in ("minimal", "low", "medium", "high") else None
        )

    def _validate_thinking_budget(self, budget: int) -> int:
        if budget == -1:
            return -1
        if budget == 0:
            return 0
        if budget > 0:
            return min(budget, 32768)
        return -1

    def pipes(self) -> List[Dict[str, str]]:
        try:
            self.name = "Google Gemini: "
            return self.get_google_models()
        except Exception as e:
            return [{"id": "error", "name": str(e)}]

    def _prepare_model_id(self, model_id: str) -> str:
        original = model_id
        model_id = self.strip_prefix(model_id)
        if not model_id.startswith("gemini-"):
            models_list = self.get_google_models()
            found = next((m["id"] for m in models_list if m["name"] == original), None)
            if found and found.startswith("gemini-"):
                model_id = found
            else:
                raise ValueError(f"Invalid model ID: {original}")
        return model_id

    # ══════════════════════════════════════════════════════════════════════════
    # CONTENT PREPARATION & HISTORY
    # ══════════════════════════════════════════════════════════════════════════

    def _prepare_content(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Convert OpenWebUI messages to Gemini content format."""
        user_system = next(
            (msg["content"] for msg in messages if msg.get("role") == "system"), None
        )
        system_message = self._combine_system_prompts(user_system)
        contents = []
        DUMMY_SIGNATURE = "skip_thought_signature_validator"

        for message in messages:
            role = message.get("role")
            if role == "system":
                continue

            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
            api_role = "model" if role == "assistant" else "user"

            # Handle Tool Calls from history
            if role == "assistant" and tool_calls:
                raw_parts = []
                first_fc = False
                if content and isinstance(content, str):
                    raw_parts.append({"text": content})

                for tc in tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name")
                    fn_args = fn.get("arguments")
                    if isinstance(fn_args, str):
                        try:
                            fn_args = json.loads(fn_args)
                        except Exception:
                            pass
                    fc_part = {"function_call": {"name": fn_name, "args": fn_args}}
                    if not first_fc:
                        fc_part["thought_signature"] = DUMMY_SIGNATURE
                        first_fc = True
                    raw_parts.append(fc_part)
                contents.append({"role": api_role, "parts": raw_parts})
                continue

            # Handle Tool Results from history
            if role == "tool":
                tool_name = message.get("name", "unknown_tool")
                try:
                    response_content = (
                        json.loads(content)
                        if isinstance(content, str)
                        else {"content": content}
                    )
                except Exception:
                    response_content = {"result": str(content)}

                # v3.0: Truncate large historical tool results
                hist_max = self.valves.HISTORY_TOOL_RESULT_MAX_CHARS
                try:
                    rc_str = (
                        json.dumps(response_content, default=str)
                        if isinstance(response_content, dict)
                        else str(response_content)
                    )
                    if len(rc_str) > hist_max:
                        if isinstance(response_content, dict):
                            response_content = self._truncate_tool_response(
                                response_content, hist_max
                            )
                        else:
                            response_content = {
                                "result": str(response_content)[:hist_max]
                                + "\n[...truncated]"
                            }
                except Exception:
                    pass

                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": tool_name,
                                    "response": response_content,
                                }
                            }
                        ],
                    }
                )
                continue

            # Handle standard content
            parts = []
            if isinstance(content, list):
                parts.extend(self._process_multimodal_content(content))
            elif isinstance(content, str) and content:
                parts.append({"text": content})
            else:
                continue

            if parts:
                contents.append({"role": api_role, "parts": parts})

        return contents, system_message

    def _process_multimodal_content(
        self, content_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        parts = []
        for item in content_list:
            if item.get("type") == "text":
                parts.append({"text": item.get("text", "")})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {}).get("url", "")
                if image_url.startswith("data:image"):
                    try:
                        optimized = self._optimize_image_for_api(image_url)
                        header, encoded = optimized.split(",", 1)
                        mime = header.split(":")[1].split(";")[0]
                        parts.append(
                            {"inline_data": {"mime_type": mime, "data": encoded}}
                        )
                    except Exception as e:
                        self.log.warning(f"Image error: {e}")
                        parts.append({"text": "[Image processing failed]"})
                else:
                    parts.append({"text": f"[Image URL not supported: {image_url}]"})
        return parts

    # ══════════════════════════════════════════════════════════════════════════
    # GENERATION CONFIG BUILDER
    # ══════════════════════════════════════════════════════════════════════════

    def _build_generation_config(
        self,
        body: Dict[str, Any],
        system_instruction: Optional[str],
        __metadata__: Dict[str, Any],
        tools: dict[str, Any] | None = None,
        model_id: str = "",
        disable_native_tools: bool = False,
        thinking_override: Optional[str] = None,
    ) -> types.GenerateContentConfig:
        """
        Build generation config. thinking_override allows tool-round thinking reduction.
        """
        params = __metadata__.get("params", {}) or {}
        native_mode = (
            not disable_native_tools
            and tools
            and params.get("function_calling") == "native"
        )

        cfg: Dict[str, Any] = {
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "top_k": body.get("top_k"),
            "max_output_tokens": body.get("max_tokens"),
            "stop_sequences": body.get("stop") or None,
            "system_instruction": system_instruction,
        }

        # Thinking config
        if self._check_thinking_support(model_id):
            try:
                tc_params = {}
                tc_params["include_thoughts"] = (
                    body.get("include_thoughts", True)
                    if self.valves.INCLUDE_THOUGHTS
                    else False
                )

                if self._is_gemini3(model_id):
                    # v4.0: Allow override for tool rounds
                    level_str = (
                        thinking_override
                        or body.get("reasoning_effort")
                        or self.valves.THINKING_LEVEL
                    )
                    level = self._validate_thinking_level(level_str)
                    if level:
                        tc_params["thinking_level"] = level
                else:
                    budget = self._validate_thinking_budget(
                        body.get("thinking_budget")
                        if body.get("thinking_budget") is not None
                        else self.valves.THINKING_BUDGET
                    )
                    if thinking_override == "minimal":
                        budget = 0  # Disable thinking for 2.5 tool rounds
                    if budget != -1:
                        tc_params["thinking_budget"] = budget

                cfg["thinking_config"] = types.ThinkingConfig(**tc_params)
            except Exception as e:
                self.log.warning(f"[CONFIG] Thinking config error: {e}")

        # Safety
        if self.valves.USE_PERMISSIVE_SAFETY:
            cfg["safety_settings"] = [
                types.SafetySetting(category=cat, threshold="BLOCK_NONE")
                for cat in [
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                ]
            ]

        # Google Search grounding
        features = __metadata__.get("features", {}) or {}
        if features.get("google_search_tool", False):
            cfg.setdefault("tools", []).append(
                types.Tool(google_search=types.GoogleSearch())
            )

        # Native tool declarations
        if native_mode:
            decls = self._build_function_declarations(tools or {})
            if decls:
                self.log.info(f"[CONFIG] Registering {len(decls)} native tools")
                cfg.setdefault("tools", []).append({"function_declarations": decls})

        filtered = {k: v for k, v in cfg.items() if v is not None}
        return types.GenerateContentConfig(**filtered)

    # ══════════════════════════════════════════════════════════════════════════
    # GROUNDING & STREAMING
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _format_grounding_chunks_as_sources(chunks):
        sources = []
        for c in chunks:
            ctx = getattr(c, "web", None) or getattr(c, "retrieved_context", None)
            if not ctx:
                continue
            uri = getattr(ctx, "uri", None)
            title = getattr(ctx, "title", None) or "Source"
            sources.append(
                {
                    "source": {"name": title, "url": uri, "type": "web_search_results"},
                    "document": ["Click the link to view the content."],
                    "metadata": [{"source": title}],
                }
            )
        return sources

    async def _process_grounding_metadata(
        self, metadata_list, text, emitter, emit_replace=True
    ):
        grounding_chunks, web_queries, grounding_supports = [], [], []

        for md in metadata_list:
            if md.grounding_chunks:
                grounding_chunks.extend(md.grounding_chunks)
            if md.web_search_queries:
                web_queries.extend(md.web_search_queries)
            if md.grounding_supports:
                grounding_supports.extend(md.grounding_supports)

        if grounding_chunks:
            sources = self._format_grounding_chunks_as_sources(grounding_chunks)
            await emitter({"type": "chat:completion", "data": {"sources": sources}})

        if web_queries:
            await emitter(
                {
                    "type": "status",
                    "data": {
                        "action": "web_search",
                        "description": "Grounded with Google Search",
                        "urls": [
                            f"https://www.google.com/search?q={q}" for q in web_queries
                        ],
                    },
                }
            )

        replaced_text = None
        if grounding_supports:
            ENCODING = "utf-8"
            text_bytes = text.encode(ENCODING)
            last_idx = 0
            cited = []
            for support in grounding_supports:
                cited.append(
                    text_bytes[last_idx : support.segment.end_index].decode(ENCODING)
                )
                footnotes = "".join(
                    [f"[{i + 1}]" for i in support.grounding_chunk_indices]
                )
                cited.append(f" {footnotes}")
                last_idx = support.segment.end_index
            if last_idx < len(text_bytes):
                cited.append(text_bytes[last_idx:].decode(ENCODING))
            replaced_text = "".join(cited)
            if emit_replace:
                await emitter({"type": "replace", "data": {"content": replaced_text}})

        if not emit_replace:
            return replaced_text if replaced_text is not None else text

    async def _handle_streaming_response(self, iterator, emitter, model_id: str = ""):
        """Handle streaming, emit thoughts as status, yield final content."""

        async def emit(event_type: str, data: Dict[str, Any]):
            if emitter:
                try:
                    await emitter({"type": event_type, "data": data})
                except Exception:
                    pass

        await emit("chat:start", {"role": "assistant"})

        answer_chunks: List[str] = []
        thought_chunks: List[str] = []
        grounding_list = []
        thinking_started: Optional[float] = None

        async for chunk in iterator:
            if not chunk.candidates:
                continue
            candidate = chunk.candidates[0]
            if candidate.grounding_metadata:
                grounding_list.append(candidate.grounding_metadata)

            for part in getattr(candidate.content, "parts", []) or []:
                if getattr(part, "thought", False) and getattr(part, "text", None):
                    if thinking_started is None:
                        thinking_started = time.time()
                    thought_chunks.append(part.text)
                    preview = part.text.replace("\n", " ").strip()[:120]
                    await emit(
                        "status",
                        {
                            "action": "thinking",
                            "description": f"Thinking… {preview}",
                            "done": False,
                            "hidden": False,
                        },
                    )
                elif getattr(part, "text", None):
                    answer_chunks.append(part.text)
                    await emit(
                        "chat:message:delta",
                        {"role": "assistant", "content": part.text},
                    )

        final_text = "".join(answer_chunks)
        if grounding_list and emitter:
            cited = await self._process_grounding_metadata(
                grounding_list, final_text, emitter, emit_replace=False
            )
            final_text = cited or final_text

        final_content = final_text
        if thought_chunks:
            duration = int(max(0, time.time() - (thinking_started or time.time())))
            thought = "".join(thought_chunks).strip()
            quoted = "\n".join(f"> {line}" for line in thought.split("\n"))
            final_content = (
                f"<details>\n<summary>Thought ({duration}s)</summary>\n\n{quoted}\n\n</details>"
                + final_text
            )

        await emit("status", {"action": "thinking", "done": True, "hidden": True})
        await emit("replace", {"role": "assistant", "content": final_content})
        await emit(
            "chat:message",
            {"role": "assistant", "content": final_content, "done": True},
        )
        await emit(
            "chat:finish", {"role": "assistant", "content": final_content, "done": True}
        )

        yield final_content

    # ══════════════════════════════════════════════════════════════════════════
    # SAFETY & RETRY (v4.0 — HANDLES 429)
    # ══════════════════════════════════════════════════════════════════════════

    def _get_safety_block_message(self, response: Any) -> Optional[str]:
        if (
            getattr(response, "prompt_feedback", None)
            and response.prompt_feedback.block_reason
        ):
            return f"[Blocked: Prompt Safety — {response.prompt_feedback.block_reason.name}]"
        if not response.candidates:
            return "[Blocked by safety settings or no candidates generated]"
        candidate = response.candidates[0]
        fr = getattr(candidate, "finish_reason", None)
        if fr == types.FinishReason.SAFETY:
            blocking = next((r for r in candidate.safety_ratings if r.blocked), None)
            reason = f" ({blocking.category.name})" if blocking else ""
            return f"[Blocked by safety settings{reason}]"
        if fr == types.FinishReason.PROHIBITED_CONTENT:
            return "[Blocked: prohibited content policy]"
        return None

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        v4.0: Determine if an error is retryable.
        429 RESOURCE_EXHAUSTED comes as ClientError on Vertex AI.
        500/503 come as ServerError.
        """
        if isinstance(error, ServerError):
            return True
        if isinstance(error, ClientError):
            err_str = str(error).lower()
            return (
                "429" in err_str or "resource_exhausted" in err_str or "rate" in err_str
            )
        if isinstance(error, APIError):
            err_str = str(error).lower()
            return "429" in err_str or "resource_exhausted" in err_str
        return False

    async def _retry_with_backoff(self, func, *args, emitter=None, **kwargs):
        """
        v4.0: Exponential backoff with jitter. Retries 429 AND 5xx.
        Emits status so user sees retry progress instead of a dead UI.
        """
        max_retries = self.valves.RETRY_COUNT
        base_delay = self.valves.RETRY_BASE_DELAY
        max_delay = self.valves.RETRY_MAX_DELAY
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if not self._is_retryable_error(e):
                    raise  # Non-retryable: raise immediately

                last_exc = e
                if attempt >= max_retries:
                    break

                # Exponential backoff with jitter
                delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)
                self.log.warning(
                    f"[RETRY] {type(e).__name__}: {str(e)[:120]}. "
                    f"Attempt {attempt + 1}/{max_retries}, waiting {delay:.1f}s"
                )

                if emitter:
                    try:
                        await emitter(
                            {
                                "type": "status",
                                "data": {
                                    "action": "retry",
                                    "description": f"Rate limited, retrying in {delay:.0f}s… (attempt {attempt + 1}/{max_retries})",
                                    "done": False,
                                },
                            }
                        )
                    except Exception:
                        pass

                await asyncio.sleep(delay)

        if last_exc:
            raise last_exc

    # ══════════════════════════════════════════════════════════════════════════
    # FORMAT FINAL RESPONSE (non-streaming)
    # ══════════════════════════════════════════════════════════════════════════

    async def _format_final_response(
        self, response: Any, start_ts: float, emitter: Callable
    ) -> str:
        safety = self._get_safety_block_message(response)
        if safety:
            return safety

        candidate = response.candidates[0]
        parts = getattr(getattr(candidate, "content", None), "parts", [])
        if not parts:
            return "[No content generated]"

        answer_parts: List[str] = []
        thought_parts: List[str] = []

        for part in parts:
            if getattr(part, "thought", False) and getattr(part, "text", None):
                thought_parts.append(part.text)
            elif getattr(part, "text", None):
                answer_parts.append(part.text)

        final = "".join(answer_parts)

        # Grounding
        if getattr(candidate, "grounding_metadata", None):
            cited = await self._process_grounding_metadata(
                [candidate.grounding_metadata], final, emitter, emit_replace=False
            )
            final = cited or final

        result = ""
        if thought_parts:
            duration = int(max(0, time.time() - start_ts))
            thought = "".join(thought_parts).strip()
            quoted = "\n".join(f"> {line}" for line in thought.split("\n"))
            result += f"<details>\n<summary>Thought ({duration}s)</summary>\n\n{quoted}\n\n</details>"

        result += final
        return result or "[No content generated]"

    # ══════════════════════════════════════════════════════════════════════════
    # v4.0: HISTORY TRIMMING FOR TOOL LOOPS
    # ══════════════════════════════════════════════════════════════════════════

    def _trim_contents_for_tool_loop(
        self, contents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Trim conversation history to reduce tokens sent on each tool-loop round.
        Keeps the last N messages (configurable). Always preserves the final user message.
        """
        max_msgs = self.valves.TOOL_LOOP_MAX_HISTORY
        if len(contents) <= max_msgs:
            return contents

        # Always keep the last user message + recent context
        trimmed = contents[-max_msgs:]

        # Ensure first message is role=user (Gemini requires user-first)
        if trimmed and trimmed[0].get("role") == "model":
            trimmed = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "[Earlier conversation trimmed for context efficiency]"
                        }
                    ],
                }
            ] + trimmed

        removed = len(contents) - len(trimmed)
        self.log.info(
            f"[v4.0 TRIM] Removed {removed} old messages, keeping {len(trimmed)}"
        )
        return trimmed

    # ══════════════════════════════════════════════════════════════════════════
    # NATIVE TOOL LOOP (v4.0)
    # ══════════════════════════════════════════════════════════════════════════

    async def _run_native_tool_loop(
        self,
        client: genai.Client,
        model_id: str,
        contents: List[Dict[str, Any]],
        body: Dict[str, Any],
        system_instruction: Optional[str],
        __metadata__: Dict[str, Any],
        effective_tools: dict[str, Any],
        __event_emitter__: Callable,
        __request__: Optional[Request],
        __user__: Optional[dict],
        disable_native_tools: bool,
        max_rounds: int = 20,
    ) -> str:
        """
        v4.0 Native tool loop with:
        - Reduced thinking during tool-decision rounds
        - History trimming before loop entry
        - Proper 429 retry with backoff
        - Streaming for the final response
        """
        self.log.info(
            f"[TOOL LOOP] Starting | model={model_id} | max_rounds={max_rounds}"
        )

        # ── Build tool registry ────────────────────────────────────────────
        tool_registry = {}
        for key, tool_def in (effective_tools or {}).items():
            if key.startswith("_"):
                continue
            spec = (
                tool_def.get("definition")
                or tool_def.get("schema")
                or tool_def.get("json_schema")
                or tool_def.get("tool")
                or tool_def.get("spec")
                or {}
            )
            name = (
                spec.get("function", {}).get("name")
                if isinstance(spec, dict) and "function" in spec
                else spec.get("name") if isinstance(spec, dict) else key
            )
            tool_registry[name or key] = tool_def

        def find_tool_ci(registry, name):
            if name in registry:
                return registry[name], name
            lower = name.lower()
            for rn, rd in registry.items():
                if rn.lower() == lower:
                    return rd, rn
            return None, None

        # ── v4.0: Trim history before entering loop ───────────────────────
        trimmed_contents = self._trim_contents_for_tool_loop(contents)

        # ── Build generation configs ──────────────────────────────────────
        # Tool rounds: minimal thinking (saves ~80% output tokens)
        tool_round_config = self._build_generation_config(
            body,
            system_instruction,
            __metadata__,
            effective_tools,
            model_id=model_id,
            disable_native_tools=disable_native_tools,
            thinking_override=self.valves.TOOL_LOOP_THINKING_LEVEL,
        )
        # Final round: full thinking (user-configured level)
        final_config = self._build_generation_config(
            body,
            system_instruction,
            __metadata__,
            tools=None,  # No tools for final response
            model_id=model_id,
            disable_native_tools=True,
        )

        # ── Convert to Gemini Content objects ─────────────────────────────
        gemini_contents = []
        for msg in trimmed_contents:
            gemini_contents.append(
                types.Content(
                    role=msg["role"],
                    parts=[
                        types.Part(**p) if isinstance(p, dict) else p
                        for p in msg["parts"]
                    ],
                )
            )

        # ── Loop state ────────────────────────────────────────────────────
        html_artifacts = []
        last_tool_calls = []
        consecutive_same = 0
        MAX_CONSECUTIVE_SAME = 3
        accumulated_chars = 0
        budget = self.valves.TOOL_LOOP_BUDGET_CHARS
        total_input_tokens = 0
        total_output_tokens = 0

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "tool_processing",
                        "description": "Processing with tools...",
                        "done": False,
                    },
                }
            )

        for round_idx in range(max_rounds):
            self.log.info(f"[TOOL LOOP] ── Round {round_idx + 1}/{max_rounds} ──")

            # ── API call with retry (now handles 429!) ────────────────────
            async def call_model():
                return await client.aio.models.generate_content(
                    model=model_id, contents=gemini_contents, config=tool_round_config
                )

            try:
                response = await self._retry_with_backoff(
                    call_model, emitter=__event_emitter__
                )
            except Exception as e:
                self.log.error(f"[TOOL LOOP] API call failed after retries: {e}")
                return f"Error: {e}"

            # ── Track token usage from response metadata ──────────────────
            usage = getattr(response, "usage_metadata", None)
            if usage:
                inp = getattr(usage, "prompt_token_count", 0) or 0
                out = getattr(usage, "candidates_token_count", 0) or 0
                cached = getattr(usage, "cached_content_token_count", 0) or 0
                total_input_tokens += inp
                total_output_tokens += out
                self.log.info(
                    f"[TOOL LOOP] Tokens this round: input={inp} (cached={cached}), output={out} | "
                    f"Running total: in={total_input_tokens}, out={total_output_tokens}"
                )

            # ── Extract function calls ────────────────────────────────────
            function_calls = []
            if response.candidates:
                parts = getattr(response.candidates[0].content, "parts", []) or []
                for part in parts:
                    if getattr(part, "function_call", None):
                        fc = part.function_call
                        self.log.info(
                            f"[TOOL LOOP] Function call: {fc.name}({dict(fc.args) if fc.args else {}})"
                        )
                        function_calls.append(fc)

            # ── No function calls → generate final response ───────────────
            if not function_calls:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "action": "tool_processing",
                                "description": "Generating response...",
                                "done": True,
                            },
                        }
                    )

                self.log.info(
                    f"[TOOL LOOP] Complete. Total tokens: in={total_input_tokens}, out={total_output_tokens}"
                )

                # v4.0: Stream the final response for better UX
                if self.valves.STREAMING_ENABLED:
                    try:

                        async def get_final_stream():
                            return await client.aio.models.generate_content_stream(
                                model=model_id,
                                contents=gemini_contents,
                                config=final_config,
                            )

                        iterator = await self._retry_with_backoff(
                            get_final_stream, emitter=__event_emitter__
                        )
                        final_text = ""
                        async for chunk in self._handle_streaming_response(
                            iterator, __event_emitter__, model_id
                        ):
                            final_text = chunk
                        if html_artifacts:
                            final_text += "\n\n" + "\n\n".join(html_artifacts)
                        return final_text
                    except Exception as stream_err:
                        self.log.warning(
                            f"[TOOL LOOP] Streaming final response failed, falling back: {stream_err}"
                        )

                # Fallback: non-streaming final response
                start_ts = time.time()
                final_text = await self._format_final_response(
                    response, start_ts, __event_emitter__
                )
                if html_artifacts:
                    final_text += "\n\n" + "\n\n".join(html_artifacts)
                return final_text

            # ── Loop detection ────────────────────────────────────────────
            current_calls = [
                (fc.name, str(dict(fc.args) if fc.args else {}))
                for fc in function_calls
            ]
            if current_calls == last_tool_calls:
                consecutive_same += 1
                if consecutive_same >= MAX_CONSECUTIVE_SAME:
                    return f"Error: Tool loop detected — model keeps requesting: {[c[0] for c in current_calls]}"
            else:
                consecutive_same = 0
            last_tool_calls = current_calls

            # ── Append model's response to context ────────────────────────
            gemini_contents.append(response.candidates[0].content)

            # ── Execute each tool call ────────────────────────────────────
            tool_response_parts = []

            for fc in function_calls:
                tool_name = fc.name
                args = dict(fc.args or {})

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "action": "tool_processing",
                                "description": f"Running {tool_name}...",
                                "done": False,
                            },
                        }
                    )

                # Chart rendering shortcut
                if "chart" in tool_name.lower() and "cfg_json" in args:
                    try:
                        config = json.loads(args["cfg_json"])
                        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><script src="https://cdn.jsdelivr.net/npm/chart.js"></script><style>body{{margin:0;padding:0;background:#fff;overflow:hidden}}canvas{{width:100%!important;height:100%!important}}</style></head><body><div style="position:relative;width:100%;height:440px"><canvas id="c"></canvas></div><script>try{{new Chart(document.getElementById('c'),{json.dumps(config)})}}catch(e){{document.body.innerHTML='<b style="color:red">'+e+'</b>'}}</script></body></html>"""
                        b64 = base64.b64encode(html.encode()).decode()
                        html_artifacts.append(
                            f'<iframe src="data:text/html;base64,{b64}" style="width:100%;height:460px;border:none;border-radius:8px;background:white" sandbox="allow-scripts"></iframe>'
                        )
                        tool_response_parts.append(
                            types.Part.from_function_response(
                                name=tool_name,
                                response={
                                    "status": "success",
                                    "message": "Chart rendered.",
                                },
                            )
                        )
                        continue
                    except Exception as e:
                        self.log.error(f"[TOOL LOOP] Chart render failed: {e}")

                # Case-insensitive lookup + MCP fallback
                tool_def, matched = find_tool_ci(tool_registry, tool_name)
                if matched and matched != tool_name:
                    tool_name = matched

                if not tool_def:
                    # Try MCP fallback
                    if (
                        "__" in tool_name
                        or self.valves.MCP_TOOL_PREFIX in tool_name
                        or "." in tool_name
                    ):
                        result = await self._execute_mcp_tool(tool_name, args)
                        result = self._ensure_dict(result)
                        result = self._truncate_tool_response(
                            result, self.valves.TOOL_RESPONSE_MAX_CHARS
                        )
                        tool_response_parts.append(
                            types.Part.from_function_response(
                                name=tool_name, response=result
                            )
                        )
                        accumulated_chars += len(json.dumps(result, default=str))
                        continue
                    else:
                        tool_response_parts.append(
                            types.Part.from_function_response(
                                name=tool_name,
                                response={
                                    "error": f"Tool '{tool_name}' not found. Available: {list(tool_registry.keys())}"
                                },
                            )
                        )
                        continue

                func = tool_def.get("callable") if isinstance(tool_def, dict) else None

                if not func:
                    # No callable → MCP HTTP
                    result = await self._execute_mcp_tool(tool_name, args)
                    result = self._ensure_dict(result)
                    result = self._truncate_tool_response(
                        result, self.valves.TOOL_RESPONSE_MAX_CHARS
                    )
                    tool_response_parts.append(
                        types.Part.from_function_response(
                            name=tool_name, response=result
                        )
                    )
                    accumulated_chars += len(json.dumps(result, default=str))
                    continue

                # Execute native callable
                result = await self._execute_callable(
                    func, args, __event_emitter__, __request__, __user__
                )

                # Handle HTML responses
                if hasattr(result, "body") and hasattr(result, "status_code"):
                    try:
                        body_str = (
                            result.body.decode("utf-8")
                            if isinstance(result.body, bytes)
                            else str(result.body)
                        )
                        b64 = base64.b64encode(body_str.encode()).decode()
                        html_artifacts.append(
                            f'<iframe src="data:text/html;base64,{b64}" style="width:100%;height:500px;border:none"></iframe>'
                        )
                        result = {
                            "status": "success",
                            "message": "HTML content displayed.",
                        }
                    except Exception:
                        result = {"output": str(result)}

                result = self._ensure_dict(result)
                result = self._truncate_tool_response(
                    result, self.valves.TOOL_RESPONSE_MAX_CHARS
                )
                accumulated_chars += len(json.dumps(result, default=str))
                tool_response_parts.append(
                    types.Part.from_function_response(name=tool_name, response=result)
                )

            # ── Append tool responses to context ──────────────────────────
            gemini_contents.append(
                types.Content(role="user", parts=tool_response_parts)
            )

            # ── Budget check ──────────────────────────────────────────────
            if accumulated_chars > budget:
                self.log.warning(
                    f"[BUDGET] Exceeded: {accumulated_chars} > {budget}. Breaking."
                )
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "action": "tool_processing",
                                "description": "Context limit reached, summarizing...",
                                "done": True,
                            },
                        }
                    )

                # v4.0: Use final_config (no tools) to force a summary, not a tool loop
                try:

                    async def get_summary():
                        return await client.aio.models.generate_content(
                            model=model_id,
                            contents=gemini_contents,
                            config=final_config,
                        )

                    summary = await self._retry_with_backoff(
                        get_summary, emitter=__event_emitter__
                    )
                    if summary.candidates:
                        parts = getattr(summary.candidates[0].content, "parts", [])
                        texts = [
                            p.text
                            for p in (parts or [])
                            if hasattr(p, "text") and p.text
                        ]
                        if texts:
                            return "".join(texts)
                except Exception as e:
                    self.log.error(f"[BUDGET] Summary call failed: {e}")
                return "Tool results gathered but context limit reached. Please try a more specific question."

            self.log.info(
                f"[TOOL LOOP] Round {round_idx + 1} done, {len(tool_response_parts)} responses added"
            )

        return "Error: Tool loop exceeded maximum iterations."

    def _ensure_dict(self, result: Any) -> dict:
        """Ensure a tool result is a JSON-serializable dict."""
        if isinstance(result, list):
            return {"result": result}
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    return {"result": parsed}
            except (json.JSONDecodeError, TypeError):
                pass
        if not isinstance(result, dict):
            return {"output": str(result)}
        try:
            json.dumps(result)
        except (TypeError, ValueError):
            return {"output": str(result)}
        return result

    async def _execute_callable(self, func, args: dict, emitter, request, user) -> Any:
        """Execute a native callable with proper arg injection."""
        try:
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            has_kwargs = any(str(p).startswith("**") for p in sig.parameters.values())
        except ValueError:
            param_names = []
            has_kwargs = True

        filtered = {
            k: v
            for k, v in args.items()
            if not param_names or k in param_names or has_kwargs
        }

        injectables = {
            "__event_emitter__": emitter,
            "__request__": request,
            "__user__": user,
            "__metadata__": {},
        }
        if param_names:
            for k, v in injectables.items():
                if k in param_names:
                    filtered[k] = v

        try:
            if inspect.iscoroutinefunction(func):
                return await func(**filtered)
            else:
                return await asyncio.to_thread(func, **filtered)
        except Exception as e:
            import traceback

            return {"error": str(e), "traceback": traceback.format_exc()[:500]}

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════════

    async def pipe(
        self,
        body: Dict[str, Any],
        __metadata__: dict[str, Any],
        __event_emitter__: Callable,
        __tools__: dict[str, Any] | None,
        __request__: Optional[Request] = None,
        __user__: Optional[dict] = None,
    ) -> Union[str, AsyncIterator[str]]:
        """Main pipeline entry point."""
        self.user = Users.get_user_by_id(__user__["id"]) if __user__ else None

        try:
            model_id = self._prepare_model_id(body.get("model", ""))
        except ValueError as ve:
            return f"Model Error: {ve}"

        stream = body.get("stream", False)
        messages = body.get("messages", [])
        params = __metadata__.get("params", {}) or {}

        self.log.info(
            f"[PIPE] model={model_id} stream={stream} messages={len(messages)}"
        )

        # ── Tool Merging (v2.9+) ──────────────────────────────────────────
        effective_tools = (__tools__ or {}).copy()

        # Merge from __metadata__['tools']
        meta_tools = __metadata__.get("tools")
        if meta_tools:
            if isinstance(meta_tools, dict):
                for k, v in meta_tools.items():
                    if k not in effective_tools:
                        effective_tools[k] = v
            elif isinstance(meta_tools, list):
                for tool in meta_tools:
                    fn_def = (
                        tool.get("function") if tool.get("type") == "function" else tool
                    )
                    if isinstance(fn_def, dict):
                        name = fn_def.get("name")
                        if name and name not in effective_tools:
                            effective_tools[name] = {
                                "definition": {"function": fn_def},
                                "spec": fn_def,
                            }

        # Merge from body['tools'] (Router-injected schemas)
        for tool in body.get("tools", []):
            fn_def = tool.get("function") if tool.get("type") == "function" else tool
            if isinstance(fn_def, dict):
                name = fn_def.get("name")
                if name and name not in effective_tools:
                    effective_tools[name] = {
                        "definition": {"function": fn_def},
                        "spec": {"function": fn_def},
                    }

        if effective_tools:
            self.log.info(
                f"[PIPE] {len(effective_tools)} tools: {list(effective_tools.keys())}"
            )

        # ── System Instruction ────────────────────────────────────────────
        user_system = next(
            (msg["content"] for msg in messages if msg.get("role") == "system"), None
        )
        system_instruction = self._combine_system_prompts(user_system)

        disable_native_tools = False
        if system_instruction and "{{TOOLS}}" in system_instruction and effective_tools:
            self.log.info("[PIPE] Detected {{TOOLS}} — using manual injection")
            system_instruction = system_instruction.replace(
                "{{TOOLS}}", self._format_tools_to_string(effective_tools)
            )
            disable_native_tools = True

        # ── Prepare Content ───────────────────────────────────────────────
        contents, _ = self._prepare_content(messages)

        native_tools_active = (
            not disable_native_tools
            and effective_tools
            and params.get("function_calling") == "native"
        )

        self.log.info(f"[PIPE] native_tools={native_tools_active}")

        # ── Native Tool Loop (non-streaming during tool rounds) ───────────
        if native_tools_active:
            return await self._run_native_tool_loop(
                client=self._get_client(),
                model_id=model_id,
                contents=contents,
                body=body,
                system_instruction=system_instruction,
                __metadata__=__metadata__,
                effective_tools=effective_tools,
                __event_emitter__=__event_emitter__,
                __request__=__request__,
                __user__=__user__,
                disable_native_tools=disable_native_tools,
            )

        # ── Standard Generation ───────────────────────────────────────────
        generation_config = self._build_generation_config(
            body,
            system_instruction,
            __metadata__,
            tools=effective_tools,
            model_id=model_id,
            disable_native_tools=disable_native_tools,
        )
        client = self._get_client()

        # Streaming
        if stream and self.valves.STREAMING_ENABLED:

            async def get_stream():
                return await client.aio.models.generate_content_stream(
                    model=model_id, contents=contents, config=generation_config
                )

            try:
                iterator = await self._retry_with_backoff(
                    get_stream, emitter=__event_emitter__
                )
                return self._handle_streaming_response(
                    iterator, __event_emitter__, model_id
                )
            except Exception as e:
                self.log.exception(f"Streaming error: {e}")
                return f"Error during streaming: {e}"

        # Non-streaming
        try:
            start_ts = time.time()

            async def get_resp():
                return await client.aio.models.generate_content(
                    model=model_id, contents=contents, config=generation_config
                )

            response = await self._retry_with_backoff(
                get_resp, emitter=__event_emitter__
            )
            return await self._format_final_response(
                response, start_ts, __event_emitter__
            )

        except (ClientError, ServerError, APIError) as api_error:
            return f"{type(api_error).__name__}: {api_error}"
        except ValueError as ve:
            return f"Configuration error: {ve}"
        except Exception as e:
            self.log.exception(f"Unexpected error: {e}")
            return f"An error occurred: {e}"
