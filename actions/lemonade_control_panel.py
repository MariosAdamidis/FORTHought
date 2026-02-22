"""
title: Lemonade Control Panel
author: Sawan Srivastava (original); Marios Adamidis / FORTHought Lab (adaptation)
version: 1.0.0
description: Open WebUI Plugin for Querying Lemonade Server Endpoints

"""

import json
import asyncio
from urllib.parse import urlparse, urlunparse
from typing import Optional, Callable, Awaitable, List, Dict, Any

import httpx
from pydantic import BaseModel, Field


class Action:
    class Valves(BaseModel):
        BASE_URL: str = Field(
            default="http://localhost:8000",
            description="Lemonade server base URL (without /api/v1).",
        )
        LEMONADE_API_KEY: str = Field(
            default="",
            description="Optional API key sent as HTTP Bearer auth for Lemonade requests.",
        )
        TIMEOUT_SECONDS: int = Field(
            default=5,
            description="Default timeout for standard API requests (health, stats, listing).",
        )

    def __init__(self):
        self.valves = self.Valves()

    def _build_base_candidates(self, base_url: str) -> List[str]:
        """Return URL candidates, preferring configured BASE_URL and then docker internal fallback."""
        primary = base_url.rstrip("/")
        parsed = urlparse(primary)
        host = (parsed.hostname or "").lower()

        if "localhost" not in host:
            return [primary]

        fallback_host = host.replace("localhost", "host.docker.internal")
        netloc = f"{fallback_host}:{parsed.port}" if parsed.port else fallback_host
        fallback = urlunparse(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        ).rstrip("/")

        return [primary] if fallback == primary else [primary, fallback]

    async def _request_with_fallback(
        self,
        client: httpx.AsyncClient,
        method: str,
        base_candidates: List[str],
        endpoint_path: str,
        *,
        timeout: Optional[int] = None,
        **kwargs,
    ):
        """Try primary URL first; if it fails, retry against docker internal fallback."""
        last_exception = None
        last_index = len(base_candidates) - 1

        for idx, base in enumerate(base_candidates):
            try:
                response = await client.request(
                    method,
                    f"{base}/api/v1{endpoint_path}",
                    timeout=timeout,
                    **kwargs,
                )

                # If primary returns API error, retry fallback candidate once.
                if response.status_code >= 400 and idx < last_index:
                    continue
                return response
            except Exception as e:
                last_exception = e
                if idx == last_index:
                    raise

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Request failed without response")

    async def _emit_status(self, emitter, description: str, done: bool = False):
        if emitter:
            await emitter(
                {"type": "status", "data": {"description": description, "done": done}}
            )

    async def _emit_notification(self, emitter, content: str, ntype: str = "info"):
        if emitter:
            await emitter(
                {"type": "notification", "data": {"type": ntype, "content": content}}
            )

    def _format_model_list(self, data: dict) -> str:
        """Helper to format model JSON into a readable string for the input dialog."""
        try:
            models = data.get("data", [])
            if not models:
                return "No models found."
            lines = []
            for m in models:
                size = m.get("size", "?")
                m_id = m.get("id", "Unknown")
                downloaded = "[Downloaded]" if m.get("downloaded") else ""
                lines.append(f"• {m_id} ({size}GB) {downloaded}")
            limit = 30
            return "\n".join(lines[:limit]) + (
                "\n... (and more)" if len(lines) > limit else ""
            )
        except Exception:
            return "Could not parse model list."

    # --- Visualization Generators ---

    def _generate_gauge_html(
        self, label: str, value: float, max_val: float, unit: str
    ) -> str:
        percent = min(100, max(0, (value / max_val) * 100)) if max_val > 0 else 0
        color = "#ef4444"
        if percent > 30:
            color = "#eab308"
        if percent > 70:
            color = "#22c55e"

        return f"""
        <div style="margin-bottom: 8px;">
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#94a3b8; margin-bottom:2px;">
                <span>{label}</span>
                <span style="color:#f1f5f9; font-weight:600; font-family:monospace;">{value} {unit}</span>
            </div>
            <div style="width:100%; background:#334155; height:6px; border-radius:3px; overflow:hidden;">
                <div style="width:{percent}%; background:{color}; height:100%;"></div>
            </div>
        </div>
        """

    def _make_raw_card(self, title: str, data: Any) -> str:
        try:
            content = json.dumps(data, indent=2)
        except:
            content = str(data)
        return f"""
        <div class="raw-card">
            <div class="raw-header"><span class="raw-title">{title}</span></div>
            <pre>{content}</pre>
        </div>
        """

    def _build_result_html(
        self, title: str, badge: str, content: str, is_error: bool = False
    ) -> str:
        """Simple wrapper for command results (Pull/Delete response)."""
        color = "#ef4444" if is_error else "#22c55e"
        return f"""
        <style>
            .res-panel {{ background: #0f172a; border: 1px solid #1f2937; border-radius: 8px; padding: 12px; font-family: sans-serif; color: #e2e8f0; }}
            .res-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1f2937; padding-bottom: 8px; margin-bottom: 8px; }}
            .res-badge {{ font-size: 0.7rem; background: {color}20; color: {color}; padding: 2px 8px; border-radius: 99px; border: 1px solid {color}40; }}
            .res-pre {{ background: #1e293b50; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 0.75rem; color: #cbd5e1; white-space: pre-wrap; }}
        </style>
        <div class="res-panel">
            <div class="res-header">
                <span style="font-weight:600;">🍋 {title}</span>
                <span class="res-badge">{badge}</span>
            </div>
            <div class="res-pre">{content}</div>
        </div>
        """

    def _build_snapshot_html(
        self, health: dict, stats: dict, system: dict, models: dict
    ) -> str:
        """The main snapshot generator."""
        unique_id = f"lemon_{id(health)}"

        cpu_name = system.get("Processor", "Unknown CPU")
        ram = system.get("Physical Memory", "Unknown RAM")
        os_ver = system.get("OS Version", "Unknown OS")
        devices = system.get("devices", {})

        badges = []
        if devices.get("npu", {}).get("available"):
            badges.append('<span class="badge badge-npu">NPU DETECTED</span>')
        gpu_list = devices.get("amd_dgpu", []) + devices.get("nvidia_dgpu", [])
        if gpu_list or devices.get("amd_igpu", {}).get("available"):
            badges.append('<span class="badge badge-gpu">GPU ACTIVE</span>')
        badges_html = (
            " ".join(badges)
            if badges
            else '<span class="badge" style="opacity:0.5">CPU ONLY</span>'
        )

        tps = stats.get("tokens_per_second", 0)
        ttft = stats.get("time_to_first_token", 0)
        prompt_tokens = stats.get("prompt_tokens", 0)
        input_tokens = stats.get("input_tokens", 0)
        output_tokens = stats.get("output_tokens", 0)
        decode_times = stats.get("decode_token_times", [])
        avg_decode = (sum(decode_times) / len(decode_times)) if decode_times else 0

        tps_fmt = f"{tps:.3f}"
        ttft_fmt = f"{ttft:.4f}"
        avg_decode_fmt = f"{avg_decode:.4f}"

        loaded_models = health.get("all_models_loaded", [])
        active_models_html = ""
        if not loaded_models:
            active_models_html = '<div style="text-align:center; padding:15px; color:#64748b; border:1px dashed #334155; border-radius:6px; font-size:0.75rem;"><i>No models loaded via API yet.</i></div>'
        else:
            for m in loaded_models:
                active_models_html += f"""
                <div class="model-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:600; color:#f59e0b; font-size:0.8rem;">{m.get("model_name", "Unknown")}</span>
                        <span style="font-size:0.6rem; background:#334155; padding:1px 5px; border-radius:4px;">{m.get("type", "llm").upper()}</span>
                    </div>
                    <div style="font-size:0.7rem; color:#94a3b8; margin-top:4px; display:grid; grid-template-columns: 1fr 1fr; gap:4px;">
                        <div>Dev: <span style="color:#e2e8f0;">{m.get("device", "cpu").replace(" ", "+").upper()}</span></div>
                        <div>Rec: {m.get("recipe", "unknown")}</div>
                    </div>
                </div>
                """

        raw_content = ""
        raw_content += self._make_raw_card("Health Status", health)
        raw_content += self._make_raw_card("Inference Stats", stats)
        raw_content += self._make_raw_card("System Info", system)
        raw_content += self._make_raw_card("Model Inventory", models)

        return f"""
        <style>
            :root {{ --bg-card: #0f172a; --border: #1e293b; --text: #e2e8f0; --accent: #f59e0b; --text-muted: #64748b; }}
            #{unique_id} .dash-container {{ font-family: -apple-system, sans-serif; color: var(--text); background: #020617; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; font-size: 14px; }}
            #{unique_id} .dash-header {{ background: var(--bg-card); padding: 10px 16px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }}
            #{unique_id} .grid {{ display: grid; grid-template-columns: 1fr; gap: 1px; background: var(--border); }}
            @media (min-width: 600px) {{ #{unique_id} .grid {{ grid-template-columns: 1fr 1fr; }} }}
            #{unique_id} .panel {{ background: #0f172a; padding: 16px; display: flex; flex-direction: column; gap: 12px; }}
            #{unique_id} .stat-big {{ font-size: 1.4rem; font-weight: 700; color: #f8fafc; font-family: monospace; }}
            #{unique_id} .stat-label {{ font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px; }}
            #{unique_id} .stat-sub {{ font-size: 0.7rem; color: #94a3b8; }}
            #{unique_id} .badge {{ font-size: 0.65rem; padding: 2px 8px; border-radius: 12px; font-weight: 600; border: 1px solid transparent; }}
            #{unique_id} .badge-npu {{ background: #7c3aed20; color: #a78bfa; border-color: #7c3aed40; }}
            #{unique_id} .badge-gpu {{ background: #05966920; color: #34d399; border-color: #05966940; }}
            #{unique_id} .model-card {{ background: #1e293b50; border: 1px solid #334155; border-left: 3px solid var(--accent); border-radius: 4px; padding: 8px; margin-bottom: 8px; }}
            #{unique_id} .toggle-btn {{ background: #1e293b; border: 1px solid #334155; color: #cbd5e1; font-size: 0.7rem; padding: 4px 10px; border-radius: 4px; cursor: pointer; }}
            #{unique_id} .toggle-btn:hover {{ background: #334155; color: white; }}
            #{unique_id} .raw-card {{ background: #1e293b50; border: 1px solid #334155; border-radius: 6px; margin-bottom: 10px; overflow: hidden; }}
            #{unique_id} .raw-header {{ background: #1e293b; padding: 6px 10px; border-bottom: 1px solid #334155; }}
            #{unique_id} .raw-title {{ font-size: 0.75rem; color: #94a3b8; font-weight: bold; text-transform: uppercase; }}
            #{unique_id} pre {{ margin: 0; padding: 10px; background: #0b1120; color: #cbd5e1; font-size: 0.7rem; overflow: auto; max-height: 200px; white-space: pre-wrap; word-wrap: break-word; }}
            #{unique_id} .stat-row {{ display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 1px solid #1e293b; padding-bottom: 8px; }}
        </style>
        <div id="{unique_id}">
            <div class="dash-container">
                <div class="dash-header">
                    <div style="display:flex; flex-direction:column;">
                        <span style="font-weight:700; font-size:0.9rem;">🍋 Lemonade Panel</span>
                        <span style="font-size:0.65rem; color:#64748b;">{os_ver}</span>
                    </div>
                    <button class="toggle-btn" onclick="var v=document.getElementById('visual-{unique_id}');var r=document.getElementById('raw-{unique_id}');if(v.style.display==='none'){{v.style.display='grid';r.style.display='none';this.innerText='Show Raw Data';}}else{{v.style.display='none';r.style.display='block';this.innerText='Show Visuals';}}">Show Raw Data</button>
                </div>
                <div id="visual-{unique_id}" class="grid">
                    <div class="panel">
                        <div>
                            {self._generate_gauge_html("Throughput", tps, 60, "T/s")}
                            <div class="stat-sub">Speed: {tps_fmt} tokens/sec</div>
                        </div>
                        <div class="stat-row">
                            <div><div class="stat-label">Initial Latency</div><div class="stat-big">{ttft_fmt}<span style="font-size:0.9rem; color:#64748b">s</span></div></div>
                            <div style="text-align:right;"><div class="stat-label">Avg Decode</div><div class="stat-big">{avg_decode_fmt}<span style="font-size:0.9rem; color:#64748b">s</span></div></div>
                        </div>
                        <div>
                            <div class="stat-label" style="margin-bottom:4px;">Token Usage</div>
                            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px;">
                                <div style="background:#1e293b; padding:6px; border-radius:4px; text-align:center;"><div class="stat-label">Input</div><div style="color:#e2e8f0; font-weight:600;">{input_tokens}</div></div>
                                <div style="background:#1e293b; padding:6px; border-radius:4px; text-align:center;"><div class="stat-label">Output</div><div style="color:#e2e8f0; font-weight:600;">{output_tokens}</div></div>
                            </div>
                        </div>
                        <div style="margin-top:auto;">
                            <div class="stat-label">System</div>
                            <div style="font-size:0.75rem; color:#e2e8f0;">{cpu_name}</div>
                            <div style="display:flex; gap:8px; margin-top:4px;">{badges_html}</div>
                        </div>
                    </div>
                    <div class="panel">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <div class="stat-label">Active Memory Slots</div>
                            <div class="stat-sub">{ram}</div>
                        </div>
                        {active_models_html}
                    </div>
                </div>
                <div id="raw-{unique_id}" style="display:none; padding: 12px;">{raw_content}</div>
            </div>
        </div>
        """

    async def action(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
        **kwargs,
    ):
        base_candidates = self._build_base_candidates(self.valves.BASE_URL)
        endpoint_key = ""

        if __user__["role"] != "admin":
            await self._emit_notification(
                __event_emitter__,
                "Admin privileges required for Lemonade Control.",
                "error",
            )
            return {"content": "This action requires admin privileges"}

        await self._emit_status(__event_emitter__, "Waiting for input...", False)

        if __event_call__:
            try:
                user_input = await __event_call__(
                    {
                        "type": "input",
                        "data": {
                            "title": "Lemonade Control",
                            "message": "Enter command (pull, delete, health) or leave EMPTY for snapshot view:",
                            "placeholder": "Leave empty for snapshot view",
                        },
                    }
                )
                if isinstance(user_input, str):
                    endpoint_key = user_input.strip().lower()
            except Exception as e:
                pass

        headers = {"Connection": "close"}
        api_key = self.valves.LEMONADE_API_KEY.strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(
            headers=headers, timeout=self.valves.TIMEOUT_SECONDS
        ) as client:

            if endpoint_key in ["pull", "delete"]:
                payload = {}

                await self._emit_status(
                    __event_emitter__, f"Fetching info for {endpoint_key}...", False
                )

                try:
                    list_path = "/models" + (
                        "?show_all=true" if endpoint_key == "pull" else ""
                    )
                    list_resp = await self._request_with_fallback(
                        client,
                        "GET",
                        base_candidates,
                        list_path,
                    )
                    model_list_str = (
                        self._format_model_list(list_resp.json())
                        if list_resp.status_code == 200
                        else "Error fetching list."
                    )
                except:
                    model_list_str = "Could not fetch model list."

                if __event_call__:
                    model_name = await __event_call__(
                        {
                            "type": "input",
                            "data": {
                                "title": f"{endpoint_key.title()} Model",
                                "message": f"Models Available:\n\n{model_list_str}\n\nEnter ID to {endpoint_key}:",
                                "placeholder": "Qwen3-14B-GGUF",
                            },
                        }
                    )
                    if model_name:
                        payload = {"model_name": model_name.strip()}
                    else:
                        return body  # Cancelled

                # Execute with LONG Timeout
                timeout = 1800 if endpoint_key == "pull" else 180
                await self._emit_status(
                    __event_emitter__,
                    f"Executing {endpoint_key} (Timeout: {timeout}s)...",
                    False,
                )

                try:
                    resp = await self._request_with_fallback(
                        client,
                        "POST",
                        base_candidates,
                        f"/{endpoint_key}",
                        json=payload,
                        timeout=timeout,
                    )
                    try:
                        resp_json = json.dumps(resp.json(), indent=2)
                    except:
                        resp_json = resp.text

                    html_out = self._build_result_html(
                        f"{endpoint_key.title()} Result",
                        str(resp.status_code),
                        resp_json,
                        is_error=(resp.status_code >= 400),
                    )
                except Exception as e:
                    html_out = self._build_result_html(
                        "Error", "Fail", str(e), is_error=True
                    )

                if body.get("messages") and isinstance(body["messages"], list):
                    body["messages"][-1]["content"] += f"\n\n{f'```html{html_out}```'}"

            else:
                await self._emit_status(
                    __event_emitter__, "Fetching Telemetry...", False
                )

                results = await asyncio.gather(
                    self._request_with_fallback(
                        client, "GET", base_candidates, "/health"
                    ),
                    self._request_with_fallback(
                        client, "GET", base_candidates, "/stats"
                    ),
                    self._request_with_fallback(
                        client, "GET", base_candidates, "/system-info"
                    ),
                    self._request_with_fallback(
                        client, "GET", base_candidates, "/models"
                    ),
                    return_exceptions=True,
                )

                data = []
                for res in results:
                    if isinstance(res, Exception):
                        data.append({"error": str(res)})
                    elif hasattr(res, "status_code") and res.status_code != 200:
                        data.append(
                            {"error": f"HTTP {res.status_code}", "text": res.text}
                        )
                    elif hasattr(res, "json"):
                        try:
                            data.append(res.json())
                        except:
                            data.append({})
                    else:
                        data.append({})

                health_d, stats_d, system_d, models_d = data

                html_dash = self._build_snapshot_html(
                    health_d, stats_d, system_d, models_d
                )

                if body.get("messages") and isinstance(body["messages"], list):
                    body["messages"][-1]["content"] += f"\n\n```html\n{html_dash}\n```"

        await self._emit_status(__event_emitter__, "Done", True)
        return body
