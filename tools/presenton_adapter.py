# presenton_adapter.py -- MCP bridge for Presenton slide generation
# Author:  Marios Adamidis (FORTHought Lab)

# mcp-presenton-adapter/presenton_adapter.py
import os, sys
from typing import Optional, Any, Dict
import httpx
from urllib.parse import quote
from mcp.server.fastmcp import FastMCP, Context

app = FastMCP("presenton-adapter")

def _canon_base(url: Optional[str], default: str) -> str:
    url = (url or "").strip() or default
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "http://" + url
    return url.rstrip("/")

def _log(ctx: Context, level: str, message: str):
    try:
        try:
            ctx.log(level=level, message=message)
        except TypeError:
            ctx.log(message)
    except Exception:
        print(f"{level.upper()}: {message}", file=sys.stderr, flush=True)

PRESENTON_BASE_URL        = _canon_base(os.getenv("PRESENTON_BASE_URL"), "http://presenton:80")
PUBLIC_FILES_BASE_URL     = _canon_base(os.getenv("PUBLIC_FILES_BASE_URL"), "http://localhost:8084")
FILESERVER_INTERNAL_URL   = _canon_base(os.getenv("FILESERVER_INTERNAL_URL"), "http://fileserver:8000")
FILESERVER_INTERNAL_TOKEN = (os.getenv("FILESERVER_INTERNAL_TOKEN") or "change_me_in_env").strip()

def _mint_public_url_from_presenton_path(ctx: Context, presenton_path: str) -> Optional[str]:
    prefix = "/app_data/"
    if not presenton_path or not presenton_path.startswith(prefix):
        _log(ctx, "warn", f"Presenton returned no usable path: {presenton_path!r}")
        return None

    rel_under_app_data = presenton_path[len(prefix):].lstrip("/")
    relative_path_in_fileserver = f"files/presentations/{rel_under_app_data}"

    try:
        with httpx.Client(timeout=10) as client:
            r = client.post(
                f"{FILESERVER_INTERNAL_URL}/internal/share",
                params={"path": relative_path_in_fileserver},
                headers={"X-Internal-Token": FILESERVER_INTERNAL_TOKEN},
            )
            r.raise_for_status()
            data = r.json()
            url_path = data.get("url")  # server now returns an encoded "/dl/..." path

            if not url_path or not url_path.startswith("/dl/"):
                _log(ctx, "error", f"Fileserver returned invalid 'url'={url_path!r} for path={relative_path_in_fileserver}")
                return None

            # If server already encoded, don't re-encode. If not, encode safely.
            if "%" in url_path:
                encoded_path = url_path
            else:
                rest = url_path[4:]
                try:
                    token, filename = rest.rsplit("/", 1)
                except ValueError:
                    _log(ctx, "error", f"Malformed fileserver url_path (no filename): {url_path!r}")
                    return None
                encoded_path = f"/dl/{quote(token, safe='')}/{quote(filename, safe='')}"

            return PUBLIC_FILES_BASE_URL + encoded_path

    except httpx.HTTPStatusError as e:
        body = ""
        try:
            body = e.response.text
        except Exception:
            pass
        _log(ctx, "error", f"Fileserver responded {e.response.status_code} for {relative_path_in_fileserver}: {body}")
        return None
    except Exception as e:
        _log(ctx, "error", f"Mint signed URL failed for {relative_path_in_fileserver}: {e}")
        return None

@app.tool()
def generate_presentation(
    ctx: Context,
    prompt: str,
    n_slides: Optional[int] = 8,
    language: Optional[str] = "English",
    template: Optional[str] = "general",
    export_as: Optional[str] = "pptx",
    instructions: Optional[str] = None,
    files: Optional[list[str]] = None,
) -> Dict[str, Any]:
    payload = {
        "content": prompt,
        "n_slides": n_slides,
        "language": language,
        "template": template,
        "export_as": export_as,
        "files": files,
    }
    if instructions:
        payload["instructions"] = instructions
    payload = {k: v for k, v in payload.items() if v is not None}

    with httpx.Client(timeout=300) as client:
        r = client.post(f"{PRESENTON_BASE_URL}/api/v1/ppt/presentation/generate", json=payload)
        r.raise_for_status()
        data = r.json()

    presenton_path = (
        data.get("path")
        or data.get("export_path")
        or data.get("output_path")
        or next((v for v in data.values() if isinstance(v, str) and v.startswith("/app_data/")), None)
    )

    public_url = _mint_public_url_from_presenton_path(ctx, presenton_path or "")

    if not public_url:
        _log(ctx, "warn", f"No public_url minted. path={presenton_path!r} keys={list(data.keys())}")

    data["public_url"] = public_url
    if public_url:
        data["message_markdown"] = f"[⬇️ Download the Presentation ({(export_as or 'pptx').upper()})]({public_url})"
    return data

if __name__ == "__main__":
    app.run()
