# fileserver_app.py -- Authenticated file server for exported documents
# Author:  Marios Adamidis (FORTHought Lab)
import os, time, mimetypes, hmac, hashlib, logging, shutil
from pathlib import Path
from typing import Optional, List
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Header, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, Response

# ---- Config ----
FILES_ROOT       = Path(os.getenv("FILES_ROOT", "/shared_data")).resolve()
SECRET           = os.getenv("FILESERVER_SECRET", "change_me")
KEY_VER          = os.getenv("FILESERVER_KEY_VERSION", "v1")
DEFAULT_TTL      = int(os.getenv("FILESERVER_DEFAULT_TTL_SECONDS", "604800"))
INTERNAL_TOKEN   = os.getenv("FILESERVER_INTERNAL_TOKEN", "change_me_long")
LEGACY_PUBLIC_BASE = "files"  # public URL base segment -> /files/...

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fileserver")

app = FastAPI(title="FORTHought Files", docs_url=None, redoc_url=None)

# ---------- Helpers ----------

def _safe_path(rel: str) -> Path:
    rel = (rel or "").lstrip("/").replace("..", "_")
    p = (FILES_ROOT / rel).resolve()
    if not str(p).startswith(str(FILES_ROOT)):
        raise HTTPException(400, "Invalid path")
    return p

def _mime(path: Path) -> Optional[str]:
    m, _ = mimetypes.guess_type(str(path))
    return m or "application/octet-stream"

def _sign(payload: str) -> str:
    return hmac.new(
        f"{KEY_VER}:{SECRET}".encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

def _token_for(relpath: str, exp_ts: int) -> str:
    # legacy-style token: rel|exp|ver|sig (rel may contain slashes/spaces)
    base = f"{relpath}|{exp_ts}|{KEY_VER}"
    sig  = _sign(base)
    return f"{base}|{sig}"

def _parse_legacy_token(tok: str) -> str:
    # returns the relative path (rel) after validation
    try:
        rel, exp_str, ver, sig = tok.split("|", 3)
        exp = int(exp_str)
    except Exception:
        raise HTTPException(403, "Bad token")
    base = f"{rel}|{exp}|{ver}"
    if not hmac.compare_digest(_sign(base), sig):
        raise HTTPException(403, "Invalid signature")
    if time.time() > exp:
        raise HTTPException(403, "Link expired")
    return rel

def _require_internal(token: Optional[str]):
    if not token or token != INTERNAL_TOKEN:
        raise HTTPException(403, "Forbidden")

def _files_root_dir() -> Path:
    """Physical directory that backs the public /files route."""
    d = (FILES_ROOT / LEGACY_PUBLIC_BASE)
    d.mkdir(parents=True, exist_ok=True)
    return d

def _dir_listing_html(dir_path: Path, base_url: str = "/files/") -> str:
    """
    Minimal HTML directory index for a given directory within /files.
    """
    title = f"Index of {base_url}"
    rows: List[str] = []

    # Parent link if not at root
    files_base = _files_root_dir()
    if dir_path != files_base:
        parent_rel = dir_path.relative_to(files_base).as_posix()
        parts = parent_rel.split("/") if parent_rel else []
        parent_url = "/files/" + ("/".join(parts[:-1]) + "/" if len(parts) > 0 else "")
        rows.append('<tr><td><a href="{}">../</a></td><td></td><td></td></tr>'.format(quote(parent_url, safe="/")))

    for child in sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        rel_child = child.relative_to(files_base).as_posix()
        href = "/files/" + rel_child
        display = child.name + ("/" if child.is_dir() else "")
        size = "-" if child.is_dir() else str(child.stat().st_size)
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(child.stat().st_mtime))
        rows.append(
            "<tr>"
            f'<td><a href="{quote(href, safe="/")}">{display}</a></td>'
            f"<td>{mtime}</td><td>{size}</td>"
            "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; padding: 1rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td {{ padding: 0.25rem 0.5rem; border-bottom: 1px solid #eee; }}
  a {{ text-decoration: none; }}
</style>
</head>
<body>
<h1>{title}</h1>
<table>
<tr><th>Name</th><th>Modified</th><th>Size</th></tr>
{''.join(rows)}
</table>
</body>
</html>"""
    return html

# ---------- Security headers ----------

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    resp: Response = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
    return resp

# ---------- Health ----------

@app.get("/health")
def health():
    return {"ok": True}

# ---------- INTERNAL API (Docker network only; requires header) ----------

@app.get("/internal/get")
def internal_get(path: str, x_internal_token: Optional[str] = Header(None)):
    _require_internal(x_internal_token)
    f = _safe_path(path)
    if not f.exists() or not f.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(str(f), media_type=_mime(f), filename=f.name)

@app.post("/internal/upload")
async def internal_upload(
    path: str,
    file: UploadFile = File(...),
    x_internal_token: Optional[str] = Header(None)
):
    """Upload a file from EdgeBox sandbox to fileserver storage"""
    _require_internal(x_internal_token)

    dest = _safe_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        log.info(f"Uploaded: {dest.name} -> {dest}")
        rel = str(dest.relative_to(FILES_ROOT))
        return {
            "success": True,
            "filename": dest.name,
            "path": rel,
            "size": dest.stat().st_size
        }
    except Exception as e:
        log.error(f"Upload failed: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")
    finally:
        file.file.close()

@app.post("/internal/share")
def internal_share(path: str,
                   ttl: Optional[int] = None,
                   x_internal_token: Optional[str] = Header(None)):
    _require_internal(x_internal_token)
    f = _safe_path(path)
    if not f.exists() or not f.is_file():
        raise HTTPException(404, "Not found")

    ttl = int(ttl or DEFAULT_TTL)
    exp = int(time.time()) + max(1, ttl)
    rel = str(f.relative_to(FILES_ROOT))

    token = _token_for(rel, exp)

    # Return an already-encoded URL so clients don't guess/encode
    encoded_url = f"/dl/{quote(token, safe='')}/{quote(f.name, safe='')}"
    return {
        "url": encoded_url,
        "raw": f"/dl/{token}/{f.name}",
        "expires_in": ttl,
        "expires_at": exp
    }

@app.post("/internal/publish_forever")
def internal_publish_forever(path: str,
                             x_internal_token: Optional[str] = Header(None)):
    _require_internal(x_internal_token)
    f = _safe_path(path)
    if not f.exists() or not f.is_file():
        raise HTTPException(404, "Not found")
    ten_years = 60 * 60 * 24 * 365 * 10
    exp = int(time.time()) + ten_years
    rel = str(f.relative_to(FILES_ROOT))
    token = _token_for(rel, exp)
    encoded_url = f"/dl/{quote(token, safe='')}/{quote(f.name, safe='')}"
    return {
        "url": encoded_url,
        "raw": f"/dl/{token}/{f.name}",
        "expires_in": ten_years,
        "expires_at": exp
    }

# ---------- PUBLIC: Signed downloads (/dl/...) ----------

# Allow slashes in {token} robustly with :path
@app.get("/dl/{token:path}/{filename}")
def public_download(token: str, filename: str):
    try:
        rel = _parse_legacy_token(token)  # decode + validate
    except HTTPException as e:
        log.info("Token parse failed: %s", e.detail)
        raise

    f = _safe_path(rel)

    if not f.exists() or not f.is_file():
        log.info("File not found for rel=%r (token ok)", rel)
        raise HTTPException(404, "Not Found")

    return FileResponse(str(f), media_type=_mime(f), filename=f.name)

# ---------- PUBLIC: Legacy /files/... (static) ----------

@app.get("/files")
def files_root_index():
    """Convenience: /files -> index of /files/"""
    base = _files_root_dir()
    html = _dir_listing_html(base, base_url="/files/")
    return HTMLResponse(content=html, status_code=200)

@app.get("/files/")
def files_root_index_slash():
    base = _files_root_dir()
    html = _dir_listing_html(base, base_url="/files/")
    return HTMLResponse(content=html, status_code=200)

@app.get("/files/{path:path}")
def files_public(path: str = ""):
    """
    Public static file server for files under /shared_data/files.
    - GET /files/… serves files
    - GET /files/<dir>/ shows a minimal directory listing
    """
    # Ensure everything resolves inside FILES_ROOT / "files"
    rel_under_files = f"{LEGACY_PUBLIC_BASE}/{path}".rstrip("/")
    target = _safe_path(rel_under_files)

    if not target.exists():
        raise HTTPException(404, "Not Found")

    if target.is_dir():
        # Ensure trailing slash in URL for relative links
        html = _dir_listing_html(target, base_url="/files/" + (path.strip("/") + "/" if path else ""))
        return HTMLResponse(content=html, status_code=200)

    # It's a file
    return FileResponse(str(target), media_type=_mime(target), filename=target.name)
