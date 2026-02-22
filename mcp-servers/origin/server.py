#!/usr/bin/env python3
"""
OriginMCP Server
Author: Marios Adamidis (FORTHought Lab)
Standalone OriginLab data engine: inspect (with user parameter rows),
extract, merge power series, single & multi-peak fitting, graphing, export.
Diagnostics in inspect, auto x_range, min_separation guard,
       nm↔eV auto-conversion, column quality filter, fallback grouping from names.
"""
import os
import io
import re
import json
import uuid
import datetime
import traceback
import urllib.parse
import threading
import hashlib
import sys
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.special import wofz

# -----------------------------------------------------------------------------
# Unbuffered output for Windows
# -----------------------------------------------------------------------------
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

# -----------------------------------------------------------------------------
# Build fingerprint
# -----------------------------------------------------------------------------
BUILD_ID = "2026-02-21T20:00:00Z-origin-mcp-v10.9.1-keyfixed"
_THIS_FILE = os.path.abspath(__file__)
try:
    with open(_THIS_FILE, "rb") as f:
        FILE_SHA12 = hashlib.sha1(f.read()).hexdigest()[:12]
except Exception:
    FILE_SHA12 = "unknown"

print(f"[origin] BUILD={BUILD_ID} SHA={FILE_SHA12}", flush=True)

# -----------------------------------------------------------------------------
# OriginPro import
# -----------------------------------------------------------------------------
try:
    import originpro as op
    ORIGIN_AVAILABLE = True
except ImportError:
    op = None
    ORIGIN_AVAILABLE = False
    print("WARNING: originpro not installed", flush=True)

# -----------------------------------------------------------------------------
# Configuration - All from environment, NO hardcoded secrets
# -----------------------------------------------------------------------------
OWUI_URL = os.getenv("OWUI_URL", "http://100.117.144.23:8081").rstrip("/")
OWUI_PUBLIC_URL = os.getenv("OWUI_PUBLIC_URL", "").rstrip("/")
AUTH_TOKEN = os.getenv("JWT_TOKEN", "") or os.getenv("OPEN_WEBUI_API_KEY", "")

if not AUTH_TOKEN:
    print("WARNING: No JWT_TOKEN set - file downloads will fail", flush=True)

DEBUG_TRACEBACK = os.getenv("DEBUG_TRACEBACK", "0").lower() in ("1", "true")
MAX_ERROR_CHARS = int(os.getenv("MAX_ERROR_CHARS", "300"))
MAX_TOOL_TEXT_CHARS = int(os.getenv("MAX_TOOL_TEXT_CHARS", "4000"))

EXPORT_DIR = os.getenv("FILE_EXPORT_DIR", r"C:\forthought\originmcp\exports").rstrip("/\\")
EXPORT_MODE = os.getenv("EXPORT_MODE", "owui").strip().lower()
os.makedirs(EXPORT_DIR, exist_ok=True)

ORIGIN_LOCK = threading.Lock()

# Dedup cache: prevents identical expensive calls from re-running
# Key: (tool_name, frozenset of args), Value: (result_dict, timestamp)
_RESULT_CACHE: Dict[str, Any] = {}
_CACHE_MAX = 20
_CACHE_TTL = 300  # 5 minutes

def _cache_key(tool: str, **kwargs) -> str:
    """Create stable cache key from tool name and arguments."""
    import json
    args_str = json.dumps(kwargs, sort_keys=True, default=str)
    return f"{tool}:{args_str}"

def _cache_get(key: str) -> Optional[Dict]:
    """Return cached result if fresh, else None."""
    import time
    entry = _RESULT_CACHE.get(key)
    if entry and (time.time() - entry[1]) < _CACHE_TTL:
        return entry[0]
    if entry:
        del _RESULT_CACHE[key]
    return None

def _cache_set(key: str, result: Dict) -> None:
    """Cache a successful result."""
    import time
    if "error" in result:
        return  # Don't cache errors
    _RESULT_CACHE[key] = (result, time.time())
    # Evict oldest if over limit
    while len(_RESULT_CACHE) > _CACHE_MAX:
        oldest = next(iter(_RESULT_CACHE))
        del _RESULT_CACHE[oldest]

print(f"Origin={ORIGIN_AVAILABLE} | ExportMode={EXPORT_MODE} | OWUI={OWUI_URL}", flush=True)
if OWUI_PUBLIC_URL:
    print(f"PublicURL={OWUI_PUBLIC_URL}", flush=True)

# =============================================================================
# Utility Functions
# =============================================================================
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def normalize_file_id(file_id: str) -> str:
    if not file_id:
        return file_id
    if UUID_RE.match(file_id):
        return file_id
    if "_" in file_id:
        head = file_id.split("_", 1)[0]
        if UUID_RE.match(head):
            return head
    return file_id


def _truncate(s: Any, limit: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= limit else s[:limit] + "..."


def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}


def _parse_content_disposition(cd: str) -> Optional[str]:
    if not cd:
        return None
    for p in [x.strip() for x in cd.split(";")]:
        if p.lower().startswith("filename*="):
            val = p.split("=", 1)[1].strip().strip('"')
            if "''" in val:
                return urllib.parse.unquote(val.split("''", 1)[1])
            return urllib.parse.unquote(val)
        if p.lower().startswith("filename="):
            return p.split("=", 1)[1].strip().strip('"')
    return None


def owui_download_bytes(file_id: str) -> Tuple[bytes, str]:
    fid = normalize_file_id(file_id)
    url = f"{OWUI_URL}/api/v1/files/{fid}/content"
    r = requests.get(url, headers=_auth_headers(), timeout=300)
    if r.status_code in (401, 403):
        raise RuntimeError(f"Unauthorized ({r.status_code}) - check JWT_TOKEN")
    if r.status_code == 404:
        raise RuntimeError(f"File not found (404) for file_id={fid}")
    r.raise_for_status()
    ctype = (r.headers.get("content-type") or "").lower()
    if "text/html" in ctype:
        raise RuntimeError(f"Got HTML instead of file - auth may have failed")
    fname = _parse_content_disposition(r.headers.get("content-disposition", "")) or fid
    return r.content, fname


def owui_upload_bytes(data: bytes, filename: str, content_type: str = "application/octet-stream") -> Dict[str, Any]:
    url = f"{OWUI_URL}/api/v1/files/"
    params = {"process": "false", "process_in_background": "false"}
    headers = {**_auth_headers(), "Accept": "application/json"}
    files = {"file": (filename, data, content_type)}
    r = requests.post(url, headers=headers, params=params, files=files, timeout=300)
    if r.status_code in (401, 403):
        raise RuntimeError(f"Upload unauthorized ({r.status_code}) - check JWT_TOKEN")
    r.raise_for_status()
    return r.json()


def owui_file_content_url(file_id: str) -> str:
    base = OWUI_PUBLIC_URL or OWUI_URL
    return f"{base}/api/v1/files/{file_id}/content"


def _new_export_folder() -> str:
    folder = f"exp_{uuid.uuid4().hex[:8]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path = os.path.join(EXPORT_DIR, folder)
    os.makedirs(path, exist_ok=True)
    return path


def export_bytes(folder: str, data: bytes, filename: str, content_type: str = "application/octet-stream") -> str:
    out_path = os.path.join(folder, filename)
    with open(out_path, "wb") as f:
        f.write(data)
    if EXPORT_MODE == "owui":
        try:
            resp = owui_upload_bytes(data, filename, content_type)
            fid = resp.get("id") or resp.get("file_id")
            if fid:
                return owui_file_content_url(str(fid))
        except Exception as e:
            print(f"OWUI upload failed: {e}", flush=True)
    return out_path


def export_figure(folder: str, fig: plt.Figure, filename: str, dpi: int = 150) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return export_bytes(folder, buf.getvalue(), filename, "image/png")


def _coerce_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _annotation_json_type(ann: Any) -> str:
    if ann is None:
        return "string"
    origin = get_origin(ann)
    if origin is not None:
        args = [a for a in get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return _annotation_json_type(args[0])
        return "string"
    if ann is int:
        return "integer"
    if ann is float:
        return "number"
    if ann is bool:
        return "boolean"
    return "string"


def _coerce_value(expected_ann: Any, v: Any) -> Any:
    if v is None:
        return None
    origin = get_origin(expected_ann)
    if origin is not None:
        args = [a for a in get_args(expected_ann) if a is not type(None)]
        if len(args) == 1:
            expected_ann = args[0]
    try:
        if expected_ann is bool:
            return _coerce_bool(v)
        if expected_ann is int and v != "":
            return int(v)
        if expected_ann is float and v != "":
            return float(v)
    except Exception:
        return v
    return v


def _compact_json(obj: Any, max_chars: int) -> str:
    def dumps(o):
        return json.dumps(o, ensure_ascii=False, separators=(",", ":"))

    s = dumps(obj)
    if len(s) <= max_chars:
        return s

    if isinstance(obj, dict):
        shrunk = {k: v for k, v in obj.items() if k not in ("trace", "traceback", "debug")}

        for k in ("x", "y", "workbooks", "results", "sheets", "data", "folders", "columns"):
            if isinstance(shrunk.get(k), list) and len(shrunk[k]) > 30:
                shrunk[f"{k}_count"] = len(shrunk[k])
                shrunk[k] = shrunk[k][:30]
                shrunk["truncated"] = True

        s2 = dumps(shrunk)
        if len(s2) <= max_chars:
            return s2

        return dumps({"keys": list(obj.keys())[:15], "truncated": True})[:max_chars]

    return s[:max_chars]


# =============================================================================
# Origin Control
# =============================================================================
_ORIGIN_READY = False
_OPJ_CACHE_DIR = os.path.join(EXPORT_DIR, "_opj_cache")
os.makedirs(_OPJ_CACHE_DIR, exist_ok=True)

_LOADED_FILE_ID: Optional[str] = None
_LOADED_PATH: Optional[str] = None


def ensure_origin_running() -> None:
    global _ORIGIN_READY
    if not ORIGIN_AVAILABLE or op is None:
        raise RuntimeError("OriginPro not available")
    if _ORIGIN_READY:
        return
    try:
        op.attach()
    except Exception:
        op.new(asksave=False)
    try:
        op.set_show(False)
    except Exception:
        pass
    _ORIGIN_READY = True


def _safe_cache_name(file_id: str, fname: str) -> str:
    fid = normalize_file_id(file_id).replace("-", "")
    base = os.path.basename(fname or "project.opj")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    if not base.lower().endswith(".opj"):
        base += ".opj"
    return f"{fid}_{base}"


def _display_name(cache_path: str) -> str:
    base = os.path.basename(cache_path or "")
    m = re.match(r"^[0-9a-fA-F]{32}_(.+)$", base)
    return m.group(1) if m else base


def ensure_project_open(file_id: str) -> str:
    global _LOADED_FILE_ID, _LOADED_PATH

    if _LOADED_FILE_ID == file_id and _LOADED_PATH and os.path.exists(_LOADED_PATH):
        return _LOADED_PATH

    raw, fname = owui_download_bytes(file_id)
    cache_name = _safe_cache_name(file_id, fname)
    cache_path = os.path.join(_OPJ_CACHE_DIR, cache_name)

    if not os.path.exists(cache_path):
        with open(cache_path, "wb") as f:
            f.write(raw)

    try:
        op.new(asksave=False)
    except Exception:
        pass

    op.open(cache_path, readonly=False, asksave=False)
    _LOADED_FILE_ID = file_id
    _LOADED_PATH = cache_path
    return cache_path


def _obj_name(obj: Any) -> str:
    if obj is None:
        return ""
    for attr in ("name",):
        try:
            v = getattr(obj, attr, None)
            if v:
                s = v() if callable(v) else v
                if isinstance(s, str) and s.strip():
                    return s.strip()
        except Exception:
            pass
    return ""


def _obj_label(obj: Any) -> str:
    if obj is None:
        return ""
    for attr in ("lname", "label"):
        try:
            v = getattr(obj, attr, None)
            if v:
                s = v() if callable(v) else v
                if isinstance(s, str) and s.strip():
                    return s.strip()
        except Exception:
            pass
    return ""


def _get_folder_path(wb: Any) -> str:
    try:
        name = _obj_name(wb)
        if name:
            path = op.pe.search(name, 0)
            return path if path else "/"
    except Exception:
        pass
    return "/"


def _find_workbook(name_or_label: str, folder_hint: str = ""):
    target = name_or_label.lower().strip()
    folder_hint_lower = folder_hint.lower().strip() if folder_hint else ""

    for wb in op.pages("w"):
        name = _obj_name(wb).lower()
        label = _obj_label(wb).lower()

        if name == target or label == target:
            if folder_hint_lower:
                path = _get_folder_path(wb).lower()
                if folder_hint_lower not in path:
                    continue
            return wb
    return None


def _find_sheet(wb: Any, sheet_name: str = "Data"):
    target = sheet_name.lower().strip()
    first_data = None
    try:
        for wks in wb:
            sname = _obj_name(wks).lower()
            if sname == target:
                return wks
            if sname not in ("graph", "note") and first_data is None:
                first_data = wks
    except Exception:
        pass
    return first_data


def _wks_col_values(wks: Any, col: int) -> List[Any]:
    """Read a single column. Prefer to_df for consistent alignment."""
    try:
        df = wks.to_df()
        return df.iloc[:, col].tolist()
    except Exception:
        pass
    if hasattr(wks, "to_list"):
        return wks.to_list(col)
    raise AttributeError("Cannot read column")


def _wks_xy_aligned(wks: Any, x_col: int, y_col: int):
    """Read X and Y columns as aligned numpy arrays, dropping NaN rows."""
    try:
        df = wks.to_df()
        x_raw = df.iloc[:, x_col].values.astype(float)
        y_raw = df.iloc[:, y_col].values.astype(float)
    except Exception:
        x_raw = np.array(_wks_col_values(wks, x_col), dtype=float)
        y_raw = np.array(_wks_col_values(wks, y_col), dtype=float)
        # Truncate to shorter length if mismatched
        n = min(len(x_raw), len(y_raw))
        x_raw, y_raw = x_raw[:n], y_raw[:n]
    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    return x_raw[mask], y_raw[mask]


def _parse_power_value(s: str) -> float:
    s = s.lower().strip()
    for suffix in ('uw', 'w', 'mw', 'nw'):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
            break
    try:
        return float(s)
    except ValueError:
        return float('inf')


def _parse_y_cols(y_cols_str: str, n_cols: int) -> List[int]:
    """
    Parse y_cols specification.  Supports:
      "1,2,3,4"      → [1, 2, 3, 4]
      "1-12"          → [1, 2, ..., 12]
      "1-6,7-12"      → [1, 2, ..., 12]
      "1,3,5-8,10"    → [1, 3, 5, 6, 7, 8, 10]
      "all"           → [1, 2, ..., n_cols-1]
      "subtracted"    → returned as sentinel string
      ""              → empty list (caller handles default)
    """
    s = y_cols_str.strip()
    if not s:
        return []
    if s.lower() == "all":
        return list(range(1, n_cols))
    if s.lower() == "subtracted":
        return []  # Caller should handle this keyword

    result = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and not part.startswith("-"):
            # Range: "1-12"
            pieces = part.split("-", 1)
            try:
                lo, hi = int(pieces[0].strip()), int(pieces[1].strip())
                result.extend(range(lo, hi + 1))
            except (ValueError, IndexError):
                continue
        else:
            try:
                result.append(int(part))
            except ValueError:
                continue
    return result


def _detect_pol_keys(user_params_set: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Detect polarization convention from user_param values across columns.
    Returns mapping: {param_value: display_label}
    e.g. {"V1": "σ+", "V2": "σ−"} or {"RCP": "σ+", "LCP": "σ−"}
    """
    # Collect all 'pol' values seen
    pol_vals = set()
    pol_key = None
    for up in user_params_set:
        for k, v in up.items():
            kl = k.lower()
            if kl in ("pol", "polarization", "polarity", "circ", "helicity"):
                pol_vals.add(v)
                pol_key = k
                break

    if not pol_key or len(pol_vals) < 2:
        return {}

    # Common conventions
    KNOWN_PAIRS = {
        frozenset({"V1", "V2"}): {"V1": "σ+", "V2": "σ−"},
        frozenset({"v1", "v2"}): {"v1": "σ+", "v2": "σ−"},
        frozenset({"s+", "s-"}): {"s+": "σ+", "s-": "σ−"},
        frozenset({"σ+", "σ-"}): {"σ+": "σ+", "σ-": "σ−"},
        frozenset({"σ+", "σ−"}): {"σ+": "σ+", "σ−": "σ−"},
        frozenset({"RCP", "LCP"}): {"RCP": "σ+", "LCP": "σ−"},
        frozenset({"R", "L"}): {"R": "σ+", "L": "σ−"},
        frozenset({"CW", "CCW"}): {"CW": "σ+", "CCW": "σ−"},
    }

    for known, mapping in KNOWN_PAIRS.items():
        if pol_vals == known or {v.lower() for v in pol_vals} == {k.lower() for k in known}:
            # Match case-insensitively, return with original case
            out = {}
            for pv in pol_vals:
                for mk, ml in mapping.items():
                    if pv.lower() == mk.lower():
                        out[pv] = ml
                        break
            if len(out) == len(pol_vals):
                return out

    # Fallback: just label them as-is
    sorted_vals = sorted(pol_vals)
    return {sorted_vals[0]: sorted_vals[0], sorted_vals[1]: sorted_vals[1]} if len(sorted_vals) == 2 else {}


# ── Grouping-key selector ─────────────────────────────────────
# Picks the user_param key that VARIES across columns, not just the last
# keyword match.  Priority: keys with more unique values > keys with fewer.
# Tie-break: prefer strain > temperature > power > others.

_GROUP_CANDIDATES = {
    "strain", "temperature", "temp", "power", "angle",
    "bias", "voltage", "field", "pressure", "cycle",
}
_POL_CANDIDATES = {
    "pol", "polarization", "polarity", "circ", "helicity",
}
_GROUP_PRIORITY = {          # lower = preferred
    "strain": 0, "temperature": 1, "temp": 1,
    "power": 2, "field": 3, "bias": 3, "voltage": 3,
    "angle": 4, "pressure": 4, "cycle": 5,
}


def _pick_group_pol_keys(
    user_params_list: List[Dict[str, str]],
) -> tuple:
    """
    From a list of per-column user_params dicts, return (group_key, pol_key).

    Selects the grouping key whose values actually *vary* across columns.
    If multiple candidates vary, prefer higher-priority physics keys
    (strain > temperature > power > cycle).
    """
    if not user_params_list:
        return None, None

    pol_key = None
    candidates = {}   # {key_name: set_of_unique_values}

    for up in user_params_list:
        for k, v in up.items():
            kl = k.lower()
            if kl in _POL_CANDIDATES:
                pol_key = k
            elif kl in _GROUP_CANDIDATES:
                candidates.setdefault(k, set()).add(str(v).strip())

    if not candidates:
        return None, pol_key

    # Filter to keys that actually vary (>1 unique value)
    varying = {k: vals for k, vals in candidates.items() if len(vals) > 1}

    if varying:
        # Pick the one with best priority (lowest number)
        group_key = min(
            varying.keys(),
            key=lambda k: (_GROUP_PRIORITY.get(k.lower(), 99), -len(varying[k]))
        )
    else:
        # None vary — fall back to highest-priority candidate
        group_key = min(
            candidates.keys(),
            key=lambda k: _GROUP_PRIORITY.get(k.lower(), 99)
        )

    return group_key, pol_key


# =============================================================================
# Column Metadata Reader
# =============================================================================

def _col_letter(ci: int) -> str:
    """Convert 0-based column index to Excel-style letter: 0->A, 25->Z, 26->AA."""
    result = ""
    n = ci
    while True:
        result = chr(ord('A') + n % 26) + result
        n = n // 26 - 1
        if n < 0:
            break
    return result


# =============================================================================
# Messy-file helpers
# =============================================================================

def _guess_x_unit(x_vals: np.ndarray) -> str:
    """Guess X-axis unit from value range.
    eV: typically 0.5–6.0.  nm: typically 200–2500.  cm-1: typically 50–5000.
    """
    if len(x_vals) < 2:
        return "unknown"
    xmin, xmax = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
    xrange = xmax - xmin
    if 0.1 < xmax < 8.0 and xrange < 5.0:
        return "eV"
    if 100 < xmax < 3000 and xrange > 20:
        if xmin > 50 and xmax < 2500:
            return "nm"
    if xmax > 50 and xrange > 30:
        return "cm-1"
    return "unknown"


def _nm_to_ev(x_nm: np.ndarray) -> np.ndarray:
    """Convert wavelength (nm) to energy (eV): E = 1239.8 / λ."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return 1239.8 / x_nm


def _auto_x_range_from_peaks(x: np.ndarray, y: np.ndarray,
                              margin_factor: float = 5.0) -> Tuple[float, float]:
    """Auto-detect fit x_range by finding the dominant peak and returning ±margin_factor×FWHM.
    Falls back to full data range if peak detection fails.
    """
    from scipy.signal import find_peaks, peak_widths
    try:
        y_range = float(np.max(y) - np.min(y))
        if y_range <= 0:
            return float(x[0]), float(x[-1])
        prom = 0.05 * y_range
        dx = abs(float(np.mean(np.diff(x)))) if len(x) > 1 else 1.0
        min_dist = max(1, int(0.005 / dx)) if dx < 1 else max(1, int(5.0 / dx))
        peaks, props = find_peaks(y, prominence=prom, distance=min_dist)
        if len(peaks) == 0:
            return float(x[0]), float(x[-1])
        # Pick strongest peak
        best = peaks[np.argmax(props["prominences"])]
        # Estimate FWHM
        try:
            widths, _, _, _ = peak_widths(y, [best], rel_height=0.5)
            fwhm = float(widths[0]) * dx
        except Exception:
            fwhm = float(x[-1] - x[0]) / 20
        margin = margin_factor * max(fwhm, 0.005)  # at least 5 meV margin for eV data
        xc = float(x[best])
        xlo = max(float(x[0]), xc - margin)
        xhi = min(float(x[-1]), xc + margin)
        return xlo, xhi
    except Exception:
        return float(x[0]), float(x[-1])


_NAME_GROUP_PATTERNS = [
    # Strain: "0.45%", "strain=0.2", "ε=0.1"
    (r'(\d+\.?\d*)\s*%', "strain"),
    (r'(?:strain|ε)\s*[=:]\s*(\d+\.?\d*)', "strain"),
    # Temperature: "300K", "T=10K", "4.2K"
    (r'(\d+\.?\d*)\s*K\b', "temperature"),
    (r'(?:temp|T)\s*[=:]\s*(\d+\.?\d*)', "temperature"),
    # Power: "100uW", "5mW", "P=200"
    (r'(\d+\.?\d*)\s*[uμnm]?W\b', "power"),
    (r'(?:power|P)\s*[=:]\s*(\d+\.?\d*)', "power"),
    # Angle: "45deg", "θ=30"
    (r'(\d+\.?\d*)\s*(?:deg|°)', "angle"),
]

_NAME_POL_PATTERNS = [
    # σ+/σ-, V1/V2, co-pol/cross-pol, RCP/LCP
    (r'σ\s*\+|sigma\s*\+|co[- ]?pol|RCP|V1\b', "+"),
    (r'σ\s*[-−]|sigma\s*[-−]|cross[- ]?pol|LCP|V2\b', "-"),
]


def _infer_params_from_names(col_meta_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """When user_params are empty, try to infer grouping and polarization from
    column names, long_names, and comments using regex patterns.
    Returns a list of synthetic user_params dicts (one per column in col_meta_list).
    """
    synthetic = []
    for cm in col_meta_list:
        params = {}
        # Combine all text fields for matching
        text_parts = []
        for key in ("name", "long_name", "comments"):
            v = cm.get(key, "")
            if v:
                text_parts.append(str(v))
        text = " ".join(text_parts)
        if not text.strip():
            synthetic.append(params)
            continue

        # Try group patterns
        for pattern, param_name in _NAME_GROUP_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                params[param_name] = m.group(1) if m.lastindex else m.group(0)
                break  # Take first match only

        # Try polarization patterns
        for pattern, pol_sign in _NAME_POL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                params["pol"] = f"σ{pol_sign}" if pol_sign == "+" else f"σ−"
                break

        synthetic.append(params)
    return synthetic


def _read_column_metadata(wks, max_user_params: int = 10) -> List[Dict[str, Any]]:
    """
    Read full metadata for every column in a worksheet.
    Returns list of dicts: index, name, long_name, units, comments,
    user_params {param_name: value}.

    API notes (discovered via diagnostics on originpro 2024+):
      - get_labels("L"/"U"/"C")       → standard label rows (all cols at once)
      - get_labels("D1"), "D2", ...    → user parameter row data (1-based, all cols)
      - get_str("UserParam1"), ...     → user parameter row NAMES (1-based)
      - cols_axis() returns 'None' string on some builds → unreliable, skip
      - lt_col_index(ci) returns 1-based number string → unreliable, generate letters
    """
    ncols = wks.cols
    columns = []

    # Standard label rows — all at once
    try:
        long_names = wks.get_labels("L")
    except Exception:
        long_names = []
    try:
        units_list = wks.get_labels("U")
    except Exception:
        units_list = []
    try:
        comments_list = wks.get_labels("C")
    except Exception:
        comments_list = []

    # Discover user parameter rows via "D1", "D2", ... (1-based)
    # Paired with UserParam1, UserParam2, ... for row names (also 1-based)
    active_user_params = {}  # {1: "pol", 2: "strain", 4: "cycle", ...}
    for pi in range(1, max_user_params + 1):
        try:
            vals = wks.get_labels(f"D{pi}")
            if vals and any(str(v).strip() for v in vals):
                row_name = f"P{pi}"
                try:
                    lt_name = wks.get_str(f"UserParam{pi}")
                    if lt_name and lt_name.strip() and lt_name.strip() != f"UserDefined{pi}":
                        row_name = lt_name.strip()
                except Exception:
                    pass
                active_user_params[pi] = {"name": row_name, "vals": vals}
        except Exception:
            continue  # Don't break — rows can be sparse (D1, D2, gap, D4)

    # Get DataFrame column names as fallback short names
    df_colnames = []
    try:
        df = wks.to_df()
        df_colnames = list(df.columns)
    except Exception:
        pass

    for ci in range(ncols):
        col_info: Dict[str, Any] = {"index": ci}

        # Short name: prefer DataFrame column name, fallback to letter
        if ci < len(df_colnames) and str(df_colnames[ci]).strip():
            col_info["name"] = str(df_colnames[ci]).strip()
        else:
            col_info["name"] = _col_letter(ci)

        # Long name
        if ci < len(long_names) and str(long_names[ci]).strip():
            col_info["long_name"] = str(long_names[ci]).strip()

        # Units
        if ci < len(units_list) and str(units_list[ci]).strip():
            col_info["units"] = str(units_list[ci]).strip()

        # Comments (often contains source filename)
        if ci < len(comments_list) and str(comments_list[ci]).strip():
            col_info["comments"] = str(comments_list[ci]).strip()

        # User parameter values
        if active_user_params:
            uparams = {}
            for pi, pdata in active_user_params.items():
                vals = pdata["vals"]
                if ci < len(vals) and str(vals[ci]).strip():
                    uparams[pdata["name"]] = str(vals[ci]).strip()
            if uparams:
                col_info["user_params"] = uparams

        columns.append(col_info)

    return columns


# =============================================================================
# Fitting Functions
# =============================================================================

def lorentz_func(x, y0, xc, w, A):
    """Origin Lorentz: y = y0 + (2*A/pi) * (w / (4*(x-xc)^2 + w^2))"""
    return y0 + (2 * A / np.pi) * (w / (4 * (x - xc) ** 2 + w ** 2))


def gaussian_func(x, y0, xc, w, A):
    """Origin Gauss with FWHM parameter w."""
    sigma = w / (2 * np.sqrt(2 * np.log(2)))
    return y0 + (A / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - xc) / sigma) ** 2)


def voigt_func(x, y0, xc, sigma, gamma, A):
    """Voigt profile via Faddeeva function."""
    z = ((x - xc) + 1j * gamma) / (sigma * np.sqrt(2))
    return y0 + A * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_peak_scipy(x: np.ndarray, y: np.ndarray, profile: str = "lorentz") -> Dict[str, Any]:
    """Single-peak curve fitting."""
    profile = profile.lower().strip()

    y0_guess = float(np.nanmin(y))
    A_guess = float(np.nanmax(y) - y0_guess) * 0.5
    xc_guess = float(x[np.nanargmax(y)])
    w_guess = float((x[-1] - x[0]) / 10) if len(x) > 1 else 1.0

    try:
        if profile in ("lorentz", "lorentzian"):
            popt, _ = curve_fit(lorentz_func, x, y, p0=[y0_guess, xc_guess, w_guess, A_guess], maxfev=10000)
            yfit = lorentz_func(x, *popt)
            return {
                "profile": "lorentz", "engine": "scipy",
                "y0": round(float(popt[0]), 6), "xc": round(float(popt[1]), 6),
                "w": round(float(popt[2]), 6), "A": round(float(popt[3]), 6),
                "r2": round(r2_score(y, yfit), 6),
            }
        elif profile in ("gauss", "gaussian"):
            popt, _ = curve_fit(gaussian_func, x, y, p0=[y0_guess, xc_guess, w_guess, A_guess], maxfev=10000)
            yfit = gaussian_func(x, *popt)
            return {
                "profile": "gaussian", "engine": "scipy",
                "y0": round(float(popt[0]), 6), "xc": round(float(popt[1]), 6),
                "w": round(float(popt[2]), 6), "A": round(float(popt[3]), 6),
                "r2": round(r2_score(y, yfit), 6),
            }
        elif profile == "voigt":
            sigma_guess = w_guess / 2
            gamma_guess = w_guess / 2
            popt, _ = curve_fit(voigt_func, x, y, p0=[y0_guess, xc_guess, sigma_guess, gamma_guess, A_guess], maxfev=20000)
            yfit = voigt_func(x, *popt)
            return {
                "profile": "voigt", "engine": "scipy",
                "y0": round(float(popt[0]), 6), "xc": round(float(popt[1]), 6),
                "sigma": round(float(popt[2]), 6), "gamma": round(float(popt[3]), 6),
                "A": round(float(popt[4]), 6), "r2": round(r2_score(y, yfit), 6),
            }
        else:
            return {"error": f"Unknown profile: {profile}"}
    except Exception as e:
        return {"error": f"Fit failed: {str(e)[:80]}"}


# =============================================================================
# Multi-peak Fitting (exciton + trion decomposition)
# =============================================================================

def _multi_lorentz(x, y0, xc1, w1, A1, xc2, w2, A2):
    """Sum of two Lorentzian peaks + baseline."""
    p1 = (2 * A1 / np.pi) * (w1 / (4 * (x - xc1) ** 2 + w1 ** 2))
    p2 = (2 * A2 / np.pi) * (w2 / (4 * (x - xc2) ** 2 + w2 ** 2))
    return y0 + p1 + p2


def _multi_lorentz_voigt(x, y0, xc1, w1, A1, xc2, sigma2, gamma2, A2):
    """Lorentzian (sharp exciton) + Voigt (broader trion) + baseline."""
    p1 = (2 * A1 / np.pi) * (w1 / (4 * (x - xc1) ** 2 + w1 ** 2))
    z = ((x - xc2) + 1j * gamma2) / (sigma2 * np.sqrt(2))
    p2 = A2 * np.real(wofz(z)) / (sigma2 * np.sqrt(2 * np.pi))
    return y0 + p1 + p2


def _multi_voigt_voigt(x, y0, xc1, sigma1, gamma1, A1, xc2, sigma2, gamma2, A2):
    """Sum of two Voigt peaks + baseline."""
    z1 = ((x - xc1) + 1j * gamma1) / (sigma1 * np.sqrt(2))
    z2 = ((x - xc2) + 1j * gamma2) / (sigma2 * np.sqrt(2))
    p1 = A1 * np.real(wofz(z1)) / (sigma1 * np.sqrt(2 * np.pi))
    p2 = A2 * np.real(wofz(z2)) / (sigma2 * np.sqrt(2 * np.pi))
    return y0 + p1 + p2


def fit_multi_peak(
    x: np.ndarray,
    y: np.ndarray,
    num_peaks: int = 2,
    model: str = "lorentz+lorentz",
    hints: Optional[Dict[str, float]] = None,
    max_fwhm: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Multi-peak curve fitting for overlapping features.

    Models:
      "lorentz+lorentz"  — two Lorentzians (default, fast)
      "lorentz+voigt"    — Lorentzian (sharp) + Voigt (broad), good for exciton+trion
      "voigt+voigt"      — two Voigts (most flexible, slowest)

    hints: optional dict with keys like xc1, xc2 for manual initial guesses.
    max_fwhm: maximum allowed FWHM in eV. Prevents one peak from becoming a background.
    Returns: per-peak parameters, composite R², individual peak arrays.
    """
    model = model.lower().strip()

    # ── Normalize model aliases ──
    MODEL_ALIASES = {
        "lorentz+lorentz": "lorentz+lorentz", "2lorentz": "lorentz+lorentz",
        "ll": "lorentz+lorentz", "lor+lor": "lorentz+lorentz",
        "lorentzian+lorentzian": "lorentz+lorentz",
        "lorentz+voigt": "lorentz+voigt", "lv": "lorentz+voigt",
        "lor+voigt": "lorentz+voigt", "lorentzian+voigt": "lorentz+voigt",
        "voigt+voigt": "voigt+voigt", "2voigt": "voigt+voigt",
        "vv": "voigt+voigt",
    }
    # Catch bare single-profile names and map to compound
    SINGLE_TO_COMPOUND = {
        "voigt": "voigt+voigt",
        "lorentz": "lorentz+lorentz",
        "lorentzian": "lorentz+lorentz",
        "gauss+gauss": None,  # not supported
        "gaussian": None,
    }
    if model in MODEL_ALIASES:
        model = MODEL_ALIASES[model]
    elif model in SINGLE_TO_COMPOUND:
        mapped = SINGLE_TO_COMPOUND[model]
        if mapped:
            model = mapped
        else:
            return {"error": f"Model '{model}' is not supported for multi-peak. "
                    f"Valid: lorentz+lorentz, lorentz+voigt, voigt+voigt"}
    elif model not in ("lorentz+lorentz", "lorentz+voigt", "voigt+voigt"):
        return {"error": f"Unknown model '{model}'. "
                f"Valid: lorentz+lorentz (or ll), lorentz+voigt (or lv), voigt+voigt (or vv)"}

    hints = hints or {}

    try:
        # --- Robust initial guess estimation ---
        y0_guess = float(np.percentile(y, 5))
        y_shifted = y - y0_guess
        y_shifted = np.maximum(y_shifted, 0)

        peak_idx = int(np.argmax(y_shifted))
        peak_height = float(y_shifted[peak_idx])
        x_range = float(x[-1] - x[0])

        if peak_height <= 0:
            return {"error": "No peak detected above baseline",
                    "debug": {"y_min": float(np.min(y)), "y_max": float(np.max(y)),
                              "n_points": len(y)}}

        # Max FWHM constraint
        if max_fwhm is None:
            max_fwhm = x_range / 3
        max_fwhm = float(max_fwhm)

        # Width estimate from half-max crossing
        half_max = peak_height / 2
        above_half = np.where(y_shifted >= half_max)[0]
        if len(above_half) >= 2:
            w_guess = float(x[above_half[-1]] - x[above_half[0]])
            w_guess = max(w_guess, x_range / 50)
        else:
            w_guess = x_range / 15
        w_guess = min(w_guess, max_fwhm * 0.8)

        # Peak centers from hints or auto-detection
        xc1_guess = hints.get("xc1", hints.get("hint_xc1", float(x[peak_idx])))
        if "xc2" in hints or "hint_xc2" in hints:
            xc2_guess = hints.get("xc2", hints.get("hint_xc2"))
        else:
            # Default: place second peak 30 meV below main peak (typical trion)
            # This is more reliable than residual-based detection for weak features
            xc2_guess = float(x[peak_idx]) - 0.030

        # Ensure xc1 > xc2 (peak1 = higher energy = exciton, peak2 = lower = trion)
        if xc1_guess < xc2_guess:
            xc1_guess, xc2_guess = xc2_guess, xc1_guess

        # Area estimates
        A1_guess = peak_height * w_guess * np.pi / 2
        A2_guess = A1_guess * 0.3
        w2_guess = min(w_guess * 1.5, max_fwhm * 0.8)

        # Baseline bound: only constrain from below (let y0 float freely upward)
        y0_lb = float(np.min(y)) - abs(peak_height)

        debug_info = {
            "y0_guess": round(y0_guess, 1), "peak_height": round(peak_height, 1),
            "xc1_guess": round(xc1_guess, 4), "xc2_guess": round(xc2_guess, 4),
            "w_guess": round(w_guess, 4), "max_fwhm": round(max_fwhm, 4),
            "n_points": len(x), "x_range": round(x_range, 4),
        }

    except Exception as e:
        return {"error": f"Initial guess failed: {str(e)[:200]}",
                "n_points": len(x), "y_range": [float(np.min(y)), float(np.max(y))]}

    try:
        if model in ("lorentz+lorentz", "2lorentz", "ll"):
            p0 = [y0_guess, xc1_guess, w_guess, A1_guess, xc2_guess, w2_guess, A2_guess]
            lb = [y0_lb, x[0], 1e-6, 0, x[0], 1e-6, 0]
            ub = [np.inf, x[-1], max_fwhm, np.inf, x[-1], max_fwhm, np.inf]
            popt, pcov = curve_fit(_multi_lorentz, x, y, p0=p0, bounds=(lb, ub), maxfev=20000)
            yfit = _multi_lorentz(x, *popt)
            y0, xc1, w1, A1, xc2, w2, A2 = popt

            # Extract uncertainties (1-sigma) from covariance matrix
            try:
                perr = np.sqrt(np.diag(pcov))
                unc = {"xc1": round(float(perr[1]), 6), "w1": round(float(perr[2]), 6),
                       "A1": round(float(perr[3]), 6), "xc2": round(float(perr[4]), 6),
                       "w2": round(float(perr[5]), 6), "A2": round(float(perr[6]), 6)}
            except Exception:
                unc = None

            peak1 = lorentz_func(x, 0, xc1, w1, A1)
            peak2 = lorentz_func(x, 0, xc2, w2, A2)
            result = {
                "model": "lorentz+lorentz",
                "peak1": {"label": "peak1", "profile": "lorentz",
                          "xc": round(float(xc1), 6), "w": round(float(w1), 6),
                          "A": round(float(A1), 6)},
                "peak2": {"label": "peak2", "profile": "lorentz",
                          "xc": round(float(xc2), 6), "w": round(float(w2), 6),
                          "A": round(float(A2), 6)},
                "y0": round(float(y0), 6),
            }
            if unc:
                result["peak1"]["xc_err"] = unc["xc1"]
                result["peak1"]["w_err"] = unc["w1"]
                result["peak2"]["xc_err"] = unc["xc2"]
                result["peak2"]["w_err"] = unc["w2"]
                result["uncertainties"] = unc

        elif model in ("lorentz+voigt", "lv"):
            sigma2_g = w2_guess / (2 * np.sqrt(2 * np.log(2)))
            gamma2_g = w2_guess / 4
            p0 = [y0_guess, xc1_guess, w_guess, A1_guess, xc2_guess, sigma2_g, gamma2_g, A2_guess]
            max_sg = max_fwhm / (2 * np.sqrt(2 * np.log(2)))
            lb = [y0_lb, x[0], 1e-6, 0, x[0], 1e-6, 1e-6, 0]
            ub = [np.inf, x[-1], max_fwhm, np.inf, x[-1], max_sg, max_fwhm / 2, np.inf]
            popt, pcov = curve_fit(_multi_lorentz_voigt, x, y, p0=p0, bounds=(lb, ub), maxfev=30000)
            yfit = _multi_lorentz_voigt(x, *popt)
            y0, xc1, w1, A1, xc2, sigma2, gamma2, A2 = popt

            try:
                perr = np.sqrt(np.diag(pcov))
                unc = {"xc1": round(float(perr[1]), 6), "xc2": round(float(perr[4]), 6)}
            except Exception:
                unc = None

            peak1 = lorentz_func(x, 0, xc1, w1, A1)
            z2 = ((x - xc2) + 1j * gamma2) / (sigma2 * np.sqrt(2))
            peak2 = A2 * np.real(wofz(z2)) / (sigma2 * np.sqrt(2 * np.pi))
            result = {
                "model": "lorentz+voigt",
                "peak1": {"label": "peak1", "profile": "lorentz",
                          "xc": round(float(xc1), 6), "w": round(float(w1), 6),
                          "A": round(float(A1), 6)},
                "peak2": {"label": "peak2", "profile": "voigt",
                          "xc": round(float(xc2), 6), "sigma": round(float(sigma2), 6),
                          "gamma": round(float(gamma2), 6), "A": round(float(A2), 6)},
                "y0": round(float(y0), 6),
            }
            if unc:
                result["peak1"]["xc_err"] = unc["xc1"]
                result["peak2"]["xc_err"] = unc["xc2"]
                result["uncertainties"] = unc

        elif model in ("voigt+voigt", "vv"):
            sg = w_guess / (2 * np.sqrt(2 * np.log(2)))
            gg = w_guess / 4
            sg2 = w2_guess / (2 * np.sqrt(2 * np.log(2)))
            gg2 = w2_guess / 4
            p0 = [y0_guess, xc1_guess, sg, gg, A1_guess, xc2_guess, sg2, gg2, A2_guess]
            max_sg = max_fwhm / (2 * np.sqrt(2 * np.log(2)))
            lb = [y0_lb, x[0], 1e-6, 1e-6, 0, x[0], 1e-6, 1e-6, 0]
            ub = [np.inf, x[-1], max_sg, max_fwhm / 2, np.inf,
                  x[-1], max_sg, max_fwhm / 2, np.inf]
            popt, pcov = curve_fit(_multi_voigt_voigt, x, y, p0=p0, bounds=(lb, ub), maxfev=40000)
            yfit = _multi_voigt_voigt(x, *popt)
            y0, xc1, s1, g1, A1, xc2, s2, g2, A2 = popt
            z1 = ((x - xc1) + 1j * g1) / (s1 * np.sqrt(2))
            z2 = ((x - xc2) + 1j * g2) / (s2 * np.sqrt(2))
            peak1 = A1 * np.real(wofz(z1)) / (s1 * np.sqrt(2 * np.pi))
            peak2 = A2 * np.real(wofz(z2)) / (s2 * np.sqrt(2 * np.pi))
            result = {
                "model": "voigt+voigt",
                "peak1": {"label": "peak1", "profile": "voigt",
                          "xc": round(float(xc1), 6), "sigma": round(float(s1), 6),
                          "gamma": round(float(g1), 6), "A": round(float(A1), 6)},
                "peak2": {"label": "peak2", "profile": "voigt",
                          "xc": round(float(xc2), 6), "sigma": round(float(s2), 6),
                          "gamma": round(float(g2), 6), "A": round(float(A2), 6)},
                "y0": round(float(y0), 6),
            }
        else:
            return {"error": f"Unknown model: {model}. Use lorentz+lorentz, lorentz+voigt, or voigt+voigt"}

        # Compute composite R²
        result["r2"] = round(r2_score(y, yfit), 6)

        # Peak intensity at center (for polarization calculations)
        for pk_key, pk_curve in [("peak1", peak1), ("peak2", peak2)]:
            pk = result[pk_key]
            xc_val = pk["xc"]
            idx_at_center = int(np.argmin(np.abs(x - xc_val)))
            pk["I_at_center"] = round(float(pk_curve[idx_at_center]), 4)

        # Separation between peaks
        sep = abs(result["peak1"]["xc"] - result["peak2"]["xc"])
        result["peak_separation_eV"] = round(sep, 6)
        result["peak_separation_meV"] = round(sep * 1000, 2)

        # ── v10.9: min_separation guard ──
        # If peaks converged to essentially the same position, the 2-peak model
        # is degenerate. Flag it but keep the two-peak format for downstream compatibility.
        x_range = float(x[-1] - x[0])
        # Threshold: 10 meV for eV-scale data, 2 nm for nm-scale data, else 1% of x_range
        if x_range < 10:  # eV scale
            min_sep_threshold = 0.010  # 10 meV
        elif x_range < 1000:  # nm scale
            min_sep_threshold = 2.0  # 2 nm
        else:  # cm-1 or other
            min_sep_threshold = x_range * 0.01
        if sep < min_sep_threshold:
            result["degenerate"] = True
            result["warning"] = (
                f"Two peaks converged (Δ={sep*1000:.1f} meV < {min_sep_threshold*1000:.0f} meV threshold). "
                f"Decomposition is marginal — treat as single peak."
            )

        # Compare with single-peak fit to assess if decomposition is justified
        try:
            single_fit = fit_peak_scipy(x, y, "lorentz")
            if "r2" in single_fit:
                result["r2_single_peak"] = single_fit["r2"]
                result["delta_r2"] = round(result["r2"] - single_fit["r2"], 6)
        except Exception:
            pass

        return result

    except Exception as e:
        return {"error": f"Multi-peak fit failed: {str(e)[:200]}",
                "debug": debug_info}


# =============================================================================
# MCP TOOL 1: opj_inspect — enhanced with column metadata + user params
# =============================================================================
def opj_inspect(file_id: str, folder: str = "", detail: str = "full") -> Dict[str, Any]:
    """
    Inspect OPJ file: workbooks, sheets, columns with full metadata.
    detail='full' returns column names, labels, units, designations, user parameter rows.
    detail='brief' returns just workbook/sheet counts.
    """
    if not ORIGIN_AVAILABLE:
        return {"error": "OriginPro not available"}

    with ORIGIN_LOCK:
        try:
            ensure_origin_running()
            fname = _display_name(ensure_project_open(file_id))

            folder_filter = folder.lower().strip() if folder else ""
            want_detail = detail.lower().strip() != "brief"

            workbooks = []
            folders_seen = set()

            for wb in op.pages("w"):
                name = _obj_name(wb)
                label = _obj_label(wb)
                path = _get_folder_path(wb)

                if path:
                    folders_seen.add(path)
                if folder_filter and folder_filter not in path.lower():
                    continue

                entry = {"name": name, "path": path}
                if label and label != name:
                    entry["label"] = label

                sheets_info = []
                try:
                    for wks in wb:
                        sname = _obj_name(wks)
                        if sname.lower() in ("graph", "note"):
                            continue
                        sheet_entry = {"name": sname, "cols": wks.cols}
                        try:
                            sheet_entry["rows"] = wks.rows
                        except Exception:
                            pass

                        if want_detail:
                            try:
                                col_meta = _read_column_metadata(wks)
                                if col_meta:
                                    sheet_entry["columns"] = col_meta
                            except Exception as e:
                                sheet_entry["columns_error"] = str(e)[:100]

                        sheets_info.append(sheet_entry)
                except Exception:
                    sheets_info = [{"name": "?", "error": "Could not enumerate sheets"}]

                entry["sheets"] = sheets_info
                workbooks.append(entry)

            workbooks.sort(key=lambda w: _parse_power_value(w.get("label", w["name"])))

            # ── v10.9: Diagnostics for messy files ──
            diagnostics = {}
            try:
                # Detect x unit and suggested range from first workbook's first sheet
                if workbooks:
                    first_wb = workbooks[0]
                    for wb_page in op.pages("w"):
                        if _obj_name(wb_page) == first_wb["name"]:
                            first_wks = _find_sheet(wb_page)
                            if first_wks and first_wks.cols >= 2:
                                try:
                                    x_raw, y_raw = _wks_xy_aligned(first_wks, 0, 1)
                                    if len(x_raw) > 10:
                                        diagnostics["x_unit_guess"] = _guess_x_unit(x_raw)
                                        diagnostics["x_range"] = [round(float(x_raw[0]), 4),
                                                                   round(float(x_raw[-1]), 4)]
                                        # Auto-detect fit region
                                        try:
                                            auto_lo, auto_hi = _auto_x_range_from_peaks(x_raw, y_raw)
                                            diagnostics["suggested_x_range"] = [round(auto_lo, 4),
                                                                                 round(auto_hi, 4)]
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                            break

                # Check for user_params across all sheets
                has_any_uparams = False
                has_pol = False
                has_group = False
                empty_cols = []
                for wb_info in workbooks:
                    for sh in wb_info.get("sheets", []):
                        for col_info in sh.get("columns", []):
                            up = col_info.get("user_params", {})
                            if up:
                                has_any_uparams = True
                                for k in up:
                                    if k.lower() in _POL_CANDIDATES:
                                        has_pol = True
                                    if k.lower() in _GROUP_CANDIDATES:
                                        has_group = True
                            # Check for empty columns (from rows count)
                            rows = sh.get("rows", 0)
                            if rows is not None and rows < 10:
                                empty_cols.append(f"{wb_info['name']}/{sh['name']}/col{col_info['index']}")

                diagnostics["has_user_params"] = has_any_uparams
                diagnostics["has_pol_pairing"] = has_pol
                diagnostics["has_grouping"] = has_group

                # If no user_params, try to infer from column names
                if not has_any_uparams and workbooks:
                    for wb_info in workbooks:
                        for sh in wb_info.get("sheets", []):
                            cols = sh.get("columns", [])
                            if cols:
                                inferred = _infer_params_from_names(cols)
                                has_inferred = any(bool(p) for p in inferred)
                                if has_inferred:
                                    diagnostics["inferred_params_available"] = True
                                    # Show a sample
                                    sample = [p for p in inferred if p][:3]
                                    if sample:
                                        diagnostics["inferred_params_sample"] = sample
                                    break
                        if diagnostics.get("inferred_params_available"):
                            break

                if empty_cols:
                    diagnostics["empty_columns"] = empty_cols[:10]  # Cap at 10
            except Exception:
                pass  # Diagnostics are best-effort, never block inspect

            return {
                "file": fname,
                "count": len(workbooks),
                "folders": sorted(list(folders_seen)),
                "diagnostics": diagnostics,
                "workbooks": workbooks,
            }

        except Exception as e:
            return {"error": str(e)[:150]}


# =============================================================================
# MCP TOOL 2: opj_get_data — extract X,Y data
# =============================================================================
def opj_get_data(
    file_id: str,
    workbook: str,
    sheet: str = "Data",
    x_col: int = 0,
    y_col: int = 1,
    folder: str = "",
    max_points: int = 200
) -> Dict[str, Any]:
    """
    Extract X,Y data from a workbook.
    Use workbook name or label. Optionally filter by folder path.
    """
    if not ORIGIN_AVAILABLE:
        return {"error": "OriginPro not available"}

    with ORIGIN_LOCK:
        try:
            ensure_origin_running()
            ensure_project_open(file_id)

            wb = _find_workbook(workbook, folder)
            if wb is None:
                hint = f" in folder '{folder}'" if folder else ""
                return {"error": f"Workbook '{workbook}' not found{hint}"}

            wks = _find_sheet(wb, sheet)
            if wks is None:
                return {"error": f"Sheet '{sheet}' not found"}

            x_clean, y_clean = _wks_xy_aligned(wks, x_col, y_col)

            if len(x_clean) == 0:
                return {"error": "No valid data"}

            step = max(1, len(x_clean) // max_points)
            x_out = x_clean[::step].tolist()
            y_out = y_clean[::step].tolist()

            peak_idx = int(np.argmax(y_clean))

            return {
                "workbook": workbook,
                "folder": _get_folder_path(wb),
                "sheet": _obj_name(wks),
                "points": len(x_clean),
                "x": [round(v, 5) for v in x_out],
                "y": [round(v, 3) for v in y_out],
                "peak": {
                    "x": round(float(x_clean[peak_idx]), 5),
                    "y": round(float(y_clean[peak_idx]), 3)
                },
            }

        except Exception as e:
            return {"error": str(e)[:150]}


# =============================================================================
# MCP TOOL 3: opj_merge_power_series — CORE WORKFLOW TOOL
# =============================================================================
def opj_merge_power_series(
    file_id: str,
    folder: str,
    bg_folder: str = "",
    output_book: str = "Merged",
    x_col: int = 0,
    y_col: int = 1
) -> Dict[str, Any]:
    """
    Merge Y columns from all power series workbooks in a folder into one sheet.
    Finds workbooks in folder sorted by power, creates merged sheet with common X
    and all Y columns. Optionally subtracts background from bg_folder.
    """
    if not ORIGIN_AVAILABLE:
        return {"error": "OriginPro not available"}

    with ORIGIN_LOCK:
        try:
            ensure_origin_running()
            ensure_project_open(file_id)

            folder_lower = folder.lower().strip()
            if not folder_lower:
                return {"error": "folder parameter required (e.g. '530/NR2')"}

            power_wbs = []
            for wb in op.pages("w"):
                path = _get_folder_path(wb).lower()
                if folder_lower in path:
                    name = _obj_name(wb)
                    label = _obj_label(wb) or name
                    power = _parse_power_value(label)
                    power_wbs.append({"wb": wb, "name": name, "label": label, "power": power})

            if not power_wbs:
                return {"error": f"No workbooks found in folder '{folder}'"}

            power_wbs.sort(key=lambda x: x["power"])

            first_wks = _find_sheet(power_wbs[0]["wb"], "Data")
            if first_wks is None:
                return {"error": "No Data sheet in first workbook"}

            x_data = np.array(_wks_col_values(first_wks, x_col), dtype=float)
            n_points = len(x_data)

            y_columns = []
            labels = []
            for pw in power_wbs:
                wks = _find_sheet(pw["wb"], "Data")
                if wks is None:
                    continue
                try:
                    y = np.array(_wks_col_values(wks, y_col), dtype=float)
                    if len(y) == n_points:
                        y_columns.append(y)
                        labels.append(pw["label"])
                except Exception:
                    continue

            if not y_columns:
                return {"error": "Could not extract Y data from workbooks"}

            bg_data = None
            bg_label = None
            if bg_folder:
                bg_folder_lower = bg_folder.lower().strip()
                for wb in op.pages("w"):
                    path = _get_folder_path(wb).lower()
                    if bg_folder_lower in path:
                        wks = _find_sheet(wb, "Data")
                        if wks:
                            try:
                                bg = np.array(_wks_col_values(wks, y_col), dtype=float)
                                if len(bg) == n_points:
                                    bg_data = bg
                                    bg_label = _obj_label(wb) or _obj_name(wb)
                                    break
                            except Exception:
                                continue

            out_book = op.new_book('w', lname=output_book)
            out_wks = out_book[0]
            out_wks.name = "Merged"

            n_y_cols = len(y_columns)
            has_bg = bg_data is not None
            total_cols = 1 + n_y_cols
            if has_bg:
                total_cols += 1 + n_y_cols

            out_wks.cols = total_cols
            out_wks.from_list(0, x_data.tolist(), lname="Energy", units="eV", axis="X")

            for i, (y, label) in enumerate(zip(y_columns, labels)):
                out_wks.from_list(i + 1, y.tolist(), lname=label, axis="Y")

            col_idx = 1 + n_y_cols

            if has_bg:
                out_wks.from_list(col_idx, bg_data.tolist(), lname=f"bg ({bg_label})", axis="Y")
                bg_col_idx = col_idx
                col_idx += 1

                for i, label in enumerate(labels):
                    subtracted = y_columns[i] - bg_data
                    out_wks.from_list(col_idx, subtracted.tolist(), lname=f"{label}-bg", axis="Y")
                    try:
                        formula_col = chr(ord('B') + i)
                        bg_col_letter = chr(ord('A') + bg_col_idx)
                        out_wks.set_formula(col_idx, f"Col({formula_col})-Col({bg_col_letter})")
                    except Exception:
                        pass
                    col_idx += 1

            folder_path = _new_export_folder()

            fig, axes = plt.subplots(1, 2 if has_bg else 1, figsize=(12 if has_bg else 7, 5))
            if not has_bg:
                axes = [axes]

            ax = axes[0]
            for i, (y, label) in enumerate(zip(y_columns, labels)):
                ax.plot(x_data, y, label=label, alpha=0.8)
            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel("Intensity (counts)")
            ax.set_title(f"Raw Data - {folder}")
            ax.legend(fontsize=7, loc='best', ncol=2)
            ax.grid(True, alpha=0.3)

            if has_bg:
                ax = axes[1]
                for i, label in enumerate(labels):
                    ax.plot(x_data, y_columns[i] - bg_data, label=f"{label}-bg", alpha=0.8)
                ax.set_xlabel("Energy (eV)")
                ax.set_ylabel("Intensity (counts)")
                ax.set_title(f"Background Subtracted - {folder}")
                ax.legend(fontsize=7, loc='best', ncol=2)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_url = export_figure(folder_path, fig, "merged_preview.png")

            opj_name = f"{output_book}.opj"
            opj_path = os.path.join(folder_path, opj_name)
            op.save(opj_path)

            with open(opj_path, "rb") as f:
                opj_data = f.read()
            opj_url = export_bytes(folder_path, opj_data, opj_name, "application/octet-stream")

            return {
                "output_book": output_book,
                "source_folder": folder,
                "workbooks_merged": len(y_columns),
                "labels": labels,
                "columns": total_cols,
                "points": n_points,
                "background": bg_label if has_bg else None,
                "plot": plot_url,
                "opj": opj_url,
            }

        except Exception as e:
            out = {"error": str(e)[:150]}
            if DEBUG_TRACEBACK:
                out["trace"] = _truncate(traceback.format_exc(), MAX_ERROR_CHARS)
            return out


# =============================================================================
# MCP TOOL 4: opj_fit_peak — single peak fit
# =============================================================================
def opj_fit_peak(
    file_id: str,
    workbook: str,
    sheet: str = "Data",
    x_col: int = 0,
    y_col: int = 1,
    folder: str = "",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    profile: str = "lorentz",
    export_plot: bool = True
) -> Dict[str, Any]:
    """
    Fit single peak with Lorentz/Gaussian/Voigt profile.
    Returns: xc (center), w (FWHM), A (area), y0 (offset), r2.
    """
    if not ORIGIN_AVAILABLE:
        return {"error": "OriginPro not available"}

    with ORIGIN_LOCK:
        try:
            ensure_origin_running()
            ensure_project_open(file_id)

            wb = _find_workbook(workbook, folder)
            if wb is None:
                return {"error": f"Workbook '{workbook}' not found"}

            wks = _find_sheet(wb, sheet)
            if wks is None:
                return {"error": f"Sheet not found"}

            x_clean, y_clean = _wks_xy_aligned(wks, x_col, y_col)

            if len(x_clean) < 10:
                return {"error": "Not enough data points"}

            if x_min is not None or x_max is not None:
                xmin = float(x_min) if x_min is not None else float(np.min(x_clean))
                xmax = float(x_max) if x_max is not None else float(np.max(x_clean))
                region_mask = (x_clean >= xmin) & (x_clean <= xmax)
                x_fit = x_clean[region_mask]
                y_fit = y_clean[region_mask]
            else:
                peak_idx = int(np.argmax(y_clean))
                window = max(20, len(x_clean) // 10)
                lo = max(0, peak_idx - window)
                hi = min(len(x_clean), peak_idx + window)
                x_fit = x_clean[lo:hi]
                y_fit = y_clean[lo:hi]

            if len(x_fit) < 5:
                return {"error": "Fit region too small"}

            fit_result = fit_peak_scipy(x_fit, y_fit, profile)

            result = {
                "workbook": workbook,
                "profile": profile,
                **fit_result,
            }

            if export_plot and "xc" in fit_result:
                folder_path = _new_export_folder()
                fig, ax = plt.subplots(figsize=(7, 4.5))

                ax.plot(x_clean, y_clean, 'b-', alpha=0.4, linewidth=0.8, label='Data')
                ax.plot(x_fit, y_fit, 'b-', linewidth=1.2, label='Fit region')

                x_curve = np.linspace(float(x_fit[0]), float(x_fit[-1]), 200)
                prof = profile.lower()
                if prof in ("lorentz", "lorentzian"):
                    y_curve = lorentz_func(x_curve, fit_result["y0"], fit_result["xc"],
                                          fit_result["w"], fit_result["A"])
                elif prof in ("gauss", "gaussian"):
                    y_curve = gaussian_func(x_curve, fit_result["y0"], fit_result["xc"],
                                           fit_result["w"], fit_result["A"])
                else:
                    y_curve = None

                if y_curve is not None:
                    ax.plot(x_curve, y_curve, 'r-', linewidth=2,
                           label=f'{profile.title()} (R\u00b2={fit_result.get("r2", 0):.4f})')

                ax.axvline(fit_result["xc"], color='green', linestyle='--', alpha=0.6,
                          label=f'xc={fit_result["xc"]:.4f}')

                ax.set_xlabel("Energy (eV)")
                ax.set_ylabel("Intensity")
                ax.set_title(f"{workbook} - {profile.title()} Fit")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                result["plot"] = export_figure(folder_path, fig, "fit.png")

            return result

        except Exception as e:
            return {"error": str(e)[:150]}


# =============================================================================
# MCP TOOL 5: opj_multi_fit — multi-peak decomposition
# =============================================================================
def opj_multi_fit(
    file_id: str,
    workbook: str,
    sheet: str = "Data",
    x_col: int = 0,
    y_col: int = 1,
    folder: str = "",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    model: str = "lorentz+lorentz",
    hint_xc1: Optional[float] = None,
    hint_xc2: Optional[float] = None,
    max_fwhm: Optional[float] = None,
    export_plot: bool = True,
) -> Dict[str, Any]:
    """
    Decompose spectrum into 2 overlapping peaks (e.g. exciton + trion).
    Models: 'lorentz+lorentz', 'lorentz+voigt', 'voigt+voigt'.
    hint_xc1, hint_xc2: optional initial guesses for peak centers.
    max_fwhm: max allowed FWHM in eV (default: x_range/3). Prevents one peak from becoming a background.
    Returns per-peak parameters, separation in meV, composite R2, plot.
    """
    if not ORIGIN_AVAILABLE:
        return {"error": "OriginPro not available"}

    with ORIGIN_LOCK:
        try:
            ensure_origin_running()
            ensure_project_open(file_id)

            wb = _find_workbook(workbook, folder)
            if wb is None:
                return {"error": f"Workbook '{workbook}' not found"}

            wks = _find_sheet(wb, sheet)
            if wks is None:
                return {"error": f"Sheet not found"}

            x_clean, y_clean = _wks_xy_aligned(wks, x_col, y_col)

            if len(x_clean) < 15:
                return {"error": "Not enough data points for multi-peak fit"}

            # Region selection
            if x_min is not None or x_max is not None:
                xmin = float(x_min) if x_min is not None else float(np.min(x_clean))
                xmax = float(x_max) if x_max is not None else float(np.max(x_clean))
                region_mask = (x_clean >= xmin) & (x_clean <= xmax)
                x_fit = x_clean[region_mask]
                y_fit = y_clean[region_mask]
            else:
                x_fit = x_clean
                y_fit = y_clean

            if len(x_fit) < 10:
                return {"error": "Fit region too small for multi-peak"}

            # Build hints dict
            hints = {}
            if hint_xc1 is not None:
                hints["xc1"] = float(hint_xc1)
            if hint_xc2 is not None:
                hints["xc2"] = float(hint_xc2)

            # Default max_fwhm: 1/3 of x range (prevents one peak becoming background)
            if max_fwhm is None:
                max_fwhm_val = float(x_fit[-1] - x_fit[0]) / 3
            else:
                max_fwhm_val = float(max_fwhm)

            fit_result = fit_multi_peak(x_fit, y_fit, model=model, hints=hints, max_fwhm=max_fwhm_val)

            if "error" in fit_result:
                return {"workbook": workbook, **fit_result}

            result = {"workbook": workbook, **fit_result}

            # Plot decomposition
            if export_plot:
                folder_path = _new_export_folder()
                fig, ax = plt.subplots(figsize=(8, 5))

                ax.plot(x_clean, y_clean, 'b-', alpha=0.3, linewidth=0.8, label='Full data')
                ax.plot(x_fit, y_fit, 'k-', linewidth=1.2, label='Fit region')

                x_curve = np.linspace(float(x_fit[0]), float(x_fit[-1]), 300)
                pk1 = fit_result["peak1"]
                pk2 = fit_result["peak2"]
                y0 = fit_result["y0"]

                # Reconstruct individual peaks for plotting
                if pk1["profile"] == "lorentz":
                    curve1 = lorentz_func(x_curve, 0, pk1["xc"], pk1["w"], pk1["A"])
                else:
                    z1 = ((x_curve - pk1["xc"]) + 1j * pk1["gamma"]) / (pk1["sigma"] * np.sqrt(2))
                    curve1 = pk1["A"] * np.real(wofz(z1)) / (pk1["sigma"] * np.sqrt(2 * np.pi))

                if pk2["profile"] == "lorentz":
                    curve2 = lorentz_func(x_curve, 0, pk2["xc"], pk2["w"], pk2["A"])
                else:
                    z2 = ((x_curve - pk2["xc"]) + 1j * pk2["gamma"]) / (pk2["sigma"] * np.sqrt(2))
                    curve2 = pk2["A"] * np.real(wofz(z2)) / (pk2["sigma"] * np.sqrt(2 * np.pi))

                composite = y0 + curve1 + curve2

                ax.fill_between(x_curve, y0, y0 + curve1, alpha=0.3, color='red',
                               label=f'Peak 1 (xc={pk1["xc"]:.4f} eV)')
                ax.fill_between(x_curve, y0, y0 + curve2, alpha=0.3, color='blue',
                               label=f'Peak 2 (xc={pk2["xc"]:.4f} eV)')
                ax.plot(x_curve, composite, 'r--', linewidth=2,
                       label=f'Composite (R\u00b2={fit_result["r2"]:.4f})')

                ax.axvline(pk1["xc"], color='red', linestyle=':', alpha=0.5)
                ax.axvline(pk2["xc"], color='blue', linestyle=':', alpha=0.5)

                sep_meV = fit_result["peak_separation_meV"]
                ax.set_xlabel("Energy (eV)")
                ax.set_ylabel("Intensity")
                ax.set_title(f"{workbook} — {model} decomposition (\u0394={sep_meV:.1f} meV)")
                ax.legend(fontsize=7, loc='best')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                result["plot"] = export_figure(folder_path, fig, "multi_fit.png")

            return result

        except Exception as e:
            return {"error": str(e)[:150]}


# =============================================================================
# MCP TOOL 5b: opj_batch_multi_fit — batch multi-peak decomposition
# =============================================================================
def opj_batch_multi_fit(
    file_id: str,
    workbook: str,
    sheet: str = "Data",
    x_col: int = 0,
    y_cols: str = "",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    model: str = "lorentz+lorentz",
    max_fwhm: Optional[float] = None,
    folder: str = "",
) -> Dict[str, Any]:
    """
    Batch multi-peak decomposition across multiple Y columns in one call.
    y_cols: comma-separated indices OR ranges e.g. "1-12", "1-6,7-12", "1,3,5-8".
           If empty, fits all Y columns (skipping column 0 = X axis).
    Returns per-column fit results with peak parameters, uncertainties,
    user_params, summary grid plot, and waterfall plot.
    """
    if not ORIGIN_AVAILABLE:
        return {"error": "OriginPro not available"}

    # Dedup: return cached result for identical calls
    ck = _cache_key("batch_multi_fit", file_id=file_id, workbook=workbook,
                     sheet=sheet, x_col=x_col, y_cols=y_cols, x_min=x_min,
                     x_max=x_max, model=model, max_fwhm=max_fwhm, folder=folder)
    cached = _cache_get(ck)
    if cached:
        return cached

    with ORIGIN_LOCK:
        try:
            ensure_origin_running()
            ensure_project_open(file_id)

            wb = _find_workbook(workbook, folder)
            if wb is None:
                return {"error": f"Workbook '{workbook}' not found"}

            wks = _find_sheet(wb, sheet)
            if wks is None:
                return {"error": f"Sheet '{sheet}' not found"}

            n_cols = wks.cols

            # Parse y_cols with range support (e.g. "1-12" or "1,3,5-8")
            col_indices = _parse_y_cols(y_cols, n_cols)
            if not col_indices:
                col_indices = list(range(1, n_cols))

            if not col_indices:
                return {"error": "No columns specified"}

            # ── v10.9: Auto x_range when not specified ──
            # Detect from first valid Y column using peak finding
            auto_x_min, auto_x_max = None, None
            x_unit_detected = "unknown"
            if x_min is None and x_max is None:
                print(f"[origin] AUTO X_RANGE: probing {col_indices[:3]} (x_min/x_max not specified)", flush=True)
                for ci_probe in col_indices[:3]:  # Try first 3 columns
                    if 0 <= ci_probe < n_cols:
                        try:
                            x_probe, y_probe = _wks_xy_aligned(wks, x_col, ci_probe)
                            if len(x_probe) > 15:
                                x_unit_detected = _guess_x_unit(x_probe)
                                auto_x_min, auto_x_max = _auto_x_range_from_peaks(x_probe, y_probe)
                                print(f"[origin] AUTO X_RANGE: col{ci_probe} → [{auto_x_min:.4f}, {auto_x_max:.4f}] unit={x_unit_detected} (from {len(x_probe)} pts, full range [{float(x_probe[0]):.4f}, {float(x_probe[-1]):.4f}])", flush=True)
                                break
                        except Exception as _e:
                            print(f"[origin] AUTO X_RANGE: col{ci_probe} FAILED: {_e}", flush=True)
                            continue
                if auto_x_min is None:
                    print(f"[origin] AUTO X_RANGE: FAILED — all probes failed, fitting full range", flush=True)
            else:
                print(f"[origin] X_RANGE: explicit x_min={x_min} x_max={x_max}", flush=True)

            # Read column metadata for labels and user_params
            col_meta = {}
            try:
                meta_list = _read_column_metadata(wks)
                for cm in meta_list:
                    col_meta[cm.get("index", -1)] = cm
            except Exception:
                pass

            # ── v10.9: Fallback grouping from column names ──
            # If no column has user_params, try to infer from names
            has_any_uparams = any(
                col_meta.get(ci, {}).get("user_params")
                for ci in col_indices
            )
            inferred_params_used = False
            if not has_any_uparams and col_meta:
                meta_for_inference = [col_meta.get(ci, {}) for ci in col_indices if ci in col_meta]
                if meta_for_inference:
                    inferred = _infer_params_from_names(meta_for_inference)
                    if any(bool(p) for p in inferred):
                        # Inject synthetic user_params
                        for idx_in_list, ci in enumerate(col_indices):
                            if ci in col_meta and idx_in_list < len(inferred) and inferred[idx_in_list]:
                                if "user_params" not in col_meta[ci] or not col_meta[ci]["user_params"]:
                                    col_meta[ci]["user_params"] = inferred[idx_in_list]
                        inferred_params_used = True

            # Default max_fwhm
            max_fwhm_val = float(max_fwhm) if max_fwhm is not None else None

            fits = []
            fit_data = []  # Store (ci, label, x, y, result) for plotting

            for ci in col_indices:
                if ci < 0 or ci >= n_cols:
                    fits.append({"y_col": ci, "error": "Column index out of range"})
                    continue

                meta = col_meta.get(ci, {})
                label = meta.get("comments") or meta.get("name") or f"Col{ci}"
                user_params = meta.get("user_params", {})

                try:
                    x_clean, y_clean = _wks_xy_aligned(wks, x_col, ci)

                    if len(x_clean) < 15:
                        fits.append({"y_col": ci, "label": label, "error": "Too few points"})
                        continue

                    # Region selection (v10.9: auto x_range fallback)
                    eff_xmin = float(x_min) if x_min is not None else (auto_x_min if auto_x_min is not None else None)
                    eff_xmax = float(x_max) if x_max is not None else (auto_x_max if auto_x_max is not None else None)
                    if eff_xmin is not None or eff_xmax is not None:
                        xmin = eff_xmin if eff_xmin is not None else float(np.min(x_clean))
                        xmax = eff_xmax if eff_xmax is not None else float(np.max(x_clean))
                        region_mask = (x_clean >= xmin) & (x_clean <= xmax)
                        x_fit = x_clean[region_mask]
                        y_fit = y_clean[region_mask]
                    else:
                        x_fit = x_clean
                        y_fit = y_clean

                    if len(x_fit) < 10:
                        fits.append({"y_col": ci, "label": label, "error": "Fit region too small"})
                        continue

                    mf = max_fwhm_val if max_fwhm_val is not None else float(x_fit[-1] - x_fit[0]) / 3
                    fr = fit_multi_peak(x_fit, y_fit, model=model, max_fwhm=mf)

                    entry = {"y_col": ci, "label": label}
                    if user_params:
                        entry["user_params"] = user_params

                    if "error" in fr:
                        entry["error"] = fr["error"]
                        fits.append(entry)
                        continue

                    entry["r2"] = fr.get("r2")
                    entry["delta_r2"] = fr.get("delta_r2")
                    entry["separation_meV"] = fr.get("peak_separation_meV")

                    for pk_key in ("peak1", "peak2"):
                        pk = fr.get(pk_key, {})
                        pk_entry = {
                            "xc": pk.get("xc"), "w": pk.get("w"),
                            "A": pk.get("A"), "I_at_center": pk.get("I_at_center"),
                        }
                        if "xc_err" in pk:
                            pk_entry["xc_err"] = pk["xc_err"]
                        if "w_err" in pk:
                            pk_entry["w_err"] = pk["w_err"]
                        if "sigma" in pk:
                            pk_entry["sigma"] = pk["sigma"]
                            pk_entry["gamma"] = pk["gamma"]
                        entry[pk_key] = pk_entry

                    entry["y0"] = fr.get("y0")
                    fits.append(entry)
                    fit_data.append((ci, label, x_fit, y_fit, fr))

                except Exception as e:
                    fits.append({"y_col": ci, "label": label, "error": str(e)[:100]})

            result = {
                "workbook": workbook, "sheet": sheet, "model": model,
                "n_fits": len(fit_data),
                "n_errors": len(col_indices) - len(fit_data),
                "fits": fits,
            }

            # v10.9 metadata
            if auto_x_min is not None and x_min is None and x_max is None:
                result["auto_x_range"] = [round(auto_x_min, 4), round(auto_x_max, 4)]
                result["x_unit_detected"] = x_unit_detected
            if inferred_params_used:
                result["inferred_params"] = True

            print(f"[origin] BATCH_MULTI_FIT: {len(fit_data)} ok, {len(col_indices)-len(fit_data)} errors"
                  f" | auto_range={'['+str(round(auto_x_min,3))+','+str(round(auto_x_max,3))+']' if auto_x_min is not None else 'none'}"
                  f" | explicit_range=[{x_min},{x_max}]", flush=True)

            folder_path = _new_export_folder()

            # === Summary grid plot ===
            if fit_data:
                n_plots = len(fit_data)
                n_grid_cols = min(3, n_plots)
                n_grid_rows = (n_plots + n_grid_cols - 1) // n_grid_cols
                fig, axes = plt.subplots(n_grid_rows, n_grid_cols,
                                         figsize=(5 * n_grid_cols, 3.5 * n_grid_rows),
                                         squeeze=False)

                for idx, (ci, label, x_fit, y_fit, fr) in enumerate(fit_data):
                    row, col = divmod(idx, n_grid_cols)
                    ax = axes[row][col]
                    ax.plot(x_fit, y_fit, 'k-', linewidth=0.8, alpha=0.7)

                    x_curve = np.linspace(float(x_fit[0]), float(x_fit[-1]), 300)
                    pk1, pk2, y0 = fr["peak1"], fr["peak2"], fr["y0"]

                    if pk1["profile"] == "lorentz":
                        curve1 = lorentz_func(x_curve, 0, pk1["xc"], pk1["w"], pk1["A"])
                    else:
                        z1 = ((x_curve - pk1["xc"]) + 1j * pk1["gamma"]) / (pk1["sigma"] * np.sqrt(2))
                        curve1 = pk1["A"] * np.real(wofz(z1)) / (pk1["sigma"] * np.sqrt(2 * np.pi))
                    if pk2["profile"] == "lorentz":
                        curve2 = lorentz_func(x_curve, 0, pk2["xc"], pk2["w"], pk2["A"])
                    else:
                        z2 = ((x_curve - pk2["xc"]) + 1j * pk2["gamma"]) / (pk2["sigma"] * np.sqrt(2))
                        curve2 = pk2["A"] * np.real(wofz(z2)) / (pk2["sigma"] * np.sqrt(2 * np.pi))

                    ax.fill_between(x_curve, y0, y0 + curve1, alpha=0.3, color='red')
                    ax.fill_between(x_curve, y0, y0 + curve2, alpha=0.3, color='blue')
                    ax.plot(x_curve, y0 + curve1 + curve2, 'r--', linewidth=1.5)

                    sep = fr.get("peak_separation_meV", 0)
                    dr2 = fr.get("delta_r2", 0) or 0
                    short_label = label[:25] if len(label) > 25 else label
                    ax.set_title(f"col{ci}: {short_label}\n\u0394={sep:.1f}meV  \u0394R\u00b2={dr2:.4f}", fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.grid(True, alpha=0.2)

                for idx in range(n_plots, n_grid_rows * n_grid_cols):
                    row, col = divmod(idx, n_grid_cols)
                    axes[row][col].set_visible(False)

                fig.suptitle(f"{workbook} \u2014 {model} batch decomposition", fontsize=11, y=1.01)
                plt.tight_layout()
                result["summary_plot"] = export_figure(folder_path, fig, "batch_multi_fit.png")

            # === Waterfall plot (publication quality) ===
            if fit_data:
                all_uparams = []
                for ci, label, x_fit, y_fit, fr in fit_data:
                    all_uparams.append(col_meta.get(ci, {}).get("user_params", {}))

                pol_map = _detect_pol_keys(all_uparams)

                # Find grouping and polarization keys (v10.8: variance-aware)
                group_key, pol_key = _pick_group_pol_keys(all_uparams)

                if pol_key and group_key and pol_map and len(pol_map) >= 2:
                    # Group by parameter, pair by polarization
                    groups = {}
                    for ci, label, x_fit, y_fit, fr in fit_data:
                        up = col_meta.get(ci, {}).get("user_params", {})
                        pv, gv = up.get(pol_key, ""), up.get(group_key, "")
                        if pv and gv:
                            groups.setdefault(gv, {})[pv] = (x_fit, y_fit, fr, ci)

                    if len(groups) > 1:
                        def _numkey(s):
                            try: return float(s)
                            except: return 0.0
                        sorted_gvals = sorted(groups.keys(), key=_numkey)
                        n_g = len(sorted_gvals)

                        # ── Publication-quality dark theme (Kourmoulakis style) ──
                        _BG   = "#1A1A2E"
                        _GR   = "#2D2D4A"
                        _SP   = "#8888AA"
                        _TX   = "#E8E8F0"
                        _S_P  = "#E53935"   # σ+ red solid
                        _S_M  = "#42A5F5"   # σ- blue dashed
                        _ABG  = "#2A2A48"

                        fig2, ax2 = plt.subplots(figsize=(9, max(6, 2.0 * n_g)))
                        fig2.patch.set_facecolor(_BG)
                        ax2.set_facecolor(_BG)

                        # Offset from max peak range
                        all_ranges = []
                        for gv in sorted_gvals:
                            for xd, yd, _, _ in groups[gv].values():
                                all_ranges.append(float(np.max(yd) - np.min(yd)))
                        offset_step = max(all_ranges) * 1.20 if all_ranges else 100

                        # Sort pol keys: σ+ first
                        pol_sorted = sorted(pol_map.keys(),
                                            key=lambda p: (0 if "+" in pol_map.get(p, "") else 1))

                        # Assign σ+ → red solid, σ- → blue dashed
                        _POL_STY = {}
                        for _pidx, _pv in enumerate(pol_sorted):
                            _disp = pol_map.get(_pv, _pv)
                            if "+" in _disp:
                                _POL_STY[_pv] = {"color": _S_P, "ls": "-",  "lw": 2.0, "label": "\u03c3 +",  "alpha": 0.95}
                            else:
                                _POL_STY[_pv] = {"color": _S_M, "ls": "--", "lw": 1.5, "label": "\u03c3 \u2212", "alpha": 0.90}

                        # Collect experiment annotations
                        ann_parts = {}
                        for up in all_uparams:
                            for k, v in up.items():
                                kl = k.lower()
                                if kl in ("temp", "temperature", "t"):
                                    val = str(v)
                                    ann_parts["T"] = val if any(c.isalpha() for c in val) else f"{val}K"
                                elif kl in ("wavelength", "lambda", "excitation", "laser"):
                                    val = str(v)
                                    ann_parts["\u03bb"] = val if any(c.isalpha() for c in val) else f"{val}nm"
                                elif kl == "material":
                                    ann_parts[""] = str(v)
                                elif kl == "power" and kl != group_key.lower():
                                    ann_parts["P"] = str(v)

                        # Get x bounds
                        x_lo = min(float(xd[0]) for gv in sorted_gvals for xd, _, _, _ in groups[gv].values())
                        x_hi = max(float(xd[-1]) for gv in sorted_gvals for xd, _, _, _ in groups[gv].values())

                        # ── Draw spectra ──
                        _legend_added = set()
                        for i, gval in enumerate(sorted_gvals):
                            g = groups[gval]
                            offset = i * offset_step

                            for pv in pol_sorted:
                                if pv not in g:
                                    continue
                                xd, yd, fr_item, ci_val = g[pv]
                                y_base = float(np.percentile(yd, 5))
                                y_plot = yd - y_base + offset

                                sty = _POL_STY.get(pv, {"color": "#AAA", "ls": "-", "lw": 1.5, "label": pv, "alpha": 0.8})
                                lbl = sty["label"] if pv not in _legend_added else None
                                if lbl:
                                    _legend_added.add(pv)

                                ax2.plot(xd, y_plot, linestyle=sty["ls"], color=sty["color"],
                                         linewidth=sty["lw"], label=lbl, alpha=sty["alpha"], zorder=3)

                            # Group label (strain/temp/power) — axes-fraction coords
                            unit = ""
                            gkl = group_key.lower()
                            if gkl == "strain":
                                unit = "%"
                            elif gkl in ("temperature", "temp"):
                                unit = "K"
                            y_frac = (offset + offset_step * 0.35) / (n_g * offset_step) if n_g > 0 else 0
                            ax2.text(-0.04, y_frac, f"{gval}{unit}",
                                     transform=ax2.transAxes,
                                     fontsize=13, fontweight='bold',
                                     ha='right', va='center',
                                     color='#FFFFFF', family='sans-serif')

                        # ── Axes styling ──
                        ax2.set_xlabel("Energy (eV)", fontsize=14, color=_TX,
                                       fontfamily='sans-serif', labelpad=8)
                        ax2.set_ylabel("")
                        ax2.text(-0.04, 1.02, "PL Intensity (arb. units)",
                                 transform=ax2.transAxes, fontsize=13, color=_TX,
                                 ha='left', va='bottom', fontfamily='sans-serif',
                                 fontstyle='italic')
                        ax2.set_yticks([])
                        ax2.set_xlim(x_lo - 0.008, x_hi + 0.008)

                        ax2.spines['top'].set_visible(False)
                        ax2.spines['right'].set_visible(False)
                        ax2.spines['left'].set_visible(False)
                        ax2.spines['bottom'].set_color(_SP)
                        ax2.spines['bottom'].set_linewidth(1.2)
                        ax2.tick_params(axis='x', labelsize=12, colors=_TX,
                                        direction='in', width=1.0, length=5)
                        ax2.tick_params(axis='y', left=False)
                        ax2.grid(True, axis='x', alpha=0.15, color=_GR,
                                 linewidth=0.5, linestyle=':')

                        # Legend
                        _leg = ax2.legend(fontsize=12, loc='upper right',
                                          framealpha=0.85, edgecolor='none',
                                          facecolor=_ABG, labelcolor=_TX,
                                          handlelength=2.5, borderpad=0.6,
                                          handletextpad=0.8)
                        _leg.set_zorder(10)

                        # Experiment annotation box
                        if ann_parts:
                            ann_text = "\n".join(f"{k}={v}" if k else v
                                                 for k, v in ann_parts.items())
                            ax2.text(0.98, 0.78, ann_text, transform=ax2.transAxes,
                                     fontsize=11, ha='right', va='top',
                                     color=_TX, fontfamily='sans-serif',
                                     bbox=dict(boxstyle='round,pad=0.4',
                                               facecolor=_ABG, alpha=0.85,
                                               edgecolor=_SP, linewidth=0.8))

                        plt.subplots_adjust(left=0.14, right=0.95, top=0.93, bottom=0.08)
                        result["waterfall_plot"] = export_figure(folder_path, fig2, "waterfall.png", dpi=200)

            # ── Pre-built markdown for LLM embedding (v10.8) ──
            _md_parts = []
            if result.get("summary_plot"):
                _md_parts.append(f'![Batch Fit Summary]({result["summary_plot"]})')
            if result.get("waterfall_plot"):
                _md_parts.append(f'![Waterfall Plot]({result["waterfall_plot"]})')

            # Reorder: embed_markdown first, then metadata, then fits (v10.8.1)
            ordered = {}
            if _md_parts:
                ordered["embed_markdown"] = "\n".join(_md_parts)
            for k in ("workbook", "sheet", "model", "n_fits", "n_errors",
                       "summary_plot", "waterfall_plot"):
                if k in result:
                    ordered[k] = result[k]
            ordered["fits"] = result.get("fits", [])

            _cache_set(ck, ordered)
            print(f"[origin] BATCH_MULTI_FIT RESULT: {len(_md_parts)} images | embed_markdown={'YES' if ordered.get('embed_markdown') else 'NO'}"
                  f" | summary={'YES' if result.get('summary_plot') else 'NO'}"
                  f" | waterfall={'YES' if result.get('waterfall_plot') else 'NO'}", flush=True)
            return ordered

        except Exception as e:
            import traceback
            print(f"[origin] *** BATCH_MULTI_FIT CRASHED: {str(e)[:500]}", flush=True)
            traceback.print_exc()
            return {"error": str(e)[:150]}



# =============================================================================
# =============================================================================
def opj_batch_fit(
    file_id: str,
    workbook: str = "Merged",
    sheet: str = "Merged",
    x_col: int = 0,
    y_cols: str = "",
    use_subtracted: bool = True,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    profile: str = "lorentz",
    export_csv: bool = True
) -> Dict[str, Any]:
    """
    Fit peaks for multiple Y columns in a merged sheet.
    y_cols: comma-separated column indices, 'all', 'subtracted', or auto-detect.
    """
    if not ORIGIN_AVAILABLE:
        return {"error": "OriginPro not available"}

    with ORIGIN_LOCK:
        try:
            ensure_origin_running()
            ensure_project_open(file_id)

            wb = _find_workbook(workbook, "")
            if wb is None:
                return {"error": f"Workbook '{workbook}' not found"}

            wks = _find_sheet(wb, sheet)
            if wks is None:
                return {"error": f"Sheet '{sheet}' not found"}

            n_cols = wks.cols
            col_labels = wks.get_labels('L')

            if y_cols:
                if y_cols.lower() == "all":
                    fit_cols = list(range(1, n_cols))
                elif y_cols.lower() == "subtracted":
                    fit_cols = [i for i in range(n_cols) if "-bg" in (col_labels[i] if i < len(col_labels) else "")]
                else:
                    fit_cols = _parse_y_cols(y_cols, n_cols)
                    if not fit_cols:
                        fit_cols = [int(c.strip()) for c in y_cols.split(",") if c.strip().isdigit()]
            else:
                bg_cols = [i for i in range(n_cols) if "-bg" in (col_labels[i] if i < len(col_labels) else "")]
                if bg_cols and use_subtracted:
                    fit_cols = bg_cols
                else:
                    fit_cols = [i for i in range(1, n_cols)
                               if i < len(col_labels) and "bg" not in col_labels[i].lower()]

            if not fit_cols:
                return {"error": "No columns to fit"}

            # ── v10.9: Detect and warn about polarization-paired data ──
            # batch_fit ignores σ+/σ- pairing — warn if we detect it
            pol_warning = None
            try:
                meta_list = _read_column_metadata(wks)
                col_up = [m.get("user_params", {}) for m in meta_list if m.get("index", -1) in fit_cols]
                _, pol_key = _pick_group_pol_keys(col_up)
                if pol_key:
                    pol_warning = (
                        "WARNING: This data appears to have polarization pairing "
                        f"('{pol_key}' key detected in user_params). batch_fit ignores "
                        "polarization and treats columns sequentially. For correct analysis, "
                        "use origin.trend_plot or origin.batch_multi_fit instead."
                    )
            except Exception:
                pass

            results = []
            folder_path = _new_export_folder()

            for col_idx in fit_cols:
                if col_idx >= n_cols:
                    continue

                label = col_labels[col_idx] if col_idx < len(col_labels) else f"Col{col_idx}"

                try:
                    x_clean, y_clean = _wks_xy_aligned(wks, x_col, col_idx)

                    if len(x_clean) < 10:
                        results.append({"label": label, "err": "too few points"})
                        continue

                    if x_min is not None or x_max is not None:
                        xmin = float(x_min) if x_min is not None else float(np.min(x_clean))
                        xmax = float(x_max) if x_max is not None else float(np.max(x_clean))
                        region_mask = (x_clean >= xmin) & (x_clean <= xmax)
                        x_fit = x_clean[region_mask]
                        y_fit = y_clean[region_mask]
                    else:
                        peak_idx = int(np.argmax(y_clean))
                        window = max(20, len(x_clean) // 10)
                        lo = max(0, peak_idx - window)
                        hi = min(len(x_clean), peak_idx + window)
                        x_fit = x_clean[lo:hi]
                        y_fit = y_clean[lo:hi]

                    if len(x_fit) < 5:
                        results.append({"label": label, "err": "region small"})
                        continue

                    fit = fit_peak_scipy(x_fit, y_fit, profile)

                    if "error" in fit:
                        results.append({"label": label, "err": fit["error"][:30]})
                    else:
                        power = _parse_power_value(label.replace("-bg", ""))
                        results.append({
                            "label": label,
                            "power": power if power != float('inf') else None,
                            "xc": fit.get("xc"),
                            "w": fit.get("w"),
                            "A": fit.get("A"),
                            "y0": fit.get("y0"),
                            "r2": fit.get("r2"),
                        })

                except Exception as e:
                    results.append({"label": label, "err": str(e)[:30]})

            successful = [r for r in results if "xc" in r]
            successful.sort(key=lambda r: r.get("power") or float('inf'))

            plot_url = None
            if len(successful) >= 2:
                fig, ax = plt.subplots(figsize=(8, 5))

                powers = [r.get("power") for r in successful if r.get("power") is not None]
                xc_vals = [r["xc"] for r in successful if r.get("power") is not None]

                if powers and len(powers) == len(xc_vals):
                    ax.plot(powers, xc_vals, 'bo-', markersize=8, linewidth=1.5)
                    ax.set_xlabel("Power (\u03bcW)")
                    ax.set_ylabel("Peak Position xc (eV)")
                else:
                    ax.plot(range(len(successful)), [r["xc"] for r in successful], 'bo-', markersize=8)
                    ax.set_xticks(range(len(successful)))
                    ax.set_xticklabels([r["label"] for r in successful], rotation=45, ha='right')
                    ax.set_ylabel("Peak Position xc (eV)")

                ax.set_title(f"Peak Position vs Power ({profile.title()} Fit)")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                plot_url = export_figure(folder_path, fig, "batch_fit_summary.png")

            csv_url = None
            if export_csv and successful:
                csv_lines = ["label,power,xc,w,A,y0,r2"]
                for r in successful:
                    csv_lines.append(f"{r['label']},{r.get('power','')},{r.get('xc','')},{r.get('w','')},{r.get('A','')},{r.get('y0','')},{r.get('r2','')}")
                csv_data = "\n".join(csv_lines).encode('utf-8')
                csv_url = export_bytes(folder_path, csv_data, "fit_results.csv", "text/csv")

            out = {
                "workbook": workbook,
                "profile": profile,
                "results": results,
                "ok": len(successful),
                "total": len(results),
                "plot": plot_url,
                "csv": csv_url,
            }
            if pol_warning:
                out["warning"] = pol_warning
            return out

        except Exception as e:
            return {"error": str(e)[:150]}


# =============================================================================
# MCP TOOL 7: opj_create_graph — create line graph
# =============================================================================
def opj_create_graph(
    file_id: str,
    workbook: str,
    sheet: str = "Merged",
    x_col: int = 0,
    y_cols: str = "",
    title: str = "",
    export_png: bool = True
) -> Dict[str, Any]:
    """
    Create a line graph from worksheet columns.
    y_cols: comma-separated column indices, or 'subtracted' for -bg columns.
    """
    if not ORIGIN_AVAILABLE:
        return {"error": "OriginPro not available"}

    with ORIGIN_LOCK:
        try:
            ensure_origin_running()
            ensure_project_open(file_id)

            wb = _find_workbook(workbook, "")
            if wb is None:
                return {"error": f"Workbook '{workbook}' not found"}

            wks = _find_sheet(wb, sheet)
            if wks is None:
                return {"error": f"Sheet '{sheet}' not found"}

            n_cols = wks.cols
            col_labels = wks.get_labels('L')

            if y_cols:
                if y_cols.lower() == "subtracted":
                    plot_cols = [i for i in range(n_cols) if "-bg" in (col_labels[i] if i < len(col_labels) else "")]
                else:
                    plot_cols = _parse_y_cols(y_cols, n_cols)
                    if not plot_cols:
                        plot_cols = [int(c.strip()) for c in y_cols.split(",") if c.strip().isdigit()]
            else:
                plot_cols = list(range(1, min(n_cols, 20)))

            if not plot_cols:
                return {"error": "No columns to plot"}

            folder_path = _new_export_folder()
            fig, ax = plt.subplots(figsize=(10, 6))

            for col_idx in plot_cols:
                if col_idx >= n_cols:
                    continue
                label = col_labels[col_idx] if col_idx < len(col_labels) else f"Col{col_idx}"
                try:
                    x_plot, y_plot = _wks_xy_aligned(wks, x_col, col_idx)
                    ax.plot(x_plot, y_plot, label=label, alpha=0.8)
                except Exception:
                    continue

            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel("Intensity (counts)")
            ax.set_title(title or f"{workbook} - {sheet}")
            ax.legend(fontsize=7, loc='best', ncol=2)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_url = export_figure(folder_path, fig, "graph.png")

            return {
                "workbook": workbook,
                "sheet": sheet,
                "columns_plotted": len(plot_cols),
                "plot": plot_url,
            }

        except Exception as e:
            return {"error": str(e)[:150]}


# =============================================================================
# MCP TOOL 8: opj_save_project — save modified OPJ
# =============================================================================
def opj_save_project(
    file_id: str,
    output_name: str = ""
) -> Dict[str, Any]:
    """Save current project state to a new OPJ file. Returns URL."""
    if not ORIGIN_AVAILABLE:
        return {"error": "OriginPro not available"}

    with ORIGIN_LOCK:
        try:
            ensure_origin_running()
            cache_path = ensure_project_open(file_id)

            folder_path = _new_export_folder()

            if not output_name:
                output_name = _display_name(cache_path)
            if not output_name.lower().endswith(".opj"):
                output_name += ".opj"

            opj_path = os.path.join(folder_path, output_name)
            op.save(opj_path)

            with open(opj_path, "rb") as f:
                opj_data = f.read()
            opj_url = export_bytes(folder_path, opj_data, output_name, "application/octet-stream")

            return {
                "file": output_name,
                "url": opj_url,
            }

        except Exception as e:
            return {"error": str(e)[:150]}


# =============================================================================
# MCP TOOL 9: opj_export_csv — export worksheet to CSV
# =============================================================================
def opj_export_csv(
    file_id: str,
    workbook: str,
    sheet: str = "Merged",
    filename: str = ""
) -> Dict[str, Any]:
    """Export worksheet data to CSV file."""
    if not ORIGIN_AVAILABLE:
        return {"error": "OriginPro not available"}

    with ORIGIN_LOCK:
        try:
            ensure_origin_running()
            ensure_project_open(file_id)

            wb = _find_workbook(workbook, "")
            if wb is None:
                return {"error": f"Workbook '{workbook}' not found"}

            wks = _find_sheet(wb, sheet)
            if wks is None:
                return {"error": f"Sheet '{sheet}' not found"}

            df = wks.to_df()

            folder_path = _new_export_folder()
            csv_name = filename if filename else f"{workbook}_{sheet}.csv"
            if not csv_name.lower().endswith(".csv"):
                csv_name += ".csv"

            csv_data = df.to_csv(index=False).encode('utf-8')
            csv_url = export_bytes(folder_path, csv_data, csv_name, "text/csv")

            return {
                "file": csv_name,
                "rows": len(df),
                "cols": len(df.columns),
                "url": csv_url,
            }

        except Exception as e:
            return {"error": str(e)[:150]}


# =============================================================================
# MCP TOOL 10: opj_trend_plot — trend analysis from batch fit results
# =============================================================================
def opj_trend_plot(
    file_id: str,
    workbook: str,
    sheet: str = "Data",
    x_col: int = 0,
    y_cols: str = "",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    model: str = "lorentz+lorentz",
    max_fwhm: Optional[float] = None,
    folder: str = "",
    plots: str = "xc,pc",
) -> Dict[str, Any]:
    """
    Trend analysis: runs batch_multi_fit then plots xc vs parameter, Pc vs parameter.
    plots: comma-separated list of plot types: 'xc' (peak position), 'pc' (circular polarization),
           'width' (FWHM), 'area' (integrated area), 'separation' (peak splitting).
    Returns batch fit results plus trend plot URLs.
    """
    # Dedup: return cached result for identical calls
    ck = _cache_key("trend_plot", file_id=file_id, workbook=workbook,
                     sheet=sheet, x_col=x_col, y_cols=y_cols, x_min=x_min,
                     x_max=x_max, model=model, max_fwhm=max_fwhm, folder=folder,
                     plots=plots)
    cached = _cache_get(ck)
    if cached:
        return cached

    # First run the batch fit
    batch_result = opj_batch_multi_fit(
        file_id=file_id, workbook=workbook, sheet=sheet, x_col=x_col,
        y_cols=y_cols, x_min=x_min, x_max=x_max, model=model,
        max_fwhm=max_fwhm, folder=folder,
    )

    print(f"[origin] TREND_PLOT: batch_result keys={list(batch_result.keys())[:10]}"
          f" | has_error={'error' in batch_result}"
          f" | n_fits={batch_result.get('n_fits', '?')}"
          f" | embed_markdown={'YES' if batch_result.get('embed_markdown') else 'NO'}", flush=True)
    if "error" in batch_result:
        print(f"[origin] TREND_PLOT: batch_multi_fit returned error: {batch_result['error']}", flush=True)
        return batch_result

    fits = batch_result.get("fits", [])
    if not fits:
        return batch_result

    # Identify grouping parameter and polarization (v10.8: variance-aware)
    all_up = [f.get("user_params", {}) for f in fits if "user_params" in f]
    group_key, pol_key = _pick_group_pol_keys(all_up)

    if not group_key:
        batch_result["trend_note"] = (
            "No grouping parameter found in user_params or column names. "
            "The file may need manual metadata (user parameter rows in OriginLab) "
            "or the columns may not represent a parametric study. "
            "Try batch_multi_fit or batch_fit instead for non-parametric data."
        )
        return batch_result

    # Build data table: {group_val: {pol_val: fit_entry}}
    grouped = {}
    for f in fits:
        if "error" in f:
            continue
        up = f.get("user_params", {})
        gv = up.get(group_key, "")
        pv = up.get(pol_key, "") if pol_key else "all"
        if gv:
            grouped.setdefault(gv, {})[pv] = f

    if not grouped:
        batch_result["trend_note"] = "Could not group fits by parameter"
        return batch_result

    # Sort group values numerically
    def _nk(s):
        try: return float(s)
        except: return 0.0
    sorted_gvals = sorted(grouped.keys(), key=_nk)
    x_param = [_nk(gv) for gv in sorted_gvals]

    # Detect pol pairs
    all_pols = set()
    for g in grouped.values():
        all_pols.update(g.keys())
    pol_map = _detect_pol_keys([f.get("user_params", {}) for f in fits if "user_params" in f])

    requested_plots = [p.strip().lower() for p in plots.split(",")]
    folder_path = _new_export_folder()
    trend_urls = {}

    # === xc trend plot ===
    if "xc" in requested_plots:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = {"red": "#e74c3c", "blue": "#3498db"}
        markers = ["o", "s", "^", "D"]

        for j, pv in enumerate(sorted(all_pols)):
            xc1_vals, xc2_vals, x_vals = [], [], []
            xc1_err, xc2_err = [], []
            for gv in sorted_gvals:
                if pv in grouped.get(gv, {}):
                    f = grouped[gv][pv]
                    x_vals.append(_nk(gv))
                    xc1_vals.append(f.get("peak1", {}).get("xc"))
                    xc2_vals.append(f.get("peak2", {}).get("xc"))
                    xc1_err.append(f.get("peak1", {}).get("xc_err", 0))
                    xc2_err.append(f.get("peak2", {}).get("xc_err", 0))

            disp = pol_map.get(pv, pv)
            mk = markers[j % len(markers)]
            if any(e > 0 for e in xc1_err):
                ax.errorbar(x_vals, xc1_vals, yerr=xc1_err, fmt=f'{mk}-',
                           label=f'Peak 1 ({disp})', markersize=7, capsize=3,
                           linewidth=1.5, color=["#e74c3c", "#3498db"][j % 2])
            else:
                ax.plot(x_vals, xc1_vals, f'{mk}-', label=f'Peak 1 ({disp})',
                       markersize=7, linewidth=1.5, color=["#e74c3c", "#3498db"][j % 2])
            if any(v is not None and v != xc1_vals[i] for i, v in enumerate(xc2_vals) if v):
                ax.plot(x_vals, xc2_vals, f'{mk}--', label=f'Peak 2 ({disp})',
                       markersize=5, linewidth=1.0, alpha=0.6,
                       color=["#e74c3c", "#3498db"][j % 2])

        unit_label = group_key.replace("_", " ").title()
        unit = ""
        if group_key.lower() == "strain":
            unit = " (%)"
        elif group_key.lower() in ("temperature", "temp"):
            unit = " (K)"
        ax.set_xlabel(f"{unit_label}{unit}", fontsize=12)
        ax.set_ylabel("Peak Position (eV)", fontsize=12)
        ax.set_title(f"Peak Position vs {unit_label}", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        trend_urls["xc_trend"] = export_figure(folder_path, fig, "xc_vs_param.png", dpi=180)

    # === Pc (circular polarization) trend plot ===
    if "pc" in requested_plots and pol_key and len(all_pols) == 2:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        pol_sorted = sorted(all_pols)

        pc_vals, pc_x = [], []
        for gv in sorted_gvals:
            g = grouped.get(gv, {})
            if len(g) == 2:
                p1, p2 = pol_sorted[0], pol_sorted[1]
                f1, f2 = g.get(p1), g.get(p2)
                if f1 and f2:
                    # Use I_at_center for Pc calculation (peak1 = main exciton)
                    I1 = f1.get("peak1", {}).get("I_at_center") or f1.get("peak1", {}).get("A", 0)
                    I2 = f2.get("peak1", {}).get("I_at_center") or f2.get("peak1", {}).get("A", 0)
                    if I1 and I2 and (I1 + I2) > 0:
                        # Convention: Pc = (I_sigma+ - I_sigma-) / (I_sigma+ + I_sigma-)
                        # Determine which pol is sigma+
                        plus_label = pol_map.get(p1, p1)
                        if "+" in plus_label or "+" in plus_label:
                            Iplus, Iminus = I1, I2
                        elif "-" in plus_label:
                            Iplus, Iminus = I2, I1
                        else:
                            Iplus, Iminus = I1, I2  # fallback

                        pc = (Iplus - Iminus) / (Iplus + Iminus) * 100
                        pc_vals.append(pc)
                        pc_x.append(_nk(gv))

        if pc_vals:
            ax.plot(pc_x, pc_vals, 'ko-', markersize=8, linewidth=2)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.fill_between(pc_x, 0, pc_vals, alpha=0.15, color='purple')

            unit_label = group_key.replace("_", " ").title()
            unit = " (%)" if group_key.lower() == "strain" else ""
            ax.set_xlabel(f"{unit_label}{unit}", fontsize=12)
            ax.set_ylabel("Circular Polarization $P_c$ (%)", fontsize=12)
            ax.set_title(f"Valley Polarization vs {unit_label}", fontsize=13)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            trend_urls["pc_trend"] = export_figure(folder_path, fig, "Pc_vs_param.png", dpi=180)

    # === Width trend ===
    if "width" in requested_plots:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for j, pv in enumerate(sorted(all_pols)):
            w_vals, x_vals = [], []
            for gv in sorted_gvals:
                if pv in grouped.get(gv, {}):
                    f = grouped[gv][pv]
                    w = f.get("peak1", {}).get("w")
                    if w:
                        x_vals.append(_nk(gv))
                        w_vals.append(w * 1000)  # meV
            disp = pol_map.get(pv, pv)
            ax.plot(x_vals, w_vals, 'o-', label=f'{disp}', markersize=7, linewidth=1.5)

        unit_label = group_key.replace("_", " ").title()
        ax.set_xlabel(f"{unit_label}", fontsize=12)
        ax.set_ylabel("FWHM (meV)", fontsize=12)
        ax.set_title(f"Linewidth vs {unit_label}", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        trend_urls["width_trend"] = export_figure(folder_path, fig, "width_vs_param.png", dpi=180)

    # === Separation trend ===
    if "separation" in requested_plots:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for j, pv in enumerate(sorted(all_pols)):
            sep_vals, x_vals = [], []
            for gv in sorted_gvals:
                if pv in grouped.get(gv, {}):
                    f = grouped[gv][pv]
                    sep = f.get("separation_meV")
                    if sep:
                        x_vals.append(_nk(gv))
                        sep_vals.append(sep)
            disp = pol_map.get(pv, pv)
            ax.plot(x_vals, sep_vals, 'o-', label=f'{disp}', markersize=7, linewidth=1.5)

        unit_label = group_key.replace("_", " ").title()
        ax.set_xlabel(f"{unit_label}", fontsize=12)
        ax.set_ylabel("Peak Separation (meV)", fontsize=12)
        ax.set_title(f"Peak Splitting vs {unit_label}", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        trend_urls["separation_trend"] = export_figure(folder_path, fig, "separation_vs_param.png", dpi=180)

    batch_result["trends"] = trend_urls

    # ── Build final response with images FIRST (v10.8.1) ──
    # LLMs process JSON linearly; embed_markdown must precede the bulky fits array
    _md_parts = []
    if batch_result.get("waterfall_plot"):
        _md_parts.append(f'![Waterfall Plot]({batch_result["waterfall_plot"]})')
    if batch_result.get("summary_plot"):
        _md_parts.append(f'![Batch Fit Summary]({batch_result["summary_plot"]})')
    _trend_labels = {
        "xc_trend": "Peak Position Trend",
        "pc_trend": "Valley Polarization Trend",
        "width_trend": "FWHM Trend",
        "separation_trend": "Peak Splitting Trend",
    }
    for tk, label in _trend_labels.items():
        if tk in trend_urls:
            _md_parts.append(f"![{label}]({trend_urls[tk]})")

    # Compact fit summary: only the fields needed for the summary table
    compact_fits = []
    for f in batch_result.get("fits", []):
        if "error" in f:
            continue
        cf = {"label": f.get("label", ""), "r2": f.get("r2")}
        up = f.get("user_params", {})
        if up:
            cf["user_params"] = up
        for pk_key in ("peak1", "peak2"):
            pk = f.get(pk_key, {})
            if pk:
                cf[pk_key] = {"xc": pk.get("xc"), "w": pk.get("w")}
                if "xc_err" in pk:
                    cf[pk_key]["xc_err"] = pk["xc_err"]
        if f.get("separation_meV") is not None:
            cf["separation_meV"] = f["separation_meV"]
        compact_fits.append(cf)

    # Build ordered response: images first, data second
    ordered = {}
    if _md_parts:
        ordered["embed_markdown"] = "\n".join(_md_parts)
    ordered["workbook"] = batch_result.get("workbook", "")
    ordered["sheet"] = batch_result.get("sheet", "")
    ordered["model"] = batch_result.get("model", "")
    ordered["group_key"] = group_key
    ordered["n_groups"] = len(grouped)
    ordered["n_fits"] = batch_result.get("n_fits", 0)
    ordered["fits_compact"] = compact_fits
    ordered["trends"] = trend_urls

    _cache_set(ck, ordered)
    return ordered


# =============================================================================
# MCP TOOL 11: opj_normalize — normalize spectra
# =============================================================================
def opj_normalize(
    file_id: str,
    workbook: str,
    sheet: str = "Data",
    x_col: int = 0,
    y_cols: str = "",
    mode: str = "peak",
    folder: str = "",
    export_plot: bool = True,
) -> Dict[str, Any]:
    """
    Normalize spectra. mode: 'peak' (max=1), 'area' (integral=1), 'value' (divide by value at x_col peak).
    Returns normalized data preview and optional overlay plot.
    """
    if not ORIGIN_AVAILABLE:
        return {"error": "OriginPro not available"}

    with ORIGIN_LOCK:
        try:
            ensure_origin_running()
            ensure_project_open(file_id)

            wb = _find_workbook(workbook, folder)
            if wb is None:
                return {"error": f"Workbook '{workbook}' not found"}
            wks = _find_sheet(wb, sheet)
            if wks is None:
                return {"error": f"Sheet '{sheet}' not found"}

            n_cols = wks.cols
            col_indices = _parse_y_cols(y_cols, n_cols)
            if not col_indices:
                col_indices = list(range(1, n_cols))

            mode = mode.lower().strip()
            normalized = []
            folder_path = _new_export_folder()

            if export_plot:
                fig, ax = plt.subplots(figsize=(8, 5))

            for ci in col_indices:
                if ci >= n_cols:
                    continue
                try:
                    x, y = _wks_xy_aligned(wks, x_col, ci)
                    if len(x) < 5:
                        continue

                    if mode == "peak":
                        ymax = float(np.max(y))
                        y_norm = y / ymax if ymax > 0 else y
                    elif mode == "area":
                        area = float(np.trapz(y, x))
                        y_norm = y / area if area > 0 else y
                    else:
                        ymax = float(np.max(y))
                        y_norm = y / ymax if ymax > 0 else y

                    meta = col_meta.get(ci, {}) if 'col_meta' in dir() else {}
                    label = meta.get("comments") or meta.get("name") or f"Col{ci}"
                    normalized.append({"y_col": ci, "label": label, "n_points": len(x)})

                    if export_plot:
                        ax.plot(x, y_norm, label=label[:30], alpha=0.8, linewidth=1.2)

                except Exception as e:
                    normalized.append({"y_col": ci, "error": str(e)[:60]})

            result = {"workbook": workbook, "mode": mode, "normalized": normalized}

            if export_plot and normalized:
                ax.set_xlabel("Energy (eV)", fontsize=11)
                ax.set_ylabel("Normalized Intensity", fontsize=11)
                ax.set_title(f"{workbook} — normalized ({mode})", fontsize=12)
                ax.legend(fontsize=7, loc='best', ncol=2)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                result["plot"] = export_figure(folder_path, fig, "normalized.png")

            return result

        except Exception as e:
            return {"error": str(e)[:150]}


# =============================================================================
# MCP TOOL 12: opj_smooth — Savitzky-Golay smoothing
# =============================================================================
def opj_smooth(
    file_id: str,
    workbook: str,
    sheet: str = "Data",
    x_col: int = 0,
    y_col: int = 1,
    window: int = 11,
    polyorder: int = 3,
    folder: str = "",
) -> Dict[str, Any]:
    """
    Smooth spectrum with Savitzky-Golay filter.
    window: filter window length (must be odd). polyorder: polynomial order.
    Returns smoothed data preview and comparison plot.
    """
    if not ORIGIN_AVAILABLE:
        return {"error": "OriginPro not available"}

    from scipy.signal import savgol_filter

    with ORIGIN_LOCK:
        try:
            ensure_origin_running()
            ensure_project_open(file_id)

            wb = _find_workbook(workbook, folder)
            if wb is None:
                return {"error": f"Workbook '{workbook}' not found"}
            wks = _find_sheet(wb, sheet)
            if wks is None:
                return {"error": f"Sheet '{sheet}' not found"}

            x, y = _wks_xy_aligned(wks, x_col, y_col)
            if len(x) < window:
                return {"error": f"Data has {len(x)} points, need at least {window}"}

            # Ensure window is odd
            if window % 2 == 0:
                window += 1
            if polyorder >= window:
                polyorder = window - 1

            y_smooth = savgol_filter(y, window, polyorder)

            folder_path = _new_export_folder()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, y, 'b-', alpha=0.4, linewidth=0.8, label='Raw')
            ax.plot(x, y_smooth, 'r-', linewidth=1.5, label=f'Smoothed (w={window}, p={polyorder})')
            ax.set_xlabel("Energy (eV)", fontsize=11)
            ax.set_ylabel("Intensity", fontsize=11)
            ax.set_title(f"{workbook} — Savitzky-Golay smoothing")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_url = export_figure(folder_path, fig, "smoothed.png")

            step = max(1, len(x) // 200)
            return {
                "workbook": workbook, "y_col": y_col,
                "window": window, "polyorder": polyorder,
                "points": len(x),
                "x": [round(float(v), 5) for v in x[::step]],
                "y_smooth": [round(float(v), 3) for v in y_smooth[::step]],
                "plot": plot_url,
            }

        except Exception as e:
            return {"error": str(e)[:150]}


# =============================================================================
# MCP TOOL 13: opj_find_peaks — auto-detect peaks
# =============================================================================
def opj_find_peaks(
    file_id: str,
    workbook: str,
    sheet: str = "Data",
    x_col: int = 0,
    y_col: int = 1,
    prominence: float = 0.1,
    min_distance_eV: float = 0.01,
    folder: str = "",
) -> Dict[str, Any]:
    """
    Auto-detect peaks in spectrum using scipy.signal.find_peaks.
    prominence: minimum peak prominence (fraction of max intensity).
    min_distance_eV: minimum distance between peaks in x-axis units.
    Returns peak positions, heights, widths, and annotated plot.
    """
    if not ORIGIN_AVAILABLE:
        return {"error": "OriginPro not available"}

    from scipy.signal import find_peaks, peak_widths

    with ORIGIN_LOCK:
        try:
            ensure_origin_running()
            ensure_project_open(file_id)

            wb = _find_workbook(workbook, folder)
            if wb is None:
                return {"error": f"Workbook '{workbook}' not found"}
            wks = _find_sheet(wb, sheet)
            if wks is None:
                return {"error": f"Sheet '{sheet}' not found"}

            x, y = _wks_xy_aligned(wks, x_col, y_col)
            if len(x) < 10:
                return {"error": "Not enough data points"}

            # Convert min_distance_eV to index distance
            dx = float(np.mean(np.diff(x))) if len(x) > 1 else 1.0
            min_dist_idx = max(1, int(min_distance_eV / abs(dx)))

            # Prominence in absolute units
            y_range = float(np.max(y) - np.min(y))
            prom_abs = prominence * y_range

            peaks, props = find_peaks(y, prominence=prom_abs, distance=min_dist_idx)

            if len(peaks) == 0:
                return {"workbook": workbook, "peaks": [], "note": "No peaks found. Try lower prominence."}

            # Get widths
            try:
                widths, w_heights, w_left, w_right = peak_widths(y, peaks, rel_height=0.5)
                # Convert from index to x units
                widths_eV = widths * abs(dx)
            except Exception:
                widths_eV = [None] * len(peaks)

            peak_list = []
            for i, pi in enumerate(peaks):
                entry = {
                    "x": round(float(x[pi]), 5),
                    "y": round(float(y[pi]), 3),
                    "prominence": round(float(props["prominences"][i]), 3),
                }
                if widths_eV[i] is not None:
                    entry["fwhm_eV"] = round(float(widths_eV[i]), 5)
                    entry["fwhm_meV"] = round(float(widths_eV[i]) * 1000, 2)
                peak_list.append(entry)

            # Plot
            folder_path = _new_export_folder()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, y, 'b-', linewidth=1.2)
            for pk in peak_list:
                ax.axvline(pk["x"], color='red', linestyle='--', alpha=0.5)
                ax.annotate(f'{pk["x"]:.4f}',
                           xy=(pk["x"], pk["y"]),
                           xytext=(0, 10), textcoords="offset points",
                           fontsize=8, ha='center', color='red',
                           arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
            ax.plot([pk["x"] for pk in peak_list], [pk["y"] for pk in peak_list],
                   'rv', markersize=10)
            ax.set_xlabel("Energy (eV)", fontsize=11)
            ax.set_ylabel("Intensity", fontsize=11)
            ax.set_title(f"{workbook} — {len(peak_list)} peaks detected")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_url = export_figure(folder_path, fig, "peaks.png")

            return {
                "workbook": workbook, "y_col": y_col,
                "n_peaks": len(peak_list),
                "peaks": peak_list,
                "plot": plot_url,
            }

        except Exception as e:
            return {"error": str(e)[:150]}


# =============================================================================
# Tool Registry
# =============================================================================
PUBLIC_TOOLS = {
    "opj_inspect": opj_inspect,
    "opj_get_data": opj_get_data,
    "opj_merge_power_series": opj_merge_power_series,
    "opj_fit_peak": opj_fit_peak,
    "opj_multi_fit": opj_multi_fit,
    "opj_batch_multi_fit": opj_batch_multi_fit,
    "opj_batch_fit": opj_batch_fit,
    "opj_create_graph": opj_create_graph,
    "opj_save_project": opj_save_project,
    "opj_export_csv": opj_export_csv,
    "opj_trend_plot": opj_trend_plot,
    "opj_normalize": opj_normalize,
    "opj_smooth": opj_smooth,
    "opj_find_peaks": opj_find_peaks,
}

TOOL_ALIASES = {
    # Legacy spec_ prefixed names (backward compat)
    "spec_opj_inspect": "opj_inspect",
    "spec_opj_get_data": "opj_get_data",
    "spec_opj_merge_power_series": "opj_merge_power_series",
    "spec_opj_fit_peak": "opj_fit_peak",
    "spec_opj_batch_fit": "opj_batch_fit",
    "spec_opj_batch_multi_fit": "opj_batch_multi_fit",
    "spec_opj_create_graph": "opj_create_graph",
    "spec_opj_save_project": "opj_save_project",
    "spec_opj_export_csv": "opj_export_csv",
    # Dot-notation aliases
    "opj_inspect": "opj_inspect",
    "opj_get_data": "opj_get_data",
    "opj_fit_peak": "opj_fit_peak",
    "opj_multi_fit": "opj_multi_fit",
    "opj_batch_fit": "opj_batch_fit",
    "opj_create_graph": "opj_create_graph",
    "opj_save_project": "opj_save_project",
    "opj_export_csv": "opj_export_csv",
    # Short names
    "opj_merge": "opj_merge_power_series",
    "opj_fit": "opj_fit_peak",
    "opj_bmfit": "opj_batch_multi_fit",
    "opj_plot": "opj_create_graph",
    "opj_save": "opj_save_project",
    "opj_csv": "opj_export_csv",
    "opj_trends": "opj_trend_plot",
    "opj_norm": "opj_normalize",
    "opj_peaks": "opj_find_peaks",
}

# Also register origin.* and spec.* dot patterns
for name in list(PUBLIC_TOOLS.keys()):
    TOOL_ALIASES[f"origin.{name}"] = name
    TOOL_ALIASES[f"spec.{name}"] = name
    TOOL_ALIASES[f"origin.{name.replace('opj_', '')}"] = name
    TOOL_ALIASES[f"spec.{name.replace('opj_', '')}"] = name


def _resolve_tool(name: str) -> str:
    if name in PUBLIC_TOOLS:
        return name
    if name in TOOL_ALIASES:
        return TOOL_ALIASES[name]
    if name.startswith("origin.") or name.startswith("spec."):
        tail = name.split(".", 1)[1]
        if tail in PUBLIC_TOOLS:
            return tail
        if f"opj_{tail}" in PUBLIC_TOOLS:
            return f"opj_{tail}"
    return name


# =============================================================================
# MCP Server
# =============================================================================
if __name__ == "__main__":
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn
    import inspect

    print(f"Starting OriginMCP Server v10.9 on port 12009...", flush=True)
    print(f"Tools: {list(PUBLIC_TOOLS.keys())}", flush=True)

    app = FastAPI(title="OriginMCP", version="1.0.0")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "origin": ORIGIN_AVAILABLE,
            "build": BUILD_ID,
            "tools": len(PUBLIC_TOOLS),
        }

    @app.post("/mcp")
    async def mcp_endpoint(request: Request):
        body = None
        try:
            body = await request.json()
            method = body.get("method")
            request_id = body.get("id")

            if method == "tools/list":
                tools_list = []
                for tool_name, tool_func in PUBLIC_TOOLS.items():
                    sig = inspect.signature(tool_func)
                    props = {}
                    required = []
                    for pname, p in sig.parameters.items():
                        ann = None if p.annotation is inspect._empty else p.annotation
                        props[pname] = {"type": _annotation_json_type(ann)}
                        if p.default is inspect._empty:
                            required.append(pname)

                    doc = (tool_func.__doc__ or "").strip().split("\n")[0]

                    tools_list.append({
                        "name": tool_name,
                        "description": doc[:150],
                        "inputSchema": {
                            "type": "object",
                            "properties": props,
                            "required": required,
                        },
                    })
                return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools_list}})

            if method == "tools/call":
                tool_name = _resolve_tool(body["params"]["name"])
                args = body["params"].get("arguments", {})

                if tool_name not in PUBLIC_TOOLS:
                    return JSONResponse({
                        "jsonrpc": "2.0", "id": request_id,
                        "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                    })

                sig = inspect.signature(PUBLIC_TOOLS[tool_name])
                typed_args = {}
                for pname, pval in args.items():
                    if pname in sig.parameters:
                        ann = sig.parameters[pname].annotation
                        ann = None if ann is inspect._empty else ann
                        typed_args[pname] = _coerce_value(ann, pval)
                    else:
                        typed_args[pname] = pval

                result = PUBLIC_TOOLS[tool_name](**typed_args)
                text = _compact_json(result, MAX_TOOL_TEXT_CHARS)

                return JSONResponse({
                    "jsonrpc": "2.0", "id": request_id,
                    "result": {"content": [{"type": "text", "text": text}]}
                })

            if method == "initialize":
                return JSONResponse({
                    "jsonrpc": "2.0", "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "origin", "version": "1.0.0"},
                    }
                })

            return JSONResponse({
                "jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            })

        except Exception as e:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": body.get("id") if isinstance(body, dict) else None,
                "error": {"code": -32000, "message": str(e)[:150]}
            })

    uvicorn.run(app, host="0.0.0.0", port=12009, log_level="info")
