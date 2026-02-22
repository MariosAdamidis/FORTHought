# micro/server.py  --  SEM Periodicity Analysis Suite
# Author:  Marios Adamidis (FORTHought Lab)
# Version: 1.0.0
#
#
#
# Architecture (from GPT-5):
#   - Disk-first file_id ingestion (OWUI HTTP fallback)
#   - Calibrate BEFORE ROI (avoids silent nm/px corruption)
#   - Dual macro/micro bands with preset param (v5 backward compat)
#   - ROI auto-selection by spectral anisotropy
#   - Per-candidate vector validation in _analyze_band_multi
#   - Structured {ok:false} on all failures (never crash)
#   - _cluster_vectors for (period, angle) dedup
#
# Analysis (from Claude, bug-fixed):
#   - Rotation sign FIX: +ang not -ang (proven on synthetic 30 deg stripes)
#   - Tile detrend FIX: sigma scales with max_period, not image width
#   - 2D autocorrelation with zero-padding + half-plane canonicalization
#   - 2D polynomial background subtraction (downsampled for efficiency)
#   - Radial PSD (azimuthal average) for isotropic periodicity
#   - Auto-OCR magnification from SEM info bar (pytesseract optional)
#
# Output (from Claude):
#   - Composite 2×3 summary figure
#   - CSV line profile export
#   - Human-readable summary text for LLM relay
#   - Detail FFT2D plot with all peaks annotated
#
# Both:
#   - Optimal FFT padding (next power of 2)
#   - Hann2D windowing
#   - Multi-peak FFT detection with conjugate pairing
#   - 1D PSD + 1D autocorrelation cross-check
#   - Spectral anisotropy scoring
#   - Particle sizing (unchanged from v5)

import os
import io
import re
import csv
import json
import glob
import uuid
import math
import datetime
import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import requests
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.ndimage import (
    gaussian_filter, gaussian_filter1d, sobel, rotate as nd_rotate,
    maximum_filter, binary_opening, binary_closing, binary_fill_holes, label,
)
from scipy.signal import find_peaks

from mcp.server.fastmcp import FastMCP

# -- Optional OCR ---------------------------------------------------------
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# =========================================================================
#  Logging
# =========================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [micro] %(message)s")
logger = logging.getLogger("micro")

# =========================================================================
#  Config
# =========================================================================
OWUI_URL = os.getenv("OWUI_URL", "").rstrip("/")
JWT_TOKEN = os.getenv("JWT_TOKEN", "")
JWT_SECRET = os.getenv("JWT_SECRET", "")
AUTH_TOKEN = (JWT_TOKEN or JWT_SECRET).strip()

EXPORT_DIR = (os.getenv("FILE_EXPORT_DIR", "/data/files")).rstrip("/")
BASE_URL = (os.getenv("FILE_EXPORT_BASE_URL", "http://localhost:8084/files")).rstrip("/")
os.makedirs(EXPORT_DIR, exist_ok=True)

UPLOAD_ROOT = os.getenv("OWUI_UPLOADS_ROOT", "/openwebui_uploads").rstrip("/")

# Reference SEM frame (JEOL 1280×1024)
REF_W = 1280
REF_H = 1024
REF_BOTTOM_BAR_PX = 64

# Calibration table: magnification -> (scalebar_nm, scalebar_px at 1280w)
SEM_CAL_TABLE: Dict[str, Dict[str, float]] = {
    "x50":     {"scalebar_nm": 500_000.0, "scalebar_px": 250.0},
    "x950":    {"scalebar_nm": 20_000.0,  "scalebar_px": 190.0},
    "x1000":   {"scalebar_nm": 10_000.0,  "scalebar_px": 100.0},
    "x2000":   {"scalebar_nm": 10_000.0,  "scalebar_px": 200.0},
    "x5000":   {"scalebar_nm": 5_000.0,   "scalebar_px": 250.0},
    "x10000":  {"scalebar_nm": 1_000.0,   "scalebar_px": 100.0},
    "x20000":  {"scalebar_nm": 1_000.0,   "scalebar_px": 200.0},
    "x30000":  {"scalebar_nm": 500.0,     "scalebar_px": 150.0},
    "x40000":  {"scalebar_nm": 500.0,     "scalebar_px": 200.0},
}
# Comma-formatted aliases for backward compat
for _k in list(SEM_CAL_TABLE.keys()):
    _num = _k[1:]
    if len(_num) >= 4:
        SEM_CAL_TABLE[f"x{int(_num):,}"] = SEM_CAL_TABLE[_k]

mcp = FastMCP(
    name="micro",
    host=os.getenv("MCP_HTTP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_HTTP_PORT", "9006")),
)


# =========================================================================
#  I/O helpers  --  disk-first file_id (from GPT-5)
# =========================================================================
def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}

def _normalize_path(p: str) -> str:
    if not p:
        return p
    p = p.replace("\\", "/").strip()
    if p.startswith(UPLOAD_ROOT + "/"):
        return p
    if "/uploads/" in p:
        return f"{UPLOAD_ROOT}/{p.split('/uploads/', 1)[1].lstrip('/')}"
    if "/" not in p:
        return f"{UPLOAD_ROOT}/{p}"
    return p

def _safe_read(path: str) -> Tuple[bytes, str]:
    path = _normalize_path(path)
    real = os.path.realpath(os.path.abspath(path))
    root = os.path.realpath(os.path.abspath(UPLOAD_ROOT))
    if not (real == root or real.startswith(root + os.sep)):
        raise RuntimeError(f"Path outside {UPLOAD_ROOT}")
    if not os.path.isfile(real):
        raise RuntimeError(f"Not found: {real}")
    with open(real, "rb") as f:
        data = f.read()
    return data, os.path.basename(real)

def _disk_find_by_file_id(file_id: str) -> Optional[str]:
    """
    Disk-first OpenWebUI upload lookup.
    Supports:
      A) /openwebui_uploads/<file_id>_<original_filename>
      B) /openwebui_uploads/<file_id>/<somefile>
    """
    base_dir = UPLOAD_ROOT
    if not os.path.exists(base_dir):
        return None
    # Flat match: <file_id>_*
    flat = glob.glob(os.path.join(base_dir, f"{file_id}_*"))
    flat = [p for p in flat if os.path.isfile(p)]
    if flat:
        return sorted(flat)[0]
    # Subdirectory match: <file_id>/<file>
    id_dir = os.path.join(base_dir, file_id)
    if os.path.isdir(id_dir):
        files = [f for f in os.listdir(id_dir) if os.path.isfile(os.path.join(id_dir, f))]
        if files:
            return os.path.join(id_dir, sorted(files)[0])
    return None

def _owui_download_bytes(file_id: str) -> Tuple[bytes, str]:
    if not OWUI_URL or not AUTH_TOKEN:
        raise RuntimeError("OWUI_URL and JWT required for file_id HTTP download fallback")
    url = f"{OWUI_URL}/api/v1/files/{file_id}/content"
    r = requests.get(url, headers=_auth_headers(), timeout=180)
    r.raise_for_status()
    fname = file_id
    cd = r.headers.get("content-disposition", "")
    if "filename=" in cd:
        fname = cd.split("filename=")[-1].strip().strip('"')
    return r.content, fname

def read_file_id_bytes(file_id: str) -> Tuple[bytes, str, Dict[str, Any]]:
    """Disk-first, then HTTP fallback."""
    p = _disk_find_by_file_id(str(file_id))
    if p:
        raw, fname = _safe_read(p)
        return raw, fname, {"mode": "disk_first", "path": p}
    raw, fname = _owui_download_bytes(str(file_id))
    return raw, fname, {"mode": "http_fallback", "file_id": file_id}

def _new_folder() -> str:
    folder = f"sem_{uuid.uuid4().hex[:8]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = os.path.join(EXPORT_DIR, folder)
    os.makedirs(out, exist_ok=True)
    return out

def _url(folder: str, fname: str) -> str:
    return f"{BASE_URL}/{os.path.basename(folder)}/{fname}"

def export_bytes(folder: str, data: bytes, fname: str) -> str:
    with open(os.path.join(folder, fname), "wb") as f:
        f.write(data)
    return _url(folder, fname)

def export_fig(folder: str, fig: plt.Figure, fname: str, dpi: int = 200) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return export_bytes(folder, buf.getvalue(), fname)


# =========================================================================
#  Image helpers
# =========================================================================
def _to_gray(img: Image.Image) -> np.ndarray:
    try:
        if getattr(img, "n_frames", 1) > 1:
            img.seek(0)
    except Exception:
        pass
    if img.mode in ("I;16", "I", "F"):
        arr = np.asarray(img, dtype=np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        return arr
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img, dtype=np.float32)

def _hann2d(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

def _next_pow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(n, 8))))

def _crop_bottom(gray: np.ndarray, crop_bottom_px: Optional[int]) -> Tuple[np.ndarray, int]:
    h, _ = gray.shape
    if crop_bottom_px is None:
        crop_bottom_px = int(round(h * (REF_BOTTOM_BAR_PX / float(REF_H))))
    crop_bottom_px = max(0, min(int(crop_bottom_px), h - 10))
    if crop_bottom_px <= 0:
        return gray, 0
    return gray[: h - crop_bottom_px, :], crop_bottom_px

def _apply_roi(gray: np.ndarray, roi: Optional[List[int]]) -> Tuple[np.ndarray, Optional[List[int]]]:
    if not roi:
        return gray, None
    if len(roi) != 4:
        raise ValueError("roi must be [x0,y0,x1,y1]")
    x0, y0, x1, y1 = [int(v) for v in roi]
    h, w = gray.shape
    x0, x1 = max(0, min(w - 1, x0)), max(0, min(w, x1))
    y0, y1 = max(0, min(h - 1, y0)), max(0, min(h, y1))
    if x1 <= x0 + 20 or y1 <= y0 + 20:
        raise ValueError("roi too small")
    return gray[y0:y1, x0:x1], [x0, y0, x1, y1]


# =========================================================================
#  Auto-calibration: OCR + filename inference
# =========================================================================
_MAG_PATTERN = re.compile(r"[x×]\s*(\d{1,3}(?:[,. ]\d{3})*)", re.IGNORECASE)

def _ocr_magnification(gray_full: np.ndarray) -> Optional[str]:
    """Read magnification from SEM info bar via pytesseract. Returns key like 'x40000' or None."""
    if not HAS_TESSERACT:
        return None
    try:
        h, w = gray_full.shape
        bar_h = int(round(h * (REF_BOTTOM_BAR_PX / float(REF_H))))
        bar = gray_full[h - bar_h:, :]
        bar_u8 = np.clip(bar, 0, 255).astype(np.uint8)
        # White text on black: threshold and binarize
        bw = np.where(bar_u8 > 140, 255, 0).astype(np.uint8)
        pil_bar = Image.fromarray(bw).resize(
            (bw.shape[1] * 2, bw.shape[0] * 2), Image.NEAREST)
        text = pytesseract.image_to_string(
            pil_bar,
            config=r"--oem 3 --psm 7 -c tessedit_char_whitelist=x×X0123456789,. ")
        logger.info(f"OCR raw text: {text!r}")
        m = _MAG_PATTERN.search(text.replace("×", "x").replace("X", "x"))
        if not m:
            return None
        num_str = m.group(1).replace(",", "").replace(".", "").replace(" ", "")
        if not num_str.isdigit():
            return None
        key = f"x{num_str}"
        if key in SEM_CAL_TABLE:
            return key
        num = int(num_str)
        for k in SEM_CAL_TABLE:
            if "," in k:
                continue  # skip aliases
            knum = int(k[1:])
            if abs(num - knum) / max(knum, 1) < 0.15:
                logger.info(f"OCR fuzzy matched: {num} -> {k}")
                return k
        return None
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return None

def _infer_mag_from_filename(fname: str) -> Optional[str]:
    """Try to extract magnification like 'x40000' from the filename."""
    s = (fname or "").lower()
    m = re.search(r"(?:^|[^a-z0-9])x\s*([0-9]{2,3}(?:[, _]?[0-9]{3})?)(?:[^a-z0-9]|$)", s)
    if not m:
        m = re.search(r"(?:^|[^a-z0-9])mag\s*([0-9]{2,3}(?:[, _]?[0-9]{3})?)(?:[^a-z0-9]|$)", s)
    if not m:
        return None
    raw = m.group(1).replace("_", ",").replace(" ", ",")
    return _normalize_mag("x" + raw)


# =========================================================================
#  Calibration
# =========================================================================
def _normalize_mag(s: str) -> str:
    s = (s or "").strip().lower().replace(" ", "").replace("×", "x")
    if s and not s.startswith("x"):
        s = "x" + s
    return s.replace(",", "")

def _get_cal(mag_label: Optional[str]) -> Optional[Dict[str, Any]]:
    key = _normalize_mag(mag_label or "")
    if not key:
        return None
    if key in SEM_CAL_TABLE:
        return {"mag_label": key, **SEM_CAL_TABLE[key]}
    # Try with comma
    key2 = key  # already stripped
    if key2 in SEM_CAL_TABLE:
        return {"mag_label": key2, **SEM_CAL_TABLE[key2]}
    return None

def _cal_to_nmpp(cal: Dict[str, Any], w: int) -> Tuple[float, Dict[str, Any]]:
    """Convert calibration table entry to nm/px for the given image width."""
    scale_factor = float(w) / float(REF_W)
    sb_px = float(cal["scalebar_px"]) * scale_factor
    sb_nm = float(cal["scalebar_nm"])
    npp = sb_nm / sb_px
    return npp, {
        "mode": "mag_table_scaled",
        "mag_label": cal["mag_label"],
        "ref_width_px": REF_W,
        "image_width_px": w,
        "scale_factor": scale_factor,
        "scalebar_nm": sb_nm,
        "scalebar_px_ref": float(cal["scalebar_px"]),
        "scalebar_px_scaled": sb_px,
        "nm_per_px": npp,
    }

def _calibrate_nm_per_px(
    w: int,
    mag_label: Optional[str],
    nm_per_px: Optional[float],
    scalebar_px: Optional[float],
    scalebar_nm: Optional[float],
    gray_full: Optional[np.ndarray] = None,
    fname: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Determine nm/px. Priority:
    1. Explicit nm_per_px
    2. Manual scalebar_px + scalebar_nm
    3. mag_label from user/model
    4. Filename inference
    5. Auto-OCR of bottom info bar
    6. Raise with helpful error
    """
    if nm_per_px and float(nm_per_px) > 0:
        return float(nm_per_px), {"mode": "nm_per_px", "nm_per_px": float(nm_per_px)}
    if scalebar_px and scalebar_nm and float(scalebar_px) > 0:
        npp = float(scalebar_nm) / float(scalebar_px)
        return npp, {"mode": "manual_scalebar", "scalebar_nm": float(scalebar_nm),
                     "scalebar_px": float(scalebar_px), "nm_per_px": npp}
    cal = _get_cal(mag_label)
    if cal:
        return _cal_to_nmpp(cal, w)
    # Filename inference
    if fname:
        inf = _infer_mag_from_filename(fname)
        if inf:
            cal = _get_cal(inf)
            if cal:
                npp, info = _cal_to_nmpp(cal, w)
                info["mode"] = "filename_inferred"
                info["inferred_mag"] = inf
                return npp, info
    # Auto-OCR
    if gray_full is not None:
        ocr_key = _ocr_magnification(gray_full)
        if ocr_key:
            cal = _get_cal(ocr_key)
            if cal:
                npp, info = _cal_to_nmpp(cal, w)
                info["mode"] = "auto_ocr"
                info["ocr_detected"] = ocr_key
                return npp, info
    # Fail with helpful message
    available = ", ".join(sorted(
        (k for k in SEM_CAL_TABLE if "," not in k),
        key=lambda k: int(k[1:])))
    raise ValueError(
        f"Calibration required. Provide one of:\n"
        f"   -  mag_label: one of [{available}]\n"
        f"   -  nm_per_px: direct scale (nm per pixel)\n"
        f"   -  scalebar_nm + scalebar_px: manual scale bar measurement\n"
        f"Tip: the model should read the magnification from the SEM image info bar.")


# =========================================================================
#  Preprocessing
# =========================================================================
def _poly_bg_2d(gray: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Fit and subtract 2D polynomial background (illumination gradients).
    Downsampled fit for efficiency: ~30ms instead of ~500ms on 1280×960.
    """
    h, w = gray.shape
    # Downsample by 4× for the fit (6 polynomial terms don't need 1M pixels)
    ds = 4
    small = gray[::ds, ::ds].astype(np.float64)
    sh, sw = small.shape
    yy, xx = np.mgrid[0:sh, 0:sw]
    yn = yy.ravel() / max(sh - 1, 1)
    xn = xx.ravel() / max(sw - 1, 1)
    z = small.ravel()
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((xn ** i) * (yn ** j))
    A = np.column_stack(terms)
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    # Evaluate on full grid
    yf, xf = np.mgrid[0:h, 0:w]
    ynf = yf.ravel().astype(np.float64) / max(h - 1, 1)
    xnf = xf.ravel().astype(np.float64) / max(w - 1, 1)
    terms_full = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms_full.append((xnf ** i) * (ynf ** j))
    bg = np.column_stack(terms_full) @ coeffs
    return (gray.astype(np.float64) - bg.reshape(h, w)).astype(np.float32)

def _sobel_grad(gray: np.ndarray) -> np.ndarray:
    gx = sobel(gray, axis=1, mode="reflect")
    gy = sobel(gray, axis=0, mode="reflect")
    return np.hypot(gx, gy).astype(np.float32)

def _highpass(gray: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0:
        return gray.astype(np.float32)
    blur = gaussian_filter(gray.astype(np.float32), sigma=float(sigma_px))
    return gray.astype(np.float32) - blur


# =========================================================================
#  2D FFT core
# =========================================================================
def _fft_power_2d(signal: np.ndarray, smooth_sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Windowed 2D FFT power spectrum. Returns (raw_P, smoothed_P)."""
    h, w = signal.shape
    work = signal.astype(np.float32) - float(np.mean(signal))
    work *= _hann2d(h, w)
    F = np.fft.fftshift(np.fft.fft2(work))
    P = np.log1p(np.abs(F) ** 2).astype(np.float32)
    Ps = gaussian_filter(P, sigma=float(smooth_sigma)) if smooth_sigma > 0 else P
    return P, Ps

def _freq_grids(h: int, w: int, nm_per_px: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx = np.fft.fftshift(np.fft.fftfreq(w, d=float(nm_per_px)))
    fy = np.fft.fftshift(np.fft.fftfreq(h, d=float(nm_per_px)))
    FX, FY = np.meshgrid(fx, fy)
    FR = np.sqrt(FX * FX + FY * FY)
    return FX, FY, FR

def _top_peaks_in_band(
    power: np.ndarray, FR: np.ndarray,
    min_period_nm: float, max_period_nm: float,
    max_peaks: int = 60, neighborhood: int = 11,
    dc_exclusion_px: int = 3,
) -> List[Dict[str, Any]]:
    h, w = power.shape
    cy, cx = h // 2, w // 2
    rr, cc = np.ogrid[:h, :w]
    r_idx = np.sqrt((rr - cy) ** 2 + (cc - cx) ** 2)
    fmin = 1.0 / float(max_period_nm)
    fmax = 1.0 / float(min_period_nm)
    band = (FR >= fmin) & (FR <= fmax) & (r_idx >= float(dc_exclusion_px))
    if not np.any(band):
        return []
    mx = maximum_filter(power, size=int(neighborhood), mode="nearest")
    local_max = (power == mx) & band
    cand = np.argwhere(local_max)
    if cand.size == 0:
        return []
    vals = power[cand[:, 0], cand[:, 1]]
    order = np.argsort(vals)[::-1][:int(max_peaks)]
    return [{"r": int(cand[i, 0]), "c": int(cand[i, 1]), "power": float(vals[i])} for i in order]

def _pair_strength(peaks: List[Dict], cy: int, cx: int, tol_px: int = 2) -> List[Dict]:
    if not peaks:
        return peaks
    pts = {(p["r"], p["c"]): p for p in peaks}
    out = []
    for p in peaks:
        rr, cc = p["r"], p["c"]
        ro, co = 2 * cy - rr, 2 * cx - cc
        best_pow = 0.0
        for dr in range(-tol_px, tol_px + 1):
            for dc in range(-tol_px, tol_px + 1):
                q = pts.get((ro + dr, co + dc))
                if q and q["power"] > best_pow:
                    best_pow = q["power"]
        out.append({**p, "pair_power": float(best_pow), "pair_score": float(p["power"] + best_pow)})
    out.sort(key=lambda x: x["pair_score"], reverse=True)
    return out

def _angle_deg_from_freq(FX: np.ndarray, FY: np.ndarray, r: int, c: int) -> float:
    ang = np.degrees(np.arctan2(float(FY[r, c]), float(FX[r, c])))
    while ang < -180:
        ang += 360
    while ang >= 180:
        ang -= 360
    return float(ang)


# =========================================================================
#  2D Autocorrelation (zero-padded, from Claude v6)
# =========================================================================
def _autocorr_2d(sig: np.ndarray) -> np.ndarray:
    """Wiener--Khinchin: ACF = IFFT(|FFT|^2), zero-padded to avoid wrap-around."""
    h, w = sig.shape
    ph, pw = _next_pow2(2 * h), _next_pow2(2 * w)
    work = sig.astype(np.float64) - float(np.mean(sig))
    padded = np.zeros((ph, pw), dtype=np.float64)
    padded[:h, :w] = work
    F = np.fft.fft2(padded)
    ac = np.fft.ifft2(F * np.conj(F)).real
    ac = np.fft.fftshift(ac)
    ac0 = float(ac[ph // 2, pw // 2])
    if ac0 > 0:
        ac /= ac0
    return ac.astype(np.float32)

def _pick_autocorr2d_peaks(
    ac2: np.ndarray, nm_per_px: float,
    min_period_nm: float, max_period_nm: float,
    max_peaks: int = 8, neighborhood: int = 21,
) -> List[Dict[str, Any]]:
    """Detect peaks in 2D autocorrelation map. Half-plane canonicalized."""
    h, w = ac2.shape
    cy, cx = h // 2, w // 2
    rr, cc = np.ogrid[:h, :w]
    r_px = np.sqrt((rr - cy).astype(np.float32) ** 2 + (cc - cx).astype(np.float32) ** 2)
    r_nm = r_px * float(nm_per_px)
    band = (r_nm >= float(min_period_nm)) & (r_nm <= float(max_period_nm))
    if not np.any(band):
        return []
    mx = maximum_filter(ac2, size=int(neighborhood), mode="nearest")
    loc = (ac2 == mx) & band & (ac2 > 0.05)
    cand = np.argwhere(loc)
    if cand.size == 0:
        return []
    vals = ac2[cand[:, 0], cand[:, 1]]
    order = np.argsort(vals)[::-1]
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for idx in order:
        r, c = int(cand[idx, 0]), int(cand[idx, 1])
        if (r, c) in seen:
            continue
        ro, co = 2 * cy - r, 2 * cx - c
        seen.add((r, c))
        seen.add((ro, co))
        ddx, ddy = float(c - cx), float(r - cy)
        if abs(ddx) < 2 and abs(ddy) < 2:
            continue
        # Canonicalize to upper half-plane
        if r > cy or (r == cy and c > cx):
            r, c = ro, co
            ddx, ddy = float(c - cx), float(r - cy)
        period_nm = float(np.hypot(ddx, ddy) * nm_per_px)
        ang = float((np.degrees(np.arctan2(ddy, ddx)) + 360.0) % 180.0)
        out.append({"dx_px": ddx, "dy_px": ddy, "period_nm": period_nm,
                     "angle_deg": ang, "value": float(vals[idx])})
        if len(out) >= int(max_peaks):
            break
    return out


# =========================================================================
#  Radial PSD (azimuthal average  --  from Claude v6)
# =========================================================================
def _radial_psd(power: np.ndarray, FR: np.ndarray,
                min_period_nm: float, max_period_nm: float,
                n_bins: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    fmin = 1.0 / max_period_nm
    fmax = 1.0 / min_period_nm
    freq_bins = np.linspace(fmin, fmax, n_bins)
    radial = np.zeros(n_bins - 1, dtype=np.float64)
    counts = np.zeros(n_bins - 1, dtype=np.float64)
    flat_fr = FR.ravel()
    flat_p = power.ravel()
    bin_idx = np.digitize(flat_fr, freq_bins) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins - 1)
    np.add.at(radial, bin_idx[valid], flat_p[valid])
    np.add.at(counts, bin_idx[valid], 1.0)
    mask = counts > 0
    radial[mask] /= counts[mask]
    freq_centers = 0.5 * (freq_bins[:-1] + freq_bins[1:])
    period_nm = np.where(freq_centers > 0, 1.0 / freq_centers, 0.0)
    return period_nm, radial


# =========================================================================
#  Spectral anisotropy (from GPT-5)
# =========================================================================
def _spectral_anisotropy(Ps: np.ndarray, FX: np.ndarray, FY: np.ndarray, FR: np.ndarray,
                         min_period_nm: float, max_period_nm: float,
                         dc_exclusion_px: int = 3, angle_bins: int = 180) -> Dict[str, Any]:
    h, w = Ps.shape
    cy, cx = h // 2, w // 2
    rr, cc = np.ogrid[:h, :w]
    r_idx = np.sqrt((rr - cy) ** 2 + (cc - cx) ** 2)
    fmin = 1.0 / float(max_period_nm)
    fmax = 1.0 / float(min_period_nm)
    band = (FR >= fmin) & (FR <= fmax) & (r_idx >= float(dc_exclusion_px))
    if not np.any(band):
        return {"ok": False, "reason": "empty_band"}
    ang = (np.degrees(np.arctan2(FY, FX)) + 180.0) % 180.0
    weights = Ps * band.astype(np.float32)
    hist = np.zeros(int(angle_bins), dtype=np.float64)
    idx = np.clip((ang * (angle_bins / 180.0)).astype(np.int32), 0, angle_bins - 1)
    np.add.at(hist, idx.ravel(), weights.ravel())
    hist_s = gaussian_filter1d(hist.astype(np.float32), sigma=2.0).astype(np.float64)
    med = float(np.median(hist_s))
    mx = float(np.max(hist_s))
    anis = float((mx + 1e-9) / (med + 1e-9))
    pk, props = find_peaks(hist_s, prominence=0.05 * (mx - med + 1e-9),
                           distance=max(3, int(angle_bins / 36)))
    peaks = [{"k_angle_deg": float(j * (180.0 / angle_bins)), "value": float(hist_s[j])} for j in pk]
    peaks.sort(key=lambda d: d["value"], reverse=True)
    return {"ok": True, "anisotropy": anis, "angle_peaks": peaks[:6]}


# =========================================================================
#  1D analysis helpers
# =========================================================================
def _rotate_keep_center(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape
    rot = nd_rotate(img, angle=float(angle_deg), reshape=True, order=1, mode="reflect")
    rh, rw = rot.shape
    y0 = max(0, (rh - h) // 2)
    x0 = max(0, (rw - w) // 2)
    return rot[y0:y0 + h, x0:x0 + w].astype(np.float32)

def _normalize_rotation_angle(k_angle_deg: float) -> float:
    """
    Normalize k-vector angle to [-90, 90] range for minimal rotation.
    CRITICAL FIX: proven correct on synthetic 30 deg stripes where raw
    k_angle=-150.9 deg must be normalized to +29.1 deg to get correct 1D profile.
    """
    ang = float(k_angle_deg) % 360
    if ang > 180:
        ang -= 360
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    return ang

def _rfft_psd_1d(x: np.ndarray, nm_per_px: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    win = np.hanning(n).astype(np.float32)
    xs = (x - float(np.mean(x))) * win
    F = np.fft.rfft(xs)
    P = (np.abs(F) ** 2).astype(np.float64)
    f = np.fft.rfftfreq(n, d=float(nm_per_px))
    return f, P

def _robust_z(x: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-12
    return med, 1.4826 * mad

def _pick_psd_peaks(
    f: np.ndarray, P: np.ndarray,
    min_period_nm: float, max_period_nm: float,
    top_n: int = 5, peak_rel_height: float = 0.25,
    peak_min_distance_bins: int = 8,
) -> List[Dict[str, Any]]:
    fmin = 1.0 / float(max_period_nm)
    fmax = 1.0 / float(min_period_nm)
    band = (f >= fmin) & (f <= fmax) & (f > 0)
    if not np.any(band):
        return []
    fb, Pb = f[band], P[band]
    base = float(np.percentile(Pb, 20))
    work = Pb - base
    p50, p90 = float(np.percentile(work, 50)), float(np.percentile(work, 90))
    height = p50 + float(peak_rel_height) * (p90 - p50)
    prominence = 0.10 * (p90 - p50)
    peaks, props = find_peaks(work, height=max(height, 1e-6),
                              prominence=max(prominence, 1e-6),
                              distance=int(peak_min_distance_bins))
    if peaks.size == 0:
        idx = int(np.argmax(Pb))
        peaks = np.array([idx], dtype=int)
        props = {"prominences": np.array([float(Pb[idx] - base)])}
    order = np.argsort(props["prominences"])[::-1]
    out = []
    med, sigma = _robust_z(Pb)
    for j in order[:int(top_n)]:
        k = int(peaks[j])
        freq = float(fb[k])
        power = float(Pb[k])
        out.append({
            "period_nm": float(1.0 / (freq + 1e-18)),
            "frequency_cycles_per_nm": freq,
            "power": power,
            "zscore": round(float((power - med) / (sigma + 1e-12)), 2),
            "prominence": float(props["prominences"][j]) if "prominences" in props else None,
            "index_in_band": int(k),
        })
    out.sort(key=lambda d: d["power"], reverse=True)
    return out

def _autocorr_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64) - float(np.mean(x))
    n = len(x)
    F = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(F * np.conj(F))[:n]
    ac0 = float(ac[0]) if ac[0] != 0 else 1.0
    return (ac / ac0).astype(np.float32)

def _pick_autocorr_peak_1d(ac: np.ndarray, nm_per_px: float,
                           min_period_nm: float, max_period_nm: float) -> Dict[str, Any]:
    n = len(ac)
    lags = np.arange(n, dtype=np.float32) * float(nm_per_px)
    lo = int(max(1, math.floor(min_period_nm / nm_per_px)))
    hi = int(min(n - 1, math.ceil(max_period_nm / nm_per_px)))
    if hi <= lo + 5:
        return {"ok": False, "reason": "lag_range_too_small"}
    y = ac[lo:hi]
    ys = gaussian_filter1d(y, sigma=1.2) if y.size > 15 else y
    peaks, props = find_peaks(ys, prominence=0.05, distance=max(3, int(0.02 * y.size)))
    if peaks.size == 0:
        return {"ok": False, "reason": "no_peak"}
    j = int(np.argmax(props["prominences"]))
    pk = int(peaks[j])
    lag_idx = int(lo + pk)
    return {"ok": True, "period_nm": float(lags[lag_idx]),
            "meta": {"lag_index": lag_idx, "prominence": float(props["prominences"][j]),
                     "lag_range_nm": [float(lags[lo]), float(lags[hi - 1])]}}

def _harmonic_consensus(psd_period: float, ac_period: float, tol_pct: float = 0.12) -> Tuple[float, str]:
    if not (psd_period and ac_period):
        return psd_period or ac_period, "single_method"
    ratio = max(psd_period, ac_period) / max(1e-9, min(psd_period, ac_period))
    for n in (2, 3, 4):
        if abs(ratio - n) / n < tol_pct:
            return max(psd_period, ac_period), f"harmonic_{n}x_resolved"
    return 0.5 * (psd_period + ac_period), "avg"

def _agreement_percent(a: float, b: float) -> float:
    if not a or not b:
        return 0.0
    return float(100.0 * (1.0 - abs(a - b) / max(a, b, 1e-12)))

def _confidence(snr_z: float, agree_1d: float, agree_2d: float,
                tile_cv: Optional[float], cycles: float, anisotropy: float) -> str:
    if cycles < 2.0:
        return "low"
    cv = tile_cv if tile_cv is not None else 0.5
    # Penalize marginal cycle counts: require stronger evidence
    cyc_penalty = cycles < 3.0  # True if 2-3 cycles (macro at high mag)
    # High: strong SNR + 1D agreement + tile consistency + directional signal
    # (2D autocorr is a bonus, not required  --  grid discretization can suppress it)
    if (snr_z >= 6 and agree_1d >= 80 and cv <= 0.20 and anisotropy >= 1.8
            and not cyc_penalty):
        return "high"
    if snr_z >= 6 and agree_1d >= 80 and cv <= 0.25 and anisotropy >= 1.8:
        # Strong evidence even with few cycles -> medium (not high)
        return "medium" if cyc_penalty else "high"
    if snr_z >= 4 and agree_1d >= 65 and cv <= 0.30 and anisotropy >= 1.3:
        return "medium"
    # 2D autocorr agreement alone can boost low -> medium
    if snr_z >= 3 and agree_2d >= 60:
        return "medium"
    return "low"

def _tile_periods(rot_signal: np.ndarray, nm_per_px: float,
                  min_period_nm: float, max_period_nm: float,
                  tiles: int, peak_rel_height: float,
                  peak_min_distance_bins: int) -> List[float]:
    """FIXED: detrend sigma scales with max_period, not image width."""
    h, _ = rot_signal.shape
    tiles = max(3, min(12, int(tiles)))
    ys = np.linspace(0, h, tiles + 1).astype(int)
    periods = []
    for i in range(tiles):
        y0, y1 = int(ys[i]), int(ys[i + 1])
        strip = rot_signal[y0:y1, :]
        if strip.size < 1000:
            continue
        prof = np.mean(strip, axis=0)
        # FIX: detrend sigma must be >> period to avoid destroying the signal
        detr_sigma = max(50.0, 0.8 * max_period_nm / nm_per_px)
        prof = prof - gaussian_filter1d(prof, sigma=float(detr_sigma))
        f, P = _rfft_psd_1d(prof, nm_per_px)
        cand = _pick_psd_peaks(f, P, min_period_nm, max_period_nm, top_n=1,
                               peak_rel_height=peak_rel_height,
                               peak_min_distance_bins=peak_min_distance_bins)
        if cand:
            periods.append(float(cand[0]["period_nm"]))
    return periods


# =========================================================================
#  Per-candidate multi-vector analysis (GPT-5 architecture, Claude fixes)
# =========================================================================
def _cluster_vectors(vs: List[Dict[str, Any]], angle_tol_deg: float = 6.0,
                     period_tol_pct: float = 0.12) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for v in vs:
        placed = False
        for u in out:
            da = abs(((v["k_angle_deg"] - u["k_angle_deg"] + 90) % 180) - 90)
            dp = abs(v["period_from_2d_fft_nm"] - u["period_from_2d_fft_nm"]) / max(u["period_from_2d_fft_nm"], 1e-9)
            if da <= angle_tol_deg and dp <= period_tol_pct:
                if v["score"] > u["score"]:
                    u.update(v)
                placed = True
                break
        if not placed:
            out.append(v)
    out.sort(key=lambda d: d["score"], reverse=True)
    return out

def _analyze_band_multi(
    gray: np.ndarray, nm_per_px: float,
    min_period_nm: float, max_period_nm: float,
    signal_mode: str, smooth_sigma: float,
    dc_exclusion_px: int, tile_rows: int,
    peak_rel_height: float, peak_min_distance_bins: int,
    top_n: int, harmonic_tol_pct: float,
    max_vectors: int, preprocess: str,
    label_name: str, bg_subtract: bool = True, bg_degree: int = 2,
) -> Dict[str, Any]:
    """
    Multi-vector band analysis. For each FFT peak candidate:
    rotate, extract 1D profile, PSD, autocorr, tile consistency, 2D autocorr check.
    Returns top periodicity vectors with confidence scores.
    """
    gray0 = gray.astype(np.float32)

    # Normalize intensity channel
    g_med = float(np.median(gray0))
    g_iqr = float(np.percentile(gray0, 75) - np.percentile(gray0, 25)) + 1e-6
    intensity = (gray0 - g_med) / g_iqr

    # Background subtraction on intensity
    if bg_subtract:
        intensity = _poly_bg_2d(intensity, degree=bg_degree)

    grad = _sobel_grad(gaussian_filter(gray0, sigma=1.0))
    grad = grad / (float(np.median(grad)) + 1e-6)

    mode = (signal_mode or "auto").strip().lower()
    candidates = []
    if mode in ("auto", "intensity"):
        candidates.append(("intensity", intensity))
    if mode in ("auto", "gradient"):
        candidates.append(("gradient", grad))
    if not candidates:
        candidates = [("intensity", intensity)]

    best_pack: Optional[Dict[str, Any]] = None
    best_score = -1e18

    for name, sig0 in candidates:
        sig = sig0.copy()

        # Optional highpass preprocessing
        prep = (preprocess or "auto").strip().lower()
        if prep in ("auto", "highpass", "dog"):
            max_p_px = max_period_nm / nm_per_px
            hp_sigma = max(8.0, 0.75 * max_p_px)
            sig = _highpass(sig, sigma_px=float(hp_sigma))
            sig = sig / (float(np.std(sig)) + 1e-6)

        _, Ps = _fft_power_2d(sig, smooth_sigma=smooth_sigma)
        h, w = sig.shape
        cy, cx = h // 2, w // 2
        FX, FY, FR = _freq_grids(h, w, nm_per_px)

        dirn = _spectral_anisotropy(Ps, FX, FY, FR, min_period_nm, max_period_nm,
                                    dc_exclusion_px=dc_exclusion_px)
        anis = float(dirn.get("anisotropy", 1.0)) if dirn.get("ok") else 1.0

        peaks = _top_peaks_in_band(Ps, FR, min_period_nm, max_period_nm,
                                   max_peaks=70, neighborhood=11,
                                   dc_exclusion_px=dc_exclusion_px)
        peaks = _pair_strength(peaks, cy, cx, tol_px=2)
        if not peaks:
            continue

        # Compute 2D autocorr once for cross-check
        ac2 = _autocorr_2d(sig)
        ac2_peaks = _pick_autocorr2d_peaks(ac2, nm_per_px, min_period_nm, max_period_nm,
                                           max_peaks=10)

        vecs: List[Dict[str, Any]] = []
        for p in peaks[:min(40, len(peaks))]:
            r, c = int(p["r"]), int(p["c"])
            # Keep half-plane to avoid duplicates
            if r > cy or (r == cy and c < cx):
                continue

            ang_k = _angle_deg_from_freq(FX, FY, r, c)
            freq_mag = float(FR[r, c])
            if freq_mag <= 0:
                continue
            period_2d = float(1.0 / freq_mag)

            # -- ROTATION FIX: normalize angle then use +ang --
            rot_ang = _normalize_rotation_angle(ang_k)
            rot = _rotate_keep_center(sig, angle_deg=rot_ang)

            # 1D profile: mean along rows (axis=0) after rotation
            prof = np.mean(rot, axis=0)
            # FIX: detrend sigma scales with max_period, not image width
            detr_sigma = max(50.0, 0.8 * max_period_nm / nm_per_px)
            prof = prof - gaussian_filter1d(prof, sigma=float(detr_sigma))

            f, P1 = _rfft_psd_1d(prof, nm_per_px)
            psd_cand = _pick_psd_peaks(f, P1, min_period_nm, max_period_nm,
                                       top_n=top_n, peak_rel_height=peak_rel_height,
                                       peak_min_distance_bins=peak_min_distance_bins)
            if not psd_cand:
                continue

            psd_best = psd_cand[0]
            ac1 = _autocorr_1d(prof)
            ac_pk = _pick_autocorr_peak_1d(ac1, nm_per_px, min_period_nm, max_period_nm)

            ac_period = float(ac_pk["period_nm"]) if ac_pk.get("ok") else 0.0
            chosen_period, method_note = _harmonic_consensus(
                float(psd_best["period_nm"]), ac_period, tol_pct=harmonic_tol_pct)
            agree_1d = _agreement_percent(float(psd_best["period_nm"]), ac_period) if ac_period else 0.0

            # 2D autocorr agreement
            agree_2d = 0.0
            if ac2_peaks:
                stripe_ang = float((ang_k + 90.0) % 180.0)
                best = None
                for q in ac2_peaks:
                    da = abs(((q["angle_deg"] - stripe_ang + 90) % 180) - 90)
                    dp = abs(q["period_nm"] - chosen_period) / max(chosen_period, 1e-9)
                    sc = (1.0 - min(1.0, da / 20.0)) * (1.0 - min(1.0, dp / 0.25))
                    if best is None or sc > best[0]:
                        best = (sc, q)
                if best:
                    agree_2d = float(100.0 * best[0])

            tile_periods = _tile_periods(rot, nm_per_px, min_period_nm, max_period_nm,
                                         tiles=tile_rows, peak_rel_height=peak_rel_height,
                                         peak_min_distance_bins=peak_min_distance_bins)
            tile_cv = None
            if len(tile_periods) >= 3:
                m_t = float(np.mean(tile_periods))
                tile_cv = float(np.std(tile_periods) / m_t) if m_t > 0 else None

            cycles = float((rot.shape[1] * nm_per_px) / max(1e-9, chosen_period))
            conf = _confidence(float(psd_best.get("zscore", 0.0)), float(agree_1d),
                               float(agree_2d), tile_cv, cycles, anis)

            score = (float(psd_best.get("zscore", 0.0)) + 0.012 * agree_1d +
                     0.010 * agree_2d + 0.35 * math.log(max(anis, 1e-9)) -
                     (tile_cv or 0.5))

            vecs.append({
                "score": float(score),
                "selected_signal": name,
                "k_angle_deg": float(ang_k),
                "stripe_angle_deg": float((ang_k + 90.0) % 180.0),
                "period_from_2d_fft_nm": float(period_2d),
                "chosen": {
                    "period_nm": float(chosen_period),
                    "method": method_note,
                    "agreement_1d_percent": float(agree_1d),
                    "agreement_2d_percent": float(agree_2d),
                    "snr_z": float(psd_best.get("zscore", 0.0)),
                    "tile_cv": tile_cv,
                    "cycles_across_width": cycles,
                    "confidence": conf,
                },
                "best_psd": psd_best,
                "psd_candidates": psd_cand[:min(5, len(psd_cand))],
                "autocorr_1d": ac_pk,
                "peak_2d_rc": [int(r), int(c)],
            })

        if not vecs:
            continue

        vecs = _cluster_vectors(vecs, angle_tol_deg=6.0, period_tol_pct=0.12)
        vecs = vecs[:int(max_vectors)]

        primary = vecs[0]
        pack = {
            "ok": True, "label": label_name, "anisotropy": anis,
            "selected_signal": primary["selected_signal"],
            "primary": primary, "periodicities": vecs,
            "artifacts_data": {
                "fft_power": Ps, "autocorr2d": ac2,
                "freq": f, "psd": P1, "signal": sig,
            }
        }
        if float(primary["score"]) > best_score:
            best_score = float(primary["score"])
            best_pack = pack

    if best_pack is None:
        return {"ok": False, "reason": "no_valid_peaks_in_band", "label": label_name}
    return best_pack


# =========================================================================
#  ROI auto-selection (from GPT-5)
# =========================================================================
def _roi_candidates_grid(h: int, w: int, grid: int, margin: int) -> List[List[int]]:
    grid = max(2, min(6, int(grid)))
    roi_w = max(256, min(w - 2 * margin, int(round(w * 0.62))))
    roi_h = max(256, min(h - 2 * margin, int(round(h * 0.62))))
    xs = np.linspace(margin, w - margin - roi_w, grid).astype(int)
    ys = np.linspace(margin, h - margin - roi_h, grid).astype(int)
    rois: List[List[int]] = []
    for y0 in ys:
        for x0 in xs:
            rois.append([int(x0), int(y0), int(x0 + roi_w), int(y0 + roi_h)])
    cx0, cy0 = (w - roi_w) // 2, (h - roi_h) // 2
    rois.insert(0, [int(cx0), int(cy0), int(cx0 + roi_w), int(cy0 + roi_h)])
    return rois

def _pick_best_roi(gray: np.ndarray, nm_per_px: float,
                   min_period_nm: float, max_period_nm: float,
                   grid: int = 3) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
    h, w = gray.shape
    rois = _roi_candidates_grid(h, w, grid=grid, margin=20)
    best = None
    for roi in rois:
        patch, _ = _apply_roi(gray, roi)
        sig = _sobel_grad(gaussian_filter(patch, sigma=1.0))
        _, Ps = _fft_power_2d(sig, smooth_sigma=1.0)
        FX, FY, FR = _freq_grids(sig.shape[0], sig.shape[1], nm_per_px)
        dirn = _spectral_anisotropy(Ps, FX, FY, FR, min_period_nm, max_period_nm,
                                    dc_exclusion_px=4)
        if not dirn.get("ok"):
            continue
        score = float(dirn.get("anisotropy", 1.0))
        if best is None or score > best[0]:
            best = (score, roi, dirn)
    if best is None:
        return gray, [0, 0, w, h], {"ok": False, "reason": "no_roi_scored"}
    _, roi, meta = best
    patch, _ = _apply_roi(gray, roi)
    return patch, roi, {"ok": True, "roi": roi, "score": float(best[0])}


# =========================================================================
#  Visualization (composite figure  --  from Claude v6)
# =========================================================================
def _plot_preview(gray: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(gray, cmap="gray", origin="upper")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig

def _plot_composite(
    gray_roi: np.ndarray, fft_power: np.ndarray, peaks_rc: List[List[int]],
    labels: List[str], ac2d: np.ndarray, ac2_peaks: List[Dict],
    radial_period: np.ndarray, radial_psd_arr: np.ndarray,
    line_profile: np.ndarray, nm_per_px: float,
    summary_lines: List[str], title: str = "SEM Periodicity Analysis",
    detected_periods_nm: Optional[List[float]] = None,
) -> plt.Figure:
    """Composite 2\u00d73 summary figure."""
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.30)

    # Panel 1: SEM preview with scale bar
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(gray_roi, cmap="gray", origin="upper")
    fov_nm = gray_roi.shape[1] * nm_per_px
    bar_nm = 10 ** int(np.log10(max(fov_nm / 4, 1)))
    bar_px = bar_nm / nm_per_px
    y_bar, x_bar = gray_roi.shape[0] - 15, 15
    ax1.plot([x_bar, x_bar + bar_px], [y_bar, y_bar], "w-", linewidth=3)
    ax1.plot([x_bar, x_bar + bar_px], [y_bar, y_bar], "k-", linewidth=1)
    unit = "nm" if bar_nm < 1000 else "\u03bcm"
    val = bar_nm if bar_nm < 1000 else bar_nm / 1000
    ax1.text(x_bar + bar_px / 2, y_bar - 8, f"{val:.0f} {unit}",
             color="white", fontsize=8, ha="center",
             bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6))
    ax1.set_title("SEM (analysis region)", fontsize=10, fontweight="bold")
    ax1.axis("off")

    # Panel 2: 2D FFT with peaks annotated (FIX 2: anti-overlap labels)
    ax2 = fig.add_subplot(gs[0, 1])
    p = fft_power.astype(np.float32)
    p = (p - np.min(p)) / (np.max(p) - np.min(p) + 1e-9)
    ax2.imshow(p, cmap="inferno", origin="lower", aspect="auto")
    colors_list = ["cyan", "lime", "magenta", "yellow", "white"]
    # Collect annotations then resolve overlap
    _ann_items = []
    for i, (rc, lbl) in enumerate(zip(peaks_rc[:5], labels[:5])):
        clr = colors_list[i % len(colors_list)]
        r_pk, c_pk = rc
        ax2.scatter([c_pk], [r_pk], s=120, marker="o",
                    facecolors="none", edgecolors=clr, linewidths=2)
        h_fft, w_fft = fft_power.shape
        ax2.scatter([w_fft - c_pk], [h_fft - r_pk], s=80, marker="o",
                    facecolors="none", edgecolors=clr, linewidths=1, alpha=0.5)
        _ann_items.append((c_pk, r_pk, lbl, clr))
    # Sort by row then stagger when labels are too close
    _ann_items.sort(key=lambda t: t[1])
    _label_h = 14  # approx pixel height of label at fontsize 7
    _last_y = -999
    _stagger = 0
    for c_pk, r_pk, lbl, clr in _ann_items:
        if abs(r_pk - _last_y) < _label_h * 1.5:
            _stagger += 1
        else:
            _stagger = 0
        _last_y = r_pk
        y_off = 8 + _stagger * (_label_h + 2)
        _arrow = dict(arrowstyle="-", color=clr, alpha=0.4, lw=0.6) if _stagger > 0 else None
        ax2.annotate(lbl, (c_pk, r_pk), color=clr, fontsize=7,
                     xytext=(8, y_off), textcoords="offset points",
                     arrowprops=_arrow,
                     bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
    ax2.set_title("2D FFT Power", fontsize=10, fontweight="bold")
    ax2.set_xlabel("kx", fontsize=8)
    ax2.set_ylabel("ky", fontsize=8)

    # Panel 3: 2D Autocorrelation (FIX 3: adaptive colormap range)
    ax3 = fig.add_subplot(gs[0, 2])
    ah, aw = ac2d.shape
    max_show = min(ah // 2, aw // 2)
    cy_ac, cx_ac = ah // 2, aw // 2
    crop_ac = ac2d[cy_ac - max_show:cy_ac + max_show, cx_ac - max_show:cx_ac + max_show]
    ext_nm = max_show * nm_per_px
    # Adaptive vmin/vmax via percentiles; fallback if range too narrow
    _ac_vmin = float(np.percentile(crop_ac, 2))
    _ac_vmax = float(np.percentile(crop_ac, 98))
    if _ac_vmax - _ac_vmin < 0.05:
        _ac_vmin, _ac_vmax = -0.15, 0.4
    ax3.imshow(crop_ac, cmap="RdBu_r", origin="lower",
               extent=[-ext_nm, ext_nm, -ext_nm, ext_nm],
               vmin=_ac_vmin, vmax=_ac_vmax)
    for i, pk in enumerate(ac2_peaks[:5]):
        clr = colors_list[i % len(colors_list)]
        dx = pk.get("dx_px", 0) * nm_per_px
        dy = pk.get("dy_px", 0) * nm_per_px
        if abs(dx) <= ext_nm and abs(dy) <= ext_nm:
            ax3.scatter([dx], [dy], s=100, marker="+", color=clr, linewidths=2)
            pnm = pk["period_nm"]
            ax3.annotate(f"{pnm:.0f}nm", (dx, dy), color=clr, fontsize=7,
                         xytext=(6, 6), textcoords="offset points",
                         bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
    ax3.set_title("2D Autocorrelation", fontsize=10, fontweight="bold")
    ax3.set_xlabel("\u0394x (nm)", fontsize=8)
    ax3.set_ylabel("\u0394y (nm)", fontsize=8)

    # Panel 4: Radial PSD (FIX 1: detected period markers)
    ax4 = fig.add_subplot(gs[1, 0])
    valid_r = radial_psd_arr > 0
    if np.any(valid_r):
        ax4.semilogx(radial_period[valid_r], radial_psd_arr[valid_r], "b-", linewidth=1)
    # Red dashed vertical lines at each detected period
    if detected_periods_nm:
        _psd_ymin, _psd_ymax = ax4.get_ylim()
        for _dp in detected_periods_nm:
            if _dp > 0:
                ax4.axvline(_dp, color="red", ls="--", alpha=0.7, linewidth=1.0)
                ax4.text(_dp, _psd_ymax * 0.92, f"{_dp:.0f}",
                         color="red", fontsize=6.5, ha="center", va="top",
                         bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                   ec="red", alpha=0.7, lw=0.5))
    ax4.set_xlabel("Period (nm, log)", fontsize=8)
    ax4.set_ylabel("Power (azimuthal avg)", fontsize=8)
    ax4.set_title("Radial PSD", fontsize=10, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    # Panel 5: 1D Line profile
    ax5 = fig.add_subplot(gs[1, 1])
    x_nm = np.arange(len(line_profile)) * nm_per_px
    ax5.plot(x_nm, line_profile, "k-", linewidth=0.8)
    ax5.set_xlabel("Position (nm)", fontsize=8)
    ax5.set_ylabel("Intensity (detrended)", fontsize=8)
    ax5.set_title("Line profile \u22a5 stripes", fontsize=10, fontweight="bold")
    ax5.grid(True, alpha=0.3)

    # Panel 6: Summary text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    ax6.text(0.05, 0.95, "\n".join(summary_lines), transform=ax6.transAxes,
             fontsize=8, fontfamily="monospace", verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8))

    return fig

def _plot_fft2d_multi(power: np.ndarray, peaks_rc: List[List[int]],
                      labels: List[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    p = power.astype(np.float32)
    p = (p - np.min(p)) / (np.max(p) - np.min(p) + 1e-9)
    ax.imshow(p, cmap="inferno", origin="lower")
    colors_list = ["cyan", "lime", "magenta", "yellow", "white"]
    for i, (rc, lbl) in enumerate(zip(peaks_rc[:10], labels[:10])):
        clr = colors_list[i % len(colors_list)]
        ax.scatter([rc[1]], [rc[0]], s=110, marker="o",
                   facecolors="none", edgecolors=clr, linewidths=2)
        ax.text(rc[1] + 8, rc[0] + 8, lbl, color="white", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("kx index")
    ax.set_ylabel("ky index")
    fig.tight_layout()
    return fig

def _plot_autocorr2d(ac2: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(ac2, cmap="magma", origin="lower")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig

def _plot_psd(f: np.ndarray, P: np.ndarray, min_p: float, max_p: float,
              chosen_p: float, title: str) -> plt.Figure:
    mask = f > 0
    period = np.zeros_like(f)
    period[mask] = 1.0 / f[mask]
    band = mask & (period >= min_p) & (period <= max_p)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(period[band], P[band])
    ax.set_xlabel("Period (nm, log)")
    ax.set_ylabel("Power")
    ax.set_title(title)
    if chosen_p and chosen_p > 0:
        ax.axvline(chosen_p, linestyle="--", label=f"best ~ {chosen_p:.1f} nm")
        ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# =========================================================================
#  Particle sizing (unchanged from v5)
# =========================================================================
def _otsu_threshold(img: np.ndarray) -> float:
    x = np.clip(img.ravel(), 0.0, 1.0)
    hist, bin_edges = np.histogram(x, bins=256, range=(0.0, 1.0))
    prob = hist.astype(np.float64) / (hist.sum() + 1e-12)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.linspace(0, 1, 256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    idx = int(np.argmax(sigma_b2))
    return float((bin_edges[idx] + bin_edges[idx + 1]) * 0.5)

def _particle_analysis(gray: np.ndarray, nm_per_px: float, mode: str,
                       min_diam_nm: float, max_diam_nm: float,
                       thresh: Optional[float], smooth_sigma: float) -> Dict[str, Any]:
    g = gray.astype(np.float32)
    g = gaussian_filter(g, sigma=float(smooth_sigma)) if smooth_sigma > 0 else g
    g0 = g - float(np.min(g))
    g01 = (g0 / (float(np.max(g0)) + 1e-9)).astype(np.float32)
    t = float(thresh) if thresh is not None else _otsu_threshold(g01)
    bright = (mode or "bright").strip().lower() != "dark"
    mask = (g01 >= t) if bright else (g01 <= t)
    mask = binary_opening(mask, iterations=1)
    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)
    lab, n = label(mask)
    if n <= 0:
        return {"ok": False, "reason": "no_components"}
    min_d_px = float(min_diam_nm) / float(nm_per_px)
    max_d_px = float(max_diam_nm) / float(nm_per_px)
    min_area = math.pi * (min_d_px / 2.0) ** 2
    max_area = math.pi * (max_d_px / 2.0) ** 2
    parts = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if ys.size < 20:
            continue
        area_px = float(ys.size)
        if area_px < min_area or area_px > max_area:
            continue
        cy_p, cx_p = float(np.mean(ys)), float(np.mean(xs))
        x = xs.astype(np.float64) - cx_p
        y = ys.astype(np.float64) - cy_p
        cov = np.cov(np.vstack([x, y]))
        eigvals = np.maximum(np.linalg.eigvalsh(cov), 1e-12)
        major_px = 4.0 * math.sqrt(float(np.max(eigvals)))
        minor_px = 4.0 * math.sqrt(float(np.min(eigvals)))
        eq_d_px = 2.0 * math.sqrt(area_px / math.pi)
        parts.append({
            "id": int(i),
            "equiv_diameter_nm": float(eq_d_px * nm_per_px),
            "major_axis_nm": float(major_px * nm_per_px),
            "minor_axis_nm": float(minor_px * nm_per_px),
            "aspect_ratio": float(major_px / max(1e-9, minor_px)),
            "centroid_px": [float(cx_p), float(cy_p)],
        })
    if not parts:
        return {"ok": False, "reason": "no_components_after_size_filter"}
    diams = np.array([p["equiv_diameter_nm"] for p in parts], dtype=np.float64)
    return {
        "ok": True, "count": int(len(parts)),
        "equiv_diameter_nm": {
            "mean": float(np.mean(diams)), "median": float(np.median(diams)),
            "std": float(np.std(diams)),
            "p10": float(np.percentile(diams, 10)), "p90": float(np.percentile(diams, 90)),
        },
        "threshold": {"mode": "otsu" if thresh is None else "manual",
                      "value": float(t), "polarity": "bright" if bright else "dark"},
        "particles": parts[:2000],
    }


# =========================================================================
#  Summary text (from Claude v6)
# =========================================================================
def _generate_summary_text(band_results: Dict[str, Dict], cal_info: Dict,
                           fov_nm: List[float], pa_result: Optional[Dict]) -> str:
    lines = []
    mag = cal_info.get("mag_label", cal_info.get("ocr_detected", "unknown"))
    npp = cal_info.get("nm_per_px", 0)
    lines.append(f"SEM image analyzed at {mag} magnification ({npp:.2f} nm/px).")
    lines.append(f"Field of view: {fov_nm[0]:.0f} x {fov_nm[1]:.0f} nm.")
    for band_name, band in sorted(band_results.items()):
        if not band.get("ok", False):
            lines.append(f"\n{band_name.upper()} band: no periodicities detected.")
            continue
        pers = band.get("periodicities", [])
        lines.append(f"\n{band_name.upper()} band ({len(pers)} vector(s)):")
        for i, v in enumerate(pers[:5]):
            ch = v.get("chosen", {})
            p_nm = ch.get("period_nm", 0)
            conf = ch.get("confidence", "?")
            s_ang = v.get("stripe_angle_deg", None)
            ang_str = f" at {s_ang:.0f} deg" if s_ang is not None else ""
            lines.append(f"  {i+1}. {p_nm:.1f} nm{ang_str} [{conf}]")
    if pa_result and pa_result.get("ok"):
        d = pa_result["equiv_diameter_nm"]
        lines.append(
            f"\nParticle analysis: {pa_result['count']} particles, "
            f"mean D {d['mean']:.1f} +/- {d['std']:.1f} nm ({d['p10']:.0f}--{d['p90']:.0f} nm).")
    return "\n".join(lines)


# =========================================================================
#  Slim response builder (model-facing output)
# =========================================================================
def _slim_response(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a compact, model-friendly response from the full analysis report.
    The full report is already saved as sem_report.json (linked in artifacts).
    This returns only what the LLM needs to present findings to the user.
    """
    slim: Dict[str, Any] = {
        "ok": report.get("ok", False),
    }

    if not slim["ok"]:
        slim["reason"] = report.get("reason", "unknown error")
        return slim

    # -- Scale info (one line) --
    scale = report.get("scale", {})
    slim["scale"] = {
        "nm_per_px": scale.get("nm_per_px"),
        "mag_label": scale.get("mag_label"),
        "field_of_view_nm": scale.get("field_of_view_nm"),
    }

    # -- Periodicities: clean table, no diagnostics --
    slim["periodicities"] = []
    for band_name in ("macro", "micro"):
        band = report.get("results", {}).get(band_name, {})
        if not band.get("ok"):
            continue
        for v in band.get("periodicities", []):
            ch = v.get("chosen", {})
            slim["periodicities"].append({
                "band": band_name,
                "period_nm": round(ch.get("period_nm", 0), 1),
                "confidence": ch.get("confidence", "?"),
                "snr": round(ch.get("snr_z", 0), 1),
                "stripe_angle_deg": round(v.get("stripe_angle_deg", 0), 1),
                "method": ch.get("method", ""),
            })

    # -- Particle summary (if present) --
    pa = report.get("results", {}).get("particles")
    if pa and pa.get("ok"):
        d = pa.get("equiv_diameter_nm", {})
        slim["particles"] = {
            "count": pa.get("count"),
            "mean_diameter_nm": round(d.get("mean", 0), 1),
            "std_nm": round(d.get("std", 0), 1),
            "range_nm": [round(d.get("p10", 0), 0), round(d.get("p90", 0), 0)],
        }

    # -- Summary text (human-readable) --
    slim["summary"] = report.get("summary", "")

    # -- Artifact URLs: top-level + per-band --
    artifacts = dict(report.get("artifacts", {}))
    for band_name in ("macro", "micro"):
        band = report.get("results", {}).get(band_name, {})
        band_arts = band.get("artifacts", {})
        for key, url in band_arts.items():
            artifacts[f"{band_name}_{key}"] = url
    slim["artifacts"] = artifacts

    return slim


# =========================================================================
#  Main MCP tool  --  dual macro/micro bands (v5 backward compat)
# =========================================================================
@mcp.tool()
def sem_analyze_fft(
    file_id: Optional[str] = None,
    image_path: Optional[str] = None,
    mag_label: Optional[str] = None,
    nm_per_px: Optional[float] = None,
    scalebar_px: Optional[float] = None,
    scalebar_nm: Optional[float] = None,
    crop_bottom_px: Optional[int] = None,
    roi: Optional[List[int]] = None,
    roi_mode: str = "auto_if_none",
    roi_grid: int = 3,
    preset: str = "both",
    macro_min_period_nm: float = 100.0,
    macro_max_period_nm: float = 2500.0,
    micro_min_period_nm: float = 10.0,
    micro_max_period_nm: float = 300.0,
    peak_rel_height: float = 0.25,
    peak_min_distance_bins: int = 8,
    top_n: int = 5,
    harmonic_tolerance_pct: float = 0.12,
    max_vectors: int = 6,
    preprocess: str = "auto",
    bg_subtract: bool = True,
    bg_degree: int = 2,
    particle_analysis: bool = False,
    particle_mode: str = "bright",
    particle_min_diam_nm: float = 20.0,
    particle_max_diam_nm: float = 2000.0,
    particle_threshold: Optional[float] = None,
    particle_smooth_sigma: float = 1.0,
) -> Dict[str, Any]:
    """
    Analyze an SEM image for periodic nanostructures.

    Uses 2D FFT, 2D autocorrelation, radial PSD, 1D line profiles,
    and per-candidate cross-validation. Returns multi-vector periodicities
    with confidence scoring and annotated plots.

    Provide EITHER file_id (from OWUI upload) OR image_path.
    The tool auto-detects magnification from the SEM info bar via OCR.
    If OCR fails, provide mag_label (e.g. 'x40000').

    preset='both' runs macro (100-2500nm) + micro (10-300nm) bands.
    Use 'macro' or 'micro' for single band analysis.
    """
    try:
        # -- Load image -----------------------------------------------
        if bool(file_id) == bool(image_path):
            return {"ok": False, "reason": "Provide exactly one: file_id OR image_path"}

        download_meta = None
        if image_path:
            raw, fname = _safe_read(image_path)
            source = {"type": "filesystem", "path": _normalize_path(image_path)}
        else:
            raw, fname, download_meta = read_file_id_bytes(str(file_id))
            source = {"type": "file_id", "file_id": file_id, "download": download_meta}

        img = Image.open(io.BytesIO(raw))
        gray_full = _to_gray(img)
        h0, w0 = gray_full.shape

        # -- Crop bottom info bar -------------------------------------
        gray, cropped_px = _crop_bottom(gray_full, crop_bottom_px)

        # -- CALIBRATE BEFORE ROI (use full-frame width) -------------
        # FIX: calibration must use pre-ROI width for correct nm/px
        inferred_mag = None
        if (not nm_per_px) and (not (scalebar_px and scalebar_nm)) and (not mag_label):
            inferred_mag = _infer_mag_from_filename(fname)
            if inferred_mag:
                mag_label = inferred_mag

        nmpp, cal_info = _calibrate_nm_per_px(
            gray.shape[1], mag_label, nm_per_px, scalebar_px, scalebar_nm,
            gray_full=gray_full, fname=fname)

        # -- ROI handling ---------------------------------------------
        roi_used = None
        roi_meta = None
        rm = (roi_mode or "auto_if_none").strip().lower()
        if rm == "manual":
            gray_roi, roi_used = _apply_roi(gray, roi)
        elif rm == "auto":
            gray_roi, roi_used, roi_meta = _pick_best_roi(
                gray, nmpp, float(macro_min_period_nm), float(macro_max_period_nm),
                grid=int(roi_grid))
        else:  # auto_if_none
            if roi:
                gray_roi, roi_used = _apply_roi(gray, roi)
            else:
                gray_roi, roi_used, roi_meta = _pick_best_roi(
                    gray, nmpp, float(macro_min_period_nm), float(macro_max_period_nm),
                    grid=int(roi_grid))

        # -- Determine which bands to run -----------------------------
        p = (preset or "both").strip().lower()
        run_macro = p in ("both", "macro", "auto")
        run_micro = p in ("both", "micro", "auto")

        folder = _new_folder()
        url_preview = export_fig(folder, _plot_preview(gray_roi, "SEM preview (analysis region)"),
                                 "sem_preview.png")

        report: Dict[str, Any] = {
            "ok": True,
            "version": "micro_sem_suite_v7_0",
            "input": {
                "source": source, "filename": fname,
                "original_dimensions_px": [int(w0), int(h0)],
                "analysis_dimensions_px": [int(gray_roi.shape[1]), int(gray_roi.shape[0])],
                "crop_bottom_px": int(cropped_px),
                "roi": roi_used, "roi_mode": rm, "roi_auto_meta": roi_meta,
            },
            "scale": {
                **cal_info,
                "mag_label": mag_label, "mag_inferred": inferred_mag,
                "nm_per_px": float(nmpp),
                "field_of_view_nm": [float(gray_roi.shape[1] * nmpp),
                                     float(gray_roi.shape[0] * nmpp)],
            },
            "results": {},
            "artifacts": {"preview_png": url_preview},
        }

        # -- Helper to export a band's results + artifacts ------------
        def _export_band(band_name: str, pack: Dict[str, Any],
                         min_p: float, max_p: float) -> Dict[str, Any]:
            ad = pack["artifacts_data"]
            per = pack["periodicities"]
            peaks_rc = [v["peak_2d_rc"] for v in per]
            labels = [f'{v["chosen"]["period_nm"]:.0f}nm @ '
                      f'{v["stripe_angle_deg"]:.0f} deg ({v["chosen"]["confidence"]})'
                      for v in per]

            url_fft = export_fig(folder,
                _plot_fft2d_multi(ad["fft_power"], peaks_rc, labels,
                                 f"{band_name.upper()} FFT2D"),
                f"{band_name}_fft2d.png")
            url_psd = export_fig(folder,
                _plot_psd(ad["freq"], ad["psd"], float(min_p), float(max_p),
                          float(pack["primary"]["chosen"]["period_nm"]),
                          f"{band_name.upper()} 1D PSD ({min_p:.0f}-{max_p:.0f} nm)"),
                f"{band_name}_psd.png")
            url_ac2 = export_fig(folder,
                _plot_autocorr2d(ad["autocorr2d"],
                                 f"{band_name.upper()} 2D Autocorrelation"),
                f"{band_name}_autocorr2d.png")

            # Radial PSD for this band
            FX_b, FY_b, FR_b = _freq_grids(
                ad["fft_power"].shape[0], ad["fft_power"].shape[1], nmpp)
            rp, rpsd = _radial_psd(ad["fft_power"], FR_b, min_p, max_p)

            # Extract 1D line profile for primary periodicity
            primary = pack["primary"]
            ang_k = primary["k_angle_deg"]
            rot_ang = _normalize_rotation_angle(ang_k)
            sig = ad.get("signal", gray_roi)
            rot = _rotate_keep_center(sig, angle_deg=rot_ang)
            prof = np.mean(rot, axis=0)
            detr = max(50.0, 0.8 * max_p / nmpp)
            prof = prof - gaussian_filter1d(prof, sigma=float(detr))

            dominant = pack["primary"]["chosen"]
            return {
                "ok": True,
                "dominant_periodicity": dominant,
                "stripe_angle_deg": primary["stripe_angle_deg"],
                "selected_signal": primary["selected_signal"],
                "period_from_2d_fft_nm": primary["period_from_2d_fft_nm"],
                "anisotropy": pack["anisotropy"],
                "periodicities": per,
                "radial_psd": {"period_nm": rp.tolist(), "power": rpsd.tolist()},
                "line_profile": prof.tolist(),
                "artifacts": {
                    "fft2d_png": url_fft, "psd_png": url_psd,
                    "autocorr2d_png": url_ac2,
                },
            }

        band_results: Dict[str, Dict] = {}

        if run_macro:
            macro = _analyze_band_multi(
                gray=gray_roi, nm_per_px=nmpp,
                min_period_nm=float(macro_min_period_nm),
                max_period_nm=float(macro_max_period_nm),
                signal_mode="auto", smooth_sigma=1.2,
                dc_exclusion_px=3, tile_rows=5,
                peak_rel_height=float(peak_rel_height),
                peak_min_distance_bins=int(peak_min_distance_bins),
                top_n=int(top_n), harmonic_tol_pct=float(harmonic_tolerance_pct),
                max_vectors=int(max_vectors), preprocess=str(preprocess),
                label_name="macro", bg_subtract=bg_subtract, bg_degree=bg_degree)
            if macro.get("ok"):
                band_results["macro"] = _export_band("macro", macro,
                    macro_min_period_nm, macro_max_period_nm)
            else:
                band_results["macro"] = macro
            report["results"]["macro"] = band_results["macro"]

        if run_micro:
            micro = _analyze_band_multi(
                gray=gray_roi, nm_per_px=nmpp,
                min_period_nm=float(micro_min_period_nm),
                max_period_nm=float(micro_max_period_nm),
                signal_mode="gradient", smooth_sigma=1.0,
                dc_exclusion_px=6, tile_rows=6,
                peak_rel_height=float(peak_rel_height),
                peak_min_distance_bins=int(max(6, peak_min_distance_bins)),
                top_n=int(top_n), harmonic_tol_pct=float(harmonic_tolerance_pct),
                max_vectors=int(max_vectors), preprocess=str(preprocess),
                label_name="micro", bg_subtract=bg_subtract, bg_degree=bg_degree)
            if micro.get("ok"):
                band_results["micro"] = _export_band("micro", micro,
                    micro_min_period_nm, micro_max_period_nm)
            else:
                band_results["micro"] = micro
            report["results"]["micro"] = band_results["micro"]

        # -- Particle analysis ----------------------------------------
        if bool(particle_analysis):
            pa = _particle_analysis(gray_roi, nmpp, str(particle_mode),
                                    float(particle_min_diam_nm),
                                    float(particle_max_diam_nm),
                                    particle_threshold, float(particle_smooth_sigma))
            report["results"]["particles"] = pa
        else:
            pa = None

        # -- Composite summary figure ---------------------------------
        # Pick the best band for the composite
        best_band = None
        for bn in ("macro", "micro"):
            br = band_results.get(bn, {})
            if br.get("ok") and br.get("periodicities"):
                best_band = br
                break

        if best_band:
            per = best_band["periodicities"]
            ad_key = "macro" if best_band is band_results.get("macro") else "micro"
            # Reconstruct data for composite
            pack = macro if ad_key == "macro" and macro.get("ok") else micro
            ad = pack["artifacts_data"]
            peaks_rc = [v["peak_2d_rc"] for v in per]
            labels = [f'{v["chosen"]["period_nm"]:.0f}nm @ '
                      f'{v["stripe_angle_deg"]:.0f} deg ({v["chosen"]["confidence"]})'
                      for v in per]
            ac2_peaks = _pick_autocorr2d_peaks(
                ad["autocorr2d"], nmpp,
                macro_min_period_nm if ad_key == "macro" else micro_min_period_nm,
                macro_max_period_nm if ad_key == "macro" else micro_max_period_nm)
            FX_c, FY_c, FR_c = _freq_grids(
                ad["fft_power"].shape[0], ad["fft_power"].shape[1], nmpp)
            min_c = macro_min_period_nm if ad_key == "macro" else micro_min_period_nm
            max_c = macro_max_period_nm if ad_key == "macro" else micro_max_period_nm
            rp_c, rpsd_c = _radial_psd(ad["fft_power"], FR_c, min_c, max_c)
            prof_c = np.array(best_band.get("line_profile", [0.0]), dtype=np.float32)

            fov_nm = [float(gray_roi.shape[1] * nmpp), float(gray_roi.shape[0] * nmpp)]
            summ_text = _generate_summary_text(band_results, cal_info, fov_nm, pa)
            summ_lines = [
                "SEM Periodicity Analysis", "-" * 40,
                f"Calibration: {cal_info.get('mode', '?')} ({mag_label or '?'})",
                f"Scale: {nmpp:.3f} nm/px", "",
            ]
            for line in summ_text.split("\n"):
                summ_lines.append(line)
            summ_lines.append("")
            summ_lines.append(f"v7.0 | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

            # Extract detected period values for PSD markers
            _det_periods = [v["chosen"]["period_nm"] for v in per
                            if v.get("chosen", {}).get("period_nm")]
            url_composite = export_fig(folder, _plot_composite(
                gray_roi, ad["fft_power"], peaks_rc, labels,
                ad["autocorr2d"], ac2_peaks,
                rp_c, rpsd_c, prof_c, nmpp, summ_lines,
                detected_periods_nm=_det_periods,
            ), "summary.png", dpi=180)
            report["artifacts"]["summary_png"] = url_composite

            # CSV export of primary line profile
            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf)
            writer.writerow(["position_nm", "intensity"])
            for i, v in enumerate(prof_c):
                writer.writerow([f"{i * nmpp:.3f}", f"{v:.6f}"])
            url_csv = export_bytes(folder, csv_buf.getvalue().encode("utf-8"),
                                   "line_profile.csv")
            report["artifacts"]["line_profile_csv"] = url_csv

        # -- Summary text ---------------------------------------------
        fov_nm = [float(gray_roi.shape[1] * nmpp), float(gray_roi.shape[0] * nmpp)]
        report["summary"] = _generate_summary_text(band_results, cal_info, fov_nm, pa)

        # -- Export full JSON report ----------------------------------
        url_json = export_bytes(folder, json.dumps(report, indent=2, default=str).encode("utf-8"),
                                "sem_report.json")
        report["artifacts"]["report_json"] = url_json

        report["quality"] = {
            "notes": [
                "v7: disk-first I/O, calibrate-before-ROI, per-candidate validation, "
                "rotation-sign fix, tile-detrend fix, poly bg subtraction, "
                "2D autocorr (zero-padded), radial PSD, composite figure.",
            ]
        }

        # Full report saved as JSON artifact above.
        # Return slim, model-friendly response (~1-2KB instead of ~100KB).
        return _slim_response(report)

    except Exception as e:
        logger.exception("sem_analyze_fft failed")
        return {"ok": False, "reason": str(e)}


# =========================================================================
#  Server entry point
# =========================================================================
if __name__ == "__main__":
    _mode = os.getenv("MODE", "stdio").lower()
    logger.info(f"Starting micro SEM server v7.0 (mode={_mode}, "
                f"tesseract={'yes' if HAS_TESSERACT else 'no'})")
    if _mode == "http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")