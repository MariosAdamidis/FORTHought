# xrd/server.py  --  XRD Phase Identification & Purity Analysis
# Author:  Marios Adamidis (FORTHought Lab)
# Version: 1.0.0
#
# Changelog v1→v2:
#   - Split single analyze_xrd into 4 MCP tools:
#       1. parse_xrd      — parse file + detect peaks, NO database search
#       2. identify_xrd   — full pipeline (parse + search + match + impurity)
#       3. search_xrd_ref — search COD/MP for a material's reference pattern only
#       4. export_xrd_origin — export processed data for OriginLab import
#   - Added Bruker RAW v4 binary parser (reverse-engineered from D8 Advance files)
#   - Improved confidence scoring: intensity-weighted, don't penalize for
#     unmatched reference peaks below 5% relative intensity
#   - Background subtraction before Rwp calculation (drops Rwp from ~97% to ~15-30%)
#   - Perovskite polymorph table expanded (CsPbBr3, CsPbI3, MAPbI3, etc.)
#   - Origin export: multi-column CSV with metadata header for direct Origin import
#
# Architecture: follows micro.py (SEM) patterns exactly
#   - Disk-first file_id ingestion (OWUI HTTP fallback)
#   - Structured {ok:false} on all failures (never crash)
#   - Slim JSON response to LLM, full data in exported artifacts
#   - FastMCP HTTP transport

import os
import io
import re
import csv
import json
import glob
import uuid
import math
import time
import struct
import zipfile
import datetime
import logging
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d

from mcp.server.fastmcp import FastMCP

# -- Optional heavy imports (fail gracefully) ---------------------------------
try:
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False

try:
    from mp_api.client import MPRester
    HAS_MP_API = True
except ImportError:
    HAS_MP_API = False


# =============================================================================
#  Logging
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [xrd] %(message)s")
logger = logging.getLogger("xrd")

# =============================================================================
#  Config (matches micro.py env vars)
# =============================================================================
OWUI_URL = os.getenv("OWUI_URL", "").rstrip("/")
JWT_TOKEN = os.getenv("JWT_TOKEN", "")
JWT_SECRET = os.getenv("JWT_SECRET", "")
AUTH_TOKEN = (JWT_TOKEN or JWT_SECRET).strip()

EXPORT_DIR = os.getenv("FILE_EXPORT_DIR", "/data/files").rstrip("/")
BASE_URL = os.getenv("FILE_EXPORT_BASE_URL", "http://localhost:8084/files").rstrip("/")
os.makedirs(EXPORT_DIR, exist_ok=True)

UPLOAD_ROOT = os.getenv("OWUI_UPLOADS_ROOT", "/openwebui_uploads").rstrip("/")

MP_API_KEY = os.getenv("MP_API_KEY", "")

# Wavelengths in Angstroms
WAVELENGTHS = {
    "CuKa":  1.54184,
    "CuKa1": 1.54056,
    "CuKa2": 1.54439,
    "MoKa":  0.71073,
    "CoKa":  1.78901,
    "FeKa":  1.93604,
    "CrKa":  2.28970,
    "AgKa":  0.55941,
}

COD_BASE = "https://www.crystallography.net/cod"
COD_SEARCH_URL = f"{COD_BASE}/result"
COD_CIF_URL = f"{COD_BASE}/"  # + {cod_id}.cif

mcp = FastMCP(
    name="xrd",
    host=os.getenv("MCP_HTTP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_HTTP_PORT", "9008")),
)


# =============================================================================
#  I/O helpers  (disk-first file_id, from micro.py)
# =============================================================================
def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}

def _normalize_path(p: str) -> str:
    p = p.strip()
    # Handle OWUI's {uploads_dir}/ prefix
    if "{uploads_dir}" in p:
        p = p.replace("{uploads_dir}/", "").replace("{uploads_dir}", "")
    # Handle absolute paths within the uploads root
    if p.startswith("/"):
        return p
    return os.path.join(UPLOAD_ROOT, p)

def _safe_read(path: str) -> Tuple[bytes, str]:
    path = _normalize_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        return f.read(), os.path.basename(path)

def _disk_find_by_file_id(file_id: str) -> Optional[str]:
    """Scan uploads dir for a file whose name starts with file_id."""
    if not UPLOAD_ROOT or not os.path.isdir(UPLOAD_ROOT):
        return None
    for root, dirs, files in os.walk(UPLOAD_ROOT):
        for fn in files:
            if fn.startswith(file_id):
                return os.path.join(root, fn)
    return None

def _owui_download_bytes(file_id: str) -> Tuple[bytes, str]:
    """Download file bytes from OWUI API."""
    if not OWUI_URL:
        raise RuntimeError("OWUI_URL not configured")
    url = f"{OWUI_URL}/api/v1/files/{file_id}/content"
    r = requests.get(url, headers=_auth_headers(), timeout=60)
    r.raise_for_status()
    cd = r.headers.get("content-disposition", "")
    m = re.search(r'filename="?([^"]+)"?', cd)
    fname = m.group(1) if m else f"{file_id}.dat"
    return r.content, fname

def read_file_id_bytes(file_id: str) -> Tuple[bytes, str, Dict[str, Any]]:
    """Resolve file_id: try disk first, fall back to OWUI HTTP."""
    disk = _disk_find_by_file_id(file_id)
    if disk:
        with open(disk, "rb") as f:
            return f.read(), os.path.basename(disk), {"mode": "disk", "path": disk}
    raw, fname = _owui_download_bytes(file_id)
    return raw, fname, {"mode": "http", "url": f"{OWUI_URL}/api/v1/files/{file_id}/content"}


# =============================================================================
#  Export helpers
# =============================================================================
def _new_folder() -> str:
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    folder = os.path.join(EXPORT_DIR, f"xrd_{uid}_{tag}")
    os.makedirs(folder, exist_ok=True)
    return folder

def _url(folder: str, fname: str) -> str:
    rel = os.path.relpath(os.path.join(folder, fname), EXPORT_DIR)
    return f"{BASE_URL}/{rel}"

def export_bytes(folder: str, data: bytes, fname: str) -> str:
    path = os.path.join(folder, fname)
    with open(path, "wb") as f:
        f.write(data)
    return _url(folder, fname)

def export_fig(folder: str, fig: plt.Figure, fname: str, dpi: int = 200) -> str:
    path = os.path.join(folder, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return _url(folder, fname)


# =============================================================================
#  File parsers
# =============================================================================

def parse_xy(raw: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Parse ASCII two-column XRD data (.xy, .dat, .csv, .txt)."""
    text = raw.decode("utf-8", errors="replace")
    lines = text.strip().replace("\r\n", "\n").split("\n")

    two_theta = []
    intensity = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("'") or line.startswith(";"):
            continue
        # Try comma, semicolon, tab, space as delimiters
        for sep in (",", ";", "\t", None):
            parts = line.split(sep) if sep else line.split()
            if len(parts) >= 2:
                try:
                    tt = float(parts[0])
                    ii = float(parts[1])
                    two_theta.append(tt)
                    intensity.append(ii)
                    break
                except ValueError:
                    continue

    if len(two_theta) < 10:
        raise ValueError(f"Only {len(two_theta)} valid data points found in ASCII file")

    return np.array(two_theta), np.array(intensity)


def parse_brml(raw: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse Bruker .brml file (ZIP archive containing XML data).
    Extracts 2θ and intensity from the XML structure.

    Bruker D8 Datum format (common):  time_per_step, flag, 2theta, theta, raw_counts
    Intensity is the LAST column; divided by time_per_step → counts-per-second.
    Reference: DxTools data_reader.py (A. Boulle, J. Appl. Cryst. 2017)
    """
    buf = io.BytesIO(raw)
    if not zipfile.is_zipfile(buf):
        raise ValueError("Not a valid .brml file (not a ZIP archive)")

    buf.seek(0)
    two_theta_all: list[float] = []
    intensity_all: list[float] = []

    with zipfile.ZipFile(buf, "r") as zf:
        xml_files = [n for n in zf.namelist()
                     if n.lower().endswith(".xml") and "rawdata" in n.lower()]
        if not xml_files:
            xml_files = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        if not xml_files:
            raise ValueError("No XML data found in .brml archive")

        import xml.etree.ElementTree as ET

        for xml_name in xml_files:
            with zf.open(xml_name) as xf:
                try:
                    tree = ET.parse(xf)
                except ET.ParseError:
                    continue
                root = tree.getroot()

            data_routes = (
                root.findall(".//DataRoutes/DataRoute")
                or root.findall(".//DataRoute")
                or []
            )

            for route in data_routes:
                tt_start = None
                tt_step = None

                # ---- Method 1: ScanAxes/ScanAxisInfo (D8 Advance) ----
                for axis in route.findall(".//ScanAxes/ScanAxisInfo"):
                    axis_id = (axis.get("AxisId", "") + axis.get("AxisName", "")).lower()
                    if "twotheta" in axis_id or "2theta" in axis_id:
                        s = axis.findtext("Start")
                        i = axis.findtext("Increment")
                        if s is not None:
                            tt_start = float(s)
                        if i is not None:
                            tt_step = float(i)

                # ---- Method 2: Drives/InfoData (older instruments) ----
                if tt_start is None:
                    for drives in route.findall(".//Drives/InfoData") + route.findall(".//AxisInformation"):
                        axis_name = drives.get("AxisName", drives.get("LogicName", ""))
                        if "2theta" in axis_name.lower() or "twotheta" in axis_name.lower():
                            start_el = drives.find(".//Start") or drives.find("Start")
                            step_el = (drives.find(".//Increment") or drives.find("Increment")
                                       or drives.find(".//Step"))
                            if start_el is not None and start_el.text:
                                tt_start = float(start_el.text)
                            if step_el is not None and step_el.text:
                                tt_step = float(step_el.text)

                # ---- Method 3: SubScan parameters ----
                if tt_start is None:
                    for sub in route.findall(".//SubScan") + route.findall(".//Subscan"):
                        start_el = sub.find(".//Start2Theta") or sub.find("Start2Theta")
                        step_el = sub.find(".//Step2Theta") or sub.find("Step2Theta")
                        if start_el is not None and start_el.text:
                            tt_start = float(start_el.text)
                        if step_el is not None and step_el.text:
                            tt_step = float(step_el.text)

                # ---- Extract Datum elements ----
                datums = route.findall(".//Datum") or route.findall("Datum")
                if not datums:
                    continue

                first_parts = datums[0].text.strip().split(",") if datums[0].text else []
                ncols = len(first_parts)

                time_per_step = 1.0
                if ncols >= 5:
                    try:
                        time_per_step = float(first_parts[0])
                    except (ValueError, IndexError):
                        pass

                raw_counts: list[float] = []
                embedded_angles: list[float] = []

                for datum in datums:
                    if not datum.text:
                        continue
                    parts = datum.text.strip().split(",")
                    if len(parts) < 2:
                        continue
                    try:
                        if ncols >= 5:
                            raw_counts.append(float(parts[-1]))
                            embedded_angles.append(float(parts[2]))
                        elif ncols >= 3:
                            raw_counts.append(float(parts[-1]))
                            embedded_angles.append(float(parts[0]))
                        else:
                            raw_counts.append(float(parts[-1]))
                            embedded_angles.append(float(parts[0]))
                    except (ValueError, IndexError):
                        continue

                if not raw_counts:
                    continue

                intensities = [c / time_per_step for c in raw_counts]

                if tt_start is not None and tt_step is not None and tt_step > 0:
                    n = len(intensities)
                    two_theta = np.arange(tt_start, tt_start + n * tt_step, tt_step)[:n]
                    two_theta_all.extend(two_theta.tolist())
                    intensity_all.extend(intensities)
                elif embedded_angles and len(embedded_angles) == len(intensities):
                    two_theta_all.extend(embedded_angles)
                    intensity_all.extend(intensities)

    if len(two_theta_all) < 10:
        raise ValueError(
            f"Could not extract sufficient data from .brml file ({len(two_theta_all)} points). "
            "The file may use an unsupported scan configuration. "
            "Try exporting to .xy from Bruker software."
        )

    tt = np.array(two_theta_all)
    ii = np.array(intensity_all)
    order = np.argsort(tt)
    return tt[order], ii[order]


def parse_raw4(raw: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse Bruker RAW v4 binary file (.raw).

    Format reverse-engineered from D8 Advance output (RAW4.00):
      Header:
        [0-7]    Magic "RAW4.00\\0"
        [12-21]  Date string "MM/DD/YYYY"
        [24-31]  Time string "HH:MM:SS"
        [56]     uint32: range_header_offset (typically 387)
        [384]    double: Kα wavelength
        [392]    double: Kα1 wavelength
        [400]    double: Kα2 wavelength
        [452]    uint32: number of data points
        [520]    double: start 2θ (degrees)
        [528]    double: step size (degrees)
        [540]    float32: time per step (seconds)
      Data:
        [1053+]  float32[n_points]: raw counts (NOT CPS)

    Verified against BRML and XY exports from the same instrument;
    CPS = raw_count / time_per_step matches to ±0.001 CPS.
    """
    if len(raw) < 8 or raw[:4] != b"RAW4":
        raise ValueError(
            f"Not a Bruker RAW v4 file (magic={raw[:8]!r}). "
            "Only RAW v4 (.00) is currently supported."
        )

    version = raw[:8].rstrip(b'\x00').decode('ascii', errors='replace')
    logger.info(f"RAW4 parser: version={version}, size={len(raw)} bytes")

    # --- Extract metadata ---
    date_str = raw[12:22].rstrip(b'\x00').decode('ascii', errors='replace')
    time_str = raw[24:32].rstrip(b'\x00').decode('ascii', errors='replace')
    logger.info(f"RAW4: date={date_str}, time={time_str}")

    # Wavelengths (doubles at fixed offsets)
    wavelength_ka = struct.unpack_from('<d', raw, 384)[0]
    wavelength_ka1 = struct.unpack_from('<d', raw, 392)[0]
    wavelength_ka2 = struct.unpack_from('<d', raw, 400)[0]
    logger.info(f"RAW4: λ Kα={wavelength_ka:.4f}, Kα1={wavelength_ka1:.4f}, Kα2={wavelength_ka2:.4f} Å")

    # Scan parameters
    n_points = struct.unpack_from('<I', raw, 452)[0]
    start_2theta = struct.unpack_from('<d', raw, 520)[0]
    step_size = struct.unpack_from('<d', raw, 528)[0]
    time_per_step = struct.unpack_from('<f', raw, 540)[0]

    logger.info(f"RAW4: n={n_points}, start={start_2theta:.4f}°, "
                f"step={step_size:.6f}°, time/step={time_per_step:.1f}s")

    # --- Validate ---
    if n_points < 10 or n_points > 100000:
        raise ValueError(f"Implausible point count: {n_points}")
    if step_size <= 0 or step_size > 1.0:
        raise ValueError(f"Implausible step size: {step_size}")
    if start_2theta < 0 or start_2theta > 170:
        raise ValueError(f"Implausible start angle: {start_2theta}")
    if time_per_step <= 0:
        time_per_step = 1.0  # fallback

    # --- Find data offset ---
    # Data is float32 array, typically starts at offset 1053 for standard
    # D8 files. We verify by checking the expected file size.
    data_offset = 1053
    expected_size = data_offset + n_points * 4

    if len(raw) < expected_size:
        # Try alternative: data might be right after the header
        # Search for the data block by checking if float values make sense
        for candidate_offset in [1053, 960, 768, 512]:
            test_end = candidate_offset + n_points * 4
            if test_end <= len(raw):
                # Check first few values look like counts
                test_vals = struct.unpack_from(f'<{min(5,n_points)}f', raw, candidate_offset)
                if all(0 <= v < 1e8 for v in test_vals):
                    data_offset = candidate_offset
                    break
        else:
            raise ValueError(
                f"Cannot locate data block: file={len(raw)} bytes, "
                f"need {expected_size} for {n_points} points @ offset {data_offset}"
            )

    # --- Extract intensity data ---
    raw_counts = np.array(
        struct.unpack_from(f'<{n_points}f', raw, data_offset),
        dtype=np.float64
    )

    # Convert to CPS
    cps = raw_counts / time_per_step

    # Build 2θ array
    two_theta = np.arange(n_points, dtype=np.float64) * step_size + start_2theta

    end_2theta = two_theta[-1]
    logger.info(f"RAW4: parsed {n_points} points, 2θ=[{start_2theta:.3f}, {end_2theta:.3f}]°, "
                f"I_max={np.max(cps):.1f} CPS")

    return two_theta, cps


def detect_format(fname: str, raw: bytes) -> str:
    ext = os.path.splitext(fname.lower())[1]
    if ext in (".xy", ".dat", ".txt", ".csv", ".asc"):
        return "xy"
    if ext == ".brml":
        return "brml"
    if ext == ".raw":
        # Sniff for RAW4 magic
        if raw[:4] == b"RAW4":
            return "raw4"
        return "raw_unknown"
    # Sniff: check if ZIP (brml)
    if raw[:4] == b"PK\x03\x04":
        return "brml"
    # Sniff: check RAW4 magic regardless of extension
    if raw[:4] == b"RAW4":
        return "raw4"
    # Sniff: check if ASCII
    try:
        raw[:500].decode("utf-8")
        return "xy"
    except UnicodeDecodeError:
        pass
    return "unknown"


def parse_file(raw: bytes, fname: str) -> Tuple[np.ndarray, np.ndarray]:
    fmt = detect_format(fname, raw)
    if fmt == "xy":
        return parse_xy(raw)
    elif fmt == "brml":
        return parse_brml(raw)
    elif fmt == "raw4":
        return parse_raw4(raw)
    elif fmt == "raw_unknown":
        raise ValueError(
            "Bruker .raw file detected but not RAW v4 format "
            f"(magic={raw[:8]!r}). Only RAW4.00 from D8 Advance is supported. "
            "Please export to .xy or .brml from DIFFRAC.EVA."
        )
    else:
        try:
            return parse_xy(raw)
        except Exception:
            raise ValueError(
                f"Unrecognized file format for '{fname}'. "
                "Supported: .xy (ASCII 2-column), .brml (Bruker XML/ZIP), "
                ".raw (Bruker RAW v4 binary)."
            )


# =============================================================================
#  Peak detection
# =============================================================================

def detect_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    height_pct: float = 5.0,
    prominence_pct: float = 3.0,
    min_distance_pts: int = 15,
    snr_threshold: float = 5.0,
) -> Dict[str, Any]:
    """
    Detect peaks in XRD pattern using scipy.signal.find_peaks with
    background-aware thresholds.
    """
    from scipy.ndimage import minimum_filter1d, median_filter

    imax = float(np.max(intensity))
    if imax <= 0:
        return {"positions": [], "intensities": [], "indices": [], "count": 0,
                "background": np.zeros_like(intensity).tolist()}

    n = len(intensity)

    # --- Background estimation ---
    win = max(int(n * 0.05), 50)
    bg_min = minimum_filter1d(intensity, size=win)
    bg = median_filter(bg_min, size=win)

    # --- Noise estimation ---
    residual = intensity - bg
    sorted_res = np.sort(residual)
    noise_region = sorted_res[: int(len(sorted_res) * 0.70)]
    sigma = float(np.std(noise_region)) if len(noise_region) > 10 else 1.0
    sigma = max(sigma, 0.01)

    # --- Thresholds ---
    pct_height = imax * (height_pct / 100.0)
    pct_prominence = imax * (prominence_pct / 100.0)
    snr_height = float(np.mean(bg)) + snr_threshold * sigma
    snr_prominence = snr_threshold * sigma

    height_thresh = max(pct_height, snr_height)
    prominence_thresh = max(pct_prominence, snr_prominence)

    indices, props = find_peaks(
        intensity,
        height=height_thresh,
        prominence=prominence_thresh,
        distance=min_distance_pts,
    )

    logger.info(
        f"Peak detection: bg_mean={np.mean(bg):.2f}, σ={sigma:.3f}, "
        f"height_thresh={height_thresh:.2f}, prom_thresh={prominence_thresh:.2f}, "
        f"found={len(indices)} peaks"
    )

    return {
        "positions": two_theta[indices].tolist(),
        "intensities": intensity[indices].tolist(),
        "indices": indices.tolist(),
        "count": len(indices),
        "background": bg.tolist(),
        "noise_sigma": sigma,
    }


# =============================================================================
#  Database search: COD
# =============================================================================

def _hill_formula(formula: str) -> str:
    """Convert a chemical formula to Hill notation for COD search."""
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    elements = {}
    for el, cnt in tokens:
        if el:
            elements[el] = elements.get(el, 0) + (int(cnt) if cnt else 1)

    ordered = []
    for special in ("C", "H"):
        if special in elements:
            n = elements.pop(special)
            ordered.append(f"{special}{n if n > 1 else ''}")
    for el in sorted(elements.keys()):
        n = elements[el]
        ordered.append(f"{el}{n if n > 1 else ''}")

    return " ".join(ordered)


def search_cod(
    formula: str,
    max_results: int = 15,
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """Search COD by chemical formula. Returns list of entries with COD IDs."""
    hill = _hill_formula(formula)
    logger.info(f"COD search: formula='{formula}' -> hill='{hill}'")

    try:
        r = requests.get(
            COD_SEARCH_URL,
            params={"formula": hill, "format": "json"},
            timeout=timeout,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"COD search failed: {e}")
        return []

    try:
        data = r.json()
    except (json.JSONDecodeError, ValueError):
        logger.warning("COD returned non-JSON response")
        return []

    results = []
    items = data.values() if isinstance(data, dict) else data
    for entry in items:
        if not isinstance(entry, dict):
            continue
        results.append({
            "cod_id": str(entry.get("file", "")),
            "formula": entry.get("formula", ""),
            "sg": entry.get("sg", ""),
            "a": entry.get("a"),
            "b": entry.get("b"),
            "c": entry.get("c"),
            "alpha": entry.get("alpha"),
            "beta": entry.get("beta"),
            "gamma": entry.get("gamma"),
            "title": entry.get("title", ""),
        })
        if len(results) >= max_results:
            break

    logger.info(f"COD returned {len(results)} entries for '{formula}'")
    return results


def fetch_cod_cif(cod_id: str, timeout: float = 20.0) -> Optional[str]:
    """Download CIF file content from COD by ID."""
    url = f"{COD_CIF_URL}{cod_id}.cif"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch CIF {cod_id}: {e}")
        return None


def cod_structure_to_pattern(
    cif_text: str,
    wavelength: str = "CuKa",
    two_theta_range: Tuple[float, float] = (10, 80),
) -> Optional[Dict[str, Any]]:
    """Parse CIF text → pymatgen Structure → compute theoretical XRD pattern."""
    if not HAS_PYMATGEN:
        return None
    try:
        s = Structure.from_str(cif_text, fmt="cif")
        wl = WAVELENGTHS.get(wavelength, 1.54184)
        calc = XRDCalculator(wavelength=wl)
        pattern = calc.get_pattern(s, two_theta_range=two_theta_range)
        sg_info = ""
        try:
            sg_info = s.get_space_group_info()[0]
        except Exception:
            pass
        return {
            "two_theta": [float(x) for x in pattern.x],
            "intensity": [float(y) for y in pattern.y],
            "hkls": [str(h) for h in pattern.hkls],
            "d_spacings": [float(d) for d in pattern.d_hkls],
            "formula": s.composition.reduced_formula,
            "sg": sg_info,
        }
    except Exception as e:
        logger.warning(f"CIF → pattern failed: {e}")
        return None


# =============================================================================
#  Database search: Materials Project
# =============================================================================

def search_mp(
    formula: str,
    wavelength: str = "CuKa",
    two_theta_range: Tuple[float, float] = (10, 80),
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search Materials Project for structures and compute XRD patterns.
    Only returns stable / near-stable phases (energy_above_hull < 0.1 eV/atom).
    """
    if not HAS_MP_API or not MP_API_KEY:
        logger.info("Materials Project unavailable (no mp-api or API key)")
        return []
    if not HAS_PYMATGEN:
        logger.info("pymatgen unavailable — cannot compute patterns from MP")
        return []

    results = []
    try:
        with MPRester(MP_API_KEY) as mpr:
            docs = mpr.materials.summary.search(
                formula=formula,
                energy_above_hull=(0, 0.1),
                fields=["material_id", "formula_pretty", "symmetry",
                         "energy_above_hull", "structure"],
            )
            logger.info(f"MP returned {len(docs)} stable phases for '{formula}'")

            wl = WAVELENGTHS.get(wavelength, 1.54184)
            calc = XRDCalculator(wavelength=wl)

            for doc in sorted(docs, key=lambda d: d.energy_above_hull)[:max_results]:
                try:
                    pattern = calc.get_pattern(doc.structure, two_theta_range=two_theta_range)
                    sg_symbol = doc.symmetry.symbol if doc.symmetry else ""
                    polymorph = _identify_polymorph(formula, sg_symbol)

                    results.append({
                        "mp_id": str(doc.material_id),
                        "formula": doc.formula_pretty,
                        "sg": sg_symbol,
                        "polymorph": polymorph,
                        "energy_above_hull": float(doc.energy_above_hull),
                        "two_theta": [float(x) for x in pattern.x],
                        "intensity": [float(y) for y in pattern.y],
                        "hkls": [str(h) for h in pattern.hkls],
                        "d_spacings": [float(d) for d in pattern.d_hkls],
                    })
                except Exception as e:
                    logger.warning(f"Pattern calc failed for {doc.material_id}: {e}")
                    continue

    except Exception as e:
        logger.warning(f"Materials Project search failed: {e}")

    return results


def _identify_polymorph(formula: str, space_group: str) -> str:
    """Map well-known formula + space group combinations to polymorph names."""
    key = f"{formula.lower()}|{space_group}"
    known = {
        # -- Oxides --
        "tio2|I4_1/amd":    "anatase",
        "tio2|P4_2/mnm":    "rutile",
        "tio2|Pbca":         "brookite",
        "sio2|P3_221":       "quartz-α",
        "sio2|P6_222":       "quartz-β",
        "sio2|I-42d":        "cristobalite-α",
        "al2o3|R-3c":        "corundum",
        "fe2o3|R-3c":        "hematite",
        "fe3o4|Fd-3m":       "magnetite",
        "caco3|R-3c":        "calcite",
        "caco3|Pmcn":        "aragonite",
        "zno|P6_3mc":        "wurtzite",
        "zno|F-43m":         "zinc blende",
        "zro2|P2_1/c":       "monoclinic",
        "zro2|P4_2/nmc":     "tetragonal",
        "zro2|Fm-3m":        "cubic",
        # -- Halide perovskites --
        "cspbbr3|Pnma":      "orthorhombic (Pnma)",
        "cspbbr3|Pm-3m":     "cubic (Pm-3m)",
        "cspbbr3|P4/mbm":    "tetragonal",
        "cspbi3|Pnma":       "δ-phase (yellow, non-perovskite)",
        "cspbi3|Pm-3m":      "α-phase (black, cubic)",
        "mapbi3|I4/mcm":     "tetragonal",
        "mapbi3|Pm-3m":      "cubic",
        "mapbi3|Pnma":       "orthorhombic",
        "fapbi3|Pm-3m":      "α-phase (cubic)",
        "fapbi3|P6_3mc":     "δ-phase (hexagonal, non-perovskite)",
        "batio3|P4mm":       "tetragonal",
        "batio3|Pm-3m":      "cubic",
        "srtio3|Pm-3m":      "cubic",
    }
    return known.get(key, "")


# =============================================================================
#  Pattern matching  (peak-position screen + Rwp)
# =============================================================================

def _peak_position_score(
    exp_peaks: List[float],
    ref_peaks: List[float],
    tolerance_deg: float = 0.3,
    ref_intensities: Optional[List[float]] = None,
    min_ref_intensity_pct: float = 5.0,
) -> Dict[str, Any]:
    """
    v2: Intensity-weighted peak position scoring.
    - Filters reference peaks below min_ref_intensity_pct of max
    - Weights matches by reference peak intensity (strong peaks matter more)
    """
    if not ref_peaks:
        return {"score": 0, "matched": 0, "total": 0, "matched_pairs": [],
                "unmatched_ref": ref_peaks, "weighted_score": 0}

    # Filter weak reference peaks
    working_refs = list(ref_peaks)
    working_ints = list(ref_intensities) if ref_intensities else [100.0] * len(ref_peaks)

    if ref_intensities and len(ref_intensities) == len(ref_peaks):
        max_int = max(ref_intensities) if ref_intensities else 1
        threshold = max_int * (min_ref_intensity_pct / 100.0)
        filtered = [(rp, ri) for rp, ri in zip(ref_peaks, ref_intensities) if ri >= threshold]
        if filtered:
            working_refs, working_ints = zip(*filtered)
            working_refs = list(working_refs)
            working_ints = list(working_ints)

    if not working_refs:
        working_refs = list(ref_peaks)
        working_ints = [100.0] * len(ref_peaks)

    exp_arr = np.array(exp_peaks)
    matched_pairs = []
    unmatched_ref = []
    matched_weight = 0.0
    total_weight = sum(working_ints)

    for rp, ri in zip(working_refs, working_ints):
        diffs = np.abs(exp_arr - rp)
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] <= tolerance_deg:
            matched_pairs.append((float(exp_arr[best_idx]), float(rp)))
            matched_weight += ri
        else:
            unmatched_ref.append(float(rp))

    matched = len(matched_pairs)
    total = len(working_refs)
    # v2: weighted score accounts for peak intensity
    weighted_score = matched_weight / total_weight if total_weight > 0 else 0

    return {
        "score": matched / total if total > 0 else 0,
        "weighted_score": weighted_score,
        "matched": matched,
        "total": total,
        "matched_pairs": matched_pairs,
        "unmatched_ref": unmatched_ref,
    }


def _estimate_background(
    two_theta: np.ndarray,
    intensity: np.ndarray,
) -> np.ndarray:
    """Estimate background for Rwp calculation using rolling minimum + smooth."""
    from scipy.ndimage import minimum_filter1d, median_filter
    n = len(intensity)
    win = max(int(n * 0.05), 50)
    bg_min = minimum_filter1d(intensity, size=win)
    bg = median_filter(bg_min, size=win)
    return bg.astype(np.float64)


def compute_rwp(
    exp_tt: np.ndarray,
    exp_int: np.ndarray,
    ref_tt: List[float],
    ref_int: List[float],
    two_theta_range: Tuple[float, float] = (10, 80),
    subtract_background: bool = True,
) -> float:
    """
    Compute weighted R-factor (Rwp) between experimental and reference patterns.

    v2 improvement: subtract estimated background from experimental data before
    computing Rwp. This prevents the flat baseline from inflating the denominator
    and producing artificially high Rwp values.
    """
    if not ref_tt or len(exp_tt) < 10:
        return 100.0

    tt_min, tt_max = two_theta_range
    step = float(np.median(np.diff(exp_tt))) if len(exp_tt) > 1 else 0.02
    grid = np.arange(tt_min, tt_max, step)

    f_exp = interp1d(exp_tt, exp_int, kind="linear", bounds_error=False, fill_value=0)
    i_exp = f_exp(grid)

    # v2: subtract background before comparison
    if subtract_background:
        bg = _estimate_background(grid, i_exp)
        i_exp_sub = np.maximum(i_exp - bg, 0)
    else:
        i_exp_sub = i_exp

    # Build broadened reference profile (pseudo-Voigt, FWHM ~ 0.15°)
    i_calc = np.zeros_like(grid)
    fwhm = 0.15
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    eta = 0.5

    for tt_r, int_r in zip(ref_tt, ref_int):
        gauss = np.exp(-0.5 * ((grid - tt_r) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        gamma = fwhm / 2
        lorentz = (gamma / np.pi) / ((grid - tt_r) ** 2 + gamma ** 2)
        pv = eta * lorentz + (1 - eta) * gauss
        i_calc += int_r * pv

    # Normalize both to max = 100
    if np.max(i_exp_sub) > 0:
        i_exp_norm = i_exp_sub / np.max(i_exp_sub) * 100.0
    else:
        i_exp_norm = i_exp_sub
    if np.max(i_calc) > 0:
        i_calc = i_calc / np.max(i_calc) * 100.0

    # Optimal scale factor
    w = 1.0 / np.maximum(i_exp_norm, 1.0)
    num = np.sum(w * i_exp_norm * i_calc)
    den = np.sum(w * i_calc ** 2)
    scale = num / den if den > 0 else 1.0

    # Rwp
    diff = i_exp_norm - scale * i_calc
    rwp_num = np.sum(w * diff ** 2)
    rwp_den = np.sum(w * i_exp_norm ** 2)
    rwp = np.sqrt(rwp_num / rwp_den) * 100.0 if rwp_den > 0 else 100.0

    return float(min(rwp, 100.0))


def _rwp_quality(rwp: float) -> str:
    if rwp < 5:
        return "excellent"
    elif rwp < 10:
        return "good"
    elif rwp < 20:
        return "fair"
    elif rwp < 40:
        return "moderate"
    else:
        return "poor"


# =============================================================================
#  Multi-phase analysis
# =============================================================================

def analyze_phases(
    exp_tt: np.ndarray,
    exp_int: np.ndarray,
    candidates: List[Dict[str, Any]],
    exp_peaks: List[float],
    two_theta_range: Tuple[float, float] = (10, 80),
    max_phases: int = 3,
) -> List[Dict[str, Any]]:
    """Sequential multi-phase analysis with v2 intensity-weighted scoring."""
    if not candidates or not exp_peaks:
        return []

    results = []
    residual_peaks = list(exp_peaks)
    used_ids = set()

    for phase_num in range(max_phases):
        if not residual_peaks:
            break

        best = None
        best_score = -1

        for cand in candidates:
            cid = cand.get("cod_id") or cand.get("mp_id", "")
            if cid in used_ids:
                continue

            ref_peaks = cand.get("two_theta", [])
            if not ref_peaks:
                continue

            ps = _peak_position_score(
                residual_peaks, ref_peaks,
                ref_intensities=cand.get("intensity"),
            )
            # v2: use weighted_score (accounts for intensity) as primary ranking
            score = ps["weighted_score"]
            if score > best_score:
                best_score = score
                best = cand
                best_ps = ps

        if best is None or best_score < 0.25:
            break

        rwp = compute_rwp(
            exp_tt, exp_int,
            best.get("two_theta", []),
            best.get("intensity", []),
            two_theta_range,
            subtract_background=True,  # v2
        )

        cid = best.get("cod_id") or best.get("mp_id", "")
        used_ids.add(cid)

        # v2: improved confidence formula
        # weighted_score (0-1) is the primary factor (how well strong peaks match)
        # rwp bonus for good profile fits
        ws = best_ps["weighted_score"]
        rwp_factor = max(0, (50 - rwp) / 50.0)  # 0 at Rwp=50, 1 at Rwp=0
        confidence = ws * 75 + rwp_factor * 25  # max 100
        confidence = max(0, min(100, confidence))

        result = {
            "phase": best.get("formula", "?"),
            "polymorph": best.get("polymorph", ""),
            "cod_id": best.get("cod_id", ""),
            "mp_id": best.get("mp_id", ""),
            "sg": best.get("sg", ""),
            "confidence": round(confidence, 1),
            "rwp": round(rwp, 1),
            "quality": _rwp_quality(rwp),
            "peaks_matched": best_ps["matched"],
            "peaks_in_ref": best_ps["total"],
        }
        results.append(result)

        matched_exp_positions = {pair[0] for pair in best_ps["matched_pairs"]}
        residual_peaks = [p for p in residual_peaks if p not in matched_exp_positions]

    return results


# =============================================================================
#  Impurity detection
# =============================================================================

def check_impurities(
    unassigned_peaks: List[float],
    reactants: List[str],
    wavelength: str = "CuKa",
    two_theta_range: Tuple[float, float] = (10, 80),
    tolerance_deg: float = 0.4,
) -> List[Dict[str, Any]]:
    """Check if unassigned peaks match known phases of reactant compounds."""
    if not unassigned_peaks or not reactants:
        return []

    impurities = []
    for reactant in reactants:
        reactant = reactant.strip()
        if not reactant:
            continue

        mp_results = search_mp(reactant, wavelength, two_theta_range, max_results=3)

        for cand in mp_results:
            ref_peaks = cand.get("two_theta", [])
            if not ref_peaks:
                continue

            ps = _peak_position_score(unassigned_peaks, ref_peaks, tolerance_deg)
            if ps["matched"] > 0:
                impurities.append({
                    "compound": cand.get("formula", reactant),
                    "mp_id": cand.get("mp_id", ""),
                    "sg": cand.get("sg", ""),
                    "peaks_matched": ps["matched"],
                    "matched_positions": [pair[0] for pair in ps["matched_pairs"]],
                })

    return impurities


# =============================================================================
#  Plotting
# =============================================================================

def plot_pattern(
    exp_tt: np.ndarray,
    exp_int: np.ndarray,
    peaks: Dict[str, Any],
    folder: str,
) -> str:
    """Plot experimental XRD pattern with detected peaks annotated."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(exp_tt, exp_int, "k-", linewidth=0.7, label="Experimental")

    peak_pos = peaks.get("positions", [])
    peak_int = peaks.get("intensities", [])
    if peak_pos:
        ax.plot(peak_pos, peak_int, "rv", markersize=5, label=f"Peaks ({len(peak_pos)})")
        if peak_pos:
            sorted_idx = np.argsort(peak_int)[::-1][:15]
            for i in sorted_idx:
                ax.annotate(
                    f"{peak_pos[i]:.1f}°",
                    (peak_pos[i], peak_int[i]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    fontsize=6,
                    ha="center",
                    rotation=60,
                )

    ax.set_xlabel("2θ (°)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("XRD Pattern — Peak Detection")
    ax.legend(fontsize=8)
    ax.set_xlim(exp_tt[0], exp_tt[-1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return export_fig(folder, fig, "pattern.png")


def plot_comparison(
    exp_tt: np.ndarray,
    exp_int: np.ndarray,
    matches: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    folder: str,
) -> str:
    """Overlay plot: experimental pattern + reference stick patterns for matched phases."""
    n_panels = 1 + len(matches)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 + 2.5 * n_panels),
                              sharex=True, gridspec_kw={"hspace": 0.25})
    if n_panels == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    axes[0].plot(exp_tt, exp_int / max(1, np.max(exp_int)) * 100,
                 "k-", linewidth=0.7, label="Experimental")
    axes[0].set_ylabel("Intensity (%)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title("XRD Phase Identification")
    axes[0].grid(True, alpha=0.3)

    cand_lookup = {}
    for c in candidates:
        cid = c.get("cod_id") or c.get("mp_id", "")
        if cid:
            cand_lookup[cid] = c

    for i, match in enumerate(matches):
        ax = axes[i + 1] if i + 1 < len(axes) else axes[-1]
        color = colors[i % len(colors)]

        cid = match.get("cod_id") or match.get("mp_id", "")
        cand = cand_lookup.get(cid, {})
        ref_tt = cand.get("two_theta", [])
        ref_int = cand.get("intensity", [])

        phase_label = match.get("phase", "?")
        poly = match.get("polymorph", "")
        if poly:
            phase_label += f" ({poly})"
        sg = match.get("sg", "")
        rwp = match.get("rwp", "?")
        conf = match.get("confidence", "?")

        label = f"{phase_label}  SG: {sg}  Rwp={rwp}%  conf={conf}%"

        if ref_tt and ref_int:
            markerline, stemlines, baseline = ax.stem(
                ref_tt, ref_int, linefmt=color, markerfmt="none", basefmt="none",
            )
            stemlines.set_linewidth(1.5)
            stemlines.set_alpha(0.8)

            ax.plot(exp_tt, exp_int / max(1, np.max(exp_int)) * 100,
                    "-", color="#999", linewidth=0.4, alpha=0.5)

        ax.set_ylabel("Intensity (%)")
        ax.legend([label], fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("2θ (°)")
    axes[-1].set_xlim(exp_tt[0], exp_tt[-1])
    fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
    return export_fig(folder, fig, "comparison.png", dpi=180)


def plot_reference(
    ref_results: List[Dict[str, Any]],
    folder: str,
    two_theta_range: Tuple[float, float] = (10, 80),
) -> str:
    """Plot reference stick patterns for search_xrd_ref results."""
    fig, ax = plt.subplots(figsize=(12, 4 + 1.5 * len(ref_results)))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for i, ref in enumerate(ref_results):
        tt = ref.get("two_theta", [])
        inten = ref.get("intensity", [])
        if not tt:
            continue

        color = colors[i % len(colors)]
        offset = i * 110  # vertical stacking

        label_parts = [ref.get("formula", "?")]
        if ref.get("sg"):
            label_parts.append(f"SG: {ref['sg']}")
        if ref.get("polymorph"):
            label_parts.append(ref["polymorph"])
        cid = ref.get("cod_id") or ref.get("mp_id", "")
        if cid:
            label_parts.append(cid)
        label = "  |  ".join(label_parts)

        markerline, stemlines, baseline = ax.stem(
            tt, [y + offset for y in inten],
            linefmt=color, markerfmt="none", basefmt="none",
        )
        stemlines.set_linewidth(1.5)
        stemlines.set_alpha(0.8)
        # Label at top-left of each pattern
        ax.text(two_theta_range[0] + 0.5, offset + 90, label,
                fontsize=8, color=color, fontweight="bold")

    ax.set_xlabel("2θ (°)")
    ax.set_ylabel("Relative Intensity (%)")
    ax.set_title("XRD Reference Patterns")
    ax.set_xlim(*two_theta_range)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return export_fig(folder, fig, "references.png")


# =============================================================================
#  Response builders
# =============================================================================

def _slim_response(report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a slim, LLM-friendly response from the full report."""
    slim: Dict[str, Any] = {
        "ok": report.get("ok", True),
        "file": report.get("file", {}),
    }

    if report.get("matches"):
        slim["matches"] = report["matches"]
    if "purity_estimate" in report:
        slim["purity_estimate"] = report["purity_estimate"]
    if report.get("unassigned_peaks"):
        slim["unassigned_peaks"] = report["unassigned_peaks"][:10]
    if report.get("possible_impurities"):
        slim["possible_impurities"] = report["possible_impurities"]

    slim["peak_count"] = report.get("peak_count", 0)
    slim["summary"] = report.get("summary", "")
    slim["artifacts"] = report.get("artifacts", {})

    return slim


def _generate_summary(
    matches: List[Dict[str, Any]],
    peaks: Dict[str, Any],
    unassigned: List[float],
    impurities: List[Dict[str, Any]],
    purity: float,
) -> str:
    lines = []

    if not matches:
        lines.append(f"Detected {peaks.get('count', 0)} peaks but no database matches found.")
        lines.append("Try providing a composition_hint to narrow the search.")
        return " ".join(lines)

    m = matches[0]
    phase = m["phase"]
    poly = f" ({m['polymorph']})" if m.get("polymorph") else ""
    lines.append(
        f"Primary phase: {phase}{poly}, "
        f"confidence {m['confidence']}%, Rwp={m['rwp']}% ({m['quality']})."
    )

    for m in matches[1:]:
        poly = f" ({m['polymorph']})" if m.get("polymorph") else ""
        lines.append(f"Secondary phase: {m['phase']}{poly}, Rwp={m['rwp']}%.")

    lines.append(f"Estimated crystal purity: {purity:.1f}%.")

    if unassigned:
        lines.append(
            f"{len(unassigned)} unassigned peak(s) at 2θ = "
            f"{', '.join(f'{p:.1f}°' for p in unassigned[:5])}"
            f"{'...' if len(unassigned) > 5 else ''}."
        )

    if impurities:
        imp_names = [imp["compound"] for imp in impurities]
        lines.append(f"Possible impurities from reactants: {', '.join(imp_names)}.")

    return " ".join(lines)


# =============================================================================
#  Shared internal: load + parse + trim
# =============================================================================

def _load_and_parse(
    file_id: str = "",
    file_path: str = "",
    two_theta_min: float = 10.0,
    two_theta_max: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]:
    """
    Shared file loading, parsing, and trimming logic for all tools.
    Returns: (two_theta, intensity, filename, io_info)
    Raises on failure.
    """
    if not file_id and not file_path:
        raise ValueError("Provide either file_id or file_path.")

    if file_id:
        raw, fname, io_info = read_file_id_bytes(file_id)
    else:
        raw, fname = _safe_read(file_path)
        io_info = {"mode": "direct_path", "path": file_path}

    logger.info(f"Loaded {fname} ({len(raw)} bytes, {io_info.get('mode')})")

    exp_tt, exp_int = parse_file(raw, fname)

    mask = (exp_tt >= two_theta_min) & (exp_tt <= two_theta_max)
    exp_tt = exp_tt[mask]
    exp_int = exp_int[mask]

    if len(exp_tt) < 10:
        raise ValueError(f"Only {len(exp_tt)} data points in range [{two_theta_min}, {two_theta_max}]°.")

    logger.info(f"Parsed: {len(exp_tt)} points, 2θ=[{exp_tt[0]:.1f}, {exp_tt[-1]:.1f}]°")

    return exp_tt, exp_int, fname, io_info


# =============================================================================
#  Shared internal: database search
# =============================================================================

def _search_databases(
    composition_hint: str,
    database: str,
    wavelength: str,
    two_theta_range: Tuple[float, float],
) -> List[Dict[str, Any]]:
    """Search COD/MP for candidate phases. Used by identify_xrd and search_xrd_ref."""
    candidates = []

    if not composition_hint:
        return candidates

    formulas = [f.strip() for f in composition_hint.split(",") if f.strip()]

    for formula in formulas:
        if database in ("cod", "both"):
            cod_entries = search_cod(formula, max_results=10)
            for entry in cod_entries[:5]:
                cif_text = fetch_cod_cif(entry["cod_id"])
                if cif_text:
                    pat = cod_structure_to_pattern(cif_text, wavelength, two_theta_range)
                    if pat and pat["two_theta"]:
                        entry.update({
                            "two_theta": pat["two_theta"],
                            "intensity": pat["intensity"],
                            "hkls": pat.get("hkls", []),
                            "d_spacings": pat.get("d_spacings", []),
                            "formula": pat.get("formula", entry.get("formula", formula)),
                            "sg": pat.get("sg") or entry.get("sg", ""),
                            "polymorph": _identify_polymorph(
                                pat.get("formula", formula),
                                pat.get("sg", entry.get("sg", ""))
                            ),
                        })
                        candidates.append(entry)

        if database in ("mp", "both"):
            mp_results = search_mp(formula, wavelength, two_theta_range, max_results=8)
            candidates.extend(mp_results)

        # COD fallback to MP
        if database == "cod" and not candidates:
            logger.info("COD returned no patterns, falling back to Materials Project")
            mp_results = search_mp(formula, wavelength, two_theta_range, max_results=8)
            candidates.extend(mp_results)

    logger.info(f"Total candidates with patterns: {len(candidates)}")
    return candidates


# =============================================================================
#  Tool 1: parse_xrd  —  Parse file + detect peaks ONLY (no database)
# =============================================================================
@mcp.tool()
def parse_xrd(
    file_id: str = "",
    file_path: str = "",
    wavelength: str = "CuKa",
    two_theta_min: float = 10.0,
    two_theta_max: float = 80.0,
    peak_height_pct: float = 5.0,
    peak_prominence_pct: float = 3.0,
) -> Dict[str, Any]:
    """
    Parse an XRD data file and detect peaks — NO database search.

    Use this for quick quality checks: is the data clean? How many peaks?
    What are the peak positions? Supports .xy, .brml, and .raw (Bruker RAW v4).

    Provide EITHER file_id (from OWUI upload) OR file_path.

    Args:
        file_id: OWUI file ID from upload.
        file_path: Direct filesystem path to data file.
        wavelength: X-ray source. Default "CuKa".
        two_theta_min: Start of 2θ range (degrees). Default 10.
        two_theta_max: End of 2θ range (degrees). Default 80.
        peak_height_pct: Peak detection height threshold as % of max. Default 5.
        peak_prominence_pct: Peak detection prominence threshold as % of max. Default 3.

    Returns:
        dict with parsed data info, peak list, and pattern plot.
    """
    try:
        t0 = time.time()
        folder = _new_folder()

        exp_tt, exp_int, fname, io_info = _load_and_parse(
            file_id, file_path, two_theta_min, two_theta_max)

        peaks = detect_peaks(
            exp_tt, exp_int,
            height_pct=peak_height_pct,
            prominence_pct=peak_prominence_pct,
        )
        logger.info(f"Detected {peaks['count']} peaks")

        # Plot
        url_pattern = plot_pattern(exp_tt, exp_int, peaks, folder)

        # Export CSV of detected peaks
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["2theta_deg", "intensity_cps", "d_spacing_A"])
        wl = WAVELENGTHS.get(wavelength, 1.54184)
        for pos, intens in zip(peaks["positions"], peaks["intensities"]):
            d = wl / (2 * np.sin(np.radians(pos / 2))) if pos > 0 else 0
            writer.writerow([f"{pos:.4f}", f"{intens:.2f}", f"{d:.4f}"])
        url_csv = export_bytes(folder, csv_buf.getvalue().encode("utf-8"), "peaks.csv")

        # Summary
        summary = (
            f"Parsed {fname}: {len(exp_tt)} points, "
            f"2θ=[{exp_tt[0]:.1f}°, {exp_tt[-1]:.1f}°], "
            f"I_max={np.max(exp_int):.1f} CPS. "
            f"Detected {peaks['count']} peaks."
        )

        report = {
            "ok": True,
            "file": {"name": fname, "points": len(exp_tt),
                     "two_theta_range": [float(exp_tt[0]), float(exp_tt[-1])],
                     "format": detect_format(fname, b""),
                     "max_intensity_cps": float(np.max(exp_int))},
            "peak_count": peaks["count"],
            "peaks": [{"two_theta": round(p, 4), "intensity": round(i, 2),
                       "d_spacing": round(wl / (2 * np.sin(np.radians(p / 2))), 4) if p > 0 else 0}
                      for p, i in zip(peaks["positions"], peaks["intensities"])],
            "summary": summary,
            "artifacts": {
                "pattern_plot": url_pattern,
                "peak_table_csv": url_csv,
            },
            "timing_sec": round(time.time() - t0, 1),
        }

        url_json = export_bytes(folder, json.dumps(report, indent=2, default=str).encode("utf-8"),
                                "parse_report.json")
        report["artifacts"]["report_json"] = url_json

        return report

    except Exception as e:
        logger.exception("parse_xrd failed")
        return {"ok": False, "reason": str(e)}


# =============================================================================
#  Tool 2: identify_xrd  —  Full pipeline (parse + search + match + impurity)
# =============================================================================
@mcp.tool()
def identify_xrd(
    file_id: str = "",
    file_path: str = "",
    composition_hint: str = "",
    reactants: str = "",
    wavelength: str = "CuKa",
    two_theta_min: float = 10.0,
    two_theta_max: float = 80.0,
    database: str = "cod",
    peak_height_pct: float = 5.0,
    peak_prominence_pct: float = 3.0,
) -> Dict[str, Any]:
    """
    Full XRD analysis: phase identification, purity, impurity detection.

    Upload a diffraction data file (.xy, .brml, .raw) and get back identified
    crystalline phases, match quality (Rwp), purity estimate, and annotated plots.

    Provide EITHER file_id (from OWUI upload) OR file_path.

    Args:
        file_id: OWUI file ID from upload.
        file_path: Direct filesystem path to data file.
        composition_hint: Expected formula(s) (e.g. "CsPbBr3" or "TiO2,SrTiO3").
            Strongly recommended — narrows database search.
        reactants: Comma-separated reactant formulas for impurity detection
            (e.g. "CsBr,PbBr2").
        wavelength: X-ray source. Default "CuKa". Options: CuKa, CuKa1, MoKa, etc.
        two_theta_min: Start of 2θ range (degrees). Default 10.
        two_theta_max: End of 2θ range (degrees). Default 80.
        database: "cod" (default), "mp" (Materials Project), or "both".
        peak_height_pct: Peak detection height threshold as % of max. Default 5.
        peak_prominence_pct: Peak detection prominence threshold as % of max. Default 3.

    Returns:
        dict with matches, purity, impurities, plots, and summary.
    """
    try:
        t0 = time.time()
        folder = _new_folder()
        two_theta_range = (two_theta_min, two_theta_max)

        # 1. Load + parse
        exp_tt, exp_int, fname, io_info = _load_and_parse(
            file_id, file_path, two_theta_min, two_theta_max)

        # 2. Detect peaks
        peaks = detect_peaks(
            exp_tt, exp_int,
            height_pct=peak_height_pct,
            prominence_pct=peak_prominence_pct,
        )
        logger.info(f"Detected {peaks['count']} peaks")

        # 3. Search databases
        candidates = _search_databases(composition_hint, database, wavelength, two_theta_range)

        # 4. Phase matching
        matches = analyze_phases(
            exp_tt, exp_int, candidates, peaks["positions"], two_theta_range,
        )

        # 5. Determine unassigned peaks
        assigned_positions = set()
        for m in matches:
            cid = m.get("cod_id") or m.get("mp_id", "")
            for c in candidates:
                c_id = c.get("cod_id") or c.get("mp_id", "")
                if c_id == cid:
                    ref_peaks = c.get("two_theta", [])
                    for ep in peaks["positions"]:
                        for rp in ref_peaks:
                            if abs(ep - rp) <= 0.3:
                                assigned_positions.add(ep)
                                break
                    break

        unassigned = [p for p in peaks["positions"] if p not in assigned_positions]

        # 6. Purity estimate
        if peaks["count"] > 0 and matches:
            purity = (1.0 - len(unassigned) / peaks["count"]) * 100.0
            purity = max(0, min(100, purity))
        elif not matches:
            purity = 0.0
        else:
            purity = 100.0

        # 7. Impurity detection
        reactant_list = [r.strip() for r in reactants.split(",") if r.strip()] if reactants else []
        impurities = []
        if unassigned and reactant_list:
            impurities = check_impurities(
                unassigned, reactant_list, wavelength, two_theta_range,
            )

        # 8. Plots
        artifacts = {}
        url_pattern = plot_pattern(exp_tt, exp_int, peaks, folder)
        artifacts["pattern_plot"] = url_pattern

        if matches and candidates:
            url_comp = plot_comparison(exp_tt, exp_int, matches, candidates, folder)
            artifacts["comparison_plot"] = url_comp

        # 9. Peak table CSV
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["2theta_deg", "intensity", "assigned_phase"])
        for pos, intens in zip(peaks["positions"], peaks["intensities"]):
            phase = ""
            if pos in assigned_positions and matches:
                phase = matches[0]["phase"]
            writer.writerow([f"{pos:.3f}", f"{intens:.1f}", phase])
        url_csv = export_bytes(folder, csv_buf.getvalue().encode("utf-8"), "peaks.csv")
        artifacts["peak_table_csv"] = url_csv

        # 10. Summary
        summary = _generate_summary(matches, peaks, unassigned, impurities, purity)

        # 11. Full report
        report = {
            "ok": True,
            "file": {"name": fname, "points": len(exp_tt),
                      "two_theta_range": [float(exp_tt[0]), float(exp_tt[-1])]},
            "matches": matches,
            "purity_estimate": round(purity, 1),
            "peak_count": peaks["count"],
            "unassigned_peaks": [round(p, 2) for p in unassigned],
            "possible_impurities": impurities,
            "artifacts": artifacts,
            "summary": summary,
            "timing_sec": round(time.time() - t0, 1),
        }

        url_json = export_bytes(
            folder, json.dumps(report, indent=2, default=str).encode("utf-8"),
            "xrd_report.json",
        )
        report["artifacts"]["report_json"] = url_json

        return _slim_response(report)

    except Exception as e:
        logger.exception("identify_xrd failed")
        return {"ok": False, "reason": str(e)}


# Keep backward compatibility: alias the old name
@mcp.tool()
def analyze_xrd(
    file_id: str = "",
    file_path: str = "",
    composition_hint: str = "",
    reactants: str = "",
    wavelength: str = "CuKa",
    two_theta_min: float = 10.0,
    two_theta_max: float = 80.0,
    database: str = "cod",
    peak_height_pct: float = 5.0,
    peak_prominence_pct: float = 3.0,
) -> Dict[str, Any]:
    """
    [DEPRECATED — use identify_xrd] Full XRD analysis pipeline.
    Kept for backward compatibility with existing SKILL.md and instrument_skills routing.
    """
    return identify_xrd(
        file_id=file_id, file_path=file_path,
        composition_hint=composition_hint, reactants=reactants,
        wavelength=wavelength, two_theta_min=two_theta_min,
        two_theta_max=two_theta_max, database=database,
        peak_height_pct=peak_height_pct, peak_prominence_pct=peak_prominence_pct,
    )


# =============================================================================
#  Tool 3: search_xrd_ref  —  Search reference pattern for a material
# =============================================================================
@mcp.tool()
def search_xrd_ref(
    formula: str,
    wavelength: str = "CuKa",
    two_theta_min: float = 10.0,
    two_theta_max: float = 80.0,
    database: str = "both",
) -> Dict[str, Any]:
    """
    Search crystallographic databases for a material's theoretical XRD pattern.

    Returns reference stick patterns (2θ positions + intensities + hkl indices)
    WITHOUT requiring an experimental data file. Useful for:
    - Looking up expected peak positions before a measurement
    - Checking what a specific phase's pattern should look like
    - Comparing multiple polymorphs of the same compound

    Args:
        formula: Chemical formula to search (e.g. "CsPbBr3", "TiO2").
            Can be comma-separated for multiple materials: "CsPbBr3,CsBr,PbBr2"
        wavelength: X-ray source. Default "CuKa".
        two_theta_min: Start of 2θ range. Default 10.
        two_theta_max: End of 2θ range. Default 80.
        database: "cod", "mp", or "both" (default).

    Returns:
        dict with reference patterns, crystal info, and stick-pattern plot.
    """
    try:
        t0 = time.time()
        folder = _new_folder()
        two_theta_range = (two_theta_min, two_theta_max)

        candidates = _search_databases(formula, database, wavelength, two_theta_range)

        if not candidates:
            return {
                "ok": False,
                "reason": f"No reference patterns found for '{formula}' in {database}. "
                          "Check the formula or try database='both'.",
            }

        # Build clean results
        results = []
        for c in candidates:
            if not c.get("two_theta"):
                continue
            wl = WAVELENGTHS.get(wavelength, 1.54184)
            entry = {
                "formula": c.get("formula", ""),
                "sg": c.get("sg", ""),
                "polymorph": c.get("polymorph", ""),
                "cod_id": c.get("cod_id", ""),
                "mp_id": c.get("mp_id", ""),
                "two_theta": c["two_theta"],
                "intensity": c["intensity"],
                "hkls": c.get("hkls", []),
                "d_spacings": c.get("d_spacings", []),
                "n_peaks": len(c["two_theta"]),
            }
            if c.get("a"):
                entry["cell"] = {
                    "a": c.get("a"), "b": c.get("b"), "c": c.get("c"),
                    "alpha": c.get("alpha"), "beta": c.get("beta"), "gamma": c.get("gamma"),
                }
            results.append(entry)

        # Plot
        url_plot = plot_reference(results, folder, two_theta_range)

        # CSV export of all peaks
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["formula", "space_group", "polymorph", "source_id",
                          "2theta_deg", "intensity_pct", "hkl", "d_spacing_A"])
        for r in results:
            src_id = r.get("cod_id") or r.get("mp_id", "")
            hkls = r.get("hkls", [])
            d_spac = r.get("d_spacings", [])
            for j, (tt, ii) in enumerate(zip(r["two_theta"], r["intensity"])):
                hkl = hkls[j] if j < len(hkls) else ""
                d = d_spac[j] if j < len(d_spac) else ""
                writer.writerow([r["formula"], r["sg"], r.get("polymorph", ""),
                                  src_id, f"{tt:.4f}", f"{ii:.1f}", hkl,
                                  f"{d:.4f}" if d else ""])
        url_csv = export_bytes(folder, csv_buf.getvalue().encode("utf-8"), "references.csv")

        summary = (
            f"Found {len(results)} reference pattern(s) for '{formula}': "
            + "; ".join(
                f"{r['formula']} ({r['sg']}{', ' + r['polymorph'] if r['polymorph'] else ''}, "
                f"{r['n_peaks']} peaks, {r.get('cod_id') or r.get('mp_id', '')})"
                for r in results
            )
        )

        report = {
            "ok": True,
            "query": formula,
            "database": database,
            "wavelength": wavelength,
            "results": results,
            "summary": summary,
            "artifacts": {
                "reference_plot": url_plot,
                "reference_csv": url_csv,
            },
            "timing_sec": round(time.time() - t0, 1),
        }

        url_json = export_bytes(folder, json.dumps(report, indent=2, default=str).encode("utf-8"),
                                "reference_report.json")
        report["artifacts"]["report_json"] = url_json

        # Slim for LLM
        slim = {
            "ok": True,
            "query": formula,
            "count": len(results),
            "results": [
                {k: v for k, v in r.items()
                 if k not in ("two_theta", "intensity", "hkls", "d_spacings")}
                for r in results
            ],
            "summary": summary,
            "artifacts": report["artifacts"],
            "timing_sec": report["timing_sec"],
        }
        return slim

    except Exception as e:
        logger.exception("search_xrd_ref failed")
        return {"ok": False, "reason": str(e)}


# =============================================================================
#  Tool 4: export_xrd_origin  —  Export processed data for OriginLab
# =============================================================================
@mcp.tool()
def export_xrd_origin(
    file_id: str = "",
    file_path: str = "",
    wavelength: str = "CuKa",
    two_theta_min: float = 10.0,
    two_theta_max: float = 80.0,
    include_background: bool = True,
    include_peak_markers: bool = True,
    peak_height_pct: float = 5.0,
    peak_prominence_pct: float = 3.0,
) -> Dict[str, Any]:
    """
    Export processed XRD data as Origin-ready CSV for manual continuation in OriginLab.

    Creates a multi-column CSV with metadata header that Origin can import directly
    via File > Import > CSV. Columns include 2θ, raw intensity, background, and
    optionally peak marker flags.

    The CSV is designed for the OriginMCP server's import workflow:
    1. This tool exports the CSV to the fileserver
    2. The model can then call spec.import_data to load it into an OPJ project
    3. Or the user downloads and opens it manually in Origin

    Args:
        file_id: OWUI file ID from upload.
        file_path: Direct filesystem path to data file.
        wavelength: X-ray source. Default "CuKa".
        two_theta_min: Start of 2θ range. Default 10.
        two_theta_max: End of 2θ range. Default 80.
        include_background: Include estimated background column. Default True.
        include_peak_markers: Include peak marker column (1=peak, 0=not). Default True.
        peak_height_pct: Peak detection threshold as % of max. Default 5.
        peak_prominence_pct: Peak detection prominence as % of max. Default 3.

    Returns:
        dict with download URL for the Origin-ready CSV and metadata.
    """
    try:
        t0 = time.time()
        folder = _new_folder()

        exp_tt, exp_int, fname, io_info = _load_and_parse(
            file_id, file_path, two_theta_min, two_theta_max)

        # Peak detection (needed for markers and background)
        peaks = detect_peaks(
            exp_tt, exp_int,
            height_pct=peak_height_pct,
            prominence_pct=peak_prominence_pct,
        )

        wl = WAVELENGTHS.get(wavelength, 1.54184)

        # Build CSV with Origin-compatible metadata header
        lines = []
        # Origin-style comment header (lines starting with # are skipped on import)
        lines.append(f"# XRD Data Export for OriginLab")
        lines.append(f"# Source: {fname}")
        lines.append(f"# Exported: {datetime.datetime.now().isoformat()}")
        lines.append(f"# Wavelength: {wavelength} ({wl:.5f} Å)")
        lines.append(f"# Range: {two_theta_min:.1f}° - {two_theta_max:.1f}°")
        lines.append(f"# Points: {len(exp_tt)}")
        lines.append(f"# Peaks detected: {peaks['count']}")
        lines.append(f"# FORTHought XRD Server v2.0")
        lines.append(f"#")

        # Column headers (Origin reads these)
        headers = ["2theta_deg", "Intensity_CPS", "d_spacing_A"]
        if include_background:
            headers.append("Background_CPS")
            headers.append("Subtracted_CPS")
        if include_peak_markers:
            headers.append("Peak_Flag")

        lines.append(",".join(headers))

        # Build peak set for marker column
        peak_indices = set(peaks.get("indices", []))
        bg = np.array(peaks.get("background", np.zeros_like(exp_int)))
        if len(bg) != len(exp_int):
            bg = np.zeros_like(exp_int)

        for i, (tt, intensity) in enumerate(zip(exp_tt, exp_int)):
            d = wl / (2 * np.sin(np.radians(tt / 2))) if tt > 0 else 0
            row = [f"{tt:.6f}", f"{intensity:.4f}", f"{d:.4f}"]
            if include_background:
                row.append(f"{bg[i]:.4f}")
                row.append(f"{max(0, intensity - bg[i]):.4f}")
            if include_peak_markers:
                row.append("1" if i in peak_indices else "0")
            lines.append(",".join(row))

        csv_data = "\n".join(lines)
        csv_fname = os.path.splitext(fname)[0] + "_origin.csv"
        url_csv = export_bytes(folder, csv_data.encode("utf-8"), csv_fname)

        # Also export a metadata JSON for programmatic import
        meta = {
            "source_file": fname,
            "wavelength": wavelength,
            "wavelength_A": wl,
            "two_theta_range": [two_theta_min, two_theta_max],
            "n_points": len(exp_tt),
            "n_peaks": peaks["count"],
            "columns": headers,
            "peak_positions": peaks["positions"],
            "peak_intensities": peaks["intensities"],
            "origin_import_hint": {
                "skip_header_lines": 9,
                "delimiter": ",",
                "x_column": "2theta_deg",
                "y_column": "Intensity_CPS",
                "suggested_graph_type": "line",
                "x_label": "2θ (°)",
                "y_label": "Intensity (CPS)",
            },
        }
        url_meta = export_bytes(folder, json.dumps(meta, indent=2).encode("utf-8"),
                                os.path.splitext(fname)[0] + "_origin_meta.json")

        return {
            "ok": True,
            "file": fname,
            "origin_csv": url_csv,
            "origin_metadata": url_meta,
            "columns": headers,
            "n_points": len(exp_tt),
            "n_peaks": peaks["count"],
            "summary": (
                f"Exported {len(exp_tt)} points from {fname} as Origin-ready CSV. "
                f"Columns: {', '.join(headers)}. "
                f"{peaks['count']} peaks flagged. "
                f"Download: {url_csv}"
            ),
            "timing_sec": round(time.time() - t0, 1),
        }

    except Exception as e:
        logger.exception("export_xrd_origin failed")
        return {"ok": False, "reason": str(e)}


# =============================================================================
#  Server entry point
# =============================================================================
if __name__ == "__main__":
    _mode = os.getenv("MODE", "stdio").lower()
    logger.info(
        f"Starting XRD analysis server v2.0 (mode={_mode}, "
        f"pymatgen={'yes' if HAS_PYMATGEN else 'no'}, "
        f"mp-api={'yes' if HAS_MP_API else 'no'}, "
        f"MP_KEY={'set' if MP_API_KEY else 'unset'})"
    )
    if _mode == "http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")