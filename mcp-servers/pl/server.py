# pl/server.py  --  PL Experimental Planning & Virtual Lab Assistant
# Author:  Marios Adamidis (FORTHought Lab)
# Version: 1.0.0
#
# Purpose: recommend laser, filters, and optical components for
# photoluminescence experiments. Also handles substrate enhancement,
# SHG/THG planning, strain analysis, PLE planning, imaging modes,
# nonlinear imaging, and valley polarization experiments.
#
#
#
#   - MoS2 B-exciton: added absorption position (600 nm / 2.05 eV) per
#     Katrisioti et al. arXiv:2504.03264 and Gerber et al. PRB 2019
#   - PLE tool: near-to-far-field offset warning for nanostructures,
#     dark-field spectroscopy recommendation (Katrisioti Fig. 2c,d)
#   - Auger saturation awareness in substrate enhancement and PLE tools
#     (exciton density >1e9 cm-2 limits PL gain to ~3x despite ~8x |E|^2)
#   - Imaging tool: added nonlinear_map mode for SHG/THG imaging
#   - Fresnel tool: SOI geometry note and multi-position awareness
#   - Added SnS to PL_MATERIALS_DB (was only in NONLINEAR_DB)
# NOT for data analysis — use the
# Spectroscopy (Origin/spec.*) tools for that.
#
# Architecture: follows server_xrd.py / micro.py patterns exactly
#   - Materials Project API for bandgap lookups (same key)
#   - Curated PL materials database for emission specifics
#   - Curated filter/laser catalog
#   - FastMCP HTTP transport
#   - Structured {ok:false} on all failures (never crash)
#   - Spectral sketch PNG artifacts via matplotlib

import os
import io
import re
import json
import math
import time
import uuid
import logging
import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mcp.server.fastmcp import FastMCP

# -- Optional: Materials Project API for bandgap lookups ----------------------
try:
    from mp_api.client import MPRester
    HAS_MP_API = True
except ImportError:
    HAS_MP_API = False

# =============================================================================
#  Logging
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [pl] %(message)s")
logger = logging.getLogger("pl")

# =============================================================================
#  Config (matches micro.py / xrd env vars)
# =============================================================================
EXPORT_DIR = os.getenv("FILE_EXPORT_DIR", "/data/files").rstrip("/")
BASE_URL = os.getenv("FILE_EXPORT_BASE_URL", "http://localhost:8084/files").rstrip("/")
os.makedirs(EXPORT_DIR, exist_ok=True)

MP_API_KEY = os.getenv("MP_API_KEY", "")

mcp = FastMCP(
    name="pl",
    host=os.getenv("MCP_HTTP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_HTTP_PORT", "9010")),
)


# =============================================================================
#  CURATED DATABASES
# =============================================================================

# ---------------------------------------------------------------------------
#  Laser catalog — common lasers found in PL labs
# ---------------------------------------------------------------------------
LASER_DB: List[Dict[str, Any]] = [
    {"id": "heCd_325",   "name": "HeCd",             "wavelength_nm": 325.0, "energy_eV": 3.815, "type": "CW",    "notes": "UV line; GaN, ZnO, wide-gap nitrides"},
    {"id": "diode_375",  "name": "Diode 375 nm",     "wavelength_nm": 375.0, "energy_eV": 3.306, "type": "CW",    "notes": "Near-UV; wide-gap semiconductors, some perovskites"},
    {"id": "diode_405",  "name": "Diode 405 nm",     "wavelength_nm": 405.0, "energy_eV": 3.061, "type": "CW",    "notes": "Violet; perovskites, QDs, organic emitters"},
    {"id": "argon_488",  "name": "Ar-ion 488 nm",    "wavelength_nm": 488.0, "energy_eV": 2.541, "type": "CW",    "notes": "Blue-green; GaN defects, QDs, TMDs"},
    {"id": "ndyag_532",  "name": "Nd:YAG 2ω (532)",  "wavelength_nm": 532.0, "energy_eV": 2.331, "type": "CW/pulsed", "notes": "Green; general purpose — TMDs, perovskites, Si QDs, most visible-gap materials"},
    {"id": "hene_633",   "name": "HeNe 633 nm",      "wavelength_nm": 632.8, "energy_eV": 1.959, "type": "CW",    "notes": "Red; near-resonant MoS2 A-exciton, narrow-gap materials"},
    {"id": "diode_660",  "name": "Diode 660 nm",     "wavelength_nm": 660.0, "energy_eV": 1.879, "type": "CW",    "notes": "Red; alternative to HeNe for TMDs"},
    {"id": "diode_785",  "name": "Diode 785 nm",     "wavelength_nm": 785.0, "energy_eV": 1.580, "type": "CW",    "notes": "NIR; Raman without PL interference, NIR emitters"},
    {"id": "tisaph",     "name": "Ti:Sapphire",      "wavelength_nm": 800.0, "energy_eV": 1.550, "type": "Pulsed (fs)", "notes": "700–1000 nm tunable; time-resolved PL, two-photon excitation"},
    {"id": "diode_980",  "name": "Diode 980 nm",     "wavelength_nm": 980.0, "energy_eV": 1.265, "type": "CW",    "notes": "NIR; upconversion materials, Er-doped systems"},
]

# ---------------------------------------------------------------------------
#  Filter catalog — common interference filters (Thorlabs/Semrock style)
# ---------------------------------------------------------------------------
FILTER_DB: List[Dict[str, Any]] = [
    # Long-pass (edge) filters
    {"id": "LP340",  "type": "long_pass", "cut_on_nm": 340,  "label": "LP340",  "od_below": 6, "notes": "UV edge; for 325 nm laser"},
    {"id": "LP400",  "type": "long_pass", "cut_on_nm": 400,  "label": "LP400",  "od_below": 6, "notes": "Passes visible; blocks UV lasers"},
    {"id": "LP425",  "type": "long_pass", "cut_on_nm": 425,  "label": "LP425",  "od_below": 6, "notes": "For 405 nm laser"},
    {"id": "LP450",  "type": "long_pass", "cut_on_nm": 450,  "label": "LP450",  "od_below": 6, "notes": "For 405 nm laser; wider window"},
    {"id": "LP500",  "type": "long_pass", "cut_on_nm": 500,  "label": "LP500",  "od_below": 6, "notes": "For 488 nm laser"},
    {"id": "LP514",  "type": "long_pass", "cut_on_nm": 514,  "label": "LP514",  "od_below": 6, "notes": "For 488/514 nm laser"},
    {"id": "LP550",  "type": "long_pass", "cut_on_nm": 550,  "label": "LP550",  "od_below": 6, "notes": "Standard for 532 nm laser"},
    {"id": "LP570",  "type": "long_pass", "cut_on_nm": 570,  "label": "LP570",  "od_below": 6, "notes": "For 532 nm; blocks Raman too"},
    {"id": "LP600",  "type": "long_pass", "cut_on_nm": 600,  "label": "LP600",  "od_below": 6, "notes": "Passes red/NIR only"},
    {"id": "LP650",  "type": "long_pass", "cut_on_nm": 650,  "label": "LP650",  "od_below": 6, "notes": "For 633 nm laser"},
    {"id": "LP700",  "type": "long_pass", "cut_on_nm": 700,  "label": "LP700",  "od_below": 6, "notes": "For 660 nm laser"},
    {"id": "LP800",  "type": "long_pass", "cut_on_nm": 800,  "label": "LP800",  "od_below": 6, "notes": "For 785 nm laser"},
    {"id": "LP830",  "type": "long_pass", "cut_on_nm": 830,  "label": "LP830",  "od_below": 6, "notes": "For 785 nm; wider margin"},
    {"id": "LP850",  "type": "long_pass", "cut_on_nm": 850,  "label": "LP850",  "od_below": 6, "notes": "NIR pass"},
    # Notch filters
    {"id": "NF325",  "type": "notch",     "center_nm": 325,  "width_nm": 25, "label": "Notch 325",  "od_at_center": 6, "notes": "Blocks HeCd 325 nm"},
    {"id": "NF405",  "type": "notch",     "center_nm": 405,  "width_nm": 25, "label": "Notch 405",  "od_at_center": 6, "notes": "Blocks 405 nm diode"},
    {"id": "NF488",  "type": "notch",     "center_nm": 488,  "width_nm": 25, "label": "Notch 488",  "od_at_center": 6, "notes": "Blocks Ar-ion 488 nm"},
    {"id": "NF532",  "type": "notch",     "center_nm": 532,  "width_nm": 25, "label": "Notch 532",  "od_at_center": 6, "notes": "Blocks Nd:YAG 532 nm"},
    {"id": "NF633",  "type": "notch",     "center_nm": 633,  "width_nm": 25, "label": "Notch 633",  "od_at_center": 6, "notes": "Blocks HeNe 633 nm"},
    {"id": "NF785",  "type": "notch",     "center_nm": 785,  "width_nm": 25, "label": "Notch 785",  "od_at_center": 6, "notes": "Blocks 785 nm diode"},
    # Band-pass filters
    {"id": "BP370_10", "type": "band_pass", "center_nm": 370, "width_nm": 10, "label": "BP370/10", "notes": "GaN band-edge isolation"},
    {"id": "BP520_20", "type": "band_pass", "center_nm": 520, "width_nm": 20, "label": "BP520/20", "notes": "CsPbBr3 PL isolation"},
    {"id": "BP550_40", "type": "band_pass", "center_nm": 550, "width_nm": 40, "label": "BP550/40", "notes": "Green emission isolation"},
    {"id": "BP620_20", "type": "band_pass", "center_nm": 620, "width_nm": 20, "label": "BP620/20", "notes": "MoS2 B-exciton isolation"},
    {"id": "BP670_30", "type": "band_pass", "center_nm": 670, "width_nm": 30, "label": "BP670/30", "notes": "MoS2 A-exciton isolation"},
    {"id": "BP700_40", "type": "band_pass", "center_nm": 700, "width_nm": 40, "label": "BP700/40", "notes": "Red PL isolation"},
    {"id": "BP800_40", "type": "band_pass", "center_nm": 800, "width_nm": 40, "label": "BP800/40", "notes": "NIR PL isolation"},
    # Dichroic mirrors
    {"id": "DM350",  "type": "dichroic",  "cut_on_nm": 350,  "label": "DM350",  "notes": "Reflects <350, transmits >350; UV excitation"},
    {"id": "DM425",  "type": "dichroic",  "cut_on_nm": 425,  "label": "DM425",  "notes": "For 405 nm laser"},
    {"id": "DM505",  "type": "dichroic",  "cut_on_nm": 505,  "label": "DM505",  "notes": "For 488 nm laser"},
    {"id": "DM550",  "type": "dichroic",  "cut_on_nm": 550,  "label": "DM550",  "notes": "For 532 nm laser"},
    {"id": "DM650",  "type": "dichroic",  "cut_on_nm": 650,  "label": "DM650",  "notes": "For 633 nm laser"},
    {"id": "DM700",  "type": "dichroic",  "cut_on_nm": 700,  "label": "DM700",  "notes": "For 660 nm laser"},
    {"id": "DM800",  "type": "dichroic",  "cut_on_nm": 800,  "label": "DM800",  "notes": "For 785 nm laser"},
    # Neutral density filters
    {"id": "ND05",   "type": "nd",        "od": 0.5,  "label": "ND 0.5", "notes": "31.6% transmission"},
    {"id": "ND10",   "type": "nd",        "od": 1.0,  "label": "ND 1.0", "notes": "10% transmission"},
    {"id": "ND20",   "type": "nd",        "od": 2.0,  "label": "ND 2.0", "notes": "1% transmission"},
    {"id": "ND30",   "type": "nd",        "od": 3.0,  "label": "ND 3.0", "notes": "0.1% transmission"},
    {"id": "ND40",   "type": "nd",        "od": 4.0,  "label": "ND 4.0", "notes": "0.01% transmission"},
]

# ---------------------------------------------------------------------------
#  Curated PL materials database — emission specifics not in Materials Project
# ---------------------------------------------------------------------------
# Key: lowercase formula (or common name)
# Values: list of emission entries (a material can have multiple PL bands)
PL_MATERIALS_DB: Dict[str, Dict[str, Any]] = {
    # ===== MATERIALS DATABASE =====
    # v3.0: Added MoS2 trion (X-), absorption peaks, nanoantenna_notes.
    #        Added SnS PL entry. B-exciton absorption at 600 nm (2.05 eV).
    # Verified against: Materials Project API, Katrisioti et al. arXiv:2504.03264,
    #   Kourmoulakis APL 2023, Katsipoulaki Adv. Opt. Mater. 2025, Maragkakis 2024
    # TMDs: Emission peaks verified via PMC, ACS Nano, Nano Lett, Sci Rep, PRX papers
    # Perovskites: Verified via Adv Funct Mater 2020 review, AIP Advances 2018, Sci Rep 2019
    # Bandgaps: Cross-checked with MP DFT values (PBE underestimates confirmed)
    # Raman lines: Verified for MoS2 (384/404), MoSe2 (242/287), WS2 (355/418)
    # FWHM values: Typical RT values — actual samples will vary with quality/substrate/T
    # ===== 2D Transition Metal Dichalcogenides =====
    "mos2": {
        "name": "Molybdenum disulfide",
        "formula": "MoS2",
        "category": "2D TMD",
        "gap_ev": 1.88, "gap_type": "direct (monolayer)",
        "emission": [
            {"label": "A exciton",  "center_nm": 670, "center_ev": 1.85, "fwhm_nm": 20, "notes": "Dominant in monolayer; trion shoulder ~1.82 eV. 660–680 nm range. Ref: PMC 5278406, PMC 9419104, ScienceDirect 2020"},
            {"label": "Trion (X-)", "center_nm": 683, "center_ev": 1.82, "fwhm_nm": 25, "notes": "Negatively charged trion; ~30 meV below X0. Visible in n-doped samples. Ref: Mak et al. Nat. Mater. 2013, Katrisioti et al. 2025 (Fig. 2a)"},
            {"label": "B exciton",  "center_nm": 620, "center_ev": 2.00, "fwhm_nm": 30, "notes": "Spin-orbit split; PL emission 605–630 nm. B/A ratio = quality indicator. Ref: McCreary arXiv:1812.01545"},
        ],
        "absorption": [
            {"label": "A exciton (abs)", "center_nm": 660, "center_ev": 1.88, "notes": "Absorption peak; Stokes-shifted from PL by ~30 meV"},
            {"label": "B exciton (abs)", "center_nm": 600, "center_ev": 2.07, "notes": "B-exciton absorption at 600 nm (2.05 eV). Katrisioti et al.: PLE avoids tuning near this. Gerber et al. PRB 2019"},
        ],
        "recommended_lasers": ["ndyag_532", "hene_633"],
        "warnings": ["532 nm is below single-particle gap but resonantly excites excitons",
                      "Power < 100 μW for monolayer (damage/heating risk)",
                      "633 nm is near-resonant with A-exciton → sharper A peak, weak B",
                      "On nanostructures: PL enhancement limited to ~3x by Auger recombination despite higher absorption (Katrisioti et al. 2025)"],
        "raman_lines_cm1": [384, 404],  # E' ~384-387, A1' ~404-406. Katrisioti 2025: 386/404
        "substrate_notes": "SiO2/Si substrate may show weak broadband PL under UV excitation",
        "nanoantenna_notes": {
            "pl_enhancement": "~3x on Si nanoantennas (Mie-coupled), limited by Auger at high exciton density >1e9 cm-2",
            "raman_enhancement": "2–8x depending on excitation wavelength and pillar diameter",
            "shg_enhancement": "20–30x due to |E|^2 pump dependence + LDOS at SHG energy",
            "strain_on_pillars": "~0.3% biaxial tensile from conforming to 120 nm tall Si pillars (30 meV PL redshift)",
            "ref": "Katrisioti et al. arXiv:2504.03264 (2025)",
        },
    },
    "ws2": {
        "name": "Tungsten disulfide",
        "formula": "WS2",
        "category": "2D TMD",
        "gap_ev": 2.05, "gap_type": "direct (monolayer)",
        "emission": [
            {"label": "A exciton",  "center_nm": 630, "center_ev": 1.97, "fwhm_nm": 15, "notes": "Bright PL; varies 1.93–2.01 eV with substrate/strain (RT). Ref: McCreary Sci Rep 2016, Zhu Sci Rep 2015"},
            {"label": "B exciton",  "center_nm": 528, "center_ev": 2.35, "fwhm_nm": 30, "notes": "Near 532 nm laser — watch for overlap. Ref: Hill Nano Lett 2015 (A/B splitting ~0.4 eV)"},
        ],
        "recommended_lasers": ["ndyag_532", "diode_405"],
        "warnings": ["B exciton near 528 nm overlaps with 532 nm laser scatter",
                      "Use LP570 instead of LP550 if B-exciton isolation not needed",
                      "A-exciton PL varies 1.93–2.01 eV depending on substrate/strain/doping"],
        "raman_lines_cm1": [355, 418],  # E2g ~355, A1g ~418
    },
    "mose2": {
        "name": "Molybdenum diselenide",
        "formula": "MoSe2",
        "category": "2D TMD",
        "gap_ev": 1.55, "gap_type": "direct (monolayer)",
        "emission": [
            {"label": "A exciton",  "center_nm": 800, "center_ev": 1.55, "fwhm_nm": 25, "notes": "NIR; 1.52–1.57 eV range at RT. Ref: McCreary arXiv:1812.01545 (1.52 eV), PMC 11653426 (1.56 eV)"},
            {"label": "B exciton",  "center_nm": 725, "center_ev": 1.71, "fwhm_nm": 35, "notes": "~190 meV above A (valence band splitting). Ref: McCreary et al."},
        ],
        "recommended_lasers": ["hene_633", "ndyag_532"],
        "warnings": ["NIR emission — Si CCD sensitivity drops; verify detector QE at 800 nm",
                      "A-exciton position varies 1.52–1.57 eV depending on substrate/quality"],
        "raman_lines_cm1": [242, 287],  # A1g ~242, E' ~287. Ref: PMC 11653426
    },
    "wse2": {
        "name": "Tungsten diselenide",
        "formula": "WSe2",
        "category": "2D TMD",
        "gap_ev": 1.65, "gap_type": "direct (monolayer)",
        "emission": [
            {"label": "A exciton",  "center_nm": 750, "center_ev": 1.65, "fwhm_nm": 15, "notes": "Dark exciton ~50 meV below bright. Ref: Nat Comms 2023 (plasmonic upconversion), Sci Rep 2016 (excitonic states)"},
            {"label": "B exciton",  "center_nm": 600, "center_ev": 2.07, "fwhm_nm": 30, "notes": "Large spin-orbit splitting ~420 meV in W-based TMDs"},
        ],
        "recommended_lasers": ["ndyag_532", "hene_633"],
        "warnings": ["Dark exciton below A-exciton → anomalous T-dependence",
                      "Trion binding energy ~30 meV",
                      "Localized exciton states vanish above ~65 K (Sci Rep 2016)"],
        "raman_lines_cm1": [248, 251],  # E2g and A1g nearly degenerate
    },
    "mote2": {
        "name": "Molybdenum ditelluride",
        "formula": "MoTe2",
        "category": "2D TMD",
        "gap_ev": 1.10, "gap_type": "direct (monolayer, 2H phase)",
        "emission": [
            {"label": "A exciton",  "center_nm": 1130, "center_ev": 1.10, "fwhm_nm": 30, "notes": "SWIR — needs InGaAs detector. Ref: PRX 8, 031073 (2018)"},
            {"label": "B exciton",  "center_nm": 920,  "center_ev": 1.35, "fwhm_nm": 40, "notes": "Spin-orbit splitting ~250 meV"},
        ],
        "recommended_lasers": ["diode_785", "hene_633"],
        "warnings": ["Si CCD CANNOT detect this — need InGaAs detector",
                      "Material is air-sensitive; measure in inert atmosphere or encapsulate",
                      "2H (semiconducting) vs 1T' (semimetallic) phase — verify crystal structure"],
        "raman_lines_cm1": [172, 235, 290],
    },
    # ===== Halide Perovskites =====
    "cspbbr3": {
        "name": "Cesium lead bromide",
        "formula": "CsPbBr3",
        "category": "Halide perovskite",
        "gap_ev": 2.36, "gap_type": "direct",
        "emission": [
            {"label": "Band-edge PL", "center_nm": 520, "center_ev": 2.38, "fwhm_nm": 20, "notes": "Green; 510–525 nm range. FWHM 18–25 nm (NCs), broader for bulk/film. Ref: Opt Lett 2016 (QDs at RT), ScienceDirect 2017"},
        ],
        "recommended_lasers": ["diode_405"],
        "warnings": ["Photodegrades under prolonged UV exposure",
                      "Use minimum power; check for time-dependent intensity",
                      "PL peak redshifts with increasing concentration (self-absorption)"],
        "raman_lines_cm1": [72, 127, 310],
    },
    "cspbi3": {
        "name": "Cesium lead iodide",
        "formula": "CsPbI3",
        "category": "Halide perovskite",
        "gap_ev": 1.73, "gap_type": "direct",
        "emission": [
            {"label": "Band-edge PL", "center_nm": 710, "center_ev": 1.75, "fwhm_nm": 25, "notes": "Red emission; α-phase unstable at RT"},
        ],
        "recommended_lasers": ["ndyag_532", "diode_405"],
        "warnings": ["α-phase (black) → δ-phase (yellow) transition at RT",
                      "Measure quickly or at elevated T to maintain black phase"],
    },
    "cspbcl3": {
        "name": "Cesium lead chloride",
        "formula": "CsPbCl3",
        "category": "Halide perovskite",
        "gap_ev": 3.00, "gap_type": "direct",
        "emission": [
            {"label": "Band-edge PL", "center_nm": 410, "center_ev": 3.02, "fwhm_nm": 15, "notes": "Violet emission; needs UV excitation"},
        ],
        "recommended_lasers": ["heCd_325", "diode_375"],
        "warnings": ["Requires UV laser — 405 nm is too close to emission",
                      "Use LP340 + notch325 for clean spectrum"],
    },
    "mapbi3": {
        "name": "Methylammonium lead iodide",
        "formula": "CH3NH3PbI3 (MAPbI3)",
        "category": "Halide perovskite",
        "gap_ev": 1.60, "gap_type": "direct",
        "emission": [
            {"label": "Band-edge PL", "center_nm": 775, "center_ev": 1.60, "fwhm_nm": 30, "notes": "760–783 nm at RT. Ref: Adv Funct Mater 2020 (~775 nm), AIP Advances 2018 (775–781 nm)"},
        ],
        "recommended_lasers": ["ndyag_532", "hene_633"],
        "warnings": ["EXTREMELY moisture-sensitive — decomposes to PbI2",
                      "High power causes thermal degradation and flickering (ScienceDirect 2020)",
                      "Always use ND filters; measure in N2 if possible"],
    },
    "mapbbr3": {
        "name": "Methylammonium lead bromide",
        "formula": "CH3NH3PbBr3 (MAPbBr3)",
        "category": "Halide perovskite",
        "gap_ev": 2.30, "gap_type": "direct",
        "emission": [
            {"label": "Band-edge PL", "center_nm": 540, "center_ev": 2.30, "fwhm_nm": 20, "notes": "Green-yellow; relatively stable"},
        ],
        "recommended_lasers": ["diode_405", "ndyag_532"],
        "warnings": ["More stable than MAPbI3 but still moisture-sensitive"],
    },
    "fapbi3": {
        "name": "Formamidinium lead iodide",
        "formula": "HC(NH2)2PbI3 (FAPbI3)",
        "category": "Halide perovskite",
        "gap_ev": 1.53, "gap_type": "direct",
        "emission": [
            {"label": "Band-edge PL", "center_nm": 810, "center_ev": 1.53, "fwhm_nm": 35, "notes": "NIR; α-phase. Ref: Wright et al. (1.53 eV at RT, Sci Rep 2019)"},
        ],
        "recommended_lasers": ["ndyag_532", "hene_633"],
        "warnings": ["α-phase (black, 1.53 eV) vs δ-phase (yellow, ~2.4 eV)",
                      "NIR emission near Si CCD limit — verify detector QE at 810 nm"],
    },
    # ===== III-V & III-Nitrides =====
    "gan": {
        "name": "Gallium nitride",
        "formula": "GaN",
        "category": "III-Nitride",
        "gap_ev": 3.40, "gap_type": "direct",
        "emission": [
            {"label": "Band-edge (NBE)", "center_nm": 365, "center_ev": 3.40, "fwhm_nm": 5, "notes": "Sharp UV peak; quality indicator"},
            {"label": "Yellow defect band", "center_nm": 560, "center_ev": 2.21, "fwhm_nm": 150, "notes": "Broad; VGa-related defects; intensity = crystal quality"},
            {"label": "Blue defect band", "center_nm": 430, "center_ev": 2.88, "fwhm_nm": 100, "notes": "Donor-acceptor pair (DAP) transitions"},
        ],
        "recommended_lasers": ["heCd_325", "diode_375"],
        "warnings": ["Requires UV excitation (E > 3.4 eV)",
                      "NBE/defect ratio = crystal quality metric",
                      "LP340 for band-edge; LP400 for defect bands only"],
        "raman_lines_cm1": [533, 568, 734],
    },
    "zno": {
        "name": "Zinc oxide",
        "formula": "ZnO",
        "category": "Wide-gap semiconductor",
        "gap_ev": 3.37, "gap_type": "direct",
        "emission": [
            {"label": "UV band-edge", "center_nm": 380, "center_ev": 3.26, "fwhm_nm": 10, "notes": "Sharp; exciton emission"},
            {"label": "Green defect band", "center_nm": 520, "center_ev": 2.38, "fwhm_nm": 120, "notes": "Oxygen vacancies; broad"},
            {"label": "Orange/red defect", "center_nm": 620, "center_ev": 2.00, "fwhm_nm": 100, "notes": "Interstitial oxygen"},
        ],
        "recommended_lasers": ["heCd_325"],
        "warnings": ["Very similar to GaN setup — UV required",
                      "UV/visible ratio = crystal quality indicator"],
        "raman_lines_cm1": [99, 437, 583],
    },
    "gaas": {
        "name": "Gallium arsenide",
        "formula": "GaAs",
        "category": "III-V semiconductor",
        "gap_ev": 1.42, "gap_type": "direct",
        "emission": [
            {"label": "Band-edge PL", "center_nm": 870, "center_ev": 1.42, "fwhm_nm": 15, "notes": "NIR; standard Si CCD barely reaches this"},
        ],
        "recommended_lasers": ["hene_633", "diode_785"],
        "warnings": ["870 nm is at the edge of Si CCD sensitivity",
                      "InGaAs detector recommended for quantitative work"],
    },
    "inp": {
        "name": "Indium phosphide",
        "formula": "InP",
        "category": "III-V semiconductor",
        "gap_ev": 1.34, "gap_type": "direct",
        "emission": [
            {"label": "Band-edge PL", "center_nm": 925, "center_ev": 1.34, "fwhm_nm": 20, "notes": "NIR; needs InGaAs detector"},
        ],
        "recommended_lasers": ["hene_633", "diode_785"],
        "warnings": ["Si CCD cannot detect 925 nm — InGaAs required"],
    },
    # ===== Quantum Dots =====
    "cdse_qd": {
        "name": "CdSe quantum dots",
        "formula": "CdSe QDs",
        "category": "Quantum dots",
        "gap_ev": 2.10, "gap_type": "direct (size-dependent: 1.7–2.5 eV)",
        "emission": [
            {"label": "QD PL (size-dependent)", "center_nm": 590, "center_ev": 2.10, "fwhm_nm": 30, "notes": "Tunable 500–650 nm; FWHM = size distribution"},
        ],
        "recommended_lasers": ["diode_405", "ndyag_532"],
        "warnings": ["Emission wavelength depends on QD size — verify sample specs",
                      "FWHM indicates size distribution homogeneity"],
    },
    "si_qd": {
        "name": "Silicon quantum dots / nanocrystals",
        "formula": "Si QDs",
        "category": "Quantum dots",
        "gap_ev": 1.70, "gap_type": "direct (quantum-confined)",
        "emission": [
            {"label": "QD PL (size-dependent)", "center_nm": 730, "center_ev": 1.70, "fwhm_nm": 80, "notes": "Tunable 600–1000 nm; broader than CdSe"},
        ],
        "recommended_lasers": ["diode_405", "ndyag_532"],
        "warnings": ["Broad emission — FWHM ~80–150 nm typical",
                      "Surface oxidation shifts emission"],
    },
    "pbs_qd": {
        "name": "Lead sulfide quantum dots",
        "formula": "PbS QDs",
        "category": "Quantum dots",
        "gap_ev": 1.00, "gap_type": "direct (size-dependent: 0.4–1.5 eV)",
        "emission": [
            {"label": "QD PL", "center_nm": 1200, "center_ev": 1.03, "fwhm_nm": 100, "notes": "SWIR; InGaAs detector required"},
        ],
        "recommended_lasers": ["diode_785", "ndyag_532"],
        "warnings": ["SWIR emission — standard Si CCD cannot detect",
                      "Air-sensitive; handle in glovebox"],
    },
    # ===== Calibration & Reference =====
    "nv_diamond": {
        "name": "NV⁻ center in diamond",
        "formula": "Diamond (NV⁻)",
        "category": "Color center",
        "gap_ev": None, "gap_type": "defect state",
        "emission": [
            {"label": "Zero-phonon line (ZPL)", "center_nm": 637, "center_ev": 1.945, "fwhm_nm": 2, "notes": "Sharp ZPL; T-dependent"},
            {"label": "Phonon sideband", "center_nm": 700, "center_ev": 1.77, "fwhm_nm": 100, "notes": "Broad 640–800 nm; dominates at RT"},
        ],
        "recommended_lasers": ["ndyag_532"],
        "warnings": ["Use LP550 to block 532; both ZPL and PSB pass cleanly"],
    },
    "rhodamine_6g": {
        "name": "Rhodamine 6G dye",
        "formula": "Rhodamine 6G",
        "category": "Fluorescent dye (calibration)",
        "gap_ev": None, "gap_type": "molecular",
        "emission": [
            {"label": "Fluorescence", "center_nm": 560, "center_ev": 2.21, "fwhm_nm": 35, "notes": "Standard calibration dye; high QY"},
        ],
        "recommended_lasers": ["ndyag_532", "argon_488"],
        "warnings": ["Use as alignment/calibration standard",
                      "Photobleaches under prolonged exposure"],
    },
    "ruby": {
        "name": "Ruby (Cr:Al₂O₃)",
        "formula": "Cr:Al₂O₃",
        "category": "Calibration standard",
        "gap_ev": None, "gap_type": "d-d transition",
        "emission": [
            {"label": "R1 line", "center_nm": 694.3, "center_ev": 1.786, "fwhm_nm": 0.5, "notes": "Pressure calibration standard"},
            {"label": "R2 line", "center_nm": 692.9, "center_ev": 1.789, "fwhm_nm": 0.5, "notes": ""},
        ],
        "recommended_lasers": ["ndyag_532", "diode_405"],
        "warnings": ["R-line shift used for pressure calibration in DACs"],
    },
    # ===== Oxides =====
    "tio2": {
        "name": "Titanium dioxide",
        "formula": "TiO2",
        "category": "Metal oxide",
        "gap_ev": 3.20, "gap_type": "indirect (anatase: 3.2, rutile: 3.0)",
        "emission": [
            {"label": "Band-edge PL", "center_nm": 390, "center_ev": 3.18, "fwhm_nm": 20, "notes": "Weak; indirect gap suppresses PL"},
            {"label": "Defect/trap PL", "center_nm": 530, "center_ev": 2.34, "fwhm_nm": 100, "notes": "Broad green; oxygen vacancy related"},
        ],
        "recommended_lasers": ["heCd_325", "diode_375"],
        "warnings": ["Indirect gap → weak PL; high power or low T may be needed",
                      "Anatase vs rutile have different emission profiles"],
    },
    "sno2": {
        "name": "Tin dioxide",
        "formula": "SnO2",
        "category": "Metal oxide",
        "gap_ev": 3.60, "gap_type": "direct",
        "emission": [
            {"label": "UV band-edge", "center_nm": 345, "center_ev": 3.59, "fwhm_nm": 10, "notes": "Requires deep UV excitation"},
            {"label": "Defect band", "center_nm": 580, "center_ev": 2.14, "fwhm_nm": 120, "notes": "Broad yellow-orange; oxygen vacancy"},
        ],
        "recommended_lasers": ["heCd_325"],
        "warnings": ["Deep UV required for band-edge excitation"],
    },
    # ===== Organic/Polymer Emitters =====
    "alq3": {
        "name": "Tris(8-hydroxyquinolinato)aluminium",
        "formula": "Alq3",
        "category": "Organic emitter",
        "gap_ev": 2.70, "gap_type": "molecular (HOMO-LUMO)",
        "emission": [
            {"label": "Fluorescence", "center_nm": 520, "center_ev": 2.38, "fwhm_nm": 60, "notes": "Classic green OLED emitter"},
        ],
        "recommended_lasers": ["diode_405"],
        "warnings": ["Organic — photobleaches easily",
                      "Use minimum power; N2 atmosphere preferred"],
    },
    # ===== II-VI bulk =====
    "cds": {
        "name": "Cadmium sulfide",
        "formula": "CdS",
        "category": "II-VI semiconductor",
        "gap_ev": 2.42, "gap_type": "direct",
        "emission": [
            {"label": "Band-edge", "center_nm": 512, "center_ev": 2.42, "fwhm_nm": 10, "notes": "Sharp green; near gap"},
            {"label": "Trap emission", "center_nm": 600, "center_ev": 2.07, "fwhm_nm": 80, "notes": "Broad; surface/bulk traps"},
        ],
        "recommended_lasers": ["diode_405", "argon_488"],
        "warnings": ["Band-edge very close to 532 nm laser — use 405 nm instead"],
    },
    "cdse_bulk": {
        "name": "Cadmium selenide (bulk)",
        "formula": "CdSe",
        "category": "II-VI semiconductor",
        "gap_ev": 1.74, "gap_type": "direct",
        "emission": [
            {"label": "Band-edge", "center_nm": 713, "center_ev": 1.74, "fwhm_nm": 15, "notes": "Red; bulk value"},
        ],
        "recommended_lasers": ["ndyag_532", "hene_633"],
    },
    "zns": {
        "name": "Zinc sulfide",
        "formula": "ZnS",
        "category": "II-VI semiconductor",
        "gap_ev": 3.68, "gap_type": "direct",
        "emission": [
            {"label": "Band-edge", "center_nm": 337, "center_ev": 3.68, "fwhm_nm": 5, "notes": "Deep UV"},
            {"label": "Blue self-activated", "center_nm": 450, "center_ev": 2.76, "fwhm_nm": 50, "notes": "Donor-acceptor; common in ZnS:Cu,Al phosphors"},
        ],
        "recommended_lasers": ["heCd_325"],
        "warnings": ["Requires UV excitation"],
    },
    # ===== Group-IV Monochalcogenides =====
    "sns": {
        "name": "Tin(II) sulfide",
        "formula": "SnS",
        "category": "Group-IV monochalcogenide",
        "gap_ev": 1.32, "gap_type": "direct (few-layer), ~1.6 eV monolayer, indirect 1.07 eV bulk",
        "emission": [
            {"label": "Band-edge PL", "center_nm": 940, "center_ev": 1.32, "fwhm_nm": 40, "notes": "NIR; few-layer direct gap. Sarkar et al. Adv. Sci. 2023 (monolayer isolation)"},
            {"label": "Defect band", "center_nm": 1050, "center_ev": 1.18, "fwhm_nm": 80, "notes": "Broad sub-gap defect emission; common in exfoliated flakes"},
        ],
        "recommended_lasers": ["hene_633", "diode_785"],
        "warnings": ["NIR emission — InGaAs detector recommended for quantitative work",
                      "Air-sensitive: oxidizes over hours in ambient — measure quickly or encapsulate",
                      "Orthorhombic C2v symmetry: strongly in-plane anisotropic (armchair vs zigzag)",
                      "Non-centrosymmetric: SHG/THG active in all thicknesses (unlike D3h TMDs)"],
        "raman_lines_cm1": [95, 160, 192, 218],  # Pnma group modes
        "substrate_notes": "SiO2/Si works but NIR emission may have weak Fresnel enhancement",
    },
    # ===== Silicon =====
    "si": {
        "name": "Silicon",
        "formula": "Si",
        "category": "Elemental semiconductor",
        "gap_ev": 1.12, "gap_type": "indirect",
        "emission": [
            {"label": "Band-edge PL", "center_nm": 1130, "center_ev": 1.10, "fwhm_nm": 50, "notes": "Extremely weak at RT due to indirect gap; needs low T or very sensitive setup"},
        ],
        "recommended_lasers": ["ndyag_532", "diode_785"],
        "warnings": ["INDIRECT GAP → PL is extremely weak at room temperature",
                      "Low-temperature measurement strongly recommended",
                      "Si CCD cannot detect 1130 nm emission — InGaAs needed"],
    },
}

# Aliases for common name lookups
_PL_ALIASES: Dict[str, str] = {
    "molybdenum disulfide": "mos2", "molybdenum disulphide": "mos2", "mos₂": "mos2",
    "tungsten disulfide": "ws2", "tungsten disulphide": "ws2", "ws₂": "ws2",
    "molybdenum diselenide": "mose2", "mose₂": "mose2",
    "tungsten diselenide": "wse2", "wse₂": "wse2",
    "molybdenum ditelluride": "mote2", "mote₂": "mote2",
    "cesium lead bromide": "cspbbr3", "cspbbr₃": "cspbbr3",
    "cesium lead iodide": "cspbi3", "cspbi₃": "cspbi3",
    "cesium lead chloride": "cspbcl3", "cspbcl₃": "cspbcl3",
    "mapbi3": "mapbi3", "mapbi₃": "mapbi3", "methylammonium lead iodide": "mapbi3", "mapi": "mapbi3",
    "mapbbr3": "mapbbr3", "methylammonium lead bromide": "mapbbr3", "mapb": "mapbbr3",
    "fapbi3": "fapbi3", "formamidinium lead iodide": "fapbi3", "fapi": "fapbi3",
    "gallium nitride": "gan",
    "zinc oxide": "zno",
    "gallium arsenide": "gaas",
    "indium phosphide": "inp",
    "cadmium selenide quantum dots": "cdse_qd", "cdse qd": "cdse_qd", "cdse qds": "cdse_qd",
    "silicon quantum dots": "si_qd", "si qd": "si_qd", "si qds": "si_qd", "si nanocrystals": "si_qd",
    "lead sulfide quantum dots": "pbs_qd", "pbs qd": "pbs_qd",
    "nv center": "nv_diamond", "nv diamond": "nv_diamond", "nv-": "nv_diamond",
    "rhodamine 6g": "rhodamine_6g", "r6g": "rhodamine_6g", "rhodamine": "rhodamine_6g",
    "ruby": "ruby", "cr:al2o3": "ruby",
    "titanium dioxide": "tio2", "tio₂": "tio2",
    "tin dioxide": "sno2", "sno₂": "sno2",
    "alq3": "alq3", "tris(8-hydroxyquinolinato)aluminium": "alq3",
    "cadmium sulfide": "cds",
    "cadmium selenide": "cdse_bulk", "cdse": "cdse_bulk",
    "zinc sulfide": "zns",
    "silicon": "si",
    "tin sulfide": "sns", "tin(ii) sulfide": "sns", "tin monosulfide": "sns",
}

# =============================================================================
#  DATABASES: Refractive indices, strain gauges, valley polarization
# =============================================================================

# ---------------------------------------------------------------------------
#  Complex refractive index data for Fresnel calculations (Palik / Aspnes)
# ---------------------------------------------------------------------------
# Format: wavelength_nm -> (n, k)
_SI_REFINDEX: List[Tuple[float, float, float]] = [
    # (wavelength_nm, n, k) — from Aspnes & Studna 1983 / Palik handbook
    (300, 5.00, 4.24), (350, 5.60, 2.99), (400, 5.57, 0.39),
    (450, 4.67, 0.15), (500, 4.30, 0.073), (532, 4.15, 0.044),
    (550, 4.08, 0.032), (600, 3.94, 0.023), (633, 3.88, 0.018),
    (650, 3.85, 0.016), (700, 3.78, 0.011), (750, 3.72, 0.008),
    (800, 3.69, 0.006), (850, 3.67, 0.005), (900, 3.65, 0.004),
    (950, 3.64, 0.003), (1000, 3.62, 0.002), (1100, 3.58, 0.001),
]

def _si_refractive_index(wl_nm: float) -> complex:
    """Interpolate complex refractive index of Si at given wavelength."""
    wls = [r[0] for r in _SI_REFINDEX]
    ns = [r[1] for r in _SI_REFINDEX]
    ks = [r[2] for r in _SI_REFINDEX]
    n_interp = float(np.interp(wl_nm, wls, ns))
    k_interp = float(np.interp(wl_nm, wls, ks))
    return complex(n_interp, k_interp)

def _sio2_refractive_index(wl_nm: float) -> float:
    """Cauchy model for SiO2 refractive index."""
    wl_um = wl_nm / 1000.0
    return 1.4580 + 0.00354 / (wl_um ** 2)


# ---------------------------------------------------------------------------
#  Strain gauge factor database — verified against group publications
# ---------------------------------------------------------------------------
# Key: "material_mode" -> gauge factor and metadata
# Positive gauge factor = blueshift under compressive / redshift under tensile
# Units: meV/% for PL modes, cm-1/% for Raman modes
STRAIN_GAUGE_DB: Dict[str, Dict[str, Any]] = {
    # === MoS2 ===
    "mos2_PL_A_exciton": {
        "material": "MoS2", "mode": "PL_A_exciton",
        "gauge_biaxial_meV_per_pct": -100.0,
        "gauge_uniaxial_meV_per_pct": -72.0,
        "reference_peak_nm": 670, "reference_peak_eV": 1.85,
        "k_gamma_crossover_pct": 1.0,  # biaxial strain where K->Gamma indirect transition occurs
        "ref": "Conley et al. Nano Lett. 2013; Katrisioti et al. 2025 (30 meV -> 0.3%)",
    },
    "mos2_Raman_E_prime": {
        "material": "MoS2", "mode": "Raman_E_prime",
        "gauge_biaxial_cm1_per_pct": -4.0,
        "gauge_uniaxial_cm1_per_pct": -2.1,
        "reference_peak_cm1": 384,
        "ref": "Rice et al. PRB 2013",
    },
    "mos2_Raman_A1_prime": {
        "material": "MoS2", "mode": "Raman_A1_prime",
        "gauge_biaxial_cm1_per_pct": -0.8,
        "gauge_uniaxial_cm1_per_pct": -0.4,
        "reference_peak_cm1": 404,
        "ref": "Out-of-plane mode — strain-insensitive. Good for cross-check.",
    },
    # === WS2 === (verified from Kioseoglou group: Kourmoulakis et al. APL 2023, J. Phys. Chem. C 2023)
    "ws2_PL_A_exciton": {
        "material": "WS2", "mode": "PL_A_exciton",
        "gauge_biaxial_meV_per_pct": -130.0,  # Kourmoulakis APL 2023: -130 meV/% measured
        "gauge_uniaxial_meV_per_pct": -65.0,  # estimated ~half of biaxial
        "reference_peak_nm": 630, "reference_peak_eV": 1.97,
        "k_gamma_crossover_pct": 0.8,
        "ref": "Kourmoulakis et al. APL 123, 223103 (2023): -130 meV/% biaxial, verified up to 0.45%",
    },
    "ws2_Raman_E_prime": {
        "material": "WS2", "mode": "Raman_E_prime",
        "gauge_biaxial_cm1_per_pct": -3.4,
        "gauge_uniaxial_cm1_per_pct": -1.7,
        "reference_peak_cm1": 355,
        "ref": "Kourmoulakis et al. J. Phys. Chem. C 2023 (Gruneisen parameter matches bulk E2g)",
    },
    # === MoSe2 ===
    "mose2_PL_A_exciton": {
        "material": "MoSe2", "mode": "PL_A_exciton",
        "gauge_biaxial_meV_per_pct": -60.0,
        "gauge_uniaxial_meV_per_pct": -30.0,
        "reference_peak_nm": 800, "reference_peak_eV": 1.55,
        "k_gamma_crossover_pct": 1.5,
        "ref": "Island et al. Nanoscale 2016",
    },
    # === WSe2 ===
    "wse2_PL_A_exciton": {
        "material": "WSe2", "mode": "PL_A_exciton",
        "gauge_biaxial_meV_per_pct": -80.0,
        "gauge_uniaxial_meV_per_pct": -40.0,
        "reference_peak_nm": 750, "reference_peak_eV": 1.65,
        "k_gamma_crossover_pct": 1.0,
        "ref": "Desai et al. Adv. Mater. 2014; Schmidt et al. 2D Mater. 2016",
    },
    "wse2_Raman_E_prime": {
        "material": "WSe2", "mode": "Raman_E_prime",
        "gauge_biaxial_cm1_per_pct": -2.0,
        "gauge_uniaxial_cm1_per_pct": -1.0,
        "reference_peak_cm1": 248,
        "ref": "Sahin et al. PRB 2013",
    },
}


# ---------------------------------------------------------------------------
#  Valley polarization reference data
# ---------------------------------------------------------------------------
VALLEY_POL_DB: Dict[str, Dict[str, Any]] = {
    "mos2": {
        "material": "MoS2", "point_group": "D3h (monolayer)",
        "X0_eV": 1.85, "X0_nm": 670,
        "trion_binding_meV": 20, "trion_type": "X-",
        "spin_orbit_valence_meV": 150,
        "typical_Pc_RT_nonres_pct": (0, 5),  # range
        "typical_Pc_RT_nearres_pct": (5, 15),
        "typical_Pc_4K_nearres_pct": (30, 50),
        "optimal_excitation": "Near-resonant: HeNe 633 nm",
        "dominant_depolarization": "Intervalley exchange (MSS mechanism)",
        "notes": [
            "First demonstration of valley polarization: Kioseoglou et al. APL 2012",
            "Near-resonant 633 nm excitation gives best Pc at RT",
            "532 nm (non-resonant) gives weak or zero Pc at RT",
            "Trion (X-) shows opposite Pc sign from X0 under some conditions",
        ],
        "ref": "Kioseoglou et al. APL 101, 221907 (2012); Zeng et al. Nat. Nano. 2012",
    },
    "ws2": {
        "material": "WS2", "point_group": "D3h (monolayer)",
        "X0_eV": 1.97, "X0_nm": 630,
        "trion_binding_meV": 30, "trion_type": "X-",
        "spin_orbit_valence_meV": 430,
        "typical_Pc_RT_nonres_pct": (0, 10),
        "typical_Pc_RT_nearres_pct": (10, 40),
        "typical_Pc_4K_nearres_pct": (40, 90),
        "optimal_excitation": "Near-resonant: 594-620 nm range",
        "dominant_depolarization": "Intervalley exchange + phonon-assisted scattering",
        "notes": [
            "Larger spin-orbit splitting (430 meV) than MoS2 -> better valley contrast",
            "Graphite substrate gives persistent RT valley polarization (Demeridou et al. 2D Mater. 2023)",
            "Photochlorination tunes Pc by 40%+ (Kioseoglou group)",
            "Anomalous T-dependent Pc reported (Hanbicki, Kioseoglou et al. Sci. Rep. 2016)",
            "B exciton near 528 nm overlaps 532 nm laser — use longer wavelength for VP",
            "Biaxial strain suppresses Pc: -130 meV/% detuning + exchange modulation (Kourmoulakis APL 2023)",
        ],
        "ref": "Hanbicki et al. Sci. Rep. 2016; Demeridou et al. 2D Mater. 2023",
    },
    "mose2": {
        "material": "MoSe2", "point_group": "D3h (monolayer)",
        "X0_eV": 1.55, "X0_nm": 800,
        "trion_binding_meV": 30, "trion_type": "X-",
        "spin_orbit_valence_meV": 190,
        "typical_Pc_RT_nonres_pct": (0, 5),
        "typical_Pc_RT_nearres_pct": (5, 15),
        "typical_Pc_4K_nearres_pct": (20, 40),
        "optimal_excitation": "Near-resonant: 750-790 nm range (Ti:Sapphire)",
        "dominant_depolarization": "Intervalley exchange (MSS mechanism)",
        "notes": [
            "NIR emission — harder to detect but spin-orbit splitting is moderate",
            "Ti:Sapphire tunable laser ideal for near-resonant excitation",
            "Kioseoglou et al. Sci. Rep. 2016: optical polarization and intervalley scattering",
        ],
        "ref": "Kioseoglou et al. Sci. Rep. 6, 25041 (2016)",
    },
    "wse2": {
        "material": "WSe2", "point_group": "D3h (monolayer)",
        "X0_eV": 1.712, "X0_nm": 724,
        "trion_binding_meV": 30, "trion_type": "X-/X+",
        "spin_orbit_valence_meV": 460,
        "typical_Pc_RT_nonres_pct": (0, 5),
        "typical_Pc_RT_nearres_pct": (5, 20),
        "typical_Pc_4K_nearres_pct": (30, 70),
        "optimal_excitation": "Near-resonant: 690-720 nm or resonant at X0",
        "dominant_depolarization": "Intervalley exchange + dark exciton reservoir",
        "notes": [
            "Dark exciton ~50 meV below bright X0 acts as depolarization reservoir",
            "Photochemical doping (photochlorination) modulates Pc non-monotonically (Katsipoulaki et al. 2025)",
            "Pristine (n-type): X- dominant, Pc ~10-20% at 78K non-resonant",
            "Near charge neutrality: Pc minimum (<10% at 78K)",
            "Hole-doped: Pc increases 3x (X+ appears, Katsipoulaki et al. Adv. Opt. Mater. 2025)",
            "Largest spin-orbit splitting among common TMDs -> strongest spin-valley locking",
            "Electron density control via photochlorination (Katsipoulaki et al. 2D Mater. 2023)",
            "For off-resonance: 532 nm gives weak Pc; use redder excitation",
        ],
        "ref": "Katsipoulaki et al. Adv. Opt. Mater. 13, 2500575 (2025); Katsipoulaki et al. 2D Mater. 10, 045008 (2023)",
    },
}


# ---------------------------------------------------------------------------
#  Nonlinear optics material data (SHG / THG)
# ---------------------------------------------------------------------------
NONLINEAR_DB: Dict[str, Dict[str, Any]] = {
    "mos2": {
        "shg_active": True, "thg_active": True,
        "point_group": "D3h", "crystal_system": "hexagonal",
        "shg_layer_rule": "Odd layers only (1L, 3L, 5L). Bilayer centrosymmetric -> SHG = 0.",
        "twisted_bilayer": "SHG restored proportional to sin(3*twist_angle) for twisted bilayer",
        "notes": "SHG intensity ~ |chi2|^2. P-SHG gives armchair direction.",
    },
    "ws2": {
        "shg_active": True, "thg_active": True,
        "point_group": "D3h", "crystal_system": "hexagonal",
        "shg_layer_rule": "Odd layers only. P-SHG maps armchair direction and detects strain.",
        "twisted_bilayer": "SHG restored proportional to sin(3*twist_angle)",
        "notes": "P-SHG strain detection: Kourmoulakis et al. Sci. Rep. 2024. "
                 "Strain breaks C3v -> anisotropic P-SHG pattern reveals strain distribution.",
    },
    "mose2": {
        "shg_active": True, "thg_active": True,
        "point_group": "D3h", "crystal_system": "hexagonal",
        "shg_layer_rule": "Odd layers only.",
        "twisted_bilayer": "SHG restored proportional to sin(3*twist_angle)",
        "notes": "NIR SHG — need appropriate detector.",
    },
    "wse2": {
        "shg_active": True, "thg_active": True,
        "point_group": "D3h", "crystal_system": "hexagonal",
        "shg_layer_rule": "Odd layers only.",
        "twisted_bilayer": "SHG restored proportional to sin(3*twist_angle)",
        "notes": "Giant SHG enhancement at exciton resonance (Wang et al. PRL 2015).",
    },
    "sns": {
        "shg_active": True, "thg_active": True,
        "point_group": "C2v", "crystal_system": "orthorhombic (Pnma)",
        "shg_layer_rule": "All layer numbers (non-centrosymmetric in all thicknesses)",
        "twisted_bilayer": "N/A — not typically stacked",
        "notes": "In-plane anisotropic SHG and THG. P-SHG/P-THG reveals armchair vs zigzag. "
                 "THG anisotropy ratio ~0.75 (Maragkakis et al. Adv. Opt. Mater. 2024). "
                 "Excitation: 1028 nm (SHG->514nm) or 1542 nm (THG->514nm).",
    },
    "csgei3": {
        "shg_active": True, "thg_active": True,
        "point_group": "R3m", "crystal_system": "rhombohedral",
        "shg_layer_rule": "Bulk nonlinear — SHG active in all forms",
        "twisted_bilayer": "N/A",
        "notes": "Lead-free nonlinear perovskite. Large chi2.",
    },
}


# ---------------------------------------------------------------------------
#  Short-pass filter database (for SHG/THG experiments)
# ---------------------------------------------------------------------------
SHORTPASS_FILTER_DB: List[Dict[str, Any]] = [
    {"id": "SP500", "type": "short_pass", "cut_off_nm": 500, "label": "SP500", "od_above": 6, "notes": "Blocks >500 nm"},
    {"id": "SP550", "type": "short_pass", "cut_off_nm": 550, "label": "SP550", "od_above": 6, "notes": "Blocks >550 nm"},
    {"id": "SP600", "type": "short_pass", "cut_off_nm": 600, "label": "SP600", "od_above": 6, "notes": "Blocks >600 nm"},
    {"id": "SP650", "type": "short_pass", "cut_off_nm": 650, "label": "SP650", "od_above": 6, "notes": "Blocks >650 nm"},
    {"id": "SP700", "type": "short_pass", "cut_off_nm": 700, "label": "SP700", "od_above": 6, "notes": "Blocks >700 nm"},
    {"id": "SP750", "type": "short_pass", "cut_off_nm": 750, "label": "SP750", "od_above": 6, "notes": "Blocks >750 nm"},
    {"id": "SP850", "type": "short_pass", "cut_off_nm": 850, "label": "SP850", "od_above": 6, "notes": "Blocks >850 nm"},
]


# ---------------------------------------------------------------------------
#  Raman shift → absolute wavelength helper
# ---------------------------------------------------------------------------
def _raman_wavelength(laser_nm: float, shift_cm1: float) -> float:
    """Convert Raman shift (cm⁻¹) relative to laser (nm) → Stokes wavelength (nm)."""
    laser_cm1 = 1e7 / laser_nm
    raman_cm1 = laser_cm1 - shift_cm1
    if raman_cm1 <= 0:
        return 0.0
    return 1e7 / raman_cm1

def _nm_to_ev(nm: float) -> float:
    if nm <= 0: return 0.0
    return 1239.841 / nm

def _ev_to_nm(ev: float) -> float:
    if ev <= 0: return 0.0
    return 1239.841 / ev


# =============================================================================
#  Materials Project bandgap lookup
# =============================================================================
def _mp_bandgap(formula: str) -> Optional[Dict[str, Any]]:
    """Query Materials Project for bandgap of a material."""
    if not HAS_MP_API or not MP_API_KEY:
        return None
    try:
        with MPRester(MP_API_KEY) as mpr:
            docs = mpr.materials.summary.search(
                formula=formula,
                energy_above_hull=(0, 0.1),
                fields=["material_id", "formula_pretty", "band_gap",
                         "is_gap_direct", "is_metal", "symmetry"],
            )
            if not docs:
                return None
            # Pick most stable (lowest e_above_hull)
            doc = docs[0]
            return {
                "mp_id": str(doc.material_id),
                "formula": doc.formula_pretty,
                "band_gap_ev": float(doc.band_gap) if doc.band_gap else None,
                "is_direct": bool(doc.is_gap_direct) if doc.is_gap_direct is not None else None,
                "is_metal": bool(doc.is_metal) if doc.is_metal is not None else None,
                "space_group": doc.symmetry.symbol if doc.symmetry else "",
                "source": "Materials Project (DFT-PBE)",
                "caveat": "DFT-PBE underestimates band gaps by ~30-40%. Experimental values are typically larger.",
            }
    except Exception as e:
        logger.warning(f"MP bandgap lookup failed for '{formula}': {e}")
        return None


# =============================================================================
#  Curated DB lookup with fuzzy matching
# =============================================================================
def _lookup_curated(query: str) -> Optional[Dict[str, Any]]:
    """Look up material in curated PL database. Tries exact key, then aliases."""
    q = query.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    # Direct key match
    if q in PL_MATERIALS_DB:
        return PL_MATERIALS_DB[q]
    # Try aliases
    q_full = query.strip().lower()
    if q_full in _PL_ALIASES:
        return PL_MATERIALS_DB.get(_PL_ALIASES[q_full])
    # Partial match on aliases
    for alias, key in _PL_ALIASES.items():
        if q in alias.replace(" ", "") or alias.replace(" ", "") in q:
            return PL_MATERIALS_DB.get(key)
    # Try formula match in DB values
    for key, mat in PL_MATERIALS_DB.items():
        if q == mat.get("formula", "").lower().replace(" ", ""):
            return mat
    return None


# =============================================================================
#  Setup recommendation engine
# =============================================================================
def _select_laser(emission_peaks: List[Dict], gap_ev: Optional[float],
                  preferred_ids: Optional[List[str]] = None) -> List[Dict]:
    """Select suitable lasers. Returns sorted by suitability."""
    candidates = []
    min_emission_nm = min(p["center_nm"] for p in emission_peaks) if emission_peaks else 500

    for laser in LASER_DB:
        score = 0
        reasons = []
        l_nm = laser["wavelength_nm"]
        l_ev = laser["energy_eV"]

        # Must be shorter wavelength than emission
        if l_nm >= min_emission_nm - 10:
            continue

        # Check if laser energy exceeds gap (if known)
        if gap_ev and gap_ev > 0:
            if l_ev >= gap_ev:
                score += 30
                reasons.append(f"Energy {l_ev:.2f} eV > gap {gap_ev:.2f} eV")
            elif l_ev >= gap_ev * 0.9:
                score += 15
                reasons.append(f"Near-resonant with gap ({l_ev:.2f} vs {gap_ev:.2f} eV)")
            else:
                score += 5
                reasons.append(f"Below gap but may excite excitons")

        # Spectral separation from emission (more is better for filtering)
        gap_nm = min_emission_nm - l_nm
        if gap_nm > 80:
            score += 20
            reasons.append(f"Good separation ({gap_nm:.0f} nm) — easy filtering")
        elif gap_nm > 30:
            score += 10
            reasons.append(f"Moderate separation ({gap_nm:.0f} nm)")
        else:
            score += 2
            reasons.append(f"Tight separation ({gap_nm:.0f} nm) — may need notch filter")

        # Preferred by curated DB
        if preferred_ids and laser["id"] in preferred_ids:
            score += 25
            reasons.append("Recommended in PL database for this material")

        # General preference for common wavelengths
        if laser["id"] in ("ndyag_532", "diode_405", "hene_633"):
            score += 5

        candidates.append({
            **laser,
            "score": score,
            "reasons": reasons,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def _select_filters(laser_nm: float, emission_peaks: List[Dict],
                    raman_lines_cm1: Optional[List[float]] = None) -> Dict[str, Any]:
    """Select optimal filter set for given laser + emission."""
    result = {
        "long_pass": [],
        "notch": [],
        "band_pass": [],
        "dichroic": [],
        "nd": [],
        "raman_warning": [],
    }

    min_em = min(p["center_nm"] for p in emission_peaks) if emission_peaks else laser_nm + 50
    max_em = max(p["center_nm"] for p in emission_peaks) if emission_peaks else laser_nm + 200

    # Long-pass: cut-on between laser and emission
    for f in FILTER_DB:
        if f["type"] != "long_pass":
            continue
        co = f["cut_on_nm"]
        # Must block laser (cut-on > laser + margin)
        if co < laser_nm + 10:
            continue
        # Must pass at least some emission
        if co > min_em - 15:
            continue
        # Score: prefer tight but safe cut-on
        gap_to_laser = co - laser_nm
        gap_to_emission = min_em - co
        score = 0
        if 15 <= gap_to_laser <= 40:
            score += 20  # ideal range
        elif gap_to_laser > 40:
            score += 10
        if gap_to_emission > 30:
            score += 15
        elif gap_to_emission > 15:
            score += 8
        result["long_pass"].append({**f, "score": score,
                                     "note": f"Blocks <{co} nm; passes emission >{co} nm"})

    result["long_pass"].sort(key=lambda x: x.get("score", 0), reverse=True)

    # Notch filter at laser wavelength
    for f in FILTER_DB:
        if f["type"] != "notch":
            continue
        if abs(f["center_nm"] - laser_nm) < 20:
            result["notch"].append({**f, "note": f"Blocks laser at {laser_nm:.0f} nm"})

    # Band-pass: for isolating specific peaks
    for f in FILTER_DB:
        if f["type"] != "band_pass":
            continue
        fc = f["center_nm"]
        fw = f["width_nm"]
        for peak in emission_peaks:
            pc = peak["center_nm"]
            if abs(fc - pc) < fw:  # filter window covers peak
                result["band_pass"].append({
                    **f, "score": 10,
                    "note": f"Isolates {peak['label']} at {pc} nm"
                })
                break

    # Dichroic mirror
    for f in FILTER_DB:
        if f["type"] != "dichroic":
            continue
        co = f["cut_on_nm"]
        if laser_nm < co < min_em + 10:
            gap = co - laser_nm
            score = 20 if 15 < gap < 50 else 10
            result["dichroic"].append({**f, "score": score,
                                        "note": f"Reflects laser <{co} nm to sample; transmits PL >{co} nm"})

    result["dichroic"].sort(key=lambda x: x.get("score", 0), reverse=True)

    # Check for Raman interference
    if raman_lines_cm1:
        for shift in raman_lines_cm1:
            raman_nm = _raman_wavelength(laser_nm, shift)
            if raman_nm <= 0:
                continue
            # Check if Raman line falls in emission window
            for peak in emission_peaks:
                if abs(raman_nm - peak["center_nm"]) < peak.get("fwhm_nm", 20) * 1.5:
                    result["raman_warning"].append({
                        "shift_cm1": shift,
                        "raman_nm": round(raman_nm, 1),
                        "near_peak": peak["label"],
                        "warning": (f"Raman line at {shift} cm⁻¹ ({raman_nm:.1f} nm) "
                                    f"is close to {peak['label']} at {peak['center_nm']} nm"),
                    })
            # Check if best LP filter would pass the Raman line
            if result["long_pass"]:
                best_lp = result["long_pass"][0]["cut_on_nm"]
                if raman_nm > best_lp:
                    result["raman_warning"].append({
                        "shift_cm1": shift,
                        "raman_nm": round(raman_nm, 1),
                        "warning": (f"Raman line at {shift} cm⁻¹ ({raman_nm:.1f} nm) "
                                    f"passes through LP{best_lp} filter — may appear in PL spectrum"),
                    })

    # ND recommendation for sensitive materials
    result["nd"] = [f for f in FILTER_DB if f["type"] == "nd"]

    return result


# =============================================================================
#  Spectral sketch plotting
# =============================================================================
def _plot_spectrum_sketch(
    material_name: str,
    laser_nm: float,
    emission_peaks: List[Dict],
    filter_set: Dict,
    raman_lines_cm1: Optional[List[float]] = None,
) -> plt.Figure:
    """Generate a publication-quality annotated spectral sketch."""

    # ── Style setup ──────────────────────────────────────────────────────
    BG       = "#FAFBFC"
    GRID_COL = "#E0E4E8"
    SPINE_COL= "#B0B8C0"
    LASER_C  = "#1565C0"   # deep blue
    BLOCK_C  = "#FFCDD2"   # soft red fill
    PASS_C   = "#C8E6C9"   # soft green fill
    NOTCH_C  = "#FFB74D"   # amber
    DM_C     = "#7E57C2"   # purple
    PEAK_COLORS = ["#E53935", "#FB8C00", "#43A047", "#8E24AA", "#6D4C41"]
    RAMAN_C  = "#90A4AE"   # blue-grey

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
    })

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 8.5),
        gridspec_kw={"height_ratios": [3.5, 1], "hspace": 0.06},
        sharex=True,
    )
    fig.patch.set_facecolor(BG)
    for ax in (ax_top, ax_bot):
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color(SPINE_COL)
            spine.set_linewidth(0.8)
        ax.tick_params(colors="#555555", labelsize=12, width=0.8, length=4)

    fig.suptitle(f"PL Setup Sketch  —  {material_name}",
                 fontsize=18, fontweight="bold", color="#212121", y=0.98)

    # ── Wavelength range ─────────────────────────────────────────────────
    all_nm = [laser_nm] + [p["center_nm"] for p in emission_peaks]
    wl_min = min(all_nm) - 80
    wl_max = max(all_nm) + 120
    wl_min = max(wl_min, 200)
    wl_max = min(wl_max, 1300)
    x = np.linspace(wl_min, wl_max, 2000)

    # ══════════════════════════════════════════════════════════════════════
    #  TOP PANEL — Spectral features
    # ══════════════════════════════════════════════════════════════════════

    # Laser line
    laser_profile = np.exp(-0.5 * ((x - laser_nm) / 1.5) ** 2) * 0.85
    ax_top.fill_between(x, laser_profile, alpha=0.20, color=LASER_C, zorder=3)
    ax_top.plot(x, laser_profile, color=LASER_C, linewidth=2, zorder=4,
                label=f"Laser {laser_nm:.0f} nm")
    ax_top.annotate(
        f"Laser\n{laser_nm:.0f} nm", xy=(laser_nm, 0.88),
        fontsize=13, ha="center", color=LASER_C, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=LASER_C, alpha=0.85, lw=0.8),
        zorder=10,
    )

    # PL emission peaks
    for i, peak in enumerate(emission_peaks):
        c = peak["center_nm"]
        fwhm = peak.get("fwhm_nm", 20)
        sigma = fwhm / 2.355
        amp = 0.65 if i == 0 else 0.35
        profile = amp * np.exp(-0.5 * ((x - c) / sigma) ** 2)
        col = PEAK_COLORS[i % len(PEAK_COLORS)]
        ev = peak.get("center_ev", _nm_to_ev(c))

        ax_top.fill_between(x, profile, alpha=0.18, color=col, zorder=2)
        ax_top.plot(x, profile, color=col, linewidth=2, zorder=3,
                    label=f"{peak['label']}  ({c:.0f} nm / {ev:.2f} eV)")

        # Smart annotation offset — stagger vertically for close peaks
        y_off = 12 + (i % 2) * 10
        ax_top.annotate(
            f"{peak['label']}  {c:.0f} nm", xy=(c, amp),
            xytext=(0, y_off), textcoords="offset points",
            fontsize=12, ha="center", color=col, fontweight="semibold",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.8, alpha=0.5),
            zorder=10,
        )

    # Raman lines
    if raman_lines_cm1:
        for j, shift in enumerate(raman_lines_cm1):
            raman_nm = _raman_wavelength(laser_nm, shift)
            if wl_min < raman_nm < wl_max:
                raman_profile = 0.12 * np.exp(-0.5 * ((x - raman_nm) / 1.8) ** 2)
                ax_top.fill_between(x, raman_profile, alpha=0.10, color=RAMAN_C,
                                    hatch="///", edgecolor=RAMAN_C, linewidth=0, zorder=1)
                ax_top.plot(x, raman_profile, color=RAMAN_C, linewidth=1.2,
                            linestyle="--", alpha=0.6, zorder=2)
                y_txt = 0.15 + (j % 2) * 0.06
                ax_top.annotate(
                    f"Raman {shift:.0f} cm⁻¹", xy=(raman_nm, y_txt),
                    fontsize=10, ha="center", color="#607D8B", style="italic",
                    zorder=10,
                )

    ax_top.set_ylabel("Relative Intensity", fontsize=13, color="#444444", labelpad=10)
    ax_top.set_ylim(0, 1.08)
    ax_top.legend(loc="upper right", fontsize=11, framealpha=0.92,
                  edgecolor=SPINE_COL, fancybox=True, borderpad=0.8)
    ax_top.grid(True, alpha=0.35, color=GRID_COL, linewidth=0.6)
    ax_top.tick_params(axis="x", labelbottom=False)

    # ══════════════════════════════════════════════════════════════════════
    #  BOTTOM PANEL — Filter windows
    # ══════════════════════════════════════════════════════════════════════

    # Long-pass filter
    if filter_set.get("long_pass"):
        best_lp = filter_set["long_pass"][0]
        co = best_lp["cut_on_nm"]
        ax_bot.axvspan(wl_min, co, alpha=0.25, color=BLOCK_C, zorder=1)
        ax_bot.axvspan(co, wl_max, alpha=0.20, color=PASS_C, zorder=1)
        ax_bot.axvline(co, color="#C62828", linewidth=2.5, linestyle="--", zorder=5)
        ax_bot.text(
            co + (wl_max - co) * 0.02, 0.78,
            f'{best_lp["label"]}  (cut-on {co} nm)',
            fontsize=12, color="#B71C1C", fontweight="bold", zorder=10,
        )
        ax_bot.text(
            (wl_min + co) / 2, 0.45, "BLOCKED",
            fontsize=14, ha="center", va="center", color="#C62828",
            fontweight="bold", alpha=0.55, zorder=10,
        )
        ax_bot.text(
            (co + wl_max) / 2, 0.45, "PASSED",
            fontsize=14, ha="center", va="center", color="#2E7D32",
            fontweight="bold", alpha=0.55, zorder=10,
        )

    # Notch filter
    if filter_set.get("notch"):
        nf = filter_set["notch"][0]
        nc = nf["center_nm"]
        nw = nf.get("width_nm", 25)
        ax_bot.axvspan(nc - nw / 2, nc + nw / 2, alpha=0.35, color=NOTCH_C, zorder=6)
        ax_bot.text(
            nc, 0.18, f'Notch {nc:.0f} nm',
            fontsize=11, ha="center", color="#E65100", fontweight="bold", zorder=10,
        )

    # Dichroic mirror
    if filter_set.get("dichroic"):
        dm = filter_set["dichroic"][0]
        dc = dm["cut_on_nm"]
        ax_bot.axvline(dc, color=DM_C, linewidth=2, linestyle=":", alpha=0.8, zorder=5)
        ax_bot.text(
            dc, 0.92, f'DM{dc}',
            fontsize=11, ha="center", color=DM_C, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=DM_C, alpha=0.85, lw=0.8),
            zorder=10,
        )

    ax_bot.set_xlabel("Wavelength (nm)", fontsize=14, color="#444444", labelpad=8)
    ax_bot.set_ylabel("Filters", fontsize=13, color="#444444", labelpad=10)
    ax_bot.set_ylim(0, 1.05)
    ax_bot.set_xlim(wl_min, wl_max)
    ax_bot.set_yticks([])
    ax_bot.grid(True, axis="x", alpha=0.35, color=GRID_COL, linewidth=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _save_figure(fig: plt.Figure, prefix: str) -> Tuple[str, str]:
    """Save figure to export dir. Returns (filepath, url)."""
    uid = uuid.uuid4().hex[:10]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"pl_{uid}_{ts}"
    out_dir = os.path.join(EXPORT_DIR, folder)
    os.makedirs(out_dir, exist_ok=True)

    fname = f"{prefix}.png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor(),
                edgecolor="#E0E0E0", pad_inches=0.3)
    plt.close(fig)
    url = f"{BASE_URL}/{folder}/{fname}"
    return path, url


# =============================================================================
#  MCP TOOL 1: pl_recommend — Main recommendation engine
# =============================================================================
@mcp.tool()
def pl_recommend(
    material: str,
    goal: str = "emission",
    laser_wavelength_nm: float = 0,
    temperature: str = "room_temperature",
) -> Dict[str, Any]:
    """
    Recommend a complete PL experimental setup for a given material.

    This is the PRIMARY tool for PL experiment planning. Given a material
    and measurement goal, it returns laser, filter, and optical component
    recommendations with physics-based reasoning.

    Args:
        material: Material name or formula (e.g. "MoS2", "CsPbBr3", "GaN",
                  "silicon quantum dots"). Case-insensitive.
        goal: Measurement goal. Options:
              "emission" — standard PL measurement (default)
              "power_dependent" — power-dependent PL (adds ND filter guidance)
              "excitation_scan" — PLE measurement (notes on scanning requirements)
              "time_resolved" — TRPL (flags pulsed laser need)
              "raman_and_pl" — simultaneous Raman + PL
        laser_wavelength_nm: Optional. If user already has a specific laser in mind,
                             validate it and build filters around it. 0 = auto-select.
        temperature: "room_temperature" (default), "low_temperature", or "cryogenic".
                     Affects warnings and spectral expectations.

    Returns:
        Dict with setup recommendation, filter list, spectral sketch, and warnings.
    """
    t0 = time.time()
    logger.info(f"pl_recommend: material='{material}', goal='{goal}', laser={laser_wavelength_nm}")

    try:
        # ---- 1. Look up material ----
        curated = _lookup_curated(material)
        mp_data = None

        if curated:
            mat_name = curated["name"]
            formula = curated.get("formula", material)
            gap_ev = curated.get("gap_ev")
            gap_type = curated.get("gap_type", "")
            emission_peaks = curated.get("emission", [])
            recommended_laser_ids = curated.get("recommended_lasers", [])
            warnings = list(curated.get("warnings", []))
            raman_cm1 = curated.get("raman_lines_cm1", [])
            source = "curated PL database"
        else:
            # Try Materials Project for bandgap
            mp_data = _mp_bandgap(material)
            if mp_data and mp_data.get("band_gap_ev"):
                mat_name = mp_data["formula"]
                formula = mp_data["formula"]
                gap_ev = mp_data["band_gap_ev"]
                gap_type = "direct" if mp_data.get("is_direct") else "indirect"
                # Estimate emission from gap (rough: emission ≈ gap - 0.05 eV for Stokes shift)
                em_ev = gap_ev - 0.05
                em_nm = _ev_to_nm(em_ev) if em_ev > 0 else _ev_to_nm(gap_ev)
                emission_peaks = [{
                    "label": "Estimated band-edge PL",
                    "center_nm": round(em_nm),
                    "center_ev": round(em_ev, 3),
                    "fwhm_nm": 30,
                    "notes": "Estimated from DFT bandgap — experimental value may differ by ±0.3 eV",
                }]
                recommended_laser_ids = []
                warnings = [
                    f"Band gap from DFT ({gap_ev:.2f} eV) typically underestimates experimental gap by 30-40%",
                    "Emission position is estimated — verify experimentally",
                ]
                if mp_data.get("is_metal"):
                    return {"ok": False, "reason": f"'{material}' ({mp_data['formula']}) is a metal according to Materials Project — metals generally do not show PL."}
                raman_cm1 = []
                source = "Materials Project (DFT-PBE estimate)"
            else:
                return {
                    "ok": False,
                    "reason": (f"Material '{material}' not found in curated PL database "
                               f"or Materials Project. Try: (1) chemical formula like 'MoS2', "
                               f"(2) common name like 'gallium nitride', or "
                               f"(3) check spelling."),
                    "available_materials": sorted(PL_MATERIALS_DB.keys()),
                }

        # ---- 2. Goal-specific adjustments ----
        if goal == "time_resolved":
            warnings.append("TIME-RESOLVED PL requires a PULSED laser (fs/ps). CW lasers will not work.")
            warnings.append("You also need a time-correlated single photon counting (TCSPC) setup or streak camera.")
        elif goal == "power_dependent":
            warnings.append("Use calibrated ND filter wheel for systematic power variation.")
            warnings.append("Log-scale power steps recommended (e.g., 1, 3, 10, 30, 100 μW).")
            warnings.append("Monitor for heating effects: if peak redshifts with power, reduce maximum power.")
        elif goal == "excitation_scan":
            warnings.append("PLE requires a tunable laser or monochromated broadband source.")
            warnings.append("Fix detection at known emission peak; scan excitation wavelength.")
        elif goal == "raman_and_pl":
            warnings.append("For simultaneous Raman + PL: use a notch filter (not LP) to see Raman peaks.")
            warnings.append("Raman peaks appear very close to laser line; use a spectrometer with good stray-light rejection.")

        if temperature in ("low_temperature", "cryogenic"):
            warnings.append("Low temperature sharpens emission lines and reveals trion/exciton fine structure.")
            warnings.append("Expect narrower FWHM and possible peak shifts (~5-20 meV blueshift at low T).")

        # ---- 3. Select laser ----
        if laser_wavelength_nm > 0:
            # User specified a laser — validate it
            chosen_laser = None
            for L in LASER_DB:
                if abs(L["wavelength_nm"] - laser_wavelength_nm) < 5:
                    chosen_laser = L
                    break
            if not chosen_laser:
                chosen_laser = {
                    "id": "user_custom",
                    "name": f"User-specified {laser_wavelength_nm:.0f} nm",
                    "wavelength_nm": laser_wavelength_nm,
                    "energy_eV": round(_nm_to_ev(laser_wavelength_nm), 3),
                    "type": "unknown",
                    "notes": "Custom laser specified by user",
                }
            laser_candidates = [chosen_laser]

            # Validate
            if gap_ev and _nm_to_ev(laser_wavelength_nm) < gap_ev * 0.85:
                warnings.append(
                    f"⚠️ Laser energy ({_nm_to_ev(laser_wavelength_nm):.2f} eV) is significantly "
                    f"below band gap ({gap_ev:.2f} eV). PL may be very weak or absent."
                )
            min_em = min(p["center_nm"] for p in emission_peaks)
            if laser_wavelength_nm >= min_em - 10:
                warnings.append(
                    f"⚠️ Laser wavelength ({laser_wavelength_nm:.0f} nm) is too close to or "
                    f"above emission ({min_em:.0f} nm). Cannot separate PL from laser scatter."
                )
        else:
            laser_candidates = _select_laser(emission_peaks, gap_ev, recommended_laser_ids)
            if not laser_candidates:
                return {"ok": False, "reason": "No suitable laser found for this material's emission range."}

        best_laser = laser_candidates[0]
        laser_nm = best_laser["wavelength_nm"]

        # ---- 4. Select filters ----
        filter_set = _select_filters(laser_nm, emission_peaks, raman_cm1)

        # ---- 5. Detector notes ----
        max_em = max(p["center_nm"] for p in emission_peaks)
        detector_notes = []
        if max_em > 1050:
            detector_notes.append("⚠️ InGaAs detector REQUIRED — Si CCD cannot detect emission above ~1050 nm")
        elif max_em > 900:
            detector_notes.append("⚠️ Si CCD sensitivity drops sharply above 900 nm — check detector QE at emission wavelength")
        elif max_em > 800:
            detector_notes.append("Si CCD works but sensitivity is reduced — longer acquisition times may be needed")
        else:
            detector_notes.append("Standard Si CCD detector is suitable")

        # ---- 6. Generate spectral sketch ----
        fig = _plot_spectrum_sketch(mat_name, laser_nm, emission_peaks, filter_set, raman_cm1)
        _, sketch_url = _save_figure(fig, "setup_sketch")

        # ---- 7. Build response ----
        # Clean filter output for readability
        def _fmt_filters(flist):
            return [{"label": f["label"], "note": f.get("note", ""),
                      "type": f.get("type", "")} for f in flist[:3]]

        return {
            "ok": True,
            "material": {
                "name": mat_name,
                "formula": formula,
                "band_gap_ev": gap_ev,
                "gap_type": gap_type,
                "emission_peaks": emission_peaks,
                "source": source,
            },
            "recommended_setup": {
                "laser": {
                    "name": best_laser.get("name", ""),
                    "wavelength_nm": laser_nm,
                    "energy_eV": best_laser.get("energy_eV", round(_nm_to_ev(laser_nm), 3)),
                    "type": best_laser.get("type", ""),
                    "reasons": best_laser.get("reasons", []),
                },
                "alternative_lasers": [
                    {"name": L.get("name", ""), "wavelength_nm": L["wavelength_nm"],
                     "score": L.get("score", 0)}
                    for L in laser_candidates[1:3]
                ],
                "filters": {
                    "primary_long_pass": _fmt_filters(filter_set.get("long_pass", [])),
                    "notch": _fmt_filters(filter_set.get("notch", [])),
                    "dichroic_mirror": _fmt_filters(filter_set.get("dichroic", [])),
                    "band_pass_optional": _fmt_filters(filter_set.get("band_pass", [])),
                },
                "optical_path_summary": (
                    f"Laser ({laser_nm:.0f} nm) → ND filter (adjust power) → "
                    f"{filter_set['dichroic'][0]['label'] if filter_set.get('dichroic') else 'mirror'} → "
                    f"objective → sample → back through objective → "
                    f"{filter_set['dichroic'][0]['label'] if filter_set.get('dichroic') else 'beamsplitter'} → "
                    f"{filter_set['long_pass'][0]['label'] if filter_set.get('long_pass') else 'filter'} → "
                    f"spectrometer → detector"
                ),
            },
            "detector": detector_notes,
            "raman_interference": filter_set.get("raman_warning", []),
            "warnings": warnings,
            "goal": goal,
            "temperature": temperature,
            "artifacts": {
                "setup_sketch": sketch_url,
            },
            "timing_sec": round(time.time() - t0, 2),
        }

    except Exception as e:
        logger.exception("pl_recommend failed")
        return {"ok": False, "reason": str(e)}


# =============================================================================
#  MCP TOOL 2: pl_material_lookup — Query material optical properties
# =============================================================================
@mcp.tool()
def pl_material_lookup(
    material: str,
    query_mp: bool = True,
) -> Dict[str, Any]:
    """
    Look up optical and emission properties of a material for PL.

    Searches the curated PL database first (detailed emission data),
    then optionally queries Materials Project for DFT bandgap.

    Args:
        material: Material name or formula (e.g. "MoS2", "GaN", "ruby").
        query_mp: Whether to also query Materials Project for bandgap (default True).

    Returns:
        Dict with all known optical/emission properties.
    """
    logger.info(f"pl_material_lookup: '{material}'")
    try:
        result = {"ok": True, "query": material, "curated": None, "materials_project": None}

        # Curated DB
        curated = _lookup_curated(material)
        if curated:
            result["curated"] = {
                "name": curated["name"],
                "formula": curated.get("formula", ""),
                "category": curated.get("category", ""),
                "band_gap_ev": curated.get("gap_ev"),
                "gap_type": curated.get("gap_type", ""),
                "emission_peaks": curated.get("emission", []),
                "recommended_lasers": [
                    next((L for L in LASER_DB if L["id"] == lid), {"id": lid})
                    for lid in curated.get("recommended_lasers", [])
                ],
                "warnings": curated.get("warnings", []),
                "raman_lines_cm1": curated.get("raman_lines_cm1", []),
                "source": "FORTHought curated PL database",
            }

        # Materials Project
        if query_mp:
            mp = _mp_bandgap(material)
            if mp:
                result["materials_project"] = mp

        if not result["curated"] and not result["materials_project"]:
            result["ok"] = False
            result["reason"] = f"Material '{material}' not found in either database."
            result["hint"] = "Try the chemical formula (e.g., 'MoS2' not 'moly disulfide')"
            result["available_curated"] = sorted(PL_MATERIALS_DB.keys())

        return result

    except Exception as e:
        logger.exception("pl_material_lookup failed")
        return {"ok": False, "reason": str(e)}


# =============================================================================
#  MCP TOOL 3: pl_filter_search — Search filter catalog
# =============================================================================
@mcp.tool()
def pl_filter_search(
    laser_nm: float = 0,
    emission_min_nm: float = 0,
    emission_max_nm: float = 0,
    filter_type: str = "all",
) -> Dict[str, Any]:
    """
    Search the optical filter catalog for suitable PL filters.

    Can search by laser wavelength (finds matching LP/notch/dichroic),
    by emission range, or both. Use filter_type to narrow results.

    Args:
        laser_nm: Laser wavelength in nm. If provided, returns filters that
                  block this laser. 0 = no laser constraint.
        emission_min_nm: Shortest emission wavelength to pass. 0 = no constraint.
        emission_max_nm: Longest emission wavelength to pass. 0 = no constraint.
        filter_type: "all" (default), "long_pass", "notch", "band_pass",
                     "dichroic", or "nd".

    Returns:
        Dict with matching filters grouped by type.
    """
    logger.info(f"pl_filter_search: laser={laser_nm}, em_range={emission_min_nm}-{emission_max_nm}")
    try:
        results = {"ok": True, "filters": [], "query": {
            "laser_nm": laser_nm, "emission_min_nm": emission_min_nm,
            "emission_max_nm": emission_max_nm, "filter_type": filter_type,
        }}

        for f in FILTER_DB:
            # Type filter
            if filter_type != "all" and f["type"] != filter_type:
                continue

            match = True
            score = 0
            notes = []

            if f["type"] == "long_pass":
                co = f["cut_on_nm"]
                if laser_nm > 0:
                    if co <= laser_nm:
                        match = False  # doesn't block laser
                    else:
                        score += 10
                        gap = co - laser_nm
                        notes.append(f"Blocks laser at {laser_nm:.0f} nm (margin {gap:.0f} nm)")
                if emission_min_nm > 0 and co > emission_min_nm - 10:
                    match = False  # blocks emission
                if match and emission_min_nm > 0:
                    notes.append(f"Passes emission above {co} nm")

            elif f["type"] == "notch":
                nc = f["center_nm"]
                if laser_nm > 0:
                    if abs(nc - laser_nm) > 20:
                        match = False
                    else:
                        score += 15
                        notes.append(f"Blocks laser at {laser_nm:.0f} nm (notch center {nc} nm)")

            elif f["type"] == "band_pass":
                fc = f["center_nm"]
                fw = f["width_nm"]
                lo = fc - fw / 2
                hi = fc + fw / 2
                if emission_min_nm > 0 and emission_max_nm > 0:
                    # Check if BP window overlaps desired emission
                    if hi < emission_min_nm or lo > emission_max_nm:
                        match = False
                    else:
                        score += 10
                        notes.append(f"Passes {lo:.0f}–{hi:.0f} nm")

            elif f["type"] == "dichroic":
                co = f["cut_on_nm"]
                if laser_nm > 0:
                    if co < laser_nm - 5 or co > laser_nm + 60:
                        match = False
                    else:
                        score += 10
                        notes.append(f"Reflects <{co} nm (laser), transmits >{co} nm (PL)")

            elif f["type"] == "nd":
                # Always include ND filters if requested
                if filter_type == "nd" or filter_type == "all":
                    notes.append(f"OD {f['od']}: {100 * 10**(-f['od']):.1f}% transmission")
                else:
                    match = False

            if match:
                results["filters"].append({
                    **f, "score": score, "match_notes": notes,
                })

        # Sort by score
        results["filters"].sort(key=lambda x: x.get("score", 0), reverse=True)
        results["count"] = len(results["filters"])

        return results

    except Exception as e:
        logger.exception("pl_filter_search failed")
        return {"ok": False, "reason": str(e)}


# =============================================================================
#  MCP TOOL 4: pl_check_setup — Validate a proposed experimental setup
# =============================================================================
@mcp.tool()
def pl_check_setup(
    material: str,
    laser_nm: float,
    filter_lp_nm: float = 0,
    filter_notch_nm: float = 0,
) -> Dict[str, Any]:
    """
    Validate a proposed PL measurement setup.

    Checks whether laser can excite the material, whether the filter
    blocks the laser, whether the filter passes emission, and warns
    about Raman interference or detector limitations.

    Args:
        material: Material name or formula.
        laser_nm: Laser wavelength in nm.
        filter_lp_nm: Long-pass filter cut-on wavelength in nm. 0 = none.
        filter_notch_nm: Notch filter center wavelength in nm. 0 = none.

    Returns:
        Dict with pass/fail for each check and overall verdict.
    """
    logger.info(f"pl_check_setup: material='{material}', laser={laser_nm}, LP={filter_lp_nm}")
    try:
        checks = []
        all_pass = True

        # Look up material
        curated = _lookup_curated(material)
        if not curated:
            mp_data = _mp_bandgap(material)
            if mp_data and mp_data.get("band_gap_ev"):
                gap_ev = mp_data["band_gap_ev"]
                em_ev = gap_ev - 0.05
                emission_peaks = [{"label": "Estimated PL", "center_nm": round(_ev_to_nm(em_ev)),
                                   "center_ev": round(em_ev, 3), "fwhm_nm": 30}]
                raman_cm1 = []
            else:
                return {"ok": False, "reason": f"Material '{material}' not found."}
        else:
            gap_ev = curated.get("gap_ev")
            emission_peaks = curated.get("emission", [])
            raman_cm1 = curated.get("raman_lines_cm1", [])

        laser_ev = _nm_to_ev(laser_nm)
        min_em = min(p["center_nm"] for p in emission_peaks) if emission_peaks else 999

        # Check 1: Laser energy vs bandgap
        if gap_ev and gap_ev > 0:
            if laser_ev >= gap_ev:
                checks.append({"check": "Laser above bandgap", "pass": True,
                                "detail": f"Laser {laser_ev:.2f} eV ≥ gap {gap_ev:.2f} eV ✓"})
            elif laser_ev >= gap_ev * 0.9:
                checks.append({"check": "Laser near-resonant", "pass": True,
                                "detail": f"Laser {laser_ev:.2f} eV is near-resonant with gap {gap_ev:.2f} eV (may still excite excitons)"})
            else:
                checks.append({"check": "Laser below bandgap", "pass": False,
                                "detail": f"Laser {laser_ev:.2f} eV < gap {gap_ev:.2f} eV — PL unlikely"})
                all_pass = False

        # Check 2: Laser vs emission separation
        if laser_nm < min_em - 15:
            checks.append({"check": "Spectral separation", "pass": True,
                            "detail": f"Laser {laser_nm:.0f} nm well below emission {min_em:.0f} nm ✓"})
        elif laser_nm < min_em:
            checks.append({"check": "Spectral separation", "pass": True,
                            "detail": f"Laser {laser_nm:.0f} nm close to emission {min_em:.0f} nm — tight filtering needed"})
        else:
            checks.append({"check": "Spectral separation", "pass": False,
                            "detail": f"Laser {laser_nm:.0f} nm ≥ emission {min_em:.0f} nm — CANNOT separate"})
            all_pass = False

        # Check 3: Long-pass filter blocks laser?
        if filter_lp_nm > 0:
            if filter_lp_nm > laser_nm + 5:
                checks.append({"check": "LP filter blocks laser", "pass": True,
                                "detail": f"LP{filter_lp_nm:.0f} blocks laser at {laser_nm:.0f} nm ✓"})
            else:
                checks.append({"check": "LP filter blocks laser", "pass": False,
                                "detail": f"LP{filter_lp_nm:.0f} does NOT block laser at {laser_nm:.0f} nm"})
                all_pass = False

            # Check LP passes emission
            if filter_lp_nm < min_em - 5:
                checks.append({"check": "LP filter passes emission", "pass": True,
                                "detail": f"LP{filter_lp_nm:.0f} passes emission at {min_em:.0f} nm ✓"})
            else:
                checks.append({"check": "LP filter passes emission", "pass": False,
                                "detail": f"LP{filter_lp_nm:.0f} may block emission at {min_em:.0f} nm!"})
                all_pass = False

        # Check 4: Raman interference
        raman_warns = []
        for shift in raman_cm1:
            rnm = _raman_wavelength(laser_nm, shift)
            if rnm > 0:
                for peak in emission_peaks:
                    if abs(rnm - peak["center_nm"]) < peak.get("fwhm_nm", 20):
                        raman_warns.append(f"Raman {shift} cm⁻¹ ({rnm:.0f} nm) overlaps {peak['label']}")
        if raman_warns:
            checks.append({"check": "Raman interference", "pass": False,
                            "detail": "; ".join(raman_warns)})
        else:
            checks.append({"check": "Raman interference", "pass": True,
                            "detail": "No Raman overlap with PL peaks"})

        # Check 5: Detector suitability
        max_em = max(p["center_nm"] for p in emission_peaks)
        if max_em > 1050:
            checks.append({"check": "Detector", "pass": False,
                            "detail": f"Emission at {max_em:.0f} nm requires InGaAs detector (Si CCD cannot see this)"})
            all_pass = False
        elif max_em > 900:
            checks.append({"check": "Detector", "pass": True,
                            "detail": f"Emission at {max_em:.0f} nm — Si CCD works but with reduced sensitivity"})
        else:
            checks.append({"check": "Detector", "pass": True,
                            "detail": f"Standard Si CCD detector is suitable for {max_em:.0f} nm emission"})

        return {
            "ok": True,
            "verdict": "PASS ✓ — setup is valid" if all_pass else "ISSUES FOUND — see failed checks",
            "all_pass": all_pass,
            "checks": checks,
            "material": material,
            "laser_nm": laser_nm,
            "emission_peaks": emission_peaks,
        }

    except Exception as e:
        logger.exception("pl_check_setup failed")
        return {"ok": False, "reason": str(e)}


# =============================================================================
#  MCP TOOL 5: pl_spectrum_sketch — Generate a visual spectral sketch
# =============================================================================
@mcp.tool()
def pl_spectrum_sketch(
    material: str,
    laser_nm: float,
    filter_lp_nm: float = 0,
) -> Dict[str, Any]:
    """
    Generate a spectral sketch showing expected PL peaks, laser line,
    filter transmission windows, and Raman lines for a material.

    Use this to create an educational visualization of what the experiment
    will see through the spectrometer.

    Args:
        material: Material name or formula.
        laser_nm: Laser wavelength in nm.
        filter_lp_nm: Long-pass filter cut-on in nm (0 = auto from laser).

    Returns:
        Dict with PNG URL of the spectral sketch.
    """
    logger.info(f"pl_spectrum_sketch: material='{material}', laser={laser_nm}")
    try:
        curated = _lookup_curated(material)
        if not curated:
            mp_data = _mp_bandgap(material)
            if mp_data and mp_data.get("band_gap_ev"):
                em_ev = mp_data["band_gap_ev"] - 0.05
                emission_peaks = [{"label": "Estimated PL", "center_nm": round(_ev_to_nm(em_ev)),
                                   "center_ev": round(em_ev, 3), "fwhm_nm": 30}]
                mat_name = mp_data["formula"]
                raman_cm1 = []
            else:
                return {"ok": False, "reason": f"Material '{material}' not found."}
        else:
            emission_peaks = curated.get("emission", [])
            mat_name = curated["name"]
            raman_cm1 = curated.get("raman_lines_cm1", [])

        # Build filter set
        if filter_lp_nm > 0:
            filter_set = {
                "long_pass": [{"label": f"LP{filter_lp_nm:.0f}", "cut_on_nm": filter_lp_nm, "type": "long_pass"}],
                "notch": [], "dichroic": [], "band_pass": [],
            }
        else:
            filter_set = _select_filters(laser_nm, emission_peaks, raman_cm1)

        fig = _plot_spectrum_sketch(mat_name, laser_nm, emission_peaks, filter_set, raman_cm1)
        _, sketch_url = _save_figure(fig, "spectrum_sketch")

        return {
            "ok": True,
            "material": mat_name,
            "laser_nm": laser_nm,
            "emission_peaks": [{"label": p["label"], "center_nm": p["center_nm"]} for p in emission_peaks],
            "artifacts": {
                "sketch_png": sketch_url,
            },
        }

    except Exception as e:
        logger.exception("pl_spectrum_sketch failed")
        return {"ok": False, "reason": str(e)}



# =============================================================================
#  TOOL 6: pl_substrate_enhancement — Fresnel multi-reflection model
# =============================================================================
@mcp.tool()
def pl_substrate_enhancement(
    material: str,
    excitation_nm: float,
    emission_nm: float,
    sio2_thickness_nm: float = 285.0,
    substrate: str = "Si",
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Calculate the Fresnel multi-reflection enhancement factor (F_total)
    for a 2D material on a layered substrate (air/SiO2/Si stack).

    This is critical for every 2D materials experiment: PL, Raman, and SHG
    signals can vary 3x or more purely from substrate interference effects.
    The tool calculates how efficiently the substrate stack enhances both
    excitation absorption and emission collection at the monolayer position.

    v3.0: Added SOI/nanoantenna geometry awareness. On nanostructured substrates
    (e.g., Si pillars on SOI), the monolayer sits at different heights depending
    on position (on-pillar vs suspended vs flat), giving DIFFERENT F_total at
    each position. Use this tool with different sio2_thickness_nm values to
    compare. Per Katrisioti et al. 2025 (Fig. 2e,f): interference alone does
    NOT explain the 3x PL enhancement on nanoantennas.

    Args:
        material: Material name (used for monolayer thickness estimate).
        excitation_nm: Excitation/laser wavelength in nm.
        emission_nm: Emission/detection wavelength in nm.
        sio2_thickness_nm: SiO2 layer thickness in nm (default 285).
                          For SOI nanoantenna substrates, use the local distance
                          from the monolayer to the Si substrate at the position
                          of interest: e.g., 145 nm (flat), 265 nm (on 120-nm pillar).
        substrate: Substrate material, currently supports "Si" (default).
        plot: Generate enhancement vs thickness plot (default True).

    Returns:
        Dict with F_exc, F_em, F_total, and optional plot URL.
    """
    logger.info(f"pl_substrate_enhancement: material={material}, exc={excitation_nm}, "
                f"em={emission_nm}, SiO2={sio2_thickness_nm} nm")
    try:
        if substrate.lower() != "si":
            return {"ok": False, "reason": f"Substrate '{substrate}' not yet supported. Currently only 'Si'."}
        if excitation_nm <= 0 or emission_nm <= 0:
            return {"ok": False, "reason": "Wavelengths must be positive."}
        if sio2_thickness_nm < 0 or sio2_thickness_nm > 1000:
            return {"ok": False, "reason": "SiO2 thickness must be 0-1000 nm."}

        def _fresnel_enhancement(wl_nm: float, d_sio2_nm: float) -> float:
            """Calculate field enhancement at air/SiO2 interface for air/SiO2/Si stack."""
            n_air = 1.0
            n_sio2 = _sio2_refractive_index(wl_nm)
            n_si = _si_refractive_index(wl_nm)

            # Fresnel reflection coefficients (normal incidence, TE=TM)
            r01 = (n_air - n_sio2) / (n_air + n_sio2)          # air -> SiO2
            r12 = (n_sio2 - n_si) / (n_sio2 + n_si)            # SiO2 -> Si (complex)
            t01 = 2 * n_air / (n_air + n_sio2)                  # transmission air -> SiO2

            # Phase accumulated in SiO2 layer (one-way)
            beta = 2 * np.pi * n_sio2 * d_sio2_nm / wl_nm

            # Total field at monolayer position (top of SiO2 = where 2D material sits)
            # Incident + reflected from SiO2/Si interface
            # E_at_surface = E_inc * (1 + r_eff) where r_eff is the effective reflection
            # from the SiO2/Si multilayer as seen from the air side
            phase_factor = np.exp(2j * beta)
            r_eff = (r01 + r12 * phase_factor) / (1 + r01 * r12 * phase_factor)

            # Field at the monolayer position: E = E_0 * (1 + r_eff)
            # where r_eff is the reflection coefficient of the full stack
            F = abs(1 + r_eff) ** 2
            return float(F)

        # Calculate at requested thickness
        F_exc = _fresnel_enhancement(excitation_nm, sio2_thickness_nm)
        F_em = _fresnel_enhancement(emission_nm, sio2_thickness_nm)
        F_total = F_exc * F_em

        result = {
            "ok": True,
            "material": material,
            "substrate_stack": f"air / SiO2 ({sio2_thickness_nm:.0f} nm) / Si",
            "excitation_nm": excitation_nm,
            "emission_nm": emission_nm,
            "sio2_thickness_nm": sio2_thickness_nm,
            "F_excitation": round(F_exc, 3),
            "F_emission": round(F_em, 3),
            "F_total": round(F_total, 3),
            "interpretation": (
                f"Signal is enhanced by {F_total:.1f}x compared to freestanding monolayer. "
                f"Excitation field enhanced {F_exc:.2f}x, emission collection enhanced {F_em:.2f}x."
            ),
        }

        # Optimal thickness search
        thicknesses = np.linspace(0, 500, 1000)
        f_totals = np.array([
            _fresnel_enhancement(excitation_nm, d) * _fresnel_enhancement(emission_nm, d)
            for d in thicknesses
        ])
        opt_idx = np.argmax(f_totals)
        result["optimal_sio2_nm"] = round(float(thicknesses[opt_idx]), 1)
        result["optimal_F_total"] = round(float(f_totals[opt_idx]), 3)

        # Common substrate comparison
        for d_common in [90, 145, 285, 300]:
            F_c = _fresnel_enhancement(excitation_nm, d_common) * _fresnel_enhancement(emission_nm, d_common)
            result[f"F_total_at_{d_common}nm"] = round(F_c, 3)

        # v3.0: SOI / nanoantenna geometry note
        result["soi_nanoantenna_note"] = {
            "description": (
                "On SOI substrates with Si nanopillars (e.g., Katrisioti et al. 2025), "
                "the monolayer sits at different heights above the bulk Si depending on position: "
                "(a) flat region: SiO2 spacer thickness (e.g., 145 nm), "
                "(b) on top of pillar: spacer + pillar height + cap (e.g., 145 + 95 + 30 = 270 nm), "
                "(c) suspended between pillars: intermediate height in air. "
                "Run this tool at each relevant thickness to compare Fresnel factors."
            ),
            "key_finding": (
                "Thin-film interference does NOT account for the ~3x PL enhancement observed "
                "on nanoantennas. The flat and suspended regions show similar F_total (~1-2), "
                "while the observed 3x PL gain arises from near-field coupling to Mie modes."
            ),
            "auger_limitation": (
                "On nanostructures, near-field enhancement can boost absorption by ~8x, "
                "but PL only increases ~3x because Auger recombination dominates at exciton "
                "densities >1e9 cm-2, capping the radiative yield (Katrisioti et al. 2025; "
                "Yang et al. Science 2015). Raman and SHG are NOT limited by this."
            ),
            "ref": "Katrisioti et al. arXiv:2504.03264 (Fig. 2e,f)",
        }

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})
            fig.patch.set_facecolor("#FAFBFC")
            for ax in (ax1, ax2):
                ax.set_facecolor("#FAFBFC")
                for spine in ax.spines.values():
                    spine.set_color("#B0B8C0")

            fig.suptitle(f"Substrate Enhancement — {material} on SiO2/Si",
                         fontsize=16, fontweight="bold", y=0.98)

            # Top panel: F_total vs thickness
            ax1.plot(thicknesses, f_totals, color="#1565C0", linewidth=2, label="F_total (PL)")
            ax1.axvline(sio2_thickness_nm, color="#E53935", linestyle="--", linewidth=2,
                        label=f"Your SiO2: {sio2_thickness_nm:.0f} nm (F={F_total:.2f})")
            ax1.axvline(thicknesses[opt_idx], color="#43A047", linestyle=":", linewidth=1.5,
                        label=f"Optimal: {thicknesses[opt_idx]:.0f} nm (F={f_totals[opt_idx]:.2f})")
            ax1.scatter([sio2_thickness_nm], [F_total], color="#E53935", s=100, zorder=5)
            ax1.set_ylabel("F_total (enhancement factor)", fontsize=12)
            ax1.legend(fontsize=11, framealpha=0.9)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 500)

            # Bottom panel: individual F_exc and F_em
            f_excs = np.array([_fresnel_enhancement(excitation_nm, d) for d in thicknesses])
            f_ems = np.array([_fresnel_enhancement(emission_nm, d) for d in thicknesses])
            ax2.plot(thicknesses, f_excs, color="#FB8C00", linewidth=1.5,
                     label=f"F_exc ({excitation_nm:.0f} nm)")
            ax2.plot(thicknesses, f_ems, color="#8E24AA", linewidth=1.5,
                     label=f"F_em ({emission_nm:.0f} nm)")
            ax2.axvline(sio2_thickness_nm, color="#E53935", linestyle="--", linewidth=1.5, alpha=0.5)
            ax2.set_xlabel("SiO2 thickness (nm)", fontsize=12)
            ax2.set_ylabel("Enhancement", fontsize=12)
            ax2.legend(fontsize=10, framealpha=0.9)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 500)

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            _, plot_url = _save_figure(fig, "substrate_enhancement")
            result["artifacts"] = {"enhancement_plot_png": plot_url}

        return result

    except Exception as e:
        logger.exception("pl_substrate_enhancement failed")
        return {"ok": False, "reason": str(e)}


# =============================================================================
#  TOOL 7: pl_nonlinear_plan — SHG / THG experiment planning
# =============================================================================
@mcp.tool()
def pl_nonlinear_plan(
    material: str,
    pump_nm: float,
    technique: str = "SHG",
    nanostructure_type: str = "",
    polarization_resolved: bool = False,
) -> Dict[str, Any]:
    """
    Plan a Second-Harmonic Generation (SHG) or Third-Harmonic Generation (THG)
    experiment for 2D materials.

    Handles the inverted filter logic (vs PL): short-pass to block fundamental,
    bandpass to collect harmonic signal. Also handles crystal symmetry, layer
    parity rules, and polarization-resolved (P-SHG/P-THG) setups.

    Args:
        material: Material name (e.g., "MoS2", "WS2", "SnS").
        pump_nm: Pump/fundamental laser wavelength in nm.
        technique: "SHG" (default) or "THG".
        nanostructure_type: Optional nanostructure description (e.g., "Si pillar D=300nm").
        polarization_resolved: If True, include P-SHG/P-THG setup guidance.

    Returns:
        Dict with harmonic wavelength, filter recommendations, symmetry info, and sketch.
    """
    logger.info(f"pl_nonlinear_plan: material={material}, pump={pump_nm}, tech={technique}")
    try:
        technique = technique.upper()
        if technique not in ("SHG", "THG"):
            return {"ok": False, "reason": f"Technique must be 'SHG' or 'THG', got '{technique}'."}
        if pump_nm <= 0:
            return {"ok": False, "reason": "Pump wavelength must be positive."}

        # Compute harmonic wavelength
        if technique == "SHG":
            harmonic_nm = pump_nm / 2.0
            harmonic_ev = _nm_to_ev(harmonic_nm)
            order = 2
        else:  # THG
            harmonic_nm = pump_nm / 3.0
            harmonic_ev = _nm_to_ev(harmonic_nm)
            order = 3

        # Look up material nonlinear properties
        mat_key = material.strip().lower().replace(" ", "")
        nl_data = NONLINEAR_DB.get(mat_key)
        curated = _lookup_curated(material)

        warnings = []
        symmetry_info = {}

        if nl_data:
            active_key = "shg_active" if technique == "SHG" else "thg_active"
            if not nl_data.get(active_key, False):
                warnings.append(f"WARNING: {material} may not be {technique}-active!")

            symmetry_info = {
                "point_group": nl_data.get("point_group", "unknown"),
                "crystal_system": nl_data.get("crystal_system", "unknown"),
                "layer_rule": nl_data.get("shg_layer_rule", ""),
                "twisted_bilayer": nl_data.get("twisted_bilayer", ""),
                "notes": nl_data.get("notes", ""),
            }

            if technique == "SHG" and "D3h" in nl_data.get("point_group", ""):
                warnings.append("D3h symmetry: SHG active ONLY in odd-layer TMDs (1L, 3L, 5L).")
                warnings.append("Bilayer with 0° twist: centrosymmetric -> SHG = 0.")
                warnings.append("Twisted bilayer: SHG restored ~ sin(3×twist_angle).")
        else:
            warnings.append(f"No nonlinear data for '{material}' in database — proceed with caution.")

        # Filter recommendations (INVERTED from PL!)
        filters = []

        # 1. Short-pass to block fundamental (most critical)
        sp_cutoff = harmonic_nm + 30  # allow 30 nm margin above harmonic
        best_sp = None
        for sp in SHORTPASS_FILTER_DB:
            if sp["cut_off_nm"] >= sp_cutoff and sp["cut_off_nm"] < pump_nm - 50:
                if best_sp is None or sp["cut_off_nm"] < best_sp["cut_off_nm"]:
                    best_sp = sp
        if best_sp:
            filters.append({"role": "Block fundamental", "filter": best_sp["label"],
                            "type": "short_pass", "note": f"Blocks pump at {pump_nm:.0f} nm (OD6+)"})
        else:
            filters.append({"role": "Block fundamental", "filter": f"Need SP~{sp_cutoff:.0f}",
                            "type": "short_pass", "note": f"Custom short-pass needed to block {pump_nm:.0f} nm"})

        # 2. Bandpass at harmonic for isolation
        harmonic_bp = None
        for f in FILTER_DB:
            if f["type"] == "band_pass":
                if abs(f["center_nm"] - harmonic_nm) < f.get("width_nm", 20):
                    harmonic_bp = f
                    break
        if harmonic_bp:
            filters.append({"role": f"Isolate {technique} signal", "filter": harmonic_bp["label"],
                            "type": "band_pass",
                            "note": f"Centers on {technique} at {harmonic_nm:.0f} nm"})
        else:
            filters.append({"role": f"Isolate {technique} signal",
                            "filter": f"Need BP{harmonic_nm:.0f}/20",
                            "type": "band_pass",
                            "note": f"Custom bandpass at {harmonic_nm:.0f} nm (FWHM ~20 nm)"})

        # 3. ND filters for pump power control
        filters.append({"role": "Pump power control", "filter": "ND filter wheel",
                        "type": "nd", "note": "Critical — 2D materials damage easily under pulsed excitation"})

        # Polarization-resolved setup
        pshg_info = {}
        if polarization_resolved:
            pshg_info = {
                "setup": [
                    "Half-wave plate (HWP) in pump beam — rotate to scan linear polarization angle",
                    "Polarizing beam splitter (PBS) before detector — resolve parallel and perpendicular components",
                    "Record intensity vs HWP angle: I_par(φ) and I_perp(φ)",
                    "For D3h TMDs: 6-fold pattern reveals armchair direction",
                    "For C2v (SnS): 2-fold pattern distinguishes armchair vs zigzag",
                ],
                "analysis": "Fit P-SHG polar plot to extract armchair orientation and detect strain-induced symmetry breaking",
                "strain_detection": "Strain breaks C3v -> distorted P-SHG pattern. Pixel-by-pixel fitting maps strain distribution.",
                "ref": "Psilodimitrakopoulos et al. 2019; Kourmoulakis et al. Sci. Rep. 2024",
            }

        # Nanostructure coupling notes
        nano_info = {}
        if nanostructure_type:
            nano_info["description"] = nanostructure_type
            nano_info["key_note"] = (f"For nanoantenna-coupled {technique}: LDOS enhancement at "
                                     f"harmonic energy ({harmonic_nm:.0f} nm / {harmonic_ev:.2f} eV) "
                                     f"is the key resonance target, NOT just pump field enhancement.")
            # Rough Mie resonance estimate for Si pillars
            if "si" in nanostructure_type.lower() and "d=" in nanostructure_type.lower():
                try:
                    d_str = nanostructure_type.lower().split("d=")[1]
                    d_nm = float(''.join(c for c in d_str if c.isdigit() or c == '.'))
                    mie_res_nm = 1.3 * d_nm  # rough magnetic dipole scaling
                    nano_info["estimated_mie_resonance_nm"] = round(mie_res_nm, 0)
                    nano_info["mie_note"] = (f"Estimated magnetic dipole resonance ~{mie_res_nm:.0f} nm "
                                             f"(rough scaling: λ ≈ 1.3×D). This is approximate — "
                                             f"verify with FDTD/GDM simulation.")
                except (ValueError, IndexError):
                    nano_info["mie_note"] = "Could not parse diameter. Format: 'Si pillar D=300nm'."

        # Check if harmonic overlaps with material exciton (resonant enhancement)
        resonance_note = ""
        if curated:
            for peak in curated.get("emission", []):
                if abs(peak["center_nm"] - harmonic_nm) < 30:
                    resonance_note = (f"NOTE: {technique} at {harmonic_nm:.0f} nm is near-resonant "
                                      f"with {peak['label']} ({peak['center_nm']} nm)! "
                                      f"Expect resonant enhancement of {technique} signal.")
                    break

        # Build result
        result = {
            "ok": True,
            "material": material,
            "technique": technique,
            "pump_nm": pump_nm,
            "pump_eV": round(_nm_to_ev(pump_nm), 3),
            f"{technique.lower()}_nm": round(harmonic_nm, 1),
            f"{technique.lower()}_eV": round(harmonic_ev, 3),
            "filter_prescription": filters,
            "symmetry": symmetry_info,
            "warnings": warnings,
        }
        if resonance_note:
            result["resonance_enhancement"] = resonance_note
        if pshg_info:
            result["polarization_resolved_setup"] = pshg_info
        if nano_info:
            result["nanostructure_coupling"] = nano_info

        # Generate spectral sketch
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#FAFBFC")
        ax.set_facecolor("#FAFBFC")
        for spine in ax.spines.values():
            spine.set_color("#B0B8C0")

        fig.suptitle(f"{technique} Plan — {material} (pump {pump_nm:.0f} nm → {harmonic_nm:.0f} nm)",
                     fontsize=15, fontweight="bold")

        x = np.linspace(min(harmonic_nm - 80, 350), max(pump_nm + 80, pump_nm + 100), 2000)

        # Pump line
        pump_profile = 0.85 * np.exp(-0.5 * ((x - pump_nm) / 3.0) ** 2)
        ax.fill_between(x, pump_profile, alpha=0.15, color="#E53935")
        ax.plot(x, pump_profile, color="#E53935", linewidth=2, label=f"Pump {pump_nm:.0f} nm")

        # Harmonic line
        harm_profile = 0.5 * np.exp(-0.5 * ((x - harmonic_nm) / 2.0) ** 2)
        ax.fill_between(x, harm_profile, alpha=0.2, color="#1565C0")
        ax.plot(x, harm_profile, color="#1565C0", linewidth=2,
                label=f"{technique} {harmonic_nm:.0f} nm ({harmonic_ev:.2f} eV)")

        # Short-pass filter region
        if best_sp:
            ax.axvspan(best_sp["cut_off_nm"], x[-1], alpha=0.12, color="#FFCDD2")
            ax.axvline(best_sp["cut_off_nm"], color="#C62828", linestyle="--", linewidth=2)
            ax.text(best_sp["cut_off_nm"] + 5, 0.75, f'{best_sp["label"]}\nBLOCKS',
                    fontsize=10, color="#C62828", fontweight="bold")

        ax.set_xlabel("Wavelength (nm)", fontsize=12)
        ax.set_ylabel("Relative Intensity", fontsize=12)
        ax.legend(fontsize=11, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        _, sketch_url = _save_figure(fig, f"{technique.lower()}_plan")
        result["artifacts"] = {"spectral_sketch_png": sketch_url}

        return result

    except Exception as e:
        logger.exception("pl_nonlinear_plan failed")
        return {"ok": False, "reason": str(e)}


# =============================================================================
#  TOOL 8: pl_strain_from_shift — Strain <-> spectral shift conversion
# =============================================================================
@mcp.tool()
def pl_strain_from_shift(
    material: str,
    mode: str = "PL_A_exciton",
    observed_peak_nm: float = 0,
    reference_peak_nm: float = 0,
    observed_peak_cm1: float = 0,
    reference_peak_cm1: float = 0,
    strain_percent: float = 0,
    strain_type: str = "biaxial",
) -> Dict[str, Any]:
    """
    Convert an observed spectral shift to strain, or predict the shift
    from a known strain value. Uses verified gauge factors from literature
    including Kioseoglou group measurements.

    Two operation modes:
    - Shift → Strain: provide observed_peak + reference_peak (or use DB default)
    - Strain → Shift: provide strain_percent to predict the peak position

    Args:
        material: Material name (e.g., "MoS2", "WS2").
        mode: Spectroscopic mode. Options:
              "PL_A_exciton", "PL_B_exciton",
              "Raman_E_prime", "Raman_A1_prime"
        observed_peak_nm: Observed PL peak in nm (for PL modes).
        reference_peak_nm: Unstrained reference peak in nm (0 = use DB).
        observed_peak_cm1: Observed Raman peak in cm-1 (for Raman modes).
        reference_peak_cm1: Unstrained reference in cm-1 (0 = use DB).
        strain_percent: If given, predicts peak shift (strain → shift mode).
        strain_type: "biaxial" (default) or "uniaxial".

    Returns:
        Dict with strain estimate, gauge factor used, and quality warnings.
    """
    logger.info(f"pl_strain_from_shift: material={material}, mode={mode}, "
                f"obs_nm={observed_peak_nm}, strain%={strain_percent}")
    try:
        # Build the lookup key
        mat_key = material.strip().lower().replace(" ", "")
        gauge_key = f"{mat_key}_{mode}"

        gauge = STRAIN_GAUGE_DB.get(gauge_key)
        if not gauge:
            available = [k for k in STRAIN_GAUGE_DB.keys() if k.startswith(mat_key)]
            if available:
                return {"ok": False,
                        "reason": f"Mode '{mode}' not found for {material}. Available: {available}"}
            else:
                return {"ok": False,
                        "reason": f"No strain gauge data for '{material}'. Available materials: "
                                  f"{sorted(set(k.split('_')[0] for k in STRAIN_GAUGE_DB.keys()))}"}

        is_raman = "Raman" in mode
        is_pl = "PL" in mode

        # Select gauge factor based on strain type
        if strain_type == "biaxial":
            if is_pl:
                gauge_factor = gauge.get("gauge_biaxial_meV_per_pct")
            else:
                gauge_factor = gauge.get("gauge_biaxial_cm1_per_pct")
        elif strain_type == "uniaxial":
            if is_pl:
                gauge_factor = gauge.get("gauge_uniaxial_meV_per_pct")
            else:
                gauge_factor = gauge.get("gauge_uniaxial_cm1_per_pct")
        else:
            return {"ok": False, "reason": f"strain_type must be 'biaxial' or 'uniaxial', got '{strain_type}'."}

        if gauge_factor is None:
            return {"ok": False, "reason": f"No {strain_type} gauge factor for {gauge_key}."}

        warnings = []
        result = {
            "ok": True,
            "material": gauge["material"],
            "mode": mode,
            "strain_type": strain_type,
            "gauge_factor": gauge_factor,
            "gauge_unit": "meV/%" if is_pl else "cm⁻¹/%",
            "reference": gauge.get("ref", ""),
        }

        # === MODE 1: Strain → Shift (predict peak position) ===
        if strain_percent != 0:
            if is_pl:
                ref_eV = gauge.get("reference_peak_eV", 0)
                ref_nm = gauge.get("reference_peak_nm", 0)
                if reference_peak_nm > 0:
                    ref_nm = reference_peak_nm
                    ref_eV = _nm_to_ev(ref_nm)

                shift_meV = gauge_factor * strain_percent
                new_eV = ref_eV + shift_meV / 1000.0
                new_nm = _ev_to_nm(new_eV) if new_eV > 0 else 0

                result["mode_type"] = "strain_to_shift"
                result["input_strain_pct"] = strain_percent
                result["reference_peak_eV"] = round(ref_eV, 4)
                result["reference_peak_nm"] = round(ref_nm, 1)
                result["predicted_shift_meV"] = round(shift_meV, 1)
                result["predicted_peak_eV"] = round(new_eV, 4)
                result["predicted_peak_nm"] = round(new_nm, 1)

            else:  # Raman
                ref_cm1 = gauge.get("reference_peak_cm1", 0)
                if reference_peak_cm1 > 0:
                    ref_cm1 = reference_peak_cm1

                shift_cm1 = gauge_factor * strain_percent
                new_cm1 = ref_cm1 + shift_cm1

                result["mode_type"] = "strain_to_shift"
                result["input_strain_pct"] = strain_percent
                result["reference_peak_cm1"] = round(ref_cm1, 1)
                result["predicted_shift_cm1"] = round(shift_cm1, 2)
                result["predicted_peak_cm1"] = round(new_cm1, 1)

        # === MODE 2: Shift → Strain (extract strain from peak) ===
        else:
            if is_pl:
                if observed_peak_nm <= 0:
                    return {"ok": False, "reason": "Provide observed_peak_nm for PL mode."}

                ref_nm = reference_peak_nm if reference_peak_nm > 0 else gauge.get("reference_peak_nm", 0)
                ref_eV = _nm_to_ev(ref_nm)
                obs_eV = _nm_to_ev(observed_peak_nm)
                shift_meV = (obs_eV - ref_eV) * 1000.0

                if gauge_factor == 0:
                    return {"ok": False, "reason": "Gauge factor is zero — cannot compute strain."}

                strain_pct = shift_meV / gauge_factor

                result["mode_type"] = "shift_to_strain"
                result["observed_peak_nm"] = observed_peak_nm
                result["observed_peak_eV"] = round(obs_eV, 4)
                result["reference_peak_nm"] = round(ref_nm, 1)
                result["reference_peak_eV"] = round(ref_eV, 4)
                result["shift_meV"] = round(shift_meV, 1)
                result["extracted_strain_pct"] = round(strain_pct, 3)
                result["strain_sign"] = "tensile" if strain_pct > 0 else "compressive" if strain_pct < 0 else "zero"

            else:  # Raman
                if observed_peak_cm1 <= 0:
                    return {"ok": False, "reason": "Provide observed_peak_cm1 for Raman mode."}

                ref_cm1 = reference_peak_cm1 if reference_peak_cm1 > 0 else gauge.get("reference_peak_cm1", 0)
                shift_cm1 = observed_peak_cm1 - ref_cm1

                if gauge_factor == 0:
                    return {"ok": False, "reason": "Gauge factor is zero — cannot compute strain."}

                strain_pct = shift_cm1 / gauge_factor

                result["mode_type"] = "shift_to_strain"
                result["observed_peak_cm1"] = observed_peak_cm1
                result["reference_peak_cm1"] = round(ref_cm1, 1)
                result["shift_cm1"] = round(shift_cm1, 2)
                result["extracted_strain_pct"] = round(strain_pct, 3)
                result["strain_sign"] = "tensile" if strain_pct > 0 else "compressive" if strain_pct < 0 else "zero"

        # Check K/Γ crossover warning
        strain_val = abs(strain_percent if strain_percent != 0 else result.get("extracted_strain_pct", 0))
        crossover = gauge.get("k_gamma_crossover_pct")
        if crossover and strain_val > crossover * 0.7:
            if strain_val >= crossover:
                warnings.append(
                    f"⚠ CRITICAL: Strain ({strain_val:.2f}%) exceeds K/Γ indirect crossover "
                    f"threshold (~{crossover}% biaxial for {gauge['material']}). "
                    f"Material is likely in the INDIRECT gap regime — PL will be quenched.")
            else:
                warnings.append(
                    f"Strain ({strain_val:.2f}%) is approaching K/Γ crossover "
                    f"threshold (~{crossover}% for {gauge['material']}). "
                    f"Watch for PL quenching as strain increases.")

        # Cross-check suggestion
        if is_pl and "A1_prime" not in mode:
            warnings.append(f"Cross-check: Raman A1' mode is strain-insensitive — use it as a control.")
        if is_raman and "E_prime" in mode:
            warnings.append("E' mode is in-plane sensitive. Combine with A1' (out-of-plane, insensitive) for validation.")

        result["warnings"] = warnings
        return result

    except Exception as e:
        logger.exception("pl_strain_from_shift failed")
        return {"ok": False, "reason": str(e)}


# =============================================================================
#  TOOL 9: pl_ple_plan — PLE experiment planning
# =============================================================================
@mcp.tool()
def pl_ple_plan(
    material: str,
    detection_nm: float,
    excitation_range_nm: Optional[List[float]] = None,
    step_nm: float = 0,
    nanostructure_type: str = "",
) -> Dict[str, Any]:
    """
    Plan a Photoluminescence Excitation (PLE) experiment. In PLE, you fix
    the detection wavelength and scan the excitation wavelength to map
    absorption resonances, Mie resonances, or excitonic fine structure.

    v3.0: Now uses ABSORPTION peaks (not just emission) for avoidance windows.
    Adds near-to-far-field offset warning and dark-field recommendation
    for nanostructure-coupled samples (per Katrisioti et al. 2025).

    Args:
        material: Material name (e.g., "MoS2", "WS2").
        detection_nm: Fixed emission wavelength to monitor (nm).
        excitation_range_nm: [min, max] scan range in nm. Auto-suggested if not given.
        step_nm: Excitation step size in nm (0 = auto-suggest).
        nanostructure_type: Optional (e.g., "Si pillar D=300nm").

    Returns:
        Dict with scan plan, wavelengths to avoid, laser requirements, and timing estimate.
    """
    logger.info(f"pl_ple_plan: material={material}, det={detection_nm}")
    try:
        curated = _lookup_curated(material)
        if not curated:
            return {"ok": False, "reason": f"Material '{material}' not found in PL database."}

        emission_peaks = curated.get("emission", [])
        absorption_peaks = curated.get("absorption", [])
        mat_name = curated.get("name", material)
        warnings = []
        avoidance_windows = []

        # v3.0: Use ABSORPTION peaks for avoidance (more relevant for PLE).
        # Fall back to emission peaks if no absorption data available.
        avoidance_source = absorption_peaks if absorption_peaks else emission_peaks
        avoidance_label = "absorption" if absorption_peaks else "emission (no absorption data; using as proxy)"

        for peak in avoidance_source:
            peak_nm = peak["center_nm"]
            fwhm = peak.get("fwhm_nm", 20)
            avoidance_min = peak_nm - max(10, fwhm / 2)
            avoidance_max = peak_nm + max(10, fwhm / 2)
            avoidance_windows.append({
                "label": peak["label"],
                "center_nm": peak_nm,
                "avoid_range_nm": [round(avoidance_min, 1), round(avoidance_max, 1)],
                "reason": f"Near-resonant excitation at {peak['label']} causes artefacts (re-emission, spectral overlap)",
            })

        # Also avoid emission peaks if not already covered by absorption
        if absorption_peaks:
            for peak in emission_peaks:
                peak_nm = peak["center_nm"]
                already_covered = any(
                    abs(peak_nm - aw["center_nm"]) < 20 for aw in avoidance_windows
                )
                if not already_covered:
                    fwhm = peak.get("fwhm_nm", 20)
                    avoidance_windows.append({
                        "label": f"{peak['label']} (emission)",
                        "center_nm": peak_nm,
                        "avoid_range_nm": [round(peak_nm - max(10, fwhm / 2), 1),
                                           round(peak_nm + max(10, fwhm / 2), 1)],
                        "reason": f"Emission peak overlap — scattered emission contaminates PLE signal",
                    })

        # Auto-suggest excitation range if not provided
        if excitation_range_nm and len(excitation_range_nm) == 2:
            exc_min, exc_max = excitation_range_nm
        else:
            # Default: from ~200 meV above gap down to ~100 nm below detection
            gap_ev = curated.get("gap_ev")
            if gap_ev:
                gap_nm = _ev_to_nm(gap_ev)
                exc_min = max(350, gap_nm - 150)  # start well above gap
                exc_max = detection_nm - 20  # stop before detection
            else:
                exc_min = 400
                exc_max = detection_nm - 20

            # v3.0: If nanostructure present, extend range to cover expected Mie resonances
            if nanostructure_type and "d=" in nanostructure_type.lower():
                try:
                    d_str = nanostructure_type.lower().split("d=")[1]
                    d_nm = float(''.join(c for c in d_str if c.isdigit() or c == '.'))
                    md_res = 1.3 * d_nm
                    if md_res < exc_min:
                        exc_min = max(350, md_res - 30)
                        warnings.append(f"Extended scan range down to {exc_min:.0f} nm to cover estimated MD Mie resonance at {md_res:.0f} nm.")
                except (ValueError, IndexError):
                    pass

            excitation_range_nm = [round(exc_min, 0), round(exc_max, 0)]

        exc_min, exc_max = excitation_range_nm[0], excitation_range_nm[1]

        # Auto-suggest step size
        if step_nm <= 0:
            if nanostructure_type:
                step_nm = 2.0  # Fine step for Mie resonance resolution
                step_reason = "Fine step (2 nm) for resolving Mie/nanostructure resonances"
            else:
                step_nm = 5.0  # Standard step
                step_reason = "Standard step (5 nm) for excitonic features"
        else:
            step_reason = "User-specified"

        n_steps = int(abs(exc_max - exc_min) / step_nm) + 1

        # Laser source recommendation
        scan_range = exc_max - exc_min
        laser_recommendation = []
        if scan_range > 200:
            laser_recommendation.append("Monochromated broadband source (Xe lamp + monochromator)")
            laser_recommendation.append("Supercontinuum laser + tunable filter")
        if exc_min >= 700 and exc_max <= 1000:
            laser_recommendation.append("Ti:Sapphire (700-1000 nm tunable)")
        if scan_range < 100:
            laser_recommendation.append("Set of discrete laser diodes with appropriate wavelengths")

        if not laser_recommendation:
            laser_recommendation.append("OPO (optical parametric oscillator) for wide tunability")
            laser_recommendation.append("Supercontinuum laser + AOTF/LCTF filter")

        # Filter requirements at each step
        filter_notes = [
            "Detection: Fixed bandpass at detection wavelength — do NOT change during scan",
            f"Recommended: BP{detection_nm:.0f}/20 (FWHM 20 nm centered on {detection_nm:.0f} nm)",
            "Excitation: At each step, the laser rejection filter must block the current excitation wavelength",
            "Option A: Set of notch filters (one per laser line) — simpler but more filters",
            "Option B: Tunable notch filter (liquid crystal or AOTF) — expensive but flexible",
            "Option C: Spatial filter in spectrometer + edge filter — most common in practice",
        ]

        # Nanostructure resonance estimation
        nano_info = {}
        if nanostructure_type:
            nano_info["description"] = nanostructure_type
            if "si" in nanostructure_type.lower() and "d=" in nanostructure_type.lower():
                try:
                    d_str = nanostructure_type.lower().split("d=")[1]
                    d_nm = float(''.join(c for c in d_str if c.isdigit() or c == '.'))
                    md_res = 1.3 * d_nm  # magnetic dipole
                    ed_res = 1.0 * d_nm  # electric dipole (rougher)
                    nano_info["estimated_resonances"] = {
                        "magnetic_dipole_nm": round(md_res, 0),
                        "electric_dipole_nm": round(ed_res, 0),
                        "note": "Rough Mie scaling. MD: lambda ~1.3*D, ED: lambda ~D. Verify with simulation (e.g., pyGDM).",
                    }
                    warnings.append(f"Focus PLE scan around {ed_res:.0f}-{md_res:.0f} nm for Mie resonances.")
                except (ValueError, IndexError):
                    pass

            # v3.0: Near-to-far-field offset warning (Katrisioti et al. 2025, Fig. 2c-d)
            nano_info["near_to_far_field_offset"] = {
                "typical_shift_meV": "80-100",
                "direction": "PLE peaks are redshifted 80-100 meV relative to dark-field (far-field) scattering peaks",
                "cause": "Near-field and far-field spectral responses of Mie resonators differ due to interference of multipolar modes",
                "ref": "Katrisioti et al. arXiv:2504.03264 (Fig. 2c,d); Alonso-Gonzalez et al. PRL 2013; Miroshnichenko et al. Nat. Comms 2015",
            }

            # v3.0: Recommend complementary dark-field measurement
            nano_info["complementary_measurements"] = [
                {
                    "technique": "Dark-field (DF) scattering spectroscopy",
                    "purpose": "Map far-field Mie resonance positions of each nanoantenna type",
                    "setup": "Pinhole in detection path (confocal) → spectrometer. Broadband illumination (halogen lamp).",
                    "why": "DF peaks identify which Mie modes are present. PLE peaks should appear redshifted by 80-100 meV.",
                    "ref": "Katrisioti et al. 2025, Fig. 2d; Estrada-Real et al. Commun. Phys. 2023",
                },
                {
                    "technique": "AFM topography",
                    "purpose": "Verify monolayer conformality on nanostructures and measure pillar height/diameter",
                    "why": "MoS2 wrinkle/bubble formation affects local strain (see Katrisioti Suppl. Fig. S1)",
                },
            ]

            # v3.0: Auger saturation warning
            nano_info["auger_warning"] = (
                "IMPORTANT: On nanostructures, near-field enhancement increases absorption (up to ~8x) "
                "but PL enhancement saturates at ~3x due to Auger recombination at exciton densities "
                ">1e9 cm-2 (Wang et al. Nano Lett. 2015; Yang et al. Science 2015; Katrisioti et al. 2025). "
                "This is NOT a setup error — it reflects the fundamental exciton density-dependent dynamics. "
                "Raman enhancement (2-8x) and SHG enhancement (20-30x) are not limited by this effect."
            )

        # Time estimate
        time_per_step_s = 30  # typical: adjust laser + acquire spectrum
        total_time_min = n_steps * time_per_step_s / 60

        result = {
            "ok": True,
            "material": mat_name,
            "experiment": "Photoluminescence Excitation (PLE)",
            "detection_nm": detection_nm,
            "excitation_scan": {
                "range_nm": excitation_range_nm,
                "step_nm": step_nm,
                "step_reason": step_reason,
                "n_steps": n_steps,
            },
            "avoidance_windows": avoidance_windows,
            "avoidance_source": avoidance_label,
            "laser_sources": laser_recommendation,
            "filter_requirements": filter_notes,
            "estimated_session_time_min": round(total_time_min, 0),
            "warnings": warnings,
        }
        if nano_info:
            result["nanostructure"] = nano_info

        # Scan plan table
        scan_plan = []
        wl = exc_min
        while wl <= exc_max:
            in_avoidance = False
            for aw in avoidance_windows:
                if aw["avoid_range_nm"][0] <= wl <= aw["avoid_range_nm"][1]:
                    in_avoidance = True
                    break
            scan_plan.append({
                "excitation_nm": round(wl, 1),
                "excitation_eV": round(_nm_to_ev(wl), 3),
                "flag": "AVOID (near exciton/absorption)" if in_avoidance else "OK",
            })
            wl += step_nm

        result["scan_plan_summary"] = {
            "total_wavelengths": len(scan_plan),
            "safe_wavelengths": sum(1 for s in scan_plan if s["flag"] == "OK"),
            "flagged_wavelengths": sum(1 for s in scan_plan if "AVOID" in s["flag"]),
            "first_5_steps": scan_plan[:5],
            "last_5_steps": scan_plan[-5:] if len(scan_plan) > 5 else [],
        }

        return result

    except Exception as e:
        logger.exception("pl_ple_plan failed")
        return {"ok": False, "reason": str(e)}


# =============================================================================
#  TOOL 10: pl_imaging_plan — Wide-field / confocal / hyperspectral PL
# =============================================================================
@mcp.tool()
def pl_imaging_plan(
    material: str,
    excitation_nm: float,
    imaging_mode: str = "widefield",
    objective_na: float = 0.9,
    target_emission_nm: float = 0,
    field_of_view_um: float = 50.0,
) -> Dict[str, Any]:
    """
    Plan a spatially-resolved PL imaging experiment. Supports wide-field
    (flood illumination), confocal mapping, hyperspectral, and nonlinear
    imaging (SHG/THG mapping) modes.

    v3.0: Added nonlinear_map mode for SHG/THG imaging (per Katrisioti et al.
    2025, Fig. 4a — SHG intensity mapping on nanoantenna arrays).

    Args:
        material: Material name.
        excitation_nm: Excitation laser wavelength in nm.
        imaging_mode: "widefield" (default), "confocal_map", "hyperspectral", or "nonlinear_map".
        objective_na: Objective numerical aperture (default 0.9).
        target_emission_nm: Specific emission peak to image (0 = auto from DB).
        field_of_view_um: Target field of view in um (default 50).

    Returns:
        Dict with filter prescription, resolution estimate, detector advice,
        and mode-specific recommendations.
    """
    logger.info(f"pl_imaging_plan: material={material}, mode={imaging_mode}, exc={excitation_nm}")
    try:
        curated = _lookup_curated(material)
        if not curated:
            mp_data = _mp_bandgap(material)
            if mp_data and mp_data.get("band_gap_ev"):
                em_ev = mp_data["band_gap_ev"] - 0.05
                emission_peaks = [{"label": "Estimated PL", "center_nm": round(_ev_to_nm(em_ev)),
                                   "center_ev": round(em_ev, 3), "fwhm_nm": 30}]
                mat_name = mp_data["formula"]
            else:
                return {"ok": False, "reason": f"Material '{material}' not found."}
        else:
            emission_peaks = curated.get("emission", [])
            mat_name = curated.get("name", material)

        # Target emission
        if target_emission_nm > 0:
            em_nm = target_emission_nm
        elif emission_peaks:
            em_nm = emission_peaks[0]["center_nm"]  # primary peak
        else:
            return {"ok": False, "reason": "No emission data available."}

        warnings = []
        imaging_mode = imaging_mode.lower()

        # Diffraction limit
        diff_limit_nm = 610 * em_nm / (objective_na * 1000)  # in nm, then convert
        diff_limit_um = 0.61 * (em_nm / 1000.0) / objective_na  # in µm

        # Filter prescription for IMAGING (stricter than spectroscopy)
        filters = []

        # Bandpass at emission (CRITICAL for imaging — LP alone is insufficient)
        bp_center = em_nm
        bp_width = 30  # typical for exciton isolation
        filters.append({
            "role": "Emission isolation (CRITICAL for imaging)",
            "recommendation": f"BP{bp_center:.0f}/{bp_width} (bandpass {bp_center:.0f}±{bp_width//2} nm)",
            "note": "LP filter ALONE is insufficient for imaging — background across detector kills contrast. "
                    "Must use bandpass to select specific emission peak.",
        })

        # Notch/LP for laser rejection
        best_notch = None
        for f in FILTER_DB:
            if f["type"] == "notch" and abs(f["center_nm"] - excitation_nm) < 20:
                best_notch = f
                break
        if best_notch:
            filters.append({"role": "Laser rejection", "recommendation": best_notch["label"],
                            "note": f"Blocks laser scatter at {excitation_nm:.0f} nm"})
        else:
            best_lp = None
            for f in FILTER_DB:
                if f["type"] == "long_pass" and f["cut_on_nm"] > excitation_nm and f["cut_on_nm"] < em_nm - 20:
                    best_lp = f
                    break
            if best_lp:
                filters.append({"role": "Laser rejection", "recommendation": best_lp["label"],
                                "note": f"Blocks laser scatter below {best_lp['cut_on_nm']} nm"})

        result = {
            "ok": True,
            "material": mat_name,
            "imaging_mode": imaging_mode,
            "excitation_nm": excitation_nm,
            "target_emission_nm": round(em_nm, 1),
            "objective_NA": objective_na,
            "diffraction_limit_um": round(diff_limit_um, 3),
            "filter_prescription": filters,
        }

        # Mode-specific recommendations
        if imaging_mode == "widefield":
            result["illumination"] = {
                "type": "Flood illumination (defocused beam or Köhler illumination)",
                "notes": [
                    "Defocus laser beam to illuminate full field uniformly",
                    "Or use Köhler illumination with lamp + excitation filter",
                    "Check illumination uniformity — flat-field correction may be needed",
                ],
            }
            result["detector"] = {
                "recommendation": "sCMOS preferred over EMCCD for large-area uniformity",
                "notes": [
                    "sCMOS: better uniformity, larger pixel count, lower read noise",
                    "EMCCD: better for very weak signals (single photon sensitivity)",
                    "For 2D materials mapping at RT: sCMOS is usually sufficient",
                ],
            }
            result["acquisition"] = {
                "typical_exposure_ms": "100-1000 ms per frame (material dependent)",
                "spatial_resolution_um": round(diff_limit_um, 3),
                "field_of_view_um": field_of_view_um,
                "pixel_size_recommendation": f"< {diff_limit_um/2:.3f} µm (Nyquist sampling)",
            }

        elif imaging_mode == "confocal_map":
            spot_size_um = diff_limit_um
            pixels_per_line = int(field_of_view_um / (spot_size_um / 2)) + 1  # Nyquist
            total_pixels = pixels_per_line ** 2
            time_per_pixel_ms = 100  # typical
            scan_time_min = total_pixels * time_per_pixel_ms / 60000

            result["scan_parameters"] = {
                "spot_size_um": round(spot_size_um, 3),
                "step_size_um": round(spot_size_um / 2, 3),
                "pixels_per_line": pixels_per_line,
                "total_pixels": total_pixels,
                "time_per_pixel_ms": time_per_pixel_ms,
                "estimated_scan_time_min": round(scan_time_min, 1),
                "field_of_view_um": field_of_view_um,
            }
            result["notes"] = [
                "Confocal gives spectral information at each pixel (full spectrum)",
                "Much slower than widefield but spectrally resolved",
                f"For {field_of_view_um}×{field_of_view_um} µm² at Nyquist: ~{scan_time_min:.0f} min",
            ]

        elif imaging_mode == "hyperspectral":
            result["setup"] = {
                "options": [
                    "Tunable filter (LCTF/AOTF) + CCD: image at each wavelength sequentially",
                    "Pushbroom spectrometer: line scan with full spectrum per pixel",
                    "Confocal raster + spectrometer: slowest but most flexible",
                ],
                "recommended_for_2d_materials": "Tunable filter + sCMOS if mapping peak position; "
                                                 "pushbroom if need full lineshape at each point",
            }
            result["spectral_resolution"] = "Tunable filter: ~2-5 nm; spectrometer: <1 nm"

        elif imaging_mode == "nonlinear_map":
            # v3.0: SHG/THG imaging mode (Katrisioti et al. 2025, Fig. 4a)
            shg_nm = excitation_nm / 2.0
            thg_nm = excitation_nm / 3.0

            result["nonlinear_imaging"] = {
                "pump_nm": excitation_nm,
                "shg_nm": round(shg_nm, 1),
                "thg_nm": round(thg_nm, 1),
                "technique_notes": [
                    "SHG imaging: raster-scan pump beam, collect SHG signal at each pixel",
                    f"SHG signal at {shg_nm:.0f} nm (from {excitation_nm:.0f} nm pump)",
                    f"THG signal at {thg_nm:.0f} nm (from {excitation_nm:.0f} nm pump)",
                ],
                "filter_chain": {
                    "step_1": f"Short-pass filter SP{shg_nm + 50:.0f} to block fundamental at {excitation_nm:.0f} nm (OD6+)",
                    "step_2": f"Bandpass BP{shg_nm:.0f}/20 to isolate SHG harmonic at {shg_nm:.0f} nm",
                    "note": "INVERTED filter logic vs PL: short-pass blocks long-wavelength pump, bandpass selects harmonic",
                },
                "detector": {
                    "SHG": f"{'Si CCD/sCMOS suitable' if shg_nm < 900 else 'InGaAs required'} for SHG at {shg_nm:.0f} nm",
                    "THG": f"{'Si CCD/sCMOS suitable' if thg_nm > 350 else 'UV-sensitive detector needed'} for THG at {thg_nm:.0f} nm",
                },
                "spatial_resolution": {
                    "diffraction_limit_um": round(0.61 * (shg_nm / 1000.0) / objective_na, 3),
                    "note": "Resolution set by SHG wavelength (shorter than pump), giving better resolution than PL imaging",
                },
                "key_physics": [
                    "SHG enhancement scales as |E_pump|^4 x LDOS(SHG) — much stronger than PL enhancement",
                    "On Si nanoantennas: 20-30x SHG enhancement observed (Katrisioti et al. 2025)",
                    "Strain does NOT significantly affect total SHG intensity (Mennel et al. APL Photonics 2019)",
                    "Odd-layer TMDs only (D3h); bilayer centrosymmetric → SHG = 0",
                    "SHG map directly reveals nanoantenna positions with high contrast",
                ],
            }

            # Scan time estimate for nonlinear map
            spot_size_um = 0.61 * (excitation_nm / 1000.0) / objective_na  # pump diffraction limit
            pixels = int(field_of_view_um / (spot_size_um / 2)) ** 2
            result["nonlinear_imaging"]["scan_estimate"] = {
                "pump_spot_um": round(spot_size_um, 3),
                "step_um": round(spot_size_um / 2, 3),
                "total_pixels": pixels,
                "time_per_pixel_ms": 200,  # longer integration for weak SHG
                "estimated_time_min": round(pixels * 200 / 60000, 1),
            }

        else:
            result["warning"] = (
                f"Unknown imaging mode '{imaging_mode}'. "
                "Supported: 'widefield', 'confocal_map', 'hyperspectral', 'nonlinear_map'."
            )

        # Key warning for 2D materials imaging
        warnings.extend([
            "2D materials on substrates: non-uniform strain from substrate topography, "
            "bubbles, and wrinkles will appear as spatial PL intensity AND peak position variations. "
            "Any spatial map should be interpreted with this in mind.",
            "If mapping PEAK POSITION: confocal or hyperspectral mode required (widefield only gives intensity).",
        ])

        if em_nm > 900:
            warnings.append(f"Emission at {em_nm:.0f} nm: Si CCD sensitivity drops. Consider InGaAs camera.")

        result["warnings"] = warnings
        return result

    except Exception as e:
        logger.exception("pl_imaging_plan failed")
        return {"ok": False, "reason": str(e)}


# =============================================================================
#  TOOL 11: pl_valley_polarization — Valley polarization experiment planning
# =============================================================================
@mcp.tool()
def pl_valley_polarization(
    material: str,
    excitation_nm: float = 0,
    temperature_K: float = 300,
    doping: str = "intrinsic",
    substrate: str = "SiO2/Si",
) -> Dict[str, Any]:
    """
    Plan a spin-valley polarization measurement for 2D TMD monolayers.
    This is the core technique of the Kioseoglou group: circularly polarized
    excitation → detect circular polarization degree (Pc) of emission to probe
    valley physics.

    Args:
        material: TMD material (MoS2, WS2, MoSe2, WSe2).
        excitation_nm: Excitation wavelength in nm (0 = auto-recommend).
        temperature_K: Sample temperature in K (default 300 = room temperature).
        doping: "intrinsic" (default), "n-type", "p-type", or "photochlorinated".
        substrate: Substrate description (default "SiO2/Si").

    Returns:
        Dict with expected Pc range, optimal excitation, excitonic species,
        setup requirements, and physics notes.
    """
    logger.info(f"pl_valley_polarization: material={material}, exc={excitation_nm}, T={temperature_K}K")
    try:
        mat_key = material.strip().lower().replace(" ", "")
        vp_data = VALLEY_POL_DB.get(mat_key)

        if not vp_data:
            available = list(VALLEY_POL_DB.keys())
            return {"ok": False,
                    "reason": f"Valley polarization data not available for '{material}'. "
                              f"Available: {available}. VP is only defined for TMD monolayers."}

        curated = _lookup_curated(material)
        warnings = []
        notes = list(vp_data.get("notes", []))

        # Determine excitation regime
        X0_nm = vp_data["X0_nm"]
        X0_eV = vp_data["X0_eV"]

        if excitation_nm > 0:
            detuning_nm = X0_nm - excitation_nm  # positive = above resonance
            detuning_eV = _nm_to_ev(excitation_nm) - X0_eV
            if abs(detuning_nm) < 15:
                exc_regime = "near-resonant"
            elif detuning_nm > 0 and detuning_nm < 60:
                exc_regime = "quasi-resonant"
            elif detuning_nm > 60:
                exc_regime = "non-resonant"
            else:
                exc_regime = "below-resonance"
                warnings.append(f"⚠ Excitation ({excitation_nm:.0f} nm) is below the A exciton "
                                f"({X0_nm} nm). This will NOT excite PL efficiently.")
        else:
            # Auto-recommend
            exc_regime = "near-resonant (recommended)"
            excitation_nm = X0_nm - 20  # ~20 nm above exciton
            notes.append(f"Auto-selected excitation: {excitation_nm:.0f} nm (~20 nm above A exciton)")

        # Estimate Pc range based on regime and temperature
        if temperature_K <= 10:
            T_regime = "cryogenic"
            if "nearres" in exc_regime or "near-resonant" in exc_regime:
                pc_range = vp_data["typical_Pc_4K_nearres_pct"]
            else:
                pc_range = (pc_range_low := vp_data["typical_Pc_4K_nearres_pct"][0] // 2,
                            vp_data["typical_Pc_4K_nearres_pct"][1] // 2)
        elif temperature_K <= 100:
            T_regime = "low_temperature"
            base_4K = vp_data["typical_Pc_4K_nearres_pct"]
            base_RT = vp_data["typical_Pc_RT_nearres_pct"]
            # Interpolate roughly
            frac = (temperature_K - 4) / (300 - 4)
            low = int(base_4K[0] * (1 - frac) + base_RT[0] * frac)
            high = int(base_4K[1] * (1 - frac) + base_RT[1] * frac)
            pc_range = (low, high)
        else:
            T_regime = "room_temperature"
            if "near-resonant" in exc_regime:
                pc_range = vp_data["typical_Pc_RT_nearres_pct"]
            else:
                pc_range = vp_data["typical_Pc_RT_nonres_pct"]

        # Doping effects
        doping_notes = []
        if doping == "n-type":
            doping_notes.append("n-type doping: X⁻ (negative trion) dominates low-energy PL.")
            doping_notes.append("X⁻ can show OPPOSITE Pc sign from X⁰ under certain conditions.")
        elif doping == "p-type":
            doping_notes.append("p-type doping: X⁺ (positive trion) appears.")
            if mat_key == "wse2":
                doping_notes.append("In WSe2: p-doping increases Pc by up to 3x (Katsipoulaki et al. 2025).")
                pc_range = (pc_range[0], min(pc_range[1] * 2, 90))
        elif doping == "photochlorinated":
            doping_notes.append("Photochlorination: progressive Cl incorporation tunes carrier density.")
            doping_notes.append("Single-shot pulses give controlled doping steps (no gate voltage needed).")
            if mat_key in ("wse2", "ws2"):
                doping_notes.append(f"For {vp_data['material']}: Pc tunable by photochlorination "
                                    f"(Katsipoulaki et al. 2D Mater. 2023, Adv. Opt. Mater. 2025).")

        # Setup requirements
        setup = {
            "excitation_optics": [
                "Circularly polarized excitation: linear polarizer + quarter-wave plate (QWP)",
                "QWP fast axis at 45° to polarizer → σ+ or σ- circular polarization",
                "Rotate QWP by 90° to switch helicity (σ+ ↔ σ-)",
            ],
            "detection_optics": [
                "Detection: QWP + linear polarizer (analyzer) before spectrometer entrance",
                "Record co-polarized (σ+/σ+) and cross-polarized (σ+/σ-) spectra separately",
                "Or use photoelastic modulator (PEM) for simultaneous lock-in detection",
            ],
            "formula": "Pc = (I_σ+ − I_σ−) / (I_σ+ + I_σ−)",
            "critical_alignment": [
                "QWP must be precisely at 45° — small errors drastically reduce measured Pc",
                "Verify circular polarization quality with a reference polarizer",
                "Backreflection from substrate can introduce artefacts — use tilted geometry if needed",
            ],
        }

        # Excitonic species identification
        species = []
        if curated:
            for peak in curated.get("emission", []):
                species.append({
                    "label": peak["label"],
                    "position_nm": peak["center_nm"],
                    "position_eV": peak.get("center_ev", round(_nm_to_ev(peak["center_nm"]), 3)),
                })
        # Add trion
        trion_eV = X0_eV - vp_data["trion_binding_meV"] / 1000.0
        species.append({
            "label": f"Trion ({vp_data['trion_type']})",
            "position_nm": round(_ev_to_nm(trion_eV), 1),
            "position_eV": round(trion_eV, 3),
            "note": f"Binding energy: {vp_data['trion_binding_meV']} meV below X⁰",
        })

        # Substrate effect
        substrate_note = ""
        if "graphite" in substrate.lower() or "graphene" in substrate.lower():
            substrate_note = ("Graphite/graphene substrate: can enhance and stabilize valley polarization "
                              "at RT (Demeridou et al. 2D Mater. 2023). Dielectric screening modifies "
                              "exchange interaction strength.")
        elif "hbn" in substrate.lower() or "bn" in substrate.lower():
            substrate_note = ("hBN encapsulation: sharpens excitonic lines and reduces inhomogeneous "
                              "broadening. May enhance Pc through reduced disorder scattering.")

        result = {
            "ok": True,
            "material": vp_data["material"],
            "experiment": "Spin-Valley Polarization Measurement",
            "excitation_nm": round(excitation_nm, 1),
            "excitation_regime": exc_regime,
            "temperature_K": temperature_K,
            "temperature_regime": T_regime,
            "doping": doping,
            "expected_Pc_range_pct": list(pc_range),
            "Pc_interpretation": (
                f"Expected circular polarization degree: {pc_range[0]}-{pc_range[1]}% "
                f"({T_regime}, {exc_regime} excitation, {doping} doping)"
            ),
            "dominant_depolarization": vp_data["dominant_depolarization"],
            "spin_orbit_splitting_meV": vp_data["spin_orbit_valence_meV"],
            "excitonic_species": species,
            "optimal_excitation": vp_data["optimal_excitation"],
            "setup_requirements": setup,
            "doping_effects": doping_notes,
            "physics_notes": notes,
            "warnings": warnings,
            "reference": vp_data.get("ref", ""),
        }
        if substrate_note:
            result["substrate_effect"] = substrate_note

        return result

    except Exception as e:
        logger.exception("pl_valley_polarization failed")
        return {"ok": False, "reason": str(e)}


# =============================================================================
#  Server entry point
# =============================================================================
if __name__ == "__main__":
    _mode = os.getenv("MODE", "stdio").lower()
    logger.info(
        f"Starting PL assistant server v2.0 (mode={_mode}, "
        f"mp-api={'yes' if HAS_MP_API else 'no'}, "
        f"MP_KEY={'set' if MP_API_KEY else 'unset'}, "
        f"curated_materials={len(PL_MATERIALS_DB)})"
    )
    if _mode == "http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")