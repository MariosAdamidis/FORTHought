# XRD Analysis Skill

## When to Use
Use these tools when the user uploads an X-ray diffraction (XRD) data file and wants:
- **Phase identification**: What crystalline phase(s) are in the sample
- **Purity analysis**: How pure is the sample, are there extra peaks
- **Impurity detection**: Do unassigned peaks come from reactants/precursors
- **Pattern plotting**: Annotated XRD plots with peak labels
- **Quick QC**: Just check peak count and data quality, no database search
- **Reference lookup**: See what a material's pattern should look like before measuring
- **Origin export**: Get processed data into OriginLab for further work

Keywords: XRD, diffraction, crystal, phase, purity, Bragg, 2-theta, powder diffraction, Bruker, .brml, .raw, .xy

## Supported File Formats
- `.xy` / `.dat` / `.csv` / `.txt` — ASCII 2-column (2θ, intensity). Priority format, always works.
- `.brml` — Bruker XML/ZIP archive from D8 Advance / DIFFRAC.EVA. Parsed automatically.
- `.raw` — Bruker RAW v4 binary (D8 Advance native). Reverse-engineered parser, validated to ±0.001 CPS against .xy exports.

Auto-detection by extension and magic bytes. If a format fails, ask the user to export to `.xy` from DIFFRAC.EVA or Bruker software.

## Available Tools (5)

### 1. `xrd.identify` — Full analysis pipeline (PRIMARY)
Parse → search COD/MP → match phases → purity → impurity detection. **This is the main tool.**

```
run(tool="xrd.identify", args={
    "file_id": "<file_id>",
    "composition_hint": "CsPbBr3",
    "reactants": "CsBr,PbBr2",
    "wavelength": "CuKa",
    "database": "cod"
})
```

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `file_id` | Yes* | — | OWUI file ID from upload |
| `file_path` | Yes* | — | OR direct path. Provide one of the two. |
| `composition_hint` | **Strongly recommended** | `""` | Expected formula (e.g. `"TiO2"`, `"CsPbBr3,CsBr"`). Without this, only peak detection runs — no phase matching. Comma-separate multiple candidates. |
| `reactants` | No | `""` | Comma-separated reactant formulas for impurity detection (e.g. `"CsBr,PbBr2"`). |
| `wavelength` | No | `"CuKa"` | X-ray source. Options: `CuKa`, `CuKa1`, `MoKa`, `CoKa`, `FeKa`, `CrKa`, `AgKa` |
| `two_theta_min` | No | `10.0` | Start of 2θ range (degrees) |
| `two_theta_max` | No | `80.0` | End of 2θ range (degrees) |
| `database` | No | `"cod"` | `"cod"` (Crystallography Open Database), `"mp"` (Materials Project), `"both"` |
| `peak_height_pct` | No | `5.0` | Peak detection height threshold (% of max intensity) |
| `peak_prominence_pct` | No | `3.0` | Peak prominence threshold (% of max) |

**Response includes:** `matches[]` (phase, polymorph, space group, confidence, Rwp), `purity_estimate`, `unassigned_peaks`, `possible_impurities`, `summary`, and `artifacts` (pattern_plot, comparison_plot, peak_table_csv, report_json).

### 2. `xrd.analyze` — Backward-compatible alias
Identical to `xrd.identify`. Same parameters, same output. Exists for backward compatibility.

```
run(tool="xrd.analyze", args={...})  # same as xrd.identify
```

### 3. `xrd.parse` — Quick QC (no database search)
Parse file and detect peaks only. Fast. Use for data quality checks before running full identification.

```
run(tool="xrd.parse", args={
    "file_id": "<file_id>",
    "wavelength": "CuKa"
})
```

Parameters: same file/range/peak params as `xrd.identify`, but NO `composition_hint`, `reactants`, or `database`.

**Response includes:** `peak_count`, `peaks[]` (with 2θ, intensity, d-spacing for each), `file` info (format, points, max intensity), pattern_plot, peak_table_csv.

### 4. `xrd.search_ref` — Reference pattern lookup (no file needed)
Search COD and/or Materials Project for a material's theoretical XRD pattern. No experimental file required.

```
run(tool="xrd.search_ref", args={
    "formula": "TiO2",
    "database": "both"
})
```

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `formula` | **Yes** | — | Chemical formula. Comma-separate for multiple: `"CsPbBr3,CsBr,PbBr2"` |
| `wavelength` | No | `"CuKa"` | X-ray source |
| `two_theta_min` | No | `10.0` | Start of 2θ range |
| `two_theta_max` | No | `80.0` | End of 2θ range |
| `database` | No | `"both"` | `"cod"`, `"mp"`, or `"both"` |

**Response includes:** For each polymorph found: formula, space group, polymorph name, COD/MP ID, peak count, unit cell params. Plus a stick-pattern plot and CSV of all reference peaks with hkl indices and d-spacings.

**Use cases:**
- User wants to know expected peaks before measuring
- Comparing polymorphs of the same compound (e.g. anatase vs rutile TiO₂)
- Looking up what a specific COD/MP entry's pattern should look like

### 5. `xrd.export_origin` — Export for OriginLab
Export processed XRD data as an Origin-ready CSV with metadata header.

```
run(tool="xrd.export_origin", args={
    "file_id": "<file_id>",
    "include_background": true,
    "include_peak_markers": true
})
```

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `file_id` / `file_path` | Yes | — | XRD data file |
| `wavelength` | No | `"CuKa"` | X-ray source |
| `include_background` | No | `true` | Add Background_CPS and Subtracted_CPS columns |
| `include_peak_markers` | No | `true` | Add Peak_Flag column (1=peak, 0=not) |

**Output CSV columns:** `2theta_deg`, `Intensity_CPS`, `d_spacing_A`, `Background_CPS`, `Subtracted_CPS`, `Peak_Flag`

**Origin import instructions** (two options):

**A. Automated (via OriginMCP):**
After `xrd.export_origin` returns the CSV URL, chain into Origin:
```
run(tool="spec.import_csv", args={
    "csv_url": "<url from xrd.export_origin result>",
    "book_name": "XRD_Data",
    "create_graph": true
})
```
This creates a new OPJ project with proper column labels (2θ, Intensity, d-spacing, Background, Subtracted, Peak Flag), an auto-generated plot, and returns a downloadable OPJ file.

**B. Manual (tell user):** File → Import → CSV, skip 9 header lines (lines starting with `#`), comma delimiter.

## Interpreting Results

### Confidence Score (0–100%)
Intensity-weighted scoring. Strong reference peaks matter more than weak ones.
- **>70%**: Strong identification — major peaks match well
- **50–70%**: Likely match — most strong peaks present, some missing
- **25–50%**: Possible match — worth investigating but not definitive
- **<25%**: Weak — consider different phase or check composition_hint

### Rwp (weighted R-factor)
Lower is better. Measures how well the theoretical profile fits the experimental pattern (after background subtraction).
- **< 5%**: Excellent match
- **5–10%**: Good match
- **10–20%**: Fair (may indicate multiple phases, preferred orientation, or peak broadening)
- **20–40%**: Moderate — real match but poor fit quality
- **> 40%**: Poor — likely wrong phase or severe issues

**Important**: Rwp is NOT a "percentage agreement." An Rwp of 7% means the fit residual is 7%, which is good. Do not say "7% match."

### Purity Estimate
Based on the fraction of detected peaks that are assigned to identified phases. 100% means every peak is accounted for. Unassigned peaks lower the purity estimate.

### Polymorph Identification
The tool recognizes common polymorphs by space group:
- TiO₂: anatase (I4₁/amd), rutile (P4₂/mnm), brookite (Pbca)
- ZrO₂: monoclinic (P2₁/c), tetragonal (P4₂/nmc), cubic (Fm-3m)
- CsPbBr₃: orthorhombic Pnma, cubic Pm-3m, tetragonal P4/mbm
- CsPbI₃: δ-phase yellow (Pnma), α-phase black cubic (Pm-3m)
- MAPbI₃: tetragonal (I4/mcm), cubic (Pm-3m), orthorhombic (Pnma)
- And more (SiO₂, Al₂O₃, Fe₂O₃, CaCO₃, ZnO, BaTiO₃, SrTiO₃, FAPbI₃)

## Workflow Decision Tree

```
User uploads XRD file
â”‚
â”œâ”€ Knows what material → xrd.identify (with composition_hint)
â”‚   â””â”€ Mentions reactants → add reactants= parameter
â”‚
â”œâ”€ Just wants to see peaks → xrd.parse
â”‚
â”œâ”€ Wants reference patterns (no file) → xrd.search_ref
â”‚
â”œâ”€ Wants data in Origin → xrd.export_origin → spec.import_csv (chain)
â”‚   â””â”€ Returns downloadable OPJ with graph
â”‚
â””â”€ Unknown material, no hint → xrd.parse first, then ask user
    what they synthesized before running xrd.identify
```

## Important Notes

1. **Always ask for or infer `composition_hint`** — without it, `xrd.identify` only detects peaks but cannot identify phases. Ask: "What material did you synthesize?" or "What compound are you expecting?"

2. **Artifacts are image URLs** — embed the `pattern_plot` and `comparison_plot` in your response so the user sees them inline.

3. **Multi-phase samples**: Handled automatically. Up to 3 phases identified sequentially. All appear in `matches[]`. The `purity_estimate` accounts for all identified phases.

4. **Impurity detection** requires the `reactants` parameter. If the user mentions precursors or what they mixed, pass those formulas.

5. **COD can be slow** (5–30s per search). If it times out, the tool falls back to Materials Project automatically. Or suggest `database="mp"` for faster results.

6. **If `ok: false`**: Report the `reason` to the user. Common issues:
   - File format not recognized → ask user to export to .xy
   - No composition hint → ask what material they expect
   - COD/MP timeout → try `database="mp"` or try again

7. **Background subtraction**: The Rwp calculation now subtracts estimated background before comparing, giving much more realistic Rwp values (typically 15–30% instead of the old ~97%).

## Example Calls

**Standard phase identification:**
```
run(tool="xrd.identify", args={
    "file_id": "abc123",
    "composition_hint": "CsPbBr3",
    "reactants": "CsBr,PbBr2",
    "database": "cod"
})
```

**Quick data quality check:**
```
run(tool="xrd.parse", args={"file_id": "abc123"})
```

**Look up TiO₂ reference patterns (all polymorphs):**
```
run(tool="xrd.search_ref", args={"formula": "TiO2", "database": "both"})
```

**Export to Origin with background and peak markers:**
```
run(tool="xrd.export_origin", args={"file_id": "abc123"})
```

**Check if sample has unreacted zinc:**
```
run(tool="xrd.identify", args={
    "file_id": "abc123",
    "composition_hint": "ZnO",
    "reactants": "Zn"
})
```

**Full XRD → Origin pipeline (export data, create OPJ with graph):**
```
# Step 1: Export processed data
result = run(tool="xrd.export_origin", args={"file_id": "abc123"})
# result contains "origin_csv": "https://files.forthought.cc/files/..."

# Step 2: Import into Origin project
run(tool="spec.import_csv", args={
    "csv_url": result["origin_csv"],
    "book_name": "XRD_CsPbBr3",
    "create_graph": true
})
# Returns: opj_url (downloadable OPJ), plot_url (preview PNG)
```