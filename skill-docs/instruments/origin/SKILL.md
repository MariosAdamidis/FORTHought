# Origin(*.opj) — OriginLab Data Analysis via OriginMCP

**Prerequisite:** OriginLab must be running on the Windows host. If tools return connection errors, tell the user to start OriginLab first.

## Which Tool? — Decision Tree (READ FIRST)

**Before calling any tool, decide here:**

```
User uploads OPJ / asks to analyze spectra
│
├─ You don't know what's in the file
│  └─► origin.inspect (detail="full")  — ALWAYS start here
│
├─ inspect shows diagnostics.has_user_params = true
│  ├─ Has polarization + grouping → origin.trend_plot (ONE call)
│  ├─ Has polarization only → origin.batch_multi_fit
│  └─ No polarization → origin.batch_fit
│
├─ inspect shows diagnostics.has_user_params = false (MESSY FILE)
│  ├─ diagnostics.inferred_params_available = true
│  │  └─ Inferred grouping/pol from column names → proceed as above
│  ├─ diagnostics.inferred_params_available = false
│  │  ├─ Ask user: "What varies across columns? (strain, power, temp?)"
│  │  ├─ OR use find_peaks + get_data on a representative column to survey
│  │  └─ Use batch_fit (not trend_plot) if no grouping can be determined
│  └─ Single column or unknown → origin.fit_peak or origin.find_peaks
│
├─ Data has NO polarization pairing (power series, single column set)
│  ├─ Multiple spectra to track one peak → origin.batch_fit
│  ├─ Need to merge workbooks first → origin.merge → origin.batch_fit
│  └─ Single spectrum → origin.fit_peak or origin.find_peaks
│
├─ User wants visualization only (no fitting)
│  └─► origin.create_graph or origin.normalize
│
└─ User wants to export
   └─► origin.export_csv or origin.save_project
```

**CRITICAL: If the data has polarization pairs (σ+/σ-) and a grouping parameter (strain, temperature, power), ALWAYS use `trend_plot` or `batch_multi_fit`. NEVER use `batch_fit` — it ignores polarization pairing and produces scrambled results.**

## Reading Diagnostics (v10.9)

`origin.inspect` now returns a `diagnostics` block before the workbook data. Use it to make smart decisions:

```json
"diagnostics": {
  "x_unit_guess": "eV",           // "eV", "nm", "cm-1", or "unknown"
  "x_range": [1.85, 2.15],        // actual X data range
  "suggested_x_range": [1.88, 2.10], // auto-detected from peak finding
  "has_user_params": true,         // any column has D-row metadata?
  "has_pol_pairing": true,         // polarization key found?
  "has_grouping": true,            // strain/temp/power key found?
  "inferred_params_available": true, // regex matched column names?
  "inferred_params_sample": [{"strain": "0.45", "pol": "σ+"}],
  "empty_columns": ["Book1/Data/col5"]
}
```

**How to use diagnostics:**
- `x_unit_guess = "nm"` → Data is in wavelength. Do NOT pass eV-scale x_min/x_max.
- `suggested_x_range` → Use as x_min/x_max if the user didn't specify.
- `has_user_params = false` → Messy file. Check `inferred_params_available`.
- `empty_columns` → Skip these in y_cols.

## Namespace

All Origin tools use the `origin.*` namespace via `run("origin.<tool>", {...})`.
Legacy `spec.*` and bare `opj_*` names are aliased automatically.

## Parameter Names — EXACT SPELLING

These are the correct parameter names. Using wrong names wastes a tool round:

| Parameter | Correct | WRONG (do not use) |
|-----------|---------|---------------------|
| Sheet name | `sheet` | ~~sheet_name~~ |
| Workbook name | `workbook` | ~~workbook_name~~ |
| Y columns | `y_cols` | ~~y_columns~~, ~~cols~~ |
| Fit model | `model` | ~~fit_model~~, ~~profile~~ (only for single-peak fit_peak) |

## Efficiency Rules

1. **Do NOT call `find("origin")` after reading this skill.** All 14 tools are documented below.

2. **ONE inspect, detail="full".** Call `origin.inspect(file_id, detail="full")` exactly once per file.

3. **y_cols supports ranges.** Use `"1-12"` not `"1,2,3,4,5,6,7,8,9,10,11,12"`.

4. **Model names:** Valid strings for multi-peak fitting:
   - `"lorentz+lorentz"` (aliases: `"ll"`, `"2lorentz"`, `"lorentz"`, `"voigt"` → auto-mapped)
   - `"lorentz+voigt"` (aliases: `"lv"`)
   - `"voigt+voigt"` (aliases: `"vv"`, `"2voigt"`)
   **Default to `"lorentz+lorentz"` unless the user specifically asks for Voigt.**

5. **NEVER repeat a tool call with the same parameters.** The server caches results (5-min TTL).

6. **CRITICAL — Image URLs:** `trend_plot` and `batch_multi_fit` return `embed_markdown` as the **first field** in the response. It contains ready-to-use `![text](url)` lines with correct OWUI file URLs. **Copy `embed_markdown` verbatim into your response. Do NOT construct, modify, or guess image URLs.** The URLs contain unique file IDs that you cannot predict.

7. **trend_plot replaces manual analysis.** Do NOT call `batch_multi_fit` + manually compute Pc in text. Call `trend_plot` once — it does the batch fit AND generates xc, Pc, width, separation trend plots AND the waterfall automatically.

8. **Maximum 4 tool rounds** for a standard analysis:
   - Round 1: view_skill
   - Round 2: origin.inspect
   - Round 3: origin.trend_plot (or batch_multi_fit)
   - Round 4: text response with embedded plots
   **STOP after getting trend_plot results.** Do NOT call find_peaks, batch_multi_fit, or batch_fit after trend_plot — trend_plot already does the batch fitting internally. Calling more tools after trend_plot wastes rounds and produces worse results.

9. **x_min/x_max are now optional.** If omitted, the server auto-detects the fit region from peak finding. Only specify x_min/x_max when you need to override the auto-detection (e.g., user asks to focus on a specific feature).

10. **min_separation guard.** Two-peak fits where peaks converge (Δ < 10 meV) are flagged with `"degenerate": true` and a warning. The two-peak result is preserved but the decomposition is marginal — report it to the user as effectively a single peak.

## General Workflow

```
1. origin.inspect   → Read diagnostics + column metadata + user_params
2. Decide tool      → Use the decision tree above, guided by diagnostics
3. Analyze          → trend_plot / batch_multi_fit / batch_fit / fit_peak
4. Present results  → Embed ALL plot URLs as images, summarize parameters
```

## Tool Reference (14 tools)

### origin.inspect — File Structure + Column Metadata + Diagnostics
```
run("origin.inspect", {"file_id": "...", "detail": "full"})
```
Returns per-column: name, long_name, comments, units, user_params dict.
**NEW (v10.9):** Returns `diagnostics` block with x_unit_guess, suggested_x_range, has_user_params, has_pol_pairing, has_grouping, inferred_params availability.
**User parameter rows are critical.** They encode experiment conditions (polarization, strain, power, temperature). Use them to decide which tool to call.

### origin.get_data — Extract X,Y Arrays
```
run("origin.get_data", {"file_id": "...", "workbook": "Book1", "y_col": 3, "max_points": 200})
```

### origin.merge — Power Series Merge + Background Subtraction
```
run("origin.merge", {"file_id": "...", "folder": "530/NR2", "bg_folder": "530/Si"})
```
**When to use:** Multiple workbooks from a power-dependent measurement need combining.

### origin.fit_peak — Single Peak Fit
```
run("origin.fit_peak", {"file_id": "...", "workbook": "Book1", "y_col": 3, "profile": "lorentz", "x_min": 500, "x_max": 550})
```
Profiles: `lorentz`, `gaussian`, `voigt`. Returns xc, w, A, y0, R², plot.

### origin.multi_fit — Two-Peak Decomposition (single column)
```
run("origin.multi_fit", {"file_id": "...", "workbook": "Book1", "y_col": 3, "model": "lorentz+lorentz", "x_min": 1.90, "x_max": 2.05, "max_fwhm": 0.050})
```
Returns peak parameters with uncertainties (xc_err, w_err), separation, delta_r2.

**delta_r2:** >0.005 = two peaks justified, 0.001–0.005 = marginal, <0.001 = single peak sufficient.

**min_separation guard (v10.9):** If Δ < 10 meV, result includes `"degenerate": true` — decomposition is marginal.

### origin.batch_multi_fit — Batch Two-Peak Decomposition (PREFERRED for 3+ columns)
```
run("origin.batch_multi_fit", {"file_id": "...", "workbook": "Book7", "sheet": "Data", "y_cols": "1-12", "model": "lorentz+lorentz", "max_fwhm": 0.050})
```
**Always prefer this over calling multi_fit in a loop.**

**x_min/x_max optional (v10.9):** If omitted, auto-detects fit region from peak finding. Result includes `auto_x_range` showing what was used.

**Fallback grouping (v10.9):** If user_params are empty, the server tries to infer grouping from column names (e.g., "0.45%", "σ+", "V1"). Result includes `inferred_params: true` when this was used.

Returns: per-column fit results, summary grid plot, and **publication-quality waterfall plot** (dark-theme, σ+ red solid / σ- blue dashed, auto-generated when paired polarization + grouping parameter detected in user_params).

### origin.trend_plot — Trend Analysis (THE GO-TO for parametric studies)
```
run("origin.trend_plot", {"file_id": "...", "workbook": "Book7", "y_cols": "1-12", "model": "lorentz+lorentz", "max_fwhm": 0.050, "plots": "xc,pc,width,separation"})
```
Runs batch_multi_fit internally, then generates trend plots:
- `xc` — peak position vs parameter (strain, power, temperature)
- `pc` — circular polarization Pc vs parameter
- `width` — FWHM vs parameter
- `separation` — peak splitting vs parameter

Returns all batch_multi_fit results (including waterfall) PLUS trend plot URLs in `result["trends"]`.

**This is the single most useful tool for the lab.** One call = waterfall + all trend plots + all fit parameters.

### origin.batch_fit — Batch Single-Peak Fit
```
run("origin.batch_fit", {"file_id": "...", "workbook": "Merged", "y_cols": "subtracted", "profile": "voigt"})
```
**When to use:** Tracking ONE peak across many spectra. Do NOT use for polarization-paired data.

### origin.normalize — Normalize Spectra
```
run("origin.normalize", {"file_id": "...", "workbook": "Book1", "y_cols": "1-6", "mode": "peak"})
```
Modes: `peak` (max=1), `area` (integral=1). Returns overlay plot.

### origin.smooth — Savitzky-Golay Smoothing
```
run("origin.smooth", {"file_id": "...", "workbook": "Book1", "y_col": 3, "window": 11, "polyorder": 3})
```

### origin.find_peaks — Auto Peak Detection
```
run("origin.find_peaks", {"file_id": "...", "workbook": "Book1", "y_col": 3, "prominence": 0.1})
```
Returns peak positions, heights, FWHM, and annotated plot. Use before fitting to survey features.

### origin.create_graph — Line Plot Overlay
```
run("origin.create_graph", {"file_id": "...", "workbook": "Book1", "y_cols": "1-6", "title": "PL Spectra"})
```

### origin.save_project / origin.export_csv — Output
```
run("origin.save_project", {"file_id": "...", "output_name": "my_analysis"})
run("origin.export_csv", {"file_id": "...", "workbook": "Book1", "sheet": "Data"})
```

## Analysis Strategies

### Strain/temperature/power-dependent PL with σ+/σ- (most common lab task)
```
origin.inspect → read diagnostics + user_params → confirm polarization + grouping keys exist
→ origin.trend_plot (y_cols="1-12", model="lorentz+lorentz", plots="xc,pc,width")
→ present: waterfall_plot, trends.xc_trend, trends.pc_trend, trends.width_trend
```
**One trend_plot call gives everything.** The waterfall uses dark-theme publication styling (σ+ red, σ- blue dashed, strain labels in white).

### Messy file — no user_params
```
origin.inspect → diagnostics.has_user_params = false
  → if inferred_params_available: proceed normally (server injects synthetic params)
  → if NOT inferred: ask user what varies, or use batch_fit for generic analysis
  → if x_unit_guess = "nm": do NOT pass eV-scale x_min/x_max; use suggested_x_range or omit
```

### Quick file survey
```
origin.inspect → origin.find_peaks (representative column) → decide tool
```

### Power-dependent series (no polarization)
```
origin.inspect → origin.merge (if multiple workbooks) → origin.batch_fit
```

### Preprocessing
```
origin.smooth (noisy data) → origin.normalize (comparing) → origin.find_peaks → fitting
```

## Image Embedding — CRITICAL

**`trend_plot` and `batch_multi_fit` return `embed_markdown` as the FIRST field in the JSON response.** It contains the exact `![text](url)` markdown you must use. 

**DO THIS:**
```
# In the tool response JSON, the first key is:
"embed_markdown": "![Waterfall Plot](https://webui.forthought.cc/api/v1/files/abc-123/content)\n![Peak Position Trend](https://webui.forthought.cc/api/v1/files/def-456/content)"

# Copy these lines directly into your response text.
```

**DO NOT:**
- Construct URLs from the file_id + filename (this produces broken `?path=` URLs)
- Use `https://mcp.forthought.ai/...` or any URL not from the tool response
- Modify, shorten, or reformat the URLs from `embed_markdown`
- Use `[text](url)` instead of `![text](url)` — the `!` prefix renders images

## Anti-Patterns — DO NOT DO THESE

- ❌ Calling `find("origin")` after reading this skill
- ❌ Calling `inspect` more than once on the same file
- ❌ Using `batch_fit` on σ+/σ- polarization data (use `trend_plot` or `batch_multi_fit`)
- ❌ Using `sheet_name` or `workbook_name` as parameter names (correct: `sheet`, `workbook`)
- ❌ Using `model="Voigt"` (correct: `"lorentz+lorentz"`)
- ❌ Calling `trend_plot` then calling `batch_multi_fit` with same params (trend_plot already does it)
- ❌ Calling the same tool with identical params multiple times
- ❌ Fitting columns one by one when `batch_multi_fit(y_cols="1-12")` does them all at once
- ❌ Using `[text](url)` for images instead of `![text](url)`
- ❌ Computing Pc manually in text when `trend_plot(plots="pc")` generates the plot
- ❌ Calling `find_peaks` or `batch_fit` after `trend_plot` already returned results — trend_plot does everything
- ❌ Fabricating or guessing plot URLs — always use URLs from `embed_markdown` or tool result fields
- ❌ Passing eV-scale x_min/x_max when diagnostics shows `x_unit_guess: "nm"`
- ❌ Ignoring `"degenerate": true` in fit results — report it to the user
