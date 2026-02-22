---
name: SEM Microscopy
description: >
  Analyze SEM images for periodicity and particle sizing using micro.sem_fft.
  Use when user uploads an SEM image or asks about nanostructures, LIPSS,
  ripples, gratings, FFT analysis, periodicity, spatial frequency, particle
  size, grain size, or diameter distribution. Covers magnification reading,
  file identification, parameter selection, and result presentation.
---

# SEM Analysis — micro.sem_fft

## Workflow

```
Step 1: Read magnification from SEM info bar  → "x40000"
Step 2: Identify file_id from upload metadata  → UUID
Step 3: Decide analysis type                   → periodicity / particles / both
Step 4: Call micro.sem_fft with exact params   → results + figures
Step 5: Present results with embedded figures  → natural language + images
```

**If the user has NOT attached an image, ask them to upload one.**

## Step 1 — Read Magnification

You are a vision model. Look at the SEM info bar (usually bottom of image) and read the magnification label: "x40000", "x10000", "x30000", etc. You MUST identify this before proceeding.

## Step 2 — Identify file_id

The upload message metadata contains the file UUID (e.g. `211631c9-fa35-42a7-a4fc-3b68d84f99bf`). Use this directly. If you cannot find it, call `get_image_path()` which returns JSON with `file_id`, `path`, and `name`.

## Step 3 — Choose Analysis Type

| User asks about | Set `particle_analysis` |
|-----------------|------------------------|
| Periodicity, spacing, LIPSS, ripples, FFT, gratings | omit (default false) |
| Particle size, grain size, diameter, distribution | `true` |
| General "analyze this" | `true` (runs both) |

## Step 4 — Call the Tool

### Periodicity only
```
run("micro.sem_fft", {
  "file_id": "<UUID>",
  "mag_label": "x40000"
})
```

### With particle sizing
```
run("micro.sem_fft", {
  "file_id": "<UUID>",
  "mag_label": "x40000",
  "particle_analysis": true
})
```

### Exact Parameter Names — NO ALIASES

| Correct | WRONG (do not use) |
|---------|-------------------|
| `file_id` | `image_id`, `imageId`, `id` |
| `mag_label` | `magnification`, `mag`, `zoom` |
| `particle_analysis` | `analyze_particles`, `particle_mode` |

Do NOT add extra parameters (crop_bottom_px, roi, preset, preprocess) unless the user explicitly requests them. Defaults are good.

## Step 5 — Present Results

1. **Embed the composite figure:** `![SEM Analysis](artifacts.summary_png)`
2. **Summarize findings** using ONLY numbers from the tool response:
   - Periodicity: period (nm), angle (°), confidence, SNR, band (macro/micro)
   - Particles: count, mean diameter (nm), std dev, p10–p90 range
3. **Embed detail plots** for bands the user cares about:
   - `![Macro FFT](artifacts.macro_fft2d_png)`
   - `![Micro PSD](artifacts.micro_psd_png)`
4. **Link downloadable files:** line profile CSV, full JSON report

### Artifact Keys

| Key | Content |
|-----|---------|
| `summary_png` | 6-panel composite (always present) |
| `preview_png` | Input image preview |
| `macro_fft2d_png` | Macro band FFT |
| `macro_psd_png` | Macro power spectral density |
| `micro_fft2d_png` | Micro band FFT |
| `micro_psd_png` | Micro power spectral density |
| `line_profile_csv` | 1D intensity data |
| `report_json` | Full machine-readable report |

## Rules

1. **Never call with empty or missing arguments.** Both `file_id` and `mag_label` are required.
2. **Only use URLs from the `artifacts` object.** Never fabricate image paths.
3. **Never add ± uncertainties** unless the tool provides them.
4. **Report physical quantities with units:** nm, °, etc.
5. **Only add `scalebar_px` and `scalebar_nm`** if the user explicitly asks you to measure the scale bar.

## Combining with Documents

After analysis, to create a report:
```
1. run("micro.sem_fft", {...})          → analysis results
2. Summarize findings
3. create_document(format="docx", ...)  → lab report with figures and data tables
```

Call `view_skill("DOCX Reports")` for the document format.
