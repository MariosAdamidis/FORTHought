---
name: Presenton
description: Use this skill when generating a full PowerPoint presentation automatically. Load this when the user asks for a presentation or slideshow.
---
# Presenton
## When to Use
Use the `generate_presentation` tool for creating a complete slide deck from a prompt. This is best for speed and AI-driven structure.
## Parameters
- `prompt`: Detailed description of the presentation topic and content.
- `n_slides`: Number of slides (e.g., 10).
- `theme`: "academic", "business", "creative", or "general".
## Workflow
1. User asks for a presentation.
2. Formulate a detailed prompt for the tool.
3. Call `generate_presentation`.
4. Return the download URL provided by the tool.
