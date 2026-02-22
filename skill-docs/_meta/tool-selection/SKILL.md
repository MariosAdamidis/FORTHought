---
name: Tool Selection Guide
description: Use this skill when deciding which FORTHought tool or skill to use for a request. Provides a lightweight decision matrix for web search vs academic search vs computation vs document generation. Load this first to understand which specialized skill to load next.
---
# Tool Selection Guide
This is a meta-skill that helps you decide which specialized skill to load for any given task.
## Decision Matrix
- **User asks about current events, news, product releases:** → Load `web-search` skill
- **User asks for papers, authors, literature review:** → Load `academic-search` skill
- **User asks for chart, graph, plot (line/bar/pie/scatter/etc):** → Load `chartjs` skill
- **User asks for data analysis, statistics, file processing:** → Load `code-interpreter` skill
- **User wants PDF/DOCX/XLSX/CSV file:** → Load `file-export` skill
- **User wants presentation/slideshow:** → Load `presenton` skill
## Workflow
1. User makes request.
2. Determine category from matrix above.
3. Load appropriate specialized skill via `get_skill`.
4. Follow that skill's detailed instructions.
