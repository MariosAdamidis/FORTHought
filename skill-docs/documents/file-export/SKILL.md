---
name: File Export
description: Use this skill when creating downloadable files (PDF, DOCX, PPTX, XLSX, CSV, etc.). Provides patterns for the file_export MCP server.
---
# File Export
## Critical Rules
1. **Single File**: Always use `create_file`.
2. **Multiple Files**: Use `create_file` multiple times.
3. **Archive**: Only use `generate_and_archive` if the user says "zip", "archive", or "compress".
4. **Links**: Return the exact `url` from the tool's response.
## Payloads
- **PDF/DOCX**: Use `content: [{"type": "paragraph", "text": "..."}]`
- **XLSX/CSV**: Use `content: [["Header1"], ["Value1"]]`
- **PPTX**: Use `slides_data: [{"title": "...", "content": ["..."]}]`
