---
name: Code Interpreter
description: Use for data analysis, file processing, statistics, and complex computations in Python.
---
# Code Interpreter
## When to Use
Use for analyzing uploaded files (CSV, etc.), performing statistical calculations, and generating complex (non-ChartJS) visualizations.
## Output Patterns
- **For ChartJS**: Do NOT save an image. Instead, `print('CHARTJS: {"title": "...", "labels": [...], "datasets": [...] }')`.
- **For Files/Images**: Save files to `/data/files/` and print the path. The assistant will handle uploading.
## File System
- User uploads are in `/data/uploads/`.
- Your outputs must go to `/data/files/`.
