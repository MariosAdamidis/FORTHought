---
name: ChartJS Visualization
description: Use this skill when creating interactive inline charts. Supports line, bar, pie, doughnut, radar, polarArea, scatter, and bubble charts. Load this whenever the user requests a visualization of these types.
---
# ChartJS Visualization
For any supported chart type (`line`, `bar`, `pie`, `scatter`, etc.), you MUST use the `chartjs` tool, not the code interpreter.
## Call Structure
Call with structured JSON. For `line`/`bar`, ensure `data.length` equals `labels.length`. For `scatter`/`bubble`, use dummy `labels: ["x"]` and data objects like `{"x": 1, "y": 2}`.
## Workflow
1. User requests a chart.
2. Identify the chart type.
3. Prepare the data arrays (`labels` and `datasets`).
4. Validate that data and label lengths match.
5. Call the `chartjs` tool with the structured JSON payload.
