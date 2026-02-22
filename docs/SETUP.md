# FORTHought Setup Guide

This guide walks you through configuring Open WebUI after deploying the Docker stack. It covers importing functions, creating model profiles, configuring RAG, and wiring everything together.

## Prerequisites

- Docker stack running (`docker compose -f config/docker-compose.yml up -d`)
- `.env` file configured from `config/.env.example`
- Open WebUI accessible at `http://localhost:8081`
- Admin account created

---

## 1. Import Functions

Go to **Admin → Functions** and import each file. The function type (Pipe, Filter, Tool, Action) is detected automatically from the file header.

### Pipes (LLM Providers)

| File | ID (auto-assigned) | Notes |
|---|---|---|
| `pipes/gemini_pipe.py` | `google_gemini_new` | Requires Google AI API key in valve settings |
| `pipes/openrouter_pipe_v2.py` | `open_webui_openrouter_integration` | Requires OpenRouter API key in valve settings |

### Filters

| File | ID | Assign to |
|---|---|---|
| `filters/markdown_normalizer.py` | `markdown_normalizer` | OpenRouter-based models (cleans leaked XML/reasoning) |
| `filters/image_description_context.py` | `image_description_context` | Non-vision models that receive image uploads |
| `filters/image_reembed_injector.py` | `image_reembed_injector` | Models that lose image context across turns |
| `filters/uploaded_filename.py` | `uploaded_filename` | All models that use file-based tools |

### Tools

| File | ID | Notes |
|---|---|---|
| `skills/lab_skills.py` | `lab_skills` | MCP gateway for Lab profile |
| `skills/coder_skills.py` | `coder_skills` | MCP gateway for Coder profile |
| `skills/instrument_skills.py` | `instrument_skills` | MCP gateway for Instrument profile |
| `tools/web_search.py` | `web_search` | Requires LangSearch API key + reranker URL in valves |
| `tools/chemistry_database.py` | `chemical_compound_intelligence_system` | PubChem/CAS lookups, no API key needed |

### Actions

| File | ID | Notes |
|---|---|---|
| `actions/export_to_word.py` | `export_to_word` | Adds "Export to Word" button to chat messages |
| `actions/lemonade_control_panel.py` | `lemonade_control_panel` | Dashboard for Lemonade Server monitoring |

---

## 2. Import Skill Documents as Knowledge

The skill documents in `skill-docs/` are loaded on-demand by the Skills tools via `view_skill(domain)`. In OWUI, these are stored as **Knowledge** entries.

Go to **Workspace → Knowledge** and create one entry per skill document:

| Knowledge Name | File to paste | Used by profile |
|---|---|---|
| `SEM Microscopy` | `skill-docs/instruments/sem/SKILL.md` | Instrument |
| `XRD Analysis` | `skill-docs/instruments/xrd/SKILL.md` | Instrument |
| `OriginLab` | `skill-docs/instruments/origin/SKILL.md` | Instrument |
| `PL Experimental Planning` | `skill-docs/research/literature/SKILL.md` | Instrument |
| `Literature Search` | `skill-docs/research/academic-search/SKILL.md` | Lab |
| `Web Research` | `skill-docs/research/web-search/SKILL.md` | Lab, Coder |
| `DOCX Reports` | `skill-docs/documents/file-export/SKILL.md` | Lab, Coder |
| `Document Editing` | `skill-docs/documents/file-export/SKILL.md` | Lab, Coder |
| `Presentations` | `skill-docs/documents/presenton/SKILL.md` | Lab, Coder |
| `HTML Artifacts` | `skill-docs/visualization/chartjs/SKILL.md` | Lab, Coder |
| `Data Analysis` | `skill-docs/compute/code-interpreter/SKILL.md` | Lab, Coder |

The names must match exactly what the system prompts reference in their routing tables (e.g., `view_skill("SEM Microscopy")`).

---

## 3. Create Model Profiles

Go to **Admin → Models** and create three model profiles. Each profile needs a base model, system prompt, and the correct tools/filters/skills assigned.

### Instrument Profile

For scientific instrument analysis (SEM, XRD, PL, OriginLab).

| Setting | Value |
|---|---|
| **Base model** | Any vision-capable model (e.g., Gemini Flash) |
| **System prompt** | Paste contents of `profiles/instrument_prompt.md` |
| **Tools** | `instrument_skills`, `web_search` |
| **Filters** | (none required for vision models) |
| **Skills** | `sem-microscopy`, `xrd`, `originlab`, `pl-experimental-planning`, `pl_2d_materials`, `web-research` |
| **Capabilities** | ☑ Vision, ☑ File Upload, ☑ Citations |

### Coder Profile

For data analysis, plotting, scripting, and library docs.

| Setting | Value |
|---|---|
| **Base model** | Any strong coding model (e.g., GPT-5.2, Gemini Pro) |
| **System prompt** | Paste contents of `profiles/coder_prompt.md` |
| **Tools** | `coder_skills`, `web_search` |
| **Filters** | `markdown_normalizer` (for OpenRouter models) |
| **Skills** | `data-analysis`, `document-editing`, `docx-reports`, `html-artifacts`, `presentations`, `web-research` |
| **Capabilities** | ☑ File Upload, ☑ Code Interpreter, ☑ Citations |
| **Builtins** | ☑ Code Interpreter |

### Lab Profile

For literature review, report writing, web research, chemistry lookups.

| Setting | Value |
|---|---|
| **Base model** | Any capable model (e.g., Gemini Pro, Gemini Flash) |
| **System prompt** | Paste contents of `profiles/lab_prompt.md` |
| **Tools** | `lab_skills`, `web_search`, `chemical_compound_intelligence_system` |
| **Filters** | `markdown_normalizer` (for OpenRouter models), `image_description_context` (for non-vision models) |
| **Skills** | `web-research`, `presentations`, `literature-search`, `html-artifacts`, `docx-reports`, `document-editing`, `data-analysis` |
| **Capabilities** | ☑ File Upload, ☑ Code Interpreter, ☑ Citations |
| **Builtins** | ☑ Code Interpreter |

### Filter Assignment Guidelines

| Filter | When to assign |
|---|---|
| `markdown_normalizer` | Models served through OpenRouter (cleans leaked `<thinking>` tags, XML artifacts) |
| `image_description_context` | Non-vision models that may receive image uploads (converts images to text descriptions) |
| `uploaded_filename` | Any model using file-based tools (usually already handled by skills) |

---

## 4. Configure RAG Pipeline

Go to **Admin → Documents** and set:

### Document Processing

| Setting | Value |
|---|---|
| Content Extraction Engine | `docling` |
| Docling Server URL | `http://<docling-host>:5005/` |
| Docling Pipeline | `standard` |
| Docling OCR Engine | `easyocr` |
| Docling Table Mode | `accurate` |
| Docling PDF Backend | `dlparse_v4` |

### Chunking

| Setting | Value |
|---|---|
| Text Splitter | `token` |
| Chunk Size | `800` |
| Chunk Overlap | `100` |
| Min Chunk Size Target | `400` |

### Embedding

| Setting | Value |
|---|---|
| Embedding Engine | `openai` |
| Embedding Model | `embed` (or your model name in LM Studio) |
| OpenAI API Base URL | `http://<lmstudio-host>:5555/v1` |
| Batch Size | `1024` |

### Search

| Setting | Value |
|---|---|
| Hybrid Search | ☑ Enabled |
| BM25 Weight | `0.6` |
| Top K | `10` |
| Relevance Threshold | `0.15` |

### Reranking

| Setting | Value |
|---|---|
| Reranking Engine | `external` |
| External Reranker URL | `http://<host>:8040/api/v1/reranking` |
| Reranking Model | `bge-reranker-v2-m3-GGUF` |
| Top K (Reranker) | `5` |

> The reranker URL should point to your AMD Lemonade Server instance. The same endpoint is shared by the RAG pipeline and the web search tool.

---

## 5. Configure Code Execution

Go to **Admin → Settings → Code Execution**:

| Setting | Value |
|---|---|
| Enable | ☑ |
| Engine | `jupyter` |
| Jupyter URL | `http://unsloth-jupyter:8888` (or your Jupyter host) |
| Auth | `password` |
| Password | Your `JUPYTER_PASSWORD` from `.env` |
| Timeout | `300` |

### Code Interpreter Prompt Template

In **Admin → Settings → Code Execution → Prompt Template**, set the instructions that teach the LLM how to use the code interpreter. Key points to include:

- Use `<code_interpreter type="code" lang="python">` XML tags (not triple backticks)
- Never use `plt.show()` — save to `/data/files/` and print a markdown image link
- Print meaningful outputs explicitly
- The plot URL pattern should match your file server: `![Plot](http://<your-files-url>/files/filename.png)`

---

## 6. MetaMCP Configuration

MetaMCP aggregates all MCP servers behind a single endpoint. Configuration is done through the MetaMCP web UI (default: `http://localhost:12008`).

Register each MCP server:

| Server | URL | Transport |
|---|---|---|
| Papers | `http://mcp-papers:9005/mcp` | Streamable HTTP |
| Files | `http://mcp-files:9004/mcp` | Streamable HTTP |
| SEM/Micro | `http://mcp-micro:9006/mcp` | Streamable HTTP |
| XRD | `http://mcp-xrd:9008/mcp` | Streamable HTTP |
| PL | `http://mcp-pl:9010/mcp` | Streamable HTTP |
| Origin | `http://<windows-host>:12009/mcp` | Streamable HTTP |

After registering, the Skills tools connect to MetaMCP at `http://metamcp:12008/metamcp/{server_name}/mcp`.

---

## 7. Tool Server Connections (Optional)

If you want to connect MCP servers directly to OWUI (bypassing Skills), go to **Admin → Settings → Tool Servers** and add:

| Name | URL | Type |
|---|---|---|
| SEM | `http://mcp-micro:9006/mcp` | MCP |
| Skills | `http://metamcp:12008/metamcp/codemode/mcp` | MCP |

Most users should use the Skills approach instead, as it provides on-demand documentation loading and lower token usage.

---

## Verification

After completing setup, run the smoke test:

```bash
bash scripts/smoke_test_core.sh
```

Then test each profile:
1. **Lab**: "Search for recent papers on LIPSS formation on MoS2"
2. **Coder**: "Plot a sine wave and save it as a file"
3. **Instrument**: Upload an SEM image and ask "Analyze this for periodic nanostructures"
