# 🧠 Project FORTHought

[![Status](https://img.shields.io/badge/Status-Alpha-orange.svg)](https://github.com/MariosAdamidis/FORTHought) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An open-source, interpretable platform for AI-accelerated scientific discovery, designed to function as a true research collaborator.

---

### 🔭 The Vision: A Digital Colleague for Every Lab

Scientific progress is drowning in a sea of its own success. We generate more data and publish more papers than ever before, but our ability to synthesize this information remains human-limited. Project FORTHought was born from a simple but ambitious question: What if every researcher, in any lab, could have an AI collaborator capable of understanding their unique context, reasoning about their complex data, and helping to accelerate the pace of discovery?

This project is my first step towards building that future. It’s not just about creating another tool, but about architecting an ecosystem where AI can be securely and effectively integrated into the day-to-day scientific process, from messy experimental data to publishable insights.

### 🌱 Core Philosophy

The development of FORTHought is guided by three core principles:

1.  **Democratization of Research:** Advanced AI capabilities should not be confined to a few well-funded institutions. By building on open-source tools and enabling deployment on local hardware, FORTHought aims to make powerful research assistants accessible to all.
2.  **Integration over Reinvention:** The project's strength comes from selecting the best open-source components and orchestrating them into a cohesive, powerful system. The innovation lies in the architecture and the solution to a specific problem, not in reinventing every wheel.
3.  **Interpretability by Design:** A "black box" AI is of limited use in science. A core goal is to build a system where the AI's reasoning can be traced, understood, and trusted, allowing it to be a true collaborator rather than just an oracle.

### ⚙️ Current Features & Validated Capabilities

The platform is currently in an alpha stage, but the foundational layer is operational and has been validated on a number of complex workflows:

* **Hybrid Document Intelligence:** A robust RAG-first system that uses a specialized engine (Docling) for primary analysis and intelligently falls back to a rich set of local PDF/OCR libraries for verification and low-level tasks.
* **Self-Correcting Code Interpreter:** An enhanced, GPU-accelerated Jupyter environment with a comprehensive scientific stack that can autonomously recover from code errors and refine its scripts to successfully complete tasks.
* **Scientific Tool Integration:** Seamlessly connects to real-world scientific databases like the **Materials Project** via a custom MCP orchestrator, allowing for complex, multi-tool chained workflows.
* **Full Office Suite Manipulation:** Programmatic creation and editing of Word (`.docx`), PowerPoint (`.pptx`), and complex multi-sheet Excel (`.xlsx`) files via the Code Interpreter.
* **Advanced Visualization:** Capable of generating publication-quality static plots (`matplotlib`, `seaborn`) and interactive 3D visualizations (`plotly`).
* **Secure Global Collaboration:** Built to allow secure, HTTPS-encrypted external access for file sharing and interaction via Cloudflare tunnels.

### 🛠️ System Requirements & Setup

This setup is designed to run on a **Windows Host with WSL2 and an AMD GPU**.

#### 1. Host System Setup (Windows)
* **LM Studio:** Download and install LM Studio.
    * Download the following models: `Qwen 3 0.6b` (for embeddings) and `Qwen 2.5 vl 3b` (for Vision/VLM).
    * Navigate to the "Local Server" tab (`</>`), load both models, and start the server on Port `1234`.

#### 2. Project Setup (WSL2)
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/MariosAdamidis/FORTHought.git](https://github.com/MariosAdamidis/FORTHought.git)
    cd FORTHought
    ```
2.  **Configure Environment Variables:**
    * Create a local `.env` file by copying the provided template:
        ```bash
        cp .env.example .env
        ```
    * Open the `.env` file (`nano .env`) and add your personal API keys.

3.  **Launch the Platform:**
    * Ensure Docker Desktop is running on your Windows host.
    * From the project's root directory in your WSL2 terminal, run:
        ```bash
        docker-compose up -d
        ```
4.  **Access Services:**
    * **Open WebUI:** `http://localhost:8081`
    * **Jupyter Lab:** `http://localhost:8888`

### 🏗️ Architecture
![image](https://github.com/user-attachments/assets/1c754032-5735-4296-b575-3fe926ecf37d)

### 🛣️ Roadmap & Future Goals

This project is under active development. The future goals are divided into technical enhancements and broader research ambitions.

#### Technical Enhancements
* [ ] **Container Security Hardening:** Transition from permissive development settings to a secure production environment by implementing non-root users and strict seccomp profiles.
* [ ] **Performance Monitoring:** Integrate Prometheus and Grafana for real-time dashboards on container health and resource utilization.
* [ ] **Expand Scientific Tooling:** Incrementally add new scientific databases and tools, such as PubChem for chemical data.

#### Research & Publication Goals
* [ ] **Publish a Systems Paper:** Formally document the FORTHought architecture, its features, and its utility in a peer-reviewed journal.
* [ ] **Investigate Mechanistic Interpretability:** Use the platform to fine-tune a "digital colleague" model on real lab data and build tools to analyze and visualize its internal reasoning processes.
* [ ] **Explore the Philosophy of AI in Science:** Publish a perspective piece on how transparent, collaborative AI systems can change the nature of scientific discovery, creativity, and knowledge generation.

### 🤝 Contributing

This is an independent project born out of a passion for science and AI. Contributions, feature requests, and discussions are all welcome. Please feel free to open an issue to report a bug or suggest a feature.

### 🙏 Acknowledgements & Community

Project FORTHought stands on the shoulders of giants. It is an integration of many incredible open-source projects, and its success would not be possible without their creators and maintainers.

#### Core Dependencies & Inspirations

* **[Open WebUI](https://github.com/open-webui/open-webui):** Provides the excellent, user-friendly chat interface that serves as the primary hub for the platform.
* **[UnslothAI](https://github.com/unslothai/unsloth):** The core library for making LLM inference and fine-tuning dramatically more efficient on local hardware. The Jupyter environment is built with Unsloth in mind.
* **[Docling](https://github.com/docling-project/docling):** The powerful document and image extraction engine that forms the core of the RAG pipeline.
* **[SearXNG](https://github.com/searxng/searxng):** The private, hackable metasearch engine that provides the `RAG_WEB_SEARCH` capability.
* **[Jupyter](https://github.com/jupyter):** The foundational project behind the interactive code interpreter environment.
* **[LM Studio](https://github.com/lmstudio-ai):** The provider of the intuitive desktop application used for serving local models to the platform.
* **[Model Context Protocol (MCP)](https://github.com/open-webui/mcpo):** The emerging standard used for orchestrating calls to external tools and services.
* **[Materials Project](https://github.com/materialsproject):** The provider of the essential scientific database and API that demonstrates the platform's utility for real-world research.
* **[Upstash Context7](https://github.com/upstash/context7):** An example of a useful MCP tool integrated into the platform's toolset.

#### Community & Special Thanks

This project was built not just on code, but on community knowledge. I want to extend a special thank you to the contributors of the following Reddit thread:

* **A huge thank you to the r/LocalLLaMA community**, specifically the contributor Ok_Ocelot2268 in the **["ROCm 6.4 current unsloth working"](https://www.reddit.com/r/LocalLLaMA/comments/1kp6gdv/rocm_64_current_unsloth_working/)** thread. Their shared solutions and patches were absolutely critical for getting the Unsloth library to function correctly with ROCm on an AMD GPU. This project would have been stalled without their help.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
