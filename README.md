# LangChain Agent Basics

A practical learning repository for building AI-powered Python applications, from basic LLM calls and tool-using agents to structured workflows, retrieval-augmented generation (RAG), observability and AI-assisted planning.

The repository accompanies a **step-by-step technical manual** that explains the concepts, the role of the LLM versus Python, and the execution flow of each example.

## What you will learn

- **LLMs and Ollama:** how language models generate responses and how to run models locally.
- **LangChain:** model integration, agents, tool calling, and multi-agent collaboration.
- **LangGraph:** stateful workflows, nodes, routers, and conditional execution.
- **RAG and Weaviate:** retrieving real recipe data through semantic search and applying Python-side filters.
- **MCP:** the concepts behind connecting AI applications to external tools and resources through a standard protocol.
- **Opik:** tracing model calls, graph execution, latency, and token usage.
- **Hermes:** using an external AI assistant.

## Requirements

- **Python 3.12** (recommended) and **Git**.
- **Ollama** installed and running, with a compatible Qwen model downloaded.
- Enough RAM/disk space to run the selected model locally.
- An internet connection for the initial package/model downloads.

The basic examples do **not** require an OpenAI API key: they use a model running locally through Ollama.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/Judiiiit/Basics-AI.git
```

### 2. Create and activate a virtual environment

**macOS / Linux / WSL (Ubuntu):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Ubuntu, if creating the environment fails because `venv` is missing, install the corresponding package (for example, `sudo apt install python3.12-venv`) and try again.

### 3. Install `requirements.txt`

The `requirements.txt` file is included in the repository and lists the Python dependencies. **You do not need to download each package manually.** Once the virtual environment is active, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This reads `requirements.txt`, downloads the required packages from the configured Python package index, and installs them into `.venv`.

> **Important:** Installing `requirements.txt` installs Python libraries only. It does **not** install Ollama, download Qwen, create a Weaviate database, configure Opik, or install the Hermes CLI.

### 4. Install Ollama and download the model

Install Ollama from [ollama.com/download](https://ollama.com/download) or by command line using:

**macOS / Linux / WSL (Ubuntu):**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows (PowerShell):**

```bash
irm https://ollama.com/install.ps1 | iex
```

Then, verify the installation with:

```bash
ollama --version
```

Download the model (the model used in this case is qwen3.5:9b, but choose a model that fits your computer's available RAM and hardware resources. For example: qwen3.5:4b (smaller) or qwen3.5:9b (more demanding)):

```bash
ollama pull qwen3.5:9b
```

Wait until the installation is finished. Check that the model is installed:

```bash
ollama list
```

Keep Ollama running while using the examples. Depending on the operating system, this may mean opening the Ollama application or starting its server with `ollama serve` (if it is not already running).

If you use a different model, update the model name in the Python scripts.

### 5. Run your first example

```bash
python test_ollama.py
```

This test sends a prompt from Python through LangChain and Ollama to Qwen. If it prints a model-generated response, the basic local setup is working.

Next, run:

```bash
python 01_simple_agent.py
```

This introduces tool calling: the LLM requests a tool, while LangChain executes the corresponding Python function.

## Examples

Run each example from the repository root, with the virtual environment activated and Ollama running.

| File                                         | Main concept                                                     |
| -------------------------------------------- | ---------------------------------------------------------------- |
| `test_ollama.py`                             | Check the Python → LangChain → Ollama → Qwen connection.         |
| `01_simple_agent.py`                         | Basic agent and a simulated weather tool.                        |
| `02_multi_agent.py`                          | Supervisor coordinating weather and flight-time agents.          |
| `03_my_drone_mission_agent.py`               | Drone-mission feasibility checks combined with LLM explanations. |
| `04_langgraph_drone_mission_agent.py`        | State and conditional workflow for drone-mission checks.         |
| `05_langgraph_autonomous_mission_planner.py` | Mission planning, validation, and possible replanning.           |
| `06_shopping_list.py`                        | Shopping-list operations using a LangChain agent.                |
| `07_shopping_list_langgraph.py`              | Shopping-list operations organized with LangGraph.               |
| `08_shopping_list_weaviate.py`               | Recipe retrieval using Weaviate and RAG.                         |
| `09_shopping_list_weaviate_opik.py`          | Weaviate-based assistant with Opik observability.                |
| `10_shopping_list_weaviate_opik_hermes.py`   | Weekly meal planning with Weaviate, Hermes, and Opik.            |

To run another example, substitute its filename:

```bash
python 07_shopping_list_langgraph.py
```

The examples are designed to be explored progressively. Later scripts have more prerequisites than earlier ones.

## Advanced examples: additional setup

### Weaviate — examples 08–10

The recipe-search examples require a **Weaviate Cloud** instance with a compatible, searchable recipe collection (called `Recetas` in the manual). Configure the connection credentials and collection expected by the corresponding script. Installing `requirements.txt` alone does not provide the database or its contents.

Keep credentials private. If the repository provides `.env.example`, copy it to `.env` and fill in the variables required by your local code:

```bash
cp .env.example .env
```

Do not commit passwords or API keys.

### Opik — examples 09–10

For cloud-based execution tracing, create an [Opik](https://www.comet.com/opik) account and configure the CLI:

```bash
opik configure
```

Provide the requested credentials/workspace and inspect the traces in the configured Opik project after running an example.

### Hermes — example 10

The final example uses **Hermes CLI as an external command**, not as a Python package or an Ollama model. Install and configure Hermes separately, ensure that the `hermes` executable is on your `PATH`, and verify that it works in the terminal before using the weekly-planning example.

The application retrieves candidate recipes from Weaviate, asks Hermes to propose a weekly menu, and uses Python to validate the selected recipes and determine missing ingredients. If Hermes is unavailable or returns unusable output, the manual describes a Python-based fallback using valid recipe candidates.

## Troubleshooting

- **`ModuleNotFoundError`** — Check that `.venv` is activated and reinstall with `python -m pip install -r requirements.txt`.
- **Ollama connection error** — Make sure Ollama is running and that the configured model appears in `ollama list`.
- **Model not found** — Download it with `ollama pull qwen3.5:9b`, or use the correct configured model name.
- **Weaviate connection or empty recipe results** — Verify credentials, collection name, and recipe data.
- **Opik traces missing** — Check `opik configure`, the workspace, and the project's tracing configuration.
- **`hermes: command not found`** — Check that Hermes CLI is installed and available on `PATH`.

To leave the virtual environment:

```bash
deactivate
```

## Documentation

The accompanying **Manual d'explicacions i exemples** provides the full explanations of LLM inference, Ollama, LangChain, LangGraph, RAG, MCP, Opik, and the Python examples. Use this README for orientation and setup; use the manual for detailed code walkthroughs and execution diagrams.
