# Irix-AI


Irix-AI is a sophisticated Multi-Agent System (MAS) designed to generate consistent and well-reasoned outputs. It orchestrates multiple specialized AI agents, each with a unique role, to deliberate on complex queries. For simpler questions, it provides a direct, efficient response. The system leverages parallel processing to run agents concurrently and can perform web searches for up-to-date information, ensuring fast and comprehensive analysis.

## Core Features

*   **Dynamic Routing:** A smart router analyzes each user query to decide between a simple `direct` response or a complex `deliberate` path involving multiple agents.
*   **Multi-Agent Deliberation:** For complex tasks, Irix-AI employs a team of agents:
    *   `Analyst Agent`: Deconstructs the problem to identify core assumptions, boundary conditions, and analytical edge cases.
    *   `Critic Agent`: Identifies flaws, risks, logical fallacies, and potential failure modes in a given premise or solution.
    *   `Builder Agent`: Proposes a direct, practical, and actionable solution to the core problem.
*   **Web Search Integration:** For queries requiring current events or real-time data, the router triggers a web search. The results are summarized and provided as context to the deliberating agents.
*   **Complexity Gating:** The system dynamically selects which agents to run. High-complexity queries (score >= 5) trigger the full agent panel, while moderately complex ones use only the `Builder Agent` for efficiency.
*   **Parallel Processing:** Agents run in parallel using threading, significantly speeding up the deliberation process.
*   **Synthesis Model:** A dedicated "heavy" model arbitrates and synthesizes the outputs from the various agents and web context into a single, coherent final answer.
*   **Automated Conversation Memory:** Manages conversation context by automatically summarizing the history when it exceeds a certain length, ensuring long-term coherence without exceeding model context limits.
*   **Telemetry and Self-Evaluation:** Logs detailed telemetry for each interaction, including latency, model usage, and a self-evaluation metric that assesses whether the use of a complex reasoning path was justified.

## How It Works

Irix-AI follows a dynamic workflow to process user input:

1.  **Input & Routing:** The user's prompt is received. A `router` model analyzes the query's complexity, intent, and conversational context. It decides on a `direct` or `deliberate` path and whether a web search is needed.
2.  **Path Selection:**
    *   **Direct Path:** For simple, factual, or low-effort questions, a "light" language model provides an immediate answer.
    *   **Deliberate Path:** For complex questions requiring analysis or if a web search is triggered, the query is dispatched to a team of specialized agents.
3.  **Agent Deliberation:** Based on the router's complexity score, either the solo `Builder Agent` or the full panel (`Analyst`, `Critic`, and `Builder`) process the query simultaneously. If a web search was performed, the summarized content is provided as context.
4.  **Synthesis:** The outputs from all agents are passed to a "heavy" synthesis model. This model resolves contradictions, removes redundancy, and crafts a single, high-quality final answer based on the expert inputs.
5.  **History Management:** The new interaction is appended to the conversation history. The `memory` module periodically summarizes older parts of the conversation to keep the context manageable.
6.  **Logging:** Key metrics from the interaction (the router's decision, models used, latency, and the self-evaluation result) are recorded in a telemetry log file (`_router_telemetry.jsonl`).

## System Architecture

*   `Irix.py`: The main application entry point that orchestrates the entire workflow, from user input to final output.
*   `IrixAI.py`: Contains the core `IrixSystem` class, managing the main logic, agent orchestration, and state.
*   `Agents.py`: Defines the `Agent` class that serves as the blueprint for all specialized agents.
*   `router.py`: Contains the logic for the routing controller, deciding the path and triggering web searches.
*   `memory.py`: Manages the conversation history, including saving, loading, and summarization logic.
*   `telemetry.py`: Implements logging and the self-evaluation mechanism to analyze router performance.
*   `_prompts.json`: A central configuration file containing all prompts for the system, router, agents, and summarizer.
*   `__extract_json_robust.py`: A utility module for reliably extracting JSON from LLM outputs, which is crucial for structured interactions between components.

## Models Used

Irix-AI relies on `ollama` to run various local language models, each chosen for a specific task:

*   **Router Model:** `granite3.1-moe:3b`
*   **Specialized Agents:** `qwen3:1.7b`
*   **Light Model (Direct Path):** `qwen2.5:7b`
*   **Heavy Model (Synthesis):** `qwen3:8b`
*   **Summary Model:** `qwen3:1.7b`

## Getting Started

### Prerequisites

*   Python 3.x
*   Ollama installed and running.

### Installation & Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/hmednejjar/irix-ai.git
    cd irix-ai
    ```

2.  Install the required Python packages:
    ```bash
    pip install ollama ddgs
    ```

3.  Download the necessary models using Ollama:
    ```bash
    ollama pull granite3.1-moe:3b
    ollama pull qwen3:1.7b
    ollama pull qwen2.5:7b
    ollama pull qwen3:8b
    ```

### Running the Application

Execute the main script from your terminal:

```bash
python Irix.py
```

You can now interact with Irix-AI through the command-line interface. Type `exit`, `quit`, or `bye` to end the session.