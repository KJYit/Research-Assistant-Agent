# 🔬 Research Assistant Agent

A multi-turn AI research agent built with **LangGraph**, **Google Gemini**, and **Tavily Search**. The agent decomposes research topics into sub-questions, retrieves information from multiple tools, evaluates its own output quality, and delivers comprehensive, cited answers.

---

## Architecture

```
[START]
   │
   ▼
[planner]  ──────── Decomposes topic into 2–4 sub-questions (Gemini)
   │
   ▼
[researcher] ─────── Tavily web search per sub-question (Tool 1)
   │
   ├── retry_count > 1 → [error_handler] → [responder] → [END]
   │
   ▼
[retriever] ──────── In-memory TF-IDF vector store lookup (Tool 2)
   │
   ▼
[synthesizer] ────── Merges results, evaluates sufficiency (Gemini)
   │
   ├── needs_more_research=True → back to [planner] (max 2 re-plans)
   │
   ▼
[responder] ──────── Drafts final cited answer (Gemini)
   │
   ▼
 [END]
```

### Nodes (6 total)

| Node | Responsibility |
|------|---------------|
| `planner` | Decomposes topic into 2–4 targeted sub-questions using Gemini |
| `researcher` | Calls Tavily Search for each sub-question (Tool 1) |
| `retriever` | Queries in-memory TF-IDF knowledge base (Tool 2) |
| `synthesizer` | Merges results, deduplicates sources, evaluates sufficiency |
| `responder` | Generates the final cited answer |
| `error_handler` | Handles tool failures gracefully |

### Conditional Edges (2)

1. **After `researcher`**: if `retry_count > 1` → `error_handler`; else → `retriever`
2. **After `synthesizer`**: if `needs_more_research=True` and `turn_count < 3` → `planner`; else → `responder`

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/KJYit/Research-Assistant-Agent.git
cd Research-Assistant-Agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Getting API keys:**
- Google Gemini: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey) (free)
- Tavily Search: [https://tavily.com](https://tavily.com) (free tier: 1,000 searches/month)

### 5. Run the Application

You can run the agent in two ways: via the Command Line Interface (CLI) or via the Web UI.

#### Option A: Run the CLI
```bash
python main.py
```

#### Option B: Run the Web UI (API + Frontend)
**1. Start the backend API:**
Open a terminal and run the FastAPI server:
```bash
uvicorn api:app --reload
```
The API will be available at `http://localhost:8000`.

**2. Start the frontend:**
Open a *new* terminal and start the React/Vite development server:
```bash
cd frontend
npm install
npm run dev
```
Open your browser to `http://localhost:5173` to use the web interface.

---

## Usage

```
======================================================
   🔬 Research Assistant Agent (powered by Gemini + Tavily)
======================================================
Commands:
  • Type a research topic and press Enter
  • Type 'new'  to start a fresh research session
  • Type 'quit' or Ctrl+C to exit
======================================================

Research topic > How does retrieval-augmented generation work?

[Planner] Decomposing topic: 'How does retrieval-augmented generation work?' (turn 1)
[Planner] Generated 3 sub-questions:
  1. What is retrieval-augmented generation (RAG)?
  2. How does the retrieval component work in RAG systems?
  3. What are the benefits of RAG over standard LLMs?
...

ANSWER
──────────────────────────────────────────────────────────────
Retrieval-Augmented Generation (RAG) is a technique that combines...

Sources:
[1] https://example.com/rag-paper
[2] internal://docs/rag_overview
──────────────────────────────────────────────────────────────

Research topic > What frameworks are used to build RAG systems?  ← follow-up turn
```

The agent **maintains full conversation history** across turns within the same session. Follow-up questions benefit from prior research context.

---

## Project Structure

```
Research-Assistant-Agent/
├── main.py              # CLI entry point and session management
├── api.py               # FastAPI backend entry point
├── graph.py             # LangGraph StateGraph definition
├── state.py             # Shared TypedDict state schema
├── requirements.txt
├── .env.example
├── frontend/            # React/Vite web user interface
├── nodes/
│   ├── planner.py       # Topic decomposition node
│   ├── researcher.py    # Tavily web search node
│   ├── retriever.py     # Vector store retrieval node
│   ├── synthesizer.py   # Result merging + sufficiency check node
│   ├── responder.py     # Final answer generation node
│   └── error_handler.py # Graceful error recovery node
└── tools/
    └── search_tools.py  # Tavily + TF-IDF retriever implementations
```