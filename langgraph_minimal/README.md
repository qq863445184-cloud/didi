# Minimal LangGraph Agent

This is a local LangGraph agent project that demonstrates a controllable agent
architecture with ReAct, tools, agentic RAG, layered memory, context assembly,
and protocol adapters.

## 当前能力

这个项目目前已经实现：

- OpenAI-compatible LLM access, default model name `gpt-5.5`.
- General ReAct + Reflection agent:
  `planner -> agent -> tools -> agent -> verifier -> reflector -> writer -> END`.
- Explicit agentic RAG graph:
  `retrieve -> verify -> retry if needed -> writer -> END`.
- Auto router graph:
  project/code/RAG questions go to RAG; general tasks go to the ReAct agent.
- Tool calling:
  time, calculator, project file listing, project file reading, project document search.
- LangChain RAG:
  document loading, recursive chunking, in-memory vector store, dense retrieval,
  sparse BM25 retrieval, RRF fusion, neighbor chunk expansion, optional query
  rewriting, optional verifier retry, optional reranking.
- Layered memory:
  working memory, conversation memory, profile memory, project memory, and
  reflection memory.
- Context engineering:
  system instruction, profile memory, project memory, conversation memory,
  question, verifier notes, and evidence are assembled with budgets and dedup.
- Chinese-first prompts:
  general agent, RAG verifier, and RAG writer all use Chinese instructions.
- Protocol adapters:
  CLI, HTTP, A2A-like JSON-RPC, SSE, WebSocket, and MCP stdio server.

## 实现与优化记录

Key implementation decisions and optimizations:

- `.env` loading uses `load_dotenv(override=True)` so local project settings win
  over stale shell environment variables.
- `.env`, `.venv`, and persisted memory JSON files are ignored by Git to avoid
  committing secrets or local runtime state.
- Default RAG mode favors fast local development:
  `EMBEDDING_PROVIDER=local_hash`, query rewriting disabled, reranker disabled,
  verifier disabled, and low top-k values.
- Higher-quality RAG is still available through
  `EMBEDDING_PROVIDER=sentence_transformers` with
  `BAAI/bge-small-zh-v1.5`, or through OpenAI-compatible embeddings if the
  endpoint supports embeddings.
- The general agent was upgraded from a simple tool loop to an explicit ReAct
  workflow with planning and verification.
- The general agent now includes a Reflection node. When verification finds an
  insufficient answer, the reflector extracts a reusable lesson and persists it
  to reflection memory.
- The ReAct verifier reads the full execution trace, including tool calls and
  tool observations, so it can correctly verify whether tool-backed answers are
  grounded.
- The writer stores `final_answer` instead of appending duplicate final messages,
  keeping conversation history smaller.
- `session_id` now flows through CLI, HTTP, SSE, WebSocket, MCP, router, general
  agent, and RAG writer.
- One-shot CLI/API/MCP calls persist conversation turns, not only `--chat` mode.
- RAG evidence is trimmed by evidence block, preserving `Source` labels as much
  as possible for citation quality.
- Reflection memory is loaded into the system context so future runs can benefit
  from previous failure lessons.
- File tools are sandboxed to `langgraph_minimal` and hide `.env`, `.venv`, and
  `__pycache__`.

## Configuration

Create `.env` from `.env.example` and set:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
MODEL_NAME=gpt-5.5
EMBEDDING_PROVIDER=local_hash
EMBEDDING_MODEL=text-embedding-3-small
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
RAG_QUERY_REWRITE=false
RAG_DENSE_TOP_K=8
RAG_SPARSE_TOP_K=8
RAG_FINAL_TOP_K=3
RAG_RERANKER_ENABLED=false
RAG_RERANKER_PROVIDER=llm
RAG_RERANKER_MODEL=BAAI/bge-reranker-base
RAG_VERIFY_ENABLED=false
RAG_MAX_ATTEMPTS=1
RECURSION_LIMIT=10
```

Available tools:

- `get_current_time`: get the current time in UTC+8.
- `calculate`: evaluate basic arithmetic.
- `list_files`: list non-hidden files inside this project.
- `read_text_file`: read a UTF-8 text file inside this project.
- `search_project_docs`: retrieve cited project chunks with LangChain RAG.

File tools are sandboxed to this project and hide `.env`, `.venv`, and `__pycache__`.

The RAG tool uses LangChain `RecursiveCharacterTextSplitter` and
`InMemoryVectorStore`. By default, `EMBEDDING_PROVIDER=local_hash` keeps CLI
startup fast. For higher-quality local embeddings, set
`EMBEDDING_PROVIDER=sentence_transformers` to use `BAAI/bge-small-zh-v1.5`.
If your OpenAI-compatible endpoint supports embeddings, set
`EMBEDDING_PROVIDER=openai` and choose `EMBEDDING_MODEL`.

The retriever uses dense vector search, sparse BM25 search, RRF fusion, and
neighbor chunk expansion by default. Query rewriting, verifier retries, and
reranking are available but disabled in the fast default config. Enable
`RAG_QUERY_REWRITE`, `RAG_VERIFY_ENABLED`, or `RAG_RERANKER_ENABLED` for higher
quality at higher latency. Set `RAG_RERANKER_PROVIDER=cross_encoder` to try a
local cross-encoder reranker.

The general agent uses an explicit ReAct + Reflection graph:

```text
planner -> agent -> tools -> agent -> verifier -> reflector -> writer -> END
```

The `agent -> tools -> agent` loop may run repeatedly when the model decides it
needs tool observations. The verifier can send the task back to the agent once
when the candidate answer is not sufficiently grounded. The reflector records
reusable failure lessons before retrying or finishing.

For explicit agentic RAG, use `--rag`. This runs:

```text
retrieve -> writer

Full mode:

retrieve -> verify -> retry if needed -> writer
```

The default one-shot CLI uses a router graph:

```text
router -> general agent
       -> agentic RAG
```

Use `--general` to force the general tool agent, or `--rag` to force RAG.

Memory is split into four layers:

- Working memory: LangGraph state, not persisted.
- Conversation memory: `memory/sessions/<session>.json`.
- Profile memory: `memory/profile.json`.
- Project memory: `memory/project.json`.
- Reflection memory: `memory/reflections.json`.

Context is assembled through `app/context.py` instead of directly concatenating
memory and evidence. It separates system instructions, user profile, project
memory, conversation memory, current question, verifier notes, and retrieved
evidence with simple character budgets and line-level deduplication.

## Setup

```powershell
cd langgraph_minimal
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Directly

```powershell
python -m app.cli
```

This runs one demo question that asks the agent to call the time tool.

## One-Shot Task

```powershell
python -m app.cli "请计算 19*24"
python -m app.cli "读取 README.md 并总结一下"
python -m app.cli --rag "这个 agent 当前 RAG 用了哪些高级检索技术？"
python -m app.cli --general "现在北京时间几点？"
python -m app.cli --session dev "继续刚才的问题"
```

One-shot mode also persists conversation memory under the selected `--session`.

## Chat Mode

```powershell
python -m app.cli --chat
python -m app.cli --chat --session dev
```

Type `exit` to quit.

Example questions:

```text
现在北京时间几点？
请计算 19*24。
读取 README.md 并总结一下。
搜索项目文档，说明这个 agent 有哪些工具，并带引用。
用一句话介绍一下 LangGraph。
```

## Optional: Run With LangGraph Dev Server

```powershell
pip install -U "langgraph-cli[inmem]"
langgraph dev
```

The graph is exported as `minimal_agent` in `langgraph.json`.

## Protocol Servers

HTTP / A2A / SSE / WebSocket:

```powershell
python -m app.server
```

Endpoints:

```text
GET  /health
POST /ask
POST /rag
GET  /.well-known/agent-card.json
POST /a2a
GET  /events/ask?question=...&mode=auto&session_id=default
WS   /ws
```

`POST /ask`, `POST /rag`, SSE, WebSocket, and A2A requests all accept
`session_id` so callers can keep separate conversation memories.

MCP stdio server:

```powershell
python -m app.mcp_server
```

MCP tools:

```text
agent_ask
project_rag
search_project_docs
calculate
get_current_time
```
