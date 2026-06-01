# Minimal LangGraph Agent

This is a tiny LangGraph agent project with an LLM node, a tool node, and conditional routing.

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
```

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
GET  /events/ask?question=...&mode=auto
WS   /ws
```

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
