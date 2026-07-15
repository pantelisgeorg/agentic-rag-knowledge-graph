# Local Setup Reference

Reference for managing the local services, credentials, and app startup for the
Agentic RAG with Knowledge Graph project.

All services were installed locally via `apt` (no Docker). See `setup_databases.sh`
for the original automated install.

---

## Service management (systemd — requires sudo)

### PostgreSQL 16 + pgvector

```bash
sudo systemctl start postgresql      # start
sudo systemctl stop postgresql       # stop
sudo systemctl restart postgresql    # restart
sudo systemctl status postgresql     # status
```

- **DB:** `agentic_rag_db`
- **User / Password:** `raguser` / `ragpass123`
- **Host:** `localhost:5432`
- **Connection string (in `.env`):** `postgresql://raguser:ragpass123@localhost:5432/agentic_rag_db`
- **Connect from terminal:**
  ```bash
  PGPASSWORD=ragpass123 psql -h localhost -U raguser -d agentic_rag_db
  ```
- **Superuser access:** `sudo -u postgres psql`

### Neo4j 5

```bash
sudo systemctl start neo4j
sudo systemctl stop neo4j
sudo systemctl restart neo4j
sudo systemctl status neo4j
```

- **URI:** `bolt://localhost:7687`
- **User / Password:** `neo4j` / `neo4jragpass`
- **Browser UI:** http://localhost:7474
- **Connect from terminal:**
  ```bash
  cypher-shell -u neo4j -p neo4jragpass
  ```

### Ollama

Currently unused for LLM/embeddings (the project switched to OpenAI), but still
installed and available as a fallback local provider.

```bash
sudo systemctl start ollama
sudo systemctl status ollama
ollama list                         # list pulled models
```

- **Endpoint:** `http://localhost:11434`
- **Pulled models:** `gemma4:26b_ctx`, `nomic-embed-text-v2-moe:latest`, and others.

### llama.cpp router (optional, user-managed)

A llama.cpp server providing better MoE GPU offload than Ollama for the
`gemma-4-26B` model. Configured via `~/Desktop/llama.cpp/models.ini`.

- **Endpoint:** `http://localhost:8083/v1`
- **Model name:** `unsloth/gemma-4-26B-A4B-it-qat-GGUF:Q4_K_XL`
- Start/stop is managed by the user's router (not systemd).

---

## Starting the app

Prerequisite: PostgreSQL and Neo4j services must be running first.

1. Activate the virtual environment:
   ```bash
   source /home/george/Desktop/rag_systems/agentic-rag-knowledge-graph/.venv/bin/activate
   ```
2. Start the API server (port 8058):
   ```bash
   python -m agent.api
   ```
3. In a separate terminal (with venv activated), start the CLI:
   ```bash
   python cli.py
   ```

Health check: `curl http://localhost:8058/health`

---

## Ingestion

```bash
# Full run (vector + knowledge graph), fresh start
python -m ingestion.ingest -d documents --clean --chunk-size 1500 --chunk-overlap 250

# Vector-only (fast, skips knowledge graph)
python -m ingestion.ingest -d documents --clean --fast --no-semantic --no-entities --chunk-size 1500 --chunk-overlap 250
```

Flags:
- `--clean` — wipe existing documents/chunks/sessions/messages in PostgreSQL and
  clear the Neo4j graph before ingesting.
- `--fast` — skip knowledge graph building (vector DB only).
- `--no-semantic` — use simple rule-based chunking (no LLM).
- `--no-entities` — skip rule-based entity metadata extraction.
- `--chunk-size` / `--chunk-overlap` — chunking sizes in characters.

---

## Credentials summary

All values live in `.env` (not committed).

| Service        | User      | Password        |
|----------------|-----------|-----------------|
| PostgreSQL     | `raguser` | `ragpass123`    |
| Neo4j          | `neo4j`   | `neo4jragpass`  |
| OpenAI API     | —         | key in `.env` (`LLM_API_KEY` / `EMBEDDING_API_KEY`) |

---

## Recreate everything from scratch

1. **Reinstall databases** (PostgreSQL + pgvector + Neo4j, users, and schema):
   ```bash
   sudo bash setup_databases.sh
   ```
2. **Reset only the PostgreSQL schema** (destructive — drops all tables/functions):
   ```bash
   PGPASSWORD=ragpass123 psql -h localhost -U raguser -d agentic_rag_db -f sql/schema.sql
   ```
3. **Clear only the Neo4j graph** (keeps PostgreSQL data):
   ```bash
   cypher-shell -u neo4j -p neo4jragpass "MATCH (n) DETACH DELETE n;"
   ```

---

## Embedding dimension note

The PostgreSQL schema's `vector(N)` dimension must match the configured embedding
model. The project is currently configured for OpenAI `text-embedding-3-small`
(1536 dimensions):

- `sql/schema.sql` → `vector(1536)` (3 locations: table column + 2 functions)
- `.env` → `VECTOR_DIMENSION=1536`

If switching embedding models, update both the schema dimension and `VECTOR_DIMENSION`,
then re-apply the schema and re-ingest. For example, Ollama's
`nomic-embed-text-v2-moe` uses 768 dimensions.
