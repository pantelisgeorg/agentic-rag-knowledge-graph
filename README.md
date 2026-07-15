# Agentic RAG with Knowledge Graph

Agentic knowledge retrieval redefined with an AI agent system that combines traditional RAG (vector search) with knowledge graph capabilities. The system uses PostgreSQL with pgvector for semantic search and Neo4j for knowledge graphs. 

Built with:

- Pydantic AI for the AI Agent Framework
- Custom lightweight GraphClient for the Knowledge Graph (replaces the heavy Graphiti client)
- Postgres with PGVector for the Vector Database
- Neo4j for the Knowledge Graph Engine
- FastAPI for the Agent API
- Claude Code for the AI Coding Assistant (See `CLAUDE.md`, `PLANNING.md`, and `TASK.md`)

## Overview

This system includes three main components:

1. **Document Ingestion Pipeline**: Processes markdown documents using semantic chunking and builds both vector embeddings and knowledge graph relationships
2. **AI Agent Interface**: A conversational agent powered by Pydantic AI that can search across both vector database and knowledge graph
3. **Streaming API**: FastAPI backend with real-time streaming responses and comprehensive search capabilities

## Prerequisites

- Python 3.11 or higher
- PostgreSQL database (such as Neon)
- Neo4j database (for knowledge graph)
- LLM Provider API key (OpenAI, Ollama, Gemini, etc.)

## Installation

### 1. Set up a virtual environment

```bash
# Create and activate virtual environment
python -m venv venv       # python3 on Linux
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up required tables in Postgres

Execute the SQL in `sql/schema.sql` to create all necessary tables, indexes, and functions.

Be sure to change the embedding dimensions on lines 31, 67, and 100 based on your embedding model. OpenAI's text-embedding-3-small is 1536 and nomic-embed-text from Ollama is 768 dimensions, for reference.

Note that this script will drop all tables before creating/recreating!

### 4. Set up Neo4j
   Using Neo4j Desktop
1. Download and install [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new project and add a local DBMS
3. Start the DBMS and set a password
4. Note the connection details (URI, username, password)

### 5. Configure environment variables

Create a `.env` file in the project root:

```bash
# Database Configuration (example Neon connection string)
DATABASE_URL=postgresql://username:password@ep-example-12345.us-east-2.aws.neon.tech/neondb

# Neo4j Configuration  
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM Provider Configuration (choose one)
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
LLM_CHOICE=gpt-4.1-mini

# Embedding Configuration
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-your-api-key
EMBEDDING_MODEL=text-embedding-3-small

# Ingestion Configuration
INGESTION_LLM_CHOICE=gpt-4.1-nano  # Faster model for processing

# Application Configuration
APP_ENV=development
LOG_LEVEL=INFO
APP_PORT=8058
```

For other LLM providers:
```bash
# Ollama (Local)
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_CHOICE=nomic-embed-text-v2-moe:latest

# llama.cpp (Local — better MoE GPU offload than Ollama)
# LLM_PROVIDER=openai
# LLM_BASE_URL=http://localhost:8083/v1
# LLM_API_KEY=not-needed
# LLM_CHOICE=unsloth/gemma-4-26B-A4B-it-qat-GGUF:Q4_K_XL

# OpenRouter
LLM_PROVIDER=openrouter
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=your-openrouter-key
LLM_CHOICE=anthropic/claude-3-5-sonnet

# Gemini
LLM_PROVIDER=gemini
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta
LLM_API_KEY=your-gemini-key
LLM_CHOICE=gemini-2.5-flash
```

## Quick Start

### 1. Prepare Your Documents

Add your markdown documents to the `documents/` folder:

```bash
mkdir -p documents
# Add your markdown files here.
```
**Note**: Documents are gitignored — add your own `.md` files here. Be aware that processing many files into the knowledge graph will take significant time (potentially 30+ minutes) due to the computational complexity of entity extraction and relationship building.

### 2. Run Document Ingestion

**Important**: You must run ingestion first to populate the databases before the agent can provide meaningful responses.

```bash
# Convenience script — ingest then consolidate near-duplicate entities
bash ingest.sh

# Same, but wipe all existing data first
bash ingest.sh --clean

# Custom similarity threshold for entity consolidation (default: 0.85)
bash ingest.sh --threshold 0.9
```

Or run the steps individually:

```bash
# Basic ingestion with semantic chunking
python -m ingestion.ingest

# Clean existing data and re-ingest everything
python -m ingestion.ingest --clean

# Consolidate near-duplicate entities after ingestion
python -m ingestion.consolidate_graph --threshold 0.85

# Dry-run consolidation to preview merges without making changes
python -m ingestion.consolidate_graph --dry-run

# Custom settings for faster processing (no knowledge graph)
python -m ingestion.ingest --chunk-size 800 --no-semantic --verbose
```

The ingestion process will:
- Parse and semantically chunk your documents
- Generate embeddings for vector search
- Extract entities and relationships for the knowledge graph
- Store everything in PostgreSQL and Neo4j

After ingestion, the optional **consolidation** step (`consolidate_graph.py`) merges near-duplicate entities in Neo4j. Entities that refer to the same real‑world thing but were extracted with slightly different names (e.g. "Αναξίμανδρος" vs "Αναξίμανδρος ο Μιλήσιος") are grouped by embedding similarity and merged into a single canonical node with all relationships preserved.

NOTE that this can take a while because knowledge graphs are very computationally expensive!

### 3. Configure Agent Behavior (Optional)

Before running the API server, you can customize when the agent uses different tools by modifying the system prompt in `agent/prompts.py`. The system prompt controls:
- When to use vector search vs knowledge graph search
- How to combine results from different sources
- The agent's reasoning strategy for tool selection

### 4. Start the API Server (Terminal 1)

```bash
# Start the FastAPI server
python -m agent.api

# Server will be available at http://localhost:8058
```

### 5. Use the Command Line Interface (Terminal 2)

The CLI provides an interactive way to chat with the agent and see which tools it uses for each query.

```bash
# Start the CLI in a separate terminal from the API (connects to default API at http://localhost:8058)
python cli.py

# Connect to a different URL
python cli.py --url http://localhost:8058

# Connect to a specific port
python cli.py --port 8080
```

#### CLI Features

- **Real-time streaming responses** - See the agent's response as it's generated
- **Tool usage visibility** - Understand which tools the agent used:
  - `vector_search` - Semantic similarity search
  - `graph_search` - Knowledge graph queries
  - `hybrid_search` - Combined search approach
- **Session management** - Maintains conversation context
- **Color-coded output** - Easy to read responses and tool information

#### Example CLI Session

```
🤖 Agentic RAG with Knowledge Graph CLI
============================================================
Connected to: http://localhost:8058

You: Query...........?

🤖 Assistant:
Answer................

🛠 Tools Used:
  1. vector_search 
  2. graph_search 
  3. hybrid_search
  4. get_entity_relationships 
────────────────────────────────────────────────────────────

#### CLI Commands

- `help` - Show available commands
- `health` - Check API connection status
- `clear` - Clear current session
- `exit` or `quit` - Exit the CLI

### 6. Test the System

#### Health Check
```bash
curl http://localhost:8058/health
```

#### Chat with the Agent (Non-streaming)
```bash
curl -X POST "http://localhost:8058/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are Google'\''s main AI initiatives?"
  }'
```

#### Streaming Chat
```bash
curl -X POST "http://localhost:8058/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the relation........",
  }'
```

## How It Works

### The Power of Hybrid RAG + Knowledge Graph

This system combines the best of both worlds:

**Vector Database (PostgreSQL + pgvector)**:
- Semantic similarity search across document chunks
- Fast retrieval of contextually relevant information
- Excellent for finding documents about similar topics

**Knowledge Graph (Neo4j:
- Temporal relationships between entities (companies, people, technologies)
- Graph traversal for discovering connections
- Perfect for understanding partnerships, acquisitions, and evolution over time

**Intelligent Agent**:
- Automatically chooses the best search strategy
- Combines results from both databases
- Provides context-aware responses with source citations

### Example Queries

The system excels at queries that benefit from both semantic search and relationship understanding:

- **Semantic Questions**

- **Relationship Questions**

- **Temporal Questions**

- **Complex Analysis** 

### Why This Architecture Works So Well

1. **Complementary Strengths**: Vector search finds semantically similar content while knowledge graphs reveal hidden connections

2. **Temporal Intelligence**: The graph client tracks how facts are related, revealing connections across documents

3. **Flexible LLM Support**: Switch between OpenAI, Ollama, OpenRouter, or Gemini based on your needs

4. **Production Ready**: Comprehensive testing, error handling, and monitoring

## API Documentation

Visit http://localhost:8058/docs for interactive API documentation once the server is running.

## Key Features

- **Hybrid Search**: Seamlessly combines vector similarity and graph traversal
- **Temporal Knowledge**: Tracks how information changes over time
- **Streaming Responses**: Real-time AI responses with Server-Sent Events
- **Flexible Providers**: Support for multiple LLM and embedding providers
- **Semantic Chunking**: Intelligent document splitting using LLM analysis
- **Production Ready**: Comprehensive testing, logging, and error handling

## Project Structure

```
agentic-rag-knowledge-graph/
├── agent/                  # AI agent and API
│   ├── agent.py           # Main Pydantic AI agent
│   ├── api.py             # FastAPI application
│   ├── providers.py       # LLM provider abstraction
│   └── models.py          # Data models
├── ingestion/             # Document processing
│   ├── ingest.py         # Main ingestion pipeline
│   ├── chunker.py        # Semantic chunking
│   ├── embedder.py       # Embedding generation
│   └── consolidate_graph.py  # Near-duplicate entity merging
├── ingest.sh              # One-shot ingest + consolidate script
├── sql/                   # Database schema
├── documents/             # Your markdown files
└── tests/                # Comprehensive test suite
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agent --cov=ingestion --cov-report=html

# Run specific test categories
pytest tests/agent/
pytest tests/ingestion/
```

## Troubleshooting

### Common Issues

**Database Connection**: Ensure your DATABASE_URL is correct and the database is accessible
```bash
# Test your connection
psql -d "$DATABASE_URL" -c "SELECT 1;"
```

**Neo4j Connection**: Verify your Neo4j instance is running and credentials are correct
```bash
# Check if Neo4j is accessible (adjust URL as needed)
curl -u neo4j:password http://localhost:7474/db/data/
```

**No Results from Agent**: Make sure you've run the ingestion pipeline first
```bash
python -m ingestion.ingest --verbose
```

**LLM API Issues**: Check your API key and provider configuration in `.env`

---

Built with ❤️ using Pydantic AI, FastAPI, PostgreSQL, and Neo4j.

Implemented by Cole Madin 2024 (https://github.com/coleam00) amd modified by George Pantelis 2026 (https://github.com/pantelisgeorg)

MIT LICENCE
