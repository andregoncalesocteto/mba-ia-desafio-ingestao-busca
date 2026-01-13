# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) system that ingests PDF documents into a PostgreSQL vector database (pgvector) and provides a CLI chat interface for querying the content using semantic search.

**Tech Stack:**
- **Vector Database**: PostgreSQL 17 with pgvector extension
- **Embeddings**: OpenAI `text-embedding-3-small` (configurable)
- **LLM**: OpenAI `gpt-4o-mini` (hardcoded in `search.py`)
- **Framework**: LangChain (document loading, text splitting, vector store, embeddings)
- **PDF Processing**: PyPDF via LangChain

## Development Setup

### 1. Start Infrastructure

Start PostgreSQL with pgvector and Adminer:
```bash
docker compose up -d
```

Verify PostgreSQL health:
```bash
docker compose ps
```

The `postgres` service must show `healthy` status.

Optional - Access Adminer at `http://localhost:8080`:
- Server: `postgres`
- Username: `postgres`
- Password: `postgres`
- Database: `rag`

Stop services:
```bash
docker compose down
```

### 2. Python Environment

Activate virtual environment:
```bash
source venv/bin/activate
```

Install dependencies (if needed):
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and configure:

**Required variables:**
- `OPENAI_API_KEY` - OpenAI API key for embeddings and LLM
- `PDF_PATH` - Absolute path to PDF file for ingestion
- `DATABASE_URL` - PostgreSQL connection (default: `postgresql://postgres:postgres@localhost:5432/rag`)

**Optional variables:**
- `PG_VECTOR_COLLECTION_NAME` - Collection name in pgvector (default: `documents`)
- `OPENAI_EMBEDDING_MODEL` - Embedding model (default: `text-embedding-3-small`)

### 4. Ingestion Workflow

Run PDF ingestion (requires PostgreSQL running):
```bash
python src/ingest.py
```

This will:
1. Load PDF from `PDF_PATH`
2. Split into chunks (1000 chars, 150 overlap)
3. Generate embeddings with OpenAI
4. Store in PostgreSQL pgvector collection

### 5. Chat Interface

Run the CLI chat:
```bash
python src/chat.py
```

## Architecture

### Core Components

**[src/ingest.py](src/ingest.py)** - PDF ingestion pipeline
- Loads PDF using `PyPDFLoader`
- Splits text with `RecursiveCharacterTextSplitter` (1000 char chunks, 150 overlap)
- Generates embeddings via `OpenAIEmbeddings`
- Persists to PostgreSQL using `PGVector.from_documents()`

**[src/search.py](src/search.py)** - RAG chain construction
- Builds LCEL chain: retriever → prompt → LLM → output parser
- Retriever fetches k=10 most similar chunks from pgvector
- Uses `PROMPT_TEMPLATE` for strict grounding (Portuguese)
- Returns `chain.invoke(question)` for query execution

**[src/chat.py](src/chat.py)** - CLI chat interface
- Terminal-based chat loop
- Delegates search to `search.py`
- Displays responses to user

### RAG Prompt Strategy

The system uses a strict grounding strategy defined in `PROMPT_TEMPLATE`:
- Answers ONLY based on retrieved context
- Returns "Não tenho informações necessárias para responder sua pergunta." if information is not in context
- No external knowledge or opinions allowed
- No fabrication or interpretation beyond source material

### Database Schema

PostgreSQL with pgvector extension stores:
- **Collection**: `documents` (configurable via `PG_VECTOR_COLLECTION_NAME`)
- **Embeddings**: 1536-dimensional vectors (OpenAI text-embedding-3-small)
- Managed automatically by LangChain's `PGVector` store

## Development Notes

- PostgreSQL must be running (`docker compose up -d`) before running any Python scripts
- The ingestion script can be run multiple times; set `pre_delete_collection=True` in `ingest.py` to clear before re-ingesting
- To change the LLM model, modify the `ChatOpenAI` instantiation in `search.py:88`