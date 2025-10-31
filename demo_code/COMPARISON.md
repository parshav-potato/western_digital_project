# Comparison: Original vs Refactored Demo

## Side-by-Side Feature Comparison

| Aspect | Original Demo | New pdf_demo |
|--------|--------------|--------------|
| **Location** | `Rag-raptor-demo/` | `pdf_demo/` |
| **Database** | PostgreSQL (needs Docker/server) | FAISS (local files) |
| **LLM Support** | OpenAI only | OpenAI + Gemini (switchable) |
| **Configuration** | Hardcoded in notebook | `.env` file + `Config` class |
| **Structure** | Single notebook | Modular `src/` package |
| **Package Manager** | Poetry | uv (faster, simpler) |
| **Dependencies** | `pyproject.toml` + manual installs | Auto-sync with `uv sync` |
| **Setup Time** | ~10 minutes | ~5 minutes |
| **Disk Space** | Larger (PostgreSQL) | Smaller (no DB) |

## Code Architecture

### Original Demo
```
Rag-raptor-demo/
├── demo/
│   └── raptor.ipynb    # 50+ cells, everything mixed
├── ingest/
│   └── raptor.py       # Some functions
└── app.py              # Streamlit app
```

### New pdf_demo
```
pdf_demo/
├── src/                # Clean separation
│   ├── config.py       # Config management
│   ├── pdf_processor.py
│   ├── raptor.py
│   └── vector_store.py
├── notebooks/
│   └── raptor_demo.ipynb  # Clean, documented
├── example_usage.py    # Python script version
└── quickstart.sh       # Easy setup
```

## Import Fixes

### Original Issues
```python
# ❌ Import errors in Rag-raptor-demo/ingest/raptor.py
from langchain.prompts import ChatPromptTemplate  # Not found
```

### Fixed in pdf_demo
```python
# ✅ Correct imports
from langchain_core.prompts import ChatPromptTemplate
```

## LLM Provider Switching

### Original
```python
# ❌ Hardcoded OpenAI only
model = ChatOpenAI(temperature=0, model="gpt-4o")
```

### New pdf_demo
```python
# ✅ Flexible provider selection
config = Config(llm_provider="openai")  # or "gemini"
model = config.get_llm_model()
```

## Vector Store Comparison

### Original: PostgreSQL
```python
# ❌ Requires PostgreSQL server
POSTGRES_URL_EMBEDDINDS = os.getenv("POSTGRES_URL_EMBEDDINDS")
vectorstore = PGVector(
    embeddings=embd,
    collection_name=collection_name,
    connection=POSTGRES_URL_EMBEDDINDS,
    use_jsonb=True,
)
```

**Pros:** Production-ready, shared access  
**Cons:** Needs database setup, external dependency

### New: FAISS
```python
# ✅ No database needed
vector_store = FAISSVectorStore(config)
vector_store.create_from_texts(texts)
vector_store.save("data/vector_store")
```

**Pros:** Local, portable, fast  
**Cons:** Single-user (but perfect for demos)

## Setup Process

### Original Demo
```bash
cd Rag-raptor-demo
poetry install  # or manual pip installs
# Setup PostgreSQL database
# Create .env with POSTGRES_URL_EMBEDDINDS
# Fix import errors manually
jupyter notebook demo/raptor.ipynb
```

### New pdf_demo
```bash
cd pdf_demo
uv sync         # Auto-installs everything
cp .env.example .env
# Add API key
jupyter notebook notebooks/raptor_demo.ipynb
```

## Performance

| Metric | Original | New pdf_demo |
|--------|----------|--------------|
| Install time | 3-5 min | 1-2 min (uv) |
| First run | Slower (DB connection) | Faster (local) |
| Dependencies | ~200+ packages | Same, cleaner management |
| Storage | Separate DB | `data/vector_store/` |

## Migration Guide

### If you want to use the new version:

1. **Copy your PDF**:
   ```bash
   cp Rag-raptor-demo/data/fy2024.pdf pdf_demo/data/
   ```

2. **Use same API keys**:
   ```bash
   # Copy from Rag-raptor-demo/.env
   cp Rag-raptor-demo/.env pdf_demo/.env
   ```

3. **Run the new demo**:
   ```bash
   cd pdf_demo
   jupyter notebook notebooks/raptor_demo.ipynb
   ```

### Converting PostgreSQL data to FAISS:

If you have existing vectors in PostgreSQL and want to move to FAISS:

```python
# In original demo - export vectors
from langchain_postgres.vectorstores import PGVector

# Load from Postgres
pg_store = PGVector(...)
docs = pg_store.similarity_search("", k=10000)  # Get all

# In new demo - import to FAISS
from src.vector_store import FAISSVectorStore
from src.config import Config

config = Config(llm_provider="openai")
faiss_store = FAISSVectorStore(config)
faiss_store.create_from_documents(docs)
faiss_store.save("data/vector_store")
```

## When to Use Each?

### Use Original (Rag-raptor-demo) when:
- ✅ You need multi-user access
- ✅ You already have PostgreSQL infrastructure
- ✅ You need persistent, shared storage
- ✅ You're deploying to production with teams

### Use New (pdf_demo) when:
- ✅ You want quick local experimentation
- ✅ You need to switch between LLM providers
- ✅ You prefer modular, maintainable code
- ✅ You don't want database overhead
- ✅ You're learning or prototyping
- ✅ You want easier setup and dependencies

## Both Projects Share:

- ✅ RAPTOR algorithm implementation
- ✅ PDF processing with unstructured
- ✅ Image analysis capabilities
- ✅ Hierarchical clustering (UMAP + GMM)
- ✅ LangChain integration
- ✅ Production-quality embeddings

## Summary

The **new pdf_demo** is a **cleaner, more flexible refactor** that:
- Removes database dependency (FAISS instead of PostgreSQL)
- Adds Gemini support (not just OpenAI)
- Better code organization (modular src/)
- Faster setup (uv instead of poetry)
- Fixed import issues
- Environment-based configuration

It's **perfect for demos, experimentation, and learning**, while the original is better for production deployments with database infrastructure.
