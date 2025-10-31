# RAPTOR Demo - PDF & Code Retrieval

A comprehensive demo showcasing RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) for hierarchical document understanding and semantic search. Includes implementations for both PDF processing and code retrieval with support for OpenAI and Google Gemini.

## Overview

This project demonstrates advanced RAG (Retrieval-Augmented Generation) techniques using RAPTOR clustering to create multi-level document representations. Unlike traditional chunking methods, RAPTOR builds a tree structure that enables retrieval at different levels of abstraction.

## Features

### Core Capabilities
- **PDF Processing**: Extract text, tables, and images from PDF documents
- **Code Retrieval**: Semantic search across Python codebases with hierarchical summaries
- **Dual LLM Support**: Switchable between OpenAI GPT-4 and Google Gemini
- **Local Embeddings**: Free offline embeddings using sentence-transformers (no API costs)
- **RAPTOR Clustering**: Multi-level hierarchical text organization
- **FAISS Vector Store**: Fast similarity search without external databases
- **Batch Processing**: Automatic batching with progress tracking to prevent crashes
- **Clean Architecture**: Professional, modular code structure

### Embedding Options
1. **Local Embeddings (Recommended)**:
   - Uses sentence-transformers (all-MiniLM-L6-v2)
   - No API costs or rate limits
   - Works completely offline
   - 384 dimensions
   - Slightly slower first run (~90MB model download)

2. **API Embeddings**:
   - OpenAI: text-embedding-ada-002 (1536 dimensions)
   - Gemini: models/embedding-001 (768 dimensions)
   - Higher quality but costs money and has rate limits

## Project Structure

```
demo_code/
├── data/                      # Data files and outputs
│   ├── images/               # Extracted PDF images
│   ├── vector_store/         # Saved FAISS indices
│   └── code_vector_store/    # Code retrieval indices
├── notebooks/                 # Interactive demos
│   ├── raptor_demo_pdf.ipynb        # PDF processing demo
│   └── raptor_code_retrieval.ipynb  # Code search demo
├── src/                       # Core modules
│   ├── __init__.py
│   ├── config.py             # Configuration & LLM setup
│   ├── pdf_processor.py      # PDF extraction with batching
│   ├── raptor.py             # RAPTOR clustering algorithm
│   └── vector_store.py       # FAISS operations
├── .env.example              # Environment template
├── example_usage.py          # Command-line demo script
├── pyproject.toml            # Dependencies (uv)
├── COMPARISON.md             # Feature comparison
├── PERFORMANCE_TIPS.md       # Optimization guide
├── SETUP_COMPLETE.md         # Setup verification
└── README.md                 # This file
```

## Installation

### Prerequisites
- Python 3.9 - 3.13 (tested on 3.13)
- uv package manager ([install](https://github.com/astral-sh/uv))
- API keys (optional if using local embeddings)

### Setup

1. **Navigate to project directory**:
   ```bash
   cd /home/parshav-potato/projects/wd_research/demo_code
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Configure environment** (optional for local embeddings):
   ```bash
   cp .env.example .env
   # Edit .env with your API keys if using API embeddings
   ```

4. **Set API keys** (skip if using local embeddings):
   ```bash
   # For OpenAI
   export OPENAI_API_KEY="your-key-here"
   
   # For Gemini
   export GOOGLE_API_KEY="your-key-here"
   ```

## Quick Start

### Option 1: Jupyter Notebooks (Recommended)

#### PDF Processing Demo
```bash
jupyter notebook notebooks/raptor_demo_pdf.ipynb
```

**Key Cells:**
- Cell 5: Configure LLM and embeddings (`USE_LOCAL_EMBEDDINGS=True` for free local embeddings)
- Cell 6: Set PDF path
- Cell 8: Process PDF with automatic batching
- Cell 10: Apply RAPTOR clustering
- Cell 15: Query with semantic search

#### Code Retrieval Demo
```bash
jupyter notebook notebooks/raptor_code_retrieval.ipynb
```

**Key Cells:**
- Cell 4: Configure settings
- Cell 5: Set codebase path
- Cell 7: Extract code chunks (functions/classes)
- Cell 9: Build RAPTOR tree
- Cell 15: Semantic code search

### Option 2: Python Script

```python
from src.config import Config
from src.pdf_processor import PDFProcessor
from src.raptor import RAPTORProcessor
from src.vector_store import FAISSVectorStore

# Use local embeddings (free, no API needed)
config = Config(
    llm_provider="gemini",           # or "openai"
    use_local_embeddings=True        # Free local embeddings
)

# Process PDF with automatic batching
processor = PDFProcessor(config)
documents, raw_texts = processor.process_pdf(
    pdf_path="data/your_file.pdf",
    output_image_dir="data/images",
    max_elements=None,               # None = process all
    skip_images=False,               # Set True to skip images
    batch_size=10                    # Elements per batch
)

# Apply RAPTOR clustering (3 levels)
raptor = RAPTORProcessor(config)
all_texts = raptor.process(texts=raw_texts, n_levels=3)

# Create and save vector store
vector_store = FAISSVectorStore(config)
vector_store.create_from_texts(all_texts)
vector_store.save("data/vector_store")

# Query the vector store
results = vector_store.similarity_search(
    "What are the main topics?",
    k=5
)
```

### Option 3: Command Line

```bash
python example_usage.py
```

## Configuration

### LLM Selection

```python
# OpenAI GPT-4
config = Config(llm_provider="openai")  # Uses gpt-4o by default

# Google Gemini
config = Config(llm_provider="gemini")  # Uses gemini-pro by default

# Custom model
config = Config(llm_provider="openai", model_name="gpt-4-turbo")
```

### Embedding Selection

```python
# Local embeddings (free, recommended)
config = Config(use_local_embeddings=True)

# OpenAI embeddings
config = Config(llm_provider="openai", use_local_embeddings=False)

# Gemini embeddings  
config = Config(llm_provider="gemini", use_local_embeddings=False)
```

### Processing Options

```python
# For large PDFs (prevent crashes and reduce costs)
documents, raw_texts = processor.process_pdf(
    pdf_path="large_document.pdf",
    output_image_dir="data/images",
    max_elements=100,        # Limit processing (None = all)
    skip_images=True,        # Skip images for speed
    batch_size=10            # Process in batches (default: 10)
)

# For complete processing
documents, raw_texts = processor.process_pdf(
    pdf_path="document.pdf",
    output_image_dir="data/images",
    max_elements=None,       # Process everything
    skip_images=False,       # Include images
    batch_size=10            # Automatic batching
)
```

## Usage Examples

### PDF Question Answering

```python
# Load existing vector store
config = Config(use_local_embeddings=True)
vector_store = FAISSVectorStore(config)
vector_store.load("data/vector_store")

# Ask questions
query = "What is the attention mechanism?"
results = vector_store.similarity_search_with_score(query, k=5)

for doc, score in results:
    print(f"Score: {score:.4f}")
    print(doc.page_content)
    print("-" * 80)
```

### Code Search

```python
# Search for code patterns
query = "How to configure embeddings?"
results = vector_store.similarity_search(query, k=3)

for doc in results:
    print(doc.page_content)
```

### Tree-Level Retrieval

RAPTOR creates multiple abstraction levels:
- **Level 0 (Leaf)**: Original text chunks
- **Level 1**: Summaries of related chunks
- **Level 2**: High-level summaries
- **Level 3**: Overall document summary

Search naturally returns relevant content from all levels.

## API Keys

### OpenAI
1. Visit https://platform.openai.com/api-keys
2. Create new key
3. Add to `.env`: `OPENAI_API_KEY=sk-...`

### Google Gemini
1. Visit https://makersuite.google.com/app/apikey
2. Create new key
3. Add to `.env`: `GOOGLE_API_KEY=...`

### No API Key Needed
- Use `use_local_embeddings=True` for free local embeddings
- Still need LLM API key for generating summaries

## Performance Optimization

### For Large Documents (>100 pages)

```python
# Test run first (fast, cheap)
documents, raw_texts = processor.process_pdf(
    pdf_path=PDF_FILE,
    max_elements=50,          # Test with 50 chunks first
    skip_images=True,         # Skip images initially
)

# Full run after testing
documents, raw_texts = processor.process_pdf(
    pdf_path=PDF_FILE,
    max_elements=None,        # Process all
    skip_images=False,        # Include images
    batch_size=10             # Automatic delays prevent crashes
)
```

### Cost Estimation

The processor shows estimates before starting:
```
Processing Plan:
  Total elements found: 1500
  Will process: 1500
  Batch size: 10
  Estimated batches: 150
  Estimated time: ~150 minutes
  Estimated cost: ~$22.50
```

See [PERFORMANCE_TIPS.md](PERFORMANCE_TIPS.md) for detailed optimization strategies.

## Comparison with Original

| Feature | Original (Rag-raptor-demo) | This Version (demo_code) |
|---------|----------------------------|--------------------------|
| **Database** | PostgreSQL (requires setup) | FAISS (local files only) |
| **LLM** | OpenAI only | OpenAI + Gemini (switchable) |
| **Embeddings** | OpenAI API only | OpenAI + Gemini + Local (free) |
| **Structure** | Single notebook | Modular src/ directory |
| **Dependencies** | Mixed pip/conda | uv package manager |
| **Config** | Hardcoded | Environment variables |
| **Batch Processing** | None (crashes on large PDFs) | Automatic batching |
| **Code Style** | With emojis | Professional, clean |
| **Error Handling** | Basic | Comprehensive with retries |
| **Progress Tracking** | None | Detailed batch progress |
| **Cost Control** | None | Estimates + limits |
| **Code Retrieval** | Not included | Full implementation |

## Features in Detail

### Automatic Batch Processing
- Processes large documents in configurable batches (default: 10 elements)
- Automatic delays between batches prevent API rate limits
- Progress tracking for each batch
- Error handling continues processing if individual elements fail
- Prevents system crashes from memory issues

### Local Embeddings
- Uses sentence-transformers library
- Model: all-MiniLM-L6-v2 (384 dimensions)
- ~90MB download on first run
- Cached locally for offline use
- No ongoing API costs
- Suitable for most semantic search tasks

### RAPTOR Algorithm
- Recursive clustering creates tree structure
- Each level summarizes clusters from level below
- Enables retrieval at different abstraction levels
- Configurable depth (typically 3 levels)
- Uses UMAP for dimensionality reduction
- Gaussian Mixture Models for clustering

## Troubleshooting

### Common Issues

**Import errors**:
```bash
# Clear cache and reinstall
rm -rf .venv
uv sync
```

**UMAP import error**:
```python
# Fixed in src/raptor.py
import umap.umap_ as umap  # Not: from umap import UMAP
```

**Gemini quota exceeded**:
```python
# Switch to local embeddings
config = Config(use_local_embeddings=True)
```

**System crashes on large PDFs**:
```python
# Use smaller batch_size or max_elements
processor.process_pdf(
    pdf_path=PDF_FILE,
    max_elements=100,
    batch_size=5  # Smaller batches
)
```

**Sentence-transformers errors**:
```bash
# Ensure compatible versions
pip install 'transformers>=4.34.0,<4.40.0' 'tokenizers<0.20.0' 'sentence-transformers>=2.2.0,<3.0.0'
```

## Documentation

- **[COMPARISON.md](COMPARISON.md)**: Detailed feature comparison
- **[PERFORMANCE_TIPS.md](PERFORMANCE_TIPS.md)**: Optimization strategies
- **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)**: Verification guide

## Requirements

**Python**: 3.9 - 3.13 (not 3.9.7)

**Key Dependencies**:
- langchain-openai / langchain-google-genai
- sentence-transformers (for local embeddings)
- faiss-cpu (vector store)
- umap-learn (dimensionality reduction)
- unstructured (PDF parsing)
- scikit-learn (clustering)

See [pyproject.toml](pyproject.toml) for complete list.

## Use Cases

1. **Document Q&A**: Build RAG systems for technical documentation
2. **Code Search**: Semantic search across large codebases
3. **Research Assistant**: Multi-level document understanding
4. **Knowledge Base**: Create searchable knowledge repositories
5. **Content Analysis**: Hierarchical document summarization

## Advanced Topics

### Custom Clustering
```python
# Adjust RAPTOR parameters
raptor = RAPTORProcessor(config)
# Modify clustering in src/raptor.py:
# - n_neighbors (UMAP)
# - n_components (GMM)
# - threshold (cluster reduction)
```

### Multiple Documents
```python
# Process multiple PDFs into one vector store
all_texts = []
for pdf_file in pdf_files:
    _, raw_texts = processor.process_pdf(pdf_file, ...)
    texts = raptor.process(raw_texts, n_levels=3)
    all_texts.extend(texts)

vector_store.create_from_texts(all_texts)
```

### Hybrid Search
```python
# Combine with keyword search
semantic_results = vector_store.similarity_search(query, k=10)
# Filter or rerank based on keywords
```

## Contributing

Contributions welcome! Please:
1. Follow the existing code style (professional, no emojis)
2. Add tests for new features
3. Update documentation
4. Submit pull requests

## License

MIT License - see LICENSE file for details

## Acknowledgments

- RAPTOR paper: [Sarthi et al., 2024](https://arxiv.org/abs/2401.18059)
- LangChain for LLM abstractions
- FAISS for efficient vector search
- sentence-transformers for local embeddings

## Support

For issues or questions:
1. Check [PERFORMANCE_TIPS.md](PERFORMANCE_TIPS.md)
2. Review [COMPARISON.md](COMPARISON.md)
3. Open an issue on GitHub
