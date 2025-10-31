# 🎉 PDF RAPTOR Demo - Setup Complete!

## 📁 Project Structure

```
pdf_demo/
├── 📄 README.md              # Full documentation
├── 📄 .env.example           # Environment template
├── 📄 .gitignore             # Git ignore rules
├── 📄 pyproject.toml         # Dependencies
├── 📄 quickstart.sh          # Quick setup script
├── 📄 example_usage.py       # Python example script
│
├── 📂 data/                  # Your PDFs and outputs
│   └── 📂 images/           # Extracted images
│
├── 📂 notebooks/             # Jupyter notebooks
│   └── 📓 raptor_demo.ipynb # Main interactive demo
│
└── 📂 src/                   # Source modules
    ├── config.py            # Configuration & API keys
    ├── pdf_processor.py     # PDF extraction
    ├── raptor.py            # RAPTOR clustering
    └── vector_store.py      # FAISS operations
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd /home/parshav-potato/projects/wd_research/pdf_demo

# Copy and edit .env file
cp .env.example .env
# Edit .env with your API keys
```

### 2. Choose Your LLM Provider

In `.env`, set:
```bash
LLM_PROVIDER=openai  # or "gemini"

# For OpenAI:
OPENAI_API_KEY=your_key_here

# For Gemini:
GOOGLE_API_KEY=your_key_here
```

### 3. Add Your PDF

```bash
# Place your PDF in the data/ directory
cp /path/to/your/file.pdf data/your_document.pdf
```

### 4. Run the Demo

**Option A: Jupyter Notebook (Recommended)**
```bash
# Activate the virtual environment
source .venv/bin/activate

# Start Jupyter
jupyter notebook notebooks/raptor_demo.ipynb
```

**Option B: Python Script**
```bash
# Edit example_usage.py to set your PDF path
# Then run:
source .venv/bin/activate
python example_usage.py
```

## ✨ Key Features

### 1. **Dual LLM Support**
- ✅ OpenAI (GPT-4o, GPT-4-turbo, GPT-3.5-turbo)
- ✅ Google Gemini (gemini-pro, gemini-1.5-pro)
- Easy switching via `.env` file

### 2. **PDF Processing**
- Extracts text chunks
- Parses tables
- Analyzes images with vision models
- Generates summaries for all content

### 3. **RAPTOR Clustering**
- Hierarchical text organization (3 levels)
- Recursive summarization
- Better retrieval at multiple abstraction levels

### 4. **FAISS Vector Store**
- Local, no database required
- Fast similarity search
- Save/load capabilities

## 🔧 Configuration Options

### In Notebooks or Scripts:

```python
from src.config import Config

# Use OpenAI
config = Config(llm_provider="openai")

# Or use Gemini
config = Config(llm_provider="gemini")

# Get embedding model
embeddings = config.get_embedding_model()

# Get LLM
llm = config.get_llm_model(temperature=0)
```

## 📊 What Happens in RAPTOR?

```
Level 0 (Leaf):     [Original texts from PDF]
                           ↓
Level 1:           [Cluster summaries of related texts]
                           ↓
Level 2:           [Summaries of Level 1 clusters]
                           ↓
Level 3:           [High-level summaries]
```

All levels are stored in the vector store for retrieval!

## 🆚 Differences from Original

| Feature | Original Demo | This Version |
|---------|--------------|--------------|
| **Database** | PostgreSQL | FAISS (local) |
| **LLM** | OpenAI only | OpenAI + Gemini |
| **Structure** | Single notebook | Modular `src/` |
| **Setup** | Poetry | uv (faster) |
| **Config** | Hardcoded | Environment-based |
| **Imports** | Fixed path issues | Clean imports |

## 🐛 Troubleshooting

### "Import could not be resolved"
```bash
cd /home/parshav-potato/projects/wd_research/pdf_demo
uv sync
```

### "API key not found"
Check your `.env` file has the correct keys:
```bash
cat .env
```

### "PDF not found"
Ensure your PDF is in the `data/` directory:
```bash
ls data/*.pdf
```

## 📚 Module Overview

### `src/config.py`
- Manages API keys
- Switches between OpenAI/Gemini
- Provides embedding and LLM models

### `src/pdf_processor.py`
- Extracts PDF content (text, tables, images)
- Generates summaries using LLM
- Handles image analysis

### `src/raptor.py`
- Implements RAPTOR algorithm
- Performs hierarchical clustering
- Generates recursive summaries

### `src/vector_store.py`
- FAISS vector store operations
- Similarity search
- Save/load functionality

## 🎯 Next Steps

1. **Try both providers**: Switch between OpenAI and Gemini
2. **Experiment with levels**: Change `n_levels` in RAPTOR
3. **Custom queries**: Test different search queries
4. **Integration**: Add to your RAG pipeline
5. **Batch processing**: Process multiple PDFs

## 📝 Example Workflow

```python
from src.config import Config
from src.pdf_processor import PDFProcessor
from src.raptor import RAPTORProcessor
from src.vector_store import FAISSVectorStore

# Setup
config = Config(llm_provider="openai")

# Process PDF
processor = PDFProcessor(config)
docs, texts = processor.process_pdf("data/my.pdf")

# Apply RAPTOR
raptor = RAPTORProcessor(config)
all_texts = raptor.process(texts, n_levels=3)

# Create vector store
store = FAISSVectorStore(config)
store.create_from_texts(all_texts)
store.save("data/vector_store")

# Query
results = store.similarity_search("my question", k=5)
```

## 🤝 Contributing

Feel free to:
- Add more LLM providers
- Improve error handling
- Add more vector store backends
- Enhance PDF processing

## 📄 License

MIT

---

**Built with** ❤️ **using uv, LangChain, FAISS, and modern Python**
