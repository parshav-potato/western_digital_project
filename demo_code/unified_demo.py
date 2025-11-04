"""
RAPTOR Unified Demo Script

This script demonstrates three RAPTOR retrieval workflows:
1. Code Retrieval - Hierarchical code search from Python codebases
2. PDF Processing - Extract and search PDF documents
3. Documentation Scraping - Crawl and index documentation websites

The demo runs with default values. Press Enter to continue through each step.
"""

import sys
import os

# Suppress all warnings and set environment variables BEFORE any imports
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

# Suppress specific warnings
import logging
logging.getLogger().setLevel(logging.ERROR)

from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath('.'))

# Remove cached modules
modules_to_remove = [key for key in sys.modules.keys() if key.startswith('src.')]
for module in modules_to_remove:
    del sys.modules[module]

# Import modules
from src.config import Config
from src.raptor import RAPTORProcessor
from src.vector_store import FAISSVectorStore
from src.code_processor import CodeProcessor
from src.pdf_processor import PDFProcessor
from src.docs_scraper import DocumentationScraper


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def wait_for_user(prompt="Press Enter to continue"):
    """Wait for user to press Enter."""
    input(f"\n{prompt}... ")


def configure_system():
    """Configure LLM provider and embeddings with defaults."""
    print_header("System Configuration")
    
    # Default configuration
    llm_provider = "gemini"
    use_local = True
    
    print("Default configuration:")
    print(f"  LLM Provider: {llm_provider}")
    print(f"  Embeddings: Local (sentence-transformers, free, offline)")
    
    config = Config(llm_provider=llm_provider, use_local_embeddings=use_local)
    
    wait_for_user("Press Enter to continue with this configuration")
    
    return config


def workflow_code_retrieval(config):
    """Execute code retrieval workflow."""
    print_header("Workflow 1: Code Retrieval")
    
    # Default configuration
    default_codebase = os.path.join(os.path.dirname(__file__), "src")
    vector_store_dir = os.path.join(os.path.dirname(__file__), "data", "code_vector_store")
    os.makedirs(vector_store_dir, exist_ok=True)
    
    print(f"Codebase path: {default_codebase}")
    print(f"Vector store directory: {vector_store_dir}")
    
    wait_for_user("Press Enter to start code extraction")
    
    # Step 1: Extract code
    print("\nStep 1: Extracting code from codebase...")
    code_processor = CodeProcessor()
    code_chunks = code_processor.extract_code_chunks(default_codebase)
    
    print(f"\nExtracted {len(code_chunks)} code chunks")
    if code_chunks:
        print(f"Sample chunk preview:\n{code_chunks[0][:300]}...")
    
    wait_for_user("Press Enter to apply RAPTOR clustering")
    
    # Step 2: Apply RAPTOR
    print("\nStep 2: Building RAPTOR tree structure...")
    raptor = RAPTORProcessor(config)
    all_code_texts = raptor.process(texts=code_chunks, n_levels=3)
    
    print(f"\nRAPTOR Results:")
    print(f"  Original code chunks: {len(code_chunks)}")
    print(f"  Total texts with summaries: {len(all_code_texts)}")
    print(f"  New summaries created: {len(all_code_texts) - len(code_chunks)}")
    
    wait_for_user("Press Enter to create vector store")
    
    # Step 3: Create vector store
    print("\nStep 3: Creating vector store...")
    vector_store = FAISSVectorStore(config)
    vector_store.create_from_texts(all_code_texts)
    
    stats = vector_store.get_stats()
    print(f"\nVector Store Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save vector store
    vector_store.save(vector_store_dir)
    print(f"\nVector store saved to {vector_store_dir}")
    
    wait_for_user("Press Enter to perform semantic search demo")
    
    # Step 4: Demo searches
    print("\nStep 4: Semantic Search Demo")
    
    demo_queries = [
        "How to configure the LLM provider?",
        "Code for RAPTOR clustering",
        "Vector store implementation"
    ]
    
    for query in demo_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        results = vector_store.similarity_search_with_score(query, k=2)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\nResult {i} (Score: {score:.4f}):")
            print("-" * 80)
            print(doc.page_content[:400] + "...")
        
        wait_for_user("Press Enter for next query")
    
    # Optional custom search
    print("\n" + "="*80)
    custom_query = input("\nEnter your own search query (or press Enter to skip): ").strip()
    if custom_query:
        print(f"\nSearching for: {custom_query}")
        results = vector_store.similarity_search_with_score(custom_query, k=3)
        
        print(f"\nTop {len(results)} Results:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\nResult {i} (Score: {score:.4f}):")
            print("-" * 80)
            print(doc.page_content[:400] + "...")
    
    print("\nCode retrieval workflow completed.")


def workflow_pdf_processing(config):
    """Execute PDF processing workflow."""
    print_header("Workflow 2: PDF Processing")
    
    # Default configuration
    default_pdf = os.path.join(os.path.dirname(__file__), "data", "1706.03762v7.pdf")
    image_output_dir = os.path.join(os.path.dirname(__file__), "data", "images")
    vector_store_dir = os.path.join(os.path.dirname(__file__), "data", "vector_store")
    
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(vector_store_dir, exist_ok=True)
    
    # Check if default PDF exists
    if not os.path.exists(default_pdf):
        print(f"Default PDF not found: {default_pdf}")
        custom_pdf = input("Enter path to PDF file: ").strip()
        if not custom_pdf or not os.path.exists(custom_pdf):
            print("No valid PDF provided. Skipping PDF workflow.")
            return
        default_pdf = custom_pdf
    
    print(f"PDF file: {default_pdf}")
    print(f"Images directory: {image_output_dir}")
    print(f"Vector store directory: {vector_store_dir}")
    
    wait_for_user("Press Enter to start PDF extraction")
    
    # Step 1: Extract from PDF
    print("\nStep 1: Extracting content from PDF...")
    pdf_processor = PDFProcessor(config)
    documents, raw_texts = pdf_processor.process_pdf(
        pdf_path=default_pdf,
        output_image_dir=image_output_dir
    )
    
    print(f"\nExtraction Summary:")
    print(f"  Total documents: {len(documents)}")
    print(f"  Total raw texts: {len(raw_texts)}")
    
    if documents:
        print(f"\nSample document:")
        print(f"  Type: {documents[0].metadata.get('type', 'unknown')}")
        print(f"  Content preview: {documents[0].page_content[:200]}...")
    
    wait_for_user("Press Enter to apply RAPTOR clustering")
    
    # Step 2: Apply RAPTOR
    print("\nStep 2: Building RAPTOR tree structure...")
    raptor = RAPTORProcessor(config)
    all_texts = raptor.process(texts=raw_texts, n_levels=3)
    
    print(f"\nRAPTOR Results:")
    print(f"  Original texts: {len(raw_texts)}")
    print(f"  Total texts with summaries: {len(all_texts)}")
    print(f"  New summaries created: {len(all_texts) - len(raw_texts)}")
    
    wait_for_user("Press Enter to create vector store")
    
    # Step 3: Create vector store
    print("\nStep 3: Creating vector store...")
    vector_store = FAISSVectorStore(config)
    vector_store.create_from_texts(all_texts)
    
    stats = vector_store.get_stats()
    print(f"\nVector Store Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save vector store
    vector_store.save(vector_store_dir)
    print(f"\nVector store saved to {vector_store_dir}")
    
    wait_for_user("Press Enter to query the vector store")
    
    # Step 4: Demo queries
    print("\nStep 4: Query Demo")
    
    demo_queries = [
        "What are the main topics discussed?",
        "Explain the key methodology",
        "What are the results?"
    ]
    
    for query in demo_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        results = vector_store.similarity_search_with_score(query, k=2)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\nResult {i} (Score: {score:.4f}):")
            print(f"{doc.page_content[:300]}...")
        
        wait_for_user("Press Enter for next query")
    
    # Optional custom query
    print("\n" + "="*80)
    custom_query = input("\nEnter your own query (or press Enter to skip): ").strip()
    if custom_query:
        print(f"\nQuerying: {custom_query}")
        results = vector_store.similarity_search_with_score(custom_query, k=3)
        
        print(f"\nTop {len(results)} Results:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\nResult {i} (Score: {score:.4f}):")
            print(f"{doc.page_content[:300]}...")
    
    print("\nPDF processing workflow completed.")


def workflow_docs_scraping(config):
    """Execute documentation scraping workflow."""
    print_header("Workflow 3: Documentation Scraping")
    
    # Default configuration
    docs_url = "https://fastapi.tiangolo.com/"
    max_depth = 2
    max_pages = 30
    delay = 1.0
    
    cache_dir = os.path.join(os.path.dirname(__file__), "data", "docs_cache")
    vector_store_dir = os.path.join(os.path.dirname(__file__), "data", "docs_vector_store")
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(vector_store_dir, exist_ok=True)
    
    print("Default configuration:")
    print(f"  Documentation URL: {docs_url}")
    print(f"  Max Depth: {max_depth}")
    print(f"  Max Pages: {max_pages}")
    print(f"  Delay: {delay}s")
    
    # Allow custom URL
    print("\n" + "="*80)
    custom_url = input("Enter custom documentation URL (or press Enter for default): ").strip()
    if custom_url:
        docs_url = custom_url
    
    wait_for_user("Press Enter to start documentation scraping")
    
    # Step 1: Scrape documentation
    print("\nStep 1: Scraping documentation...")
    scraper = DocumentationScraper(
        base_url=docs_url,
        max_depth=max_depth,
        max_pages=max_pages,
        delay=delay
    )
    
    scraped_pages = scraper.scrape()
    
    print(f"\nScraped {len(scraped_pages)} pages")
    if scraped_pages:
        print(f"\nSample pages:")
        for i, page in enumerate(scraped_pages[:3], 1):
            print(f"  {i}. {page['title'][:60]}... ({page['length']} chars)")
    
    # Convert to chunks
    doc_chunks = []
    for page in scraped_pages:
        chunk = f"""# {page['title']}
URL: {page['url']}
Depth: {page['depth']}

{page['content']}
"""
        doc_chunks.append(chunk)
    
    print(f"\nCreated {len(doc_chunks)} documentation chunks")
    
    # Save scraped data
    cache_file = os.path.join(cache_dir, f"docs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(scraped_pages, f, indent=2, ensure_ascii=False)
    print(f"Scraped data saved to cache")
    
    wait_for_user("Press Enter to apply RAPTOR clustering")
    
    # Step 2: Apply RAPTOR
    print("\nStep 2: Building RAPTOR tree structure...")
    raptor = RAPTORProcessor(config)
    all_doc_texts = raptor.process(texts=doc_chunks, n_levels=3)
    
    print(f"\nRAPTOR Results:")
    print(f"  Original documentation pages: {len(doc_chunks)}")
    print(f"  Total texts with summaries: {len(all_doc_texts)}")
    print(f"  New summaries created: {len(all_doc_texts) - len(doc_chunks)}")
    
    wait_for_user("Press Enter to create vector store")
    
    # Step 3: Create vector store
    print("\nStep 3: Creating vector store...")
    vector_store = FAISSVectorStore(config)
    vector_store.create_from_texts(all_doc_texts)
    
    stats = vector_store.get_stats()
    print(f"\nVector Store Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save vector store
    vector_store.save(vector_store_dir)
    print(f"\nVector store saved to {vector_store_dir}")
    
    wait_for_user("Press Enter to search documentation")
    
    # Step 4: Demo searches
    print("\nStep 4: Documentation Search Demo")
    
    demo_queries = [
        "How do I get started?",
        "What are the main features?",
        "How to deploy to production?"
    ]
    
    for query in demo_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        results = vector_store.similarity_search_with_score(query, k=2)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\nResult {i} (Score: {score:.4f}):")
            print("-" * 80)
            content_preview = doc.page_content[:400]
            print(content_preview + "..." if len(doc.page_content) > 400 else content_preview)
        
        wait_for_user("Press Enter for next query")
    
    # Optional custom search
    print("\n" + "="*80)
    custom_query = input("\nEnter your own search query (or press Enter to skip): ").strip()
    if custom_query:
        print(f"\nSearching: {custom_query}")
        results = vector_store.similarity_search_with_score(custom_query, k=3)
        
        print(f"\nTop {len(results)} Results:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\nResult {i} (Score: {score:.4f}):")
            print("-" * 80)
            content_preview = doc.page_content[:400]
            print(content_preview + "..." if len(doc.page_content) > 400 else content_preview)
    
    print("\nDocumentation scraping workflow completed.")


def main():
    """Main execution function."""
    print_header("RAPTOR Unified Demo")
    
    print("This demo demonstrates three hierarchical retrieval workflows:")
    print("  1. Code Retrieval - Search Python codebases")
    print("  2. PDF Processing - Extract and search PDF documents")
    print("  3. Documentation Scraping - Index documentation websites")
    print("\nThe demo uses default values. Press Enter to continue through each step.")
    
    wait_for_user("Press Enter to begin")
    
    # Configure system
    config = configure_system()
    
    # Workflow selection
    print("\nAvailable workflows:")
    print("  1. Code Retrieval")
    print("  2. PDF Processing")
    print("  3. Documentation Scraping")
    print("  4. All workflows (default)")
    
    choice = input("\nSelect workflow (1-4, or press Enter for all): ").strip() or "4"
    
    # Execute workflows
    try:
        if choice in ['1', '4']:
            workflow_code_retrieval(config)
        
        if choice in ['2', '4']:
            workflow_pdf_processing(config)
        
        if choice in ['3', '4']:
            workflow_docs_scraping(config)
        
        print_header("Demo Completed")
        print("All selected workflows have been executed successfully.")
        print("\nThank you for using the RAPTOR demo!")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
