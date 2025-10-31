"""
Example script demonstrating PDF RAPTOR processing.
Run this as: python example_usage.py
"""
import os
from src.config import Config
from src.pdf_processor import PDFProcessor
from src.raptor import RAPTORProcessor
from src.vector_store import FAISSVectorStore


def main():
    print("=" * 60)
    print("PDF RAPTOR Demo - Example Usage")
    print("=" * 60)
    
    # Configuration
    # Change to "gemini" to use Google Gemini instead
    LLM_PROVIDER = "openai"
    
    # File paths
    PDF_FILE = "data/your_document.pdf"  # Change this to your PDF file
    IMAGE_DIR = "data/images"
    VECTOR_STORE_DIR = "data/vector_store"
    
    # Create directories
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    
    try:
        # Initialize configuration
        print(f"\n1. Initializing with {LLM_PROVIDER.upper()}...")
        config = Config(llm_provider=LLM_PROVIDER)
        print(f"Config: {config}")
        
        # Process PDF
        print(f"\n2. Processing PDF: {PDF_FILE}")
        if not os.path.exists(PDF_FILE):
            print(f"Error: PDF file not found at {PDF_FILE}")
            print(f"Please place your PDF in the data/ directory")
            return
        
        pdf_processor = PDFProcessor(config)
        documents, raw_texts = pdf_processor.process_pdf(PDF_FILE, IMAGE_DIR)
        
        # Apply RAPTOR
        print(f"\n3. Applying RAPTOR clustering...")
        raptor = RAPTORProcessor(config)
        all_texts = raptor.process(raw_texts, n_levels=3)
        
        # Create vector store
        print(f"\n4. Creating FAISS vector store...")
        vector_store = FAISSVectorStore(config)
        vector_store.create_from_texts(all_texts)
        
        # Save vector store
        print(f"\n5. Saving vector store...")
        vector_store.save(VECTOR_STORE_DIR)
        
        # Test query
        print(f"\n6. Testing similarity search...")
        query = "What are the main topics in this document?"
        results = vector_store.similarity_search(query, k=3)
        
        print(f"\nQuery: {query}")
        print("\nTop 3 Results:")
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(doc.page_content[:200] + "...")
        
        print("\n" + "=" * 60)
        print("Processing complete!")
        print(f"Vector store saved to: {VECTOR_STORE_DIR}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease check:")
        print("  1. Your .env file has the correct API keys")
        print(f"  2. LLM_PROVIDER is set to '{LLM_PROVIDER}'")
        print("  3. The PDF file exists and is accessible")


if __name__ == "__main__":
    main()
