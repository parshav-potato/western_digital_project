"""
FAISS vector store module for efficient similarity search.
"""
import os
import pickle
from typing import List, Optional, Tuple
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class FAISSVectorStore:
    """Manage FAISS vector store for document embeddings."""
    
    def __init__(self, config):
        """
        Initialize FAISS vector store.
        
        Args:
            config: Config instance with embedding settings
        """
        self.config = config
        self.embeddings = config.get_embedding_model()
        self.vector_store = None
        
    def create_from_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None
    ) -> FAISS:
        """
        Create a new FAISS vector store from texts.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts for each text
            
        Returns:
            FAISS vector store instance
        """
        print(f"\nCreating FAISS vector store from {len(texts)} texts...")
        
        if metadatas is None:
            metadatas = [{"id": str(i)} for i in range(len(texts))]
        
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        print(f"Vector store created successfully")
        return self.vector_store
    
    def create_from_documents(self, documents: List[Document]) -> FAISS:
        """
        Create a new FAISS vector store from Document objects.
        
        Args:
            documents: List of Document objects
            
        Returns:
            FAISS vector store instance
        """
        print(f"\nCreating FAISS vector store from {len(documents)} documents...")
        
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        print(f"Vector store created successfully")
        return self.vector_store
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None
    ):
        """
        Add texts to existing vector store.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Use create_from_texts first.")
        
        print(f"Adding {len(texts)} texts to vector store...")
        
        if metadatas is None:
            metadatas = [{"id": str(i)} for i in range(len(texts))]
        
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        print(f"Texts added successfully")
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to existing vector store.
        
        Args:
            documents: List of Document objects
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Use create_from_documents first.")
        
        print(f"Adding {len(documents)} documents to vector store...")
        self.vector_store.add_documents(documents=documents)
        print(f"Documents added successfully")
    
    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of similar Documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def save(self, directory: str, index_name: str = "faiss_index"):
        """
        Save vector store to disk.
        
        Args:
            directory: Directory to save the vector store
            index_name: Name for the index file
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save.")
        
        os.makedirs(directory, exist_ok=True)
        
        print(f"\nSaving vector store to {directory}...")
        
        # Save FAISS index
        index_path = os.path.join(directory, index_name)
        self.vector_store.save_local(index_path)
        
        # Save config info
        config_path = os.path.join(directory, "config.pkl")
        config_info = {
            "llm_provider": self.config.llm_provider,
            "model_name": self.config.model_name
        }
        with open(config_path, "wb") as f:
            pickle.dump(config_info, f)
        
        print(f"Vector store saved successfully")
    
    def load(self, directory: str, index_name: str = "faiss_index"):
        """
        Load vector store from disk.
        
        Args:
            directory: Directory containing the vector store
            index_name: Name of the index file
        """
        print(f"\n📂 Loading vector store from {directory}...")
        
        index_path = os.path.join(directory, index_name)
        
        if not os.path.exists(index_path):
            raise ValueError(f"Vector store not found at {index_path}")
        
        self.vector_store = FAISS.load_local(
            index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Load config info if available
        config_path = os.path.join(directory, "config.pkl")
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config_info = pickle.load(f)
                print(f"  Loaded config: {config_info}")
        
        print(f"Vector store loaded successfully")
    
    def get_stats(self) -> dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        if self.vector_store is None:
            return {"status": "not initialized"}
        
        # Get the number of vectors
        try:
            n_vectors = self.vector_store.index.ntotal
        except:
            n_vectors = "unknown"
        
        return {
            "status": "initialized",
            "n_vectors": n_vectors,
            "embedding_provider": self.config.llm_provider,
        }
