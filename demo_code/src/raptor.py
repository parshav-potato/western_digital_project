"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) implementation.
Performs hierarchical clustering and summarization of text documents.
"""
import numpy as np
import pandas as pd
import umap.umap_ as umap
from typing import Dict, List, Optional, Tuple
from sklearn.mixture import GaussianMixture
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class RAPTORProcessor:
    """RAPTOR hierarchical text clustering and summarization."""
    
    def __init__(self, config):
        """
        Initialize RAPTOR processor.
        
        Args:
            config: Config instance with LLM and embedding settings
        """
        self.config = config
        self.embeddings = config.get_embedding_model()
        self.llm = config.get_llm_model(temperature=0)
        self.random_seed = config.random_seed
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        text_embeddings = self.embeddings.embed_documents(texts)
        return np.array(text_embeddings)
    
    def global_cluster_embeddings(
        self,
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Perform global dimensionality reduction using UMAP.
        
        Args:
            embeddings: Input embeddings
            dim: Target dimensionality
            n_neighbors: Number of neighbors (default: sqrt of num embeddings)
            metric: Distance metric
            
        Returns:
            Reduced embeddings
        """
        if n_neighbors is None:
            n_neighbors = int((len(embeddings) - 1) ** 0.5)
        
        return umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=dim,
            metric=metric,
            random_state=self.random_seed
        ).fit_transform(embeddings)
    
    def local_cluster_embeddings(
        self,
        embeddings: np.ndarray,
        dim: int,
        num_neighbors: int = 10,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Perform local dimensionality reduction using UMAP.
        
        Args:
            embeddings: Input embeddings
            dim: Target dimensionality
            num_neighbors: Number of neighbors
            metric: Distance metric
            
        Returns:
            Reduced embeddings
        """
        return umap.UMAP(
            n_neighbors=num_neighbors,
            n_components=dim,
            metric=metric,
            random_state=self.random_seed
        ).fit_transform(embeddings)
    
    def get_optimal_clusters(
        self,
        embeddings: np.ndarray,
        max_clusters: int = 50
    ) -> int:
        """
        Determine optimal number of clusters using BIC with GMM.
        
        Args:
            embeddings: Input embeddings
            max_clusters: Maximum number of clusters to consider
            
        Returns:
            Optimal number of clusters
        """
        max_clusters = min(max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        
        for n in n_clusters:
            gm = GaussianMixture(
                n_components=n,
                random_state=self.random_seed
            )
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        
        return n_clusters[np.argmin(bics)]
    
    def gmm_cluster(
        self,
        embeddings: np.ndarray,
        threshold: float
    ) -> Tuple[List[np.ndarray], int]:
        """
        Cluster embeddings using Gaussian Mixture Model.
        
        Args:
            embeddings: Input embeddings
            threshold: Probability threshold for cluster assignment
            
        Returns:
            Tuple of (cluster labels, number of clusters)
        """
        n_clusters = self.get_optimal_clusters(embeddings)
        gm = GaussianMixture(
            n_components=n_clusters,
            random_state=self.random_seed
        )
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, n_clusters
    
    def perform_clustering(
        self,
        embeddings: np.ndarray,
        dim: int,
        threshold: float,
    ) -> List[np.ndarray]:
        """
        Perform hierarchical clustering on embeddings.
        
        Args:
            embeddings: Input embeddings
            dim: Target dimensionality for UMAP
            threshold: Probability threshold for GMM
            
        Returns:
            List of cluster labels for each embedding
        """
        if len(embeddings) <= dim + 1:
            # Not enough data for clustering
            return [np.array([0]) for _ in range(len(embeddings))]
        
        # Global dimensionality reduction
        reduced_embeddings_global = self.global_cluster_embeddings(embeddings, dim)
        
        # Global clustering
        global_clusters, n_global_clusters = self.gmm_cluster(
            reduced_embeddings_global, threshold
        )
        
        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0
        
        # Local clustering within each global cluster
        for i in range(n_global_clusters):
            # Extract embeddings for this global cluster
            global_cluster_embeddings_ = embeddings[
                np.array([i in gc for gc in global_clusters])
            ]
            
            if len(global_cluster_embeddings_) == 0:
                continue
            
            if len(global_cluster_embeddings_) <= dim + 1:
                # Small cluster - direct assignment
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
                n_local_clusters = 1
            else:
                # Local dimensionality reduction and clustering
                reduced_embeddings_local = self.local_cluster_embeddings(
                    global_cluster_embeddings_, dim
                )
                local_clusters, n_local_clusters = self.gmm_cluster(
                    reduced_embeddings_local, threshold
                )
            
            # Assign local cluster IDs
            for j in range(n_local_clusters):
                local_cluster_embeddings_ = global_cluster_embeddings_[
                    np.array([j in lc for lc in local_clusters])
                ]
                indices = np.where(
                    (embeddings == local_cluster_embeddings_[:, None]).all(-1)
                )[1]
                for idx in indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )
            
            total_clusters += n_local_clusters
        
        return all_local_clusters
    
    def embed_cluster_texts(self, texts: List[str]) -> pd.DataFrame:
        """
        Embed and cluster texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with texts, embeddings, and cluster labels
        """
        text_embeddings = self.embed_texts(texts)
        cluster_labels = self.perform_clustering(text_embeddings, 10, 0.1)
        
        df = pd.DataFrame()
        df["text"] = texts
        df["embd"] = list(text_embeddings)
        df["cluster"] = cluster_labels
        return df
    
    def format_cluster_texts(self, df: pd.DataFrame) -> str:
        """
        Format texts from a cluster into a single string.
        
        Args:
            df: DataFrame containing text column
            
        Returns:
            Formatted string
        """
        texts = df["text"].tolist()
        return "--- --- \n --- --- ".join(texts)
    
    def embed_cluster_summarize_texts(
        self,
        texts: List[str],
        level: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Embed, cluster, and summarize texts.
        
        Args:
            texts: List of text strings
            level: Current hierarchical level
            
        Returns:
            Tuple of (clusters DataFrame, summaries DataFrame)
        """
        # Embed and cluster
        df_clusters = self.embed_cluster_texts(texts)
        
        # Expand DataFrame for processing
        expanded_list = []
        for _, row in df_clusters.iterrows():
            for cluster in row["cluster"]:
                expanded_list.append({
                    "text": row["text"],
                    "embd": row["embd"],
                    "cluster": cluster
                })
        
        expanded_df = pd.DataFrame(expanded_list)
        all_clusters = expanded_df["cluster"].unique()
        
        print(f"  Level {level}: Generated {len(all_clusters)} clusters")
        
        # Summarize each cluster
        template = """Please provide a comprehensive summary of the following content. 
Be accurate with numbers and facts, do not make things up.

Content:
{context}

Summary:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        summaries = []
        for i in all_clusters:
            df_cluster = expanded_df[expanded_df["cluster"] == i]
            formatted_txt = self.format_cluster_texts(df_cluster)
            summary = chain.invoke({"context": formatted_txt})
            summaries.append(summary)
        
        df_summary = pd.DataFrame({
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        })
        
        return df_clusters, df_summary
    
    def recursive_embed_cluster_summarize(
        self,
        texts: List[str],
        level: int = 1,
        n_levels: int = 3
    ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Recursively embed, cluster, and summarize texts.
        
        Args:
            texts: List of text strings
            level: Current recursion level
            n_levels: Maximum recursion depth
            
        Returns:
            Dictionary mapping levels to (clusters, summaries) DataFrames
        """
        results = {}
        
        # Process current level
        df_clusters, df_summary = self.embed_cluster_summarize_texts(texts, level)
        results[level] = (df_clusters, df_summary)
        
        # Recurse if needed
        unique_clusters = df_summary["cluster"].nunique()
        if level < n_levels and unique_clusters > 1:
            new_texts = df_summary["summaries"].tolist()
            next_level_results = self.recursive_embed_cluster_summarize(
                new_texts, level + 1, n_levels
            )
            results.update(next_level_results)
        
        return results
    
    def process(
        self,
        texts: List[str],
        n_levels: int = 3
    ) -> List[str]:
        """
        Process texts through RAPTOR algorithm.
        
        Args:
            texts: List of text strings (leaf nodes)
            n_levels: Number of hierarchical levels
            
        Returns:
            List of all texts including summaries from all levels
        """
        print(f"\nBuilding RAPTOR tree with {n_levels} levels...")
        print(f"Starting with {len(texts)} leaf texts")
        
        # Build the tree
        results = self.recursive_embed_cluster_summarize(texts, level=1, n_levels=n_levels)
        
        # Collect all texts from all levels
        all_texts = texts.copy()
        
        for level in sorted(results.keys()):
            summaries = results[level][1]["summaries"].tolist()
            all_texts.extend(summaries)
            print(f"  Level {level}: Added {len(summaries)} summaries")
        
        print(f"RAPTOR processing complete: {len(all_texts)} total texts")
        return all_texts
