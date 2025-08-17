"""
RAPTOR indexer implementation for hierarchical tree-based document organization
Based on the RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) approach
Uses Google Gemini for summarization instead of OpenAI as per steering rules
"""

import logging
import uuid
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document as LangChainDocument

from ..core.interfaces import BaseIndexer
from ..core.models import Document
from ..core.config import PipelineConfig


logger = logging.getLogger(__name__)


@dataclass
class RAPTORNode:
    """Represents a node in the RAPTOR tree"""
    node_id: str
    content: str
    summary: str
    embedding: Optional[List[float]]
    level: int
    children: List[str]  # IDs of child nodes
    parent: Optional[str]  # ID of parent node
    metadata: Dict[str, Any]


class RAPTORIndexer(BaseIndexer):
    """
    RAPTOR indexer that creates hierarchical tree structures with clustered summaries.
    Implements tree-based document organization for multi-level retrieval.
    """
    
    def __init__(self, config: PipelineConfig, max_levels: int = 3, cluster_size_threshold: int = 5):
        """
        Initialize the RAPTORIndexer with configuration.
        
        Args:
            config: Pipeline configuration containing model and embedding settings
            max_levels: Maximum number of levels in the tree
            cluster_size_threshold: Minimum cluster size to create a new level
        """
        self.config = config
        self.max_levels = max_levels
        self.cluster_size_threshold = cluster_size_threshold
        
        # Initialize Google Gemini embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model if config.embedding_model != "text-embedding-ada-002" 
                  else "models/embedding-001"  # Default Google embedding model
        )
        
        # Initialize Google Gemini LLM for summarization
        self.llm = ChatGoogleGenerativeAI(
            model=config.llm_model if config.llm_model != "gpt-3.5-turbo" 
                  else "gemini-1.5-flash",  # Default Google model
            temperature=config.temperature
        )
        
        # Create summarization chain for clusters
        self.cluster_summary_chain = (
            {"docs": lambda x: x}
            | ChatPromptTemplate.from_template(
                "Summarize the following documents into a coherent overview:\n\n{docs}"
            )
            | self.llm
            | StrOutputParser()
        )
        
        # Tree structure storage
        self.nodes: Dict[str, RAPTORNode] = {}
        self.root_nodes: List[str] = []  # IDs of root level nodes
        self.level_nodes: Dict[int, List[str]] = {}  # Nodes by level
        
        # Vector store for embeddings at each level
        self.vectorstores: Dict[int, Chroma] = {}
        
        self._document_count = 0
        
        logger.info(f"RAPTORIndexer initialized with max_levels={max_levels}, "
                   f"cluster_size_threshold={cluster_size_threshold}")
    
    def index_documents(self, documents: List[Document]) -> bool:
        """
        Index documents by creating a hierarchical tree structure with clustering and summarization.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            bool: True if indexing was successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided for indexing")
                return True
            
            logger.info(f"Building RAPTOR tree for {len(documents)} documents")
            
            # Step 1: Create leaf nodes from documents
            leaf_nodes = self._create_leaf_nodes(documents)
            
            # Step 2: Build the tree bottom-up
            current_level = 0
            current_nodes = leaf_nodes
            
            while len(current_nodes) > 1 and current_level < self.max_levels:
                logger.info(f"Processing level {current_level} with {len(current_nodes)} nodes")
                
                # Cluster nodes at current level
                clusters = self._cluster_nodes(current_nodes)
                
                # Create parent nodes from clusters
                parent_nodes = []
                for cluster in clusters:
                    if len(cluster) >= self.cluster_size_threshold:
                        parent_node = self._create_parent_node(cluster, current_level + 1)
                        parent_nodes.append(parent_node)
                    else:
                        # If cluster is too small, promote nodes to next level
                        parent_nodes.extend(cluster)
                
                # Store nodes at this level
                self.level_nodes[current_level] = [node.node_id for node in current_nodes]
                
                # Create vector store for this level
                self._create_vectorstore_for_level(current_level, current_nodes)
                
                current_nodes = parent_nodes
                current_level += 1
            
            # Handle the final level
            if current_nodes:
                self.level_nodes[current_level] = [node.node_id for node in current_nodes]
                self._create_vectorstore_for_level(current_level, current_nodes)
                self.root_nodes = [node.node_id for node in current_nodes]
            
            self._document_count += len(documents)
            logger.info(f"Successfully built RAPTOR tree with {current_level + 1} levels. "
                       f"Total documents: {self._document_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error building RAPTOR tree: {str(e)}")
            return False
    
    def _create_leaf_nodes(self, documents: List[Document]) -> List[RAPTORNode]:
        """Create leaf nodes from input documents"""
        leaf_nodes = []
        
        logger.info("Creating leaf nodes from documents")
        
        for doc in documents:
            node_id = str(uuid.uuid4())
            
            # Generate embedding for the document
            embedding = self.embeddings.embed_query(doc.content)
            
            # For leaf nodes, content and summary are the same
            node = RAPTORNode(
                node_id=node_id,
                content=doc.content,
                summary=doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                embedding=embedding,
                level=0,
                children=[],
                parent=None,
                metadata=doc.metadata
            )
            
            self.nodes[node_id] = node
            leaf_nodes.append(node)
        
        logger.info(f"Created {len(leaf_nodes)} leaf nodes")
        return leaf_nodes
    
    def _cluster_nodes(self, nodes: List[RAPTORNode]) -> List[List[RAPTORNode]]:
        """Cluster nodes based on their embeddings"""
        if len(nodes) <= self.cluster_size_threshold:
            return [nodes]
        
        # Extract embeddings
        embeddings = np.array([node.embedding for node in nodes])
        
        # Determine optimal number of clusters
        n_clusters = min(len(nodes) // self.cluster_size_threshold, len(nodes) - 1)
        n_clusters = max(2, n_clusters)  # At least 2 clusters
        
        logger.info(f"Clustering {len(nodes)} nodes into {n_clusters} clusters")
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Group nodes by cluster
        clusters = [[] for _ in range(n_clusters)]
        for node, label in zip(nodes, cluster_labels):
            clusters[label].append(node)
        
        # Filter out empty clusters
        clusters = [cluster for cluster in clusters if cluster]
        
        logger.info(f"Created {len(clusters)} clusters")
        return clusters
    
    def _create_parent_node(self, child_nodes: List[RAPTORNode], level: int) -> RAPTORNode:
        """Create a parent node from a cluster of child nodes"""
        node_id = str(uuid.uuid4())
        
        # Combine content from child nodes for summarization
        combined_content = "\n\n".join([
            f"Document {i+1}: {node.summary}" 
            for i, node in enumerate(child_nodes)
        ])
        
        # Generate summary using the LLM
        try:
            summary = self.cluster_summary_chain.invoke(combined_content)
        except Exception as e:
            logger.warning(f"Failed to generate summary for cluster: {str(e)}")
            # Fallback to truncated combined content
            summary = combined_content[:1000] + "..." if len(combined_content) > 1000 else combined_content
        
        # Generate embedding for the summary
        embedding = self.embeddings.embed_query(summary)
        
        # Create parent node
        parent_node = RAPTORNode(
            node_id=node_id,
            content=combined_content,
            summary=summary,
            embedding=embedding,
            level=level,
            children=[child.node_id for child in child_nodes],
            parent=None,
            metadata={
                "cluster_size": len(child_nodes),
                "child_levels": list(set(child.level for child in child_nodes))
            }
        )
        
        # Update child nodes to point to parent
        for child in child_nodes:
            child.parent = node_id
            self.nodes[child.node_id] = child
        
        self.nodes[node_id] = parent_node
        return parent_node
    
    def _create_vectorstore_for_level(self, level: int, nodes: List[RAPTORNode]) -> None:
        """Create a vector store for nodes at a specific level"""
        if not nodes:
            return
        
        # Convert nodes to LangChain documents
        langchain_docs = []
        for node in nodes:
            # Filter out complex metadata (lists, dicts) that Chroma can't handle
            filtered_metadata = {}
            for key, value in node.metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    filtered_metadata[key] = value
                else:
                    # Convert complex types to strings
                    filtered_metadata[key] = str(value)
            
            doc = LangChainDocument(
                page_content=node.summary,
                metadata={
                    "node_id": node.node_id,
                    "level": node.level,
                    "parent": node.parent if node.parent else "",
                    "children_count": len(node.children),  # Store count instead of list
                    **filtered_metadata
                }
            )
            langchain_docs.append(doc)
        
        # Create vector store for this level
        vectorstore = Chroma.from_documents(
            documents=langchain_docs,
            embedding=self.embeddings,
            collection_name=f"raptor_level_{level}",
            **self.config.vector_store_config
        )
        
        self.vectorstores[level] = vectorstore
        logger.info(f"Created vector store for level {level} with {len(langchain_docs)} nodes")
    
    def get_document_count(self) -> int:
        """
        Get the number of indexed documents.
        
        Returns:
            int: Number of documents that have been indexed
        """
        return self._document_count
    
    def clear_index(self) -> bool:
        """
        Clear the RAPTOR tree and all vector stores.
        
        Returns:
            bool: True if clearing was successful, False otherwise
        """
        try:
            self.nodes.clear()
            self.root_nodes.clear()
            self.level_nodes.clear()
            self.vectorstores.clear()
            self._document_count = 0
            
            logger.info("RAPTOR index cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing RAPTOR index: {str(e)}")
            return False
    
    def search_at_level(self, query: str, level: int, k: int = 5) -> List[RAPTORNode]:
        """
        Search for relevant nodes at a specific level.
        
        Args:
            query: Search query
            level: Tree level to search
            k: Number of results to return
            
        Returns:
            List[RAPTORNode]: Relevant nodes at the specified level
        """
        if level not in self.vectorstores:
            logger.warning(f"No vector store found for level {level}")
            return []
        
        try:
            # Search the vector store at the specified level
            vectorstore = self.vectorstores[level]
            docs = vectorstore.similarity_search(query, k=k)
            
            # Convert back to RAPTORNode objects
            nodes = []
            for doc in docs:
                node_id = doc.metadata.get("node_id")
                if node_id and node_id in self.nodes:
                    nodes.append(self.nodes[node_id])
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error searching at level {level}: {str(e)}")
            return []
    
    def hierarchical_search(self, query: str, k: int = 5) -> List[RAPTORNode]:
        """
        Perform hierarchical search starting from the top level and drilling down.
        
        Args:
            query: Search query
            k: Number of results to return at each level
            
        Returns:
            List[RAPTORNode]: Relevant nodes from hierarchical search
        """
        if not self.vectorstores:
            logger.warning("No vector stores available for search")
            return []
        
        try:
            # Start from the highest level
            max_level = max(self.vectorstores.keys())
            current_nodes = self.search_at_level(query, max_level, k)
            
            # Drill down through the hierarchy
            for level in range(max_level - 1, -1, -1):
                if level not in self.vectorstores:
                    continue
                
                # Get children of current nodes
                child_candidates = []
                for node in current_nodes:
                    for child_id in node.children:
                        if child_id in self.nodes:
                            child_candidates.append(self.nodes[child_id])
                
                if not child_candidates:
                    break
                
                # Search among child candidates
                child_embeddings = np.array([child.embedding for child in child_candidates])
                query_embedding = np.array(self.embeddings.embed_query(query)).reshape(1, -1)
                
                # Calculate similarities
                similarities = cosine_similarity(query_embedding, child_embeddings)[0]
                
                # Get top k children
                top_indices = np.argsort(similarities)[-k:][::-1]
                current_nodes = [child_candidates[i] for i in top_indices]
            
            return current_nodes
            
        except Exception as e:
            logger.error(f"Error in hierarchical search: {str(e)}")
            return []
    
    def get_tree_info(self) -> Dict[str, Any]:
        """
        Get information about the RAPTOR tree structure.
        
        Returns:
            Dict[str, Any]: Tree information
        """
        return {
            "total_nodes": len(self.nodes),
            "total_levels": len(self.level_nodes),
            "nodes_per_level": {level: len(nodes) for level, nodes in self.level_nodes.items()},
            "root_nodes": len(self.root_nodes),
            "document_count": self._document_count,
            "max_levels": self.max_levels,
            "cluster_size_threshold": self.cluster_size_threshold
        }
    
    def get_node(self, node_id: str) -> Optional[RAPTORNode]:
        """
        Get a specific node by ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Optional[RAPTORNode]: The node if found, None otherwise
        """
        return self.nodes.get(node_id)
    
    def get_node_path(self, node_id: str) -> List[RAPTORNode]:
        """
        Get the path from a node to the root.
        
        Args:
            node_id: ID of the starting node
            
        Returns:
            List[RAPTORNode]: Path from node to root
        """
        path = []
        current_node = self.nodes.get(node_id)
        
        while current_node:
            path.append(current_node)
            if current_node.parent:
                current_node = self.nodes.get(current_node.parent)
            else:
                break
        
        return path