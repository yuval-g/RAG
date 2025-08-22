"""
Multi-representation indexer implementation adapting from workplan/03AdvancedIndexingStrategies.md
Uses Google Gemini for summarization instead of OpenAI as per steering rules
"""

import uuid
import logging
from typing import List, Optional, Dict, Any
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document as LangChainDocument

from ..core.interfaces import BaseIndexer
from ..core.models import Document
from ..core.config import PipelineConfig


logger = logging.getLogger(__name__)


class MultiRepresentationIndexer(BaseIndexer):
    """
    Multi-representation indexer that creates summaries for retrieval while storing full documents for generation.
    Adapts the implementation from workplan/03AdvancedIndexingStrategies.md but uses Google Gemini.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the MultiRepresentationIndexer with configuration.
        
        Args:
            config: Pipeline configuration containing model and embedding settings
        """
        self.config = config
        
        # Initialize Google Gemini embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model if config.embedding_model != "text-embedding-ada-002" 
                  else "models/embedding-001"  # Default Google embedding model
        )
        
        # Initialize Google Gemini LLM for summarization
        self.llm = ChatGoogleGenerativeAI(
            model=config.llm_model if config.llm_model != "gpt-3.5-turbo" 
                  else "gemini-2.0-flash-lite",  # Default Google model
            temperature=config.temperature
        )
        
        # Create summarization chain
        self.summary_chain = (
            {"doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
            | self.llm
            | StrOutputParser()
        )
        
        # Vector store for summary embeddings
        self.vectorstore: Optional[Chroma] = None
        
        # Document store for original documents
        self.docstore = InMemoryByteStore()
        
        # Key to link summaries to parent documents
        self.id_key = "doc_id"
        
        # Multi-vector retriever
        self.retriever: Optional[MultiVectorRetriever] = None
        
        self._document_count = 0
        
        logger.info("MultiRepresentationIndexer initialized with Google Gemini")
    
    def index_documents(self, documents: List[Document]) -> bool:
        """
        Index documents by creating summaries for retrieval and storing full documents for generation.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            bool: True if indexing was successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided for indexing")
                return True
            
            # Convert Document objects to LangChain document format for summarization
            langchain_docs = []
            for doc in documents:
                langchain_docs.append(LangChainDocument(
                    page_content=doc.content,
                    metadata=doc.metadata
                ))
            
            logger.info(f"Generating summaries for {len(langchain_docs)} documents")
            
            # Generate summaries using the chain with batch processing for efficiency
            summaries = self.summary_chain.batch(langchain_docs, {"max_concurrency": 5})
            
            logger.info(f"Generated {len(summaries)} summaries")
            
            # Initialize vector store if not already done
            if self.vectorstore is None:
                self.vectorstore = Chroma(
                    collection_name="summaries", 
                    embedding_function=self.embeddings,
                    **self.config.vector_store_config
                )
                
                # Initialize retriever
                self.retriever = MultiVectorRetriever(
                    vectorstore=self.vectorstore,
                    byte_store=self.docstore,
                    id_key=self.id_key,
                )
            
            # Generate unique IDs for each document
            doc_ids = [str(uuid.uuid4()) for _ in documents]
            
            # Create summary documents with doc_id metadata
            summary_docs = [
                LangChainDocument(
                    page_content=summary, 
                    metadata={self.id_key: doc_ids[i]}
                )
                for i, summary in enumerate(summaries)
            ]
            
            # Add summaries to vector store
            logger.info("Adding summaries to vector store")
            self.retriever.vectorstore.add_documents(summary_docs)
            
            # Add original documents to docstore, linking them by the same IDs
            logger.info("Adding original documents to docstore")
            doc_pairs = list(zip(doc_ids, langchain_docs))
            self.retriever.docstore.mset(doc_pairs)
            
            self._document_count += len(documents)
            logger.info(f"Successfully indexed {len(documents)} documents with multi-representation. "
                       f"Total documents: {self._document_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error indexing documents with multi-representation: {str(e)}")
            return False
    
    def get_document_count(self) -> int:
        """
        Get the number of indexed documents.
        
        Returns:
            int: Number of documents that have been indexed
        """
        return self._document_count
    
    def clear_index(self) -> bool:
        """
        Clear all indexed documents from both vector store and docstore.
        
        Returns:
            bool: True if clearing was successful, False otherwise
        """
        try:
            if self.vectorstore is not None:
                self.vectorstore = None
            
            if self.docstore is not None:
                # Clear the in-memory byte store
                self.docstore = InMemoryByteStore()
            
            self.retriever = None
            self._document_count = 0
            
            logger.info("Multi-representation index cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing multi-representation index: {str(e)}")
            return False
    
    def get_retriever(self) -> Optional[MultiVectorRetriever]:
        """
        Get the multi-vector retriever for use in retrieval operations.
        
        Returns:
            Optional[MultiVectorRetriever]: The retriever instance, or None if not initialized
        """
        return self.retriever
    
    def search_summaries(self, query: str, k: int = 5) -> List[LangChainDocument]:
        """
        Search over summaries only (for debugging/inspection purposes).
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List[LangChainDocument]: Summary documents with metadata
        """
        if self.vectorstore is None:
            logger.warning("Vector store not initialized")
            return []
        
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error searching summaries: {str(e)}")
            return []
    
    def get_full_documents(self, query: str, k: int = 5) -> List[LangChainDocument]:
        """
        Get full documents using the multi-vector retriever.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List[LangChainDocument]: Full documents retrieved via summaries
        """
        if self.retriever is None:
            logger.warning("Retriever not initialized")
            return []
        
        try:
            return self.retriever.get_relevant_documents(query, n_results=k)
        except Exception as e:
            logger.error(f"Error retrieving full documents: {str(e)}")
            return []
    
    def get_summary_count(self) -> int:
        """
        Get the number of summaries in the vector store.
        
        Returns:
            int: Number of summaries stored
        """
        if self.vectorstore is not None:
            try:
                collection = self.vectorstore._collection
                return collection.count()
            except Exception as e:
                logger.warning(f"Could not get summary count: {str(e)}")
                return 0
        return 0
    
    def get_docstore_count(self) -> int:
        """
        Get the number of documents in the docstore.
        
        Returns:
            int: Number of documents in docstore
        """
        try:
            # InMemoryByteStore doesn't have a direct count method
            # We'll use our internal counter
            return self._document_count
        except Exception as e:
            logger.warning(f"Could not get docstore count: {str(e)}")
            return 0