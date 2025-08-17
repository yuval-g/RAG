"""
Basic document indexer implementation adapting from workplan/00BasicRAGSystem.md
Uses Google Gemini embeddings instead of OpenAI as per steering rules
"""

from typing import List, Optional, Dict, Any
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
import bs4

from ..core.interfaces import BaseIndexer
from ..core.models import Document
from ..core.config import PipelineConfig


logger = logging.getLogger(__name__)


class BasicIndexer(BaseIndexer):
    """
    Basic document indexer that chunks documents and stores them in a vector database.
    Adapts the implementation from workplan/00BasicRAGSystem.md but uses Google Gemini embeddings.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the BasicIndexer with configuration.
        
        Args:
            config: Pipeline configuration containing chunking and embedding settings
        """
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        # Initialize Google Gemini embeddings instead of OpenAI
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model if config.embedding_model != "text-embedding-ada-002" 
                  else "models/embedding-001"  # Default Google embedding model
        )
        
        self.vectorstore: Optional[Chroma] = None
        self._document_count = 0
        
        logger.info(f"BasicIndexer initialized with chunk_size={config.chunk_size}, "
                   f"chunk_overlap={config.chunk_overlap}")
    
    def index_documents(self, documents: List[Document]) -> bool:
        """
        Index a list of documents by chunking them and storing in vector database.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            bool: True if indexing was successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided for indexing")
                return True
            
            # Convert Document objects to LangChain document format
            langchain_docs = []
            for doc in documents:
                # Create a simple document-like object that text_splitter can handle
                class SimpleDoc:
                    def __init__(self, page_content: str, metadata: Dict[str, Any]):
                        self.page_content = page_content
                        self.metadata = metadata
                
                langchain_docs.append(SimpleDoc(doc.content, doc.metadata))
            
            # Split documents into chunks using RecursiveCharacterTextSplitter
            logger.info(f"Splitting {len(langchain_docs)} documents into chunks")
            splits = self.text_splitter.split_documents(langchain_docs)
            logger.info(f"Created {len(splits)} chunks from documents")
            
            # Create or update vector store with embeddings
            if self.vectorstore is None:
                logger.info("Creating new Chroma vector store")
                self.vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    **self.config.vector_store_config
                )
            else:
                logger.info("Adding documents to existing vector store")
                self.vectorstore.add_documents(splits)
            
            self._document_count += len(documents)
            logger.info(f"Successfully indexed {len(documents)} documents "
                       f"({len(splits)} chunks). Total documents: {self._document_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
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
        Clear all indexed documents from the vector store.
        
        Returns:
            bool: True if clearing was successful, False otherwise
        """
        try:
            if self.vectorstore is not None:
                # Chroma doesn't have a direct clear method, so we'll recreate it
                self.vectorstore = None
                self._document_count = 0
                logger.info("Index cleared successfully")
                return True
            else:
                logger.info("No index to clear")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing index: {str(e)}")
            return False
    
    def load_web_documents(self, urls: List[str], 
                          bs_kwargs: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Load documents from web URLs using WebBaseLoader.
        Adapts the web loading logic from workplan/00BasicRAGSystem.md
        
        Args:
            urls: List of URLs to load
            bs_kwargs: Optional BeautifulSoup parsing arguments
            
        Returns:
            List[Document]: Loaded documents
        """
        try:
            if bs_kwargs is None:
                # Default parsing configuration from workplan
                bs_kwargs = dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    )
                )
            
            loader = WebBaseLoader(
                web_paths=urls,
                bs_kwargs=bs_kwargs
            )
            
            # Load documents
            langchain_docs = loader.load()
            
            # Convert to our Document format
            documents = []
            for i, doc in enumerate(langchain_docs):
                documents.append(Document(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    doc_id=f"web_doc_{i}"
                ))
            
            logger.info(f"Loaded {len(documents)} documents from {len(urls)} URLs")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading web documents: {str(e)}")
            return []
    
    def get_vectorstore(self) -> Optional[Chroma]:
        """
        Get the underlying vector store for retrieval operations.
        
        Returns:
            Optional[Chroma]: The Chroma vector store instance, or None if not initialized
        """
        return self.vectorstore
    
    def get_chunk_count(self) -> int:
        """
        Get the number of chunks in the vector store.
        
        Returns:
            int: Number of chunks stored
        """
        if self.vectorstore is not None:
            try:
                # Get collection info to determine chunk count
                collection = self.vectorstore._collection
                return collection.count()
            except Exception as e:
                logger.warning(f"Could not get chunk count: {str(e)}")
                return 0
        return 0