#!/usr/bin/env python3
"""
Document Q&A System Example using RAG Engine

This example demonstrates how to build a document-based question-answering system
that can process various document formats and provide accurate answers with source citations.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import mimetypes

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_engine.core.engine import RAGEngine
from rag_engine.core.config import PipelineConfig, IndexingStrategy
from rag_engine.core.models import Document, TestCase


class DocumentQASystem:
    """A document-based Q&A system using RAG Engine"""
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize the Q&A system"""
        self.config = config or PipelineConfig(
            llm_model="gemini-1.5-flash",
            temperature=0.1,  # Lower temperature for factual answers
            chunk_size=1000,
            chunk_overlap=200,
            retrieval_k=5,
            indexing_strategy=IndexingStrategy.BASIC
        )
        self.engine = RAGEngine(self.config)
        self.document_metadata = {}
        
    def load_documents_from_directory(self, directory_path: str, 
                                    file_extensions: List[str] = None) -> bool:
        """Load all documents from a directory"""
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.json']
        
        directory = Path(directory_path)
        if not directory.exists():
            print(f"❌ Directory not found: {directory_path}")
            return False
        
        documents = []
        loaded_files = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                try:
                    doc = self._load_single_document(file_path)
                    if doc:
                        documents.append(doc)
                        loaded_files.append(str(file_path))
                except Exception as e:
                    print(f"⚠️  Error loading {file_path}: {e}")
        
        if documents:
            result = self.engine.add_documents(documents)
            if result:
                print(f"✅ Successfully loaded {len(documents)} documents from {len(loaded_files)} files")
                return True
        
        print("❌ No documents were loaded")
        return False
    
    def load_documents_from_files(self, file_paths: List[str]) -> bool:
        """Load documents from specific files"""
        documents = []
        loaded_files = []
        
        for file_path in file_paths:
            try:
                doc = self._load_single_document(Path(file_path))
                if doc:
                    documents.append(doc)
                    loaded_files.append(file_path)
            except Exception as e:
                print(f"⚠️  Error loading {file_path}: {e}")
        
        if documents:
            result = self.engine.add_documents(documents)
            if result:
                print(f"✅ Successfully loaded {len(documents)} documents from {len(loaded_files)} files")
                return True
        
        print("❌ No documents were loaded")
        return False
    
    def _load_single_document(self, file_path: Path) -> Optional[Document]:
        """Load a single document file"""
        if not file_path.exists():
            return None
        
        # Determine file type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        try:
            if file_path.suffix.lower() == '.json':
                content = self._load_json_document(file_path)
            else:
                # Default to text loading
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            if not content.strip():
                return None
            
            # Create document metadata
            metadata = {
                "filename": file_path.name,
                "filepath": str(file_path),
                "file_size": file_path.stat().st_size,
                "file_type": file_path.suffix.lower(),
                "mime_type": mime_type or "text/plain",
                "directory": str(file_path.parent)
            }
            
            # Store metadata for later reference
            doc_id = f"doc_{file_path.stem}_{hash(str(file_path)) % 10000}"
            self.document_metadata[doc_id] = metadata
            
            return Document(
                content=content,
                metadata=metadata,
                doc_id=doc_id
            )
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    
    def _load_json_document(self, file_path: Path) -> str:
        """Load and format JSON document"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON to readable text format
        if isinstance(data, dict):
            content_parts = []
            for key, value in data.items():
                if isinstance(value, (str, int, float)):
                    content_parts.append(f"{key}: {value}")
                elif isinstance(value, list):
                    content_parts.append(f"{key}: {', '.join(map(str, value))}")
                else:
                    content_parts.append(f"{key}: {json.dumps(value, indent=2)}")
            return "\n".join(content_parts)
        else:
            return json.dumps(data, indent=2)
    
    def ask_question(self, question: str, strategy: str = "basic", 
                    include_sources: bool = True) -> Dict[str, Any]:
        """Ask a question and get an answer with sources"""
        try:
            # Query the RAG engine
            response = self.engine.query(question, strategy=strategy)
            
            result = {
                "question": question,
                "answer": response.answer,
                "confidence": response.confidence_score,
                "processing_time": response.processing_time,
                "strategy_used": strategy,
                "sources": []
            }
            
            if include_sources and response.source_documents:
                for i, doc in enumerate(response.source_documents):
                    source_info = {
                        "rank": i + 1,
                        "filename": doc.metadata.get("filename", "Unknown"),
                        "filepath": doc.metadata.get("filepath", "Unknown"),
                        "file_type": doc.metadata.get("file_type", "Unknown"),
                        "content_preview": doc.content[:300] + "..." if len(doc.content) > 300 else doc.content,
                        "relevance_score": getattr(doc, 'relevance_score', None)
                    }
                    result["sources"].append(source_info)
            
            return result
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "processing_time": 0.0,
                "error": str(e),
                "sources": []
            }
    
    def batch_questions(self, questions: List[str], strategy: str = "basic") -> List[Dict[str, Any]]:
        """Process multiple questions in batch"""
        results = []
        
        print(f"Processing {len(questions)} questions...")
        for i, question in enumerate(questions, 1):
            print(f"Processing question {i}/{len(questions)}: {question[:50]}...")
            result = self.ask_question(question, strategy)
            results.append(result)
        
        return results
    
    def evaluate_system(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Evaluate the Q&A system with test cases"""
        try:
            evaluation_result = self.engine.evaluate(test_cases)
            
            return {
                "overall_score": evaluation_result.overall_score,
                "metric_scores": evaluation_result.metric_scores,
                "test_case_count": len(test_cases),
                "recommendations": evaluation_result.recommendations
            }
        except Exception as e:
            return {
                "error": str(e),
                "overall_score": 0.0,
                "metric_scores": {},
                "test_case_count": len(test_cases),
                "recommendations": []
            }
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents"""
        system_info = self.engine.get_system_info()
        
        # Analyze document metadata
        file_types = {}
        total_size = 0
        
        for doc_id, metadata in self.document_metadata.items():
            file_type = metadata.get("file_type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1
            total_size += metadata.get("file_size", 0)
        
        return {
            "total_documents": system_info["stats"]["indexed_documents"],
            "file_types": file_types,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "system_ready": system_info["stats"]["retriever_ready"]
        }
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for documents similar to the query"""
        try:
            # Use the engine's retrieval capability
            response = self.engine.query(query, k=limit)
            
            results = []
            for doc in response.source_documents:
                results.append({
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "filepath": doc.metadata.get("filepath", "Unknown"),
                    "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "metadata": doc.metadata
                })
            
            return results
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []


def create_sample_documents():
    """Create sample documents for demonstration"""
    sample_dir = Path("sample_docs")
    sample_dir.mkdir(exist_ok=True)
    
    # AI Overview document
    ai_doc = sample_dir / "ai_overview.txt"
    ai_doc.write_text("""
    Artificial Intelligence: A Comprehensive Overview
    
    Artificial Intelligence (AI) represents one of the most significant technological 
    advancements of the 21st century. At its core, AI is the simulation of human 
    intelligence processes by machines, particularly computer systems.
    
    Key Components of AI:
    1. Machine Learning: Algorithms that improve automatically through experience
    2. Natural Language Processing: Understanding and generating human language
    3. Computer Vision: Interpreting and analyzing visual information
    4. Robotics: Physical manifestation of AI in mechanical systems
    
    Applications of AI:
    - Healthcare: Diagnostic imaging, drug discovery, personalized treatment
    - Finance: Fraud detection, algorithmic trading, risk assessment
    - Transportation: Autonomous vehicles, traffic optimization
    - Entertainment: Recommendation systems, content generation
    
    The future of AI holds immense promise, with potential applications in climate 
    change mitigation, space exploration, and solving complex global challenges.
    """)
    
    # Machine Learning document
    ml_doc = sample_dir / "machine_learning.md"
    ml_doc.write_text("""
    # Machine Learning Fundamentals
    
    Machine Learning (ML) is a subset of artificial intelligence that enables 
    computers to learn and make decisions from data without being explicitly programmed.
    
    ## Types of Machine Learning
    
    ### Supervised Learning
    - Uses labeled training data
    - Examples: Classification, Regression
    - Algorithms: Linear Regression, Decision Trees, Neural Networks
    
    ### Unsupervised Learning
    - Finds patterns in unlabeled data
    - Examples: Clustering, Dimensionality Reduction
    - Algorithms: K-Means, PCA, Autoencoders
    
    ### Reinforcement Learning
    - Learns through interaction with environment
    - Uses rewards and penalties
    - Applications: Game playing, Robotics, Autonomous systems
    
    ## Popular ML Libraries
    - **Python**: scikit-learn, TensorFlow, PyTorch
    - **R**: caret, randomForest, e1071
    - **Java**: Weka, Deeplearning4j
    
    ## Best Practices
    1. Data preprocessing and cleaning
    2. Feature engineering and selection
    3. Model validation and testing
    4. Hyperparameter tuning
    5. Regular model monitoring and updates
    """)
    
    # Programming guide
    prog_doc = sample_dir / "programming_guide.txt"
    prog_doc.write_text("""
    Programming for AI and Machine Learning
    
    Getting started with AI programming requires understanding both the theoretical 
    concepts and practical implementation skills.
    
    Essential Programming Languages:
    
    Python:
    - Most popular for AI/ML development
    - Rich ecosystem of libraries
    - Easy to learn and use
    - Great for prototyping
    
    R:
    - Excellent for statistical analysis
    - Strong data visualization capabilities
    - Preferred in academic research
    
    Java:
    - Good for enterprise applications
    - Strong performance
    - Platform independent
    
    C++:
    - High performance applications
    - Real-time systems
    - Embedded AI systems
    
    Key Skills to Develop:
    1. Statistics and Mathematics
    2. Data manipulation and analysis
    3. Algorithm implementation
    4. Model evaluation and validation
    5. Software engineering best practices
    
    Learning Path:
    1. Start with Python basics
    2. Learn pandas and numpy for data manipulation
    3. Practice with scikit-learn for ML algorithms
    4. Explore deep learning with TensorFlow or PyTorch
    5. Work on real-world projects
    """)
    
    # Configuration file
    config_doc = sample_dir / "config.json"
    config_doc.write_text(json.dumps({
        "system_name": "AI Documentation System",
        "version": "1.0.0",
        "supported_formats": ["txt", "md", "json", "pdf"],
        "max_file_size": "10MB",
        "indexing_strategy": "basic",
        "retrieval_settings": {
            "k": 5,
            "chunk_size": 1000,
            "overlap": 200
        },
        "llm_settings": {
            "model": "gemini-1.5-flash",
            "temperature": 0.1,
            "max_tokens": 500
        }
    }, indent=2))
    
    print(f"✅ Created sample documents in {sample_dir}")
    return str(sample_dir)


def run_interactive_qa():
    """Run interactive Q&A session"""
    print("📚 Document Q&A System")
    print("=" * 50)
    
    # Initialize system
    qa_system = DocumentQASystem()
    
    # Check if sample documents exist, create if not
    sample_dir = Path("sample_docs")
    if not sample_dir.exists():
        print("Creating sample documents...")
        create_sample_documents()
    
    # Load documents
    print("Loading documents...")
    if not qa_system.load_documents_from_directory(str(sample_dir)):
        print("❌ Failed to load documents. Exiting.")
        return
    
    # Show document stats
    stats = qa_system.get_document_stats()
    print(f"\n📊 Document Statistics:")
    print(f"Total documents: {stats['total_documents']}")
    print(f"File types: {stats['file_types']}")
    print(f"Total size: {stats['total_size_mb']} MB")
    
    print("\n🎯 You can now ask questions about the loaded documents.")
    print("Available strategies: basic, multi_query, rag_fusion")
    print("Commands: 'quit' to exit, 'stats' for statistics, 'search <query>' to search documents")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n❓ Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                stats = qa_system.get_document_stats()
                print(f"\n📊 Current Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            if user_input.lower().startswith('search '):
                query = user_input[7:]  # Remove 'search ' prefix
                results = qa_system.search_documents(query, limit=5)
                print(f"\n🔍 Search results for '{query}':")
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['filename']}")
                    print(f"   {result['content_preview']}")
                continue
            
            if not user_input:
                continue
            
            # Ask for strategy
            strategy = input("Strategy (basic/multi_query/rag_fusion) [basic]: ").strip() or "basic"
            
            # Process question
            print("🤔 Processing...")
            result = qa_system.ask_question(user_input, strategy=strategy)
            
            # Display answer
            print(f"\n💡 Answer: {result['answer']}")
            print(f"📊 Confidence: {result['confidence']:.2f}")
            print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
            print(f"🔧 Strategy: {result['strategy_used']}")
            
            # Show sources
            if result['sources']:
                print(f"\n📚 Sources ({len(result['sources'])}):")
                for source in result['sources']:
                    print(f"  {source['rank']}. {source['filename']} ({source['file_type']})")
                    print(f"     Preview: {source['content_preview'][:100]}...")
            
            if 'error' in result:
                print(f"⚠️  Error: {result['error']}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def run_batch_demo():
    """Run batch processing demo"""
    print("📚 Document Q&A System - Batch Demo")
    print("=" * 50)
    
    # Initialize system
    qa_system = DocumentQASystem()
    
    # Create and load sample documents
    sample_dir = create_sample_documents()
    if not qa_system.load_documents_from_directory(sample_dir):
        print("❌ Failed to load documents. Exiting.")
        return
    
    # Predefined questions
    questions = [
        "What is artificial intelligence?",
        "What are the main types of machine learning?",
        "Which programming languages are best for AI?",
        "What are some applications of AI in healthcare?",
        "How do I get started with machine learning?",
        "What is the difference between supervised and unsupervised learning?",
        "What libraries are available for Python ML development?"
    ]
    
    print(f"\n🎯 Processing {len(questions)} questions...")
    
    # Process questions
    results = qa_system.batch_questions(questions, strategy="multi_query")
    
    # Display results
    print("\n📋 Results Summary:")
    print("-" * 50)
    
    total_confidence = 0
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Q: {result['question']}")
        print(f"   A: {result['answer'][:150]}...")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Sources: {len(result['sources'])}")
        total_confidence += result['confidence']
    
    avg_confidence = total_confidence / len(results)
    print(f"\n📊 Average confidence: {avg_confidence:.2f}")
    
    # Show document stats
    stats = qa_system.get_document_stats()
    print(f"📚 Documents processed: {stats['total_documents']}")


def run_evaluation_demo():
    """Run evaluation demo with test cases"""
    print("📚 Document Q&A System - Evaluation Demo")
    print("=" * 50)
    
    # Initialize system
    qa_system = DocumentQASystem()
    
    # Create and load sample documents
    sample_dir = create_sample_documents()
    if not qa_system.load_documents_from_directory(sample_dir):
        print("❌ Failed to load documents. Exiting.")
        return
    
    # Create test cases
    test_cases = [
        TestCase(
            question="What is artificial intelligence?",
            expected_answer="AI is the simulation of human intelligence processes by machines",
            metadata={"category": "definition", "difficulty": "basic"}
        ),
        TestCase(
            question="What are the types of machine learning?",
            expected_answer="Supervised learning, unsupervised learning, and reinforcement learning",
            metadata={"category": "classification", "difficulty": "intermediate"}
        ),
        TestCase(
            question="Which programming language is most popular for AI?",
            expected_answer="Python is the most popular programming language for AI development",
            metadata={"category": "tools", "difficulty": "basic"}
        )
    ]
    
    print(f"🧪 Evaluating system with {len(test_cases)} test cases...")
    
    # Run evaluation
    evaluation_result = qa_system.evaluate_system(test_cases)
    
    # Display results
    print("\n📊 Evaluation Results:")
    print(f"Overall Score: {evaluation_result['overall_score']:.2f}")
    
    if evaluation_result['metric_scores']:
        print("Metric Scores:")
        for metric, score in evaluation_result['metric_scores'].items():
            print(f"  {metric}: {score:.2f}")
    
    if evaluation_result['recommendations']:
        print("Recommendations:")
        for rec in evaluation_result['recommendations']:
            print(f"  - {rec}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Q&A System Example")
    parser.add_argument("--mode", choices=["interactive", "batch", "evaluation"], 
                       default="interactive", help="Run mode")
    parser.add_argument("--directory", help="Directory containing documents to load")
    parser.add_argument("--files", nargs="*", help="Specific files to load")
    parser.add_argument("--strategy", default="basic", 
                       choices=["basic", "multi_query", "rag_fusion"],
                       help="Query processing strategy")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "batch":
            run_batch_demo()
        elif args.mode == "evaluation":
            run_evaluation_demo()
        else:
            if args.directory or args.files:
                # Custom document loading
                qa_system = DocumentQASystem()
                
                if args.directory:
                    success = qa_system.load_documents_from_directory(args.directory)
                elif args.files:
                    success = qa_system.load_documents_from_files(args.files)
                
                if success:
                    run_interactive_qa()
                else:
                    print("❌ Failed to load documents.")
            else:
                run_interactive_qa()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)