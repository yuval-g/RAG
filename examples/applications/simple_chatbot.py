#!/usr/bin/env python3
"""
Simple Chatbot Example using RAG Engine

This example demonstrates how to build a simple chatbot interface
using the RAG Engine for knowledge-based question answering.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_engine.core.engine import RAGEngine
from rag_engine.core.config import PipelineConfig
from rag_engine.core.models import Document


class SimpleChatbot:
    """A simple chatbot using RAG Engine for knowledge-based responses"""
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize the chatbot with RAG engine"""
        self.config = config or PipelineConfig(
            llm_model="gemini-1.5-flash",
            temperature=0.7,
            chunk_size=800,
            retrieval_k=3
        )
        self.engine = RAGEngine(self.config)
        self.conversation_history = []
        
    def load_knowledge_base(self, documents: List[Document]) -> bool:
        """Load documents into the knowledge base"""
        try:
            result = self.engine.add_documents(documents)
            print(f"✅ Loaded {len(documents)} documents into knowledge base")
            return result
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            return False
    
    def load_from_files(self, file_paths: List[str]) -> bool:
        """Load documents from text files"""
        documents = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                doc = Document(
                    content=content,
                    metadata={
                        "source": file_path,
                        "filename": Path(file_path).name,
                        "type": "text_file"
                    },
                    doc_id=f"file_{Path(file_path).stem}"
                )
                documents.append(doc)
                
            except Exception as e:
                print(f"⚠️  Error reading {file_path}: {e}")
        
        if documents:
            return self.load_knowledge_base(documents)
        return False
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """Process user input and return chatbot response"""
        if not user_input.strip():
            return {
                "response": "Please ask me a question!",
                "confidence": 0.0,
                "sources": []
            }
        
        try:
            # Query the RAG engine
            rag_response = self.engine.query(user_input)
            
            # Add to conversation history
            self.conversation_history.append({
                "user": user_input,
                "bot": rag_response.answer,
                "confidence": rag_response.confidence_score,
                "timestamp": rag_response.metadata.get("timestamp")
            })
            
            # Format sources
            sources = []
            for doc in rag_response.source_documents:
                sources.append({
                    "title": doc.metadata.get("filename", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                })
            
            return {
                "response": rag_response.answer,
                "confidence": rag_response.confidence_score,
                "sources": sources,
                "processing_time": rag_response.processing_time
            }
            
        except Exception as e:
            return {
                "response": f"Sorry, I encountered an error: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "error": str(e)
            }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return self.engine.get_system_info()


def create_sample_knowledge_base() -> List[Document]:
    """Create a sample knowledge base about AI and technology"""
    documents = [
        Document(
            content="""
            Artificial Intelligence (AI) is a branch of computer science that aims to create 
            intelligent machines capable of performing tasks that typically require human intelligence. 
            These tasks include visual perception, speech recognition, decision-making, and language 
            translation. AI can be categorized into narrow AI (designed for specific tasks) and 
            general AI (human-like intelligence across all domains).
            """,
            metadata={"title": "What is AI", "category": "AI Basics", "difficulty": "beginner"},
            doc_id="ai_intro"
        ),
        Document(
            content="""
            Machine Learning (ML) is a subset of AI that enables computers to learn and improve 
            from experience without being explicitly programmed. ML algorithms build mathematical 
            models based on training data to make predictions or decisions. Common types include 
            supervised learning (with labeled data), unsupervised learning (finding patterns in 
            unlabeled data), and reinforcement learning (learning through interaction with an environment).
            """,
            metadata={"title": "Machine Learning Basics", "category": "ML", "difficulty": "intermediate"},
            doc_id="ml_intro"
        ),
        Document(
            content="""
            Natural Language Processing (NLP) is a field of AI that focuses on the interaction 
            between computers and human language. NLP enables machines to understand, interpret, 
            and generate human language in a valuable way. Applications include sentiment analysis, 
            machine translation, chatbots, text summarization, and question-answering systems. 
            Modern NLP relies heavily on deep learning and transformer architectures.
            """,
            metadata={"title": "NLP Overview", "category": "NLP", "difficulty": "intermediate"},
            doc_id="nlp_intro"
        ),
        Document(
            content="""
            Deep Learning is a subset of machine learning that uses artificial neural networks 
            with multiple layers (hence 'deep') to model and understand complex patterns in data. 
            Deep learning has revolutionized fields like computer vision, natural language processing, 
            and speech recognition. Popular architectures include Convolutional Neural Networks (CNNs) 
            for image processing and Recurrent Neural Networks (RNNs) for sequential data.
            """,
            metadata={"title": "Deep Learning Explained", "category": "Deep Learning", "difficulty": "advanced"},
            doc_id="dl_intro"
        ),
        Document(
            content="""
            Python is the most popular programming language for AI and machine learning development. 
            It offers excellent libraries like TensorFlow, PyTorch, scikit-learn, pandas, and numpy. 
            Python's simplicity and readability make it ideal for rapid prototyping and experimentation. 
            Other languages used in AI include R (for statistics), Java (for enterprise applications), 
            and C++ (for performance-critical applications).
            """,
            metadata={"title": "Programming for AI", "category": "Programming", "difficulty": "beginner"},
            doc_id="programming_ai"
        )
    ]
    return documents


def run_interactive_chat():
    """Run an interactive chat session"""
    print("🤖 RAG Engine Chatbot")
    print("=" * 50)
    print("Loading knowledge base...")
    
    # Initialize chatbot
    chatbot = SimpleChatbot()
    
    # Load sample knowledge base
    sample_docs = create_sample_knowledge_base()
    if not chatbot.load_knowledge_base(sample_docs):
        print("❌ Failed to load knowledge base. Exiting.")
        return
    
    print("✅ Knowledge base loaded successfully!")
    print("\nYou can now ask questions about AI, ML, NLP, and programming.")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Type 'history' to see conversation history.")
    print("Type 'info' to see system information.")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\n🧑 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Thanks for chatting!")
                break
            
            # Check for special commands
            if user_input.lower() == 'history':
                history = chatbot.get_conversation_history()
                if history:
                    print("\n📜 Conversation History:")
                    for i, entry in enumerate(history, 1):
                        print(f"{i}. You: {entry['user']}")
                        print(f"   Bot: {entry['bot'][:100]}...")
                        print(f"   Confidence: {entry['confidence']:.2f}")
                else:
                    print("No conversation history yet.")
                continue
            
            if user_input.lower() == 'info':
                info = chatbot.get_system_info()
                print(f"\n📊 System Info:")
                print(f"Version: {info['version']}")
                print(f"Documents: {info['stats']['indexed_documents']}")
                print(f"Ready: {info['stats']['retriever_ready']}")
                continue
            
            if not user_input:
                continue
            
            # Process the query
            print("🤔 Thinking...")
            response = chatbot.chat(user_input)
            
            # Display response
            print(f"\n🤖 Bot: {response['response']}")
            
            if response['confidence'] > 0:
                print(f"📊 Confidence: {response['confidence']:.2f}")
                print(f"⏱️  Processing time: {response.get('processing_time', 0):.2f}s")
                
                # Show sources if available
                if response['sources']:
                    print("\n📚 Sources:")
                    for i, source in enumerate(response['sources'], 1):
                        print(f"  {i}. {source['title']} ({source['source']})")
            
            if 'error' in response:
                print(f"⚠️  Error: {response['error']}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")


def run_batch_demo():
    """Run a batch demo with predefined questions"""
    print("🤖 RAG Engine Chatbot - Batch Demo")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = SimpleChatbot()
    
    # Load sample knowledge base
    sample_docs = create_sample_knowledge_base()
    if not chatbot.load_knowledge_base(sample_docs):
        print("❌ Failed to load knowledge base. Exiting.")
        return
    
    # Predefined questions
    questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What programming languages are used for AI?",
        "What is the difference between AI and ML?",
        "What are some applications of NLP?",
        "What is deep learning?",
        "How do I get started with AI programming?"
    ]
    
    print("🎯 Running batch demo with sample questions...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        response = chatbot.chat(question)
        
        print(f"Answer: {response['response'][:200]}...")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Sources: {len(response['sources'])}")
        print("-" * 50)
    
    # Show conversation history
    history = chatbot.get_conversation_history()
    print(f"\n📊 Demo completed! Processed {len(history)} questions.")
    
    avg_confidence = sum(entry['confidence'] for entry in history) / len(history)
    print(f"Average confidence: {avg_confidence:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Engine Chatbot Example")
    parser.add_argument("--mode", choices=["interactive", "batch"], default="interactive",
                       help="Run mode: interactive chat or batch demo")
    parser.add_argument("--files", nargs="*", help="Text files to load as knowledge base")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "batch":
            run_batch_demo()
        else:
            if args.files:
                print(f"Loading knowledge base from files: {args.files}")
                chatbot = SimpleChatbot()
                if chatbot.load_from_files(args.files):
                    # Continue with interactive chat using loaded files
                    run_interactive_chat()
                else:
                    print("❌ Failed to load files. Using sample knowledge base.")
                    run_interactive_chat()
            else:
                run_interactive_chat()
    
    except Exception as e:
        print(f"❌ Error running chatbot: {e}")
        sys.exit(1)