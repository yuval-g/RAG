"""
Example demonstrating self-correction mechanisms (CRAG and Self-RAG)
Based on workplan/04AdvancedRetrieval-Generation.md
"""

import os
import sys
from typing import List

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_engine.core.config import PipelineConfig
from rag_engine.core.models import Document
from rag_engine.retrieval.self_correction import (
    SelfCorrectionEngine,
    CRAGRelevanceChecker,
    SelfRAGValidator,
    RelevanceGrade,
    FactualityGrade
)


def create_sample_documents() -> List[Document]:
    """Create sample documents for testing self-correction"""
    return [
        Document(
            content="""
            Machine learning is a subset of artificial intelligence (AI) that focuses on the development 
            of algorithms and statistical models that enable computer systems to improve their performance 
            on a specific task through experience, without being explicitly programmed. It involves training 
            algorithms on data to make predictions or decisions.
            """.strip(),
            metadata={"source": "ml_textbook.pdf", "topic": "machine_learning"}
        ),
        Document(
            content="""
            The weather forecast for tomorrow shows sunny skies with a high temperature of 75°F and 
            low humidity. There's a 10% chance of rain in the evening. Wind speeds will be light 
            at 5-10 mph from the southwest.
            """.strip(),
            metadata={"source": "weather_report.txt", "topic": "weather"}
        ),
        Document(
            content="""
            Deep learning is a subset of machine learning that uses artificial neural networks with 
            multiple layers (hence "deep") to model and understand complex patterns in data. It has 
            been particularly successful in areas like image recognition, natural language processing, 
            and speech recognition.
            """.strip(),
            metadata={"source": "deep_learning_guide.pdf", "topic": "deep_learning"}
        ),
        Document(
            content="""
            Python is a high-level, interpreted programming language known for its simplicity and 
            readability. It was created by Guido van Rossum and first released in 1991. Python 
            supports multiple programming paradigms and has a large standard library.
            """.strip(),
            metadata={"source": "python_manual.pdf", "topic": "programming"}
        ),
        Document(
            content="""
            The stock market experienced significant volatility today with the Dow Jones dropping 
            2.3% and the S&P 500 falling 1.8%. Technology stocks were particularly affected due to 
            concerns about rising interest rates and inflation.
            """.strip(),
            metadata={"source": "financial_news.txt", "topic": "finance"}
        )
    ]


def demonstrate_crag_relevance_checking():
    """Demonstrate CRAG-style relevance checking"""
    print("=" * 60)
    print("CRAG RELEVANCE CHECKING DEMONSTRATION")
    print("=" * 60)
    
    # Create configuration
    config = PipelineConfig(
        llm_model="gemini-2.0-flash-lite",
        temperature=0.0,
        enable_self_correction=True,
        relevance_threshold=0.7,
        min_relevant_docs=2
    )
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Initialize CRAG relevance checker
    print("Initializing CRAG relevance checker...")
    try:
        relevance_checker = CRAGRelevanceChecker(config)
        print("✓ CRAG relevance checker initialized successfully")
    except Exception as e:
        print(f"⚠ CRAG relevance checker initialization failed (expected without credentials): {type(e).__name__}")
        print("  Proceeding with simulated functionality for demonstration...")
    
    # Test query about machine learning
    query = "What is machine learning and how does it work?"
    print(f"\nQuery: {query}")
    print("\nAssessing document relevance...")
    
    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1} ({doc.metadata.get('topic', 'unknown')}):")
        print(f"Content preview: {doc.content[:100]}...")
        
        try:
            # Assess relevance (this would normally call the LLM)
            # For demo purposes, we'll simulate the assessment
            if doc.metadata.get('topic') in ['machine_learning', 'deep_learning']:
                grade = RelevanceGrade.RELEVANT
                confidence = 0.9 if doc.metadata.get('topic') == 'machine_learning' else 0.85
                reasoning = f"Document is about {doc.metadata.get('topic')} which is directly relevant to the query"
            elif doc.metadata.get('topic') == 'programming':
                grade = RelevanceGrade.AMBIGUOUS
                confidence = 0.6
                reasoning = "Document is about programming which is somewhat related but not directly answering the question"
            else:
                grade = RelevanceGrade.IRRELEVANT
                confidence = 0.8
                reasoning = f"Document is about {doc.metadata.get('topic')} which is not relevant to machine learning"
            
            print(f"  Grade: {grade.value}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Reasoning: {reasoning}")
            
        except Exception as e:
            print(f"  Error assessing relevance: {e}")
    
    # Filter relevant documents
    print(f"\nFiltering documents with relevance threshold: {config.relevance_threshold}")
    try:
        # Simulate filtering (normally done by the relevance checker)
        relevant_docs = [doc for doc in documents 
                        if doc.metadata.get('topic') in ['machine_learning', 'deep_learning']]
        
        print(f"✓ Filtered {len(relevant_docs)} relevant documents from {len(documents)} total")
        
        for i, doc in enumerate(relevant_docs):
            print(f"  Relevant Doc {i+1}: {doc.metadata.get('topic')}")
        
        # Check if we have enough relevant documents
        if len(relevant_docs) >= config.min_relevant_docs:
            print(f"✓ Sufficient relevant documents found (>= {config.min_relevant_docs})")
        else:
            print(f"⚠ Insufficient relevant documents, fallback strategy would be triggered")
            
    except Exception as e:
        print(f"✗ Error filtering documents: {e}")


def demonstrate_self_rag_validation():
    """Demonstrate Self-RAG response validation"""
    print("\n" + "=" * 60)
    print("SELF-RAG RESPONSE VALIDATION DEMONSTRATION")
    print("=" * 60)
    
    # Create configuration
    config = PipelineConfig(
        llm_model="gemini-2.0-flash-lite",
        temperature=0.0,
        enable_self_correction=True,
        factuality_threshold=0.7
    )
    
    # Create context documents (relevant to machine learning)
    context_docs = [
        Document(
            content="Machine learning is a subset of AI that uses algorithms to learn from data.",
            metadata={"source": "ml_guide.pdf"}
        ),
        Document(
            content="Deep learning uses neural networks with multiple layers to process complex data.",
            metadata={"source": "dl_guide.pdf"}
        )
    ]
    
    # Initialize Self-RAG validator
    print("Initializing Self-RAG validator...")
    try:
        validator = SelfRAGValidator(config)
        print("✓ Self-RAG validator initialized successfully")
    except Exception as e:
        print(f"⚠ Self-RAG validator initialization failed (expected without credentials): {type(e).__name__}")
        print("  Proceeding with simulated functionality for demonstration...")
    
    # Test different types of responses
    test_cases = [
        {
            "query": "What is machine learning?",
            "response": "Machine learning is a subset of AI that uses algorithms to learn from data and make predictions.",
            "expected_grade": FactualityGrade.GROUNDED,
            "description": "Well-grounded response"
        },
        {
            "query": "What is machine learning?",
            "response": "Machine learning was invented by aliens in 1950 and uses quantum computers to predict the future.",
            "expected_grade": FactualityGrade.NOT_GROUNDED,
            "description": "Not grounded response with false claims"
        },
        {
            "query": "What is machine learning?",
            "response": "Machine learning is related to AI and involves some form of data processing, though the exact mechanisms are complex.",
            "expected_grade": FactualityGrade.PARTIALLY_GROUNDED,
            "description": "Partially grounded response"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print(f"Response: {test_case['response']}")
        
        try:
            # Simulate validation (normally done by the validator)
            # For demo purposes, we'll use the expected grades
            grade = test_case['expected_grade']
            
            if grade == FactualityGrade.GROUNDED:
                confidence = 0.9
                reasoning = "Response is fully supported by the provided context"
                citations_found = True
            elif grade == FactualityGrade.NOT_GROUNDED:
                confidence = 0.85
                reasoning = "Response contains claims not supported by the context"
                citations_found = False
            else:  # PARTIALLY_GROUNDED
                confidence = 0.6
                reasoning = "Response is somewhat supported but lacks specific details from context"
                citations_found = False
            
            print(f"  Validation Result:")
            print(f"    Grade: {grade.value}")
            print(f"    Confidence: {confidence:.2f}")
            print(f"    Citations Found: {citations_found}")
            print(f"    Reasoning: {reasoning}")
            
            # Determine if correction is needed
            needs_correction = (
                grade == FactualityGrade.NOT_GROUNDED or
                (grade == FactualityGrade.PARTIALLY_GROUNDED and confidence < config.factuality_threshold)
            )
            
            if needs_correction:
                print(f"  ⚠ Correction needed (confidence {confidence:.2f} < threshold {config.factuality_threshold})")
                
                # Simulate correction
                if grade == FactualityGrade.NOT_GROUNDED:
                    corrected_response = (
                        "Based on the provided context, I cannot fully verify all claims in the original response. "
                        f"Here's what I can confirm from the available information:\n\n{test_case['response']}\n\n"
                        "Please note that some information may not be fully supported by the provided sources."
                    )
                else:
                    corrected_response = (
                        f"{test_case['response']}\n\n"
                        "Note: This response is based on the provided context. "
                        "Some details may require additional verification."
                    )
                
                print(f"  Corrected Response: {corrected_response[:100]}...")
            else:
                print(f"  ✓ No correction needed")
                
        except Exception as e:
            print(f"  ✗ Error validating response: {e}")


def demonstrate_full_self_correction_pipeline():
    """Demonstrate the full self-correction pipeline"""
    print("\n" + "=" * 60)
    print("FULL SELF-CORRECTION PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Create configuration
    config = PipelineConfig(
        llm_model="gemini-2.0-flash-lite",
        temperature=0.0,
        enable_self_correction=True,
        relevance_threshold=0.7,
        factuality_threshold=0.7,
        min_relevant_docs=2
    )
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Initialize self-correction engine
    print("Initializing self-correction engine...")
    try:
        correction_engine = SelfCorrectionEngine(config)
        print("✓ Self-correction engine initialized successfully")
    except Exception as e:
        print(f"⚠ Self-correction engine initialization failed (expected without credentials): {type(e).__name__}")
        print("  Proceeding with simulated functionality for demonstration...")
    
    # Test query and response
    query = "What is machine learning and how does it relate to AI?"
    original_response = "Machine learning is a subset of artificial intelligence that uses algorithms to learn from data."
    
    print(f"\nOriginal Query: {query}")
    print(f"Original Documents: {len(documents)} documents")
    print(f"Original Response: {original_response}")
    
    # Apply full self-correction pipeline
    print("\nApplying self-correction pipeline...")
    try:
        # Simulate the full pipeline
        # 1. Retrieval correction (CRAG)
        relevant_docs = [doc for doc in documents 
                        if doc.metadata.get('topic') in ['machine_learning', 'deep_learning']]
        
        retrieval_metadata = {
            "original_count": len(documents),
            "filtered_count": len(relevant_docs),
            "correction_applied": len(relevant_docs) < len(documents),
            "fallback_triggered": len(relevant_docs) < config.min_relevant_docs
        }
        
        print(f"  Retrieval Correction:")
        print(f"    Original documents: {retrieval_metadata['original_count']}")
        print(f"    Filtered documents: {retrieval_metadata['filtered_count']}")
        print(f"    Correction applied: {retrieval_metadata['correction_applied']}")
        print(f"    Fallback triggered: {retrieval_metadata['fallback_triggered']}")
        
        # 2. Response validation (Self-RAG)
        # Simulate validation - this response should be grounded
        validation_metadata = {
            "grade": "grounded",
            "confidence": 0.9,
            "reasoning": "Response is well supported by the context",
            "citations_found": True,
            "correction_needed": False
        }
        
        print(f"  Response Validation:")
        print(f"    Grade: {validation_metadata['grade']}")
        print(f"    Confidence: {validation_metadata['confidence']}")
        print(f"    Citations found: {validation_metadata['citations_found']}")
        print(f"    Correction needed: {validation_metadata['correction_needed']}")
        
        # Final results
        final_docs = relevant_docs if not retrieval_metadata['fallback_triggered'] else documents
        final_response = original_response  # No correction needed in this case
        
        total_corrections = (
            int(retrieval_metadata['correction_applied']) +
            int(validation_metadata['correction_needed'])
        )
        
        print(f"\nFinal Results:")
        print(f"  Final documents: {len(final_docs)}")
        print(f"  Final response: {final_response}")
        print(f"  Total corrections applied: {total_corrections}")
        
        # Show correction statistics
        stats = {
            "relevance_threshold": config.relevance_threshold,
            "factuality_threshold": config.factuality_threshold,
            "min_relevant_docs": config.min_relevant_docs
        }
        
        print(f"\nCorrection Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("✓ Self-correction pipeline completed successfully")
        
    except Exception as e:
        print(f"✗ Error in self-correction pipeline: {e}")


def main():
    """Main function to run all demonstrations"""
    print("SELF-CORRECTION MECHANISMS DEMONSTRATION")
    print("Based on CRAG and Self-RAG approaches from workplan/04AdvancedRetrieval-Generation.md")
    print()
    
    try:
        # Demonstrate CRAG relevance checking
        demonstrate_crag_relevance_checking()
        
        # Demonstrate Self-RAG validation
        demonstrate_self_rag_validation()
        
        # Demonstrate full pipeline
        demonstrate_full_self_correction_pipeline()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print()
        print("Key Features Demonstrated:")
        print("• CRAG-style relevance checking for retrieved documents")
        print("• Self-RAG validation for generated responses")
        print("• Fallback strategies for low-confidence results")
        print("• Configurable thresholds for correction decisions")
        print("• Integration with existing RAG pipeline components")
        print()
        print("Note: This demonstration uses simulated LLM responses.")
        print("In a real implementation, actual LLM calls would be made.")
        
    except Exception as e:
        print(f"\n✗ Demonstration failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())