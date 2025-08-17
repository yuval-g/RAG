"""
Example demonstrating grounded generation features in the RAG system.
Shows how to use citation generation, response validation, and full grounding.
"""

import os
import sys
from typing import List
from unittest.mock import patch

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_engine.generation.generation_engine import GenerationEngine
from rag_engine.core.models import Document
from rag_engine.core.config import PipelineConfig


def create_sample_documents() -> List[Document]:
    """Create sample documents for demonstration"""
    return [
        Document(
            content="Python is a high-level, interpreted programming language created by Guido van Rossum. It was first released in 1991 and emphasizes code readability with its notable use of significant whitespace.",
            metadata={
                "source": "python_docs",
                "title": "Python Programming Language",
                "url": "https://docs.python.org/3/",
                "doc_id": "doc1"
            },
            doc_id="doc1"
        ),
        Document(
            content="Python supports multiple programming paradigms, including structured, object-oriented, and functional programming. It has a comprehensive standard library and is often described as a 'batteries included' language.",
            metadata={
                "source": "python_guide",
                "title": "Python Features",
                "url": "https://python-guide.org/",
                "doc_id": "doc2"
            },
            doc_id="doc2"
        ),
        Document(
            content="Python is widely used in data science, machine learning, web development, and automation. Popular libraries include NumPy, Pandas, TensorFlow, Django, and Flask.",
            metadata={
                "source": "python_applications",
                "title": "Python Use Cases",
                "url": "https://python.org/applications/",
                "doc_id": "doc3"
            },
            doc_id="doc3"
        )
    ]


def demonstrate_basic_generation():
    """Demonstrate basic generation vs grounded generation"""
    print("=" * 60)
    print("BASIC vs GROUNDED GENERATION COMPARISON")
    print("=" * 60)
    
    config = PipelineConfig(
        llm_provider="google",
        llm_model="gemini-2.0-flash-lite",
        temperature=0.0
    )
    
    try:
        # Mock the LLM to avoid credential issues
        with patch('rag_engine.generation.generation_engine.ChatGoogleGenerativeAI'):
            engine = GenerationEngine(config)
            documents = create_sample_documents()
            query = "What is Python and what is it used for?"
            
            print(f"Query: {query}")
            print("\nSample Documents:")
            for i, doc in enumerate(documents, 1):
                print(f"[{i}] {doc.content[:100]}...")
            
            print("\n" + "-" * 40)
            print("BASIC GENERATION:")
            print("-" * 40)
            print("Uses standard RAG prompt without strict grounding requirements")
            print("May include information not strictly from the context")
            
            print("\n" + "-" * 40)
            print("GROUNDED GENERATION:")
            print("-" * 40)
            print("Uses enhanced prompt that strictly enforces grounding in provided context")
            print("Includes explicit instructions to only use provided information")
            
            # Show the difference in prompts
            basic_prompt = engine.prompt_template.format(
                context=engine.format_docs(documents),
                question=query
            )
            
            grounded_prompt = engine.grounded_prompt_template.format(
                context=engine.format_docs(documents),
                question=query
            )
            
            print(f"\nBasic prompt length: {len(basic_prompt)} characters")
            print(f"Grounded prompt length: {len(grounded_prompt)} characters")
            print("Grounded prompt includes explicit grounding instructions")
            
            # Show key differences
            print("\nKEY DIFFERENCES:")
            print("-" * 40)
            if "Use ONLY the following pieces" in grounded_prompt:
                print("✓ Grounded prompt includes 'Use ONLY' instruction")
            if "Base your answer ONLY on" in grounded_prompt:
                print("✓ Grounded prompt emphasizes context-only responses")
            if "don't have enough information" in grounded_prompt:
                print("✓ Grounded prompt includes fallback for insufficient context")
        
    except Exception as e:
        print(f"Error in basic generation demonstration: {e}")


def demonstrate_citation_generation():
    """Demonstrate citation generation features"""
    print("\n" + "=" * 60)
    print("CITATION GENERATION DEMONSTRATION")
    print("=" * 60)
    
    config = PipelineConfig()
    
    try:
        with patch('rag_engine.generation.generation_engine.ChatGoogleGenerativeAI'):
            engine = GenerationEngine(config)
            documents = create_sample_documents()
            
            # Show how numbered context is created
            numbered_context, source_metadata = engine._create_numbered_context(documents)
            
            print("NUMBERED CONTEXT FOR CITATIONS:")
            print("-" * 40)
            print(numbered_context[:500] + "...")
            
            print("\nSOURCE METADATA:")
            print("-" * 40)
            for metadata in source_metadata:
                print(f"Citation [{metadata['citation_number']}]: {metadata['title']}")
                print(f"  Source: {metadata['source']}")
                print(f"  URL: {metadata.get('url', 'N/A')}")
                print()
            
            print("CITATION PROMPT FEATURES:")
            print("-" * 40)
            print("• Includes numbered sources [1], [2], [3]")
            print("• Instructs LLM to include citations in response")
            print("• Provides source attribution for fact-checking")
            print("• Enables traceability of information")
        
    except Exception as e:
        print(f"Error in citation demonstration: {e}")


def demonstrate_response_validation():
    """Demonstrate response validation features"""
    print("\n" + "=" * 60)
    print("RESPONSE VALIDATION DEMONSTRATION")
    print("=" * 60)
    
    config = PipelineConfig()
    
    try:
        with patch('rag_engine.generation.generation_engine.ChatGoogleGenerativeAI'):
            engine = GenerationEngine(config)
            documents = create_sample_documents()
            query = "What is Python?"
            
            # Example responses for validation
            good_response = "Python is a high-level programming language created by Guido van Rossum, first released in 1991."
            bad_response = "Python is a programming language invented in 2010 by Microsoft for web development only."
            
            print("VALIDATION EXAMPLES:")
            print("-" * 40)
            
            print(f"Query: {query}")
            print(f"\nGood Response: {good_response}")
            print("Expected Validation: GROUNDED (information matches context)")
            
            print(f"\nBad Response: {bad_response}")
            print("Expected Validation: NOT_GROUNDED (incorrect date and creator)")
            
            print("\nVALIDATION CRITERIA:")
            print("-" * 40)
            print("• Factual accuracy against provided context")
            print("• Detection of hallucinated information")
            print("• Identification of contradictions")
            print("• Relevance to the original question")
            print("• Confidence scoring (0.0 - 1.0)")
            
            print("\nVALIDATION OUTPUT FORMAT:")
            print("-" * 40)
            print("GROUNDED: [YES/NO/PARTIALLY]")
            print("CONFIDENCE: [0.0-1.0]")
            print("ISSUES: [List of specific problems]")
            print("EXPLANATION: [Reasoning for assessment]")
            
            # Show validation prompt template
            validation_prompt = engine.validation_prompt_template.format(
                question=query,
                context=engine.format_docs(documents),
                answer=good_response
            )
            
            print(f"\nValidation prompt includes:")
            print("• Context documents for fact-checking")
            print("• Original question for relevance assessment")
            print("• Answer to be validated")
            print("• Structured output format requirements")
        
    except Exception as e:
        print(f"Error in validation demonstration: {e}")


def demonstrate_full_grounding_workflow():
    """Demonstrate the complete grounding workflow"""
    print("\n" + "=" * 60)
    print("FULL GROUNDING WORKFLOW DEMONSTRATION")
    print("=" * 60)
    
    config = PipelineConfig()
    
    try:
        with patch('rag_engine.generation.generation_engine.ChatGoogleGenerativeAI'):
            engine = GenerationEngine(config)
            documents = create_sample_documents()
            query = "What is Python and what are its main applications?"
            
            print("FULL GROUNDING WORKFLOW STEPS:")
            print("-" * 40)
            print("1. Generate response with citations")
            print("2. Validate response grounding")
            print("3. Calculate confidence score")
            print("4. Provide comprehensive metadata")
            
            print(f"\nQuery: {query}")
            print(f"Context Documents: {len(documents)}")
            
            print("\nEXPECTED RESPONSE FEATURES:")
            print("-" * 40)
            print("• Citations: [1], [2], [3] referencing source documents")
            print("• Grounding: Only information from provided context")
            print("• Validation: Automatic fact-checking against sources")
            print("• Confidence: Score based on validation results")
            print("• Metadata: Complete processing information")
            
            print("\nMETADATA INCLUDES:")
            print("-" * 40)
            print("• grounding_applied: True")
            print("• citations_included: True/False")
            print("• validation_applied: True/False")
            print("• source_count: Number of source documents")
            print("• source_metadata: Citation mapping")
            print("• validation_results: Grounding assessment")
            print("• processing_time: Performance metrics")
            
            print("\nCONFIDENCE SCORING:")
            print("-" * 40)
            print("• GROUNDED response: 0.8+ confidence")
            print("• PARTIALLY_GROUNDED: 0.5-0.7 confidence")
            print("• NOT_GROUNDED: 0.0-0.3 confidence")
        
    except Exception as e:
        print(f"Error in full grounding demonstration: {e}")


def demonstrate_citation_extraction():
    """Demonstrate citation extraction from responses"""
    print("\n" + "=" * 60)
    print("CITATION EXTRACTION DEMONSTRATION")
    print("=" * 60)
    
    config = PipelineConfig()
    
    try:
        with patch('rag_engine.generation.generation_engine.ChatGoogleGenerativeAI'):
            engine = GenerationEngine(config)
            
            # Example responses with citations
            responses = [
                "Python is a programming language [1] created by Guido van Rossum [2].",
                "Python supports multiple paradigms [2] and is used in data science [3].",
                "Python has no citations in this response.",
                "Multiple citations [1] can appear [2] throughout [3] the text [1]."
            ]
            
            print("CITATION EXTRACTION EXAMPLES:")
            print("-" * 40)
            
            for i, response in enumerate(responses, 1):
                citations = engine.extract_citations_from_response(response)
                print(f"Response {i}: {response}")
                print(f"Citations found: {citations}")
                print()
            
            print("CITATION EXTRACTION FEATURES:")
            print("-" * 40)
            print("• Regex pattern matching: \\[\\d+\\]")
            print("• Returns list of citation numbers")
            print("• Handles duplicate citations")
            print("• Useful for citation validation")
        
    except Exception as e:
        print(f"Error in citation extraction demonstration: {e}")


def main():
    """Run all demonstrations"""
    print("GROUNDED GENERATION FEATURES DEMONSTRATION")
    print("This example shows the enhanced generation capabilities")
    print("implementing requirement 5.4 for grounded responses.")
    
    demonstrate_basic_generation()
    demonstrate_citation_generation()
    demonstrate_response_validation()
    demonstrate_full_grounding_workflow()
    demonstrate_citation_extraction()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("The enhanced GenerationEngine provides:")
    print("• Grounded generation with strict context adherence")
    print("• Citation generation with source attribution")
    print("• Response validation against retrieved documents")
    print("• Comprehensive metadata and confidence scoring")
    print("• Full integration with existing RAG pipeline")
    print("\nThese features implement requirement 5.4:")
    print("'WHEN generation occurs THEN the system SHALL ensure")
    print("responses are grounded in retrieved context'")


if __name__ == "__main__":
    main()