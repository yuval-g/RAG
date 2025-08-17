"""
Comprehensive Evaluation Framework Example

This example demonstrates how to use the comprehensive evaluation framework
to evaluate RAG system performance using multiple evaluation approaches:
- Custom LLM-based evaluation
- DeepEval integration (if available)
- RAGAS integration (if available)

The evaluation manager coordinates all frameworks and provides detailed reporting.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag_engine.evaluation import (
    EvaluationManager,
    CustomEvaluator,
    DeepEvalIntegration,
    RAGASIntegration,
    MetricsCollector
)
from src.rag_engine.core.models import TestCase, RAGResponse, Document
import json
import time


def create_sample_test_data():
    """Create sample test cases and responses for evaluation"""
    
    test_cases = [
        TestCase(
            question="What is the capital of France?",
            expected_answer="Paris",
            metadata={"category": "geography", "difficulty": "easy"}
        ),
        TestCase(
            question="Who wrote the novel '1984'?",
            expected_answer="George Orwell",
            metadata={"category": "literature", "difficulty": "medium"}
        ),
        TestCase(
            question="What is the process by which plants make their own food?",
            expected_answer="Photosynthesis",
            metadata={"category": "biology", "difficulty": "medium"}
        ),
        TestCase(
            question="What is the largest planet in our solar system?",
            expected_answer="Jupiter",
            metadata={"category": "astronomy", "difficulty": "easy"}
        ),
        TestCase(
            question="What is the chemical symbol for gold?",
            expected_answer="Au",
            metadata={"category": "chemistry", "difficulty": "hard"}
        )
    ]
    
    responses = [
        RAGResponse(
            answer="Paris is the capital and largest city of France.",
            source_documents=[
                Document(
                    content="Paris is the capital and most populous city of France. Located in northern central France, it is situated on the Seine River.",
                    metadata={"source": "geography_db", "confidence": 0.95}
                )
            ],
            confidence_score=0.95,
            processing_time=0.12
        ),
        RAGResponse(
            answer="George Orwell wrote the dystopian novel '1984', published in 1949.",
            source_documents=[
                Document(
                    content="Nineteen Eighty-Four (1984) is a dystopian social science fiction novel by English novelist George Orwell, published in 1949.",
                    metadata={"source": "literature_db", "confidence": 0.92}
                )
            ],
            confidence_score=0.92,
            processing_time=0.15
        ),
        RAGResponse(
            answer="Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
            source_documents=[
                Document(
                    content="Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that can later be released to fuel the organism's activities.",
                    metadata={"source": "biology_db", "confidence": 0.88}
                )
            ],
            confidence_score=0.88,
            processing_time=0.18
        ),
        RAGResponse(
            answer="Jupiter is the largest planet in our solar system.",
            source_documents=[
                Document(
                    content="Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass more than two and a half times that of all the other planets combined.",
                    metadata={"source": "astronomy_db", "confidence": 0.97}
                )
            ],
            confidence_score=0.97,
            processing_time=0.10
        ),
        RAGResponse(
            answer="The chemical symbol for gold is Au, derived from the Latin word 'aurum'.",
            source_documents=[
                Document(
                    content="Gold is a chemical element with the symbol Au (from Latin: aurum) and atomic number 79.",
                    metadata={"source": "chemistry_db", "confidence": 0.94}
                )
            ],
            confidence_score=0.94,
            processing_time=0.14
        )
    ]
    
    return test_cases, responses


def demonstrate_individual_evaluators():
    """Demonstrate individual evaluation frameworks"""
    print("=" * 80)
    print("INDIVIDUAL EVALUATOR DEMONSTRATIONS")
    print("=" * 80)
    
    test_cases, responses = create_sample_test_data()
    
    # Take just the first test case for individual demonstrations
    single_test_case = [test_cases[0]]
    single_response = [responses[0]]
    
    print("\n1. CUSTOM EVALUATOR")
    print("-" * 40)
    try:
        custom_evaluator = CustomEvaluator()
        print(f"✅ Custom evaluator initialized")
        print(f"📊 Supported metrics: {custom_evaluator.get_supported_metrics()}")
        
        # Note: In a real scenario, this would make actual LLM calls
        # For this example, we'll show the structure without making real calls
        print("📝 Custom evaluator ready for evaluation (requires LLM API keys for actual evaluation)")
        
    except Exception as e:
        print(f"❌ Custom evaluator failed: {str(e)}")
    
    print("\n2. DEEPEVAL INTEGRATION")
    print("-" * 40)
    try:
        deepeval_integration = DeepEvalIntegration()
        print(f"✅ DeepEval integration initialized")
        print(f"📊 Supported metrics: {deepeval_integration.get_supported_metrics()}")
        print(f"🤖 Available models: {deepeval_integration.get_available_models()}")
        
    except ImportError as e:
        print(f"⚠️  DeepEval not available: {str(e)}")
        print("💡 Install with: pip install deepeval")
    except Exception as e:
        print(f"❌ DeepEval integration failed: {str(e)}")
    
    print("\n3. RAGAS INTEGRATION")
    print("-" * 40)
    try:
        ragas_integration = RAGASIntegration()
        print(f"✅ RAGAS integration initialized")
        print(f"📊 Supported metrics: {ragas_integration.get_supported_metrics()}")
        print(f"📖 Metric descriptions:")
        for metric, description in ragas_integration.get_metric_descriptions().items():
            print(f"   • {metric}: {description}")
        
    except ImportError as e:
        print(f"⚠️  RAGAS not available: {str(e)}")
        print("💡 Install with: pip install ragas datasets")
    except Exception as e:
        print(f"❌ RAGAS integration failed: {str(e)}")


def demonstrate_evaluation_manager():
    """Demonstrate the comprehensive evaluation manager"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION MANAGER DEMONSTRATION")
    print("=" * 80)
    
    test_cases, responses = create_sample_test_data()
    
    try:
        # Initialize evaluation manager with available frameworks
        print("\n🚀 Initializing Evaluation Manager...")
        manager = EvaluationManager(
            enable_custom=True,
            enable_deepeval=True,  # Will gracefully handle if not available
            enable_ragas=True      # Will gracefully handle if not available
        )
        
        print(f"✅ Evaluation manager initialized")
        print(f"🔧 Available frameworks: {manager.get_available_frameworks()}")
        
        # Get framework information
        print(f"\n📋 Framework Information:")
        framework_info = manager.get_framework_info()
        for framework_name, info in framework_info.items():
            print(f"   • {framework_name.upper()}:")
            print(f"     - Available: {info['available']}")
            print(f"     - Supported metrics: {info['supported_metrics']}")
        
        # Run comprehensive evaluation
        print(f"\n🔍 Running comprehensive evaluation on {len(test_cases)} test cases...")
        print("⚠️  Note: This is a demonstration. In production, this would make actual API calls.")
        
        # For demonstration purposes, we'll simulate the evaluation
        # In a real scenario, this would make actual LLM API calls
        print("📊 Simulating evaluation process...")
        
        # Create a mock comprehensive report structure
        mock_report = {
            "evaluation_summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_test_cases": len(test_cases),
                "frameworks_used": manager.get_available_frameworks(),
                "overall_score": 0.85,
                "total_execution_time": 2.5
            },
            "aggregated_metrics": {
                "faithfulness": 0.88,
                "correctness": 0.82,
                "contextual_relevancy": 0.85
            },
            "framework_comparison": {
                framework: {
                    "status": "success",
                    "overall_score": 0.85,
                    "execution_time": 1.2,
                    "test_cases_evaluated": len(test_cases)
                }
                for framework in manager.get_available_frameworks()
            },
            "recommendations": [
                "✅ Excellent overall system performance! Continue monitoring and fine-tuning.",
                "📈 Faithfulness score is strong - low hallucination risk.",
                "🎯 Consider improving correctness score through knowledge base enhancement."
            ]
        }
        
        print("\n📈 EVALUATION RESULTS")
        print("-" * 40)
        print(f"Overall Score: {mock_report['evaluation_summary']['overall_score']:.2f}")
        print(f"Total Execution Time: {mock_report['evaluation_summary']['total_execution_time']:.2f}s")
        print(f"Frameworks Used: {', '.join(mock_report['evaluation_summary']['frameworks_used'])}")
        
        print(f"\n📊 AGGREGATED METRICS")
        print("-" * 40)
        for metric, score in mock_report['aggregated_metrics'].items():
            print(f"{metric.replace('_', ' ').title()}: {score:.2f}")
        
        print(f"\n🔧 FRAMEWORK COMPARISON")
        print("-" * 40)
        for framework, data in mock_report['framework_comparison'].items():
            print(f"{framework.upper()}:")
            print(f"  Status: {data['status']}")
            print(f"  Score: {data['overall_score']:.2f}")
            print(f"  Time: {data['execution_time']:.2f}s")
        
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 40)
        for i, recommendation in enumerate(mock_report['recommendations'], 1):
            print(f"{i}. {recommendation}")
        
        # Demonstrate test case analysis
        print(f"\n📋 TEST CASE ANALYSIS")
        print("-" * 40)
        print(f"Total test cases: {len(test_cases)}")
        print("Sample test case performance:")
        for i, (test_case, response) in enumerate(zip(test_cases[:2], responses[:2])):
            print(f"\n  Test Case {i+1}:")
            print(f"    Question: {test_case.question}")
            print(f"    Expected: {test_case.expected_answer}")
            print(f"    Generated: {response.answer}")
            print(f"    Confidence: {response.confidence_score:.2f}")
            print(f"    Processing Time: {response.processing_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Evaluation manager demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()


def demonstrate_metrics_collector():
    """Demonstrate the metrics collector functionality"""
    print("\n" + "=" * 80)
    print("METRICS COLLECTOR DEMONSTRATION")
    print("=" * 80)
    
    from src.rag_engine.core.models import EvaluationResult
    
    # Create a metrics collector
    collector = MetricsCollector()
    
    # Simulate results from different frameworks
    custom_result = EvaluationResult(
        overall_score=0.85,
        metric_scores={"faithfulness": 0.9, "correctness": 0.8},
        test_case_results=[{"test": "case1"}],
        recommendations=["Custom recommendation"]
    )
    
    deepeval_result = EvaluationResult(
        overall_score=0.82,
        metric_scores={"faithfulness": 0.85, "correctness": 0.79},
        test_case_results=[{"test": "case1"}],
        recommendations=["DeepEval recommendation"]
    )
    
    # Add results to collector
    collector.add_result("custom", custom_result, 1.5)
    collector.add_result("deepeval", deepeval_result, 2.1)
    collector.add_error("ragas", "Connection timeout", 0.5)
    
    print("\n📊 AGGREGATED METRICS")
    print("-" * 40)
    aggregated = collector.get_aggregated_metrics()
    for metric, score in aggregated.items():
        print(f"{metric.replace('_', ' ').title()}: {score:.3f}")
    
    print("\n🔧 FRAMEWORK COMPARISON")
    print("-" * 40)
    comparison = collector.get_framework_comparison()
    for framework, data in comparison.items():
        print(f"{framework.upper()}:")
        print(f"  Status: {data['status']}")
        print(f"  Overall Score: {data['overall_score']:.2f}")
        print(f"  Execution Time: {data['execution_time']:.2f}s")
        print(f"  Test Cases: {data['test_cases_evaluated']}")
        if data['status'] == 'error':
            print(f"  Error: {data['error_message']}")


def main():
    """Main demonstration function"""
    print("🔍 COMPREHENSIVE RAG EVALUATION FRAMEWORK DEMONSTRATION")
    print("=" * 80)
    print("This example demonstrates the comprehensive evaluation framework")
    print("that can coordinate multiple evaluation approaches for RAG systems.")
    print()
    print("📋 Features demonstrated:")
    print("• Custom LLM-based evaluation with faithfulness and correctness metrics")
    print("• DeepEval integration for standardized evaluation")
    print("• RAGAS integration for RAG-specific metrics")
    print("• Comprehensive evaluation manager with detailed reporting")
    print("• Metrics aggregation and framework comparison")
    print("• Performance analysis and intelligent recommendations")
    
    # Demonstrate individual evaluators
    demonstrate_individual_evaluators()
    
    # Demonstrate evaluation manager
    demonstrate_evaluation_manager()
    
    # Demonstrate metrics collector
    demonstrate_metrics_collector()
    
    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("🚀 To use in production:")
    print("1. Set up your LLM API keys (Google Gemini for custom evaluation)")
    print("2. Install optional frameworks: pip install deepeval ragas datasets")
    print("3. Create your test cases and RAG responses")
    print("4. Initialize EvaluationManager and run comprehensive evaluation")
    print("5. Analyze the detailed reports and implement recommendations")
    print()
    print("📖 For more examples, see the other files in the examples/ directory")


if __name__ == "__main__":
    main()