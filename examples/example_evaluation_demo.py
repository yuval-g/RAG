"""
Evaluation Framework Demo - No API Keys Required

This example demonstrates the evaluation framework structure and capabilities
without requiring actual API keys or external dependencies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag_engine.evaluation import MetricsCollector
from src.rag_engine.core.models import TestCase, RAGResponse, Document, EvaluationResult
import json


def demonstrate_evaluation_models():
    """Demonstrate the evaluation data models"""
    print("=" * 80)
    print("EVALUATION DATA MODELS DEMONSTRATION")
    print("=" * 80)
    
    # Create sample test cases
    test_cases = [
        TestCase(
            question="What is the capital of France?",
            expected_answer="Paris",
            metadata={"category": "geography", "difficulty": "easy"}
        ),
        TestCase(
            question="Who wrote '1984'?",
            expected_answer="George Orwell",
            metadata={"category": "literature", "difficulty": "medium"}
        )
    ]
    
    # Create sample RAG responses
    responses = [
        RAGResponse(
            answer="Paris is the capital and largest city of France.",
            source_documents=[
                Document(
                    content="Paris is the capital and most populous city of France.",
                    metadata={"source": "geography_db", "confidence": 0.95}
                )
            ],
            confidence_score=0.95,
            processing_time=0.12,
            metadata={"retrieval_method": "semantic_search"}
        ),
        RAGResponse(
            answer="George Orwell wrote the dystopian novel '1984'.",
            source_documents=[
                Document(
                    content="Nineteen Eighty-Four is a dystopian novel by George Orwell.",
                    metadata={"source": "literature_db", "confidence": 0.92}
                )
            ],
            confidence_score=0.92,
            processing_time=0.15,
            metadata={"retrieval_method": "hybrid_search"}
        )
    ]
    
    print(f"📋 Created {len(test_cases)} test cases:")
    for i, test_case in enumerate(test_cases, 1):
        print(f"  {i}. {test_case.question}")
        print(f"     Expected: {test_case.expected_answer}")
        print(f"     Category: {test_case.metadata.get('category', 'N/A')}")
    
    print(f"\n🤖 Created {len(responses)} RAG responses:")
    for i, response in enumerate(responses, 1):
        print(f"  {i}. {response.answer}")
        print(f"     Confidence: {response.confidence_score:.2f}")
        print(f"     Sources: {len(response.source_documents)}")
        print(f"     Time: {response.processing_time:.3f}s")
    
    return test_cases, responses


def demonstrate_evaluation_results():
    """Demonstrate evaluation result structures"""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS DEMONSTRATION")
    print("=" * 80)
    
    # Create sample evaluation results from different frameworks
    custom_result = EvaluationResult(
        overall_score=0.85,
        metric_scores={
            "faithfulness": 0.90,
            "correctness": 0.82,
            "contextual_relevancy": 0.83
        },
        test_case_results=[
            {
                "question": "What is the capital of France?",
                "expected_answer": "Paris",
                "generated_answer": "Paris is the capital and largest city of France.",
                "metrics": {
                    "faithfulness": {"score": 0.95, "reasoning": "Fully grounded in context", "metric": "faithfulness"},
                    "correctness": {"score": 0.90, "reasoning": "Factually accurate", "metric": "correctness"},
                    "contextual_relevancy": {"score": 0.85, "reasoning": "Highly relevant context", "metric": "contextual_relevancy"}
                }
            },
            {
                "question": "Who wrote '1984'?",
                "expected_answer": "George Orwell",
                "generated_answer": "George Orwell wrote the dystopian novel '1984'.",
                "metrics": {
                    "faithfulness": {"score": 0.85, "reasoning": "Well grounded", "metric": "faithfulness"},
                    "correctness": {"score": 0.75, "reasoning": "Mostly accurate", "metric": "correctness"},
                    "contextual_relevancy": {"score": 0.80, "reasoning": "Good context match", "metric": "contextual_relevancy"}
                }
            }
        ],
        recommendations=[
            "✅ Excellent faithfulness score - low hallucination risk",
            "📈 Consider improving correctness through knowledge base enhancement",
            "🎯 Contextual relevancy is strong - retrieval system performing well"
        ]
    )
    
    deepeval_result = EvaluationResult(
        overall_score=0.78,
        metric_scores={
            "correctness": 0.80,
            "faithfulness": 0.82,
            "contextual_relevancy": 0.72
        },
        test_case_results=[
            {
                "question": "What is the capital of France?",
                "expected_answer": "Paris",
                "generated_answer": "Paris is the capital and largest city of France.",
                "metrics": {
                    "correctness": {"score": 0.85, "reasoning": "DeepEval correctness assessment", "metric": "correctness"}
                }
            }
        ],
        recommendations=[
            "📊 DeepEval assessment shows good overall performance",
            "🔧 Consider tuning contextual relevancy threshold"
        ]
    )
    
    ragas_result = EvaluationResult(
        overall_score=0.82,
        metric_scores={
            "faithfulness": 0.88,
            "answer_relevancy": 0.79,
            "context_recall": 0.85,
            "answer_correctness": 0.76
        },
        test_case_results=[
            {
                "question": "What is the capital of France?",
                "expected_answer": "Paris",
                "generated_answer": "Paris is the capital and largest city of France.",
                "metrics": {
                    "faithfulness": {"score": 0.90, "reasoning": "RAGAS faithfulness evaluation", "metric": "faithfulness"},
                    "answer_relevancy": {"score": 0.85, "reasoning": "RAGAS relevancy evaluation", "metric": "answer_relevancy"}
                }
            }
        ],
        recommendations=[
            "🎯 RAGAS shows strong RAG-specific performance",
            "📚 Context recall is excellent - retrieval system working well"
        ]
    )
    
    results = {
        "custom": custom_result,
        "deepeval": deepeval_result,
        "ragas": ragas_result
    }
    
    print("📊 EVALUATION RESULTS FROM DIFFERENT FRAMEWORKS:")
    print("-" * 60)
    
    for framework_name, result in results.items():
        print(f"\n{framework_name.upper()} FRAMEWORK:")
        print(f"  Overall Score: {result.overall_score:.2f}")
        print(f"  Metrics: {list(result.metric_scores.keys())}")
        print(f"  Test Cases: {len(result.test_case_results)}")
        print(f"  Recommendations: {len(result.recommendations)}")
        
        # Show top metric scores
        top_metrics = sorted(result.metric_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        print(f"  Top Metrics: {', '.join([f'{k}={v:.2f}' for k, v in top_metrics])}")
    
    return results


def demonstrate_metrics_collector():
    """Demonstrate the metrics collector functionality"""
    print("\n" + "=" * 80)
    print("METRICS COLLECTOR DEMONSTRATION")
    print("=" * 80)
    
    # Get sample results
    results = demonstrate_evaluation_results()
    
    # Create metrics collector
    collector = MetricsCollector()
    
    # Add results with execution times
    collector.add_result("custom", results["custom"], 1.5)
    collector.add_result("deepeval", results["deepeval"], 2.1)
    collector.add_result("ragas", results["ragas"], 3.2)
    
    # Add an error case
    collector.add_error("failed_framework", "Connection timeout", 0.5)
    
    print("\n📊 AGGREGATED METRICS ACROSS ALL FRAMEWORKS:")
    print("-" * 50)
    aggregated = collector.get_aggregated_metrics()
    for metric, score in aggregated.items():
        print(f"  {metric.replace('_', ' ').title()}: {score:.3f}")
    
    print("\n🔧 FRAMEWORK COMPARISON:")
    print("-" * 50)
    comparison = collector.get_framework_comparison()
    for framework, data in comparison.items():
        print(f"\n  {framework.upper()}:")
        print(f"    Status: {data['status']}")
        print(f"    Overall Score: {data['overall_score']:.2f}")
        print(f"    Execution Time: {data['execution_time']:.1f}s")
        print(f"    Test Cases: {data['test_cases_evaluated']}")
        if data['status'] == 'error':
            print(f"    Error: {data['error_message']}")
    
    return collector


def demonstrate_comprehensive_reporting():
    """Demonstrate comprehensive evaluation reporting"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION REPORTING")
    print("=" * 80)
    
    collector = demonstrate_metrics_collector()
    
    # Simulate comprehensive report structure
    aggregated_metrics = collector.get_aggregated_metrics()
    framework_comparison = collector.get_framework_comparison()
    
    # Generate mock comprehensive report
    report = {
        "evaluation_summary": {
            "timestamp": "2024-01-15 14:30:00",
            "total_test_cases": 2,
            "frameworks_used": ["custom", "deepeval", "ragas"],
            "overall_score": sum(aggregated_metrics.values()) / len(aggregated_metrics) if aggregated_metrics else 0,
            "total_execution_time": sum(data["execution_time"] for data in framework_comparison.values())
        },
        "aggregated_metrics": aggregated_metrics,
        "framework_comparison": framework_comparison,
        "performance_analysis": {
            "fastest_framework": min(
                [(name, data["execution_time"]) for name, data in framework_comparison.items() if data["status"] == "success"],
                key=lambda x: x[1]
            )[0] if any(data["status"] == "success" for data in framework_comparison.values()) else "none",
            "most_accurate_framework": max(
                [(name, data["overall_score"]) for name, data in framework_comparison.items() if data["status"] == "success"],
                key=lambda x: x[1]
            )[0] if any(data["status"] == "success" for data in framework_comparison.values()) else "none"
        },
        "recommendations": [
            "🎯 Overall system performance is strong across multiple evaluation frameworks",
            "⚡ Custom framework provides fastest evaluation times",
            "📊 RAGAS offers most comprehensive RAG-specific metrics",
            "🔧 Consider addressing failed framework connection issues",
            "📈 Faithfulness scores are consistently high - low hallucination risk"
        ]
    }
    
    print("\n📈 COMPREHENSIVE EVALUATION REPORT:")
    print("=" * 60)
    
    print(f"\n📋 EVALUATION SUMMARY:")
    summary = report["evaluation_summary"]
    print(f"  Timestamp: {summary['timestamp']}")
    print(f"  Test Cases: {summary['total_test_cases']}")
    print(f"  Frameworks: {', '.join(summary['frameworks_used'])}")
    print(f"  Overall Score: {summary['overall_score']:.3f}")
    print(f"  Total Time: {summary['total_execution_time']:.1f}s")
    
    print(f"\n📊 AGGREGATED METRICS:")
    for metric, score in report["aggregated_metrics"].items():
        print(f"  {metric.replace('_', ' ').title()}: {score:.3f}")
    
    print(f"\n⚡ PERFORMANCE ANALYSIS:")
    perf = report["performance_analysis"]
    print(f"  Fastest Framework: {perf['fastest_framework']}")
    print(f"  Most Accurate: {perf['most_accurate_framework']}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"  {i}. {rec}")
    
    # Show JSON export capability
    print(f"\n💾 REPORT EXPORT (JSON):")
    print("-" * 30)
    print(json.dumps({
        "summary": report["evaluation_summary"],
        "metrics": report["aggregated_metrics"]
    }, indent=2))


def main():
    """Main demonstration function"""
    print("🔍 EVALUATION FRAMEWORK DEMONSTRATION")
    print("=" * 80)
    print("This example demonstrates the evaluation framework capabilities")
    print("without requiring API keys or external dependencies.")
    print()
    print("📋 Components demonstrated:")
    print("• Evaluation data models (TestCase, RAGResponse, EvaluationResult)")
    print("• Multiple evaluation framework result structures")
    print("• MetricsCollector for aggregating results")
    print("• Comprehensive reporting and analysis")
    print("• JSON export capabilities")
    
    # Run demonstrations
    test_cases, responses = demonstrate_evaluation_models()
    results = demonstrate_evaluation_results()
    collector = demonstrate_metrics_collector()
    demonstrate_comprehensive_reporting()
    
    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("🚀 Key takeaways:")
    print("• The evaluation framework supports multiple evaluation approaches")
    print("• Results can be aggregated and compared across frameworks")
    print("• Comprehensive reporting provides actionable insights")
    print("• The system gracefully handles framework failures")
    print("• All data structures are JSON-serializable for export")
    print()
    print("📖 Next steps:")
    print("1. Set up API keys for actual LLM-based evaluation")
    print("2. Install optional frameworks (deepeval, ragas)")
    print("3. Run example_comprehensive_evaluation.py for full demo")
    print("4. Integrate evaluation into your RAG pipeline")


if __name__ == "__main__":
    main()