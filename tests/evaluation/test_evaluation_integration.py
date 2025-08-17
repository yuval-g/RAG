"""
Integration tests for the complete evaluation framework
"""

import pytest
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.evaluation import (
    EvaluationManager, 
    CustomEvaluator, 
    DeepEvalIntegration, 
    RAGASIntegration,
    MetricsCollector
)
from src.rag_engine.core.models import TestCase, RAGResponse, Document, EvaluationResult


class TestEvaluationIntegration:
    """Integration tests for the complete evaluation framework"""
    
    @patch('src.rag_engine.evaluation.evaluation_manager.DEEPEVAL_AVAILABLE', False)
    @patch('src.rag_engine.evaluation.evaluation_manager.RAGAS_AVAILABLE', False)
    def test_end_to_end_evaluation_custom_only(self):
        """Test end-to-end evaluation with custom evaluator only"""
        # Create test data
        test_cases = [
            TestCase(
                question="What is the capital of France?",
                expected_answer="Paris"
            ),
            TestCase(
                question="What is 2+2?",
                expected_answer="4"
            )
        ]
        
        responses = [
            RAGResponse(
                answer="Paris is the capital of France.",
                source_documents=[
                    Document(content="Paris is the capital of France and the largest city in the country.")
                ]
            ),
            RAGResponse(
                answer="2+2 equals 4.",
                source_documents=[
                    Document(content="Basic arithmetic: 2+2=4")
                ]
            )
        ]
        
        # Mock the custom evaluator to return predictable results
        with patch('src.rag_engine.evaluation.evaluation_manager.CustomEvaluator') as mock_custom:
            mock_evaluator = Mock()
            mock_result = EvaluationResult(
                overall_score=0.85,
                metric_scores={
                    "correctness": 0.9,
                    "faithfulness": 0.8,
                    "contextual_relevancy": 0.85
                },
                test_case_results=[
                    {
                        "question": "What is the capital of France?",
                        "expected_answer": "Paris",
                        "generated_answer": "Paris is the capital of France.",
                        "metrics": {
                            "correctness": {"score": 0.9, "reasoning": "Correct answer", "metric": "correctness"},
                            "faithfulness": {"score": 0.8, "reasoning": "Faithful to context", "metric": "faithfulness"},
                            "contextual_relevancy": {"score": 0.85, "reasoning": "Relevant context", "metric": "contextual_relevancy"}
                        }
                    },
                    {
                        "question": "What is 2+2?",
                        "expected_answer": "4",
                        "generated_answer": "2+2 equals 4.",
                        "metrics": {
                            "correctness": {"score": 0.9, "reasoning": "Correct answer", "metric": "correctness"},
                            "faithfulness": {"score": 0.8, "reasoning": "Faithful to context", "metric": "faithfulness"},
                            "contextual_relevancy": {"score": 0.85, "reasoning": "Relevant context", "metric": "contextual_relevancy"}
                        }
                    }
                ],
                recommendations=["System performing well"]
            )
            mock_evaluator.evaluate.return_value = mock_result
            mock_custom.return_value = mock_evaluator
            
            # Initialize evaluation manager
            manager = EvaluationManager(enable_deepeval=False, enable_ragas=False)
            
            # Run comprehensive evaluation
            result = manager.evaluate_comprehensive(test_cases, responses)
            
            # Verify the comprehensive report structure
            assert "evaluation_summary" in result
            assert "aggregated_metrics" in result
            assert "framework_comparison" in result
            assert "test_case_analysis" in result
            assert "performance_analysis" in result
            assert "recommendations" in result
            assert "detailed_results" in result
            
            # Verify evaluation summary
            summary = result["evaluation_summary"]
            assert summary["total_test_cases"] == 2
            assert summary["frameworks_used"] == ["custom"]
            assert summary["overall_score"] > 0
            assert "timestamp" in summary
            assert "total_execution_time" in summary
            
            # Verify aggregated metrics
            metrics = result["aggregated_metrics"]
            assert "correctness" in metrics
            assert "faithfulness" in metrics
            assert "contextual_relevancy" in metrics
            assert all(0 <= score <= 1 for score in metrics.values())
            
            # Verify framework comparison
            comparison = result["framework_comparison"]
            assert "custom" in comparison
            assert comparison["custom"]["status"] == "success"
            assert comparison["custom"]["overall_score"] == 0.85
            assert comparison["custom"]["test_cases_evaluated"] == 2
            
            # Verify test case analysis
            analysis = result["test_case_analysis"]
            assert analysis["total_cases"] == 2
            assert len(analysis["case_performance"]) == 2
            assert len(analysis["worst_performing_cases"]) <= 2
            assert len(analysis["best_performing_cases"]) <= 2
            
            # Verify performance analysis
            performance = result["performance_analysis"]
            assert "total_execution_time" in performance
            assert "framework_performance" in performance
            assert "efficiency_ranking" in performance
            assert "custom" in performance["framework_performance"]
            
            # Verify recommendations
            recommendations = result["recommendations"]
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0
            
            # Verify detailed results
            detailed = result["detailed_results"]
            assert "custom" in detailed
    
    def test_metrics_collector_comprehensive(self):
        """Test MetricsCollector with comprehensive data"""
        collector = MetricsCollector()
        
        # Add results from multiple frameworks
        custom_result = EvaluationResult(
            overall_score=0.8,
            metric_scores={"faithfulness": 0.9, "correctness": 0.7},
            test_case_results=[{"test": "case1"}],
            recommendations=["Custom rec"]
        )
        
        deepeval_result = EvaluationResult(
            overall_score=0.75,
            metric_scores={"faithfulness": 0.8, "correctness": 0.7},
            test_case_results=[{"test": "case1"}],
            recommendations=["DeepEval rec"]
        )
        
        collector.add_result("custom", custom_result, 1.5)
        collector.add_result("deepeval", deepeval_result, 2.0)
        collector.add_error("ragas", "Connection timeout", 0.5)
        
        # Test aggregated metrics
        aggregated = collector.get_aggregated_metrics()
        assert len(aggregated) == 2  # faithfulness and correctness
        assert abs(aggregated["faithfulness"] - 0.85) < 0.001  # (0.9 + 0.8) / 2
        assert aggregated["correctness"] == 0.7  # (0.7 + 0.7) / 2
        
        # Test framework comparison
        comparison = collector.get_framework_comparison()
        assert len(comparison) == 3  # custom, deepeval, ragas
        
        # Check successful frameworks
        assert comparison["custom"]["status"] == "success"
        assert comparison["custom"]["overall_score"] == 0.8
        assert comparison["custom"]["execution_time"] == 1.5
        assert comparison["custom"]["test_cases_evaluated"] == 1
        
        assert comparison["deepeval"]["status"] == "success"
        assert comparison["deepeval"]["overall_score"] == 0.75
        assert comparison["deepeval"]["execution_time"] == 2.0
        assert comparison["deepeval"]["test_cases_evaluated"] == 1
        
        # Check failed framework
        assert comparison["ragas"]["status"] == "error"
        assert comparison["ragas"]["overall_score"] == 0.0
        assert comparison["ragas"]["execution_time"] == 0.5
        assert comparison["ragas"]["test_cases_evaluated"] == 0
        assert comparison["ragas"]["error_message"] == "Connection timeout"
    
    @patch('src.rag_engine.evaluation.evaluation_manager.DEEPEVAL_AVAILABLE', False)
    @patch('src.rag_engine.evaluation.evaluation_manager.RAGAS_AVAILABLE', False)
    def test_evaluation_manager_framework_info(self):
        """Test getting framework information from evaluation manager"""
        with patch('src.rag_engine.evaluation.evaluation_manager.CustomEvaluator') as mock_custom:
            mock_evaluator = Mock()
            mock_evaluator.get_supported_metrics.return_value = ["faithfulness", "correctness", "contextual_relevancy"]
            mock_custom.return_value = mock_evaluator
            
            manager = EvaluationManager(enable_deepeval=False, enable_ragas=False)
            
            # Test available frameworks
            frameworks = manager.get_available_frameworks()
            assert frameworks == ["custom"]
            
            # Test framework info
            info = manager.get_framework_info()
            assert "custom" in info
            assert info["custom"]["name"] == "custom"
            assert info["custom"]["available"] == True
            assert info["custom"]["supported_metrics"] == ["faithfulness", "correctness", "contextual_relevancy"]
    
    @patch('src.rag_engine.evaluation.evaluation_manager.DEEPEVAL_AVAILABLE', False)
    @patch('src.rag_engine.evaluation.evaluation_manager.RAGAS_AVAILABLE', False)
    def test_evaluation_with_mixed_results(self):
        """Test evaluation with mixed success and failure results"""
        test_cases = [TestCase(question="Test?", expected_answer="Answer")]
        responses = [RAGResponse(answer="Response", source_documents=[])]
        
        with patch('src.rag_engine.evaluation.evaluation_manager.CustomEvaluator') as mock_custom:
            # Mock evaluator that fails
            mock_evaluator = Mock()
            mock_evaluator.evaluate.side_effect = Exception("Network error")
            mock_custom.return_value = mock_evaluator
            
            manager = EvaluationManager(enable_deepeval=False, enable_ragas=False)
            
            result = manager.evaluate_comprehensive(test_cases, responses)
            
            # Should still return a comprehensive report even with failures
            assert "evaluation_summary" in result
            assert "framework_comparison" in result
            assert "recommendations" in result
            
            # Check that the failure is properly recorded
            comparison = result["framework_comparison"]
            assert "custom" in comparison
            assert comparison["custom"]["status"] == "error"
            assert "error_message" in comparison["custom"]
            
            # Should have recommendations about the failure
            recommendations = result["recommendations"]
            assert any("failed" in rec.lower() for rec in recommendations)
    
    def test_evaluation_result_model_compatibility(self):
        """Test that EvaluationResult model works correctly with the framework"""
        # Test creating an EvaluationResult
        result = EvaluationResult(
            overall_score=0.85,
            metric_scores={"faithfulness": 0.9, "correctness": 0.8},
            test_case_results=[
                {
                    "question": "Test question",
                    "expected_answer": "Expected",
                    "generated_answer": "Generated",
                    "metrics": {
                        "faithfulness": {"score": 0.9, "reasoning": "Good", "metric": "faithfulness"}
                    }
                }
            ],
            recommendations=["System is performing well"]
        )
        
        # Test that the model can be serialized
        if hasattr(result, 'model_dump'):
            serialized = result.model_dump()
        elif hasattr(result, 'dict'):
            serialized = result.dict()
        else:
            serialized = result.__dict__
        
        assert "overall_score" in serialized
        assert "metric_scores" in serialized
        assert "test_case_results" in serialized
        assert "recommendations" in serialized
        
        assert serialized["overall_score"] == 0.85
        assert serialized["metric_scores"]["faithfulness"] == 0.9
        assert len(serialized["test_case_results"]) == 1
        assert len(serialized["recommendations"]) == 1


if __name__ == "__main__":
    pytest.main([__file__])