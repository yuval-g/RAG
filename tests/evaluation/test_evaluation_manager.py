"""
Unit tests for evaluation manager and metrics collector
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.evaluation.evaluation_manager import EvaluationManager, MetricsCollector
from src.rag_engine.core.models import TestCase, RAGResponse, Document, EvaluationResult


class TestMetricsCollector:
    """Test the MetricsCollector class"""
    
    def test_initialization(self):
        """Test MetricsCollector initialization"""
        collector = MetricsCollector()
        assert collector.results == {}
        assert collector.execution_times == {}
        assert collector.errors == {}
    
    def test_add_result(self):
        """Test adding evaluation result"""
        collector = MetricsCollector()
        
        result = EvaluationResult(
            overall_score=0.8,
            metric_scores={"faithfulness": 0.9, "correctness": 0.7},
            test_case_results=[],
            recommendations=[]
        )
        
        collector.add_result("custom", result, 1.5)
        
        assert "custom" in collector.results
        assert collector.results["custom"] == result
        assert collector.execution_times["custom"] == 1.5
    
    def test_add_error(self):
        """Test adding error"""
        collector = MetricsCollector()
        
        collector.add_error("deepeval", "Connection failed", 0.5)
        
        assert "deepeval" in collector.errors
        assert collector.errors["deepeval"] == "Connection failed"
        assert collector.execution_times["deepeval"] == 0.5
    
    def test_get_aggregated_metrics(self):
        """Test getting aggregated metrics"""
        collector = MetricsCollector()
        
        # Add results from multiple frameworks
        result1 = EvaluationResult(
            overall_score=0.8,
            metric_scores={"faithfulness": 0.9, "correctness": 0.7},
            test_case_results=[],
            recommendations=[]
        )
        
        result2 = EvaluationResult(
            overall_score=0.7,
            metric_scores={"faithfulness": 0.8, "correctness": 0.6},
            test_case_results=[],
            recommendations=[]
        )
        
        collector.add_result("custom", result1, 1.0)
        collector.add_result("deepeval", result2, 2.0)
        
        aggregated = collector.get_aggregated_metrics()
        
        assert "faithfulness" in aggregated
        assert "correctness" in aggregated
        assert abs(aggregated["faithfulness"] - 0.85) < 0.001  # (0.9 + 0.8) / 2
        assert abs(aggregated["correctness"] - 0.65) < 0.001   # (0.7 + 0.6) / 2
    
    def test_get_aggregated_metrics_with_errors(self):
        """Test aggregated metrics ignores error metrics"""
        collector = MetricsCollector()
        
        result = EvaluationResult(
            overall_score=0.0,
            metric_scores={"error": 0.0, "faithfulness": 0.8},
            test_case_results=[],
            recommendations=[]
        )
        
        collector.add_result("custom", result, 1.0)
        
        aggregated = collector.get_aggregated_metrics()
        
        assert "error" not in aggregated
        assert "faithfulness" in aggregated
        assert aggregated["faithfulness"] == 0.8
    
    def test_get_framework_comparison(self):
        """Test getting framework comparison"""
        collector = MetricsCollector()
        
        result = EvaluationResult(
            overall_score=0.8,
            metric_scores={"faithfulness": 0.9},
            test_case_results=[{"question": "test"}],
            recommendations=[]
        )
        
        collector.add_result("custom", result, 1.5)
        collector.add_error("deepeval", "Failed", 0.5)
        
        comparison = collector.get_framework_comparison()
        
        assert "custom" in comparison
        assert "deepeval" in comparison
        
        # Check successful framework
        assert comparison["custom"]["status"] == "success"
        assert comparison["custom"]["overall_score"] == 0.8
        assert comparison["custom"]["execution_time"] == 1.5
        assert comparison["custom"]["test_cases_evaluated"] == 1
        
        # Check failed framework
        assert comparison["deepeval"]["status"] == "error"
        assert comparison["deepeval"]["overall_score"] == 0.0
        assert comparison["deepeval"]["execution_time"] == 0.5
        assert comparison["deepeval"]["test_cases_evaluated"] == 0
        assert comparison["deepeval"]["error_message"] == "Failed"


class TestEvaluationManager:
    """Test the EvaluationManager class"""
    
    @patch('src.rag_engine.evaluation.evaluation_manager.DEEPEVAL_AVAILABLE', False)
    @patch('src.rag_engine.evaluation.evaluation_manager.RAGAS_AVAILABLE', False)
    def test_initialization_custom_only(self):
        """Test initialization with only custom evaluator"""
        with patch('src.rag_engine.evaluation.evaluation_manager.CustomEvaluator') as mock_custom:
            mock_custom.return_value = Mock()
            
            manager = EvaluationManager(enable_deepeval=False, enable_ragas=False)
            
            assert "custom" in manager.evaluators
            assert manager.enabled_frameworks == ["custom"]
    
    @patch('src.rag_engine.evaluation.evaluation_manager.DEEPEVAL_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.evaluation_manager.RAGAS_AVAILABLE', True)
    def test_initialization_all_frameworks(self):
        """Test initialization with all frameworks"""
        with patch('src.rag_engine.evaluation.evaluation_manager.CustomEvaluator') as mock_custom, \
             patch('src.rag_engine.evaluation.evaluation_manager.DeepEvalIntegration') as mock_deepeval, \
             patch('src.rag_engine.evaluation.evaluation_manager.RAGASIntegration') as mock_ragas:
            
            mock_custom.return_value = Mock()
            mock_deepeval.return_value = Mock()
            mock_ragas.return_value = Mock()
            
            manager = EvaluationManager()
            
            assert "custom" in manager.evaluators
            assert "deepeval" in manager.evaluators
            assert "ragas" in manager.evaluators
            assert len(manager.enabled_frameworks) == 3
    
    def test_initialization_no_frameworks(self):
        """Test initialization when no frameworks are available"""
        with patch('src.rag_engine.evaluation.evaluation_manager.CustomEvaluator', side_effect=Exception("Failed")):
            with pytest.raises(RuntimeError, match="No evaluation frameworks could be initialized"):
                EvaluationManager(enable_deepeval=False, enable_ragas=False)
    
    @patch('src.rag_engine.evaluation.evaluation_manager.DEEPEVAL_AVAILABLE', False)
    @patch('src.rag_engine.evaluation.evaluation_manager.RAGAS_AVAILABLE', False)
    def test_get_available_frameworks(self):
        """Test getting available frameworks"""
        with patch('src.rag_engine.evaluation.evaluation_manager.CustomEvaluator') as mock_custom:
            mock_custom.return_value = Mock()
            
            manager = EvaluationManager(enable_deepeval=False, enable_ragas=False)
            frameworks = manager.get_available_frameworks()
            
            assert frameworks == ["custom"]
    
    @patch('src.rag_engine.evaluation.evaluation_manager.DEEPEVAL_AVAILABLE', False)
    @patch('src.rag_engine.evaluation.evaluation_manager.RAGAS_AVAILABLE', False)
    def test_get_framework_info(self):
        """Test getting framework information"""
        with patch('src.rag_engine.evaluation.evaluation_manager.CustomEvaluator') as mock_custom:
            mock_evaluator = Mock()
            mock_evaluator.get_supported_metrics.return_value = ["faithfulness", "correctness"]
            mock_custom.return_value = mock_evaluator
            
            manager = EvaluationManager(enable_deepeval=False, enable_ragas=False)
            info = manager.get_framework_info()
            
            assert "custom" in info
            assert info["custom"]["name"] == "custom"
            assert info["custom"]["available"] == True
            assert info["custom"]["supported_metrics"] == ["faithfulness", "correctness"]
    
    @patch('src.rag_engine.evaluation.evaluation_manager.DEEPEVAL_AVAILABLE', False)
    @patch('src.rag_engine.evaluation.evaluation_manager.RAGAS_AVAILABLE', False)
    def test_evaluate_comprehensive_mismatched_lengths(self):
        """Test comprehensive evaluation with mismatched test cases and responses"""
        with patch('src.rag_engine.evaluation.evaluation_manager.CustomEvaluator') as mock_custom:
            mock_custom.return_value = Mock()
            
            manager = EvaluationManager(enable_deepeval=False, enable_ragas=False)
            
            test_cases = [TestCase(question="Test?", expected_answer="Answer")]
            responses = []  # Empty responses
            
            with pytest.raises(ValueError, match="Number of test cases must match number of responses"):
                manager.evaluate_comprehensive(test_cases, responses)
    
    @patch('src.rag_engine.evaluation.evaluation_manager.DEEPEVAL_AVAILABLE', False)
    @patch('src.rag_engine.evaluation.evaluation_manager.RAGAS_AVAILABLE', False)
    def test_evaluate_comprehensive_invalid_frameworks(self):
        """Test comprehensive evaluation with invalid framework names"""
        with patch('src.rag_engine.evaluation.evaluation_manager.CustomEvaluator') as mock_custom:
            mock_custom.return_value = Mock()
            
            manager = EvaluationManager(enable_deepeval=False, enable_ragas=False)
            
            test_cases = [TestCase(question="Test?", expected_answer="Answer")]
            responses = [RAGResponse(answer="Response", source_documents=[])]
            
            with pytest.raises(ValueError, match="Invalid frameworks requested"):
                manager.evaluate_comprehensive(test_cases, responses, frameworks=["invalid"])
    
    @patch('src.rag_engine.evaluation.evaluation_manager.DEEPEVAL_AVAILABLE', False)
    @patch('src.rag_engine.evaluation.evaluation_manager.RAGAS_AVAILABLE', False)
    def test_evaluate_comprehensive_success(self):
        """Test successful comprehensive evaluation"""
        with patch('src.rag_engine.evaluation.evaluation_manager.CustomEvaluator') as mock_custom:
            mock_evaluator = Mock()
            mock_result = EvaluationResult(
                overall_score=0.8,
                metric_scores={"faithfulness": 0.9, "correctness": 0.7},
                test_case_results=[{
                    "question": "Test?",
                    "expected_answer": "Answer",
                    "generated_answer": "Response",
                    "metrics": {
                        "faithfulness": {"score": 0.9, "reasoning": "Good", "metric": "faithfulness"}
                    }
                }],
                recommendations=["Good performance"]
            )
            mock_evaluator.evaluate.return_value = mock_result
            mock_custom.return_value = mock_evaluator
            
            manager = EvaluationManager(enable_deepeval=False, enable_ragas=False)
            
            test_cases = [TestCase(question="Test?", expected_answer="Answer")]
            responses = [RAGResponse(answer="Response", source_documents=[])]
            
            result = manager.evaluate_comprehensive(test_cases, responses)
            
            assert "evaluation_summary" in result
            assert "aggregated_metrics" in result
            assert "framework_comparison" in result
            assert "test_case_analysis" in result
            assert "performance_analysis" in result
            assert "recommendations" in result
            assert "detailed_results" in result
            
            # Check evaluation summary
            assert result["evaluation_summary"]["total_test_cases"] == 1
            assert result["evaluation_summary"]["frameworks_used"] == ["custom"]
            assert result["evaluation_summary"]["overall_score"] > 0
            
            # Check aggregated metrics
            assert "faithfulness" in result["aggregated_metrics"]
            assert "correctness" in result["aggregated_metrics"]
            
            # Check framework comparison
            assert "custom" in result["framework_comparison"]
            assert result["framework_comparison"]["custom"]["status"] == "success"
    
    @patch('src.rag_engine.evaluation.evaluation_manager.DEEPEVAL_AVAILABLE', False)
    @patch('src.rag_engine.evaluation.evaluation_manager.RAGAS_AVAILABLE', False)
    def test_evaluate_comprehensive_with_error(self):
        """Test comprehensive evaluation when framework fails"""
        with patch('src.rag_engine.evaluation.evaluation_manager.CustomEvaluator') as mock_custom:
            mock_evaluator = Mock()
            mock_evaluator.evaluate.side_effect = Exception("Evaluation failed")
            mock_custom.return_value = mock_evaluator
            
            manager = EvaluationManager(enable_deepeval=False, enable_ragas=False)
            
            test_cases = [TestCase(question="Test?", expected_answer="Answer")]
            responses = [RAGResponse(answer="Response", source_documents=[])]
            
            result = manager.evaluate_comprehensive(test_cases, responses)
            
            # Should still return a result even with errors
            assert "evaluation_summary" in result
            assert "framework_comparison" in result
            
            # Check that error is recorded
            assert "custom" in result["framework_comparison"]
            assert result["framework_comparison"]["custom"]["status"] == "error"
    
    @patch('src.rag_engine.evaluation.evaluation_manager.DEEPEVAL_AVAILABLE', False)
    @patch('src.rag_engine.evaluation.evaluation_manager.RAGAS_AVAILABLE', False)
    def test_generate_comprehensive_recommendations_low_scores(self):
        """Test recommendation generation for low scores"""
        with patch('src.rag_engine.evaluation.evaluation_manager.CustomEvaluator') as mock_custom:
            mock_custom.return_value = Mock()
            
            manager = EvaluationManager(enable_deepeval=False, enable_ragas=False)
            
            aggregated_metrics = {
                "faithfulness": 0.5,
                "correctness": 0.4,
                "relevancy": 0.3
            }
            
            framework_comparison = {
                "custom": {"status": "success", "overall_score": 0.4}
            }
            
            recommendations = manager._generate_comprehensive_recommendations(
                aggregated_metrics, framework_comparison
            )
            
            assert len(recommendations) > 0
            assert any("overall system performance" in rec.lower() for rec in recommendations)
            assert any("faithfulness" in rec.lower() for rec in recommendations)
            assert any("correctness" in rec.lower() for rec in recommendations)
    
    @patch('src.rag_engine.evaluation.evaluation_manager.DEEPEVAL_AVAILABLE', False)
    @patch('src.rag_engine.evaluation.evaluation_manager.RAGAS_AVAILABLE', False)
    def test_generate_comprehensive_recommendations_high_scores(self):
        """Test recommendation generation for high scores"""
        with patch('src.rag_engine.evaluation.evaluation_manager.CustomEvaluator') as mock_custom:
            mock_custom.return_value = Mock()
            
            manager = EvaluationManager(enable_deepeval=False, enable_ragas=False)
            
            aggregated_metrics = {
                "faithfulness": 0.9,
                "correctness": 0.9,
                "relevancy": 0.9
            }
            
            framework_comparison = {
                "custom": {"status": "success", "overall_score": 0.9}
            }
            
            recommendations = manager._generate_comprehensive_recommendations(
                aggregated_metrics, framework_comparison
            )
            
            assert len(recommendations) > 0
            assert any("excellent" in rec.lower() for rec in recommendations)
    
    @patch('src.rag_engine.evaluation.evaluation_manager.DEEPEVAL_AVAILABLE', False)
    @patch('src.rag_engine.evaluation.evaluation_manager.RAGAS_AVAILABLE', False)
    def test_analyze_test_cases(self):
        """Test test case analysis"""
        with patch('src.rag_engine.evaluation.evaluation_manager.CustomEvaluator') as mock_custom:
            mock_custom.return_value = Mock()
            
            manager = EvaluationManager(enable_deepeval=False, enable_ragas=False)
            
            # Create mock collector with results
            collector = MetricsCollector()
            result = EvaluationResult(
                overall_score=0.8,
                metric_scores={"faithfulness": 0.9},
                test_case_results=[{
                    "question": "Test?",
                    "expected_answer": "Answer",
                    "generated_answer": "Response",
                    "metrics": {
                        "faithfulness": {"score": 0.9, "reasoning": "Good", "metric": "faithfulness"}
                    }
                }],
                recommendations=[]
            )
            collector.add_result("custom", result, 1.0)
            
            test_cases = [TestCase(question="Test?", expected_answer="Answer")]
            responses = [RAGResponse(answer="Response", source_documents=[])]
            
            analysis = manager._analyze_test_cases(collector, test_cases, responses)
            
            assert "total_cases" in analysis
            assert "case_performance" in analysis
            assert "worst_performing_cases" in analysis
            assert "best_performing_cases" in analysis
            
            assert analysis["total_cases"] == 1
            assert len(analysis["case_performance"]) == 1
            
            case_perf = analysis["case_performance"][0]
            assert case_perf["question"] == "Test?"
            assert case_perf["expected_answer"] == "Answer"
            assert case_perf["generated_answer"] == "Response"
            assert case_perf["average_score"] > 0
    
    @patch('src.rag_engine.evaluation.evaluation_manager.DEEPEVAL_AVAILABLE', False)
    @patch('src.rag_engine.evaluation.evaluation_manager.RAGAS_AVAILABLE', False)
    def test_analyze_performance(self):
        """Test performance analysis"""
        with patch('src.rag_engine.evaluation.evaluation_manager.CustomEvaluator') as mock_custom:
            mock_custom.return_value = Mock()
            
            manager = EvaluationManager(enable_deepeval=False, enable_ragas=False)
            
            # Create mock collector with results
            collector = MetricsCollector()
            result = EvaluationResult(
                overall_score=0.8,
                metric_scores={"faithfulness": 0.9},
                test_case_results=[{"test": "case"}],
                recommendations=[]
            )
            collector.add_result("custom", result, 1.5)
            collector.add_error("deepeval", "Failed", 0.5)
            
            analysis = manager._analyze_performance(collector, 2.0)
            
            assert "total_execution_time" in analysis
            assert "framework_performance" in analysis
            assert "efficiency_ranking" in analysis
            
            assert analysis["total_execution_time"] == 2.0
            
            # Check framework performance
            assert "custom" in analysis["framework_performance"]
            assert "deepeval" in analysis["framework_performance"]
            
            custom_perf = analysis["framework_performance"]["custom"]
            assert custom_perf["status"] == "success"
            assert custom_perf["execution_time"] == 1.5
            assert custom_perf["overall_score"] == 0.8
            assert custom_perf["efficiency"] > 0
            
            deepeval_perf = analysis["framework_performance"]["deepeval"]
            assert deepeval_perf["status"] == "error"
            assert deepeval_perf["execution_time"] == 0.5
            assert deepeval_perf["overall_score"] == 0.0
            assert deepeval_perf["efficiency"] == 0.0
            
            # Check efficiency ranking
            assert len(analysis["efficiency_ranking"]) == 1  # Only successful frameworks
            assert analysis["efficiency_ranking"][0][0] == "custom"


if __name__ == "__main__":
    pytest.main([__file__])