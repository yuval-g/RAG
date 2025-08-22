"""
Unit tests for DeepEval integration
"""

import pytest
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.evaluation.deepeval_integration import DeepEvalIntegration, DEEPEVAL_AVAILABLE
from src.rag_engine.core.models import TestCase, RAGResponse, Document


class TestDeepEvalIntegration:
    """Test the DeepEvalIntegration class"""
    
    def test_initialization_without_deepeval(self):
        """Test initialization when DeepEval is not available"""
        with patch('src.rag_engine.evaluation.deepeval_integration.DEEPEVAL_AVAILABLE', False):
            with pytest.raises(ImportError, match="DeepEval is not installed"):
                DeepEvalIntegration()
    
    @patch('src.rag_engine.evaluation.deepeval_integration.DEEPEVAL_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.deepeval_integration.GEval')
    @patch('src.rag_engine.evaluation.deepeval_integration.FaithfulnessMetric')
    @patch('src.rag_engine.evaluation.deepeval_integration.ContextualRelevancyMetric')
    def test_initialization_with_deepeval(self, mock_relevancy, mock_faith, mock_geval):
        """Test successful initialization with DeepEval available"""
        integration = DeepEvalIntegration(model_name="gpt-4")
        
        assert integration.model_name == "gpt-4"
        assert hasattr(integration, 'metrics')
        assert "correctness" in integration.metrics
        assert "faithfulness" in integration.metrics
        assert "contextual_relevancy" in integration.metrics
    
    @patch('src.rag_engine.evaluation.deepeval_integration.DEEPEVAL_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.deepeval_integration.GEval')
    @patch('src.rag_engine.evaluation.deepeval_integration.FaithfulnessMetric')
    @patch('src.rag_engine.evaluation.deepeval_integration.ContextualRelevancyMetric')
    def test_get_supported_metrics(self, mock_relevancy, mock_faith, mock_geval):
        """Test getting supported metrics"""
        integration = DeepEvalIntegration()
        metrics = integration.get_supported_metrics()
        expected_metrics = ["correctness", "faithfulness", "contextual_relevancy"]
        assert set(metrics) == set(expected_metrics)
    
    @patch('src.rag_engine.evaluation.deepeval_integration.DEEPEVAL_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.deepeval_integration.GEval')
    @patch('src.rag_engine.evaluation.deepeval_integration.FaithfulnessMetric')
    @patch('src.rag_engine.evaluation.deepeval_integration.ContextualRelevancyMetric')
    def test_get_available_models(self, mock_relevancy, mock_faith, mock_geval):
        """Test getting available models"""
        integration = DeepEvalIntegration()
        models = integration.get_available_models()
        assert "gpt-4" in models
        assert "gemini-2.0-flash-lite" in models
        assert "claude-3-opus" in models
    
    @patch('src.rag_engine.evaluation.deepeval_integration.DEEPEVAL_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.deepeval_integration.GEval')
    @patch('src.rag_engine.evaluation.deepeval_integration.FaithfulnessMetric')
    @patch('src.rag_engine.evaluation.deepeval_integration.ContextualRelevancyMetric')
    @patch('src.rag_engine.evaluation.deepeval_integration.LLMTestCase')
    def test_create_test_case(self, mock_test_case, mock_relevancy, mock_faith, mock_geval):
        """Test creating DeepEval test case from our format"""
        integration = DeepEvalIntegration()
        
        test_case = TestCase(
            question="What is the capital of France?",
            expected_answer="Paris"
        )
        
        response = RAGResponse(
            answer="Paris is the capital of France.",
            source_documents=[
                Document(content="Paris is the capital of France.")
            ]
        )
        
        mock_llm_test_case = Mock()
        mock_test_case.return_value = mock_llm_test_case
        
        result = integration.create_test_case(test_case, response)
        
        # Verify LLMTestCase was called with correct parameters
        mock_test_case.assert_called_once_with(
            input="What is the capital of France?",
            expected_output="Paris",
            actual_output="Paris is the capital of France.",
            retrieval_context=["Paris is the capital of France."]
        )
        
        assert result == mock_llm_test_case
    
    @patch('src.rag_engine.evaluation.deepeval_integration.DEEPEVAL_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.deepeval_integration.GEval')
    @patch('src.rag_engine.evaluation.deepeval_integration.FaithfulnessMetric')
    @patch('src.rag_engine.evaluation.deepeval_integration.ContextualRelevancyMetric')
    @patch('src.rag_engine.evaluation.deepeval_integration.LLMTestCase')
    def test_create_test_case_no_context(self, mock_test_case, mock_relevancy, mock_faith, mock_geval):
        """Test creating test case when response has no source documents"""
        integration = DeepEvalIntegration()
        
        test_case = TestCase(
            question="What is 2+2?",
            expected_answer="4"
        )
        
        response = RAGResponse(
            answer="4",
            source_documents=[]
        )
        
        mock_llm_test_case = Mock()
        mock_test_case.return_value = mock_llm_test_case
        
        result = integration.create_test_case(test_case, response)
        
        # Should have empty retrieval context
        mock_test_case.assert_called_once_with(
            input="What is 2+2?",
            expected_output="4",
            actual_output="4",
            retrieval_context=[]
        )
    
    @patch('src.rag_engine.evaluation.deepeval_integration.DEEPEVAL_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.deepeval_integration.GEval')
    @patch('src.rag_engine.evaluation.deepeval_integration.FaithfulnessMetric')
    @patch('src.rag_engine.evaluation.deepeval_integration.ContextualRelevancyMetric')
    def test_evaluate_mismatched_lengths(self, mock_relevancy, mock_faith, mock_geval):
        """Test evaluation with mismatched test cases and responses"""
        integration = DeepEvalIntegration()
        test_cases = [TestCase(question="Test?", expected_answer="Answer")]
        responses = []  # Empty responses
        
        with pytest.raises(ValueError, match="Number of test cases must match number of responses"):
            integration.evaluate(test_cases, responses)
    
    @patch('src.rag_engine.evaluation.deepeval_integration.DEEPEVAL_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.deepeval_integration.GEval')
    @patch('src.rag_engine.evaluation.deepeval_integration.FaithfulnessMetric')
    @patch('src.rag_engine.evaluation.deepeval_integration.ContextualRelevancyMetric')
    def test_evaluate_with_error_result(self, mock_relevancy, mock_faith, mock_geval):
        """Test evaluation when DeepEval returns error"""
        integration = DeepEvalIntegration()
        
        test_cases = [
            TestCase(question="Test question?", expected_answer="Test answer")
        ]
        
        responses = [
            RAGResponse(answer="Test response", source_documents=[])
        ]
        
        # Mock evaluate_with_deepeval to return error
        with patch.object(integration, 'evaluate_with_deepeval', return_value={"error": "Test error"}):
            result = integration.evaluate(test_cases, responses)
            
            assert result.overall_score == 0.0
            assert "error" in result.metric_scores
            assert len(result.test_case_results) == 1
            assert "error" in result.test_case_results[0]
            assert "DeepEval evaluation failed" in result.recommendations[0]
    
    @patch('src.rag_engine.evaluation.deepeval_integration.DEEPEVAL_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.deepeval_integration.GEval')
    @patch('src.rag_engine.evaluation.deepeval_integration.FaithfulnessMetric')
    @patch('src.rag_engine.evaluation.deepeval_integration.ContextualRelevancyMetric')
    def test_evaluate_successful_with_results(self, mock_relevancy, mock_faith, mock_geval):
        """Test successful evaluation with proper results processing"""
        integration = DeepEvalIntegration()
        
        test_cases = [
            TestCase(question="Test question?", expected_answer="Test answer")
        ]
        
        responses = [
            RAGResponse(answer="Test response", source_documents=[])
        ]
        
        # Mock DeepEval results structure
        mock_metric_result = Mock()
        mock_metric_result.metric.name = "Correctness"
        mock_metric_result.score = 0.8
        mock_metric_result.reason = "Good answer"
        
        mock_test_result = Mock()
        mock_test_result.metrics_data = [mock_metric_result]
        
        mock_deepeval_results = Mock()
        mock_deepeval_results.test_results = [mock_test_result]
        
        with patch.object(integration, 'evaluate_with_deepeval', return_value=mock_deepeval_results):
            result = integration.evaluate(test_cases, responses)
            
            assert result.overall_score > 0
            assert "correctness" in result.metric_scores
            assert result.metric_scores["correctness"] == 0.8
            assert len(result.test_case_results) == 1
            assert "correctness" in result.test_case_results[0]["metrics"]
    
    @patch('src.rag_engine.evaluation.deepeval_integration.DEEPEVAL_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.deepeval_integration.GEval')
    @patch('src.rag_engine.evaluation.deepeval_integration.FaithfulnessMetric')
    @patch('src.rag_engine.evaluation.deepeval_integration.ContextualRelevancyMetric')
    def test_evaluate_fallback_processing(self, mock_relevancy, mock_faith, mock_geval):
        """Test evaluation with fallback result processing"""
        integration = DeepEvalIntegration()
        
        test_cases = [
            TestCase(question="Test question?", expected_answer="Test answer")
        ]
        
        responses = [
            RAGResponse(answer="Test response", source_documents=[])
        ]
        
        # Mock DeepEval results without test_results attribute
        mock_deepeval_results = Mock()
        del mock_deepeval_results.test_results  # Remove the attribute
        
        with patch.object(integration, 'evaluate_with_deepeval', return_value=mock_deepeval_results):
            result = integration.evaluate(test_cases, responses)
            
            assert result.overall_score == 0.5  # Default fallback score
            assert "deepeval_score" in result.metric_scores
            assert result.metric_scores["deepeval_score"] == 0.5
            assert len(result.test_case_results) == 1
    
    @patch('src.rag_engine.evaluation.deepeval_integration.DEEPEVAL_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.deepeval_integration.GEval')
    @patch('src.rag_engine.evaluation.deepeval_integration.FaithfulnessMetric')
    @patch('src.rag_engine.evaluation.deepeval_integration.ContextualRelevancyMetric')
    def test_generate_recommendations_low_scores(self, mock_relevancy, mock_faith, mock_geval):
        """Test recommendation generation for low scores"""
        integration = DeepEvalIntegration()
        
        metric_scores = {
            "correctness": 0.5,
            "faithfulness": 0.6,
            "contextual_relevancy": 0.4
        }
        
        recommendations = integration._generate_recommendations(metric_scores)
        
        assert len(recommendations) == 3  # All metrics are below threshold
        assert any("correctness" in rec.lower() for rec in recommendations)
        assert any("faithfulness" in rec.lower() for rec in recommendations)
        assert any("relevancy" in rec.lower() for rec in recommendations)
    
    @patch('src.rag_engine.evaluation.deepeval_integration.DEEPEVAL_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.deepeval_integration.GEval')
    @patch('src.rag_engine.evaluation.deepeval_integration.FaithfulnessMetric')
    @patch('src.rag_engine.evaluation.deepeval_integration.ContextualRelevancyMetric')
    def test_generate_recommendations_high_scores(self, mock_relevancy, mock_faith, mock_geval):
        """Test recommendation generation for high scores"""
        integration = DeepEvalIntegration()
        
        metric_scores = {
            "correctness": 0.9,
            "faithfulness": 0.9,
            "contextual_relevancy": 0.9
        }
        
        recommendations = integration._generate_recommendations(metric_scores)
        
        assert len(recommendations) == 1
        assert "excellent performance" in recommendations[0].lower()
    
    @patch('src.rag_engine.evaluation.deepeval_integration.DEEPEVAL_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.deepeval_integration.GEval')
    @patch('src.rag_engine.evaluation.deepeval_integration.FaithfulnessMetric')
    @patch('src.rag_engine.evaluation.deepeval_integration.ContextualRelevancyMetric')
    def test_generate_recommendations_with_error(self, mock_relevancy, mock_faith, mock_geval):
        """Test recommendation generation when there's an error"""
        integration = DeepEvalIntegration()
        
        metric_scores = {"error": 0.0}
        
        recommendations = integration._generate_recommendations(metric_scores)
        
        assert len(recommendations) == 1
        assert "deepeval evaluation encountered errors" in recommendations[0].lower()


@pytest.mark.skipif(not DEEPEVAL_AVAILABLE, reason="DeepEval not installed")
class TestDeepEvalIntegrationReal:
    """Test DeepEval integration with real DeepEval (if available)"""
    
    def test_real_initialization(self):
        """Test real initialization if DeepEval is available"""
        try:
            integration = DeepEvalIntegration()
            assert integration is not None
            assert len(integration.get_supported_metrics()) > 0
        except ImportError:
            pytest.skip("DeepEval not available for real testing")


if __name__ == "__main__":
    pytest.main([__file__])