"""
Unit tests for RAGAS integration
"""

import pytest
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.evaluation.ragas_integration import RAGASIntegration, RAGAS_AVAILABLE
from src.rag_engine.core.models import TestCase, RAGResponse, Document


class TestRAGASIntegration:
    """Test the RAGASIntegration class"""
    
    def test_initialization_without_ragas(self):
        """Test initialization when RAGAS is not available"""
        with patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', False):
            with pytest.raises(ImportError, match="RAGAS is not installed"):
                RAGASIntegration()
    
    @patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.ragas_integration.faithfulness')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_relevancy')
    @patch('src.rag_engine.evaluation.ragas_integration.context_recall')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_correctness')
    def test_initialization_with_ragas(self, mock_correctness, mock_recall, mock_relevancy, mock_faithfulness):
        """Test successful initialization with RAGAS available"""
        integration = RAGASIntegration()
        
        assert hasattr(integration, 'metrics')
        assert "faithfulness" in integration.metrics
        assert "answer_relevancy" in integration.metrics
        assert "context_recall" in integration.metrics
        assert "answer_correctness" in integration.metrics
    
    @patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.ragas_integration.faithfulness')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_relevancy')
    @patch('src.rag_engine.evaluation.ragas_integration.context_recall')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_correctness')
    def test_get_supported_metrics(self, mock_correctness, mock_recall, mock_relevancy, mock_faithfulness):
        """Test getting supported metrics"""
        integration = RAGASIntegration()
        metrics = integration.get_supported_metrics()
        expected_metrics = ["faithfulness", "answer_relevancy", "context_recall", "answer_correctness"]
        assert set(metrics) == set(expected_metrics)
    
    @patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.ragas_integration.faithfulness')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_relevancy')
    @patch('src.rag_engine.evaluation.ragas_integration.context_recall')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_correctness')
    def test_get_metric_descriptions(self, mock_correctness, mock_recall, mock_relevancy, mock_faithfulness):
        """Test getting metric descriptions"""
        integration = RAGASIntegration()
        descriptions = integration.get_metric_descriptions()
        
        assert "faithfulness" in descriptions
        assert "answer_relevancy" in descriptions
        assert "context_recall" in descriptions
        assert "answer_correctness" in descriptions
        assert "hallucination" in descriptions["faithfulness"].lower()
    
    @patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.ragas_integration.faithfulness')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_relevancy')
    @patch('src.rag_engine.evaluation.ragas_integration.context_recall')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_correctness')
    @patch('src.rag_engine.evaluation.ragas_integration.Dataset')
    def test_prepare_dataset(self, mock_dataset, mock_correctness, mock_recall, mock_relevancy, mock_faithfulness):
        """Test preparing dataset in RAGAS format"""
        integration = RAGASIntegration()
        
        test_cases = [
            TestCase(
                question="What is the capital of France?",
                expected_answer="Paris"
            )
        ]
        
        responses = [
            RAGResponse(
                answer="Paris is the capital of France.",
                source_documents=[
                    Document(content="Paris is the capital of France.")
                ]
            )
        ]
        
        mock_dataset_instance = Mock()
        mock_dataset.from_dict.return_value = mock_dataset_instance
        
        result = integration.prepare_dataset(test_cases, responses)
        
        # Verify Dataset.from_dict was called with correct data
        mock_dataset.from_dict.assert_called_once()
        call_args = mock_dataset.from_dict.call_args[0][0]
        
        assert "question" in call_args
        assert "answer" in call_args
        assert "contexts" in call_args
        assert "ground_truth" in call_args
        
        assert call_args["question"] == ["What is the capital of France?"]
        assert call_args["answer"] == ["Paris is the capital of France."]
        assert call_args["contexts"] == [["Paris is the capital of France."]]
        assert call_args["ground_truth"] == ["Paris"]
        
        assert result == mock_dataset_instance
    
    @patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.ragas_integration.faithfulness')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_relevancy')
    @patch('src.rag_engine.evaluation.ragas_integration.context_recall')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_correctness')
    @patch('src.rag_engine.evaluation.ragas_integration.Dataset')
    def test_prepare_dataset_no_context(self, mock_dataset, mock_correctness, mock_recall, mock_relevancy, mock_faithfulness):
        """Test preparing dataset when response has no source documents"""
        integration = RAGASIntegration()
        
        test_cases = [
            TestCase(
                question="What is 2+2?",
                expected_answer="4"
            )
        ]
        
        responses = [
            RAGResponse(
                answer="4",
                source_documents=[]
            )
        ]
        
        mock_dataset_instance = Mock()
        mock_dataset.from_dict.return_value = mock_dataset_instance
        
        result = integration.prepare_dataset(test_cases, responses)
        
        # Verify contexts is empty list
        call_args = mock_dataset.from_dict.call_args[0][0]
        assert call_args["contexts"] == [[]]  # Empty context list
    
    @patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.ragas_integration.faithfulness')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_relevancy')
    @patch('src.rag_engine.evaluation.ragas_integration.context_recall')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_correctness')
    def test_prepare_dataset_mismatched_lengths(self, mock_correctness, mock_recall, mock_relevancy, mock_faithfulness):
        """Test preparing dataset with mismatched test cases and responses"""
        integration = RAGASIntegration()
        
        test_cases = [TestCase(question="Test?", expected_answer="Answer")]
        responses = []  # Empty responses
        
        with pytest.raises(ValueError, match="Number of test cases must match number of responses"):
            integration.prepare_dataset(test_cases, responses)
    
    @patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.ragas_integration.faithfulness')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_relevancy')
    @patch('src.rag_engine.evaluation.ragas_integration.context_recall')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_correctness')
    @patch('src.rag_engine.evaluation.ragas_integration.evaluate')
    def test_evaluate_with_ragas_success(self, mock_evaluate, mock_correctness, mock_recall, mock_relevancy, mock_faithfulness):
        """Test successful RAGAS evaluation"""
        integration = RAGASIntegration()
        
        mock_dataset = Mock()
        mock_results = Mock()
        mock_evaluate.return_value = mock_results
        
        result = integration.evaluate_with_ragas(mock_dataset)
        
        assert result == mock_results
        mock_evaluate.assert_called_once()
    
    @patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.ragas_integration.faithfulness')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_relevancy')
    @patch('src.rag_engine.evaluation.ragas_integration.context_recall')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_correctness')
    @patch('src.rag_engine.evaluation.ragas_integration.evaluate')
    def test_evaluate_with_ragas_error(self, mock_evaluate, mock_correctness, mock_recall, mock_relevancy, mock_faithfulness):
        """Test RAGAS evaluation with error"""
        integration = RAGASIntegration()
        
        mock_dataset = Mock()
        mock_evaluate.side_effect = Exception("RAGAS error")
        
        result = integration.evaluate_with_ragas(mock_dataset)
        
        assert "error" in result
        assert result["error"] == "RAGAS error"
    
    @patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.ragas_integration.faithfulness')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_relevancy')
    @patch('src.rag_engine.evaluation.ragas_integration.context_recall')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_correctness')
    def test_evaluate_mismatched_lengths(self, mock_correctness, mock_recall, mock_relevancy, mock_faithfulness):
        """Test evaluation with mismatched test cases and responses"""
        integration = RAGASIntegration()
        
        test_cases = [TestCase(question="Test?", expected_answer="Answer")]
        responses = []  # Empty responses
        
        with pytest.raises(ValueError, match="Number of test cases must match number of responses"):
            integration.evaluate(test_cases, responses)
    
    @patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.ragas_integration.faithfulness')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_relevancy')
    @patch('src.rag_engine.evaluation.ragas_integration.context_recall')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_correctness')
    def test_evaluate_dataset_preparation_error(self, mock_correctness, mock_recall, mock_relevancy, mock_faithfulness):
        """Test evaluation when dataset preparation fails"""
        integration = RAGASIntegration()
        
        test_cases = [
            TestCase(question="Test question?", expected_answer="Test answer")
        ]
        
        responses = [
            RAGResponse(answer="Test response", source_documents=[])
        ]
        
        # Mock prepare_dataset to raise an exception
        with patch.object(integration, 'prepare_dataset', side_effect=Exception("Dataset error")):
            result = integration.evaluate(test_cases, responses)
            
            assert result.overall_score == 0.0
            assert "error" in result.metric_scores
            assert "Dataset preparation failed" in result.test_case_results[0]["error"]
            assert "RAGAS dataset preparation failed" in result.recommendations[0]
    
    @patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.ragas_integration.faithfulness')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_relevancy')
    @patch('src.rag_engine.evaluation.ragas_integration.context_recall')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_correctness')
    def test_evaluate_with_error_result(self, mock_correctness, mock_recall, mock_relevancy, mock_faithfulness):
        """Test evaluation when RAGAS returns error"""
        integration = RAGASIntegration()
        
        test_cases = [
            TestCase(question="Test question?", expected_answer="Test answer")
        ]
        
        responses = [
            RAGResponse(answer="Test response", source_documents=[])
        ]
        
        # Mock both prepare_dataset and evaluate_with_ragas to return error
        with patch.object(integration, 'prepare_dataset', return_value=Mock()), \
             patch.object(integration, 'evaluate_with_ragas', return_value={"error": "Test error"}):
            result = integration.evaluate(test_cases, responses)
            
            assert result.overall_score == 0.0
            assert "error" in result.metric_scores
            assert "Test error" in result.test_case_results[0]["error"]
            assert "RAGAS evaluation failed" in result.recommendations[0]
    
    @patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.ragas_integration.faithfulness')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_relevancy')
    @patch('src.rag_engine.evaluation.ragas_integration.context_recall')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_correctness')
    def test_evaluate_successful_with_results(self, mock_correctness, mock_recall, mock_relevancy, mock_faithfulness):
        """Test successful evaluation with proper results processing"""
        integration = RAGASIntegration()
        
        test_cases = [
            TestCase(question="Test question?", expected_answer="Test answer")
        ]
        
        responses = [
            RAGResponse(answer="Test response", source_documents=[])
        ]
        
        # Mock RAGAS results structure with to_pandas method
        mock_ragas_results = Mock()
        
        # Use MagicMock for DataFrame to support magic methods
        from unittest.mock import MagicMock
        mock_df = MagicMock()
        mock_df.columns = ['faithfulness', 'answer_relevancy', 'context_recall', 'answer_correctness']
        mock_df.__len__.return_value = 1
        
        # Mock pandas series for each metric
        mock_series = Mock()
        mock_series.dropna.return_value = [0.8, 0.9]
        mock_series.mean.return_value = 0.85
        
        # Configure mock_df to return mock_series for any column access
        mock_df.__getitem__.return_value = mock_series
        
        # Mock iloc for individual row access
        mock_row = MagicMock()
        mock_row.__contains__.return_value = True
        mock_row.__getitem__.return_value = 0.8
        mock_df.iloc = [mock_row]  # Make it indexable
        
        mock_ragas_results.to_pandas.return_value = mock_df
        
        with patch.object(integration, 'prepare_dataset', return_value=Mock()), \
             patch.object(integration, 'evaluate_with_ragas', return_value=mock_ragas_results):
            result = integration.evaluate(test_cases, responses)
            
            assert result.overall_score > 0
            assert len(result.metric_scores) > 0
            assert len(result.test_case_results) == 1
    
    @patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.ragas_integration.faithfulness')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_relevancy')
    @patch('src.rag_engine.evaluation.ragas_integration.context_recall')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_correctness')
    def test_evaluate_fallback_processing(self, mock_correctness, mock_recall, mock_relevancy, mock_faithfulness):
        """Test evaluation with fallback result processing"""
        integration = RAGASIntegration()
        
        test_cases = [
            TestCase(question="Test question?", expected_answer="Test answer")
        ]
        
        responses = [
            RAGResponse(answer="Test response", source_documents=[])
        ]
        
        # Mock RAGAS results without to_pandas method
        mock_ragas_results = Mock()
        del mock_ragas_results.to_pandas  # Remove the attribute
        
        with patch.object(integration, 'prepare_dataset', return_value=Mock()), \
             patch.object(integration, 'evaluate_with_ragas', return_value=mock_ragas_results):
            result = integration.evaluate(test_cases, responses)
            
            assert result.overall_score == 0.5  # Default fallback score
            assert "ragas_score" in result.metric_scores
            assert result.metric_scores["ragas_score"] == 0.5
            assert len(result.test_case_results) == 1
    
    @patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.ragas_integration.faithfulness')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_relevancy')
    @patch('src.rag_engine.evaluation.ragas_integration.context_recall')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_correctness')
    def test_generate_recommendations_low_scores(self, mock_correctness, mock_recall, mock_relevancy, mock_faithfulness):
        """Test recommendation generation for low scores"""
        integration = RAGASIntegration()
        
        metric_scores = {
            "faithfulness": 0.5,
            "answer_relevancy": 0.6,
            "context_recall": 0.4,
            "answer_correctness": 0.3
        }
        
        recommendations = integration._generate_recommendations(metric_scores)
        
        assert len(recommendations) == 4  # All metrics are below threshold
        assert any("faithfulness" in rec.lower() for rec in recommendations)
        assert any("answer_relevancy" in rec.lower() for rec in recommendations)
        assert any("context_recall" in rec.lower() for rec in recommendations)
        assert any("answer_correctness" in rec.lower() for rec in recommendations)
    
    @patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.ragas_integration.faithfulness')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_relevancy')
    @patch('src.rag_engine.evaluation.ragas_integration.context_recall')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_correctness')
    def test_generate_recommendations_high_scores(self, mock_correctness, mock_recall, mock_relevancy, mock_faithfulness):
        """Test recommendation generation for high scores"""
        integration = RAGASIntegration()
        
        metric_scores = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.9,
            "context_recall": 0.9,
            "answer_correctness": 0.9
        }
        
        recommendations = integration._generate_recommendations(metric_scores)
        
        assert len(recommendations) == 1
        assert "excellent performance" in recommendations[0].lower()
    
    @patch('src.rag_engine.evaluation.ragas_integration.RAGAS_AVAILABLE', True)
    @patch('src.rag_engine.evaluation.ragas_integration.faithfulness')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_relevancy')
    @patch('src.rag_engine.evaluation.ragas_integration.context_recall')
    @patch('src.rag_engine.evaluation.ragas_integration.answer_correctness')
    def test_generate_recommendations_with_error(self, mock_correctness, mock_recall, mock_relevancy, mock_faithfulness):
        """Test recommendation generation when there's an error"""
        integration = RAGASIntegration()
        
        metric_scores = {"error": 0.0}
        
        recommendations = integration._generate_recommendations(metric_scores)
        
        assert len(recommendations) == 1
        assert "ragas evaluation encountered errors" in recommendations[0].lower()


@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="RAGAS not installed")
class TestRAGASIntegrationReal:
    """Test RAGAS integration with real RAGAS (if available)"""
    
    def test_real_initialization(self):
        """Test real initialization if RAGAS is available"""
        try:
            integration = RAGASIntegration()
            assert integration is not None
            assert len(integration.get_supported_metrics()) > 0
        except ImportError:
            pytest.skip("RAGAS not available for real testing")


if __name__ == "__main__":
    pytest.main([__file__])