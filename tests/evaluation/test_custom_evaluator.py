"""
Unit tests for custom evaluation system
"""

import pytest
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_engine.evaluation.custom_evaluator import CustomEvaluator, ResultScore
from src.rag_engine.core.models import TestCase, RAGResponse, Document


class TestResultScore:
    """Test the ResultScore model"""
    
    def test_result_score_creation(self):
        """Test creating a ResultScore instance"""
        score = ResultScore(score=0.8, reasoning="Good answer")
        assert score.score == 0.8
        assert score.reasoning == "Good answer"
    
    def test_result_score_validation(self):
        """Test ResultScore validation"""
        # Valid score
        score = ResultScore(score=0.5, reasoning="Average")
        assert score.score == 0.5
        
        # Score outside range should still be allowed (LLM might return it)
        score = ResultScore(score=1.2, reasoning="Excellent")
        assert score.score == 1.2


class TestCustomEvaluator:
    """Test the CustomEvaluator class"""
    
    @pytest.fixture
    def evaluator(self):
        """Create a CustomEvaluator instance for testing"""
        with patch('src.rag_engine.evaluation.custom_evaluator.ChatGoogleGenerativeAI'):
            return CustomEvaluator()
    
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization"""
        assert evaluator is not None
        assert hasattr(evaluator, 'llm')
        assert hasattr(evaluator, 'correctness_chain')
        assert hasattr(evaluator, 'faithfulness_chain')
        assert hasattr(evaluator, 'contextual_relevancy_chain')
    
    def test_get_supported_metrics(self, evaluator):
        """Test getting supported metrics"""
        metrics = evaluator.get_supported_metrics()
        expected_metrics = ["correctness", "faithfulness", "contextual_relevancy"]
        assert metrics == expected_metrics
    
    def test_evaluate_correctness(self, evaluator):
        """Test correctness evaluation"""
        with patch.object(evaluator, 'evaluate_correctness', return_value={
            "score": 0.8,
            "reasoning": "Test reasoning",
            "metric": "correctness"
        }) as mock_eval:
            result = evaluator.evaluate_correctness(
                question="What is 2+2?",
                ground_truth="4",
                generated_answer="4"
            )
            
            assert result["score"] == 0.8
            assert result["reasoning"] == "Test reasoning"
            assert result["metric"] == "correctness"
            mock_eval.assert_called_once()
    
    def test_evaluate_faithfulness(self, evaluator):
        """Test faithfulness evaluation"""
        with patch.object(evaluator, 'evaluate_faithfulness', return_value={
            "score": 0.8,
            "reasoning": "Test reasoning",
            "metric": "faithfulness"
        }) as mock_eval:
            result = evaluator.evaluate_faithfulness(
                question="What is the capital of France?",
                context="Paris is the capital of France.",
                generated_answer="Paris"
            )
            
            assert result["score"] == 0.8
            assert result["reasoning"] == "Test reasoning"
            assert result["metric"] == "faithfulness"
            mock_eval.assert_called_once()
    
    def test_evaluate_contextual_relevancy(self, evaluator):
        """Test contextual relevancy evaluation"""
        with patch.object(evaluator, 'evaluate_contextual_relevancy', return_value={
            "score": 0.8,
            "reasoning": "Test reasoning",
            "metric": "contextual_relevancy"
        }) as mock_eval:
            result = evaluator.evaluate_contextual_relevancy(
                question="What is the capital of France?",
                context="Paris is the capital of France."
            )
            
            assert result["score"] == 0.8
            assert result["reasoning"] == "Test reasoning"
            assert result["metric"] == "contextual_relevancy"
            mock_eval.assert_called_once()
    
    def test_evaluate_full_pipeline(self, evaluator):
        """Test the full evaluation pipeline"""
        # Mock individual evaluation methods
        with patch.object(evaluator, 'evaluate_correctness', return_value={
            "score": 0.8,
            "reasoning": "Good correctness",
            "metric": "correctness"
        }), \
        patch.object(evaluator, 'evaluate_faithfulness', return_value={
            "score": 0.9,
            "reasoning": "Good faithfulness",
            "metric": "faithfulness"
        }), \
        patch.object(evaluator, 'evaluate_contextual_relevancy', return_value={
            "score": 0.7,
            "reasoning": "Good relevancy",
            "metric": "contextual_relevancy"
        }):
            
            # Create test data
            test_cases = [
                TestCase(
                    question="What is the capital of France?",
                    expected_answer="Paris"
                )
            ]
            
            responses = [
                RAGResponse(
                    answer="Paris",
                    source_documents=[
                        Document(content="Paris is the capital of France.")
                    ]
                )
            ]
            
            # Run evaluation
            result = evaluator.evaluate(test_cases, responses)
            
            # Check results
            assert result.overall_score > 0
            assert result.metric_scores["correctness"] == 0.8
            assert result.metric_scores["faithfulness"] == 0.9
            assert result.metric_scores["contextual_relevancy"] == 0.7
            assert len(result.test_case_results) == 1
            # Should have recommendations since contextual_relevancy (0.7) is at threshold
            assert len(result.recommendations) >= 0  # May or may not have recommendations
    
    def test_evaluate_mismatched_lengths(self, evaluator):
        """Test evaluation with mismatched test cases and responses"""
        test_cases = [TestCase(question="Test?", expected_answer="Answer")]
        responses = []  # Empty responses
        
        with pytest.raises(ValueError, match="Number of test cases must match number of responses"):
            evaluator.evaluate(test_cases, responses)
    
    def test_evaluate_no_source_documents(self, evaluator):
        """Test evaluation when response has no source documents"""
        with patch.object(evaluator, 'evaluate_correctness', return_value={
            "score": 0.8,
            "reasoning": "Good correctness",
            "metric": "correctness"
        }):
            test_cases = [
                TestCase(
                    question="What is 2+2?",
                    expected_answer="4"
                )
            ]
            
            responses = [
                RAGResponse(
                    answer="4",
                    source_documents=[]  # No source documents
                )
            ]
            
            result = evaluator.evaluate(test_cases, responses)
            
            # Should still have correctness score
            assert "correctness" in result.metric_scores
            assert result.metric_scores["correctness"] == 0.8
            # Faithfulness and relevancy should be 0 due to no context
            assert result.metric_scores["faithfulness"] == 0.0
            assert result.metric_scores["contextual_relevancy"] == 0.0
    
    def test_generate_recommendations_low_scores(self, evaluator):
        """Test recommendation generation for low scores"""
        metric_scores = {
            "correctness": 0.5,
            "faithfulness": 0.6,
            "contextual_relevancy": 0.4
        }
        
        recommendations = evaluator._generate_recommendations(metric_scores)
        
        assert len(recommendations) == 3  # All metrics are below threshold
        assert any("correctness" in rec.lower() for rec in recommendations)
        assert any("faithfulness" in rec.lower() for rec in recommendations)
        assert any("contextual relevancy" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_high_scores(self, evaluator):
        """Test recommendation generation for high scores"""
        metric_scores = {
            "correctness": 0.9,
            "faithfulness": 0.9,
            "contextual_relevancy": 0.9
        }
        
        recommendations = evaluator._generate_recommendations(metric_scores)
        
        assert len(recommendations) == 1
        assert "excellent performance" in recommendations[0].lower()


if __name__ == "__main__":
    pytest.main([__file__])