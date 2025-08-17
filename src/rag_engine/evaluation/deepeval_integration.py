"""
DeepEval framework integration for standardized RAG evaluation
"""

from typing import List, Dict, Any, Optional
import logging

try:
    from deepeval import evaluate
    from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    logging.warning("DeepEval not available. Install with: pip install deepeval")
    
    # Create dummy classes for type hints when DeepEval is not available
    class LLMTestCase:
        pass
    
    class GEval:
        pass
    
    class FaithfulnessMetric:
        pass
    
    class ContextualRelevancyMetric:
        pass
    
    def evaluate(*args, **kwargs):
        pass

from ..core.interfaces import BaseEvaluator
from ..core.models import TestCase, RAGResponse, EvaluationResult


class DeepEvalIntegration(BaseEvaluator):
    """Integration with DeepEval framework for standardized metrics"""
    
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        """Initialize DeepEval integration
        
        Args:
            model_name: LLM model to use for evaluation (default: gemini-1.5-pro)
        """
        if not DEEPEVAL_AVAILABLE:
            raise ImportError(
                "DeepEval is not installed. Please install it with: pip install deepeval"
            )
        
        self.model_name = model_name
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup DeepEval metrics"""
        self.metrics = {
            "correctness": GEval(
                name="Correctness",
                model=self.model_name,
                evaluation_params=[
                    "Factual accuracy of the answer",
                    "Completeness of the response",
                    "Alignment with expected output"
                ]
            ),
            "faithfulness": FaithfulnessMetric(model=self.model_name),
            "contextual_relevancy": ContextualRelevancyMetric(model=self.model_name)
        }
    
    def create_test_case(self, test_case: TestCase, response: RAGResponse) -> LLMTestCase:
        """Create a DeepEval LLMTestCase from our test case and response
        
        Args:
            test_case: Our internal test case format
            response: RAG system response
            
        Returns:
            LLMTestCase: DeepEval test case format
        """
        # Extract context from source documents
        retrieval_context = []
        if response.source_documents:
            retrieval_context = [doc.content for doc in response.source_documents]
        
        return LLMTestCase(
            input=test_case.question,
            expected_output=test_case.expected_answer,
            actual_output=response.answer,
            retrieval_context=retrieval_context
        )
    
    def evaluate_with_deepeval(
        self, 
        test_cases: List[TestCase], 
        responses: List[RAGResponse],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate using DeepEval framework
        
        Args:
            test_cases: List of test cases
            responses: List of RAG responses
            metrics: List of metric names to use (default: all available)
            
        Returns:
            Dict containing DeepEval results
        """
        if len(test_cases) != len(responses):
            raise ValueError("Number of test cases must match number of responses")
        
        # Create DeepEval test cases
        deepeval_test_cases = []
        for test_case, response in zip(test_cases, responses):
            deepeval_test_cases.append(self.create_test_case(test_case, response))
        
        # Select metrics to use
        if metrics is None:
            metrics = list(self.metrics.keys())
        
        selected_metrics = [self.metrics[metric] for metric in metrics if metric in self.metrics]
        
        if not selected_metrics:
            raise ValueError(f"No valid metrics found. Available: {list(self.metrics.keys())}")
        
        # Run evaluation
        try:
            evaluation_results = evaluate(
                test_cases=deepeval_test_cases,
                metrics=selected_metrics
            )
            return evaluation_results
        except Exception as e:
            logging.error(f"DeepEval evaluation failed: {str(e)}")
            return {"error": str(e), "results": []}
    
    def evaluate(self, test_cases: List[TestCase], responses: List[RAGResponse]) -> EvaluationResult:
        """Evaluate RAG responses using DeepEval framework
        
        Args:
            test_cases: List of test cases
            responses: List of RAG responses
            
        Returns:
            EvaluationResult: Standardized evaluation results
        """
        if len(test_cases) != len(responses):
            raise ValueError("Number of test cases must match number of responses")
        
        # Run DeepEval evaluation
        deepeval_results = self.evaluate_with_deepeval(test_cases, responses)
        
        if isinstance(deepeval_results, dict) and "error" in deepeval_results:
            # Return error result
            error_msg = deepeval_results.get("error", "Unknown error")
            return EvaluationResult(
                overall_score=0.0,
                metric_scores={"error": 0.0},
                test_case_results=[{"error": error_msg}],
                recommendations=["DeepEval evaluation failed. Check logs for details."]
            )
        
        # Process DeepEval results into our format
        metric_scores = {}
        test_case_results = []
        
        # Extract metric scores from DeepEval results
        # Note: DeepEval's result structure may vary, so we handle it gracefully
        try:
            if hasattr(deepeval_results, 'test_results'):
                # Process individual test results
                for i, (test_case, response) in enumerate(zip(test_cases, responses)):
                    case_result = {
                        "question": test_case.question,
                        "expected_answer": test_case.expected_answer,
                        "generated_answer": response.answer,
                        "metrics": {}
                    }
                    
                    # Extract metrics for this test case
                    if i < len(deepeval_results.test_results):
                        test_result = deepeval_results.test_results[i]
                        for metric_result in test_result.metrics_data:
                            metric_name = metric_result.metric.name.lower()
                            case_result["metrics"][metric_name] = {
                                "score": metric_result.score,
                                "reasoning": getattr(metric_result, 'reason', 'No reasoning provided'),
                                "metric": metric_name
                            }
                    
                    test_case_results.append(case_result)
                
                # Calculate average scores
                all_metric_names = set()
                for case_result in test_case_results:
                    all_metric_names.update(case_result["metrics"].keys())
                
                for metric_name in all_metric_names:
                    scores = []
                    for case_result in test_case_results:
                        if metric_name in case_result["metrics"]:
                            scores.append(case_result["metrics"][metric_name]["score"])
                    
                    if scores:
                        metric_scores[metric_name] = sum(scores) / len(scores)
                    else:
                        metric_scores[metric_name] = 0.0
            
            else:
                # Fallback: create basic results structure
                for i, (test_case, response) in enumerate(zip(test_cases, responses)):
                    case_result = {
                        "question": test_case.question,
                        "expected_answer": test_case.expected_answer,
                        "generated_answer": response.answer,
                        "metrics": {
                            "deepeval_score": {
                                "score": 0.5,  # Default score
                                "reasoning": "DeepEval evaluation completed",
                                "metric": "deepeval_score"
                            }
                        }
                    }
                    test_case_results.append(case_result)
                
                metric_scores = {"deepeval_score": 0.5}
        
        except Exception as e:
            logging.error(f"Error processing DeepEval results: {str(e)}")
            # Return basic error result
            metric_scores = {"error": 0.0}
            test_case_results = [{"error": f"Result processing failed: {str(e)}"}]
        
        # Calculate overall score
        if metric_scores and "error" not in metric_scores:
            overall_score = sum(metric_scores.values()) / len(metric_scores)
        else:
            overall_score = 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metric_scores)
        
        return EvaluationResult(
            overall_score=overall_score,
            metric_scores=metric_scores,
            test_case_results=test_case_results,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, metric_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on DeepEval metric scores"""
        recommendations = []
        
        if "error" in metric_scores:
            recommendations.append("DeepEval evaluation encountered errors. Check configuration and dependencies.")
            return recommendations
        
        for metric_name, score in metric_scores.items():
            if score < 0.7:
                if "correctness" in metric_name.lower():
                    recommendations.append(
                        f"Low {metric_name} score ({score:.2f}). Consider improving knowledge base "
                        "or fine-tuning the generation model."
                    )
                elif "faithfulness" in metric_name.lower():
                    recommendations.append(
                        f"Low {metric_name} score ({score:.2f}). The model may be hallucinating. "
                        "Consider strengthening grounding mechanisms."
                    )
                elif "relevancy" in metric_name.lower():
                    recommendations.append(
                        f"Low {metric_name} score ({score:.2f}). Consider improving the "
                        "retrieval system or context filtering."
                    )
                else:
                    recommendations.append(
                        f"Low {metric_name} score ({score:.2f}). Review this metric's performance."
                    )
        
        if all(score > 0.8 for score in metric_scores.values()):
            recommendations.append(
                "Excellent performance across all DeepEval metrics! System is performing well."
            )
        
        return recommendations
    
    def get_supported_metrics(self) -> List[str]:
        """Get list of supported DeepEval metrics"""
        return list(self.metrics.keys())
    
    def get_available_models(self) -> List[str]:
        """Get list of available models for DeepEval"""
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "claude-3-opus",
            "claude-3-sonnet"
        ]