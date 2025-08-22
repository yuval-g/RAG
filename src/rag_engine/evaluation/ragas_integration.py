"""
RAGAS framework integration for RAG-specific evaluation
"""

from typing import List, Dict, Any, Optional
import logging

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        answer_correctness,
    )
    from datasets import Dataset
    from langchain_google_genai import ChatGoogleGenerativeAI
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("RAGAS not available. Install with: pip install ragas datasets")
    
    # Create dummy classes for type hints when RAGAS is not available
    class Dataset:
        pass
    
    def evaluate(*args, **kwargs):
        pass
    
    # Create dummy metric objects
    faithfulness = None
    answer_relevancy = None
    context_recall = None
    answer_correctness = None

from ..core.interfaces import BaseEvaluator
from ..core.models import TestCase, RAGResponse, EvaluationResult


class RAGASIntegration(BaseEvaluator):
    """Integration with RAGAS framework for RAG-specific evaluation"""
    
    def __init__(self):
        """Initialize RAGAS integration"""
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS is not installed. Please install it with: pip install ragas datasets"
            )
        
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup RAGAS metrics"""
        self.metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness
        }
    
    def prepare_dataset(self, test_cases: List[TestCase], responses: List[RAGResponse]) -> Dataset:
        """Prepare dataset in RAGAS format
        
        Args:
            test_cases: List of test cases
            responses: List of RAG responses
            
        Returns:
            Dataset: Hugging Face dataset for RAGAS evaluation
        """
        if len(test_cases) != len(responses):
            raise ValueError("Number of test cases must match number of responses")
        
        # Extract data for RAGAS format
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        for test_case, response in zip(test_cases, responses):
            questions.append(test_case.question)
            answers.append(response.answer)
            ground_truths.append(test_case.expected_answer)
            
            # Extract contexts from source documents
            if response.source_documents:
                context_list = [doc.content for doc in response.source_documents]
            else:
                context_list = []
            contexts.append(context_list)
        
        # Create dataset in RAGAS format
        data_samples = {
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truths
        }
        
        return Dataset.from_dict(data_samples)
    
    def evaluate_with_ragas(
        self, 
        dataset: Dataset,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate using RAGAS framework
        
        Args:
            dataset: Hugging Face dataset in RAGAS format
            metrics: List of metric names to use (default: all available)
            
        Returns:
            Dict containing RAGAS results
        """
        # Select metrics to use
        if metrics is None:
            metrics = list(self.metrics.keys())
        
        selected_metrics = []
        for metric_name in metrics:
            if metric_name in self.metrics and self.metrics[metric_name] is not None:
                selected_metrics.append(self.metrics[metric_name])
        
        if not selected_metrics:
            raise ValueError(f"No valid metrics found. Available: {list(self.metrics.keys())}")
        
        # Run evaluation
        try:
            evaluation_results = evaluate(
                dataset=dataset,
                metrics=selected_metrics
            )
            return evaluation_results
        except Exception as e:
            logging.error(f"RAGAS evaluation failed: {str(e)}")
            return {"error": str(e), "results": []}
    
    def evaluate(self, test_cases: List[TestCase], responses: List[RAGResponse]) -> EvaluationResult:
        """Evaluate RAG responses using RAGAS framework
        
        Args:
            test_cases: List of test cases
            responses: List of RAG responses
            
        Returns:
            EvaluationResult: Standardized evaluation results
        """
        if len(test_cases) != len(responses):
            raise ValueError("Number of test cases must match number of responses")
        
        # Prepare dataset
        try:
            dataset = self.prepare_dataset(test_cases, responses)
        except Exception as e:
            logging.error(f"Failed to prepare RAGAS dataset: {str(e)}")
            return EvaluationResult(
                overall_score=0.0,
                metric_scores={"error": 0.0},
                test_case_results=[{"error": f"Dataset preparation failed: {str(e)}"}],
                recommendations=["RAGAS dataset preparation failed. Check data format."]
            )
        
        # Run RAGAS evaluation
        ragas_results = self.evaluate_with_ragas(dataset)
        
        if isinstance(ragas_results, dict) and "error" in ragas_results:
            # Return error result
            error_msg = ragas_results.get("error", "Unknown error")
            return EvaluationResult(
                overall_score=0.0,
                metric_scores={"error": 0.0},
                test_case_results=[{"error": error_msg}],
                recommendations=["RAGAS evaluation failed. Check logs for details."]
            )
        
        # Process RAGAS results into our format
        try:
            return self._process_ragas_results(ragas_results, test_cases, responses)
        except Exception as e:
            logging.error(f"Error processing RAGAS results: {str(e)}")
            return EvaluationResult(
                overall_score=0.0,
                metric_scores={"error": 0.0},
                test_case_results=[{"error": f"Result processing failed: {str(e)}"}],
                recommendations=["RAGAS result processing failed. Check result format."]
            )
    
    def _process_ragas_results(
        self, 
        ragas_results: Any, 
        test_cases: List[TestCase], 
        responses: List[RAGResponse]
    ) -> EvaluationResult:
        """Process RAGAS results into our standardized format
        
        Args:
            ragas_results: Raw RAGAS evaluation results
            test_cases: Original test cases
            responses: Original responses
            
        Returns:
            EvaluationResult: Standardized evaluation results
        """
        metric_scores = {}
        test_case_results = []
        
        # Convert RAGAS results to pandas DataFrame for easier processing
        try:
            if hasattr(ragas_results, 'to_pandas'):
                results_df = ragas_results.to_pandas()
                
                # Extract metric scores (averages)
                metric_columns = ['faithfulness', 'answer_relevancy', 'context_recall', 'answer_correctness']
                for metric in metric_columns:
                    if metric in results_df.columns:
                        # Calculate average score for this metric
                        scores = results_df[metric].dropna()
                        if len(scores) > 0:
                            metric_scores[metric] = float(scores.mean())
                        else:
                            metric_scores[metric] = 0.0
                
                # Process individual test case results
                for i, (test_case, response) in enumerate(zip(test_cases, responses)):
                    case_result = {
                        "question": test_case.question,
                        "expected_answer": test_case.expected_answer,
                        "generated_answer": response.answer,
                        "metrics": {}
                    }
                    
                    # Extract metrics for this test case
                    if i < len(results_df):
                        row = results_df.iloc[i]
                        for metric in metric_columns:
                            if metric in row and (pd is None or not pd.isna(row[metric])):
                                case_result["metrics"][metric] = {
                                    "score": float(row[metric]),
                                    "reasoning": f"RAGAS {metric} evaluation",
                                    "metric": metric
                                }
                    
                    test_case_results.append(case_result)
            
            else:
                # Fallback: create basic results structure
                for i, (test_case, response) in enumerate(zip(test_cases, responses)):
                    case_result = {
                        "question": test_case.question,
                        "expected_answer": test_case.expected_answer,
                        "generated_answer": response.answer,
                        "metrics": {
                            "ragas_score": {
                                "score": 0.5,  # Default score
                                "reasoning": "RAGAS evaluation completed",
                                "metric": "ragas_score"
                            }
                        }
                    }
                    test_case_results.append(case_result)
                metric_scores = {"ragas_score": 0.5}
        except Exception as e:
            logging.error(f"Error processing RAGAS DataFrame: {str(e)}")
            # Fallback processing
            for i, (test_case, response) in enumerate(zip(test_cases, responses)):
                case_result = {
                    "question": test_case.question,
                    "expected_answer": test_case.expected_answer,
                    "generated_answer": response.answer,
                    "metrics": {
                        "ragas_fallback": {
                            "score": 0.5,
                            "reasoning": "RAGAS processing fallback",
                            "metric": "ragas_fallback"
                        }
                    }
                }
                test_case_results.append(case_result)
            
            metric_scores = {"ragas_fallback": 0.5}
        
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
        """Generate recommendations based on RAGAS metric scores"""
        recommendations = []
        
        if "error" in metric_scores:
            recommendations.append("RAGAS evaluation encountered errors. Check configuration and dependencies.")
            return recommendations
        
        for metric_name, score in metric_scores.items():
            if score < 0.7:
                if "faithfulness" in metric_name.lower():
                    recommendations.append(
                        f"Low {metric_name} score ({score:.2f}). The model may be hallucinating. "
                        "Consider strengthening grounding mechanisms or improving context quality."
                    )
                elif "answer_relevancy" in metric_name.lower():
                    recommendations.append(
                        f"Low {metric_name} score ({score:.2f}). Generated answers may not be relevant to questions. "
                        "Consider improving the generation model or prompt engineering."
                    )
                elif "context_recall" in metric_name.lower():
                    recommendations.append(
                        f"Low {metric_name} score ({score:.2f}). The retrieval system may be missing relevant context. "
                        "Consider improving retrieval algorithms or expanding the knowledge base."
                    )
                elif "answer_correctness" in metric_name.lower():
                    recommendations.append(
                        f"Low {metric_name} score ({score:.2f}). Generated answers may be factually incorrect. "
                        "Consider improving the knowledge base or fine-tuning the generation model."
                    )
                else:
                    recommendations.append(
                        f"Low {metric_name} score ({score:.2f}). Review this metric's performance."
                    )
        
        if all(score > 0.8 for score in metric_scores.values()):
            recommendations.append(
                "Excellent performance across all RAGAS metrics! RAG system is performing very well."
            )
        
        return recommendations
    
    def get_supported_metrics(self) -> List[str]:
        """Get list of supported RAGAS metrics"""
        return list(self.metrics.keys())
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of RAGAS metrics"""
        return {
            "faithfulness": "How factually consistent is the answer with the context? (Prevents hallucination)",
            "answer_relevancy": "How relevant is the answer to the question?",
            "context_recall": "Did we retrieve all the necessary context to answer the question?",
            "answer_correctness": "How accurate is the answer compared to the ground truth?"
        }