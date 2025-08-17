"""
Evaluation manager that coordinates all evaluation approaches
"""

from typing import List, Dict, Any, Optional, Union
import logging
import time
from datetime import datetime

from ..core.interfaces import BaseEvaluator
from ..core.models import TestCase, RAGResponse, EvaluationResult
from .custom_evaluator import CustomEvaluator
from .deepeval_integration import DeepEvalIntegration, DEEPEVAL_AVAILABLE
from .ragas_integration import RAGASIntegration, RAGAS_AVAILABLE


class MetricsCollector:
    """Collects and aggregates metrics from multiple evaluation frameworks"""
    
    def __init__(self):
        self.results = {}
        self.execution_times = {}
        self.errors = {}
    
    def add_result(self, framework_name: str, result: EvaluationResult, execution_time: float):
        """Add evaluation result from a framework
        
        Args:
            framework_name: Name of the evaluation framework
            result: Evaluation result
            execution_time: Time taken for evaluation in seconds
        """
        self.results[framework_name] = result
        self.execution_times[framework_name] = execution_time
    
    def add_error(self, framework_name: str, error: str, execution_time: float):
        """Add error from a framework
        
        Args:
            framework_name: Name of the evaluation framework
            error: Error message
            execution_time: Time taken before error in seconds
        """
        self.errors[framework_name] = error
        self.execution_times[framework_name] = execution_time
    
    def get_aggregated_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics across all frameworks
        
        Returns:
            Dict mapping metric names to aggregated scores
        """
        all_metrics = {}
        metric_counts = {}
        
        for framework_name, result in self.results.items():
            for metric_name, score in result.metric_scores.items():
                if metric_name == "error":
                    continue
                
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = 0.0
                    metric_counts[metric_name] = 0
                
                all_metrics[metric_name] += score
                metric_counts[metric_name] += 1
        
        # Calculate averages
        aggregated = {}
        for metric_name, total_score in all_metrics.items():
            if metric_counts[metric_name] > 0:
                aggregated[metric_name] = total_score / metric_counts[metric_name]
            else:
                aggregated[metric_name] = 0.0
        
        return aggregated
    
    def get_framework_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Get comparison of frameworks
        
        Returns:
            Dict with framework comparison data
        """
        comparison = {}
        
        for framework_name, result in self.results.items():
            comparison[framework_name] = {
                "overall_score": result.overall_score,
                "metric_scores": result.metric_scores,
                "execution_time": self.execution_times.get(framework_name, 0.0),
                "test_cases_evaluated": len(result.test_case_results),
                "status": "success"
            }
        
        for framework_name, error in self.errors.items():
            comparison[framework_name] = {
                "overall_score": 0.0,
                "metric_scores": {"error": 0.0},
                "execution_time": self.execution_times.get(framework_name, 0.0),
                "test_cases_evaluated": 0,
                "status": "error",
                "error_message": error
            }
        
        return comparison


class EvaluationManager:
    """Manages and coordinates all evaluation approaches"""
    
    def __init__(self, enable_custom: bool = True, enable_deepeval: bool = True, enable_ragas: bool = True):
        """Initialize evaluation manager
        
        Args:
            enable_custom: Whether to enable custom evaluation
            enable_deepeval: Whether to enable DeepEval integration
            enable_ragas: Whether to enable RAGAS integration
        """
        self.evaluators = {}
        self.enabled_frameworks = []
        
        # Initialize custom evaluator
        if enable_custom:
            try:
                self.evaluators["custom"] = CustomEvaluator()
                self.enabled_frameworks.append("custom")
                logging.info("Custom evaluator initialized")
            except Exception as e:
                logging.error(f"Failed to initialize custom evaluator: {str(e)}")
        
        # Initialize DeepEval integration
        if enable_deepeval and DEEPEVAL_AVAILABLE:
            try:
                self.evaluators["deepeval"] = DeepEvalIntegration()
                self.enabled_frameworks.append("deepeval")
                logging.info("DeepEval integration initialized")
            except Exception as e:
                logging.error(f"Failed to initialize DeepEval integration: {str(e)}")
        elif enable_deepeval:
            logging.warning("DeepEval requested but not available")
        
        # Initialize RAGAS integration
        if enable_ragas and RAGAS_AVAILABLE:
            try:
                self.evaluators["ragas"] = RAGASIntegration()
                self.enabled_frameworks.append("ragas")
                logging.info("RAGAS integration initialized")
            except Exception as e:
                logging.error(f"Failed to initialize RAGAS integration: {str(e)}")
        elif enable_ragas:
            logging.warning("RAGAS requested but not available")
        
        if not self.evaluators:
            raise RuntimeError("No evaluation frameworks could be initialized")
        
        logging.info(f"Evaluation manager initialized with frameworks: {self.enabled_frameworks}")
    
    def evaluate_comprehensive(
        self, 
        test_cases: List[TestCase], 
        responses: List[RAGResponse],
        frameworks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation using multiple frameworks
        
        Args:
            test_cases: List of test cases
            responses: List of RAG responses
            frameworks: List of framework names to use (default: all available)
            
        Returns:
            Dict containing comprehensive evaluation results
        """
        if len(test_cases) != len(responses):
            raise ValueError("Number of test cases must match number of responses")
        
        # Determine which frameworks to use
        if frameworks is None:
            frameworks = self.enabled_frameworks
        else:
            # Validate requested frameworks
            invalid_frameworks = set(frameworks) - set(self.enabled_frameworks)
            if invalid_frameworks:
                raise ValueError(f"Invalid frameworks requested: {invalid_frameworks}")
        
        collector = MetricsCollector()
        evaluation_start_time = time.time()
        
        # Run evaluation with each framework
        for framework_name in frameworks:
            if framework_name not in self.evaluators:
                logging.warning(f"Framework {framework_name} not available, skipping")
                continue
            
            evaluator = self.evaluators[framework_name]
            framework_start_time = time.time()
            
            try:
                logging.info(f"Running evaluation with {framework_name}")
                result = evaluator.evaluate(test_cases, responses)
                execution_time = time.time() - framework_start_time
                
                collector.add_result(framework_name, result, execution_time)
                logging.info(f"{framework_name} evaluation completed in {execution_time:.2f}s")
                
            except Exception as e:
                execution_time = time.time() - framework_start_time
                error_msg = f"Evaluation failed: {str(e)}"
                collector.add_error(framework_name, error_msg, execution_time)
                logging.error(f"{framework_name} evaluation failed: {str(e)}")
        
        total_execution_time = time.time() - evaluation_start_time
        
        # Generate comprehensive report
        return self._generate_comprehensive_report(
            collector, 
            test_cases, 
            responses, 
            total_execution_time
        )
    
    def _generate_comprehensive_report(
        self, 
        collector: MetricsCollector, 
        test_cases: List[TestCase], 
        responses: List[RAGResponse],
        total_execution_time: float
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report
        
        Args:
            collector: Metrics collector with results
            test_cases: Original test cases
            responses: Original responses
            total_execution_time: Total time for all evaluations
            
        Returns:
            Dict containing comprehensive report
        """
        # Get aggregated metrics
        aggregated_metrics = collector.get_aggregated_metrics()
        framework_comparison = collector.get_framework_comparison()
        
        # Calculate overall score
        if aggregated_metrics:
            overall_score = sum(aggregated_metrics.values()) / len(aggregated_metrics)
        else:
            overall_score = 0.0
        
        # Generate recommendations
        recommendations = self._generate_comprehensive_recommendations(
            aggregated_metrics, 
            framework_comparison
        )
        
        # Create detailed test case analysis
        test_case_analysis = self._analyze_test_cases(collector, test_cases, responses)
        
        # Generate performance analysis
        performance_analysis = self._analyze_performance(collector, total_execution_time)
        
        return {
            "evaluation_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_test_cases": len(test_cases),
                "frameworks_used": list(framework_comparison.keys()),
                "overall_score": overall_score,
                "total_execution_time": total_execution_time
            },
            "aggregated_metrics": aggregated_metrics,
            "framework_comparison": framework_comparison,
            "test_case_analysis": test_case_analysis,
            "performance_analysis": performance_analysis,
            "recommendations": recommendations,
            "detailed_results": {
                framework: result.model_dump() if hasattr(result, 'model_dump') else (
                    result.dict() if hasattr(result, 'dict') else result.__dict__
                )
                for framework, result in collector.results.items()
            }
        }
    
    def _generate_comprehensive_recommendations(
        self, 
        aggregated_metrics: Dict[str, float], 
        framework_comparison: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate comprehensive recommendations based on all evaluation results
        
        Args:
            aggregated_metrics: Aggregated metrics across frameworks
            framework_comparison: Comparison of framework results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Overall performance recommendations
        if aggregated_metrics:
            avg_score = sum(aggregated_metrics.values()) / len(aggregated_metrics)
            
            if avg_score < 0.6:
                recommendations.append(
                    "⚠️ Overall system performance is below acceptable levels. "
                    "Consider comprehensive system review and improvements."
                )
            elif avg_score < 0.8:
                recommendations.append(
                    "📈 System performance is moderate. Focus on specific weak areas for improvement."
                )
            else:
                recommendations.append(
                    "✅ Excellent overall system performance! Continue monitoring and fine-tuning."
                )
        
        # Metric-specific recommendations
        for metric_name, score in aggregated_metrics.items():
            if score < 0.7:
                if "faithfulness" in metric_name.lower():
                    recommendations.append(
                        f"🚨 Low {metric_name} score ({score:.2f}). "
                        "High risk of hallucination - strengthen grounding mechanisms immediately."
                    )
                elif "correctness" in metric_name.lower():
                    recommendations.append(
                        f"📚 Low {metric_name} score ({score:.2f}). "
                        "Review knowledge base quality and generation model accuracy."
                    )
                elif "relevancy" in metric_name.lower() or "relevance" in metric_name.lower():
                    recommendations.append(
                        f"🎯 Low {metric_name} score ({score:.2f}). "
                        "Improve retrieval system and context filtering."
                    )
                elif "recall" in metric_name.lower():
                    recommendations.append(
                        f"🔍 Low {metric_name} score ({score:.2f}). "
                        "Retrieval system may be missing relevant information - expand search scope."
                    )
        
        # Framework-specific recommendations
        successful_frameworks = [
            name for name, data in framework_comparison.items() 
            if data["status"] == "success"
        ]
        failed_frameworks = [
            name for name, data in framework_comparison.items() 
            if data["status"] == "error"
        ]
        
        if failed_frameworks:
            recommendations.append(
                f"🔧 Some evaluation frameworks failed: {', '.join(failed_frameworks)}. "
                "Check dependencies and configuration."
            )
        
        if len(successful_frameworks) > 1:
            # Compare framework results
            scores = [
                framework_comparison[name]["overall_score"] 
                for name in successful_frameworks
            ]
            if max(scores) - min(scores) > 0.2:
                recommendations.append(
                    "📊 Significant variance between evaluation frameworks detected. "
                    "Review evaluation criteria and consider framework-specific tuning."
                )
        
        return recommendations
    
    def _analyze_test_cases(
        self, 
        collector: MetricsCollector, 
        test_cases: List[TestCase], 
        responses: List[RAGResponse]
    ) -> Dict[str, Any]:
        """Analyze individual test case performance across frameworks
        
        Args:
            collector: Metrics collector with results
            test_cases: Original test cases
            responses: Original responses
            
        Returns:
            Dict containing test case analysis
        """
        analysis = {
            "total_cases": len(test_cases),
            "case_performance": [],
            "worst_performing_cases": [],
            "best_performing_cases": []
        }
        
        # Analyze each test case
        for i, (test_case, response) in enumerate(zip(test_cases, responses)):
            case_scores = []
            case_metrics = {}
            
            # Collect scores from all frameworks
            for framework_name, result in collector.results.items():
                if i < len(result.test_case_results):
                    case_result = result.test_case_results[i]
                    if "metrics" in case_result:
                        for metric_name, metric_data in case_result["metrics"].items():
                            if isinstance(metric_data, dict) and "score" in metric_data:
                                score = metric_data["score"]
                                case_scores.append(score)
                                
                                if metric_name not in case_metrics:
                                    case_metrics[metric_name] = []
                                case_metrics[metric_name].append(score)
            
            # Calculate average score for this case
            avg_score = sum(case_scores) / len(case_scores) if case_scores else 0.0
            
            case_analysis = {
                "case_index": i,
                "question": test_case.question,
                "expected_answer": test_case.expected_answer,
                "generated_answer": response.answer,
                "average_score": avg_score,
                "metric_averages": {
                    metric: sum(scores) / len(scores) 
                    for metric, scores in case_metrics.items()
                },
                "framework_scores": case_scores
            }
            
            analysis["case_performance"].append(case_analysis)
        
        # Identify best and worst performing cases
        if analysis["case_performance"]:
            sorted_cases = sorted(
                analysis["case_performance"], 
                key=lambda x: x["average_score"]
            )
            
            # Get worst 3 cases (or all if less than 3)
            worst_count = min(3, len(sorted_cases))
            analysis["worst_performing_cases"] = sorted_cases[:worst_count]
            
            # Get best 3 cases (or all if less than 3)
            best_count = min(3, len(sorted_cases))
            analysis["best_performing_cases"] = sorted_cases[-best_count:]
        
        return analysis
    
    def _analyze_performance(
        self, 
        collector: MetricsCollector, 
        total_execution_time: float
    ) -> Dict[str, Any]:
        """Analyze performance characteristics of evaluation frameworks
        
        Args:
            collector: Metrics collector with results
            total_execution_time: Total execution time
            
        Returns:
            Dict containing performance analysis
        """
        analysis = {
            "total_execution_time": total_execution_time,
            "framework_performance": {},
            "efficiency_ranking": []
        }
        
        # Analyze each framework's performance
        for framework_name in collector.execution_times:
            execution_time = collector.execution_times[framework_name]
            
            framework_perf = {
                "execution_time": execution_time,
                "percentage_of_total": (execution_time / total_execution_time) * 100 if total_execution_time > 0 else 0
            }
            
            # Add success/error status
            if framework_name in collector.results:
                result = collector.results[framework_name]
                framework_perf["status"] = "success"
                framework_perf["test_cases_processed"] = len(result.test_case_results)
                framework_perf["overall_score"] = result.overall_score
                
                # Calculate efficiency (score per second)
                if execution_time > 0:
                    framework_perf["efficiency"] = result.overall_score / execution_time
                else:
                    framework_perf["efficiency"] = 0.0
            else:
                framework_perf["status"] = "error"
                framework_perf["test_cases_processed"] = 0
                framework_perf["overall_score"] = 0.0
                framework_perf["efficiency"] = 0.0
            
            analysis["framework_performance"][framework_name] = framework_perf
        
        # Create efficiency ranking
        successful_frameworks = [
            (name, data) for name, data in analysis["framework_performance"].items()
            if data["status"] == "success"
        ]
        
        analysis["efficiency_ranking"] = sorted(
            successful_frameworks,
            key=lambda x: x[1]["efficiency"],
            reverse=True
        )
        
        return analysis
    
    def get_available_frameworks(self) -> List[str]:
        """Get list of available evaluation frameworks
        
        Returns:
            List of framework names
        """
        return self.enabled_frameworks.copy()
    
    def get_framework_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available frameworks
        
        Returns:
            Dict with framework information
        """
        info = {}
        
        for framework_name, evaluator in self.evaluators.items():
            framework_info = {
                "name": framework_name,
                "available": True,
                "supported_metrics": evaluator.get_supported_metrics()
            }
            
            # Add framework-specific information
            if hasattr(evaluator, 'get_available_models'):
                framework_info["available_models"] = evaluator.get_available_models()
            
            if hasattr(evaluator, 'get_metric_descriptions'):
                framework_info["metric_descriptions"] = evaluator.get_metric_descriptions()
            
            info[framework_name] = framework_info
        
        return info