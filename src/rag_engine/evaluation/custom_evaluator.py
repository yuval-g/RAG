"""
Custom evaluation system with faithfulness and correctness chains
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from ..core.interfaces import BaseEvaluator
from ..core.models import TestCase, RAGResponse, EvaluationResult


class ResultScore(BaseModel):
    """Structured output schema for evaluation scores"""
    score: float = Field(..., description="The score of the result, ranging from 0 to 1 where 1 is the best possible score.")
    reasoning: str = Field(..., description="Explanation of the scoring decision")


class CustomEvaluator(BaseEvaluator):
    """Custom evaluator with LLM-based judges for faithfulness and correctness"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash-lite", temperature: float = 0.0):
        """Initialize the custom evaluator with Google Gemini LLM"""
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_tokens=4000
        )
        
        # Initialize evaluation chains
        self._setup_correctness_chain()
        self._setup_faithfulness_chain()
        self._setup_contextual_relevancy_chain()
    
    def _setup_correctness_chain(self):
        """Setup the correctness evaluation chain"""
        self.correctness_prompt = PromptTemplate(
            input_variables=["question", "ground_truth", "generated_answer"],
            template="""
            Question: {question}
            Ground Truth: {ground_truth}
            Generated Answer: {generated_answer}

            Evaluate the correctness of the generated answer compared to the ground truth.
            Score from 0 to 1, where 1 is perfectly correct and 0 is completely incorrect.
            
            Consider:
            - Factual accuracy
            - Completeness of the answer
            - Partial credit for partially correct answers
            
            Provide both a score and reasoning for your evaluation.
            """
        )
        
        self.correctness_chain = self.correctness_prompt | self.llm.with_structured_output(ResultScore)
    
    def _setup_faithfulness_chain(self):
        """Setup the faithfulness evaluation chain"""
        self.faithfulness_prompt = PromptTemplate(
            input_variables=["question", "context", "generated_answer"],
            template="""
            Question: {question}
            Context: {context}
            Generated Answer: {generated_answer}

            Evaluate if the generated answer to the question can be deduced from the context.
            Score of 0 or 1, where 1 is perfectly faithful AND CAN BE DERIVED FROM THE CONTEXT and 0 otherwise.
            You don't care if the answer is factually correct; all you care about is if the answer can be deduced from the context.
            
            Examples:
            Question: What is the capital of France and Spain?
            Context: Paris is the capital of France and Madrid is the capital of Spain.
            Generated Answer: Paris
            Score: 1 (the answer can be derived from the context, even if incomplete)
            
            Question: What is 2+2?
            Context: 4.
            Generated Answer: 4.
            Score: 0 (the context states '4', but doesn't provide information to deduce the answer to 'What is 2+2?')
            
            Provide both a score and reasoning for your evaluation.
            """
        )
        
        self.faithfulness_chain = self.faithfulness_prompt | self.llm.with_structured_output(ResultScore)
    
    def _setup_contextual_relevancy_chain(self):
        """Setup the contextual relevancy evaluation chain"""
        self.contextual_relevancy_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            Question: {question}
            Context: {context}

            Evaluate if the provided context is relevant to answering the question.
            Score from 0 to 1, where 1 means the context is highly relevant and contains information needed to answer the question,
            and 0 means the context is completely irrelevant.
            
            Consider:
            - Does the context contain information that helps answer the question?
            - Is the context on-topic and related to the question domain?
            - Would this context be useful for generating a good answer?
            
            Provide both a score and reasoning for your evaluation.
            """
        )
        
        self.contextual_relevancy_chain = self.contextual_relevancy_prompt | self.llm.with_structured_output(ResultScore)
    
    def evaluate_correctness(self, question: str, ground_truth: str, generated_answer: str) -> Dict[str, Any]:
        """Evaluate correctness of a generated answer against ground truth"""
        try:
            result = self.correctness_chain.invoke({
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer
            })
            return {
                "score": result.score,
                "reasoning": result.reasoning,
                "metric": "correctness"
            }
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"Error during evaluation: {str(e)}",
                "metric": "correctness"
            }
    
    def evaluate_faithfulness(self, question: str, context: str, generated_answer: str) -> Dict[str, Any]:
        """Evaluate faithfulness of a generated answer to the provided context"""
        try:
            result = self.faithfulness_chain.invoke({
                "question": question,
                "context": context,
                "generated_answer": generated_answer
            })
            return {
                "score": result.score,
                "reasoning": result.reasoning,
                "metric": "faithfulness"
            }
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"Error during evaluation: {str(e)}",
                "metric": "faithfulness"
            }
    
    def evaluate_contextual_relevancy(self, question: str, context: str) -> Dict[str, Any]:
        """Evaluate relevancy of context to the question"""
        try:
            result = self.contextual_relevancy_chain.invoke({
                "question": question,
                "context": context
            })
            return {
                "score": result.score,
                "reasoning": result.reasoning,
                "metric": "contextual_relevancy"
            }
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"Error during evaluation: {str(e)}",
                "metric": "contextual_relevancy"
            }
    
    def evaluate(self, test_cases: List[TestCase], responses: List[RAGResponse]) -> EvaluationResult:
        """Evaluate RAG responses against test cases using custom metrics"""
        if len(test_cases) != len(responses):
            raise ValueError("Number of test cases must match number of responses")
        
        all_scores = {"correctness": [], "faithfulness": [], "contextual_relevancy": []}
        test_case_results = []
        
        for test_case, response in zip(test_cases, responses):
            case_result = {
                "question": test_case.question,
                "expected_answer": test_case.expected_answer,
                "generated_answer": response.answer,
                "metrics": {}
            }
            
            # Evaluate correctness
            correctness_result = self.evaluate_correctness(
                test_case.question,
                test_case.expected_answer,
                response.answer
            )
            case_result["metrics"]["correctness"] = correctness_result
            all_scores["correctness"].append(correctness_result["score"])
            
            # Evaluate faithfulness (using source documents as context)
            if response.source_documents:
                context = "\n".join([doc.content for doc in response.source_documents])
                faithfulness_result = self.evaluate_faithfulness(
                    test_case.question,
                    context,
                    response.answer
                )
                case_result["metrics"]["faithfulness"] = faithfulness_result
                all_scores["faithfulness"].append(faithfulness_result["score"])
                
                # Evaluate contextual relevancy
                relevancy_result = self.evaluate_contextual_relevancy(
                    test_case.question,
                    context
                )
                case_result["metrics"]["contextual_relevancy"] = relevancy_result
                all_scores["contextual_relevancy"].append(relevancy_result["score"])
            else:
                # No context available
                case_result["metrics"]["faithfulness"] = {
                    "score": 0.0,
                    "reasoning": "No source documents provided",
                    "metric": "faithfulness"
                }
                case_result["metrics"]["contextual_relevancy"] = {
                    "score": 0.0,
                    "reasoning": "No source documents provided",
                    "metric": "contextual_relevancy"
                }
                all_scores["faithfulness"].append(0.0)
                all_scores["contextual_relevancy"].append(0.0)
            
            test_case_results.append(case_result)
        
        # Calculate average scores
        metric_scores = {}
        for metric, scores in all_scores.items():
            if scores:
                metric_scores[metric] = sum(scores) / len(scores)
            else:
                metric_scores[metric] = 0.0
        
        # Calculate overall score (weighted average)
        overall_score = (
            metric_scores["correctness"] * 0.4 +
            metric_scores["faithfulness"] * 0.4 +
            metric_scores["contextual_relevancy"] * 0.2
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metric_scores)
        
        return EvaluationResult(
            overall_score=overall_score,
            metric_scores=metric_scores,
            test_case_results=test_case_results,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, metric_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on metric scores"""
        recommendations = []
        
        if metric_scores["correctness"] < 0.7:
            recommendations.append(
                "Low correctness score detected. Consider improving the knowledge base "
                "or fine-tuning the generation model."
            )
        
        if metric_scores["faithfulness"] < 0.8:
            recommendations.append(
                "Low faithfulness score detected. The model may be hallucinating. "
                "Consider strengthening grounding mechanisms or adjusting prompts."
            )
        
        if metric_scores["contextual_relevancy"] < 0.7:
            recommendations.append(
                "Low contextual relevancy score detected. Consider improving the "
                "retrieval system or re-ranking mechanisms."
            )
        
        if all(score > 0.8 for score in metric_scores.values()):
            recommendations.append(
                "Excellent performance across all metrics! System is performing well."
            )
        
        return recommendations
    
    def get_supported_metrics(self) -> List[str]:
        """Get list of supported evaluation metrics"""
        return ["correctness", "faithfulness", "contextual_relevancy"]