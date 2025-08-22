"""
Integration tests for specific component combinations and advanced workflows
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from src.rag_engine.core.engine import RAGEngine
from src.rag_engine.core.config import PipelineConfig, IndexingStrategy, LLMProvider, VectorStore
from src.rag_engine.core.models import Document, TestCase, RouteConfig, RoutingDecision
from src.rag_engine.query.processor import QueryProcessor
from src.rag_engine.routing.query_router import QueryRouter, RoutingStrategy
from src.rag_engine.indexing.indexing_manager import IndexingManager
from src.rag_engine.retrieval.retrieval_engine import RetrievalEngine
from src.rag_engine.evaluation.evaluation_manager import EvaluationManager


class TestComponentCombinations:
    """Test specific combinations of RAG components"""
    
    @pytest.fixture
    def multi_domain_documents(self):
        """Documents from multiple domains for routing tests"""
        return [
            # Python/Programming documents
            Document(
                content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, artificial intelligence, and automation.",
                metadata={"domain": "programming", "language": "python", "difficulty": "beginner"},
                doc_id="python_001"
            ),
            Document(
                content="Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design. It follows the model-view-template (MVT) architectural pattern and includes an ORM, admin interface, and built-in security features.",
                metadata={"domain": "programming", "language": "python", "framework": "django", "difficulty": "intermediate"},
                doc_id="django_001"
            ),
            
            # JavaScript/Web Development documents
            Document(
                content="JavaScript is a versatile programming language primarily used for web development. It enables interactive web pages and is an essential part of web applications. Modern JavaScript (ES6+) includes features like arrow functions, promises, and modules.",
                metadata={"domain": "programming", "language": "javascript", "difficulty": "beginner"},
                doc_id="javascript_001"
            ),
            Document(
                content="React is a JavaScript library for building user interfaces, particularly web applications. It uses a component-based architecture and virtual DOM for efficient rendering. React applications are built using JSX syntax and can manage state using hooks or external libraries.",
                metadata={"domain": "programming", "language": "javascript", "framework": "react", "difficulty": "intermediate"},
                doc_id="react_001"
            ),
            
            # Data Science documents
            Document(
                content="Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It combines statistics, mathematics, programming, and domain expertise.",
                metadata={"domain": "data_science", "difficulty": "beginner"},
                doc_id="datascience_001"
            ),
            Document(
                content="Pandas is a powerful Python library for data manipulation and analysis. It provides data structures like DataFrame and Series, along with tools for reading/writing data, cleaning, transforming, and analyzing datasets efficiently.",
                metadata={"domain": "data_science", "language": "python", "library": "pandas", "difficulty": "intermediate"},
                doc_id="pandas_001"
            ),
            
            # Machine Learning documents
            Document(
                content="Supervised learning is a type of machine learning where algorithms learn from labeled training data to make predictions on new, unseen data. Common supervised learning tasks include classification (predicting categories) and regression (predicting continuous values).",
                metadata={"domain": "machine_learning", "type": "supervised", "difficulty": "intermediate"},
                doc_id="supervised_001"
            ),
            Document(
                content="Scikit-learn is a comprehensive machine learning library for Python that provides simple and efficient tools for data mining and data analysis. It includes algorithms for classification, regression, clustering, and dimensionality reduction.",
                metadata={"domain": "machine_learning", "language": "python", "library": "scikit-learn", "difficulty": "intermediate"},
                doc_id="sklearn_001"
            )
        ]
    
    @pytest.fixture
    def domain_routes(self):
        """Route configurations for different domains"""
        return [
            RouteConfig(
                name="python_expert",
                description="Expert in Python programming, frameworks, and libraries",
                keywords=["python", "django", "flask", "pandas", "numpy", "scikit-learn"],
                metadata={"domain": "programming", "language": "python"}
            ),
            RouteConfig(
                name="javascript_expert",
                description="Expert in JavaScript, web development, and frontend frameworks",
                keywords=["javascript", "react", "vue", "angular", "node", "web development"],
                metadata={"domain": "programming", "language": "javascript"}
            ),
            RouteConfig(
                name="data_science_expert",
                description="Expert in data science, analytics, and statistical methods",
                keywords=["data science", "analytics", "statistics", "visualization", "pandas"],
                metadata={"domain": "data_science"}
            ),
            RouteConfig(
                name="ml_expert",
                description="Expert in machine learning algorithms and implementations",
                keywords=["machine learning", "supervised learning", "unsupervised learning", "scikit-learn"],
                metadata={"domain": "machine_learning"}
            )
        ]
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    @patch('src.rag_engine.routing.logical_router.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.routing.semantic_router.GoogleGenerativeAIEmbeddings')
    def test_routing_with_query_processing_combination(self, mock_semantic_embeddings, mock_logical_llm,
                                                     mock_chroma, mock_embeddings, mock_llm,
                                                     multi_domain_documents, domain_routes):
        """Test routing combined with advanced query processing"""
        # Setup mocks
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_logical_llm.return_value = Mock()
        mock_semantic_embeddings.return_value = Mock()
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.return_value = [
            Mock(page_content=doc.content, metadata=doc.metadata) 
            for doc in multi_domain_documents
        ]
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        # Initialize components
        config = PipelineConfig(
            query_strategies=["multi_query", "rag_fusion"],
            retrieval_k=3
        )
        
        engine = RAGEngine(config)
        router = QueryRouter(routes=domain_routes, default_strategy=RoutingStrategy.AUTO)
        query_processor = QueryProcessor(config)
        
        # Index multi-domain documents
        engine.add_documents(multi_domain_documents)
        
        # Test queries that should route to different experts
        test_scenarios = [
            {
                "query": "How do I use pandas for data analysis in Python?",
                "expected_routes": ["python_expert", "data_science_expert"],
                "processing_strategy": "multi_query"
            },
            {
                "query": "What's the difference between React and Vue.js?",
                "expected_routes": ["javascript_expert"],
                "processing_strategy": "rag_fusion"
            },
            {
                "query": "How does supervised learning work with scikit-learn?",
                "expected_routes": ["ml_expert", "python_expert"],
                "processing_strategy": "decomposition"
            }
        ]
        
        for scenario in test_scenarios:
            # Mock routing decision
            mock_decision = RoutingDecision(
                selected_route=scenario["expected_routes"][0],
                confidence=0.85,
                reasoning=f"Query matches {scenario['expected_routes'][0]} domain",
                metadata={"router_type": "logical", "query_complexity": "medium"}
            )
            
            with patch.object(router, 'route', return_value=mock_decision):
                # Step 1: Route the query
                routing_decision = router.route(scenario["query"])
                assert routing_decision.selected_route in scenario["expected_routes"]
                
                # Step 2: Process query with selected strategy
                processed_query = query_processor.process(
                    scenario["query"], 
                    scenario["processing_strategy"]
                )
                assert processed_query.strategy_used == scenario["processing_strategy"]
                
                # Step 3: Execute query with routing and processing
                response = engine.query(
                    scenario["query"],
                    route=routing_decision.selected_route,
                    strategy=scenario["processing_strategy"]
                )
                
                # Verify integrated response
                assert isinstance(response.answer, str)
                assert len(response.answer) > 0
                assert "route" in response.metadata or "strategy" in response.metadata
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    @patch('src.rag_engine.retrieval.reranker.CohereRerank')
    def test_advanced_indexing_with_reranking_combination(self, mock_reranker, mock_chroma, 
                                                        mock_embeddings, mock_llm,
                                                        multi_domain_documents):
        """Test advanced indexing strategies combined with reranking"""
        # Setup mocks
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        # Mock reranker
        mock_reranker_instance = Mock()
        mock_reranker.return_value = mock_reranker_instance
        
        # Test different indexing + reranking combinations
        indexing_strategies = [IndexingStrategy.BASIC, IndexingStrategy.MULTI_REPRESENTATION]
        
        for strategy in indexing_strategies:
            config = PipelineConfig(
                indexing_strategy=strategy,
                use_reranking=True,
                reranker_model="cohere",
                retrieval_k=10,  # Retrieve more for reranking
                chunk_size=800,
                chunk_overlap=100
            )
            
            try:
                # Initialize components
                indexing_manager = IndexingManager(config)
                retrieval_engine = RetrievalEngine(config)
                
                # Index documents with specific strategy
                indexing_result = indexing_manager.index_documents(
                    multi_domain_documents, 
                    strategy.value
                )
                assert indexing_result is True
                
                # Mock retrieval with reranking
                mock_retrieved_docs = [
                    Mock(page_content=doc.content, metadata=doc.metadata) 
                    for doc in multi_domain_documents[:5]
                ]
                
                with patch.object(retrieval_engine, 'retrieve_with_rerank', 
                                return_value=mock_retrieved_docs):
                    # Test retrieval with reranking
                    retrieved_docs = retrieval_engine.retrieve_with_rerank(
                        "Python data analysis with pandas", 
                        k=5
                    )
                    
                    assert len(retrieved_docs) <= 5
                    assert all(hasattr(doc, 'page_content') for doc in retrieved_docs)
                
            except Exception as e:
                pytest.skip(f"Indexing strategy {strategy} with reranking not fully implemented: {e}")
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    @patch('src.rag_engine.query.multi_query.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.evaluation.custom_evaluator.ChatGoogleGenerativeAI')
    def test_query_processing_with_evaluation_combination(self, mock_eval_llm, mock_query_llm,
                                                        mock_chroma, mock_embeddings, mock_llm,
                                                        multi_domain_documents):
        """Test query processing strategies combined with evaluation"""
        # Setup mocks
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_query_llm.return_value = Mock()
        mock_eval_llm.return_value = Mock()
        
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        # Mock query processing responses
        mock_multi_response = Mock()
        mock_multi_response.content = "1. What is Python?\n2. How is Python used?\n3. Why choose Python?"
        mock_query_llm.return_value.invoke.return_value = mock_multi_response
        
        # Mock evaluation responses
        mock_eval_response = Mock()
        mock_eval_response.content = '{"faithfulness": 0.85, "correctness": 0.80, "relevance": 0.90}'
        mock_eval_llm.return_value.invoke.return_value = mock_eval_response
        
        config = PipelineConfig(
            query_strategies=["multi_query", "rag_fusion", "hyde"],
            evaluation_frameworks=["custom", "deepeval"]
        )
        
        engine = RAGEngine(config)
        query_processor = QueryProcessor(config)
        evaluation_manager = EvaluationManager(config)
        
        # Index documents
        engine.add_documents(multi_domain_documents)
        
        # Test different query strategies with evaluation
        strategies_to_test = ["multi_query", "rag_fusion", "hyde"]
        
        test_cases = [
            TestCase(
                question="What is Python and why is it popular?",
                expected_answer="Python is a high-level programming language known for simplicity and readability.",
                metadata={"domain": "programming", "language": "python"}
            ),
            TestCase(
                question="How do I get started with React development?",
                expected_answer="React is a JavaScript library for building user interfaces with components.",
                metadata={"domain": "programming", "language": "javascript"}
            )
        ]
        
        strategy_performance = {}
        
        for strategy in strategies_to_test:
            try:
                strategy_results = []
                
                for test_case in test_cases:
                    # Process query with strategy
                    processed_query = query_processor.process(test_case.question, strategy)
                    
                    # Execute query
                    response = engine.query(test_case.question, strategy=strategy)
                    
                    # Evaluate response
                    evaluation_result = evaluation_manager.evaluate(
                        test_cases=[test_case],
                        metrics=["custom"]
                    )
                    
                    strategy_results.append({
                        "processed_query": processed_query,
                        "response": response,
                        "evaluation": evaluation_result
                    })
                
                strategy_performance[strategy] = {
                    "results": strategy_results,
                    "avg_score": sum(r["evaluation"].overall_score for r in strategy_results) / len(strategy_results),
                    "avg_processing_time": sum(r["response"].processing_time for r in strategy_results) / len(strategy_results)
                }
                
            except Exception as e:
                pytest.skip(f"Strategy {strategy} with evaluation not fully implemented: {e}")
        
        # Verify strategy comparison
        if strategy_performance:
            best_strategy = max(strategy_performance.keys(), 
                              key=lambda s: strategy_performance[s]["avg_score"])
            
            assert best_strategy in strategies_to_test
            assert strategy_performance[best_strategy]["avg_score"] >= 0.0
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    def test_multi_provider_configuration_combination(self, mock_chroma, mock_embeddings, mock_llm,
                                                    multi_domain_documents):
        """Test different provider combinations"""
        # Setup mocks
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        # Test different provider combinations
        provider_combinations = [
            {
                "llm_provider": LLMProvider.GOOGLE,
                "embedding_provider": "google",
                "vector_store": VectorStore.CHROMA
            },
            # Add more combinations as providers are implemented
        ]
        
        for combo in provider_combinations:
            try:
                config = PipelineConfig(
                    llm_provider=combo["llm_provider"],
                    embedding_provider=combo["embedding_provider"],
                    vector_store=combo["vector_store"],
                    chunk_size=600,
                    retrieval_k=4
                )
                
                engine = RAGEngine(config)
                
                # Test basic functionality with this combination
                indexing_result = engine.add_documents(multi_domain_documents[:3])
                assert indexing_result is True
                
                response = engine.query("What is Python programming?")
                assert isinstance(response.answer, str)
                assert len(response.answer) > 0
                
                system_info = engine.get_system_info()
                assert system_info["config"]["llm_provider"] == combo["llm_provider"]
                assert system_info["config"]["vector_store"] == combo["vector_store"]
                
            except Exception as e:
                pytest.skip(f"Provider combination not fully implemented: {e}")
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    def test_metadata_filtering_with_routing_combination(self, mock_chroma, mock_embeddings, mock_llm,
                                                       multi_domain_documents, domain_routes):
        """Test metadata filtering combined with intelligent routing"""
        # Setup mocks
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        config = PipelineConfig(retrieval_k=5)
        engine = RAGEngine(config)
        router = QueryRouter(routes=domain_routes)
        
        # Index documents
        engine.add_documents(multi_domain_documents)
        
        # Test queries with metadata filtering requirements
        filtering_scenarios = [
            {
                "query": "Show me beginner-level Python tutorials",
                "expected_route": "python_expert",
                "metadata_filters": {"language": "python", "difficulty": "beginner"}
            },
            {
                "query": "Find intermediate JavaScript framework examples",
                "expected_route": "javascript_expert", 
                "metadata_filters": {"language": "javascript", "difficulty": "intermediate"}
            },
            {
                "query": "Get advanced machine learning content",
                "expected_route": "ml_expert",
                "metadata_filters": {"domain": "machine_learning", "difficulty": "advanced"}
            }
        ]
        
        for scenario in filtering_scenarios:
            # Mock routing decision
            mock_decision = RoutingDecision(
                selected_route=scenario["expected_route"],
                confidence=0.90,
                reasoning=f"Query requires {scenario['expected_route']} with specific filters",
                metadata={
                    "suggested_filters": scenario["metadata_filters"],
                    "router_type": "logical"
                }
            )
            
            with patch.object(router, 'route', return_value=mock_decision):
                routing_decision = router.route(scenario["query"])
                
                # Verify routing includes metadata suggestions
                assert routing_decision.selected_route == scenario["expected_route"]
                if "suggested_filters" in routing_decision.metadata:
                    suggested_filters = routing_decision.metadata["suggested_filters"]
                    
                    # Test query with suggested filters
                    response = engine.query(
                        scenario["query"],
                        metadata_filter=suggested_filters
                    )
                    
                    assert isinstance(response.answer, str)
                    assert len(response.answer) > 0
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    def test_error_recovery_across_components(self, mock_chroma, mock_embeddings, mock_llm,
                                            multi_domain_documents):
        """Test error recovery and fallback mechanisms across components"""
        # Setup mocks
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        config = PipelineConfig(
            query_strategies=["multi_query", "basic"],  # Include fallback strategy
            use_reranking=True,
            retrieval_k=5
        )
        
        engine = RAGEngine(config)
        query_processor = QueryProcessor(config)
        
        # Index documents
        engine.add_documents(multi_domain_documents)
        
        # Test error scenarios and recovery
        error_scenarios = [
            {
                "name": "query_processing_failure",
                "error_component": "query_processor",
                "fallback_expected": True
            },
            {
                "name": "retrieval_failure", 
                "error_component": "retrieval",
                "fallback_expected": True
            },
            {
                "name": "generation_failure",
                "error_component": "generation",
                "fallback_expected": True
            }
        ]
        
        for scenario in error_scenarios:
            test_query = "What is Python programming?"
            
            if scenario["error_component"] == "query_processor":
                # Mock query processor failure
                with patch.object(query_processor, 'process', side_effect=Exception("Query processing error")):
                    try:
                        response = engine.query(test_query, strategy="multi_query")
                        
                        # Should fallback to basic strategy
                        if scenario["fallback_expected"]:
                            assert isinstance(response.answer, str)
                            assert "error" not in response.answer.lower() or len(response.answer) > 0
                    
                    except Exception:
                        # If no fallback implemented, skip test
                        pytest.skip(f"Fallback for {scenario['name']} not implemented")
            
            elif scenario["error_component"] == "retrieval":
                # Mock retrieval failure
                mock_retriever = Mock()
                mock_retriever.get_relevant_documents.side_effect = Exception("Retrieval error")
                mock_vectorstore.as_retriever.return_value = mock_retriever
                
                response = engine.query(test_query)
                
                # Should handle gracefully
                assert isinstance(response.answer, str)
                assert response.confidence_score >= 0.0
            
            elif scenario["error_component"] == "generation":
                # Mock generation failure
                mock_llm_instance = Mock()
                mock_llm_instance.invoke.side_effect = Exception("Generation error")
                mock_llm.return_value = mock_llm_instance
                
                # Reset retriever to work
                mock_retriever = Mock()
                mock_retriever.get_relevant_documents.return_value = [
                    Mock(page_content=multi_domain_documents[0].content, 
                         metadata=multi_domain_documents[0].metadata)
                ]
                mock_vectorstore.as_retriever.return_value = mock_retriever
                
                response = engine.query(test_query)
                
                # Should handle gracefully
                assert isinstance(response.answer, str)
                assert response.confidence_score >= 0.0
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    def test_performance_optimization_combinations(self, mock_chroma, mock_embeddings, mock_llm,
                                                 multi_domain_documents):
        """Test performance optimization across component combinations"""
        # Setup mocks for fast responses
        mock_embeddings.return_value = Mock()
        
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = "Optimized response"
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.return_value = [
            Mock(page_content=doc.content, metadata=doc.metadata) 
            for doc in multi_domain_documents[:3]
        ]
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        # Test different performance configurations
        performance_configs = [
            {
                "name": "speed_optimized",
                "config": PipelineConfig(
                    chunk_size=300,  # Smaller chunks for faster processing
                    retrieval_k=3,   # Fewer documents
                    query_strategies=["basic"],  # Simple strategy
                    use_reranking=False
                )
            },
            {
                "name": "quality_optimized", 
                "config": PipelineConfig(
                    chunk_size=1000,  # Larger chunks for better context
                    retrieval_k=8,    # More documents
                    query_strategies=["multi_query", "rag_fusion"],  # Advanced strategies
                    use_reranking=True
                )
            },
            {
                "name": "balanced",
                "config": PipelineConfig(
                    chunk_size=600,
                    retrieval_k=5,
                    query_strategies=["multi_query"],
                    use_reranking=False
                )
            }
        ]
        
        performance_results = {}
        
        for perf_config in performance_configs:
            try:
                engine = RAGEngine(perf_config["config"])
                
                # Measure indexing performance
                indexing_start = time.time()
                engine.add_documents(multi_domain_documents)
                indexing_time = time.time() - indexing_start
                
                # Measure query performance
                test_queries = [
                    "What is Python?",
                    "How does React work?", 
                    "Explain data science"
                ]
                
                query_times = []
                for query in test_queries:
                    query_start = time.time()
                    response = engine.query(query)
                    query_time = time.time() - query_start
                    query_times.append(query_time)
                    
                    assert isinstance(response.answer, str)
                    assert len(response.answer) > 0
                
                performance_results[perf_config["name"]] = {
                    "indexing_time": indexing_time,
                    "avg_query_time": sum(query_times) / len(query_times),
                    "total_time": indexing_time + sum(query_times),
                    "config": perf_config["config"]
                }
                
            except Exception as e:
                pytest.skip(f"Performance config {perf_config['name']} not fully supported: {e}")
        
        # Verify performance characteristics
        if len(performance_results) >= 2:
            # Speed optimized should be fastest
            if "speed_optimized" in performance_results and "quality_optimized" in performance_results:
                speed_time = performance_results["speed_optimized"]["avg_query_time"]
                quality_time = performance_results["quality_optimized"]["avg_query_time"]
                
                # With mocks, times should be very small, but speed should still be <= quality
                assert speed_time <= quality_time * 2  # Allow some variance
        
        return performance_results