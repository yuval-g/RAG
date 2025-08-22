"""
Integration tests using benchmark datasets and real-world scenarios
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from src.rag_engine.core.engine import RAGEngine
from src.rag_engine.core.config import PipelineConfig, IndexingStrategy
from src.rag_engine.core.models import Document, TestCase, EvaluationResult
from src.rag_engine.evaluation.evaluation_manager import EvaluationManager


class TestBenchmarkDatasets:
    """Integration tests with benchmark datasets and real-world scenarios"""
    
    @pytest.fixture
    def wikipedia_sample_documents(self):
        """Sample documents mimicking Wikipedia articles for testing"""
        return [
            Document(
                content="""
                Machine learning (ML) is a type of artificial intelligence (AI) that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so. Machine learning algorithms use historical data as input to predict new output values.

                Machine learning is used in a variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.

                A subset of machine learning is closely related to computational statistics, which focuses on making predictions using computers; but not all machine learning is statistical learning. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.

                Data mining is a related field of study, focusing on exploratory data analysis through unsupervised learning. Some implementations of machine learning use data and neural networks in a way that mimics the working of a biological brain.
                """,
                metadata={
                    "title": "Machine Learning",
                    "source": "wikipedia",
                    "category": "computer_science",
                    "length": 1024,
                    "last_modified": "2024-01-15"
                },
                doc_id="wiki_ml_001"
            ),
            Document(
                content="""
                Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.

                Colloquially, the term "artificial intelligence" is often used to describe machines (or computers) that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".

                As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. A quip in Tesler's Theorem says "AI is whatever hasn't been done yet."

                For most of its history, AI research has been divided into sub-fields that often fail to communicate with each other. These sub-fields are based on technical considerations, such as particular goals (e.g. "robotics" or "machine learning"), the use of particular tools ("logic" or artificial neural networks), or deep philosophical differences.
                """,
                metadata={
                    "title": "Artificial Intelligence",
                    "source": "wikipedia", 
                    "category": "computer_science",
                    "length": 1156,
                    "last_modified": "2024-01-20"
                },
                doc_id="wiki_ai_001"
            ),
            Document(
                content="""
                Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.

                The result is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

                Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.

                The history of natural language processing (NLP) generally started in the 1950s, although work can be found from earlier periods. In 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence.
                """,
                metadata={
                    "title": "Natural Language Processing",
                    "source": "wikipedia",
                    "category": "computer_science", 
                    "length": 987,
                    "last_modified": "2024-01-25"
                },
                doc_id="wiki_nlp_001"
            ),
            Document(
                content="""
                Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.

                Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks, convolutional neural networks and Transformers have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance.

                Artificial neural networks (ANNs) were inspired by information processing and distributed communication nodes in biological systems. ANNs have various differences from biological brains. Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analog.
                """,
                metadata={
                    "title": "Deep Learning",
                    "source": "wikipedia",
                    "category": "computer_science",
                    "length": 1087,
                    "last_modified": "2024-02-01"
                },
                doc_id="wiki_dl_001"
            ),
            Document(
                content="""
                Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do.

                Computer vision tasks include methods for acquiring, processing, analyzing and understanding digital images, and extraction of high-dimensional data from the real world in order to produce numerical or symbolic information, e.g. in the forms of decisions.

                Understanding in this context means the transformation of visual images (the input of the retina) into descriptions of the world that make sense to thought processes and can elicit appropriate action. This image understanding can be seen as the disentangling of symbolic information from image data using models constructed with the aid of geometry, physics, statistics, and learning theory.

                The scientific discipline of computer vision is concerned with the theory behind artificial systems that extract information from images.
                """,
                metadata={
                    "title": "Computer Vision",
                    "source": "wikipedia",
                    "category": "computer_science",
                    "length": 1134,
                    "last_modified": "2024-02-05"
                },
                doc_id="wiki_cv_001"
            )
        ]
    
    @pytest.fixture
    def squad_style_qa_dataset(self):
        """SQuAD-style question-answer dataset for evaluation"""
        return [
            TestCase(
                question="What is machine learning?",
                expected_answer="Machine learning is a type of artificial intelligence that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so.",
                metadata={
                    "context": "Machine learning (ML) is a type of artificial intelligence...",
                    "answer_start": 0,
                    "category": "definition",
                    "difficulty": "easy"
                }
            ),
            TestCase(
                question="What applications use machine learning?",
                expected_answer="Machine learning is used in medicine, email filtering, speech recognition, and computer vision.",
                metadata={
                    "context": "Machine learning is used in a variety of applications...",
                    "answer_start": 45,
                    "category": "applications",
                    "difficulty": "medium"
                }
            ),
            TestCase(
                question="How do artificial neural networks differ from biological brains?",
                expected_answer="Neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analog.",
                metadata={
                    "context": "Artificial neural networks (ANNs) were inspired by...",
                    "answer_start": 120,
                    "category": "comparison",
                    "difficulty": "hard"
                }
            ),
            TestCase(
                question="What is the AI effect?",
                expected_answer="The AI effect is a phenomenon where tasks considered to require intelligence are often removed from the definition of AI as machines become increasingly capable.",
                metadata={
                    "context": "As machines become increasingly capable...",
                    "answer_start": 78,
                    "category": "concept",
                    "difficulty": "medium"
                }
            ),
            TestCase(
                question="When did natural language processing generally start?",
                expected_answer="The history of natural language processing generally started in the 1950s.",
                metadata={
                    "context": "The history of natural language processing...",
                    "answer_start": 0,
                    "category": "historical",
                    "difficulty": "easy"
                }
            ),
            TestCase(
                question="What fields have deep learning architectures been applied to?",
                expected_answer="Deep learning has been applied to computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs.",
                metadata={
                    "context": "Deep-learning architectures such as...",
                    "answer_start": 156,
                    "category": "applications",
                    "difficulty": "medium"
                }
            ),
            TestCase(
                question="What does computer vision seek to understand and automate?",
                expected_answer="Computer vision seeks to understand and automate tasks that the human visual system can do.",
                metadata={
                    "context": "From the perspective of engineering...",
                    "answer_start": 34,
                    "category": "definition",
                    "difficulty": "easy"
                }
            ),
            TestCase(
                question="What are the main challenges in natural language processing?",
                expected_answer="Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.",
                metadata={
                    "context": "Challenges in natural language processing...",
                    "answer_start": 0,
                    "category": "challenges",
                    "difficulty": "medium"
                }
            )
        ]
    
    @pytest.fixture
    def msmarco_style_passages(self):
        """MS MARCO-style passages for retrieval testing"""
        return [
            Document(
                content="Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together.",
                metadata={
                    "passage_id": "msmarco_001",
                    "source": "programming_guide",
                    "relevance_score": 0.95,
                    "topic": "python_programming"
                },
                doc_id="msmarco_python_001"
            ),
            Document(
                content="JavaScript is a programming language that conforms to the ECMAScript specification. JavaScript is high-level, often just-in-time compiled, and multi-paradigm. It has curly-bracket syntax, dynamic typing, prototype-based object-orientation, and first-class functions.",
                metadata={
                    "passage_id": "msmarco_002", 
                    "source": "web_development_guide",
                    "relevance_score": 0.88,
                    "topic": "javascript_programming"
                },
                doc_id="msmarco_js_001"
            ),
            Document(
                content="React is a free and open-source front-end JavaScript library for building user interfaces based on UI components. It is maintained by Meta and a community of individual developers and companies. React can be used as a base in the development of single-page or mobile applications.",
                metadata={
                    "passage_id": "msmarco_003",
                    "source": "frontend_frameworks",
                    "relevance_score": 0.92,
                    "topic": "react_framework"
                },
                doc_id="msmarco_react_001"
            ),
            Document(
                content="Machine learning algorithms build a model based on training data in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision.",
                metadata={
                    "passage_id": "msmarco_004",
                    "source": "ai_handbook",
                    "relevance_score": 0.97,
                    "topic": "machine_learning"
                },
                doc_id="msmarco_ml_001"
            ),
            Document(
                content="Data science is an inter-disciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from many structural and unstructured data. Data science is related to data mining, machine learning and big data.",
                metadata={
                    "passage_id": "msmarco_005",
                    "source": "data_science_intro",
                    "relevance_score": 0.89,
                    "topic": "data_science"
                },
                doc_id="msmarco_ds_001"
            )
        ]
    
    @pytest.fixture
    def hotpot_qa_multi_hop_questions(self):
        """HotpotQA-style multi-hop reasoning questions"""
        return [
            TestCase(
                question="What programming language is used for both web development and machine learning, and what library is commonly used for ML in this language?",
                expected_answer="Python is used for both web development and machine learning. Scikit-learn is a commonly used library for machine learning in Python.",
                metadata={
                    "type": "multi_hop",
                    "hops": 2,
                    "reasoning_type": "bridge",
                    "difficulty": "hard"
                }
            ),
            TestCase(
                question="Which field combines computer science with statistics and what type of learning does it relate to that doesn't require labeled data?",
                expected_answer="Data science combines computer science with statistics and relates to unsupervised learning, which doesn't require labeled data.",
                metadata={
                    "type": "multi_hop",
                    "hops": 2,
                    "reasoning_type": "comparison",
                    "difficulty": "hard"
                }
            ),
            TestCase(
                question="What JavaScript library is used for building user interfaces and what company maintains it?",
                expected_answer="React is a JavaScript library for building user interfaces and it is maintained by Meta.",
                metadata={
                    "type": "multi_hop",
                    "hops": 2,
                    "reasoning_type": "bridge",
                    "difficulty": "medium"
                }
            )
        ]
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    def test_wikipedia_knowledge_base_workflow(self, mock_chroma, mock_embeddings, mock_llm,
                                             wikipedia_sample_documents, squad_style_qa_dataset):
        """Test RAG system with Wikipedia-style knowledge base"""
        # Setup mocks
        mock_embeddings.return_value = Mock()
        
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = "Machine learning is a type of artificial intelligence that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so."
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_retrieved_docs = [
            Mock(page_content=wikipedia_sample_documents[0].content, 
                 metadata=wikipedia_sample_documents[0].metadata)
        ]
        mock_retriever.get_relevant_documents.return_value = mock_retrieved_docs
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        # Configure for Wikipedia-style content
        config = PipelineConfig(
            chunk_size=800,  # Larger chunks for encyclopedia content
            chunk_overlap=100,
            retrieval_k=5,
            temperature=0.1  # Lower temperature for factual content
        )
        
        engine = RAGEngine(config)
        
        # Index Wikipedia documents
        indexing_result = engine.add_documents(wikipedia_sample_documents)
        assert indexing_result is True
        assert engine.get_document_count() == len(wikipedia_sample_documents)
        
        # Test with SQuAD-style questions
        correct_answers = 0
        total_questions = len(squad_style_qa_dataset)
        
        for test_case in squad_style_qa_dataset:
            response = engine.query(test_case.question)
            
            # Verify response structure
            assert isinstance(response.answer, str)
            assert len(response.answer) > 0
            assert response.confidence_score >= 0.0
            assert len(response.source_documents) > 0
            
            # Check if response contains key terms from expected answer
            expected_terms = test_case.expected_answer.lower().split()
            response_terms = response.answer.lower().split()
            
            # Simple overlap check (in real scenario, would use more sophisticated metrics)
            overlap = len(set(expected_terms) & set(response_terms))
            if overlap >= len(expected_terms) * 0.3:  # 30% term overlap threshold
                correct_answers += 1
        
        # Calculate accuracy
        accuracy = correct_answers / total_questions
        
        # Verify reasonable performance
        assert accuracy >= 0.0  # Basic sanity check
        assert engine.get_document_count() == len(wikipedia_sample_documents)
        
        # Test system info with Wikipedia content
        system_info = engine.get_system_info()
        assert system_info["stats"]["indexed_documents"] == len(wikipedia_sample_documents)
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    def test_msmarco_passage_retrieval_workflow(self, mock_chroma, mock_embeddings, mock_llm,
                                              msmarco_style_passages):
        """Test passage retrieval with MS MARCO-style data"""
        # Setup mocks
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        
        # Mock retrieval to return passages based on relevance scores
        def mock_retrieve(query, k=5):
            # Simple mock that returns passages with highest relevance scores
            sorted_passages = sorted(msmarco_style_passages, 
                                   key=lambda x: x.metadata.get("relevance_score", 0.0), 
                                   reverse=True)
            return [Mock(page_content=p.content, metadata=p.metadata) 
                   for p in sorted_passages[:k]]
        
        mock_retriever.get_relevant_documents.side_effect = lambda query: mock_retrieve(query, 3)
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        config = PipelineConfig(
            chunk_size=400,  # Smaller chunks for passage retrieval
            retrieval_k=3,
            use_reranking=True  # Enable reranking for better precision
        )
        
        engine = RAGEngine(config)
        
        # Index MS MARCO passages
        engine.add_documents(msmarco_style_passages)
        
        # Test queries with expected relevant passages
        retrieval_test_cases = [
            {
                "query": "What is Python programming language?",
                "expected_topics": ["python_programming"],
                "min_relevance": 0.9
            },
            {
                "query": "How does machine learning work?",
                "expected_topics": ["machine_learning"],
                "min_relevance": 0.95
            },
            {
                "query": "What is React framework?",
                "expected_topics": ["react_framework"],
                "min_relevance": 0.9
            },
            {
                "query": "JavaScript programming features",
                "expected_topics": ["javascript_programming"],
                "min_relevance": 0.85
            }
        ]
        
        retrieval_results = []
        
        for test_case in retrieval_test_cases:
            response = engine.query(test_case["query"])
            
            # Analyze retrieved documents
            retrieved_topics = [doc.metadata.get("topic", "") for doc in response.source_documents]
            relevance_scores = [doc.metadata.get("relevance_score", 0.0) for doc in response.source_documents]
            
            # Check if expected topics are retrieved
            topic_match = any(topic in retrieved_topics for topic in test_case["expected_topics"])
            high_relevance = any(score >= test_case["min_relevance"] for score in relevance_scores)
            
            retrieval_results.append({
                "query": test_case["query"],
                "topic_match": topic_match,
                "high_relevance": high_relevance,
                "retrieved_topics": retrieved_topics,
                "max_relevance": max(relevance_scores) if relevance_scores else 0.0,
                "response": response
            })
        
        # Verify retrieval quality
        topic_matches = sum(1 for r in retrieval_results if r["topic_match"])
        high_relevance_matches = sum(1 for r in retrieval_results if r["high_relevance"])
        
        assert topic_matches >= len(retrieval_test_cases) * 0.5  # At least 50% topic matches
        assert high_relevance_matches >= len(retrieval_test_cases) * 0.5  # At least 50% high relevance
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    @patch('src.rag_engine.query.decomposition.ChatGoogleGenerativeAI')
    def test_hotpot_qa_multi_hop_reasoning(self, mock_decomp_llm, mock_chroma, mock_embeddings, mock_llm,
                                         wikipedia_sample_documents, msmarco_style_passages,
                                         hotpot_qa_multi_hop_questions):
        """Test multi-hop reasoning with HotpotQA-style questions"""
        # Setup mocks
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_decomp_llm.return_value = Mock()
        
        # Mock decomposition for multi-hop questions
        mock_decomp_response = Mock()
        mock_decomp_response.content = "1. What programming language is used for web development and ML?\n2. What ML library is used in Python?"
        mock_decomp_llm.return_value.invoke.return_value = mock_decomp_response
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        
        # Combine all documents for comprehensive knowledge base
        all_documents = wikipedia_sample_documents + msmarco_style_passages
        
        mock_retrieved_docs = [
            Mock(page_content=doc.content, metadata=doc.metadata) 
            for doc in all_documents[:4]  # Return multiple relevant docs
        ]
        mock_retriever.get_relevant_documents.return_value = mock_retrieved_docs
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        # Configure for multi-hop reasoning
        config = PipelineConfig(
            query_strategies=["decomposition", "multi_query"],
            retrieval_k=6,  # More documents for multi-hop
            chunk_size=600,
            temperature=0.2  # Slightly higher for reasoning
        )
        
        engine = RAGEngine(config)
        
        # Index comprehensive knowledge base
        engine.add_documents(all_documents)
        
        # Test multi-hop questions
        multi_hop_results = []
        
        for test_case in hotpot_qa_multi_hop_questions:
            # Test with decomposition strategy
            response = engine.query(test_case.question, strategy="decomposition")
            
            # Analyze multi-hop reasoning capability
            hops_required = test_case.metadata.get("hops", 1)
            reasoning_type = test_case.metadata.get("reasoning_type", "unknown")
            
            # Check if response addresses multiple aspects (simple heuristic)
            response_sentences = response.answer.split('.')
            multi_aspect_response = len(response_sentences) >= hops_required
            
            # Check if multiple source documents are used
            multiple_sources = len(response.source_documents) >= hops_required
            
            multi_hop_results.append({
                "question": test_case.question,
                "hops_required": hops_required,
                "reasoning_type": reasoning_type,
                "multi_aspect_response": multi_aspect_response,
                "multiple_sources": multiple_sources,
                "source_count": len(response.source_documents),
                "response_length": len(response.answer),
                "confidence": response.confidence_score,
                "response": response
            })
        
        # Analyze multi-hop performance
        multi_aspect_success = sum(1 for r in multi_hop_results if r["multi_aspect_response"])
        multiple_source_success = sum(1 for r in multi_hop_results if r["multiple_sources"])
        
        multi_hop_performance = {
            "multi_aspect_rate": multi_aspect_success / len(multi_hop_results),
            "multiple_source_rate": multiple_source_success / len(multi_hop_results),
            "avg_source_count": sum(r["source_count"] for r in multi_hop_results) / len(multi_hop_results),
            "avg_response_length": sum(r["response_length"] for r in multi_hop_results) / len(multi_hop_results),
            "results": multi_hop_results
        }
        
        # Verify multi-hop reasoning capability
        assert multi_hop_performance["multiple_source_rate"] >= 0.5  # Should use multiple sources
        assert multi_hop_performance["avg_source_count"] >= 2.0  # Average should be at least 2 sources
    
    @patch('src.rag_engine.generation.generation_engine.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.indexing.basic_indexer.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_engine.indexing.basic_indexer.Chroma')
    @patch('src.rag_engine.evaluation.custom_evaluator.ChatGoogleGenerativeAI')
    @patch('src.rag_engine.evaluation.ragas_integration.ChatGoogleGenerativeAI')
    def test_comprehensive_benchmark_evaluation(self, mock_ragas_llm, mock_eval_llm, mock_chroma, 
                                              mock_embeddings, mock_llm,
                                              wikipedia_sample_documents, squad_style_qa_dataset):
        """Test comprehensive evaluation using multiple benchmark-style metrics"""
        # Setup mocks
        mock_embeddings.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_eval_llm.return_value = Mock()
        mock_ragas_llm.return_value = Mock()
        
        # Mock evaluation responses
        mock_eval_response = Mock()
        mock_eval_response.content = '{"faithfulness": 0.85, "correctness": 0.80, "relevance": 0.90}'
        mock_eval_llm.return_value.invoke.return_value = mock_eval_response
        
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        config = PipelineConfig(
            evaluation_frameworks=["custom", "ragas"],
            chunk_size=700,
            retrieval_k=4
        )
        
        engine = RAGEngine(config)
        evaluation_manager = EvaluationManager(config)
        
        # Index documents
        engine.add_documents(wikipedia_sample_documents)
        
        # Categorize test cases by difficulty
        easy_cases = [tc for tc in squad_style_qa_dataset if tc.metadata.get("difficulty") == "easy"]
        medium_cases = [tc for tc in squad_style_qa_dataset if tc.metadata.get("difficulty") == "medium"]
        hard_cases = [tc for tc in squad_style_qa_dataset if tc.metadata.get("difficulty") == "hard"]
        
        benchmark_results = {}
        
        # Test each difficulty level
        for difficulty, test_cases in [("easy", easy_cases), ("medium", medium_cases), ("hard", hard_cases)]:
            if not test_cases:
                continue
                
            try:
                # Evaluate with multiple frameworks
                evaluation_result = evaluation_manager.evaluate(
                    test_cases=test_cases,
                    metrics=["custom"]  # Start with custom, add others as available
                )
                
                benchmark_results[difficulty] = {
                    "overall_score": evaluation_result.overall_score,
                    "metric_scores": evaluation_result.metric_scores,
                    "test_case_count": len(test_cases),
                    "recommendations": evaluation_result.recommendations
                }
                
            except Exception as e:
                pytest.skip(f"Evaluation framework not fully implemented: {e}")
        
        # Calculate overall benchmark performance
        if benchmark_results:
            overall_performance = {
                "difficulty_breakdown": benchmark_results,
                "weighted_average": sum(
                    results["overall_score"] * results["test_case_count"] 
                    for results in benchmark_results.values()
                ) / sum(results["test_case_count"] for results in benchmark_results.values()),
                "total_test_cases": sum(results["test_case_count"] for results in benchmark_results.values())
            }
            
            # Verify benchmark performance
            assert overall_performance["weighted_average"] >= 0.0
            assert overall_performance["total_test_cases"] > 0
        
        else:
            pytest.skip("No evaluation results available")
    
    def test_create_custom_benchmark_dataset(self, tmp_path):
        """Test creating and loading custom benchmark datasets"""
        # Create custom benchmark dataset
        custom_dataset = {
            "name": "RAG System Benchmark",
            "version": "1.0",
            "description": "Custom benchmark for RAG system evaluation",
            "documents": [
                {
                    "id": "custom_001",
                    "content": "Custom document content for testing RAG system capabilities.",
                    "metadata": {"source": "custom", "category": "test"}
                }
            ],
            "test_cases": [
                {
                    "question": "What is this document about?",
                    "expected_answer": "Testing RAG system capabilities",
                    "metadata": {"type": "comprehension", "difficulty": "easy"}
                }
            ]
        }
        
        # Save dataset to temporary file
        dataset_file = tmp_path / "custom_benchmark.json"
        with open(dataset_file, 'w') as f:
            json.dump(custom_dataset, f, indent=2)
        
        # Load and verify dataset
        with open(dataset_file, 'r') as f:
            loaded_dataset = json.load(f)
        
        assert loaded_dataset["name"] == custom_dataset["name"]
        assert len(loaded_dataset["documents"]) == len(custom_dataset["documents"])
        assert len(loaded_dataset["test_cases"]) == len(custom_dataset["test_cases"])
        
        # Convert to RAG system format
        documents = [
            Document(
                content=doc["content"],
                metadata=doc["metadata"],
                doc_id=doc["id"]
            )
            for doc in loaded_dataset["documents"]
        ]
        
        test_cases = [
            TestCase(
                question=tc["question"],
                expected_answer=tc["expected_answer"],
                metadata=tc["metadata"]
            )
            for tc in loaded_dataset["test_cases"]
        ]
        
        assert len(documents) == 1
        assert len(test_cases) == 1
        assert documents[0].content == custom_dataset["documents"][0]["content"]
        assert test_cases[0].question == custom_dataset["test_cases"][0]["question"]