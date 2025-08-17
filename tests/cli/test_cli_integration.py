"""
Integration tests for the RAG CLI
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from src.rag_engine.cli.main import cli
from src.rag_engine.core.models import Document, RAGResponse, EvaluationResult, EvaluationTestCase
from src.rag_engine.core.config import PipelineConfig


@pytest.fixture
def runner():
    """CLI test runner"""
    return CliRunner()


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return PipelineConfig(
        llm_provider="google",
        llm_model="gemini-2.0-flash-lite",
        embedding_provider="openai",
        vector_store="chroma",
        enable_logging=False
    )


@pytest.fixture
def mock_rag_engine():
    """Mock RAG engine for testing"""
    engine = Mock()
    
    # Mock system info
    engine.get_system_info.return_value = {
        "version": "0.1.0",
        "config": {
            "llm_provider": "google",
            "llm_model": "gemini-2.0-flash-lite",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-ada-002",
            "vector_store": "chroma",
            "indexing_strategy": "basic",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_k": 5,
        },
        "components": {
            "indexer": True,
            "retriever": True,
            "query_processor": False,
            "router": False,
            "generator": True,
            "evaluator": False,
        },
        "stats": {
            "indexed_documents": 5,
            "indexed_chunks": 25,
            "retriever_ready": True,
        }
    }
    
    # Mock query response
    engine.query.return_value = RAGResponse(
        answer="This is a test answer from the RAG system.",
        source_documents=[
            Document(
                content="Test document content for the CLI test",
                metadata={"source": "test.txt"},
                doc_id="test-doc-1"
            )
        ],
        confidence_score=0.85,
        processing_time=1.23,
        metadata={"query": "test question", "retrieved_count": 1}
    )
    
    # Mock document operations
    engine.add_documents.return_value = True
    engine.clear_documents.return_value = True
    engine.load_web_documents.return_value = True
    engine.get_document_count.return_value = 5
    engine.get_chunk_count.return_value = 25
    engine.is_ready.return_value = True
    
    # Mock evaluation
    engine.evaluate.return_value = EvaluationResult(
        overall_score=0.8,
        metric_scores={"faithfulness": 0.85, "relevancy": 0.75},
        test_case_results=[],
        recommendations=["Consider improving document quality"]
    )
    
    return engine


@pytest.fixture
def mock_engine_creation(mock_rag_engine, mock_config):
    """Mock engine creation for CLI tests"""
    with patch('src.rag_engine.cli.main.create_engine') as mock_create:
        mock_create.return_value = mock_rag_engine
        with patch('src.rag_engine.cli.main.load_config') as mock_load_config:
            mock_load_config.return_value = mock_config
            yield mock_create


class TestCLIBasicCommands:
    """Test basic CLI commands"""
    
    def test_cli_help(self, runner):
        """Test CLI help command"""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'RAG Engine CLI' in result.output
        assert 'serve' in result.output
        assert 'query' in result.output
        assert 'index' in result.output
        assert 'evaluate' in result.output
        assert 'status' in result.output
        assert 'config' in result.output
    
    def test_cli_version_info(self, runner, mock_engine_creation):
        """Test CLI status command"""
        result = runner.invoke(cli, ['status'])
        assert result.exit_code == 0
        # Should contain system information
        assert 'RAG System Status' in result.output or 'Version' in result.output


class TestQueryCommand:
    """Test query command"""
    
    def test_query_success_text_format(self, runner, mock_engine_creation):
        """Test successful query with text output"""
        result = runner.invoke(cli, ['query', 'What is the capital of France?'])
        assert result.exit_code == 0
        assert 'Answer' in result.output
        assert 'This is a test answer' in result.output
        assert 'Confidence' in result.output
        assert 'Processing time' in result.output
    
    def test_query_success_json_format(self, runner, mock_engine_creation):
        """Test successful query with JSON output"""
        result = runner.invoke(cli, [
            'query', 'What is the capital of France?',
            '--output-format', 'json'
        ])
        assert result.exit_code == 0
        
        # Parse JSON output
        output_data = json.loads(result.output)
        assert 'question' in output_data
        assert 'answer' in output_data
        assert 'confidence_score' in output_data
        assert output_data['confidence_score'] == 0.85
    
    def test_query_with_options(self, runner, mock_engine_creation):
        """Test query with various options"""
        result = runner.invoke(cli, [
            'query', 'Test question',
            '--k', '3',
            '--include-sources'
        ])
        assert result.exit_code == 0
        # Check that the query succeeded and basic elements are present
        assert 'Answer' in result.output
        assert 'Confidence' in result.output
        assert 'Processing time' in result.output
    
    def test_query_with_sources_json(self, runner, mock_engine_creation):
        """Test query with sources in JSON format"""
        result = runner.invoke(cli, [
            'query', 'Test question',
            '--include-sources',
            '--output-format', 'json'
        ])
        assert result.exit_code == 0
        
        output_data = json.loads(result.output)
        assert 'source_documents' in output_data
        assert len(output_data['source_documents']) > 0
        assert output_data['source_documents'][0]['metadata']['source'] == 'test.txt'
    
    def test_query_without_sources(self, runner, mock_engine_creation):
        """Test query without source documents"""
        result = runner.invoke(cli, [
            'query', 'Test question',
            '--output-format', 'json'
        ])
        assert result.exit_code == 0
        
        output_data = json.loads(result.output)
        assert 'source_documents' in output_data
    
    def test_query_engine_not_ready(self, runner, mock_engine_creation):
        """Test query when engine is not ready"""
        mock_engine_creation.return_value.is_ready.return_value = False
        
        result = runner.invoke(cli, ['query', 'Test question'])
        assert result.exit_code == 1
        assert 'No documents indexed' in result.output


class TestIndexCommands:
    """Test indexing commands"""
    
    def test_index_files_help(self, runner):
        """Test index files help"""
        result = runner.invoke(cli, ['index', 'files', '--help'])
        assert result.exit_code == 0
        assert 'Index documents from files' in result.output
    
    def test_index_files_success(self, runner, mock_engine_creation):
        """Test successful file indexing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file1 = Path(temp_dir) / "test1.txt"
            test_file2 = Path(temp_dir) / "test2.md"
            
            test_file1.write_text("This is test content 1")
            test_file2.write_text("This is test content 2")
            
            result = runner.invoke(cli, [
                'index', 'files', str(test_file1), str(test_file2)
            ])
            assert result.exit_code == 0
            assert 'Successfully indexed' in result.output
    
    def test_index_files_recursive(self, runner, mock_engine_creation):
        """Test recursive file indexing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested directory structure
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()
            
            test_file = subdir / "test.txt"
            test_file.write_text("This is test content")
            
            result = runner.invoke(cli, [
                'index', 'files', temp_dir,
                '--recursive'
            ])
            assert result.exit_code == 0
            assert 'Successfully indexed' in result.output
    
    def test_index_files_with_pattern(self, runner, mock_engine_creation):
        """Test file indexing with pattern"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            txt_file = Path(temp_dir) / "test.txt"
            py_file = Path(temp_dir) / "test.py"
            
            txt_file.write_text("This is a text file")
            py_file.write_text("print('This is a Python file')")
            
            result = runner.invoke(cli, [
                'index', 'files', temp_dir,
                '--recursive',
                '--pattern', '*.txt'
            ])
            assert result.exit_code == 0
    
    def test_index_web_success(self, runner, mock_engine_creation):
        """Test successful web indexing"""
        result = runner.invoke(cli, [
            'index', 'web',
            'https://example.com',
            'https://example.org'
        ])
        assert result.exit_code == 0
        assert 'Successfully loaded' in result.output
    
    def test_index_web_with_options(self, runner, mock_engine_creation):
        """Test web indexing with options"""
        result = runner.invoke(cli, [
            'index', 'web',
            'https://example.com',
            '--max-depth', '2',
            '--include-links'
        ])
        assert result.exit_code == 0
    
    def test_clear_index_with_confirm(self, runner, mock_engine_creation):
        """Test clearing index with confirmation"""
        result = runner.invoke(cli, [
            'index', 'clear',
            '--confirm'
        ])
        assert result.exit_code == 0
        assert 'Successfully cleared' in result.output
    
    def test_clear_index_interactive(self, runner, mock_engine_creation):
        """Test clearing index with interactive confirmation"""
        result = runner.invoke(cli, ['index', 'clear'], input='y\n')
        assert result.exit_code == 0
        assert 'Successfully cleared' in result.output


class TestEvaluateCommand:
    """Test evaluate command"""
    
    def test_evaluate_success(self, runner, mock_engine_creation):
        """Test successful evaluation"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {
                "test_cases": [
                    {
                        "question": "What is the capital of France?",
                        "expected_answer": "Paris",
                        "context": [
                            {
                                "content": "Paris is the capital of France.",
                                "metadata": {"source": "geography.txt"}
                            }
                        ],
                        "metadata": {"category": "geography"}
                    }
                ]
            }
            json.dump(test_data, f)
            test_file = f.name
        
        try:
            result = runner.invoke(cli, ['evaluate', test_file])
            assert result.exit_code == 0
            assert 'Evaluation Results' in result.output
            assert 'Overall Score' in result.output
        finally:
            os.unlink(test_file)
    
    def test_evaluate_json_output(self, runner, mock_engine_creation):
        """Test evaluation with JSON output"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {
                "test_cases": [
                    {
                        "question": "Test question",
                        "expected_answer": "Test answer",
                        "context": [],
                        "metadata": {}
                    }
                ]
            }
            json.dump(test_data, f)
            test_file = f.name
        
        try:
            result = runner.invoke(cli, [
                'evaluate', test_file,
                '--output-format', 'json'
            ])
            assert result.exit_code == 0
            
            # Parse JSON output (extract JSON part if there's other text)
            output_lines = result.output.strip().split('\n')
            json_start = -1
            for i, line in enumerate(output_lines):
                if line.strip().startswith('{'):
                    json_start = i
                    break
            
            if json_start >= 0:
                json_output = '\n'.join(output_lines[json_start:])
                output_data = json.loads(json_output)
            else:
                output_data = json.loads(result.output)
            
            assert 'overall_score' in output_data
            assert 'metric_scores' in output_data
        finally:
            os.unlink(test_file)
    
    def test_evaluate_with_output_file(self, runner, mock_engine_creation):
        """Test evaluation with output file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {
                "test_cases": [
                    {
                        "question": "Test question",
                        "expected_answer": "Test answer",
                        "context": [],
                        "metadata": {}
                    }
                ]
            }
            json.dump(test_data, f)
            test_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_f:
            output_file = output_f.name
        
        try:
            result = runner.invoke(cli, [
                'evaluate', test_file,
                '--output', output_file
            ])
            assert result.exit_code == 0
            assert f'saved to {output_file}' in result.output
            
            # Check output file exists and contains data
            assert Path(output_file).exists()
            with open(output_file, 'r') as f:
                output_data = json.load(f)
                assert 'overall_score' in output_data
        finally:
            os.unlink(test_file)
            if Path(output_file).exists():
                os.unlink(output_file)
    
    def test_evaluate_missing_file(self, runner, mock_engine_creation):
        """Test evaluation with missing test file"""
        result = runner.invoke(cli, ['evaluate', 'nonexistent.json'])
        assert result.exit_code == 1
        assert 'Test file not found' in result.output


class TestConfigCommands:
    """Test configuration commands"""
    
    def test_config_show_yaml(self, runner, mock_engine_creation):
        """Test showing configuration in YAML format"""
        result = runner.invoke(cli, ['config', 'show'])
        assert result.exit_code == 0
        # Should contain configuration data
        assert 'llm_provider' in result.output or 'google' in result.output
    
    def test_config_show_json(self, runner, mock_engine_creation):
        """Test showing configuration in JSON format"""
        result = runner.invoke(cli, ['config', 'show', '--format', 'json'])
        assert result.exit_code == 0
        
        # Should be valid JSON
        try:
            json.loads(result.output)
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")
    
    def test_config_validate_success(self, runner):
        """Test configuration validation with valid file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'llm_provider': 'google',
                'llm_model': 'gemini-2.0-flash-lite',
                'embedding_provider': 'openai',
                'vector_store': 'chroma'
            }
            import yaml
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            with patch('src.rag_engine.cli.main.ConfigurationManager') as MockConfigManager:
                mock_manager = Mock()
                mock_manager.validate_config_file.return_value = True
                MockConfigManager.return_value = mock_manager
                
                result = runner.invoke(cli, ['config', 'validate', config_file])
                assert result.exit_code == 0
                assert 'is valid' in result.output
        finally:
            os.unlink(config_file)


class TestServeCommand:
    """Test serve command"""
    
    def test_serve_help(self, runner):
        """Test serve command help"""
        result = runner.invoke(cli, ['serve', '--help'])
        assert result.exit_code == 0
        assert 'Start the RAG API server' in result.output
        assert '--host' in result.output
        assert '--port' in result.output
        assert '--reload' in result.output


class TestCLIOptions:
    """Test CLI global options"""
    
    def test_global_config_option(self, runner, mock_engine_creation):
        """Test global config option"""
        result = runner.invoke(cli, [
            '--config', 'test_config.yaml',
            'status'
        ])
        # Should not fail due to config option
        assert result.exit_code == 0
    
    def test_global_environment_option(self, runner, mock_engine_creation):
        """Test global environment option"""
        result = runner.invoke(cli, [
            '--environment', 'testing',
            'status'
        ])
        # Should not fail due to environment option
        assert result.exit_code == 0
    
    def test_global_verbose_option(self, runner, mock_engine_creation):
        """Test global verbose option"""
        result = runner.invoke(cli, [
            '--verbose',
            'status'
        ])
        # Should not fail due to verbose option
        assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__])