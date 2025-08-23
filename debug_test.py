import pytest
from unittest.mock import Mock, patch

from src.rag_engine.query.processor import QueryProcessor
from src.rag_engine.core.exceptions import QueryProcessingError
from src.rag_engine.query.multi_query import MultiQueryGenerator

def test_init_failure_debug():
    """Debug test to see what exception is actually raised"""
    # Test proper initialization
    with patch.object(MultiQueryGenerator, '__init__') as mock_multi:
        mock_multi.side_effect = Exception("Initialization failed")
        
        try:
            processor = QueryProcessor(llm_model="gemini-2.0-flash-lite")
            print("No exception was raised!")
        except Exception as e:
            print(f"Exception raised: {type(e).__name__}: {e}")
            # This should be the expected behavior
            assert isinstance(e, QueryProcessingError)
            assert "Failed to initialize query processors" in str(e)

if __name__ == "__main__":
    test_init_failure_debug()