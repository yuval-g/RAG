import pytest
from unittest.mock import Mock, patch

from src.rag_engine.query.processor import QueryProcessor
from src.rag_engine.core.exceptions import QueryProcessingError
from src.rag_engine.query.multi_query import MultiQueryGenerator

def test_init_failure_debug():
    """Debug test to see what exception is actually raised"""
    # Let's manually call the _initialize_processors method to see what happens
    processor = QueryProcessor.__new__(QueryProcessor)  # Create instance without calling __init__
    processor.llm_model = "gemini-2.0-flash-lite"
    processor.temperature = 0.0
    processor.llm_kwargs = {}
    
    # Patch the MultiQueryGenerator class directly
    with patch.object(MultiQueryGenerator, '__init__') as mock_multi:
        mock_multi.side_effect = Exception("Initialization failed")
        
        try:
            processor._initialize_processors()
            print("No exception was raised!")
        except Exception as e:
            print(f"Exception raised: {type(e).__name__}: {e}")
            raise

if __name__ == "__main__":
    test_init_failure_debug()