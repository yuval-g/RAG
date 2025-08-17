"""
Main entry point for the RAG API server
"""

import os
import logging
from typing import Optional

import uvicorn
from .app import create_app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    config_path: Optional[str] = None,
    environment: Optional[str] = None,
    reload: bool = False,
    log_level: str = "info"
):
    """Run the RAG API server"""
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create the FastAPI app
    app = create_app(config_path=config_path, environment=environment)
    
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )


if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("RAG_API_HOST", "0.0.0.0")
    port = int(os.getenv("RAG_API_PORT", "8000"))
    config_path = os.getenv("RAG_CONFIG_PATH")
    environment = os.getenv("RAG_ENVIRONMENT", "development")
    reload = os.getenv("RAG_API_RELOAD", "false").lower() == "true"
    log_level = os.getenv("RAG_LOG_LEVEL", "info")
    
    run_server(
        host=host,
        port=port,
        config_path=config_path,
        environment=environment,
        reload=reload,
        log_level=log_level
    )