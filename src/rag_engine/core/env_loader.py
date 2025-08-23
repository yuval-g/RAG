"""
Environment variable loading utility.
Ensures consistent dotenv loading across all entry points.
"""

import os
from pathlib import Path
from typing import Optional


def load_environment_variables(dotenv_path: Optional[str] = None) -> None:
    """
    Load environment variables from .env file.
    
    This function ensures consistent environment variable loading
    across all entry points and modules.
    
    Args:
        dotenv_path: Optional path to .env file. If None, searches for .env
                    in current directory and parent directories.
    """
    try:
        from dotenv import load_dotenv
        
        if dotenv_path:
            # Load from specific path
            load_dotenv(dotenv_path)
        else:
            # Search for .env file in current and parent directories
            current_dir = Path.cwd()
            
            # Check current directory and up to 3 parent directories
            for i in range(4):
                env_file = current_dir / ".env"
                if env_file.exists():
                    load_dotenv(env_file)
                    break
                current_dir = current_dir.parent
            else:
                # If no .env file found, just load from environment
                load_dotenv()
                
    except ImportError:
        # python-dotenv not installed, skip loading
        pass


def ensure_env_loaded() -> None:
    """
    Ensure environment variables are loaded.
    
    This is a safe function that can be called multiple times
    without side effects.
    """
    # Check if we've already loaded (simple heuristic)
    if not hasattr(ensure_env_loaded, '_loaded'):
        load_environment_variables()
        ensure_env_loaded._loaded = True


# Load environment variables when this module is imported
ensure_env_loaded()