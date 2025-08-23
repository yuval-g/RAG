# Environment Variable Loading

This document explains how environment variables are loaded consistently across all entry points in the RAG Engine.

## Problem

Previously, `load_dotenv()` was not being called early enough in some entry points, causing environment variables from `.env` files to not be available when modules were imported and initialized.

## Solution

We've implemented a comprehensive solution that ensures environment variables are loaded early and consistently:

### 1. Core Environment Loader

**File**: `src/rag_engine/core/env_loader.py`

- Provides `ensure_env_loaded()` function that safely loads environment variables
- Can be called multiple times without side effects
- Automatically searches for `.env` files in current and parent directories
- Handles cases where `python-dotenv` is not installed

### 2. Early Loading in Entry Points

All major entry points now load environment variables early:

- **CLI**: `rag_cli.py` and `src/rag_engine/cli/main.py`
- **API Server**: `src/rag_engine/api/main.py` and `src/rag_engine/api/app.py`
- **Core Config**: `src/rag_engine/core/config.py`
- **LLM Providers**: `src/rag_engine/generation/llm_providers.py`

### 3. Backward Compatibility

Examples and tests still use the traditional `load_dotenv()` pattern for backward compatibility and clarity.

## Usage

### For New Modules

```python
# At the top of your module
from ..core.env_loader import ensure_env_loaded
ensure_env_loaded()

# Now you can safely use os.getenv()
import os
api_key = os.getenv("GOOGLE_API_KEY")
```

### For Entry Points

```python
#!/usr/bin/env python3
"""
Your entry point script
"""
# Load environment variables first
from src.rag_engine.core.env_loader import ensure_env_loaded
ensure_env_loaded()

# Then import other modules
from your_module import main

if __name__ == "__main__":
    main()
```

### For Examples and Tests

```python
# Traditional pattern (still works)
from dotenv import load_dotenv
load_dotenv()

# Your code here
```

## Testing

Use the provided test script to verify environment loading:

```bash
python test_env_loading.py
```

This will test:
- Direct env_loader import
- Config system import
- LLM providers import
- Environment variable access
- CLI and API imports

## Environment File Discovery

The env_loader automatically searches for `.env` files in:

1. Current working directory
2. Parent directory
3. Grandparent directory
4. Great-grandparent directory

This ensures `.env` files are found regardless of where the script is run from.

## Best Practices

1. **Always load early**: Import and call `ensure_env_loaded()` before importing modules that need environment variables

2. **Use in entry points**: All entry points should load environment variables before importing application modules

3. **Safe to call multiple times**: `ensure_env_loaded()` can be called multiple times without issues

4. **Fallback gracefully**: The system works even if `python-dotenv` is not installed

5. **Test your setup**: Use `test_env_loading.py` to verify your environment is configured correctly

## Common Issues Fixed

- ✅ API server not finding environment variables when started directly
- ✅ CLI commands failing when `.env` file exists but isn't loaded
- ✅ LLM providers not finding API keys during initialization
- ✅ Configuration system not seeing environment overrides
- ✅ Inconsistent behavior between different entry points

## Migration Guide

If you have existing code that's not loading environment variables properly:

1. Add the env_loader import at the top of your module:
   ```python
   from ..core.env_loader import ensure_env_loaded
   ensure_env_loaded()
   ```

2. For entry point scripts, load before importing application modules:
   ```python
   from src.rag_engine.core.env_loader import ensure_env_loaded
   ensure_env_loaded()
   ```

3. Test with `python test_env_loading.py` to verify the fix works