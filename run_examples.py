#!/usr/bin/env python3
"""
Run all examples and log results to examples.log
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_example(example_path: str, log_file):
    """Run a single example and log the results"""
    example_name = os.path.basename(example_path)
    
    print(f"Running {example_name}...")
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"RUNNING: {example_name}\n")
    log_file.write(f"TIME: {datetime.now().isoformat()}\n")
    log_file.write(f"{'='*80}\n")
    log_file.flush()
    
    try:
        # Run the example with timeout using uv
        result = subprocess.run(
            ["uv", "run", "python", example_path],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Log stdout
        if result.stdout:
            log_file.write("STDOUT:\n")
            log_file.write(result.stdout)
            log_file.write("\n")
        
        # Log stderr
        if result.stderr:
            log_file.write("STDERR:\n")
            log_file.write(result.stderr)
            log_file.write("\n")
        
        # Log return code
        log_file.write(f"RETURN CODE: {result.returncode}\n")
        
        if result.returncode == 0:
            print(f"✅ {example_name} - SUCCESS")
            return True
        else:
            print(f"❌ {example_name} - FAILED (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        log_file.write("ERROR: Example timed out after 120 seconds\n")
        print(f"⏰ {example_name} - TIMEOUT")
        return False
    except Exception as e:
        log_file.write(f"ERROR: {str(e)}\n")
        print(f"💥 {example_name} - ERROR: {str(e)}")
        return False
    finally:
        log_file.write(f"\nCOMPLETED: {datetime.now().isoformat()}\n")
        log_file.write("-" * 80 + "\n")
        log_file.flush()

def main():
    """Run all examples and log results"""
    
    # Get all Python example files
    examples_dir = "examples"
    example_files = []
    
    for file in os.listdir(examples_dir):
        if file.startswith("example_") and file.endswith(".py"):
            example_files.append(os.path.join(examples_dir, file))
    
    example_files.sort()
    
    print(f"🚀 Running {len(example_files)} examples...")
    print(f"📝 Logging to examples.log")
    
    # Open log file
    with open("examples.log", "w") as log_file:
        # Write header
        log_file.write(f"RAG ENGINE EXAMPLES TEST RUN\n")
        log_file.write(f"Started: {datetime.now().isoformat()}\n")
        log_file.write(f"Python: {sys.version}\n")
        log_file.write(f"Working Directory: {os.getcwd()}\n")
        log_file.write(f"Examples to run: {len(example_files)}\n")
        log_file.write("=" * 80 + "\n")
        log_file.flush()
        
        # Run each example
        successful = 0
        failed = 0
        
        for example_path in example_files:
            if run_example(example_path, log_file):
                successful += 1
            else:
                failed += 1
            
            # Small delay between examples
            time.sleep(1)
        
        # Write summary
        log_file.write(f"\n{'='*80}\n")
        log_file.write("SUMMARY\n")
        log_file.write(f"{'='*80}\n")
        log_file.write(f"Total examples: {len(example_files)}\n")
        log_file.write(f"Successful: {successful}\n")
        log_file.write(f"Failed: {failed}\n")
        log_file.write(f"Success rate: {successful/len(example_files)*100:.1f}%\n")
        log_file.write(f"Completed: {datetime.now().isoformat()}\n")
        
        print(f"\n📊 Results:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📈 Success rate: {successful/len(example_files)*100:.1f}%")
        print(f"   📝 Full log saved to examples.log")

if __name__ == "__main__":
    main()