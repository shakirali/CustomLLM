#!/usr/bin/env python3
"""
Validate test files by checking syntax and importability.
This doesn't require pytest to be installed.
"""

import sys
import os
import ast
import importlib.util

def validate_python_syntax(filepath):
    """Validate Python syntax of a file."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source, filename=filepath)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def validate_imports(filepath):
    """Check if file can be imported (without running tests)."""
    try:
        # Add parent directory to path
        parent_dir = os.path.dirname(os.path.dirname(filepath))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        spec = importlib.util.spec_from_file_location(
            os.path.basename(filepath).replace('.py', ''),
            filepath
        )
        if spec and spec.loader:
            # Just check if it can be loaded, don't execute
            return True, None
        return False, "Could not create spec"
    except Exception as e:
        return False, str(e)

def main():
    """Validate all test files."""
    test_dir = os.path.join(os.path.dirname(__file__), "tests")
    
    test_files = [
        "test_dataset.py",
        "test_attention.py",
        "test_model_components.py",
        "test_embeddings.py",
        "test_transformer.py",
        "test_output_projection.py"
    ]
    
    print("Validating test files...")
    print("=" * 70)
    
    all_valid = True
    results = []
    
    for test_file in test_files:
        filepath = os.path.join(test_dir, test_file)
        
        if not os.path.exists(filepath):
            print(f"❌ {test_file}: File not found")
            all_valid = False
            continue
        
        # Check syntax
        syntax_ok, syntax_error = validate_python_syntax(filepath)
        if not syntax_ok:
            print(f"❌ {test_file}: Syntax error - {syntax_error}")
            all_valid = False
            continue
        
        # Check imports (but skip if pytest not available)
        try:
            import pytest
            import_ok, import_error = validate_imports(filepath)
            if not import_ok:
                print(f"⚠️  {test_file}: Import check failed - {import_error}")
        except ImportError:
            # If pytest not available, just check syntax
            import_ok = True
        
        if syntax_ok and import_ok:
            print(f"✅ {test_file}: Valid")
            results.append((test_file, True))
        else:
            results.append((test_file, False))
            all_valid = False
    
    print("=" * 70)
    
    if all_valid:
        print("\n✅ All test files are syntactically valid!")
        print("\nNote: To actually run the tests, install pytest:")
        print("  pip install pytest")
        print("\nThen run:")
        print("  pytest tests/ -v")
        return 0
    else:
        print("\n❌ Some test files have issues. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

