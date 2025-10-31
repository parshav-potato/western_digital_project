"""
Code processing module for extracting code from Python files.
"""
import os
import glob
import ast
from typing import List, Optional


class CodeProcessor:
    """Extract and process code from Python codebases."""
    
    def __init__(self):
        """Initialize code processor."""
        pass
    
    def extract_code_chunks(
        self, 
        codebase_path: str, 
        max_files: Optional[int] = None
    ) -> List[str]:
        """
        Extract code chunks from Python files in a codebase.
        
        Extracts:
        - Module-level docstrings
        - Classes with their docstrings, methods, and source code
        - Top-level functions with docstrings, arguments, and source code
        
        Args:
            codebase_path: Path to the codebase directory
            max_files: Maximum number of files to process (None = all)
            
        Returns:
            List of code chunk strings formatted for embedding
        """
        code_chunks = []
        
        # Find all Python files
        python_files = glob.glob(
            os.path.join(codebase_path, "**/*.py"), 
            recursive=True
        )
        
        if max_files:
            python_files = python_files[:max_files]
        
        print(f"Found {len(python_files)} Python files")
        
        for file_path in python_files:
            try:
                chunks = self._extract_from_file(file_path, codebase_path)
                code_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Warning: Error processing {file_path}: {str(e)[:50]}")
                continue
        
        return code_chunks
    
    def _extract_from_file(
        self, 
        file_path: str, 
        codebase_path: str
    ) -> List[str]:
        """
        Extract code chunks from a single Python file.
        
        Args:
            file_path: Path to Python file
            codebase_path: Base path for relative path calculation
            
        Returns:
            List of code chunks from this file
        """
        chunks = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the AST
        tree = ast.parse(source)
        relative_path = os.path.relpath(file_path, codebase_path)
        
        # Extract module-level docstring
        module_doc = ast.get_docstring(tree)
        if module_doc:
            chunk = f"# File: {relative_path}\n\n{module_doc}\n"
            chunks.append(chunk)
        
        # Extract classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                chunk = self._extract_class(node, relative_path, source)
                if chunk:
                    chunks.append(chunk)
            
            elif isinstance(node, ast.FunctionDef):
                # Only get top-level functions (not methods)
                if node.col_offset == 0:
                    chunk = self._extract_function(node, relative_path, source)
                    if chunk:
                        chunks.append(chunk)
        
        return chunks
    
    def _extract_class(
        self, 
        node: ast.ClassDef, 
        relative_path: str, 
        source: str
    ) -> Optional[str]:
        """
        Extract class information including docstring and methods.
        
        Args:
            node: AST ClassDef node
            relative_path: Relative file path
            source: Full source code
            
        Returns:
            Formatted class chunk string
        """
        class_doc = ast.get_docstring(node) or "No description"
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        
        chunk = f"# File: {relative_path}\n"
        chunk += f"# Class: {node.name}\n\n"
        chunk += f"{class_doc}\n\n"
        chunk += f"Methods: {', '.join(methods)}\n"
        
        # Get the source code for the class (first 20 lines)
        try:
            source_lines = source.split('\n')
            class_start = node.lineno - 1
            class_source = '\n'.join(source_lines[class_start:class_start + 20])
            chunk += f"\n```python\n{class_source}\n...\n```"
        except:
            pass
        
        return chunk
    
    def _extract_function(
        self, 
        node: ast.FunctionDef, 
        relative_path: str, 
        source: str
    ) -> Optional[str]:
        """
        Extract function information including docstring and signature.
        
        Args:
            node: AST FunctionDef node
            relative_path: Relative file path
            source: Full source code
            
        Returns:
            Formatted function chunk string
        """
        func_doc = ast.get_docstring(node) or "No description"
        args = [arg.arg for arg in node.args.args]
        
        chunk = f"# File: {relative_path}\n"
        chunk += f"# Function: {node.name}\n\n"
        chunk += f"Arguments: {', '.join(args)}\n\n"
        chunk += f"{func_doc}\n"
        
        # Get the source code for the function (first 15 lines)
        try:
            source_lines = source.split('\n')
            func_start = node.lineno - 1
            func_source = '\n'.join(source_lines[func_start:func_start + 15])
            chunk += f"\n```python\n{func_source}\n...\n```"
        except:
            pass
        
        return chunk
    
    def get_file_stats(self, codebase_path: str) -> dict:
        """
        Get statistics about a codebase.
        
        Args:
            codebase_path: Path to the codebase directory
            
        Returns:
            Dictionary with codebase statistics
        """
        python_files = glob.glob(
            os.path.join(codebase_path, "**/*.py"), 
            recursive=True
        )
        
        total_classes = 0
        total_functions = 0
        total_lines = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                    total_lines += len(source.split('\n'))
                
                tree = ast.parse(source)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        total_classes += 1
                    elif isinstance(node, ast.FunctionDef):
                        if node.col_offset == 0:  # Top-level only
                            total_functions += 1
            except:
                continue
        
        return {
            "total_files": len(python_files),
            "total_classes": total_classes,
            "total_functions": total_functions,
            "total_lines": total_lines,
        }
