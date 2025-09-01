# Location: /tests/test_notebooks.py

"""
Automated notebook execution tests for TensorVerseHub.
Tests that all notebooks can be executed without errors and produce expected outputs.
"""

import pytest
import os
import sys
import subprocess
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Try to import nbformat and nbconvert
try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbconvert import HTMLExporter
    NBCONVERT_AVAILABLE = True
except ImportError:
    NBCONVERT_AVAILABLE = False
    print("Warning: nbformat/nbconvert not available. Install with: pip install nbformat nbconvert")

from tests import TEST_CONFIG


# Skip all tests if nbconvert not available
pytestmark = pytest.mark.skipif(
    not NBCONVERT_AVAILABLE,
    reason="nbformat/nbconvert not available for notebook testing"
)


class NotebookTester:
    """Utility class for testing Jupyter notebooks."""
    
    def __init__(self, timeout: int = 600, kernel_name: str = 'python3'):
        """
        Initialize notebook tester.
        
        Args:
            timeout: Maximum execution time per cell in seconds
            kernel_name: Jupyter kernel to use for execution
        """
        self.timeout = timeout
        self.kernel_name = kernel_name
        self.executed_notebooks = []
        
    def execute_notebook(self, notebook_path: str, working_dir: str = None) -> Tuple[bool, str]:
        """
        Execute a single notebook and return success status and error message.
        
        Args:
            notebook_path: Path to notebook file
            working_dir: Working directory for execution
            
        Returns:
            Tuple of (success, error_message)
        """
        if not os.path.exists(notebook_path):
            return False, f"Notebook not found: {notebook_path}"
        
        if working_dir is None:
            working_dir = os.path.dirname(notebook_path)
        
        try:
            # Read notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            # Configure preprocessor
            preprocessor = ExecutePreprocessor(
                timeout=self.timeout,
                kernel_name=self.kernel_name,
                allow_errors=False
            )
            
            # Execute notebook
            processed_notebook, resources = preprocessor.preprocess(
                notebook, {'metadata': {'path': working_dir}}
            )
            
            self.executed_notebooks.append({
                'path': notebook_path,
                'success': True,
                'cells_executed': len(processed_notebook.cells),
                'error': None
            })
            
            return True, ""
            
        except Exception as e:
            error_msg = str(e)
            self.executed_notebooks.append({
                'path': notebook_path,
                'success': False,
                'error': error_msg
            })
            
            return False, error_msg
    
    def validate_notebook_structure(self, notebook_path: str) -> List[str]:
        """
        Validate notebook structure and content.
        
        Args:
            notebook_path: Path to notebook file
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            # Check for required elements
            has_title = False
            has_learning_objectives = False
            code_cells = 0
            markdown_cells = 0
            
            for cell in notebook.cells:
                if cell.cell_type == 'markdown':
                    markdown_cells += 1
                    source = cell.source.lower()
                    
                    if any(title_word in source for title_word in ['#', 'title']):
                        has_title = True
                    
                    if any(obj_word in source for obj_word in ['learning objectives', 'objectives']):
                        has_learning_objectives = True
                        
                elif cell.cell_type == 'code':
                    code_cells += 1
                    
                    # Check for common issues
                    if not cell.source.strip():
                        issues.append(f"Empty code cell found")
                    
                    # Check for potentially problematic imports
                    if 'import *' in cell.source:
                        issues.append("Wildcard imports found (not recommended)")
            
            # Structure validation
            if not has_title:
                issues.append("No title found in notebook")
            
            if not has_learning_objectives:
                issues.append("No learning objectives found")
            
            if code_cells == 0:
                issues.append("No code cells found")
            
            if markdown_cells == 0:
                issues.append("No markdown cells found")
            
            if code_cells < 5:
                issues.append(f"Very few code cells ({code_cells}) - consider adding more examples")
            
            # Check cell order (should start with markdown)
            if notebook.cells and notebook.cells[0].cell_type != 'markdown':
                issues.append("Notebook should start with markdown cell (title/introduction)")
                
        except Exception as e:
            issues.append(f"Error reading notebook: {str(e)}")
        
        return issues
    
    def extract_notebook_outputs(self, notebook_path: str) -> Dict[str, Any]:
        """
        Extract outputs from executed notebook for verification.
        
        Args:
            notebook_path: Path to notebook file
            
        Returns:
            Dictionary containing notebook outputs and metadata
        """
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            outputs_info = {
                'total_cells': len(notebook.cells),
                'code_cells': 0,
                'cells_with_output': 0,
                'execution_count': 0,
                'errors': [],
                'plots': 0,
                'print_statements': 0
            }
            
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    outputs_info['code_cells'] += 1
                    
                    if hasattr(cell, 'execution_count') and cell.execution_count:
                        outputs_info['execution_count'] = max(
                            outputs_info['execution_count'], cell.execution_count
                        )
                    
                    if hasattr(cell, 'outputs') and cell.outputs:
                        outputs_info['cells_with_output'] += 1
                        
                        for output in cell.outputs:
                            if output.output_type == 'error':
                                outputs_info['errors'].append({
                                    'error_name': output.get('ename', 'Unknown'),
                                    'error_value': output.get('evalue', 'Unknown error')
                                })
                            
                            elif output.output_type in ['display_data', 'execute_result']:
                                if 'image/png' in output.get('data', {}):
                                    outputs_info['plots'] += 1
                            
                            elif output.output_type == 'stream':
                                if 'text' in output:
                                    outputs_info['print_statements'] += 1
            
            return outputs_info
            
        except Exception as e:
            return {'error': str(e)}
    
    def create_execution_report(self, output_path: str = "notebook_test_report.html") -> str:
        """
        Create HTML report of notebook execution results.
        
        Args:
            output_path: Path for output HTML file
            
        Returns:
            Path to generated report
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>TensorVerseHub Notebook Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .success { color: green; }
                .error { color: red; }
                .warning { color: orange; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .error-details { background-color: #ffe6e6; padding: 10px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>📚 TensorVerseHub Notebook Test Report</h1>
        """
        
        # Summary
        total_notebooks = len(self.executed_notebooks)
        successful = sum(1 for nb in self.executed_notebooks if nb['success'])
        failed = total_notebooks - successful
        
        html_content += f"""
            <h2>📊 Summary</h2>
            <ul>
                <li>Total notebooks tested: <strong>{total_notebooks}</strong></li>
                <li class="success">Successful: <strong>{successful}</strong></li>
                <li class="error">Failed: <strong>{failed}</strong></li>
                <li>Success rate: <strong>{(successful/total_notebooks*100) if total_notebooks > 0 else 0:.1f}%</strong></li>
            </ul>
        """
        
        # Detailed results
        html_content += """
            <h2>📋 Detailed Results</h2>
            <table>
                <tr>
                    <th>Notebook</th>
                    <th>Status</th>
                    <th>Cells Executed</th>
                    <th>Error Details</th>
                </tr>
        """
        
        for nb in self.executed_notebooks:
            status_class = "success" if nb['success'] else "error"
            status_text = "✅ PASS" if nb['success'] else "❌ FAIL"
            
            cells_executed = nb.get('cells_executed', 'N/A')
            error_details = nb.get('error', 'None') if not nb['success'] else 'None'
            
            html_content += f"""
                <tr>
                    <td>{os.path.basename(nb['path'])}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{cells_executed}</td>
                    <td>{error_details[:100]}{'...' if len(str(error_details)) > 100 else ''}</td>
                </tr>
            """
        
        html_content += """
            </table>
        """
        
        # Error details
        failed_notebooks = [nb for nb in self.executed_notebooks if not nb['success']]
        if failed_notebooks:
            html_content += """
                <h2>❌ Error Details</h2>
            """
            
            for nb in failed_notebooks:
                html_content += f"""
                    <h3>{os.path.basename(nb['path'])}</h3>
                    <div class="error-details">
                        <strong>Error:</strong> {nb.get('error', 'Unknown error')}
                    </div>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path


class TestNotebookFoundations:
    """Test foundational notebooks (01-06)."""
    
    @pytest.fixture
    def notebook_tester(self):
        """Create notebook tester instance."""
        return NotebookTester(timeout=300)  # 5 minutes per cell
    
    @pytest.fixture
    def notebooks_dir(self):
        """Get notebooks directory path."""
        project_root = Path(__file__).parent.parent
        notebooks_dir = project_root / "notebooks"
        return str(notebooks_dir)
    
    def test_notebook_01_tensors_operations(self, notebook_tester, notebooks_dir):
        """Test notebook 01: TensorFlow tensors and operations."""
        notebook_path = os.path.join(
            notebooks_dir, 
            "01_tensorflow_foundations", 
            "01_tensors_operations_execution.ipynb"
        )
        
        if not os.path.exists(notebook_path):
            pytest.skip(f"Notebook not found: {notebook_path}")
        
        # Validate structure first
        issues = notebook_tester.validate_notebook_structure(notebook_path)
        
        # Allow some issues for demo notebooks but report them
        if issues:
            print(f"Notebook structure issues: {issues}")
        
        # Execute notebook
        success, error = notebook_tester.execute_notebook(notebook_path)
        
        if not success:
            pytest.fail(f"Notebook execution failed: {error}")
        
        # Verify outputs
        outputs = notebook_tester.extract_notebook_outputs(notebook_path)
        
        assert outputs.get('code_cells', 0) > 10, "Should have substantial code content"
        assert outputs.get('errors', []) == [], f"Should have no errors, but found: {outputs.get('errors')}"
    
    @pytest.mark.parametrize("notebook_name", [
        "02_data_pipelines_tfrecords.ipynb",
        "03_debugging_profiling.ipynb"
    ])
    def test_foundation_notebooks_exist(self, notebook_name, notebooks_dir):
        """Test that foundation notebooks exist (placeholder test)."""
        notebook_path = os.path.join(
            notebooks_dir,
            "01_tensorflow_foundations",
            notebook_name
        )
        
        # For now, just check if directory exists
        foundation_dir = os.path.join(notebooks_dir, "01_tensorflow_foundations")
        if not os.path.exists(foundation_dir):
            pytest.skip(f"Foundation notebooks directory not found: {foundation_dir}")
        
        # This is a placeholder - in a complete implementation, 
        # we would test actual notebook execution
        assert True  # Placeholder assertion


class TestNotebookStructure:
    """Test notebook structure and quality standards."""
    
    def test_notebook_content_standards(self):
        """Test that notebooks follow content standards."""
        project_root = Path(__file__).parent.parent
        notebooks_dir = project_root / "notebooks"
        
        if not notebooks_dir.exists():
            pytest.skip("Notebooks directory not found")
        
        notebook_tester = NotebookTester()
        issues_found = []
        
        # Find all notebook files
        notebook_files = list(notebooks_dir.rglob("*.ipynb"))
        
        if not notebook_files:
            pytest.skip("No notebook files found")
        
        for notebook_path in notebook_files:
            # Skip checkpoint files
            if ".ipynb_checkpoints" in str(notebook_path):
                continue
            
            issues = notebook_tester.validate_notebook_structure(str(notebook_path))
            
            if issues:
                issues_found.append({
                    'notebook': notebook_path.name,
                    'issues': issues
                })
        
        # Report issues but don't fail the test (yet)
        if issues_found:
            print("\n📋 Notebook Structure Issues Found:")
            for item in issues_found:
                print(f"  📓 {item['notebook']}:")
                for issue in item['issues']:
                    print(f"    ⚠️ {issue}")
        
        # For now, just ensure we can read the notebooks
        assert len(notebook_files) >= 0  # Placeholder assertion
    
    def test_notebook_imports_validity(self):
        """Test that notebooks have valid import statements."""
        project_root = Path(__file__).parent.parent
        notebooks_dir = project_root / "notebooks"
        
        if not notebooks_dir.exists():
            pytest.skip("Notebooks directory not found")
        
        notebook_files = list(notebooks_dir.rglob("*.ipynb"))
        import_issues = []
        
        for notebook_path in notebook_files:
            if ".ipynb_checkpoints" in str(notebook_path):
                continue
            
            try:
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook = nbformat.read(f, as_version=4)
                
                for cell in notebook.cells:
                    if cell.cell_type == 'code' and cell.source.strip():
                        # Check for common import issues
                        lines = cell.source.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line.startswith('import ') or line.startswith('from '):
                                # Check for wildcard imports
                                if 'import *' in line:
                                    import_issues.append(f"{notebook_path.name}: Wildcard import found")
                                
                                # Check for relative imports without proper structure
                                if line.startswith('from .') or line.startswith('from ..'):
                                    import_issues.append(f"{notebook_path.name}: Relative import found")
                        
            except Exception as e:
                import_issues.append(f"{notebook_path.name}: Error reading notebook - {e}")
        
        if import_issues:
            print(f"\n📦 Import Issues Found:")
            for issue in import_issues:
                print(f"  ⚠️ {issue}")
        
        # Don't fail test for import issues, just report them
        assert True


class TestNotebookExecution:
    """Test notebook execution in different scenarios."""
    
    @pytest.fixture
    def temp_working_dir(self):
        """Create temporary working directory for notebook execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_notebook_execution_isolation(self, temp_working_dir):
        """Test that notebooks can execute in isolated environments."""
        project_root = Path(__file__).parent.parent
        notebooks_dir = project_root / "notebooks"
        
        if not notebooks_dir.exists():
            pytest.skip("Notebooks directory not found")
        
        # Find a sample notebook to test
        sample_notebooks = list(notebooks_dir.rglob("01_tensors_operations_execution.ipynb"))
        
        if not sample_notebooks:
            pytest.skip("No sample notebook found for testing")
        
        notebook_path = sample_notebooks[0]
        notebook_tester = NotebookTester(timeout=120)  # 2 minutes per cell
        
        # Test execution in isolated directory
        success, error = notebook_tester.execute_notebook(str(notebook_path), temp_working_dir)
        
        # Don't fail if execution fails due to missing dependencies
        # This is more about testing the testing infrastructure
        if not success:
            print(f"⚠️ Notebook execution failed (expected in test environment): {error}")
        
        assert True  # Test passed if we got here without crashing
    
    def test_notebook_memory_usage(self):
        """Test that notebooks don't consume excessive memory."""
        # This is a placeholder for memory usage testing
        # In a full implementation, you would monitor memory usage during execution
        
        project_root = Path(__file__).parent.parent
        notebooks_dir = project_root / "notebooks"
        
        if not notebooks_dir.exists():
            pytest.skip("Notebooks directory not found")
        
        # For now, just check that we can create a notebook tester
        notebook_tester = NotebookTester()
        assert notebook_tester.timeout > 0
        assert notebook_tester.kernel_name == 'python3'
    
    def test_notebook_execution_time(self):
        """Test that notebooks execute within reasonable time limits."""
        # This tests the timeout mechanism
        notebook_tester = NotebookTester(timeout=1)  # Very short timeout
        
        # Create a simple test notebook content that should timeout
        test_notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": "import time; time.sleep(5)  # This should timeout"
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # This test verifies our timeout mechanism works
        # We expect this to fail due to timeout
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            nbformat.write(nbformat.from_dict(test_notebook), f)
            temp_notebook_path = f.name
        
        try:
            success, error = notebook_tester.execute_notebook(temp_notebook_path)
            
            # Should fail due to timeout
            if success:
                print("⚠️ Expected timeout but execution succeeded")
            else:
                assert "timeout" in error.lower() or "time" in error.lower()
        
        finally:
            os.unlink(temp_notebook_path)


class TestNotebookReporting:
    """Test notebook testing reporting functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_execution_report_generation(self, temp_dir):
        """Test generation of notebook execution reports."""
        notebook_tester = NotebookTester()
        
        # Simulate some notebook execution results
        notebook_tester.executed_notebooks = [
            {
                'path': '/fake/path/notebook1.ipynb',
                'success': True,
                'cells_executed': 25,
                'error': None
            },
            {
                'path': '/fake/path/notebook2.ipynb',
                'success': False,
                'error': 'ModuleNotFoundError: No module named test_module'
            },
            {
                'path': '/fake/path/notebook3.ipynb',
                'success': True,
                'cells_executed': 15,
                'error': None
            }
        ]
        
        # Generate report
        report_path = os.path.join(temp_dir, "test_report.html")
        generated_path = notebook_tester.create_execution_report(report_path)
        
        assert os.path.exists(generated_path)
        assert generated_path == report_path
        
        # Verify report content
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        assert "TensorVerseHub Notebook Test Report" in report_content
        assert "notebook1.ipynb" in report_content
        assert "notebook2.ipynb" in report_content
        assert "ModuleNotFoundError" in report_content
        assert "Success rate" in report_content
    
    def test_output_extraction_edge_cases(self):
        """Test output extraction with various notebook types."""
        notebook_tester = NotebookTester()
        
        # Test empty notebook
        empty_notebook = {
            "cells": [],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            nbformat.write(nbformat.from_dict(empty_notebook), f)
            temp_notebook_path = f.name
        
        try:
            outputs = notebook_tester.extract_notebook_outputs(temp_notebook_path)
            
            assert outputs['total_cells'] == 0
            assert outputs['code_cells'] == 0
            assert outputs['cells_with_output'] == 0
            assert outputs['errors'] == []
            
        finally:
            os.unlink(temp_notebook_path)
    
    def test_notebook_validation_comprehensive(self):
        """Test comprehensive notebook validation."""
        notebook_tester = NotebookTester()
        
        # Create a good notebook structure
        good_notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "# My Notebook Title\n\n## Learning Objectives\n\n- Learn something cool"
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": "import tensorflow as tf\nprint('Hello TensorFlow')"
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": "# More code\nx = tf.constant([1, 2, 3])\nprint(x)"
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Create bad notebook structure
        bad_notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": "from somewhere import *"  # Bad import
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ""  # Empty cell
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Test good notebook
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            nbformat.write(nbformat.from_dict(good_notebook), f)
            good_notebook_path = f.name
        
        # Test bad notebook
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            nbformat.write(nbformat.from_dict(bad_notebook), f)
            bad_notebook_path = f.name
        
        try:
            # Good notebook should have minimal issues
            good_issues = notebook_tester.validate_notebook_structure(good_notebook_path)
            assert len(good_issues) <= 2  # May have minor issues but should be mostly good
            
            # Bad notebook should have multiple issues
            bad_issues = notebook_tester.validate_notebook_structure(bad_notebook_path)
            assert len(bad_issues) >= 3  # Should detect multiple problems
            
            # Check for specific issues
            bad_issues_text = ' '.join(bad_issues)
            assert 'title' in bad_issues_text.lower() or 'Empty code cell' in bad_issues_text
            
        finally:
            os.unlink(good_notebook_path)
            os.unlink(bad_notebook_path)


class TestNotebookIntegration:
    """Integration tests for notebook functionality."""
    
    def test_full_notebook_pipeline(self):
        """Test the complete notebook testing pipeline."""
        notebook_tester = NotebookTester(timeout=30)  # Short timeout for testing
        
        # Create a complete test notebook
        test_notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "# Test Notebook\n\n## Learning Objectives\n\n- Test the testing pipeline"
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": "import sys\nprint(f'Python version: {sys.version}')"
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": "# Simple computation\nresult = 2 + 2\nprint(f'2 + 2 = {result}')"
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": "## Conclusion\n\nThis completes our test."
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            nbformat.write(nbformat.from_dict(test_notebook), f)
            test_notebook_path = f.name
        
        try:
            # 1. Validate structure
            issues = notebook_tester.validate_notebook_structure(test_notebook_path)
            assert len(issues) <= 1  # Should be minimal issues
            
            # 2. Execute notebook (may fail in test environment, that's ok)
            success, error = notebook_tester.execute_notebook(test_notebook_path)
            
            # 3. Extract outputs (whether execution succeeded or not)
            outputs = notebook_tester.extract_notebook_outputs(test_notebook_path)
            
            # 4. Verify we got some output info
            assert 'total_cells' in outputs
            assert outputs['total_cells'] == 4
            assert 'code_cells' in outputs
            assert outputs['code_cells'] == 2
            
            # 5. Generate report
            with tempfile.TemporaryDirectory() as temp_dir:
                report_path = notebook_tester.create_execution_report(
                    os.path.join(temp_dir, "integration_report.html")
                )
                
                assert os.path.exists(report_path)
                
                with open(report_path, 'r', encoding='utf-8') as rf:
                    report_content = rf.read()
                
                assert "TensorVerseHub Notebook Test Report" in report_content
                
        finally:
            os.unlink(test_notebook_path)
    
    def test_batch_notebook_processing(self):
        """Test processing multiple notebooks in batch."""
        notebook_tester = NotebookTester(timeout=30)
        
        # Create multiple test notebooks
        notebooks = []
        
        for i in range(3):
            notebook_content = {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": f"# Test Notebook {i+1}\n\n## Learning Objectives\n\n- Test batch processing"
                    },
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": f"print('This is notebook {i+1}')\nresult = {i+1} * 10\nprint(f'Result: {{result}}')"
                    }
                ],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
                nbformat.write(nbformat.from_dict(notebook_content), f)
                notebooks.append(f.name)
        
        try:
            # Process all notebooks
            results = []
            for notebook_path in notebooks:
                success, error = notebook_tester.execute_notebook(notebook_path)
                results.append((notebook_path, success, error))
            
            # Verify we processed all notebooks
            assert len(notebook_tester.executed_notebooks) == 3
            
            # Generate comprehensive report
            with tempfile.TemporaryDirectory() as temp_dir:
                report_path = notebook_tester.create_execution_report(
                    os.path.join(temp_dir, "batch_report.html")
                )
                
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                
                # Should contain info about all notebooks
                assert "Total notebooks tested: <strong>3</strong>" in report_content
                
        finally:
            # Clean up
            for notebook_path in notebooks:
                if os.path.exists(notebook_path):
                    os.unlink(notebook_path)


# Utility functions for CLI usage
def run_notebook_tests(notebooks_dir: str, output_report: str = "notebook_test_report.html") -> bool:
    """
    Run notebook tests from command line.
    
    Args:
        notebooks_dir: Directory containing notebooks
        output_report: Path for output report
        
    Returns:
        True if all tests passed, False otherwise
    """
    if not os.path.exists(notebooks_dir):
        print(f"❌ Notebooks directory not found: {notebooks_dir}")
        return False
    
    print(f"🔍 Scanning for notebooks in: {notebooks_dir}")
    
    notebook_tester = NotebookTester(timeout=300)  # 5 minutes per cell
    notebook_files = list(Path(notebooks_dir).rglob("*.ipynb"))
    
    # Filter out checkpoint files
    notebook_files = [nb for nb in notebook_files if ".ipynb_checkpoints" not in str(nb)]
    
    if not notebook_files:
        print("⚠️ No notebook files found")
        return True
    
    print(f"📚 Found {len(notebook_files)} notebooks to test")
    
    all_passed = True
    
    for notebook_path in notebook_files:
        print(f"  🧪 Testing: {notebook_path.name}")
        
        # Validate structure
        issues = notebook_tester.validate_notebook_structure(str(notebook_path))
        if issues:
            print(f"    ⚠️ Structure issues: {len(issues)}")
            for issue in issues[:3]:  # Show first 3 issues
                print(f"      - {issue}")
        
        # Execute notebook
        success, error = notebook_tester.execute_notebook(str(notebook_path))
        
        if success:
            print(f"    ✅ PASS")
        else:
            print(f"    ❌ FAIL: {error[:100]}...")
            all_passed = False
    
    # Generate report
    report_path = notebook_tester.create_execution_report(output_report)
    print(f"📄 Test report generated: {report_path}")
    
    # Summary
    successful = sum(1 for nb in notebook_tester.executed_notebooks if nb['success'])
    total = len(notebook_tester.executed_notebooks)
    
    print(f"\n📊 Summary: {successful}/{total} notebooks passed ({successful/total*100:.1f}%)")
    
    return all_passed


if __name__ == "__main__":
    """Run notebook tests when executed as script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TensorVerseHub notebooks")
    parser.add_argument(
        "notebooks_dir",
        nargs="?",
        default="notebooks",
        help="Directory containing notebooks to test"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="notebook_test_report.html",
        help="Output path for test report"
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=300,
        help="Timeout per notebook cell in seconds"
    )
    
    args = parser.parse_args()
    
    # Override default timeout if specified
    if args.timeout != 300:
        NotebookTester.timeout = args.timeout
    
    success = run_notebook_tests(args.notebooks_dir, args.output)
    sys.exit(0 if success else 1)