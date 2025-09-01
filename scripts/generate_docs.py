# Location: /scripts/generate_docs.py

"""
Documentation generation script for TensorVerseHub.
Automatically generates comprehensive documentation from code, notebooks, and metadata.
"""

import os
import sys
import json
import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Core imports
import re
import ast
from datetime import datetime
import subprocess

# Optional imports
try:
    import nbformat
    from nbconvert import HTMLExporter, MarkdownExporter
    NBCONVERT_AVAILABLE = True
except ImportError:
    NBCONVERT_AVAILABLE = False
    print("Warning: nbformat/nbconvert not available. Notebook documentation will be limited.")

try:
    import markdown
    from markdown.extensions import codehilite, toc, tables
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    print("Warning: markdown not available. Markdown processing will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('documentation_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DocumentationGenerator:
    """Comprehensive documentation generator for TensorVerseHub."""
    
    def __init__(self, project_root: str, output_dir: str):
        """
        Initialize documentation generator.
        
        Args:
            project_root: Root directory of the project
            output_dir: Output directory for generated documentation
        """
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        
        # Create output directory structure
        self.docs_dir = self.output_dir / "docs"
        self.api_docs_dir = self.docs_dir / "api"
        self.notebook_docs_dir = self.docs_dir / "notebooks"
        self.tutorial_docs_dir = self.docs_dir / "tutorials"
        self.static_dir = self.docs_dir / "static"
        
        for dir_path in [self.docs_dir, self.api_docs_dir, self.notebook_docs_dir, 
                        self.tutorial_docs_dir, self.static_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Documentation metadata
        self.project_info = self._extract_project_info()
        self.module_docs = {}
        self.notebook_docs = {}
        self.tutorial_docs = {}
    
    def _extract_project_info(self) -> Dict[str, Any]:
        """Extract project information from various sources."""
        info = {
            'name': 'TensorVerseHub',
            'description': 'Comprehensive TensorFlow learning resource with practical examples and implementations',
            'version': '1.0.0',
            'author': 'TensorVerseHub Team',
            'generated_at': datetime.now().isoformat()
        }
        
        # Try to extract from setup.py or pyproject.toml
        setup_py = self.project_root / "setup.py"
        if setup_py.exists():
            try:
                with open(setup_py, 'r', encoding='utf-8') as f:
                    setup_content = f.read()
                
                # Extract version
                version_match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", setup_content)
                if version_match:
                    info['version'] = version_match.group(1)
                
                # Extract description
                desc_match = re.search(r"description\s*=\s*['\"]([^'\"]+)['\"]", setup_content)
                if desc_match:
                    info['description'] = desc_match.group(1)
                    
            except Exception as e:
                logger.warning(f"Could not parse setup.py: {e}")
        
        # Try to extract from README
        readme_files = ['README.md', 'README.rst', 'README.txt']
        for readme_file in readme_files:
            readme_path = self.project_root / readme_file
            if readme_path.exists():
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        readme_content = f.read()
                    
                    # Extract title and description from README
                    lines = readme_content.split('\n')
                    if lines:
                        # First non-empty line is often the title
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                continue
                            if line.startswith('# '):
                                info['name'] = line[2:].strip()
                                break
                    
                    info['readme_content'] = readme_content
                    break
                    
                except Exception as e:
                    logger.warning(f"Could not parse {readme_file}: {e}")
        
        return info
    
    def analyze_python_modules(self) -> Dict[str, Any]:
        """Analyze Python modules and extract documentation."""
        logger.info("Analyzing Python modules...")
        
        module_docs = {}
        
        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        python_files = [f for f in python_files if not any(part.startswith('.') for part in f.parts)]
        
        for py_file in python_files:
            try:
                module_info = self._analyze_python_file(py_file)
                if module_info:
                    relative_path = py_file.relative_to(self.project_root)
                    module_name = str(relative_path).replace('/', '.').replace('.py', '')
                    module_docs[module_name] = module_info
                    
            except Exception as e:
                logger.warning(f"Failed to analyze {py_file}: {e}")
        
        logger.info(f"Analyzed {len(module_docs)} Python modules")
        return module_docs
    
    def _analyze_python_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            module_info = {
                'path': str(file_path),
                'docstring': ast.get_docstring(tree) or '',
                'classes': [],
                'functions': [],
                'imports': [],
                'constants': []
            }
            
            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node) or '',
                        'methods': [],
                        'line_number': node.lineno
                    }
                    
                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_info = {
                                'name': item.name,
                                'docstring': ast.get_docstring(item) or '',
                                'args': [arg.arg for arg in item.args.args],
                                'line_number': item.lineno
                            }
                            class_info['methods'].append(method_info)
                    
                    module_info['classes'].append(class_info)
                
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    # Top-level functions only
                    func_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node) or '',
                        'args': [arg.arg for arg in node.args.args],
                        'line_number': node.lineno
                    }
                    module_info['functions'].append(func_info)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module_info['imports'].append(alias.name)
                    else:
                        module_name = node.module or ''
                        for alias in node.names:
                            module_info['imports'].append(f"{module_name}.{alias.name}")
                
                elif isinstance(node, ast.Assign):
                    # Extract constants (uppercase variables)
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            module_info['constants'].append(target.id)
            
            return module_info
            
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return None
    
    def analyze_notebooks(self) -> Dict[str, Any]:
        """Analyze Jupyter notebooks and extract documentation."""
        if not NBCONVERT_AVAILABLE:
            logger.warning("nbconvert not available, skipping notebook analysis")
            return {}
        
        logger.info("Analyzing Jupyter notebooks...")
        
        notebook_docs = {}
        
        # Find all notebook files
        notebook_files = list(self.project_root.rglob("*.ipynb"))
        notebook_files = [f for f in notebook_files if '.ipynb_checkpoints' not in str(f)]
        
        for nb_file in notebook_files:
            try:
                notebook_info = self._analyze_notebook_file(nb_file)
                if notebook_info:
                    relative_path = nb_file.relative_to(self.project_root)
                    notebook_name = str(relative_path)
                    notebook_docs[notebook_name] = notebook_info
                    
            except Exception as e:
                logger.warning(f"Failed to analyze notebook {nb_file}: {e}")
        
        logger.info(f"Analyzed {len(notebook_docs)} notebooks")
        return notebook_docs
    
    def _analyze_notebook_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a single Jupyter notebook."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            notebook_info = {
                'path': str(file_path),
                'title': '',
                'description': '',
                'learning_objectives': [],
                'code_cells': 0,
                'markdown_cells': 0,
                'total_cells': len(notebook.cells),
                'sections': [],
                'imports': set(),
                'outputs': []
            }
            
            current_section = None
            
            for cell in notebook.cells:
                if cell.cell_type == 'markdown':
                    notebook_info['markdown_cells'] += 1
                    source = cell.source
                    
                    # Extract title (first heading)
                    if not notebook_info['title']:
                        title_match = re.match(r'^#\s+(.+)', source)
                        if title_match:
                            notebook_info['title'] = title_match.group(1).strip()
                    
                    # Extract sections
                    section_match = re.match(r'^(#+)\s+(.+)', source)
                    if section_match:
                        level = len(section_match.group(1))
                        title = section_match.group(2).strip()
                        current_section = {
                            'level': level,
                            'title': title,
                            'content': source
                        }
                        notebook_info['sections'].append(current_section)
                    
                    # Look for learning objectives
                    if 'learning objective' in source.lower() or 'objectives' in source.lower():
                        objectives = re.findall(r'[-*]\s+(.+)', source)
                        notebook_info['learning_objectives'].extend(objectives)
                    
                    # Extract description from first few cells
                    if not notebook_info['description'] and len(source.strip()) > 50:
                        # Take first paragraph that's not a heading
                        paragraphs = [p.strip() for p in source.split('\n\n') if p.strip()]
                        for para in paragraphs:
                            if not para.startswith('#') and len(para) > 50:
                                notebook_info['description'] = para[:200] + '...' if len(para) > 200 else para
                                break
                
                elif cell.cell_type == 'code':
                    notebook_info['code_cells'] += 1
                    
                    # Extract imports
                    for line in cell.source.split('\n'):
                        line = line.strip()
                        if line.startswith('import ') or line.startswith('from '):
                            notebook_info['imports'].add(line)
                    
                    # Analyze outputs
                    if hasattr(cell, 'outputs') and cell.outputs:
                        for output in cell.outputs:
                            if output.output_type == 'display_data':
                                if 'image/png' in output.get('data', {}):
                                    notebook_info['outputs'].append('plot')
                            elif output.output_type == 'stream':
                                notebook_info['outputs'].append('text')
                            elif output.output_type == 'execute_result':
                                notebook_info['outputs'].append('result')
            
            # Convert imports set to list
            notebook_info['imports'] = list(notebook_info['imports'])
            
            return notebook_info
            
        except Exception as e:
            logger.warning(f"Failed to parse notebook {file_path}: {e}")
            return None
    
    def generate_api_documentation(self, module_docs: Dict[str, Any]) -> None:
        """Generate API documentation from module analysis."""
        logger.info("Generating API documentation...")
        
        # Generate index page
        index_content = self._generate_api_index(module_docs)
        with open(self.api_docs_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        # Generate individual module pages
        for module_name, module_info in module_docs.items():
            module_content = self._generate_module_page(module_name, module_info)
            
            # Create module file name
            module_file = module_name.replace('.', '_') + '.html'
            with open(self.api_docs_dir / module_file, 'w', encoding='utf-8') as f:
                f.write(module_content)
    
    def _generate_api_index(self, module_docs: Dict[str, Any]) -> str:
        """Generate API documentation index page."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.project_info['name']} - API Documentation</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .module-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .module-card {{ border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
                .module-card h3 {{ margin-top: 0; color: #007bff; }}
                .stats {{ background-color: #e9ecef; padding: 10px; border-radius: 3px; margin: 10px 0; }}
                a {{ color: #007bff; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📚 {self.project_info['name']} API Documentation</h1>
                <p>{self.project_info['description']}</p>
                <p><strong>Version:</strong> {self.project_info['version']} | 
                   <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>📦 Modules ({len(module_docs)})</h2>
            <div class="module-grid">
        """
        
        for module_name, module_info in sorted(module_docs.items()):
            module_file = module_name.replace('.', '_') + '.html'
            
            # Extract first line of docstring as summary
            summary = module_info.get('docstring', '').split('\n')[0] if module_info.get('docstring') else 'No description available'
            
            html_content += f"""
                <div class="module-card">
                    <h3><a href="{module_file}">{module_name}</a></h3>
                    <p>{summary[:150]}{'...' if len(summary) > 150 else ''}</p>
                    <div class="stats">
                        Classes: {len(module_info.get('classes', []))} | 
                        Functions: {len(module_info.get('functions', []))} | 
                        Imports: {len(module_info.get('imports', []))}
                    </div>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_module_page(self, module_name: str, module_info: Dict[str, Any]) -> str:
        """Generate documentation page for a specific module."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{module_name} - {self.project_info['name']} API</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin: 30px 0; }}
                .item {{ border-left: 3px solid #007bff; padding-left: 15px; margin: 20px 0; }}
                .docstring {{ background-color: #f8f9fa; padding: 15px; border-radius: 3px; white-space: pre-wrap; }}
                .args {{ background-color: #fff3cd; padding: 10px; border-radius: 3px; }}
                code {{ background-color: #f1f1f1; padding: 2px 4px; border-radius: 3px; }}
                .toc {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                a {{ color: #007bff; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📄 {module_name}</h1>
                <p><a href="index.html">← Back to API Index</a></p>
                <p><strong>Path:</strong> {module_info.get('path', 'Unknown')}</p>
            </div>
        """
        
        # Module docstring
        if module_info.get('docstring'):
            html_content += f"""
                <div class="section">
                    <h2>📖 Description</h2>
                    <div class="docstring">{module_info['docstring']}</div>
                </div>
            """
        
        # Table of contents
        toc_items = []
        if module_info.get('classes'):
            toc_items.append(f'<a href="#classes">Classes ({len(module_info["classes"])})</a>')
        if module_info.get('functions'):
            toc_items.append(f'<a href="#functions">Functions ({len(module_info["functions"])})</a>')
        if module_info.get('constants'):
            toc_items.append(f'<a href="#constants">Constants ({len(module_info["constants"])})</a>')
        if module_info.get('imports'):
            toc_items.append(f'<a href="#imports">Imports ({len(module_info["imports"])})</a>')
        
        if toc_items:
            html_content += f"""
                <div class="toc">
                    <h3>📋 Contents</h3>
                    {' | '.join(toc_items)}
                </div>
            """
        
        # Classes
        if module_info.get('classes'):
            html_content += '<div class="section" id="classes"><h2>🏗️ Classes</h2>'
            
            for class_info in module_info['classes']:
                html_content += f"""
                    <div class="item">
                        <h3>class <code>{class_info['name']}</code></h3>
                        <p><strong>Line:</strong> {class_info['line_number']}</p>
                """
                
                if class_info.get('docstring'):
                    html_content += f'<div class="docstring">{class_info["docstring"]}</div>'
                
                # Methods
                if class_info.get('methods'):
                    html_content += '<h4>Methods:</h4><ul>'
                    for method in class_info['methods']:
                        args_str = ', '.join(method['args']) if method['args'] else ''
                        html_content += f'<li><code>{method["name"]}({args_str})</code>'
                        if method.get('docstring'):
                            html_content += f'<br><small>{method["docstring"].split(".")[0]}.</small>'
                        html_content += '</li>'
                    html_content += '</ul>'
                
                html_content += '</div>'
            
            html_content += '</div>'
        
        # Functions
        if module_info.get('functions'):
            html_content += '<div class="section" id="functions"><h2>⚙️ Functions</h2>'
            
            for func_info in module_info['functions']:
                args_str = ', '.join(func_info['args']) if func_info['args'] else ''
                
                html_content += f"""
                    <div class="item">
                        <h3><code>{func_info['name']}({args_str})</code></h3>
                        <p><strong>Line:</strong> {func_info['line_number']}</p>
                """
                
                if func_info.get('docstring'):
                    html_content += f'<div class="docstring">{func_info["docstring"]}</div>'
                
                if func_info['args']:
                    html_content += f'<div class="args"><strong>Arguments:</strong> {", ".join(func_info["args"])}</div>'
                
                html_content += '</div>'
            
            html_content += '</div>'
        
        # Constants
        if module_info.get('constants'):
            html_content += f"""
                <div class="section" id="constants">
                    <h2>📋 Constants</h2>
                    <ul>
                        {''.join(f'<li><code>{const}</code></li>' for const in module_info['constants'])}
                    </ul>
                </div>
            """
        
        # Imports
        if module_info.get('imports'):
            html_content += f"""
                <div class="section" id="imports">
                    <h2>📦 Imports</h2>
                    <ul>
                        {''.join(f'<li><code>{imp}</code></li>' for imp in sorted(set(module_info['imports'])))}
                    </ul>
                </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content
    
    def generate_notebook_documentation(self, notebook_docs: Dict[str, Any]) -> None:
        """Generate documentation for notebooks."""
        logger.info("Generating notebook documentation...")
        
        # Generate notebook index
        index_content = self._generate_notebook_index(notebook_docs)
        with open(self.notebook_docs_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        # Generate individual notebook pages
        for notebook_path, notebook_info in notebook_docs.items():
            notebook_content = self._generate_notebook_page(notebook_path, notebook_info)
            
            # Create safe filename
            safe_name = notebook_path.replace('/', '_').replace('.ipynb', '.html')
            with open(self.notebook_docs_dir / safe_name, 'w', encoding='utf-8') as f:
                f.write(notebook_content)
    
    def _generate_notebook_index(self, notebook_docs: Dict[str, Any]) -> str:
        """Generate notebook documentation index."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.project_info['name']} - Notebook Documentation</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .notebook-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
                .notebook-card {{ border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
                .notebook-card h3 {{ margin-top: 0; color: #28a745; }}
                .stats {{ background-color: #e9ecef; padding: 10px; border-radius: 3px; margin: 10px 0; }}
                .objectives {{ background-color: #fff3cd; padding: 10px; border-radius: 3px; margin: 10px 0; }}
                a {{ color: #007bff; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📚 {self.project_info['name']} Notebook Documentation</h1>
                <p>Interactive learning materials and tutorials</p>
                <p><strong>Total Notebooks:</strong> {len(notebook_docs)}</p>
            </div>
            
            <div class="notebook-grid">
        """
        
        for notebook_path, notebook_info in sorted(notebook_docs.items()):
            safe_name = notebook_path.replace('/', '_').replace('.ipynb', '.html')
            title = notebook_info.get('title') or Path(notebook_path).stem.replace('_', ' ').title()
            
            html_content += f"""
                <div class="notebook-card">
                    <h3><a href="{safe_name}">📓 {title}</a></h3>
                    <p><strong>Path:</strong> {notebook_path}</p>
            """
            
            if notebook_info.get('description'):
                html_content += f'<p>{notebook_info["description"]}</p>'
            
            html_content += f"""
                <div class="stats">
                    Code Cells: {notebook_info.get('code_cells', 0)} | 
                    Markdown Cells: {notebook_info.get('markdown_cells', 0)} | 
                    Total: {notebook_info.get('total_cells', 0)}
                </div>
            """
            
            if notebook_info.get('learning_objectives'):
                html_content += f"""
                    <div class="objectives">
                        <strong>Learning Objectives:</strong>
                        <ul>
                            {''.join(f'<li>{obj}</li>' for obj in notebook_info['learning_objectives'][:3])}
                        </ul>
                    </div>
                """
            
            html_content += '</div>'
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_notebook_page(self, notebook_path: str, notebook_info: Dict[str, Any]) -> str:
        """Generate documentation page for a specific notebook."""
        title = notebook_info.get('title') or Path(notebook_path).stem.replace('_', ' ').title()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title} - {self.project_info['name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin: 30px 0; }}
                .objectives {{ background-color: #d4edda; padding: 15px; border-radius: 5px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .stat-card {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; text-align: center; }}
                .imports {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                code {{ background-color: #f1f1f1; padding: 2px 4px; border-radius: 3px; }}
                a {{ color: #007bff; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📓 {title}</h1>
                <p><a href="index.html">← Back to Notebook Index</a></p>
                <p><strong>Path:</strong> {notebook_path}</p>
            </div>
        """
        
        # Description
        if notebook_info.get('description'):
            html_content += f"""
                <div class="section">
                    <h2>📖 Description</h2>
                    <p>{notebook_info['description']}</p>
                </div>
            """
        
        # Learning objectives
        if notebook_info.get('learning_objectives'):
            html_content += f"""
                <div class="section">
                    <div class="objectives">
                        <h2>🎯 Learning Objectives</h2>
                        <ul>
                            {''.join(f'<li>{obj}</li>' for obj in notebook_info['learning_objectives'])}
                        </ul>
                    </div>
                </div>
            """
        
        # Statistics
        html_content += f"""
            <div class="section">
                <h2>📊 Notebook Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>{notebook_info.get('total_cells', 0)}</h3>
                        <p>Total Cells</p>
                    </div>
                    <div class="stat-card">
                        <h3>{notebook_info.get('code_cells', 0)}</h3>
                        <p>Code Cells</p>
                    </div>
                    <div class="stat-card">
                        <h3>{notebook_info.get('markdown_cells', 0)}</h3>
                        <p>Markdown Cells</p>
                    </div>
                    <div class="stat-card">
                        <h3>{len(notebook_info.get('sections', []))}</h3>
                        <p>Sections</p>
                    </div>
                </div>
            </div>
        """
        
        # Sections
        if notebook_info.get('sections'):
            html_content += """
                <div class="section">
                    <h2>📋 Sections</h2>
                    <ul>
            """
            
            for section in notebook_info['sections']:
                indent = "  " * (section['level'] - 1)
                html_content += f'<li>{indent}{section["title"]}</li>'
            
            html_content += """
                    </ul>
                </div>
            """
        
        # Imports
        if notebook_info.get('imports'):
            html_content += f"""
                <div class="section">
                    <h2>📦 Key Imports</h2>
                    <div class="imports">
                        {''.join(f'<code>{imp}</code><br>' for imp in sorted(notebook_info['imports'])[:10])}
                        {f'<p><em>... and {len(notebook_info["imports"]) - 10} more</em></p>' if len(notebook_info['imports']) > 10 else ''}
                    </div>
                </div>
            """
        
        # Outputs summary
        if notebook_info.get('outputs'):
            output_counts = {}
            for output in notebook_info['outputs']:
                output_counts[output] = output_counts.get(output, 0) + 1
            
            html_content += f"""
                <div class="section">
                    <h2>🖼️ Outputs</h2>
                    <ul>
                        {''.join(f'<li>{output_type.title()}: {count}</li>' for output_type, count in output_counts.items())}
                    </ul>
                </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content
    
    def generate_main_index(self) -> None:
        """Generate main documentation index page."""
        logger.info("Generating main documentation index...")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.project_info['name']} - Documentation</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0; padding: 0; line-height: 1.6; color: #333;
                }}
                .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 60px 20px; text-align: center; margin-bottom: 40px;
                }}
                .header h1 {{ margin: 0; font-size: 3em; font-weight: 300; }}
                .header p {{ font-size: 1.2em; margin: 20px 0; }}
                .nav-grid {{ 
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 30px; margin: 40px 0;
                }}
                .nav-card {{
                    background: white; border: 1px solid #e1e8ed; border-radius: 10px;
                    padding: 30px; text-align: center; transition: transform 0.2s, box-shadow 0.2s;
                }}
                .nav-card:hover {{ 
                    transform: translateY(-5px); box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                }}
                .nav-card h3 {{ color: #1da1f2; margin: 0 0 15px 0; font-size: 1.5em; }}
                .nav-card p {{ color: #657786; margin: 15px 0; }}
                .nav-card a {{ 
                    color: #1da1f2; text-decoration: none; font-weight: 500;
                    display: inline-block; padding: 12px 24px; border: 2px solid #1da1f2;
                    border-radius: 25px; transition: all 0.2s;
                }}
                .nav-card a:hover {{ 
                    background-color: #1da1f2; color: white;
                }}
                .stats {{ 
                    background-color: #f7f9fa; padding: 30px; border-radius: 10px; margin: 40px 0;
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px;
                }}
                .stat {{ text-align: center; }}
                .stat h3 {{ font-size: 2.5em; color: #1da1f2; margin: 0; }}
                .stat p {{ margin: 5px 0; color: #657786; }}
                .footer {{ 
                    background-color: #15202b; color: #8899a6; text-align: center; 
                    padding: 40px 20px; margin-top: 60px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 {self.project_info['name']}</h1>
                <p>{self.project_info['description']}</p>
                <p><strong>Version {self.project_info['version']}</strong> • Generated {datetime.now().strftime('%B %d, %Y')}</p>
            </div>
            
            <div class="container">
        """
        
        # Navigation cards
        html_content += """
            <div class="nav-grid">
                <div class="nav-card">
                    <h3>📚 API Documentation</h3>
                    <p>Complete reference for all Python modules, classes, and functions</p>
                    <a href="api/index.html">Browse API Docs</a>
                </div>
                
                <div class="nav-card">
                    <h3>📓 Notebook Gallery</h3>
                    <p>Interactive Jupyter notebooks with tutorials and examples</p>
                    <a href="notebooks/index.html">View Notebooks</a>
                </div>
                
                <div class="nav-card">
                    <h3>🎓 Tutorials</h3>
                    <p>Step-by-step guides and learning materials</p>
                    <a href="tutorials/index.html">Start Learning</a>
                </div>
                
                <div class="nav-card">
                    <h3>🔧 Getting Started</h3>
                    <p>Installation, setup, and quick start guide</p>
                    <a href="#getting-started">Get Started</a>
                </div>
            </div>
        """
        
        # Statistics
        module_count = len(self.module_docs)
        notebook_count = len(self.notebook_docs)
        
        html_content += f"""
            <div class="stats">
                <div class="stat">
                    <h3>{module_count}</h3>
                    <p>Python Modules</p>
                </div>
                <div class="stat">
                    <h3>{notebook_count}</h3>
                    <p>Jupyter Notebooks</p>
                </div>
                <div class="stat">
                    <h3>∞</h3>
                    <p>Learning Opportunities</p>
                </div>
            </div>
        """
        
        # Getting started section
        if self.project_info.get('readme_content'):
            # Extract installation section from README
            readme_content = self.project_info['readme_content']
            
            html_content += f"""
                <div id="getting-started" style="margin: 40px 0;">
                    <h2>🚀 Getting Started</h2>
                    <div style="background-color: #f7f9fa; padding: 30px; border-radius: 10px;">
                        <h3>Quick Installation</h3>
                        <pre style="background-color: #1d1f21; color: #c5c8c6; padding: 20px; border-radius: 5px; overflow-x: auto;">
<code>git clone https://github.com/tensorversehub/tensorversehub.git
cd tensorversehub
pip install -r requirements.txt</code>
                        </pre>
                        
                        <h3>First Steps</h3>
                        <ol>
                            <li>Explore the <a href="notebooks/index.html">notebook collection</a></li>
                            <li>Check out the <a href="api/index.html">API documentation</a></li>
                            <li>Run your first example: <code>python examples/basic_tensorflow_example.py</code></li>
                        </ol>
                    </div>
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="footer">
                <p>📖 Documentation generated automatically from source code and notebooks</p>
                <p>Built with ❤️ for the TensorFlow community</p>
            </div>
        </body>
        </html>
        """
        
        with open(self.docs_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def copy_static_assets(self) -> None:
        """Copy static assets like CSS and images."""
        logger.info("Copying static assets...")
        
        # Create a simple CSS file for consistency
        css_content = """
        /* TensorVerseHub Documentation Styles */
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --accent-color: #1da1f2;
            --text-color: #333;
            --light-bg: #f7f9fa;
            --border-color: #e1e8ed;
        }
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            margin: 0;
            padding: 0;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Common utility classes */
        .text-center { text-align: center; }
        .mt-2 { margin-top: 2rem; }
        .mb-2 { margin-bottom: 2rem; }
        .p-2 { padding: 2rem; }
        
        /* Code blocks */
        pre, code {
            font-family: 'Monaco', 'Consolas', monospace;
        }
        
        pre {
            background-color: #1d1f21;
            color: #c5c8c6;
            padding: 1rem;
            border-radius: 5px;
            overflow-x: auto;
        }
        
        code {
            background-color: #f1f1f1;
            padding: 2px 4px;
            border-radius: 3px;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
        }
        """
        
        with open(self.static_dir / "styles.css", 'w', encoding='utf-8') as f:
            f.write(css_content)
    
    def generate_full_documentation(self) -> str:
        """Generate complete documentation suite."""
        logger.info("🚀 Starting comprehensive documentation generation...")
        
        # Analyze project structure
        logger.info("📊 Analyzing project structure...")
        self.module_docs = self.analyze_python_modules()
        self.notebook_docs = self.analyze_notebooks()
        
        # Generate documentation
        self.generate_api_documentation(self.module_docs)
        self.generate_notebook_documentation(self.notebook_docs)
        self.generate_main_index()
        self.copy_static_assets()
        
        # Generate simple tutorials index (placeholder)
        tutorials_index = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tutorials - {self.project_info['name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎓 {self.project_info['name']} Tutorials</h1>
                <p><a href="../index.html">← Back to Documentation Home</a></p>
            </div>
            
            <h2>Coming Soon</h2>
            <p>Detailed tutorials are being prepared. For now, check out our 
               <a href="../notebooks/index.html">notebook collection</a> for hands-on learning.</p>
        </body>
        </html>
        """
        
        with open(self.tutorial_docs_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(tutorials_index)
        
        main_index_path = self.docs_dir / "index.html"
        
        logger.info("🎉 Documentation generation completed successfully!")
        logger.info(f"📍 Main documentation index: {main_index_path}")
        logger.info(f"📚 Generated docs for {len(self.module_docs)} modules and {len(self.notebook_docs)} notebooks")
        
        return str(main_index_path)


def main():
    """Main documentation generation script."""
    parser = argparse.ArgumentParser(description="Generate TensorVerseHub documentation")
    
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Root directory of the project"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_docs",
        help="Output directory for documentation"
    )
    
    parser.add_argument(
        "--modules-only",
        action="store_true",
        help="Generate only API documentation"
    )
    
    parser.add_argument(
        "--notebooks-only",
        action="store_true",
        help="Generate only notebook documentation"
    )
    
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open documentation in web browser after generation"
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting TensorVerseHub Documentation Generation")
    logger.info(f"Project root: {args.project_root}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize generator
    doc_generator = DocumentationGenerator(args.project_root, args.output_dir)
    
    if args.modules_only:
        # Generate only API docs
        module_docs = doc_generator.analyze_python_modules()
        doc_generator.generate_api_documentation(module_docs)
        logger.info("✅ API documentation generated")
        
    elif args.notebooks_only:
        # Generate only notebook docs
        notebook_docs = doc_generator.analyze_notebooks()
        doc_generator.generate_notebook_documentation(notebook_docs)
        logger.info("✅ Notebook documentation generated")
        
    else:
        # Generate full documentation suite
        index_path = doc_generator.generate_full_documentation()
        
        # Open in browser if requested
        if args.open_browser:
            try:
                import webbrowser
                webbrowser.open(f"file://{Path(index_path).absolute()}")
                logger.info("🌐 Documentation opened in web browser")
            except Exception as e:
                logger.warning(f"Could not open browser: {e}")
        
        logger.info(f"📖 Documentation available at: file://{Path(index_path).absolute()}")


if __name__ == "__main__":
    main()