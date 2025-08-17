"""
Command-line interface for the RAG engine
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax

from ..core.engine import RAGEngine
from ..core.config import ConfigurationManager, PipelineConfig
from ..core.models import Document, EvaluationTestCase
from ..api.main import run_server


console = Console()


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: Optional[str] = None, environment: Optional[str] = None) -> PipelineConfig:
    """Load configuration from file or environment"""
    try:
        config_manager = ConfigurationManager(config_path=config_path, environment=environment)
        return config_manager.load_config()
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)


def create_engine(config_path: Optional[str] = None, environment: Optional[str] = None) -> RAGEngine:
    """Create and initialize RAG engine"""
    try:
        config = load_config(config_path, environment)
        return RAGEngine(config)
    except Exception as e:
        console.print(f"[red]Error creating RAG engine: {e}[/red]")
        sys.exit(1)


@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--environment', '-e', help='Environment (development, testing, production)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, environment, verbose):
    """RAG Engine CLI - Production-ready RAG system command-line interface"""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['environment'] = environment
    ctx.obj['verbose'] = verbose
    
    setup_logging(verbose)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind the server to')
@click.option('--port', default=8000, help='Port to bind the server to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.pass_context
def serve(ctx, host, port, reload):
    """Start the RAG API server"""
    console.print("[blue]Starting RAG API server...[/blue]")
    
    try:
        run_server(
            host=host,
            port=port,
            config_path=ctx.obj['config_path'],
            environment=ctx.obj['environment'],
            reload=reload,
            log_level='debug' if ctx.obj['verbose'] else 'info'
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('question')
@click.option('--k', default=5, help='Number of documents to retrieve')
@click.option('--include-sources/--no-include-sources', default=True, help='Include source documents in output')
@click.option('--output-format', type=click.Choice(['text', 'json']), default='text', help='Output format')
@click.pass_context
def query(ctx, question, k, include_sources, output_format):
    """Query the RAG system"""
    if output_format != 'json':
        console.print(f"[blue]Processing query: {question}[/blue]")
    
    try:
        engine = create_engine(ctx.obj['config_path'], ctx.obj['environment'])
        
        if not engine.is_ready():
            if output_format == 'json':
                error_output = {"error": "No documents indexed. Please add documents first."}
                print(json.dumps(error_output, indent=2))
            else:
                console.print("[red]No documents indexed. Please add documents first using 'rag index' command.[/red]")
            sys.exit(1)
        
        if output_format != 'json':
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Querying...", total=None)
                response = engine.query(question, k=k)
                progress.remove_task(task)
        else:
            response = engine.query(question, k=k)
        
        if output_format == 'json':
            output = {
                'question': question,
                'answer': response.answer,
                'confidence_score': response.confidence_score,
                'processing_time': response.processing_time,
                'source_documents': [
                    {
                        'content': doc.content[:200] + '...' if len(doc.content) > 200 else doc.content,
                        'metadata': doc.metadata,
                        'doc_id': doc.doc_id
                    }
                    for doc in response.source_documents
                ] if include_sources else [],
                'metadata': response.metadata
            }
            print(json.dumps(output, indent=2))
        else:
            # Text format
            console.print(Panel(response.answer, title="Answer", border_style="green"))
            
            console.print(f"\n[dim]Confidence: {response.confidence_score:.2f}[/dim]")
            console.print(f"[dim]Processing time: {response.processing_time:.2f}s[/dim]")
            
            if include_sources and response.source_documents:
                console.print("\n[bold]Source Documents:[/bold]")
                for i, doc in enumerate(response.source_documents, 1):
                    content_preview = doc.content[:200] + '...' if len(doc.content) > 200 else doc.content
                    console.print(f"\n[dim]{i}. {doc.metadata.get('source', 'Unknown source')}[/dim]")
                    console.print(f"[dim]{content_preview}[/dim]")
    
    except Exception as e:
        if output_format == 'json':
            error_output = {"error": str(e)}
            print(json.dumps(error_output, indent=2))
        else:
            console.print(f"[red]Error processing query: {e}[/red]")
        sys.exit(1)


@cli.group()
def index():
    """Document indexing commands"""
    pass


@index.command('files')
@click.argument('paths', nargs=-1, required=True)
@click.option('--clear', is_flag=True, help='Clear existing documents before indexing')
@click.option('--recursive', '-r', is_flag=True, help='Recursively index directories')
@click.option('--pattern', help='File pattern to match (e.g., "*.txt", "*.md")')
@click.pass_context
def index_files(ctx, paths, clear, recursive, pattern):
    """Index documents from files or directories"""
    console.print(f"[blue]Indexing documents from {len(paths)} path(s)...[/blue]")
    
    try:
        engine = create_engine(ctx.obj['config_path'], ctx.obj['environment'])
        
        if clear:
            console.print("[yellow]Clearing existing documents...[/yellow]")
            engine.clear_documents()
        
        documents = []
        
        for path_str in paths:
            path = Path(path_str)
            
            if path.is_file():
                if pattern and not path.match(pattern):
                    continue
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.append(Document(
                        content=content,
                        metadata={'source': str(path), 'filename': path.name},
                        doc_id=str(path)
                    ))
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not read {path}: {e}[/yellow]")
            
            elif path.is_dir() and recursive:
                for file_path in path.rglob(pattern or '*'):
                    if file_path.is_file() and file_path.suffix in ['.txt', '.md', '.py', '.json', '.yaml', '.yml']:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            documents.append(Document(
                                content=content,
                                metadata={'source': str(file_path), 'filename': file_path.name},
                                doc_id=str(file_path)
                            ))
                        except Exception as e:
                            console.print(f"[yellow]Warning: Could not read {file_path}: {e}[/yellow]")
        
        if not documents:
            console.print("[yellow]No documents found to index[/yellow]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Indexing {len(documents)} documents...", total=None)
            success = engine.add_documents(documents)
            progress.remove_task(task)
        
        if success:
            console.print(f"[green]Successfully indexed {len(documents)} documents[/green]")
            console.print(f"[dim]Total documents: {engine.get_document_count()}[/dim]")
            console.print(f"[dim]Total chunks: {engine.get_chunk_count()}[/dim]")
        else:
            console.print("[red]Failed to index documents[/red]")
            sys.exit(1)
    
    except Exception as e:
        console.print(f"[red]Error indexing files: {e}[/red]")
        sys.exit(1)


@index.command('web')
@click.argument('urls', nargs=-1, required=True)
@click.option('--clear', is_flag=True, help='Clear existing documents before indexing')
@click.option('--max-depth', default=1, help='Maximum crawling depth')
@click.option('--include-links', is_flag=True, help='Include links in the content')
@click.pass_context
def index_web(ctx, urls, clear, max_depth, include_links):
    """Index documents from web URLs"""
    console.print(f"[blue]Loading documents from {len(urls)} URL(s)...[/blue]")
    
    try:
        engine = create_engine(ctx.obj['config_path'], ctx.obj['environment'])
        
        if clear:
            console.print("[yellow]Clearing existing documents...[/yellow]")
            engine.clear_documents()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading web documents...", total=None)
            success = engine.load_web_documents(
                list(urls),
                max_depth=max_depth,
                include_links=include_links
            )
            progress.remove_task(task)
        
        if success:
            console.print(f"[green]Successfully loaded documents from {len(urls)} URL(s)[/green]")
            console.print(f"[dim]Total documents: {engine.get_document_count()}[/dim]")
            console.print(f"[dim]Total chunks: {engine.get_chunk_count()}[/dim]")
        else:
            console.print("[red]Failed to load web documents[/red]")
            sys.exit(1)
    
    except Exception as e:
        console.print(f"[red]Error loading web documents: {e}[/red]")
        sys.exit(1)


@index.command('clear')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def clear_index(ctx, confirm):
    """Clear all indexed documents"""
    if not confirm:
        if not click.confirm("Are you sure you want to clear all indexed documents?"):
            console.print("[yellow]Operation cancelled[/yellow]")
            return
    
    try:
        engine = create_engine(ctx.obj['config_path'], ctx.obj['environment'])
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Clearing documents...", total=None)
            success = engine.clear_documents()
            progress.remove_task(task)
        
        if success:
            console.print("[green]Successfully cleared all documents[/green]")
        else:
            console.print("[red]Failed to clear documents[/red]")
            sys.exit(1)
    
    except Exception as e:
        console.print(f"[red]Error clearing documents: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('test_file')
@click.option('--frameworks', multiple=True, default=['custom'], help='Evaluation frameworks to use')
@click.option('--output', help='Output file for results')
@click.option('--output-format', type=click.Choice(['text', 'json']), default='text', help='Output format')
@click.pass_context
def evaluate(ctx, test_file, frameworks, output, output_format):
    """Evaluate the RAG system using test cases from a JSON file"""
    if output_format != 'json':
        console.print(f"[blue]Evaluating RAG system using {test_file}...[/blue]")
    
    try:
        # Load test cases from file
        test_path = Path(test_file)
        if not test_path.exists():
            console.print(f"[red]Test file not found: {test_file}[/red]")
            sys.exit(1)
        
        with open(test_path, 'r') as f:
            test_data = json.load(f)
        
        test_cases = []
        for case in test_data.get('test_cases', []):
            test_cases.append(EvaluationTestCase(
                question=case['question'],
                expected_answer=case['expected_answer'],
                context=[Document(**doc) for doc in case.get('context', [])],
                metadata=case.get('metadata', {})
            ))
        
        if not test_cases:
            console.print("[yellow]No test cases found in file[/yellow]")
            return
        
        engine = create_engine(ctx.obj['config_path'], ctx.obj['environment'])
        
        if output_format != 'json':
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Evaluating {len(test_cases)} test cases...", total=None)
                result = engine.evaluate(test_cases)
                progress.remove_task(task)
        else:
            result = engine.evaluate(test_cases)
        
        # Format output
        if output_format == 'json':
            output_data = {
                'overall_score': result.overall_score,
                'metric_scores': result.metric_scores,
                'test_case_results': result.test_case_results,
                'recommendations': result.recommendations,
                'frameworks_used': list(frameworks),
                'total_test_cases': len(test_cases)
            }
            
            if output:
                with open(output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                console.print(f"[green]Results saved to {output}[/green]")
            else:
                print(json.dumps(output_data, indent=2))
        else:
            # Text format
            console.print(Panel(f"Overall Score: {result.overall_score:.2f}", title="Evaluation Results", border_style="green"))
            
            if result.metric_scores:
                table = Table(title="Metric Scores")
                table.add_column("Metric", style="cyan")
                table.add_column("Score", style="green")
                
                for metric, score in result.metric_scores.items():
                    table.add_row(metric, f"{score:.2f}")
                
                console.print(table)
            
            if result.recommendations:
                console.print("\n[bold]Recommendations:[/bold]")
                for i, rec in enumerate(result.recommendations, 1):
                    console.print(f"{i}. {rec}")
            
            if output:
                output_data = {
                    'overall_score': result.overall_score,
                    'metric_scores': result.metric_scores,
                    'test_case_results': result.test_case_results,
                    'recommendations': result.recommendations,
                    'frameworks_used': list(frameworks),
                    'total_test_cases': len(test_cases)
                }
                with open(output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                console.print(f"\n[green]Detailed results saved to {output}[/green]")
    
    except Exception as e:
        console.print(f"[red]Error during evaluation: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Show RAG system status and information"""
    try:
        engine = create_engine(ctx.obj['config_path'], ctx.obj['environment'])
        info = engine.get_system_info()
        
        # System information table
        table = Table(title="RAG System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")
        
        table.add_row("Version", info['version'], "")
        table.add_row("LLM Provider", info['config']['llm_provider'], info['config']['llm_model'])
        table.add_row("Embedding Provider", info['config']['embedding_provider'], info['config']['embedding_model'])
        table.add_row("Vector Store", info['config']['vector_store'], "")
        table.add_row("Indexing Strategy", info['config']['indexing_strategy'], f"Chunk size: {info['config']['chunk_size']}")
        
        console.print(table)
        
        # Statistics
        stats_table = Table(title="Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Indexed Documents", str(info['stats']['indexed_documents']))
        stats_table.add_row("Indexed Chunks", str(info['stats']['indexed_chunks']))
        stats_table.add_row("Retriever Ready", "Yes" if info['stats']['retriever_ready'] else "No")
        
        console.print(stats_table)
        
        # Component status
        components_table = Table(title="Components")
        components_table.add_column("Component", style="cyan")
        components_table.add_column("Status", style="green")
        
        for component, status in info['components'].items():
            status_text = "✓ Active" if status else "✗ Inactive"
            components_table.add_row(component.title(), status_text)
        
        console.print(components_table)
    
    except Exception as e:
        console.print(f"[red]Error getting system status: {e}[/red]")
        sys.exit(1)


@cli.group()
def config():
    """Configuration management commands"""
    pass


@config.command('show')
@click.option('--format', 'output_format', type=click.Choice(['yaml', 'json']), default='yaml', help='Output format')
@click.pass_context
def show_config(ctx, output_format):
    """Show current configuration"""
    try:
        config = load_config(ctx.obj['config_path'], ctx.obj['environment'])
        
        # Convert dataclass to dict
        config_dict = {
            field.name: getattr(config, field.name)
            for field in config.__dataclass_fields__.values()
        }
        
        if output_format == 'json':
            console.print(json.dumps(config_dict, indent=2, default=str))
        else:
            import yaml
            yaml_output = yaml.dump(config_dict, default_flow_style=False, sort_keys=True)
            syntax = Syntax(yaml_output, "yaml", theme="monokai", line_numbers=True)
            console.print(syntax)
    
    except Exception as e:
        console.print(f"[red]Error showing configuration: {e}[/red]")
        sys.exit(1)


@config.command('validate')
@click.argument('config_file')
@click.pass_context
def validate_config(ctx, config_file):
    """Validate a configuration file"""
    try:
        from ..core.config import ConfigurationManager
        
        config_manager = ConfigurationManager()
        is_valid = config_manager.validate_config_file(config_file)
        
        if is_valid:
            console.print(f"[green]Configuration file {config_file} is valid[/green]")
        else:
            console.print(f"[red]Configuration file {config_file} is invalid[/red]")
            sys.exit(1)
    
    except Exception as e:
        console.print(f"[red]Error validating configuration: {e}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    cli()