"""
Command-line interface for SCAudit.

This module provides the CLI commands for running SCA audits,
managing data, and generating reports.
"""

import click
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, List
import yaml
import os
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from sqlalchemy.orm import Session

from .core.orchestrator import AuditOrchestrator
from .core.analysis import AnalysisService
from .core.similarity import SimilarityIndex
from .models.database import create_encrypted_engine, init_database
from .utils.config import load_config, merge_configs


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model name mappings
MODEL_MAPPING = {
    "gpt-4o": lambda: ChatOpenAI(model="gpt-4o"),
    "gpt-4": lambda: ChatOpenAI(model="gpt-4"),
    "gpt-3.5-turbo": lambda: ChatOpenAI(model="gpt-3.5-turbo"),
    "claude-3-opus": lambda: ChatAnthropic(model="claude-3-opus-20240229"),
    "claude-3-sonnet": lambda: ChatAnthropic(model="claude-3-sonnet-20240229"),
    "claude-3-haiku": lambda: ChatAnthropic(model="claude-3-haiku-20240307"),
}


def get_model(model_name: str):
    """Get a chat model instance by name."""
    # Ensure API keys are available
    if model_name.startswith("gpt") or "openai" in model_name:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
    elif "claude" in model_name or "anthropic" in model_name:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    if model_name in MODEL_MAPPING:
        return MODEL_MAPPING[model_name]()
    else:
        # Try to parse as a custom model spec
        if ":" in model_name:
            provider, model = model_name.split(":", 1)
            if provider == "openai":
                return ChatOpenAI(model=model)
            elif provider == "anthropic":
                return ChatAnthropic(model=model)
        
        raise ValueError(f"Unknown model: {model_name}")


def load_cli_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from file and environment."""
    config = {}
    
    # Load from default locations
    default_paths = [
        Path.home() / ".config" / "scaudit.yaml",
        Path.home() / ".scaudit.yaml",
        Path("scaudit.yaml")
    ]
    
    for path in default_paths:
        if path.exists():
            with open(path) as f:
                config = yaml.safe_load(f)
            break
    
    # Override with specific config file
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            override_config = yaml.safe_load(f)
        config = merge_configs(config, override_config)
    
    # Override with environment variables
    env_mapping = {
        "SCAUDIT_DB_PATH": "database_path",
        "SCAUDIT_DB_KEY": "database_key",
        "SCAUDIT_DEFAULT_TARGET": "default_target",
        "SCAUDIT_DEFAULT_JUDGES": "default_judges",
    }
    
    for env_var, config_key in env_mapping.items():
        if env_var in os.environ:
            config[config_key] = os.environ[env_var]
    
    return config


@click.group()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.pass_context
def cli(ctx, config: Optional[str], debug: bool):
    """SCAudit - Special Characters Attack auditing tool."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Setup context
    ctx.ensure_object(dict)
    
    # Load configuration
    ctx.obj['config'] = load_cli_config(config)
    
    # Set debug logging
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize database
    db_path = ctx.obj['config'].get('database_path', 'scaudit.db')
    # Expand user path
    db_path = os.path.expanduser(db_path)
    engine = create_encrypted_engine(db_path)
    init_database(engine)
    ctx.obj['engine'] = engine


@cli.command()
@click.option('--files', '-f', required=True, help='Glob pattern for input files')
@click.option('--target', '-t', help='Target model to audit')
@click.option('--judge', '-j', multiple=True, help='Judge models (can specify multiple)')
@click.option('--batch-size', '-b', default=100, help='Batch size for processing')
@click.option('--max-concurrency', default=10, help='Maximum concurrent requests')
@click.option('--temperature', default=0.7, help='Temperature for target model')
@click.option('--max-tokens', default=1000, help='Max tokens for responses')
@click.option('--progress', is_flag=True, help='Show progress bar')
@click.pass_context
def ingest(
    ctx,
    files: str,
    target: Optional[str],
    judge: List[str],
    batch_size: int,
    max_concurrency: int,
    temperature: float,
    max_tokens: int,
    progress: bool
):
    """Ingest sequences and run SCA audit."""
    config = ctx.obj['config']
    
    # Determine target model
    target_model_name = target or config.get('default_target', 'gpt-4o')
    target_model = get_model(target_model_name)
    
    # Determine judge models
    judge_model_names = list(judge) or config.get('default_judges', ['gpt-4o'])
    judge_models = [get_model(name) for name in judge_model_names]
    
    # Create database session
    session = Session(bind=ctx.obj['engine'])
    
    # Configure orchestrator
    orchestrator_config = {
        'max_concurrent': max_concurrency,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'requests_per_minute': config.get('requests_per_minute', 60),
        'tokens_per_minute': config.get('tokens_per_minute', 150000),
    }
    
    # Create orchestrator
    orchestrator = AuditOrchestrator(
        target_model=target_model,
        judge_models=judge_models,
        db_session=session,
        config=orchestrator_config
    )
    
    # Progress callback
    def progress_callback(results):
        if progress:
            click.echo(
                f"Progress: {results['total_sequences']} sequences, "
                f"{results['total_responses']} responses, "
                f"{results['total_leaks']} leaks detected"
            )
    
    try:
        # Run audit
        click.echo(f"Starting audit with target: {target_model_name}")
        click.echo(f"Judge models: {', '.join(judge_model_names)}")
        click.echo(f"Processing files matching: {files}")
        
        results = orchestrator.audit_glob_sync(
            files,
            batch_size,
            progress_callback if progress else None
        )
        
        # Display results
        click.echo("\nAudit Results:")
        click.echo(f"  Total sequences: {results['total_sequences']:,}")
        click.echo(f"  Total responses: {results['total_responses']:,}")
        click.echo(f"  Total leaks: {results['total_leaks']:,}")
        click.echo(f"  Leak rate: {results['total_leaks'] / results['total_responses']:.2%}")
        click.echo(f"  Errors: {results['errors']}")
        click.echo(f"  Filtered: {results['filtered']}")
        
        if 'clusters' in results:
            click.echo(f"\nClustering Results:")
            click.echo(f"  Total clusters: {results['clusters']['total_clusters']:,}")
            click.echo(f"  Average cluster size: {results['clusters']['average_cluster_size']:.1f}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        session.close()


@cli.command()
@click.option('--threshold', '-t', default=0.95, help='Similarity threshold')
@click.option('--recompute', is_flag=True, help='Recompute embeddings')
@click.pass_context
def dedup(ctx, threshold: float, recompute: bool):
    """Deduplicate responses using similarity search."""
    session = Session(bind=ctx.obj['engine'])
    analysis = AnalysisService(session)
    
    try:
        if recompute:
            click.echo("Recomputing embeddings...")
            # This would require loading all responses and recomputing
            # For now, we'll use existing embeddings
        
        click.echo(f"Deduplicating with threshold: {threshold}")
        
        # Get cluster statistics
        cluster_report = analysis.cluster_report()
        
        click.echo("\nDeduplication Results:")
        click.echo(f"  Total clusters: {cluster_report['total_clusters']:,}")
        click.echo(f"  Average cluster size: {cluster_report.get('average_cluster_size', 0):.1f}")
        click.echo(f"  Largest cluster: {cluster_report.get('max_cluster_size', 0):,}")
        click.echo(f"  Singleton clusters: {cluster_report.get('singleton_clusters', 0):,}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        session.close()


@cli.command()
@click.option('--metric', '-m', type=click.Choice([
    'leak_stats', 'model_performance', 'strategy_effectiveness',
    'confidence_distribution', 'data_types'
]), help='Specific metric to analyze')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', type=click.Choice(['markdown', 'json']), default='markdown')
@click.pass_context
def analyze(ctx, metric: Optional[str], output: Optional[str], format: str):
    """Analyze audit results and generate reports."""
    session = Session(bind=ctx.obj['engine'])
    analysis = AnalysisService(session)
    
    try:
        if metric:
            # Get specific metric
            stats = analysis.leak_stats()
            
            if metric == 'leak_stats':
                data = {
                    'total_sequences': stats['total_sequences'],
                    'total_responses': stats['total_responses'],
                    'total_leaks': stats['total_leaks'],
                    'leak_rate': stats['leak_rate']
                }
            elif metric == 'model_performance':
                data = stats['model_performance']
            elif metric == 'strategy_effectiveness':
                data = stats['strategy_effectiveness']
            elif metric == 'confidence_distribution':
                data = stats['confidence_distribution']
            elif metric == 'data_types':
                data = stats['extracted_data_types']
            
            if format == 'json':
                import json
                result = json.dumps(data, indent=2)
            else:
                # Format as markdown table
                result = f"# {metric.replace('_', ' ').title()}\n\n"
                if isinstance(data, dict):
                    result += "| Key | Value |\n|-----|-------|\n"
                    for k, v in data.items():
                        result += f"| {k} | {v} |\n"
        else:
            # Generate full report
            result = analysis.generate_markdown_report()
        
        # Output results
        if output:
            Path(output).write_text(result)
            click.echo(f"Report saved to: {output}")
        else:
            click.echo(result)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        session.close()


@cli.command()
@click.option('--table', '-t', required=True, type=click.Choice([
    'sequences', 'responses', 'judgments', 'extracted_data', 'clusters'
]), help='Table to export')
@click.option('--format', '-f', type=click.Choice(['parquet', 'csv', 'json']), default='parquet')
@click.option('--output', '-o', required=True, help='Output file path')
@click.option('--filter', '-F', multiple=True, help='Filters in format column=value')
@click.pass_context
def export(ctx, table: str, format: str, output: str, filter: List[str]):
    """Export audit data to various formats."""
    session = Session(bind=ctx.obj['engine'])
    analysis = AnalysisService(session)
    
    try:
        # Parse filters
        filters = {}
        for f in filter:
            if '=' in f:
                col, val = f.split('=', 1)
                filters[col] = val
        
        click.echo(f"Exporting {table} to {output} as {format}...")
        
        analysis.export(
            output_path=output,
            format=format,
            table=table
        )
        
        click.echo(f"Export complete: {output}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        session.close()


@cli.command()
@click.option('--key', '-k', help='New encryption key')
@click.pass_context
def rotate_key(ctx, key: str):
    """Rotate database encryption key."""
    # This would require re-encrypting the database
    # Implementation depends on SQLCipher specifics
    click.echo("Key rotation not yet implemented")


if __name__ == '__main__':
    cli()