"""
Configuration management utilities for SCAudit.

This module provides functions for loading, merging, and validating
configuration from various sources.
"""

from typing import Dict, Any, Optional
import yaml
import json
from pathlib import Path
import os
from dotenv import load_dotenv


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Supports YAML and JSON formats.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path) as f:
        if path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif path.suffix == '.json':
            return json.load(f)
        else:
            # Try to detect format
            content = f.read()
            f.seek(0)
            
            try:
                return json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                return yaml.safe_load(f)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Later configurations override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration
    """
    result = {}
    
    for config in configs:
        if config:
            result = deep_merge(result, config)
    
    return result


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        update: Dictionary with updates
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration and apply defaults.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Validated configuration with defaults
    """
    defaults = {
        'database_path': 'scaudit.db',
        'default_target': 'gpt-4o',
        'default_judges': ['gpt-4o'],
        'requests_per_minute': 60,
        'tokens_per_minute': 150000,
        'max_concurrent': 10,
        'temperature': 0.7,
        'max_tokens': 1000,
        'similarity_threshold': 0.95,
        'judge_ensemble_size': 1,
        'use_llm_extraction': True,
        'vector_backend': 'faiss',
        'filter_min_length': 20,
        'filter_min_special_ratio': 0.15,
        'filter_min_entropy': 2.0,
    }
    
    # Apply defaults
    result = defaults.copy()
    result.update(config)
    
    # Validate specific fields
    if not isinstance(result['default_judges'], list):
        result['default_judges'] = [result['default_judges']]
    
    # Ensure numeric fields are correct type
    numeric_fields = [
        'requests_per_minute', 'tokens_per_minute', 'max_concurrent',
        'temperature', 'max_tokens', 'similarity_threshold',
        'judge_ensemble_size', 'filter_min_length', 'filter_min_special_ratio',
        'filter_min_entropy'
    ]
    
    for field in numeric_fields:
        if field in result:
            result[field] = float(result[field]) if '.' in str(result[field]) else int(result[field])
    
    return result


def load_env_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Returns:
        Configuration dictionary from environment
    """
    # Load .env file if it exists
    load_dotenv()
    
    config = {}
    
    # Mapping of environment variables to config keys
    env_mapping = {
        'SCAUDIT_DB_PATH': 'database_path',
        'SCAUDIT_DB_KEY': 'database_key',
        'SCAUDIT_DEFAULT_TARGET': 'default_target',
        'SCAUDIT_DEFAULT_JUDGES': 'default_judges',
        'SCAUDIT_REQUESTS_PER_MINUTE': 'requests_per_minute',
        'SCAUDIT_TOKENS_PER_MINUTE': 'tokens_per_minute',
        'SCAUDIT_MAX_CONCURRENT': 'max_concurrent',
        'SCAUDIT_TEMPERATURE': 'temperature',
        'SCAUDIT_MAX_TOKENS': 'max_tokens',
        'SCAUDIT_SIMILARITY_THRESHOLD': 'similarity_threshold',
        'SCAUDIT_VECTOR_BACKEND': 'vector_backend',
    }
    
    for env_var, config_key in env_mapping.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            
            # Parse lists
            if config_key == 'default_judges' and ',' in value:
                value = [v.strip() for v in value.split(',')]
            
            # Parse numbers
            elif config_key in ['requests_per_minute', 'tokens_per_minute', 
                              'max_concurrent', 'max_tokens']:
                value = int(value)
            elif config_key in ['temperature', 'similarity_threshold']:
                value = float(value)
            
            config[config_key] = value
    
    return config


def save_config(config: Dict[str, Any], path: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        path: Path to save to
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path_obj, 'w') as f:
        if path_obj.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        else:
            json.dump(config, f, indent=2)


def get_default_config_path() -> Optional[Path]:
    """
    Get the default configuration file path.
    
    Returns:
        Path to default config file, or None if not found
    """
    search_paths = [
        Path.home() / '.config' / 'scaudit.yaml',
        Path.home() / '.scaudit.yaml',
        Path('scaudit.yaml'),
        Path('.scaudit.yaml')
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None