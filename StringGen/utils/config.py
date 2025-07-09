#!/usr/bin/env python3
"""
Configuration management for SCA generator.

Handles loading and saving configuration files for reproducible
generation runs and parameter presets.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
import argparse


@dataclass
class Config:
    """
    Configuration container for SCA generator parameters.
    
    All parameters match command-line arguments for consistency.
    """
    # Generation mode
    mode: str = 'sample'
    
    # Generation parameters
    strategy: Optional[str] = None
    length: Union[int, str] = '10'
    count: int = 1
    min_length: int = 1
    max_length: int = 100
    seed: Optional[int] = None
    
    # Output parameters
    output: str = 'sca_output.txt'
    format: str = 'pipe'
    overwrite: bool = False
    append: bool = False
    
    # Performance parameters
    workers: Optional[int] = None
    use_threads: bool = False
    buffer_size: int = 10000
    chunk_size: int = 1000
    
    # Progress and logging
    progress: bool = False
    metrics: bool = False
    log_level: str = 'INFO'
    quiet: bool = False
    
    # Advanced options
    validate: bool = False
    dry_run: bool = False
    resume: Optional[str] = None
    
    # Metadata
    version: str = field(default='2.0.0', init=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = asdict(self)
        # Convert Path objects to strings
        if isinstance(data.get('output'), Path):
            data['output'] = str(data['output'])
        if isinstance(data.get('resume'), Path):
            data['resume'] = str(data['resume'])
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        # Filter out unknown keys
        valid_keys = {f.name for f in cls.__dataclass_fields__.values() if f.init}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """Create configuration from command-line arguments."""
        # Convert argparse namespace to dict
        args_dict = vars(args)
        
        # Remove non-config arguments
        exclude_keys = {'config', 'save_config', 'benchmark'}
        config_dict = {
            k: v for k, v in args_dict.items() 
            if k not in exclude_keys and v is not None
        }
        
        # Convert Path objects to strings
        if 'output' in config_dict and isinstance(config_dict['output'], Path):
            config_dict['output'] = str(config_dict['output'])
        if 'resume' in config_dict and isinstance(config_dict['resume'], Path):
            config_dict['resume'] = str(config_dict['resume'])
        
        return cls.from_dict(config_dict)
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate mode
        if self.mode not in ['sample', 'exhaustive']:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # Validate strategy
        if self.strategy and self.strategy not in ['INSET1', 'INSET2', 'CROSS1', 'CROSS2', 'CROSS3']:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        
        # Validate format
        if self.format not in ['pipe', 'json', 'binary']:
            raise ValueError(f"Invalid format: {self.format}")
        
        # Validate length ranges
        if self.min_length < 1:
            raise ValueError("min_length must be >= 1")
        if self.max_length < self.min_length:
            raise ValueError("max_length must be >= min_length")
        
        # Validate counts
        if self.count < 1:
            raise ValueError("count must be >= 1")
        if self.buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        
        # Validate file conflicts
        if self.overwrite and self.append:
            raise ValueError("Cannot use both --overwrite and --append")


def load_config(path: Union[str, Path]) -> Config:
    """
    Load configuration from file.
    
    Supports JSON and YAML formats based on file extension.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Loaded configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is invalid
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, 'r') as f:
        if path.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                data = yaml.safe_load(f)
            except ImportError:
                raise ImportError(
                    "PyYAML is required for YAML config files. "
                    "Install with: pip install pyyaml"
                )
        elif path.suffix.lower() == '.json':
            data = json.load(f)
        else:
            # Default to JSON
            data = json.load(f)
    
    config = Config.from_dict(data)
    config.validate()
    return config


def save_config(config: Config, path: Union[str, Path]) -> None:
    """
    Save configuration to file.
    
    Format is determined by file extension (.json or .yaml/.yml).
    
    Args:
        config: Configuration object to save
        path: Output file path
    """
    path = Path(path)
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict
    data = config.to_dict()
    
    with open(path, 'w') as f:
        if path.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                raise ImportError(
                    "PyYAML is required for YAML config files. "
                    "Install with: pip install pyyaml"
                )
        else:
            # Default to JSON
            json.dump(data, f, indent=2)


def create_default_configs():
    """Create example configuration files for common use cases."""
    configs_dir = Path('configs')
    configs_dir.mkdir(exist_ok=True)
    
    # Fast sampling config
    fast_sample = Config(
        mode='sample',
        count=10000,
        length='100-1000',
        workers=None,  # Use all CPUs
        buffer_size=50000,
        chunk_size=5000,
        progress=True
    )
    save_config(fast_sample, configs_dir / 'fast_sample.json')
    
    # Exhaustive INSET1 config
    exhaustive_inset1 = Config(
        mode='exhaustive',
        strategy='INSET1',
        min_length=1,
        max_length=100,
        format='pipe',
        progress=True,
        metrics=True
    )
    save_config(exhaustive_inset1, configs_dir / 'exhaustive_inset1.json')
    
    # Benchmark config
    benchmark = Config(
        mode='sample',
        count=100000,
        length='1000',
        workers=None,
        use_threads=False,
        buffer_size=100000,
        metrics=True,
        log_level='WARNING'
    )
    save_config(benchmark, configs_dir / 'benchmark.json')
    
    # Large dataset config
    large_dataset = Config(
        mode='sample',
        count=10000000,  # 10 million
        length='10-10000',
        workers=None,
        buffer_size=100000,
        chunk_size=10000,
        progress=True,
        metrics=True,
        format='binary'  # More compact
    )
    save_config(large_dataset, configs_dir / 'large_dataset.json')
    
    print(f"Created example configurations in {configs_dir}/")


if __name__ == '__main__':
    # Create example configs when run directly
    create_default_configs()