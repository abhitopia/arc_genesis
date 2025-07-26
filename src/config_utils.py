"""
config_utils.py
===============
Configuration utilities for YAML serialization/deserialization.

Author 2025 – MIT licence
"""

import yaml
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Dict, Type, TypeVar
from pathlib import Path

T = TypeVar('T')


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert a dataclass (potentially nested) to a dictionary suitable for YAML.
    
    Args:
        obj: The dataclass instance to convert
        
    Returns:
        Dictionary representation of the dataclass
    """
    if is_dataclass(obj):
        result = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = dataclass_to_dict(value)
        return result
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    else:
        return obj


def dict_to_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    """
    Convert a dictionary to a dataclass instance with support for polymorphic model configs.
    
    Args:
        cls: The dataclass class to instantiate
        data: Dictionary with the data
        
    Returns:
        Instance of the dataclass
    """
    if not is_dataclass(cls):
        return data
    
    # Special handling for ExperimentConfig with polymorphic model field
    if cls.__name__ == 'ExperimentConfig' and 'model' in data:
        model_data = data['model']
        if isinstance(model_data, dict):
            # Import here to avoid circular imports
            from .train import create_model_config_from_dict
            data = data.copy()  # Don't modify original
            data['model'] = create_model_config_from_dict(model_data)
    
    field_types = {f.name: f.type for f in fields(cls)}
    kwargs = {}
    
    for field_name, field_type in field_types.items():
        if field_name in data:
            value = data[field_name]
            
            # Skip model field for ExperimentConfig as it's already handled above
            if cls.__name__ == 'ExperimentConfig' and field_name == 'model':
                kwargs[field_name] = value
            # Handle nested dataclasses
            elif is_dataclass(field_type) and isinstance(value, dict):
                kwargs[field_name] = dict_to_dataclass(field_type, value)
            else:
                kwargs[field_name] = value
    
    return cls(**kwargs)


def save_config_to_yaml(config: Any, filepath: Path) -> None:
    """
    Save a configuration dataclass to a YAML file.
    
    Args:
        config: The configuration dataclass instance
        filepath: Path where to save the YAML file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = dataclass_to_dict(config)
    
    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)


def load_config_from_yaml(cls: Type[T], filepath: Path) -> T:
    """
    Load a configuration dataclass from a YAML file with support for polymorphic models.
    
    Args:
        cls: The dataclass class to instantiate
        filepath: Path to the YAML file
        
    Returns:
        Instance of the dataclass loaded from YAML
    """
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    return dict_to_dataclass(cls, data)


def merge_configs(base_config: T, override_dict: Dict[str, Any]) -> T:
    """
    Merge a base configuration with override values from a dictionary.
    
    Args:
        base_config: Base configuration dataclass
        override_dict: Dictionary with override values (can be nested)
        
    Returns:
        New configuration instance with merged values
    """
    # Convert base config to dict
    base_dict = dataclass_to_dict(base_config)
    
    # Deep merge the override values
    merged_dict = _deep_merge_dicts(base_dict, override_dict)
    
    # Convert back to dataclass
    return dict_to_dataclass(type(base_config), merged_dict)


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary with override values
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result 