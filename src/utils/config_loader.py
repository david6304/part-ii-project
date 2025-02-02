import yaml
from pathlib import Path
import os

def find_project_root() -> Path:
    """Find the project root by searching for config/ directory"""
    current = Path(os.getcwd())
    while current != current.parent:
        if (current / "config").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root containing config/")

def convert_paths(config_dict: dict) -> dict:
    """Recursively convert string paths to Path objects"""
    for key, value in config_dict.items():
        if isinstance(value, dict):
            config_dict[key] = convert_paths(value)
        elif isinstance(value, str):
            config_dict[key] = Path(value)
    return config_dict

def load_config(config_file: str = "config/paths.yaml") -> dict:
    """Load YAML configuration file"""
    project_root = find_project_root()
    config_path = project_root / config_file
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert strings to Path objects
    for category in ['data', 'output']:
        config[category] = convert_paths(config[category])
    
    return config

# Test immediately
if __name__ == "__main__":
    config = load_config()
    print("Loaded config:", config)