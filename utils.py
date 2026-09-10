from pathlib import Path
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parent

def load_config(filepath="config.toml"):
    """Loads the main project configuration from a TOML file."""
    config_path = PROJECT_ROOT / filepath if not Path(filepath).is_absolute() else Path(filepath)
    with open(config_path, "rb") as f:
        return tomllib.load(f)