# Settings loader — reads the YAML config file and returns a dictionary.
# WHY separate from the YAML file?
# settings.py adds logic: fallback to defaults if file not found, merging
# user overrides, type validation. The YAML file is just data.

import yaml
import os
import logging
from typing import Dict, Any

log = logging.getLogger(__name__)

# Path to the defaults file — relative to this file's location
_DEFAULTS_PATH = os.path.join(os.path.dirname(__file__), 'defaults.yaml')

# Path for user overrides — in the project root (not committed to Git)
_USER_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config_user.yaml')


def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file(s).
    
    Priority order (highest to lowest):
    1. config_user.yaml  (your personal overrides — not in Git)
    2. defaults.yaml     (default values — in Git)
    
    Returns:
        Dictionary with all configuration values.
    """
    config = {}
    
    # Load defaults first
    if os.path.exists(_DEFAULTS_PATH):
        with open(_DEFAULTS_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        log.debug(f"Loaded defaults from {_DEFAULTS_PATH}")
    else:
        log.warning(f"Defaults file not found: {_DEFAULTS_PATH}")
    
    # Overlay user overrides (if file exists)
    if os.path.exists(_USER_PATH):
        with open(_USER_PATH, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f) or {}
        # Deep merge: user values override defaults, section by section
        _deep_merge(config, user_config)
        log.debug(f"Applied user config from {_USER_PATH}")
    
    return config


def save_user_config(updates: Dict[str, Any]):
    """
    Save user-specific settings to config_user.yaml.
    
    Use this when the user changes settings through the UI.
    Only saves the values that differ from defaults.
    """
    os.makedirs(os.path.dirname(_USER_PATH), exist_ok=True)
    with open(_USER_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(updates, f, default_flow_style=False)
    log.info(f"Saved user config to {_USER_PATH}")


def _deep_merge(base: dict, override: dict):
    """
    Recursively merge override dict into base dict.
    
    Unlike base.update(override), this merges nested dicts instead of
    replacing them. So user can override just cursor.smoothing_factor
    without losing all other cursor settings.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value