import os
from pathlib import Path
from typing import Optional, Dict, Any

# Default Wandb Configuration
DEFAULT_WANDB_CONFIG = {
    "project": None,  # Will be set from config
    "entity": None,  # Your wandb username/team name if needed
    "api_key": os.getenv("WANDB_API_KEY", None),
    "mode": "online",  # Set to "disabled" to disable wandb
    "run_names": {}  # Will be set from config
}

# Wandb Logging Configurations
LOG_CONFIG = {
    "inference": {
        "log_interval": 10,  # Log every N examples
        "log_examples": True,  # Log individual examples
        "log_metrics": True,  # Log aggregated metrics
    },
    "evaluation": {
        "log_interval": 1,  # Log after each evaluation
        "log_examples": True,
        "log_metrics": True,
    },
    "analysis": {
        "log_interval": 1,  # Log after each analysis
        "log_examples": True,
        "log_metrics": True,
    }
}


def validate_wandb_config(config: Dict[str, Any]) -> bool:
    """Validate wandb configuration.

    Args:
        config: Wandb configuration dictionary

    Returns:
        bool: True if config is valid, False otherwise
    """
    if not config.get('project'):
        print("Warning: No project name specified in wandb config")
        return False

    if not config.get('api_key'):
        print("Warning: No API key specified in wandb config")
        return False

    if not config.get('run_names'):
        print("Warning: No run names specified in wandb config")
        return False

    return True


def init_wandb(run_type: str, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """Initialize wandb with the appropriate configuration.

    Args:
        run_type: Type of run ('inference', 'evaluation', 'analysis')
        config: Additional configuration to log

    Returns:
        Optional[Any]: Wandb run instance if successful, None otherwise
    """
    import wandb

    if not config or 'wandb' not in config:
        print("Warning: No wandb configuration found in config file")
        return None

    # Get wandb config from the provided config
    wandb_config = DEFAULT_WANDB_CONFIG.copy()
    wandb_config.update(config['wandb'])

    # Validate configuration
    if not validate_wandb_config(wandb_config):
        return None

    # Set environment variables for wandb
    os.environ["WANDB_API_KEY"] = wandb_config['api_key']
    os.environ["WANDB_MODE"] = wandb_config['mode']
    os.environ["WANDB_SILENT"] = "true"  # Disable wandb prompts

    # Base configuration
    run_config = {
        "run_type": run_type,
        **LOG_CONFIG[run_type]
    }

    # Add additional config if provided
    if config:
        run_config.update(config)

    try:
        # Get run name from config
        run_names = wandb_config.get('run_names', {})
        if run_type not in run_names:
            print(
                f"Warning: No run name specified for {run_type} in wandb config")
            return None

        run_name = run_names[run_type]

        # Initialize wandb
        wandb.init(
            project=wandb_config['project'],
            entity=wandb_config['entity'],
            name=run_name,
            config=run_config,
            mode=wandb_config['mode']
        )
        return wandb
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        print("Continuing without wandb logging...")
        return None
