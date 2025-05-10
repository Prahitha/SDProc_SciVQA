import os
from pathlib import Path

# Default Wandb Configuration
DEFAULT_WANDB_CONFIG = {
    "project": "sci-vqa-phi4-prompt",
    "entity": None,  # Your wandb username/team name if needed
    "api_key": os.getenv("WANDB_API_KEY", 'd7daa36e132a83d1ef62f1a3c08e0c27f3f6a666'),
    "mode": "online"  # Set to "disabled" to disable wandb
}

# Wandb Run Names
RUN_NAMES = {
    "inference": "phi4-inference",
    "evaluation": "phi4-evaluation",
    "analysis": "phi4-analysis"
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


def init_wandb(run_type: str, config: dict = None):
    """Initialize wandb with the appropriate configuration.

    Args:
        run_type: Type of run ('inference', 'evaluation', 'analysis')
        config: Additional configuration to log
    """
    import wandb

    # Get wandb config from the provided config
    wandb_config = DEFAULT_WANDB_CONFIG.copy()
    if config and 'wandb' in config:
        wandb_config.update(config['wandb'])

    # Set environment variables for wandb
    os.environ["WANDB_API_KEY"] = wandb_config['api_key'] if wandb_config['api_key'] else ""
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
        # Initialize wandb
        wandb.init(
            project=wandb_config['project'],
            entity=wandb_config['entity'],
            name=RUN_NAMES[run_type],
            config=run_config,
            mode=wandb_config['mode']
        )
        return wandb
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        print("Continuing without wandb logging...")
        return None
