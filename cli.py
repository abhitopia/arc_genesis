"""
cli.py
======
Command-line interface for multi-model training (GenesisV2, SlotAttention) using typer.

Author 2025 – MIT licence
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import typer
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# Import our modules
from src.train import (
    ExperimentConfig, DataModule, TrainingModule, 
    ModelConfig, TrainingConfig, DataConfig,
    create_trainer, create_default_experiment_for_model,
    create_model_config_from_dict
)
from src.config_utils import save_config_to_yaml, load_config_from_yaml


def load_config_from_checkpoint(checkpoint_path: str) -> ExperimentConfig:
    """
    Load configuration from a checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        ExperimentConfig loaded from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'hyper_parameters' not in checkpoint:
        raise ValueError(f"Checkpoint does not contain 'hyper_parameters' key. Cannot load config from checkpoint.")
    
    hyper_params = checkpoint['hyper_parameters']
    
    # Convert hyperparameters back to ExperimentConfig
    # PyTorch Lightning saves the vars(config) as hyper_parameters
    config = ExperimentConfig()
    
    # Handle nested dataclass objects - PyTorch Lightning may save these as actual objects or dicts
    if 'model' in hyper_params:
        model_params = hyper_params['model']
        if hasattr(model_params, '__dict__'):  # It's an object
            config.model = model_params
        elif isinstance(model_params, dict):  # It's a dictionary
            config.model = create_model_config_from_dict(model_params)
        
    if 'training' in hyper_params:
        training_params = hyper_params['training']
        if hasattr(training_params, '__dict__'):  # It's an object
            config.training = training_params
        elif isinstance(training_params, dict):  # It's a dictionary
            config.training = TrainingConfig(**training_params)
            
    if 'data' in hyper_params:
        data_params = hyper_params['data']
        if hasattr(data_params, '__dict__'):  # It's an object
            config.data = data_params
        elif isinstance(data_params, dict):  # It's a dictionary
            config.data = DataConfig(**data_params)
    
    # Update top-level config attributes
    for key, value in hyper_params.items():
        if hasattr(config, key) and key not in ['model', 'training', 'data']:
            setattr(config, key, value)
    
    return config


def load_weights_from_checkpoint(model: TrainingModule, checkpoint_path: str):
    """
    Load only model weights and GECO state from checkpoint, ignoring optimizer/scheduler state.
    
    This handles torch.compile compatibility by automatically detecting and handling
    state dict key differences between compiled and non-compiled models.
    
    Args:
        model: Fresh TrainingModule instance
        checkpoint_path: Path to checkpoint file
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' not in checkpoint:
        raise ValueError(f"Checkpoint does not contain 'state_dict' key. Invalid checkpoint format.")
    
    full_state_dict = checkpoint['state_dict']
    
    # Filter state dict to include only:
    # 1. Model weights (model.*)
    # 2. GECO state (geco.*)
    # Exclude: optimizer states, lr scheduler states, etc.
    
    model_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith('model.') or key.startswith('geco.'):
            model_state_dict[key] = value
    
    # Handle torch.compile compatibility
    model_state_dict = _handle_compile_compatibility(model, model_state_dict)
    
    # Load the filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    
    # Report what was loaded/ignored
    typer.echo(f"   Loaded {len(model_state_dict)} parameters")
    if missing_keys and len(missing_keys) > 0:
        typer.echo(f"   Missing keys (will use random init): {len(missing_keys)} parameters")
        if len(missing_keys) <= 5:
            typer.echo(f"     Examples: {missing_keys[:5]}")
    if unexpected_keys and len(unexpected_keys) > 0:
        typer.echo(f"   Unexpected keys (ignored): {len(unexpected_keys)} parameters")
        if len(unexpected_keys) <= 5:
            typer.echo(f"     Examples: {unexpected_keys[:5]}")


def _handle_compile_compatibility(model: TrainingModule, checkpoint_state_dict: dict) -> dict:
    """
    Handle state dict compatibility between compiled and non-compiled models.
    
    torch.compile can add prefixes like '_orig_mod.' to parameter names.
    This function attempts to match parameters correctly regardless of compilation state.
    
    Args:
        model: Target model to load weights into
        checkpoint_state_dict: State dict from checkpoint
        
    Returns:
        Compatible state dict with corrected keys
    """
    # Get current model state dict to see what keys we expect
    current_state_dict = model.state_dict()
    current_keys = set(current_state_dict.keys())
    checkpoint_keys = set(checkpoint_state_dict.keys())
    
    # Check if we have exact matches (no compilation mismatch)
    exact_matches = current_keys.intersection(checkpoint_keys)
    if len(exact_matches) > len(current_keys) * 0.8:  # Most keys match
        return checkpoint_state_dict
    
    # Attempt to handle compilation prefix differences
    corrected_state_dict = {}
    
    for checkpoint_key, value in checkpoint_state_dict.items():
        # Try various key transformations
        possible_keys = [
            checkpoint_key,  # Original key
            checkpoint_key.replace('._orig_mod.', '.'),  # Remove _orig_mod prefix
            checkpoint_key.replace('.model._orig_mod.', '.model.'),  # Remove nested _orig_mod
        ]
        
        # Also try adding _orig_mod if checkpoint doesn't have it but current model does
        if '._orig_mod.' not in checkpoint_key:
            base_parts = checkpoint_key.split('.')
            if len(base_parts) >= 2:
                # Insert _orig_mod after 'model'
                if base_parts[0] == 'model':
                    new_key = 'model._orig_mod.' + '.'.join(base_parts[1:])
                    possible_keys.append(new_key)
        
        # Find the best matching key
        best_key = None
        for candidate_key in possible_keys:
            if candidate_key in current_keys:
                best_key = candidate_key
                break
        
        if best_key:
            corrected_state_dict[best_key] = value
        else:
            # Keep original key and let PyTorch Lightning handle it
            corrected_state_dict[checkpoint_key] = value
    
    return corrected_state_dict


app = typer.Typer(
    name="genesis-cli",
    help="Multi-model training CLI (GenesisV2, SlotAttention)",
    no_args_is_help=True
)





@app.command()
def generate_config(
    output_path: str = typer.Option(
        "config.yaml",
        "--output", "-o",
        help="Output path for the generated config file"
    ),
    model_type: str = typer.Option(
        "genesis_v2",
        "--model-type", "-MT",
        help="Type of model to generate config for ('genesis_v2' or 'slot_attention')"
    )
):
    """
    Generate a default configuration file.
    
    Creates a YAML configuration file with default parameters that can be
    modified and used for training.
    """
    output_file = Path(output_path)
    
    # Create default config
    config = create_default_experiment_for_model(model_type)
    typer.echo(f"Generating default experiment config for model type: {model_type}...")
    
    # Save config to YAML
    try:
        save_config_to_yaml(config, output_file)
        typer.echo(f"✅ Config saved to: {output_file}")
        typer.echo(f"   You can now edit this file and use it with: genesis-cli train --config {output_file}")
    except Exception as e:
        typer.echo(f"❌ Error saving config: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def train(
    config_path: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to the YAML configuration file (required unless using --resume)"
    ),
    checkpoint_path: Optional[str] = typer.Option(
        None,
        "--checkpoint", "-k",
        help="Path to checkpoint file to load from"
    ),
    resume: bool = typer.Option(
        False,
        "--resume", "-r",
        help="Resume full training state from checkpoint (loads config from checkpoint)"
    ),
    cpu: bool = typer.Option(
        False,
        "--cpu", "-C",
        help="Force CPU usage (disable GPU)"
    ),
    compile_model: bool = typer.Option(
        False,
        "--compile",
        help="Compile model with torch.compile for maximum performance (auto-handled in checkpoint loading)"
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-D",
        help="Debug mode (adds 'debug_' prefix to project name)"
    ),
    disable_wandb: bool = typer.Option(
        False,
        "--disable-wandb", "-DW",
        help="Disable Weights & Biases logging"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run", "-DR",
        help="Print configuration and exit without training"
    ),
    max_steps: Optional[int] = typer.Option(
        None,
        "--max-steps",
        help="Override max_steps for quick debugging"
    ),
    num_workers: int = typer.Option(
        8,
        "--num-workers", "-NW",
        help="Override number of data loading workers"
    )
):
    """
    Train model (GenesisV2 or SlotAttention) with the specified configuration.
    
    Supports three training modes:
    
    1. From scratch: --config my_config.yaml
    2. Transfer learning: --config my_config.yaml --checkpoint path/to/weights.ckpt
    3. Resume training: --checkpoint path/to/checkpoint.ckpt --resume
    
    When using --resume, the configuration is loaded from the checkpoint file.
    Command-line options can override specific config values.
    """
    
    # Check if config file exists (when required)
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            typer.echo(f"❌ Config file not found: {config_file}", err=True)
            typer.echo(f"   Generate one with: genesis-cli generate-config --output {config_file}")
            raise typer.Exit(1)
    
    # Validate input parameters
    if resume and checkpoint_path is None:
        typer.echo(f"❌ --resume requires --checkpoint to be specified", err=True)
        raise typer.Exit(1)
    
    if not resume and config_path is None:
        typer.echo(f"❌ --config is required when not using --resume", err=True)
        typer.echo(f"   Use --resume to load config from checkpoint, or provide --config for fresh/transfer training")
        raise typer.Exit(1)
    
    # Check if checkpoint file exists (if provided)
    if checkpoint_path is not None:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            typer.echo(f"❌ Checkpoint file not found: {checkpoint_file}", err=True)
            raise typer.Exit(1)
    
    # Load configuration
    try:
        if resume:
            # Load config from checkpoint
            config = load_config_from_checkpoint(checkpoint_path)
            typer.echo(f"✅ Loaded config from checkpoint: {checkpoint_path}")
        else:
            # Load config from YAML file
            config = load_config_from_yaml(ExperimentConfig, config_file)
            typer.echo(f"✅ Loaded config from: {config_file}")
    except Exception as e:
        typer.echo(f"❌ Error loading config: {e}", err=True)
        raise typer.Exit(1)
    
    # Determine hardware settings (separate from config)
    if cpu:
        accelerator, devices, precision = "cpu", "auto", "32-true"
        typer.echo(f"   Override: CPU mode enabled (accelerator=cpu, precision=32-true)")
    else:
        # Auto-detect based on CUDA availability
        if torch.cuda.is_available():
            accelerator, devices, precision = "cuda", "auto", "bf16-mixed"
        else:
            accelerator, devices, precision = "cpu", "auto", "32-true"
    
    # Set high precision matrix multiplication
    torch.set_float32_matmul_precision("high")
    typer.echo(f"   Override: High precision matrix multiplication enabled")
    
    if debug:
        if not config.project_name.startswith("debug_"):
            config.project_name = f"debug_{config.project_name}"
        # Enable anomaly detection for debugging gradient issues
        typer.echo(f"   Override: Debug mode enabled (project_name = {config.project_name})")
        # torch.autograd.set_detect_anomaly(True)
        # typer.echo(f"   Override: Gradient anomaly detection enabled")
    
    if compile_model:
        typer.echo(f"   Override: Model compilation enabled for maximum performance")
    
    # Apply max_steps override if provided (for debugging)
    if max_steps is not None:
        config.training.max_steps = max_steps
        # For very short debug runs, adjust logging frequency to ensure we see metrics
        if max_steps <= 10:
            config.log_every_n_steps = 1
            typer.echo(f"   Override: log_every_n_steps = 1 (adjusted for short debug run)")
        typer.echo(f"   Override: max_steps = {max_steps}")
    
    # Print configuration summary
    typer.echo("\n📋 Training Configuration:")
    typer.echo(f"   Project: {config.project_name}")
    typer.echo(f"   Run name: {config.run_name}")
    typer.echo(f"   Dataset: {config.data.dataset}")
    # Build model description dynamically based on model type
    if config.model.model_type == 'genesis_v2':
        model_desc = f"GenesisV2 (K={config.model.K}, img_size={config.model.img_size})"
    elif config.model.model_type == 'slot_attention':
        model_desc = f"SlotAttention (K={config.model.K}, img_size={config.model.img_size})"
    else:
        model_desc = f"{config.model.model_type} (K={config.model.K}, img_size={config.model.img_size})"
    typer.echo(f"   Model: {model_desc}")
    typer.echo(f"   Training: {config.training.max_steps} steps, batch size {config.training.batch_size}")
    typer.echo(f"   Device: {accelerator.upper()} (precision: {precision})")
    typer.echo(f"   W&B logging: {'Disabled' if disable_wandb else 'Enabled'}")
    if resume:
        typer.echo(f"   Mode: Resume full training state from checkpoint")
    elif checkpoint_path is not None:
        typer.echo(f"   Mode: Transfer learning (weights only from checkpoint)")
        typer.echo(f"   Load weights from: {checkpoint_path}")
    
    if dry_run:
        typer.echo("\n🏃‍♂️ Dry run mode - exiting without training")
        return
    
    # Save the final configuration used for training
    run_dir = Path(config.save_dir) / config.project_name / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_save_path = run_dir / "config.yaml"
    save_config_to_yaml(config, config_save_path)
    typer.echo(f"📁 Saved training config to: {config_save_path}")
    
    # Suppress some warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    
    # Set seeds for reproducibility
    pl.seed_everything(config.data.seed, workers=True)
    
    # Create data module
    datamodule = DataModule(
        config.data, 
        config.training.batch_size,
        num_workers=num_workers
    )
    
    # Get number of batches for validation interval check
    datamodule.setup('fit')
    num_train_batches = len(datamodule.train_dataloader())
    
    # Ensure val_check_interval is not greater than the number of batches
    if config.val_check_interval > num_train_batches:
        config.val_check_interval = num_train_batches
        typer.echo(f"   Override: val_check_interval adjusted to {num_train_batches} (number of batches in train set)")

    # Create model
    model = TrainingModule(config)
    
    # Handle transfer learning mode (load only model + GECO state, not full training state)
    if checkpoint_path is not None and not resume:
        typer.echo(f"🔧 Loading model weights and GECO state from checkpoint...")
        try:
            load_weights_from_checkpoint(model, checkpoint_path)
            typer.echo(f"✅ Successfully loaded weights from: {checkpoint_path}")
        except Exception as e:
            typer.echo(f"❌ Error loading weights: {e}", err=True)
            raise typer.Exit(1)
    
    # Compile model for maximum performance if requested
    if compile_model:
        typer.echo(f"🔧 Compiling model for maximum performance...")
        try:
            # Compile the underlying model within the Lightning module
            model.model = torch.compile(model.model, mode="reduce-overhead", backend="inductor", fullgraph=False)
            typer.echo(f"✅ Model compiled successfully (mode=reduce-overhead, backend=inductor, fullgraph=False)")
        except Exception as e:
            typer.echo(f"⚠️  Model compilation failed: {e}")
            typer.echo(f"   Continuing without compilation...")
    
    # Create logger
    if disable_wandb:
        logger = None
        typer.echo("⚠️  W&B logging disabled")
    else:
        logger = WandbLogger(
            name=config.run_name,
            project=config.project_name,
            save_dir=os.path.join(config.save_dir, config.project_name, config.run_name),
            log_model=False,
        )
    
    # Create trainer with hardware settings
    trainer = create_trainer(config, logger=logger, accelerator=accelerator, devices=devices, precision=precision)
    
    # Start training
    if resume:
        typer.echo(f"\n🚀 Resuming training from checkpoint...")
        ckpt_path = checkpoint_path
    elif checkpoint_path is not None:
        typer.echo(f"\n🚀 Starting fresh training with loaded weights...")
        ckpt_path = None  # Don't pass checkpoint to trainer.fit() in transfer learning mode
    else:
        typer.echo(f"\n🚀 Starting training from scratch...")
        ckpt_path = None
    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        typer.echo(f"✅ Training completed!")
        typer.echo(f"   Checkpoints saved to: {trainer.checkpoint_callback.dirpath}")
    except KeyboardInterrupt:
        typer.echo(f"\n⏹️  Training interrupted by user")
        raise typer.Exit(1)
    except Exception as e:
        import traceback
        typer.echo(f"\n❌ Training failed: {e}", err=True)
        typer.echo(f"\nFull traceback:", err=True)
        typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(1)


@app.command()
def test(
    config_path: str = typer.Option(
        ...,
        "--config", "-c",
        help="Path to the YAML configuration file"
    ),
    checkpoint_path: str = typer.Option(
        ...,
        "--checkpoint", "-k",
        help="Path to model checkpoint file"
    ),
    disable_wandb: bool = typer.Option(
        False,
        "--disable-wandb", "-NW",
        help="Disable Weights & Biases logging"
    )
):
    """
    Test model (GenesisV2 or SlotAttention) with a trained checkpoint.
    
    Loads configuration and checkpoint, then runs testing.
    """
    config_file = Path(config_path)
    checkpoint_file = Path(checkpoint_path)
    
    # Check if files exist
    if not config_file.exists():
        typer.echo(f"❌ Config file not found: {config_file}", err=True)
        raise typer.Exit(1)
    
    if not checkpoint_file.exists():
        typer.echo(f"❌ Checkpoint file not found: {checkpoint_file}", err=True)
        raise typer.Exit(1)
    
    # Load configuration
    try:
        config = load_config_from_yaml(ExperimentConfig, config_file)
        typer.echo(f"✅ Loaded config from: {config_file}")
    except Exception as e:
        typer.echo(f"❌ Error loading config: {e}", err=True)
        raise typer.Exit(1)
    
    # Set seeds for reproducibility
    pl.seed_everything(config.data.seed, workers=True)
    
    # Create data module
    datamodule = DataModule(config.data, config.training.batch_size)
    
    # Load model from checkpoint
    try:
        model = TrainingModule.load_from_checkpoint(
            checkpoint_file,
            config=config
        )
        typer.echo(f"✅ Loaded model from: {checkpoint_file}")
    except Exception as e:
        typer.echo(f"❌ Error loading checkpoint: {e}", err=True)
        raise typer.Exit(1)
    
    # Create logger
    if disable_wandb:
        logger = None
    else:
        logger = WandbLogger(
            name=f"{config.run_name}_test",
            project=config.project_name,
            save_dir=os.path.join(config.save_dir, config.project_name, f"{config.run_name}_test"),
            log_model=False,
        )
    
    # Create trainer (uses auto-detection for hardware)
    trainer = create_trainer(config, logger=logger)
    
    # Run testing
    typer.echo(f"\n🧪 Starting testing...")
    try:
        trainer.test(model, datamodule=datamodule)
        typer.echo(f"✅ Testing completed!")
    except Exception as e:
        import traceback
        typer.echo(f"\n❌ Testing failed: {e}", err=True)
        typer.echo(f"\nFull traceback:", err=True)
        typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(1)


@app.command()
def info():
    """
    Display information about the CLI and available commands.
    """
    typer.echo("🧠 Multi-Model Training CLI")
    typer.echo("=" * 50)
    typer.echo("")
    typer.echo("Available commands:")
    typer.echo("  generate-config  Generate default configuration files")
    typer.echo("  train           Train a model with configuration")
    typer.echo("  test            Test a trained model")
    typer.echo("  info            Show this information")
    typer.echo("")
    typer.echo("Quick start:")
    typer.echo("  1. python cli.py generate-config --output my_config.yaml")
    typer.echo("  2. # Edit my_config.yaml as needed")
    typer.echo("  3. python cli.py train --config my_config.yaml --debug")
    typer.echo("")
    typer.echo("Transfer learning (pretrained weights):")
    typer.echo("  python cli.py train --config my_config.yaml --checkpoint path/to/weights.ckpt")
    typer.echo("")
    typer.echo("Resume training (full state, config from checkpoint):")
    typer.echo("  python cli.py train --checkpoint path/to/checkpoint.ckpt --resume")
    typer.echo("")
    typer.echo("For detailed help on any command:")
    typer.echo("  python cli.py COMMAND --help")


if __name__ == "__main__":
    app() 