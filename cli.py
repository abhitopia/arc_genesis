"""
cli.py
======
Command-line interface for Genesis V2 training using typer.

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
    create_trainer
)
from src.config_utils import save_config_to_yaml, load_config_from_yaml


app = typer.Typer(
    name="genesis-cli",
    help="Genesis V2 training CLI",
    no_args_is_help=True
)


def create_default_experiment() -> ExperimentConfig:
    """Create a default experiment configuration."""
    return ExperimentConfig()


@app.command()
def generate_config(
    output_path: str = typer.Option(
        "config.yaml",
        "--output", "-o",
        help="Output path for the generated config file"
    )
):
    """
    Generate a default configuration file.
    
    Creates a YAML configuration file with default parameters that can be
    modified and used for training.
    """
    output_file = Path(output_path)
    
    # Create default config
    config = create_default_experiment()
    typer.echo(f"Generating default experiment config...")
    
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
    config_path: str = typer.Option(
        ...,
        "--config", "-c",
        help="Path to the YAML configuration file"
    ),
    cpu: bool = typer.Option(
        False,
        "--cpu", "-C",
        help="Force CPU usage (disable GPU)"
    ),
    compile_model: bool = typer.Option(
        False,
        "--compile",
        help="Compile model with torch.compile for maximum performance"
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
        4,
        "--num-workers", "-NW",
        help="Override number of data loading workers"
    )
):
    """
    Train Genesis V2 model with the specified configuration.
    
    Loads configuration from a YAML file and starts training. 
    Command-line options can override specific config values.
    """
    config_file = Path(config_path)
    
    # Check if config file exists
    if not config_file.exists():
        typer.echo(f"❌ Config file not found: {config_file}", err=True)
        typer.echo(f"   Generate one with: genesis-cli generate-config --output {config_file}")
        raise typer.Exit(1)
    
    # Load configuration
    try:
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
    typer.echo(f"   Model: Genesis V2 (K_steps={config.model.K_steps}, img_size={config.model.img_size})")
    typer.echo(f"   Training: {config.training.max_steps} steps, batch size {config.training.batch_size}")
    typer.echo(f"   Device: {accelerator.upper()} (precision: {precision})")
    typer.echo(f"   W&B logging: {'Disabled' if disable_wandb else 'Enabled'}")
    
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
    if config.training.val_check_interval > num_train_batches:
        config.training.val_check_interval = num_train_batches
        typer.echo(f"   Override: val_check_interval adjusted to {num_train_batches} (number of batches in train set)")

    # Create model
    model = TrainingModule(config)
    
    # Compile model for maximum performance if requested
    if compile_model:
        typer.echo(f"🔧 Compiling model for maximum performance...")
        try:
            # Compile the underlying Genesis V2 model within the Lightning module
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
    typer.echo(f"\n🚀 Starting training...")
    try:
        trainer.fit(model, datamodule=datamodule)
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
    Test Genesis V2 model with a trained checkpoint.
    
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
    typer.echo("🧠 Genesis V2 Training CLI")
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
    typer.echo("For detailed help on any command:")
    typer.echo("  python cli.py COMMAND --help")


if __name__ == "__main__":
    app() 