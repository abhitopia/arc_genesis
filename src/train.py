import os
import warnings
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

# Import our modules
from data.d_sprites import VariableDSpritesConfig
from utils import cosine_schedule



@dataclass
class ModelConfig:
    """Configuration for PCAE model parameters."""
    
    # Model architecture
    num_affine_caps: int = 8
    num_similarity_caps: int = 8
    num_templates: Optional[int] = None
    hidden_dim: int = 128
    n_conv_layers: int = 4
    n_transform_layers: int = 2
    n_heads: int = 4
    
    # Template settings
    template_size: int = 11
    use_alpha_channel: bool = True
    colored_templates: bool = False
    allow_flip: bool = True
    
    # Input/output settings
    input_channels: int = 3  # For RGB: 3, for grayscale: 1, for discrete: depends on encoding
    num_colors: Optional[int] = None  # None for RGB/grayscale, palette size for discrete


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Training setup
    batch_size: int = 32
    max_steps: int = 10000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Loss weights
    nll_weight: float = 1.0
    mse_weight: float = 0.0
    neighborhood_agreement_weight: float = 0.01
    centre_of_mass_weight: float = 0.01
    compactness_weight: float = 0.01
    
    # Training dynamics
    gumbel_tau_start: float = 1.0
    gumbel_tau_end: float = 0.0
    gumbel_tau_decay_steps: int = 5000
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_schedule_patience: int = 5  # validation epochs to wait before reducing LR
    lr_schedule_factor: float = 0.5
    lr_schedule_min_lr: float = 1e-6


@dataclass
class DataConfig:
    dataset: 'str' = 'd_sprites'
    variable_size: bool = False
    is_discrete: bool = True
    seed: int = 42

    def __post_init__(self):
        if self.dataset == 'd_sprites':
            self.dataset_config = VariableDSpritesConfig(
                is_discrete=self.is_discrete,
                seed=self.seed,
                min_size=32 if self.variable_size else 64,
                max_size=64,
                num_colors=10,
                num_objects=3,
                unique_colors=True,
                fixed_background=False
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")


@dataclass
class ExperimentConfig:
    """Main experiment configuration combining all sub-configs."""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment metadata
    project_name: str = "genesis_v2"
    run_name: str = "test_run"
    save_dir: str = "./experiments"
    
    # Logging
    log_every_n_steps: int = 10
    val_check_interval: float = 500  # Check validation every 500 steps
    
    # Checkpointing
    save_top_k: int = 3
    monitor_metric: str = "loss/val_nll_loss"  # For checkpointing
    monitor_mode: str = "min"
    
    # Learning rate scheduler monitoring (separate from checkpointing)
    lr_monitor_metric: str = "loss/val_nll_loss"  # Monitor validation loss


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for synthetic data."""
    
    def __init__(self, config: DataConfig, batch_size: int, num_workers: int = 4):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages."""
        if stage == "fit" or stage is None:
            self.train_dataset = self.config.dataset_config.get_dataset('train')
            self.val_dataset = self.config.dataset_config.get_dataset('val')
        if stage == "test" or stage is None:
            self.test_dataset = self.config.dataset_config.get_dataset('test')
    
    def train_dataloader(self):
        return self.train_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self):
        return self.val_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def test_dataloader(self):
        return self.test_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


class LightningModule(pl.LightningModule):
    """PyTorch Lightning module wrapping PCAE."""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(vars(config))
        
        # Initialize model
        self.model = self._create_model()
        
        # Flag to log static parameters only once
        self._logged_static_params = False
    
    def _create_model(self) -> PCAE:
        """Create PCAE model based on configuration."""
        model_cfg = self.config.model
        data_cfg = self.config.data
        
        # Determine input channels and num_colors based on data mode
        if data_cfg.mode.upper() == "RGB":
            assert model_cfg.input_channels == 3, "RGB data must have 3 input channels"
            num_colors = None
        elif data_cfg.mode.upper() == "DISCRETE":
            # For discrete mode, we typically use embeddings or one-hot encoding
            num_colors = data_cfg.n_colours
        else:
            raise ValueError(f"Unsupported data mode: {data_cfg.mode}")
        
        return PCAE(
            num_affine_caps=model_cfg.num_affine_caps,
            num_similarity_caps=model_cfg.num_similarity_caps,
            input_channels=model_cfg.input_channels,
            use_alpha_channel=model_cfg.use_alpha_channel,
            hidden_dim=model_cfg.hidden_dim,
            n_conv_layers=model_cfg.n_conv_layers,
            n_transform_layers=model_cfg.n_transform_layers,
            n_heads=model_cfg.n_heads,
            input_size=data_cfg.max_hw,  # Use maximum image size from data config
            template_size=model_cfg.template_size,
            allow_flip=model_cfg.allow_flip,
            colored_templates=model_cfg.colored_templates,
            num_templates=model_cfg.num_templates,
            num_colors=num_colors,
        )
    
    def _get_gumbel_tau(self) -> float:
        """Get current Gumbel temperature using cosine decay schedule."""
        return cosine_schedule(
            start_value=self.config.training.gumbel_tau_start,
            end_value=self.config.training.gumbel_tau_end,
            current_step=self.global_step,
            total_steps=self.config.training.gumbel_tau_decay_steps
        )
    
    def _visualize_templates(self) -> Optional[Dict[str, Any]]:
        """
        Visualize templates using utility functions.
        
        Returns:
            Dictionary with W&B template visualizations or None if W&B unavailable
        """
        try:
            # Get template visualizations as numpy arrays
            templates_dict = visualize_templates(self.model.decoder, return_format="numpy")
            
            # Convert to W&B format
            wandb_dict = templates_to_wandb_images(templates_dict)
            
            # Add "templates/" prefix to keys
            return {f"templates/{key}": images for key, images in wandb_dict.items()}
            
        except ImportError:
            warnings.warn("wandb not available, template visualization disabled")
            return None
        except Exception as e:
            warnings.warn(f"Template visualization failed: {e}")
            return None

    # ------------------------------------------------------------------
    # New: full model visualisation on a *single* validation sample
    # ------------------------------------------------------------------
    def _visualize_sample(self, output: Dict[str, Any], sample_idx: int = 0):
        """Create a figure showing original, reconstruction and per-capsule details.

        Returns a matplotlib Figure or None if visualisation fails.
        """
        try:
            from src.utils import visualize_model_prediction  # imported lazily to avoid heavy deps at import time
            fig = visualize_model_prediction(output, self.model.decoder, sample_idx=sample_idx)
            return fig
        except Exception as e:
            warnings.warn(f"Sample visualisation failed: {e}")
            return None
    
    def _log_static_parameters(self):
        """Log static configuration parameters once."""
        if self._logged_static_params:
            return
        
        # Log loss weights
        static_params = {
            "parameters/nll_weight": self.config.training.nll_weight,
            "parameters/mse_weight": self.config.training.mse_weight,
            "parameters/neighborhood_agreement_weight": self.config.training.neighborhood_agreement_weight,
            "parameters/centre_of_mass_weight": self.config.training.centre_of_mass_weight,
            "parameters/compactness_weight": self.config.training.compactness_weight,
            "parameters/learning_rate": self.config.training.learning_rate,
            "parameters/weight_decay": self.config.training.weight_decay,
            "parameters/batch_size": self.config.training.batch_size,
            "parameters/gumbel_tau_start": self.config.training.gumbel_tau_start,
            "parameters/gumbel_tau_end": self.config.training.gumbel_tau_end,
            "parameters/gumbel_tau_decay_steps": self.config.training.gumbel_tau_decay_steps,
        }
        
        self.log_dict(static_params, on_step=False, on_epoch=True)
        self._logged_static_params = True
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor], mode: str = "train", return_output: bool = False) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]]:
        """Compute loss and metrics."""
        
        image = batch["image"]
        # Create mask - synthetic data doesn't always provide masks
        if "mask" in batch:
            mask = batch["mask"]
        else:
            B, *spatial_dims = image.shape[:1] + image.shape[-2:]  # Handle both RGB and discrete
            mask = torch.ones(B, *spatial_dims[-2:], device=image.device, dtype=torch.bool)
        
        # Forward pass
        output = self.model(image, mask=mask, tau=self._get_gumbel_tau())
        
        # Extract individual loss components
        nll_loss = output["nll_loss"]
        mse_loss = output["mse_loss"]
        neighborhood_agreement = output["neighborhood_agreement"]
        centre_of_mass = output["centre_of_mass"]
        compactness = output["compactness"]
        
        # Compute weighted total loss
        training_cfg = self.config.training
        total_loss = (
            training_cfg.nll_weight * nll_loss +
            training_cfg.mse_weight * mse_loss +
            training_cfg.neighborhood_agreement_weight * neighborhood_agreement +
            training_cfg.centre_of_mass_weight * centre_of_mass +
            training_cfg.compactness_weight * compactness
        )
        
        # Prepare metrics dict with organized sections
        metrics = {
            f"loss/{mode}_total_loss": total_loss,
            f"loss/{mode}_nll_loss": nll_loss,
            f"loss/{mode}_mse_loss": mse_loss,
            f"regularization/{mode}_neighborhood_agreement": neighborhood_agreement,
            f"regularization/{mode}_centre_of_mass": centre_of_mass,
            f"regularization/{mode}_compactness": compactness,
        }
        
        if return_output:
            return total_loss, metrics, output
        else:
            return total_loss, metrics
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Log static parameters once
        self._log_static_parameters()
        
        loss, metrics = self._compute_loss(batch, mode="train")
        
        # Log loss and regularization metrics (step-level only for training)
        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        # Log scheduled parameters that change during training
        scheduled_params = {
            "scheduled_params/gumbel_tau": self._get_gumbel_tau(),
            "scheduled_params/learning_rate": self.optimizers().param_groups[0]['lr'],
        }
        self.log_dict(scheduled_params, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Get loss, metrics, and full output for visualization
        loss, metrics, output = self._compute_loss(batch, mode="val", return_output=True)
        
        # Log metrics
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Log template visualizations (only for the first validation batch to avoid too many images)
        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            template_viz = self._visualize_templates()
            if template_viz:
                # Log template visualizations
                for key, images in template_viz.items():
                    self.logger.experiment.log({key: images}, step=self.global_step)

            # Log *sample* visualisations (first few images in the batch)
            import matplotlib.pyplot as plt, warnings as _w
            max_samples_to_log = 3
            try:
                import wandb
            except ImportError:
                _w.warn("wandb not available – skipping sample visualisations")
                return loss

            for i in range(min(max_samples_to_log, output['images'].shape[0])):
                fig = self._visualize_sample(output, sample_idx=i)
                if fig is not None:
                    self.logger.experiment.log({f"samples/val_sample_{i}": wandb.Image(fig)}, step=self.global_step)
                    plt.close(fig)
  
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss, metrics = self._compute_loss(batch, mode="test")
        
        # Log metrics
        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        if not self.config.training.use_lr_scheduler:
            return optimizer
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.training.lr_schedule_factor,
            patience=self.config.training.lr_schedule_patience,
            min_lr=self.config.training.lr_schedule_min_lr
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.config.lr_monitor_metric,  # Monitor validation loss
                "frequency": 1,  # Check scheduler every validation epoch
                "interval": "epoch"
                # Note: With epoch-level monitoring, patience controls validation epochs
                # Current: patience=5 means wait 5 validation epochs before reducing LR
                # Since validation runs every 500 training steps, this gives time for improvement
            }
        }


def create_trainer(config: ExperimentConfig, logger=None, accelerator: str = None, devices=None, precision: str = None) -> pl.Trainer:
    """Create PyTorch Lightning trainer with callbacks."""
    
    # Create callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.save_dir, config.project_name, config.run_name, "checkpoints"),
        filename="epoch_{epoch:02d}-step_{step}-{val_loss/val_nll_loss:.4f}",
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=config.save_top_k,
        save_last=True,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring
    if config.training.use_lr_scheduler:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    
    # Progress bar - show step-based progress when using max_steps
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)
    
    # Use provided hardware settings or auto-detect
    if accelerator is None or devices is None or precision is None:
        if torch.cuda.is_available():
            accelerator = accelerator or "cuda"
            devices = devices or "auto"
            precision = precision or "bf16-mixed"
        else:
            accelerator = accelerator or "cpu"
            devices = devices or "auto"
            precision = precision or "32-true"
    
    trainer = pl.Trainer(
        max_steps=config.training.max_steps,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        gradient_clip_val=1.0,  # Gradient clipping for stability
        deterministic=False,
        num_sanity_val_steps=0,  # Disable sanity checking
    )
    
    return trainer


def create_discrete_experiment():
    """Example function to create a discrete data experiment."""
    config = ExperimentConfig()
    
    # Configure for discrete data
    config.run_name = "pcae_discrete_experiment"
    config.data.mode = "DISCRETE"
    config.data.n_colours = 10
    config.model.input_channels = 8         # Will be set automatically but shown for clarity
    config.model.num_colors = 10
    config.model.colored_templates = False  # Use monochrome templates for discrete data
    
    return config


def create_rgb_experiment():
    """Example function to create an RGB data experiment."""
    config = ExperimentConfig()
    
    # Configure for RGB data
    config.run_name = "pcae_rgb_experiment"
    config.data.mode = "RGB"
    config.model.input_channels = 3
    config.model.num_colors = None
    config.model.colored_templates = False
    
    return config



