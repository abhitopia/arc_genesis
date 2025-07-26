import os
import warnings
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
import wandb

# Import our modules
from .data.d_sprites import VariableDSpritesConfig
from .models.genesis_v2 import GenesisV2Config, GenesisV2
from .models.slot_attention import SlotAttentionConfig, SlotAttentionModel
from .modules.geco import create_geco_for_image_size
from utils.visualisation import make_slot_figure, extract_slot_stats_for_sample
import matplotlib.pyplot as plt


@dataclass
class BaseModelConfig:
    """Base configuration class with common parameters for all models."""
    model_type: str  # Must be specified in subclasses
    K: int = 5  # Number of slots/steps - unified across models
    in_chnls: int = 3
    img_size: int = 64
    feat_dim: int = 64
    broadcast_size: int = 4
    add_coords_every_layer: bool = False
    normal_std: float = 0.7        # std for normal distribution for the mixture model
    use_vae: bool = True           # whether to use VAE (stochastic) latents or deterministic latents
    lstm_hidden_dim: int = 256     # hidden dimension for autoregressive KL loss LSTM
    num_layers: Optional[int] = 4  # Number of layers in LatentDecoder, None = auto (num upsampling stages)


@dataclass
class GenesisV2ModelConfig(BaseModelConfig):
    """Configuration for GenesisV2 model."""
    model_type: str = 'genesis_v2'
    kernel: str = 'gaussian'
    detach_recon_masks: bool = True  # whether to detach reconstructed masks in KL loss


@dataclass
class SlotAttentionModelConfig(BaseModelConfig):
    """Configuration for SlotAttention model."""
    model_type: str = 'slot_attention'
    # K: int = 7 inherited from base (good default for SlotAttention)
    num_iterations: int = 3
    num_heads: int = 4
    slot_dim: Optional[int] = None  # If None, uses feat_dim
    implicit_grads: bool = False


def create_model_config_from_dict(config_dict: Dict[str, Any]) -> BaseModelConfig:
    """
    Factory function to create appropriate model config from dictionary.
    
    Args:
        config_dict: Dictionary containing model configuration
        
    Returns:
        Appropriate model config instance (GenesisV2ModelConfig or SlotAttentionModelConfig)
    """
    model_type = config_dict.get('model_type', 'genesis_v2')
    
    if model_type == 'genesis_v2':
        return GenesisV2ModelConfig(**config_dict)
    elif model_type == 'slot_attention':
        return SlotAttentionModelConfig(**config_dict)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: ['genesis_v2', 'slot_attention']")


def create_model_from_config(config: BaseModelConfig):
    """
    Factory function to create model instance from config.
    
    Args:
        config: Model configuration instance
        
    Returns:
        Model instance (GenesisV2 or SlotAttentionModel)
    """
    config_dict = {k: v for k, v in asdict(config).items() if k != 'model_type'}

    if isinstance(config, GenesisV2ModelConfig):
        # Convert to GenesisV2Config for the model
        model_config = GenesisV2Config(**config_dict)
        return GenesisV2(model_config)
    elif isinstance(config, SlotAttentionModelConfig):
        # Convert to SlotAttentionConfig for the model
        model_config = SlotAttentionConfig(**config_dict)
        return SlotAttentionModel(model_config)
    else:
        raise ValueError(f"Unsupported model config type: {type(config)}")


# Legacy ModelConfig for backward compatibility - maps to GenesisV2ModelConfig
@dataclass 
class ModelConfig(GenesisV2ModelConfig):
    """Legacy ModelConfig - now an alias for GenesisV2ModelConfig for backward compatibility."""
    K: int = 5  # Keep GenesisV2's default for backward compatibility


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Training setup
    batch_size: int = 32
    max_steps: int = 10000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_schedule_patience: int = 5  # validation epochs to wait before reducing LR
    lr_schedule_factor: float = 0.5
    lr_schedule_min_lr: float = 1e-6
    
    # Reconstruction loss weights - control sharpness vs probabilistic behavior
    mixture_weight: float = 1.0     # Weight for mixture model loss (probabilistic)
    mse_weight: float = 0.0         # Weight for MSE loss (sharp reconstructions)
    
    # Auxiliary loss weights - set to 0 to disable computation
    latent_kl_weight: float = 1.0   # Weight for latent KL loss (set to 0 to disable VAE)
    mask_kl_weight: float = 1.0     # Weight for mask KL loss (set to 0 to disable mask consistency)
  
    # Training stability
    elbo_divergence_threshold: float = 1e8  # Stop training if ELBO exceeds this
      
    # GECO control
    use_geco: bool = True           # Whether to use GECO for dynamic beta adjustment
    
    # GECO (Generalized ELBO Constrained Optimization) parameters
    geco_goal: float = 0.5655  # Target reconstruction error per pixel
    geco_lr: float = 1e-5      # GECO learning rate for beta updates
    geco_alpha: float = 0.99   # GECO momentum for error EMA
    geco_beta_init: float = 1.0 # Initial beta value
    geco_beta_min: float = 1e-10 # Minimum beta value
    geco_speedup: float = 10.0  # Scale GECO lr if constraint violation is positive


@dataclass
class DataConfig:
    dataset: 'str' = 'd_sprites'
    variable_size: bool = False
    is_discrete: bool = False  # Use RGB images instead of discrete/categorical
    seed: int = 42

    def __post_init__(self):
        if self.dataset == 'd_sprites':
            self.dataset_config = VariableDSpritesConfig(
                is_discrete=self.is_discrete,
                seed=self.seed,
                min_size=32 if self.variable_size else 64,
                max_size=64,
                num_colors=10,
                num_objects=None,
                unique_colors=True,
                fixed_background=False
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")


@dataclass
class ExperimentConfig:
    """Main experiment configuration combining all sub-configs."""
    
    # Sub-configurations - model can be either GenesisV2 or SlotAttention
    model: Union[GenesisV2ModelConfig, SlotAttentionModelConfig] = field(default_factory=GenesisV2ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment metadata
    project_name: str = "genesis_v2"
    run_name: str = "test_run"
    save_dir: str = "./experiments"
    
    # Logging
    log_every_n_steps: int = 10
    val_check_interval: float = 500  # Check validation every 500 steps
    num_visualizations: int = 8
    
    # Checkpointing
    save_top_k: int = 3
    monitor_metric: str = "loss_val/total_loss"  # For checkpointing
    monitor_mode: str = "min"
    
    def __post_init__(self):
        """Validate compatibility between model and training configurations."""
        # Check VAE compatibility
        if not self.model.use_vae and self.training.latent_kl_weight > 0:
            raise ValueError(
                f"Incompatible configuration: model.use_vae=False but training.latent_kl_weight={self.training.latent_kl_weight} > 0. "
                f"Cannot apply latent KL loss with deterministic latents. "
                f"Either set model.use_vae=True or training.latent_kl_weight=0.0"
            )


def create_default_experiment_for_model(model_type: str = 'genesis_v2') -> ExperimentConfig:
    """
    Create a default experiment configuration for the specified model type.
    
    Args:
        model_type: Type of model ('genesis_v2' or 'slot_attention')
        
    Returns:
        ExperimentConfig with appropriate model configuration
    """
    if model_type == 'genesis_v2':
        model_config = GenesisV2ModelConfig()
    elif model_type == 'slot_attention':
        model_config = SlotAttentionModelConfig()
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: ['genesis_v2', 'slot_attention']")
    
    return ExperimentConfig(model=model_config)


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
        )
    
    def val_dataloader(self):
        return self.val_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return self.test_dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class TrainingModule(pl.LightningModule):    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(vars(config))
        
        # Initialize model
        self.model = create_model_from_config(self.config.model)
        
        # Initialize GECO only if enabled
        if self.config.training.use_geco:
            self.geco = create_geco_for_image_size(
                img_size=self.config.model.img_size,
                channels=self.config.model.in_chnls,
                goal_per_pixel=self.config.training.geco_goal,
                base_step_size=self.config.training.geco_lr,
                alpha=self.config.training.geco_alpha,
                beta_init=self.config.training.geco_beta_init,
                beta_min=self.config.training.geco_beta_min,
                speedup=self.config.training.geco_speedup
            )
        else:
            self.geco = None
            
        # Flag to log static parameters only once
        self._logged_static_params = False
    
    def on_train_start(self):
        """Called when training begins."""
        # Log static parameters once at the start of training
        self._log_static_parameters()
    
    def _log_static_parameters(self):
        """Log static configuration parameters once."""
        if self._logged_static_params:
            return
        
        # Log training configuration
        static_params = {
            "parameters/learning_rate": self.config.training.learning_rate,
            "parameters/weight_decay": self.config.training.weight_decay,
            "parameters/batch_size": self.config.training.batch_size,
            "parameters/max_steps": self.config.training.max_steps,
            "parameters/mixture_weight": self.config.training.mixture_weight,
            "parameters/mse_weight": self.config.training.mse_weight,
            "parameters/latent_kl_weight": self.config.training.latent_kl_weight,
            "parameters/mask_kl_weight": self.config.training.mask_kl_weight,
            "parameters/use_geco": self.config.training.use_geco,
            "parameters/elbo_divergence_threshold": self.config.training.elbo_divergence_threshold,
            "parameters/geco_goal": self.config.training.geco_goal,
            "parameters/geco_lr": self.config.training.geco_lr,
            "parameters/geco_alpha": self.config.training.geco_alpha,
            "parameters/geco_beta_init": self.config.training.geco_beta_init,
            "parameters/geco_beta_min": self.config.training.geco_beta_min,
            "parameters/geco_speedup": self.config.training.geco_speedup,
            "model/img_size": self.config.model.img_size,
            "model/feat_dim": self.config.model.feat_dim,
            "model/normal_std": self.config.model.normal_std,
            "model/lstm_hidden_dim": self.config.model.lstm_hidden_dim,
            "model/K": self.config.model.K,
        }
        
        # Add model-specific parameters based on model type
        if self.config.model.model_type == 'genesis_v2':
            static_params.update({
                "model/kernel": self.config.model.kernel,
                "model/detach_recon_masks": self.config.model.detach_recon_masks,
            })
        elif self.config.model.model_type == 'slot_attention':
            slot_params = {
                "model/num_iterations": self.config.model.num_iterations,
                "model/num_heads": self.config.model.num_heads,
                "model/slot_dim": self.config.model.slot_dim,
                "model/implicit_grads": self.config.model.implicit_grads,
            }
            # Filter out None values (PyTorch Lightning can't log None)
            slot_params = {k: v for k, v in slot_params.items() if v is not None}
            static_params.update(slot_params)
        
        self.log_dict(static_params, on_step=False, on_epoch=True)
        self._logged_static_params = True
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor], mode: str = "train") -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any], bool]:
        """Compute loss and metrics."""
        
        image = batch["image"]
        
        # Forward pass through GenesisV2
        output = self.model(image)
        
        # Extract loss components from GenesisV2 output
        mixture_loss = output["mixture_loss"]  # Probabilistic reconstruction loss
        mse_loss = output["mse_loss"]          # Sharp reconstruction loss
        mask_kl_loss = output["mask_kl_loss"]  # Always computed (cheap)
        
        # Extract KL losses (will be None if compute_latent_kl=False)
        latent_kl_loss = output["latent_kl_loss"]  # Either tensor or None
        latent_kl_loss_per_slot = output["latent_kl_loss_per_slot"]  # Either tensor or None
        
        # Compute normalized reconstruction error (resolution-independent)
        num_elements = image.shape[1:].numel()  # channels * height * width
        mixture_loss_per_element = mixture_loss.mean() / num_elements
        mse_loss_per_element = mse_loss.mean() / num_elements
        
        # Build total reconstruction loss
        total_reconstruction_loss = (
            self.config.training.mixture_weight * mixture_loss + 
            self.config.training.mse_weight * mse_loss
        )
        
        # Build total weighted KL and collect metrics by only adding non-zero weighted terms
        total_weighted_kl = torch.zeros_like(mixture_loss)
        
        # Prepare base metrics
        metrics = {
            # Core loss terms
            f"loss_{mode}/total_loss": None,  # Will be filled after total_loss computation
            f"loss_{mode}/mixture_loss": mixture_loss.mean(),
            f"loss_{mode}/mse_loss": mse_loss.mean(),
            f"loss_{mode}/total_reconstruction": total_reconstruction_loss.mean(),
            f"loss_{mode}/err_per_elem": mixture_loss_per_element,
            f"loss_{mode}/mse_per_elem": mse_loss_per_element,
            f"loss_{mode}/mask_kl": mask_kl_loss.mean(),
        }
        
        # Add latent KL terms is computed (for VAE)
        if latent_kl_loss is not None:
            total_weighted_kl = total_weighted_kl + self.config.training.latent_kl_weight * latent_kl_loss
            metrics.update({
                f"loss_{mode}/latent_kl": latent_kl_loss.mean(),
                f"loss_{mode}/weighted_latent_kl": (self.config.training.latent_kl_weight * latent_kl_loss).mean(),
            })
            
            # Add per-slot latent KL metrics (computed together with latent_kl_loss)
            latent_kl_per_slot_mean = latent_kl_loss_per_slot.mean(0)  # [K] - mean over batch
            for slot_idx in range(latent_kl_per_slot_mean.shape[0]):
                metrics[f"slot_{mode}/kl_latent_{slot_idx}"] = latent_kl_per_slot_mean[slot_idx].item()
        
        # Add mask KL terms and metrics (always - weight of 0 automatically nullifies impact)
        total_weighted_kl = total_weighted_kl + self.config.training.mask_kl_weight * mask_kl_loss
        metrics[f"loss_{mode}/weighted_mask_kl"] = (self.config.training.mask_kl_weight * mask_kl_loss).mean()
        
        # Compute ELBO (reconstruction + weighted KL) - this is our actual training objective
        elbo = total_reconstruction_loss.mean() + total_weighted_kl.mean()
        
        # Compute total loss - either with GECO or fixed weights
        if self.config.training.use_geco:
            # Use GECO for dynamic beta adjustment
            total_loss = self.geco.loss(total_reconstruction_loss.mean(), total_weighted_kl.mean())
            current_beta = self.geco.beta.item()
        else:
            # Use fixed weights - total loss is just the ELBO
            total_loss = elbo
            current_beta = 1.0  # Fixed beta for logging
        
        # Check for ELBO divergence
        elbo_diverged = elbo.item() > self.config.training.elbo_divergence_threshold
        
        # Complete the metrics that needed values computed later
        metrics.update({
            f"loss_{mode}/total_loss": total_loss,
            f"loss_{mode}/total_weighted_kl": total_weighted_kl.mean(),
            f"loss_{mode}/elbo": elbo,
        })
        
        # Add GECO or fixed weight metrics
        if self.config.training.use_geco:
            metrics.update({
                f"geco_{mode}/beta": current_beta,
                f"geco_{mode}/err_ema": self.geco.err_ema.item() if self.geco._err_ema_initialized else 0.0,
                f"geco_{mode}/constraint": (self.geco.goal - self.geco.err_ema).item() if self.geco._err_ema_initialized else 0.0,
            })
        
        return total_loss, metrics, output, elbo_diverged
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, metrics, output, elbo_diverged = self._compute_loss(batch, mode="train")
        
        # Check for ELBO divergence and stop training if detected
        if elbo_diverged:
            elbo_value = metrics["loss_train/elbo"]
            self.print(f"ELBO DIVERGENCE DETECTED: {elbo_value:.2e} > {self.config.training.elbo_divergence_threshold:.2e}")
            self.print("Stopping training to prevent further divergence.")
            self.trainer.should_stop = True
        
        # Log loss and metrics (step-level only for training)
        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        # Log scheduled parameters that change during training
        scheduled_params = {
            "scheduled_params/learning_rate": self.optimizers().param_groups[0]['lr'],
        }
        self.log_dict(scheduled_params, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        loss, metrics, output, elbo_diverged = self._compute_loss(batch, mode="val")
        
        # Log metrics
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # On the first validation batch of each epoch, log visualizations
        if batch_idx == 0:
            self.log_visualizations(batch, output, max_samples=self.config.num_visualizations)
            
        return loss
        
    def log_visualizations(self, batch: Dict[str, torch.Tensor], output: Dict[str, torch.Tensor], max_samples: int = 2):
        """Log visualizations to Wandb."""
        # Ensure we have a logger and it's Wandb
        if not self.logger or not isinstance(self.logger, WandbLogger):
            print("[DEBUG] No Wandb logger found, skipping visualization.")
            return
                    
        for i in range(max_samples):
            image = batch['image'][i]
            recon = output['recon'][i]
            
            # Extract the slot statistics for this specific sample (works for both model types now)
            slot_stats = extract_slot_stats_for_sample(output, sample_idx=i)
            
            # Use unified K parameter for max_slots
            max_slots = self.config.model.K
            
            # Create the figure (works for both model types)
            fig = make_slot_figure(
                image=image,
                recon=recon,
                slot_stats=slot_stats,
                max_slots=max_slots,
                figsize_per_slot=(1.5, 1.5)
            )
      
            # Log to Wandb with a lower DPI for smaller file size
            try:
                self.logger.experiment.log({
                    f"Viz/val_sample_{i}": wandb.Image(fig, caption=f"Sample {i}"),
                    "global_step": self.global_step
                })
            except Exception as e:
                print(f"[DEBUG] ERROR logging val_sample_{i} to Wandb: {e}")
            finally:
                plt.close(fig)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss, metrics, output, elbo_diverged = self._compute_loss(batch, mode="test")
        
        # Log metrics (divergence detection not needed during testing)
        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
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
                "monitor": self.config.monitor_metric,  # Monitor validation loss
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
        filename="epoch_{epoch:02d}-step_{step}-{" + config.monitor_metric + ":.4f}",
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=config.save_top_k,
        save_last=True,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring - always useful for debugging
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
