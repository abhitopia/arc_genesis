import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from data import DSpritesDataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.d_sprites import VariableDSpritesConfig
from model import SlotAttentionModel

def beta_schedule(step, warmup=5000, max_beta=1e-4):
    """
    Beta schedule for KL loss weight in VAE training.
    
    Args:
        step: Current training step
        warmup: Number of steps for linear ramp-up
        max_beta: Maximum beta value after warmup
    
    Returns:
        Current beta value
    """
    if step < warmup:
        return max_beta * (step / warmup)        # linear ramp
    return max_beta

class Trainer:

    def __init__(self, model, optimizer, device, mask_entropy_weight=1e-4):

        self.model  = model
        self.optimizer = optimizer
        self.loss = nn.MSELoss()
        self.device = device
        self.mask_entropy_weight = mask_entropy_weight
        self.model.to(self.device)
        self.step =0

    def train_step(self, inputs, beta=0.0):

        inputs = inputs.to(self.device)
        recon_combined, recon, masks, mask_logits, scopes, slots, kl_loss = self.model(inputs)
        recon_loss = self.loss(recon_combined, inputs)
        
        # Compute mask entropy regularization
        # masks shape: (B, K, 1, H, W)
        log_masks = (masks + 1e-8).log()
        mask_entropy = -(masks * log_masks).sum(dim=[2,3,4]).mean()   # average over batch & slots
        
        # Total loss with entropy regularization and KL loss
        loss = recon_loss + self.mask_entropy_weight * mask_entropy + beta * kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, recon_loss, mask_entropy, kl_loss

    def test_step(self, inputs, beta=0.0):

        inputs = inputs.to(self.device)
        with torch.no_grad():
            recon_combined, recon, masks, mask_logits, scopes, slots, kl_loss = self.model(inputs)
            recon_loss = self.loss(recon_combined, inputs)
            
            # Compute mask entropy for logging
            log_masks = (masks + 1e-8).log()
            mask_entropy = -(masks * log_masks).sum(dim=[2,3,4]).mean()
            
            loss = recon_loss + self.mask_entropy_weight * mask_entropy + beta * kl_loss
        return loss, recon_loss, mask_entropy, kl_loss

    def visualize(self, inputs):

        inputs = inputs.to(self.device)
        with torch.no_grad():
            recon_combined, recon, masks, mask_logits, scopes, slots, kl_loss = self.model(inputs)

        batch_size, num_slots, C, H, W = recon.shape
        
        # Use structured layout for ordered slots (stick-breaking), simple layout for unordered (softmax)
        use_structured_layout = mask_logits is not None and self.model.ordered_slots
        
        if not use_structured_layout:
            # Fallback to original single row visualization for softmax normalization
            out = torch.cat(
                [
                    inputs.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recon * masks + (1 - masks),  # each slot
                ],
                dim=1,
            )
            out = (out * 0.5 + 0.5).clamp(0, 1)
            images = make_grid(
                out.view(batch_size * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
            )
            return images
        
        # Structured layout for stick-breaking/ordered slots
        
        # Compute raw alphas (what each slot "thinks")
        raw_alphas = torch.sigmoid(mask_logits)  # B, S, 1, H, W
        
        # Convert single-channel alphas to 3-channel for visualization (grayscale)
        raw_alphas_rgb = raw_alphas.repeat(1, 1, 3, 1, 1)  # B, S, 3, H, W
        
        # Check if we're in sequential mode (has scopes)
        is_sequential = scopes is not None
        
        if is_sequential:
            # Sequential mode: 4 rows per sample
            # Row 0: [Original,      Slot1_scope,      Slot2_scope,      ...]
            # Row 1: [Reconstruction, Slot1_recon,     Slot2_recon,     ...]
            # Row 2: [Padding,       Slot1_raw_alpha,  Slot2_raw_alpha,  ...]  
            # Row 3: [Padding,       Slot1_recon×raw,  Slot2_recon×raw,  ...]
            
            # Convert scopes to 3-channel for visualization
            scopes_rgb = scopes.repeat(1, 1, 3, 1, 1)  # B, S, 3, H, W
            
            grid = torch.zeros(batch_size, 4, num_slots + 1, C, H, W, device=inputs.device)
            
            # Row 0: Original + slot scopes
            grid[:, 0, 0] = inputs  # Original images
            grid[:, 0, 1:] = scopes_rgb  # Scopes (grayscale)
            
            # Row 1: Reconstruction + individual slot reconstructions
            grid[:, 1, 0] = recon_combined  # Combined reconstruction
            grid[:, 1, 1:] = recon  # Individual slot reconstructions
            
            # Row 2: Padding + raw slot alphas
            grid[:, 2, 0] = torch.ones_like(inputs)  # White padding
            grid[:, 2, 1:] = raw_alphas_rgb  # Raw slot alphas (grayscale)
            
            # Row 3: Padding + reconstruction weighted by raw alphas
            grid[:, 3, 0] = torch.ones_like(inputs)  # White padding
            grid[:, 3, 1:] = recon * raw_alphas + (1 - raw_alphas)  # Recon × raw alphas with white background
            
            num_rows = 4
            
        else:
            # Parallel mode: 3 rows per sample  
            # Row 0: [Original,       Slot1_normalized, Slot2_normalized, ...]
            # Row 1: [Reconstruction, Slot1_raw_alpha,  Slot2_raw_alpha,  ...]  
            # Row 2: [Padding,        Slot1_recon×raw,  Slot2_recon×raw,  ...]
            
            grid = torch.zeros(batch_size, 3, num_slots + 1, C, H, W, device=inputs.device)
            
            # Row 0: Original image + normalized slot masks
            grid[:, 0, 0] = inputs  # Original images
            grid[:, 0, 1:] = recon * masks + (1 - masks)  # Normalized slot masks with white background
            
            # Row 1: Reconstruction + raw slot alphas  
            grid[:, 1, 0] = recon_combined  # Combined reconstruction
            grid[:, 1, 1:] = raw_alphas_rgb  # Raw slot alphas (grayscale)
            
            # Row 2: Padding + reconstruction weighted by raw alphas
            grid[:, 2, 0] = torch.ones_like(inputs)  # White padding for first column
            grid[:, 2, 1:] = recon * raw_alphas + (1 - raw_alphas)  # Recon × raw alphas with white background
            
            num_rows = 3
        
        # Normalize to [0, 1] range
        grid = (grid * 0.5 + 0.5).clamp(0, 1)
        
        # Reshape for make_grid
        grid_reshaped = grid.view(batch_size * num_rows, num_slots + 1, C, H, W)
        grid_flat = grid_reshaped.view(batch_size * num_rows * (num_slots + 1), C, H, W)
        
        # Create final grid with nrow=num_slots+1 to maintain column structure
        images = make_grid(
            grid_flat.cpu(), normalize=False, nrow=num_slots + 1,
        )

        return images

    def save(self, dir_path):

        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step
            }, os.path.join(dir_path, f'ckpt_{self.step}.pt'))

    def restore(self, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            state_dict = torch.load(f)
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.step = state_dict['step']

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--use_implicit_grads", action='store_true')
    parser.add_argument("--n_samples", type=int, default=8, help='number of sample imgs to visualize')
    parser.add_argument("--exp_name", type=str, default='slot_attention', help='name of experiment')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_slots", type=int, default=5)
    parser.add_argument("--num_slot_iters", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=500000)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument('--warmup_steps', default=10000, type=int, help='warmup steps for the learning rate.')
    parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
    parser.add_argument('--decay_steps', default=100000, type=int, help='Steps for the learning rate decay.')
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--ckpt_interval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--restore", action='store_true')
    parser.add_argument("--restore_path", type=str, help='checkpoint path to restore')
    parser.add_argument("--compile", action='store_true', help='compile model with torch.compile for performance')
    parser.add_argument("--decoder_num_layers", type=int, default=7, help='number of layers in LatentDecoder (default: 6)')
    parser.add_argument("--base_ch", type=int, default=64, help='base channels for encoder (default: 32)')
    parser.add_argument("--bottleneck_hw", type=int, default=8, help='encoder bottleneck spatial size (default: 8)')
    parser.add_argument("--mask_entropy_weight", type=float, default=1e-4, help='weight for mask entropy regularization (default: 1e-4)')
    parser.add_argument("--no_encoder_pos_embed", action='store_true', help='disable encoder position embedding')
    parser.add_argument("--no_ordered_slots", action='store_true', help='disable ordered slot initialization (use shared mu/sigma instead)')
    parser.add_argument("--max_beta", type=float, default=1e-4, help='maximum beta for VAE KL loss (default: 0.0, disables VAE)')
    parser.add_argument("--beta_warmup", type=int, default=5000, help='warmup steps for beta schedule (default: 5000)')
    parser.add_argument("--heads", type=int, default=4, help='number of attention heads for multi-head slot attention (default: 4)')
    parser.add_argument("--sequential", action='store_true', help='use sequential (MONet-style) slot attention instead of parallel')
  
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/')
    torch.set_float32_matmul_precision('high')

    if not (os.path.exists(results_dir)):
        os.makedirs(results_dir)

    logdir = args.exp_name + '_' + time.strftime("%d-%m-%Y-%H-%M-%S")
    logdir = os.path.join(results_dir, logdir)

    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')

    # Create VariableDSpritesDataset configuration
    # Using RGB images (is_discrete=False) to match slot attention expectations
    config = VariableDSpritesConfig(
        num_train=50000,
        num_val=2000, 
        num_test=2000,
        min_size=32,  # Match the model's expected resolution
        max_size=32,  # Fixed size for slot attention
        num_colors=10,
        is_discrete=False,  # Use RGB images
        seed=42,
        num_objects=None,  # Random 1-4 objects
        unique_colors=True,
        fixed_background=False
    )
    
    train_dataset = DSpritesDataset(config=config, d_set='train')
    test_dataset = DSpritesDataset(config=config, masks=True, d_set='test')

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True
                              )

    test_loader   = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              pin_memory=True)

    # Determine if VAE should be used based on max_beta
    use_vae = args.max_beta > 0.0

    model = SlotAttentionModel(
                    resolution=32,  # Single int since image is square
                    num_slots = args.num_slots,
                    num_iters = args.num_slot_iters,
                    in_channels =3,
                    base_ch = args.base_ch,  # Now configurable
                    bottleneck_hw = args.bottleneck_hw,  # Now configurable
                    slot_size = args.base_ch,
                    slot_mlp_size = 2 * args.base_ch,
                    decoder_num_layers=args.decoder_num_layers,
                    use_encoder_pos_embed=not args.no_encoder_pos_embed,
                    sequential=args.sequential,  # Enable sequential (MONet-style) attention
                    ordered_slots=not args.no_ordered_slots,
                    implicit_grads = args.use_implicit_grads,
                    use_vae = use_vae,
                    heads = args.heads)
    
    # Compile model if requested and available
    if args.compile:
        if hasattr(torch, 'compile'):
            print("Compiling model with torch.compile...")
            model = torch.compile(model)
        else:
            print("Warning: torch.compile not available, skipping compilation")
    
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)

    trainer = Trainer(model, optimizer, device, mask_entropy_weight=args.mask_entropy_weight)
    writer = SummaryWriter(logdir)

    perm = torch.randperm(args.batch_size)
    idx = perm[: args.n_samples]
    fixed_imgs = (next(iter(test_loader))['imgs'])[idx]  # for visualizing slots during training

    if args.train:

        print("Training")
        start_time = time.time()
        step=0
        if args.restore:
            trainer.restore(args.restore_path)
            if trainer.step !=0:
                step = trainer.step + 1
        
        while step <= args.max_steps:

            for batch in train_loader:

                trainer.model.train()
                if step < args.warmup_steps:
                    learning_rate = args.learning_rate * (step/ args.warmup_steps)
                else:
                    learning_rate = args.learning_rate

                learning_rate = learning_rate * (args.decay_rate ** (
                    step / args.decay_steps))

                trainer.optimizer.param_groups[0]['lr'] = learning_rate

                # Compute current beta for VAE
                current_beta = beta_schedule(step, warmup=args.beta_warmup, max_beta=args.max_beta)

                loss, recon_loss, mask_entropy, kl_loss = trainer.train_step(batch['imgs'], beta=current_beta)

                if step % args.log_interval == 0:
                    # Log training metrics
                    writer.add_scalar('train_loss', loss.item(), step)
                    writer.add_scalar('train_recon_loss', recon_loss.item(), step)
                    writer.add_scalar('train_mask_entropy', mask_entropy.item(), step)
                    writer.add_scalar('train_kl_loss', kl_loss.item(), step)
                    writer.add_scalar('beta', current_beta, step)
                    
                    trainer.model.eval()
                    sample_imgs = trainer.visualize(fixed_imgs)
                    writer.add_image(f'slots at epoch {step}', sample_imgs, step)
                    save_image(sample_imgs, os.path.join(logdir, f'slots_at_{step}.jpg'))

                    # Compute test metrics
                    total_loss = 0
                    total_recon_loss = 0
                    total_mask_entropy = 0
                    total_kl_loss = 0
                    for batch in test_loader:
                        test_loss, test_recon_loss, test_mask_entropy, test_kl_loss = trainer.test_step(batch['imgs'], beta=current_beta)
                        total_loss += test_loss.item()
                        total_recon_loss += test_recon_loss.item()
                        total_mask_entropy += test_mask_entropy.item()
                        total_kl_loss += test_kl_loss.item()
                    
                    avg_test_loss = total_loss / len(test_loader)
                    avg_test_recon_loss = total_recon_loss / len(test_loader)
                    avg_test_mask_entropy = total_mask_entropy / len(test_loader)
                    avg_test_kl_loss = total_kl_loss / len(test_loader)

                    # Log test metrics
                    writer.add_scalar('test_loss', avg_test_loss, step)
                    writer.add_scalar('test_recon_loss', avg_test_recon_loss, step)
                    writer.add_scalar('test_mask_entropy', avg_test_mask_entropy, step)
                    writer.add_scalar('test_kl_loss', avg_test_kl_loss, step)

                    print("###############################")
                    print(f"At training step {step}")
                    print("###############################")
                    print(f"Train_loss: {loss.item():.6f}")
                    print(f"  └─ Recon: {recon_loss.item():.6f}")
                    print(f"  └─ Entropy: {mask_entropy.item():.6f} (λ*H: {(trainer.mask_entropy_weight * mask_entropy).item():.6f})")
                    print(f"  └─ KL: {kl_loss.item():.6f} (β*KL: {(current_beta * kl_loss).item():.6f})")
                    print(f"Test_loss: {avg_test_loss:.6f}")
                    print(f"  └─ Recon: {avg_test_recon_loss:.6f}")
                    print(f"  └─ Entropy: {avg_test_mask_entropy:.6f} (λ*H: {(trainer.mask_entropy_weight * avg_test_mask_entropy):.6f})")
                    print(f"  └─ KL: {avg_test_kl_loss:.6f} (β*KL: {(current_beta * avg_test_kl_loss):.6f})")
                    print(f"Beta: {current_beta:.6f}")

                    time_since_start = time.time() - start_time
                    print(f"Time Since Start {time_since_start:.6f}")

                if step % args.ckpt_interval == 0:
                    trainer.save(logdir)

                step+=1
                trainer.step = step
         
    if args.test:
        print("Testing")
        trainer.model.eval()
        current_beta = beta_schedule(trainer.step, warmup=args.beta_warmup, max_beta=args.max_beta)
        total_loss = 0
        total_recon_loss = 0
        total_mask_entropy = 0
        total_kl_loss = 0
        for batch in test_loader:
            loss, recon_loss, mask_entropy, kl_loss = trainer.test_step(batch['imgs'], beta=current_beta)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_mask_entropy += mask_entropy.item()
            total_kl_loss += kl_loss.item()
        avg_test_loss = total_loss / len(test_loader)
        avg_test_recon_loss = total_recon_loss / len(test_loader)
        avg_test_mask_entropy = total_mask_entropy / len(test_loader)
        avg_test_kl_loss = total_kl_loss / len(test_loader)
        print(f"Val_loss: {avg_test_loss:.6f}")
        print(f"  └─ Recon: {avg_test_recon_loss:.6f}")
        print(f"  └─ Entropy: {avg_test_mask_entropy:.6f} (λ*H: {(trainer.mask_entropy_weight * avg_test_mask_entropy):.6f})")
        print(f"  └─ KL: {avg_test_kl_loss:.6f} (β*KL: {(current_beta * avg_test_kl_loss):.6f})")
        print(f"Beta: {current_beta:.6f}")

    if args.visualize:
        print("Visualize Slots")
        trainer.model.eval()
        perm = torch.randperm(args.batch_size)
        idx = perm[: args.n_samples]
        batch = next(iter(test_loader))['imgs'][idx]
        images = trainer.visualize(batch)
        save_image(images, os.path.join(logdir, f'slots_at_test.jpg'))
        

if __name__ == '__main__':
    main()