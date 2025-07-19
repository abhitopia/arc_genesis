

import os
import random
import subprocess
import tempfile

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def download_dsprites_dataset():
    """Download dSprites dataset if not already available."""
    # Use system temp directory
    temp_dir = tempfile.gettempdir()
    dsprites_dir = os.path.join(temp_dir, 'dsprites-dataset')
    npz_path = os.path.join(dsprites_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    
    if os.path.exists(npz_path):
        print(f"dSprites dataset already exists at {npz_path}")
        return npz_path
    
    print(f"Downloading dSprites dataset to {dsprites_dir}...")
    
    # Clone the repository
    try:
        subprocess.run([
            'git', 'clone', 
            'https://github.com/deepmind/dsprites-dataset.git',
            dsprites_dir
        ], check=True, capture_output=True, text=True)
        
        if os.path.exists(npz_path):
            print(f"Successfully downloaded dSprites dataset to {npz_path}")
            return npz_path
        else:
            raise FileNotFoundError(f"Dataset file not found at expected path: {npz_path}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


class VariableDSpritesDataset(Dataset):
    """Variable-sized dSprites dataset with configurable color palettes."""

    def __init__(self, num_samples, min_h=5, min_w=5, max_h=32, max_w=32,
                 num_colors=10, is_discrete=True, seed=42, 
                 num_objects=None, unique_colors=True, fixed_background=False):
        """
        Args:
            num_samples: Number of samples to generate
            min_h, min_w: Minimum image dimensions
            max_h, max_w: Maximum image dimensions
            num_colors: Total number of colors in palette (including background)
            is_discrete: True for categorical output, False for RGB output
            seed: Random seed for reproducibility
            num_objects: Fixed number of objects per image (None for random 1-4)
            unique_colors: Whether to enforce unique colors per image
            fixed_background: If True, always use same background color; if False, random background
        """
        # Create RNG instance with composite seed to avoid overlapping datasets
        composite_seed = seed + num_samples + min_h + min_w + max_h + max_w + num_colors
        self.py_rng = random.Random(composite_seed)
        
        # Download and load dSprites dataset
        dsprites_path = download_dsprites_dataset()
        print("Loading dSprites dataset...")
        dataset_zip = np.load(dsprites_path, encoding="latin1")
        self.sprites = dataset_zip['imgs']
        self.num_samples = num_samples
        self.min_h, self.min_w = min_h, min_w
        self.max_h, self.max_w = max_h, max_w
        self.num_colors = num_colors
        self.is_discrete = is_discrete
        self.num_objects = num_objects
        self.unique_colors = unique_colors
        self.fixed_background = fixed_background
        
        # Generate color palette
        self.color_palette = self._generate_color_palette()
        
        # Pre-generate all images and masks
        print(f"Generating {num_samples} variable-sized images...")
        self.images, self.masks = self._generate_dataset()
        print("Dataset generation complete!")
        
        self.to_tensor = transforms.ToTensor()

    def _generate_color_palette(self):
        """Generate a fixed color palette."""
        # Use a mix of distinctive colors
        base_colors = [
            (228, 26, 28),   # Red
            (55, 126, 184),  # Blue
            (77, 175, 74),   # Green
            (152, 78, 163),  # Purple
            (255, 127, 0),   # Orange
            (255, 255, 51),  # Yellow
            (166, 86, 40),   # Brown
            (247, 129, 191), # Pink
            (153, 153, 153), # Gray
            (0, 0, 0),       # Black
            (255, 255, 255), # White
            (128, 0, 0),     # Dark Red
            (0, 128, 0),     # Dark Green
            (0, 0, 128),     # Dark Blue
            (128, 128, 0),   # Olive
        ]
        
        # Limit num_colors to available base colors
        max_colors = len(base_colors)
        if self.num_colors > max_colors:
            print(f"Warning: Requested {self.num_colors} colors, but only {max_colors} base colors available. Using {max_colors} colors.")
            self.num_colors = max_colors
        
        # Take the first num_colors colors
        palette = base_colors[:self.num_colors]
        
        return palette

    def _choose_color_from_palette(self, used_indices=None, is_background=False):
        """Choose a color from the palette."""
        if is_background:
            if self.fixed_background:
                # Always use the last color for background
                idx = self.num_colors - 1
            else:
                # Choose any color from the palette for background
                available_indices = list(range(self.num_colors))
                if used_indices and self.unique_colors:
                    available_indices = [i for i in available_indices if i not in used_indices]
                    if not available_indices:
                        # Fallback if we run out of unique colors
                        available_indices = list(range(self.num_colors))
                idx = self.py_rng.choice(available_indices)
        else:
            # For objects, use any color from the palette
            available_indices = list(range(self.num_colors))
            if used_indices and self.unique_colors:
                available_indices = [i for i in available_indices if i not in used_indices]
                if not available_indices:
                    # Fallback if we run out of unique colors
                    available_indices = list(range(self.num_colors))
            idx = self.py_rng.choice(available_indices)
        
        return idx, self.color_palette[idx]

    def _generate_dataset(self):
        """Generate the complete dataset."""
        all_images = []
        all_masks = []
        
        for i in range(self.num_samples):
            if (i + 1) % 1000 == 0:
                print(f"Processing [{i+1} | {self.num_samples}]")
            
            # Generate random dimensions
            h = self.py_rng.randint(self.min_h, self.max_h)
            w = self.py_rng.randint(self.min_w, self.max_w)
            
            # Create full 64x64 image first
            full_img, full_mask = self._generate_single_image_64x64()
            
            # Crop to desired size
            if h < 64 or w < 64:
                # Intelligent crop - find region with objects
                top, left = self._find_crop_with_objects(full_mask, h, w)
                cropped_img = full_img[top:top+h, left:left+w]
                cropped_mask = full_mask[top:top+h, left:left+w]
            else:
                # If target size is >= 64, we need to pad or resize
                # For simplicity, let's resize
                if not self.is_discrete:
                    pil_img = Image.fromarray((full_img * 255).astype(np.uint8))
                    pil_img = pil_img.resize((w, h), Image.NEAREST)
                    cropped_img = np.array(pil_img).astype(np.float32) / 255.0
                else:
                    pil_img = Image.fromarray(full_img.squeeze().astype(np.uint8))
                    pil_img = pil_img.resize((w, h), Image.NEAREST)
                    cropped_img = np.array(pil_img)[..., np.newaxis]
                
                # Resize each binary mask separately
                cropped_mask = np.zeros((h, w, self.num_colors), dtype=np.uint8)
                for c in range(self.num_colors):
                    pil_mask = Image.fromarray(full_mask[:, :, c].astype(np.uint8))
                    pil_mask = pil_mask.resize((w, h), Image.NEAREST)
                    cropped_mask[:, :, c] = np.array(pil_mask)
            
            all_images.append(cropped_img)
            all_masks.append(cropped_mask)
        
        return all_images, all_masks

    def _find_crop_with_objects(self, mask, crop_h, crop_w):
        """
        Find a crop position that contains objects with at least 60% visibility.
        
        Args:
            mask: Binary mask array of shape (64, 64, num_colors)
            crop_h, crop_w: Desired crop dimensions
        
        Returns:
            (top, left): Crop position
        """
        # Try to find a crop with good object visibility
        best_score = -1
        best_top, best_left = 0, 0
        
        # Try multiple random positions and pick the best one
        num_attempts = min(50, (64 - crop_h + 1) * (64 - crop_w + 1))
        
        for _ in range(num_attempts):
            top = self.py_rng.randint(0, 64 - crop_h)
            left = self.py_rng.randint(0, 64 - crop_w)
            
            score = self._evaluate_crop_quality(mask, top, left, crop_h, crop_w)
            
            if score > best_score:
                best_score = score
                best_top, best_left = top, left
                
                # If we found a crop with good quality, use it
                if score >= 1.0:  # At least one well-visible object
                    break
        
        # Fallback: if no good crop found, use random crop
        if best_score < 0:
            best_top = self.py_rng.randint(0, 64 - crop_h)
            best_left = self.py_rng.randint(0, 64 - crop_w)
        
        return best_top, best_left
    
    def _evaluate_crop_quality(self, mask, top, left, crop_h, crop_w):
        """
        Evaluate crop quality based on object visibility.
        Objects must be either ≥60% visible or not present at all.
        
        Args:
            mask: Binary mask array of shape (64, 64, num_colors)
            top, left: Crop position
            crop_h, crop_w: Crop dimensions
            
        Returns:
            score: Quality score (higher is better, -1 means invalid)
        """
        score = 0
        
        for color_idx in range(mask.shape[2]):
            color_mask = mask[:, :, color_idx]
            
            # Skip if this color has no objects
            if not np.any(color_mask):
                continue
                
            # Count total pixels for this color in the full image
            total_object_pixels = np.sum(color_mask)
            
            # Count pixels for this color in the crop region
            crop_region = color_mask[top:top+crop_h, left:left+crop_w]
            visible_object_pixels = np.sum(crop_region)
            
            # Calculate visibility percentage
            visibility = visible_object_pixels / total_object_pixels
            
            if visible_object_pixels > 0:  # Object is present in crop
                if visibility >= 0.6:  # At least 60% visible
                    score += visibility  # Reward well-visible objects
                else:  # Less than 60% visible - invalid crop
                    return -1
            # If visibility == 0, object is not in crop at all, which is fine
        
        return score

    def _generate_single_image_64x64(self):
        """Generate a single 64x64 image with objects."""
        # Track used colors for uniqueness
        used_color_indices = []
        
        # Choose background color
        bg_idx, bg_rgb = self._choose_color_from_palette(
            used_indices=used_color_indices, is_background=True)
        
        if self.unique_colors:
            used_color_indices.append(bg_idx)
        
        if not self.is_discrete:
            # Create RGB image
            image = np.array(Image.new('RGB', (64, 64), bg_rgb)).astype(np.float32) / 255.0
        else:
            # Create categorical image
            image = np.full((64, 64, 1), bg_idx, dtype=np.uint8)
        
        # Binary masks - one per object color (excluding background)
        binary_masks = np.zeros((64, 64, self.num_colors), dtype=np.uint8)
        
        # Add objects
        if self.num_objects is None:
            num_sprites = self.py_rng.randint(1, 4)
        else:
            num_sprites = self.num_objects
            
        for obj_idx in range(num_sprites):
            # Choose random sprite
            object_index = self.py_rng.randint(0, len(self.sprites) - 1)
            sprite_mask = np.array(self.sprites[object_index], dtype=bool)
            
            # Choose object color
            obj_color_idx, obj_rgb = self._choose_color_from_palette(
                used_indices=used_color_indices, is_background=False)
            
            if self.unique_colors:
                used_color_indices.append(obj_color_idx)
            
            # Find sprite coordinates
            crop_index = np.where(sprite_mask == True)
            
            if not self.is_discrete:
                # Update RGB image
                obj_color_normalized = np.array(obj_rgb, dtype=np.float32) / 255.0
                image[crop_index] = obj_color_normalized
            else:
                # Update categorical image
                image[crop_index] = obj_color_idx
            
            # Update binary masks - set object's color mask
            binary_masks[crop_index[0], crop_index[1], obj_color_idx] = 1
        
        return image, binary_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        
        if not self.is_discrete:
            # Convert to tensor and ensure correct shape (C, H, W)
            img = self.to_tensor(img)
        else:
            # For categorical data, convert to tensor manually
            img = torch.from_numpy(img.transpose(2, 0, 1)).long()
        
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        
        return {
            'input': img,
            'target': mask
        }


def create_dataset(num_samples=1000, min_h=5, min_w=5, max_h=32, max_w=32,
                   num_colors=10, is_discrete=True, seed=42,
                   num_objects=None, unique_colors=True, fixed_background=False):
    """
    Create a variable-sized dSprites dataset.
    
    Args:
        num_samples: Number of samples to generate
        min_h, min_w: Minimum image dimensions
        max_h, max_w: Maximum image dimensions
        num_colors: Total number of colors in palette
        is_discrete: True for categorical output, False for RGB output
        seed: Random seed for reproducibility
        num_objects: Fixed number of objects per image (None for random 1-4)
        unique_colors: Whether to enforce unique colors per image
        fixed_background: If True, always use same background color
    
    Returns:
        VariableDSpritesDataset: The created dataset
    """
    dataset = VariableDSpritesDataset(
        num_samples=num_samples,
        min_h=min_h,
        min_w=min_w,
        max_h=max_h,
        max_w=max_w,
        num_colors=num_colors,
        is_discrete=is_discrete,
        seed=seed,
        num_objects=num_objects,
        unique_colors=unique_colors,
        fixed_background=fixed_background
    )
    
    return dataset


def visualize_samples(dataset, num_samples=5):
    """
    Visualize the first few samples from the dataset.
    
    Args:
        dataset: VariableDSpritesDataset instance
        num_samples: Number of samples to visualize
    """
    for sample_idx in range(min(num_samples, len(dataset))):
        sample = dataset[sample_idx]
        img = sample['input']
        mask = sample['target']
        
        # Convert tensors to numpy for visualization
        if dataset.is_discrete:
            # For discrete images, convert to RGB using the color palette
            img_np = img.squeeze().numpy()  # Remove channel dimension
            h, w = img_np.shape
            rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
            
            for color_idx in range(dataset.num_colors):
                color_mask = img_np == color_idx
                if np.any(color_mask):
                    rgb_img[color_mask] = dataset.color_palette[color_idx]
        else:
            # For RGB images, convert CHW to HWC
            img_np = img.permute(1, 2, 0).numpy()
            rgb_img = (img_np * 255).astype(np.uint8)
        
        # Get binary masks
        mask_np = mask.numpy()  # Shape: (num_colors, H, W)
        active_classes = [idx for idx in range(mask_np.shape[0]) if np.any(mask_np[idx] > 0)]
        
        # Create subplot grid: 1 row for image + N columns for active binary masks
        num_cols = len(active_classes) + 1
        fig, axes = plt.subplots(1, num_cols, figsize=(3*num_cols, 3))
        if num_cols == 1:
            axes = [axes]
        
        # Show original image
        axes[0].imshow(rgb_img)
        axes[0].set_title(f'Sample {sample_idx+1}\n{rgb_img.shape[:2]}')
        axes[0].axis('off')
        
                 # Show each active binary mask (only for object colors, not background)
        for col_idx, class_idx in enumerate(active_classes):
            binary_mask = mask_np[class_idx]
            axes[col_idx + 1].imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
            axes[col_idx + 1].set_title(f'Object Color {class_idx}\nRGB: {dataset.color_palette[class_idx]}')
            axes[col_idx + 1].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """Example usage of the dataset creation function."""
    # Create a sample dataset
    dataset = create_dataset(
        num_samples=5000,
        min_h=16,
        min_w=16,
        max_h=32,
        max_w=32,
        num_colors=10,
        is_discrete=True,
        seed=44
    )
    
    # Print some statistics
    print(f"\nDataset Statistics:")
    print(f"Number of samples: {len(dataset)}")
    print(f"Number of colors: {dataset.num_colors}")
    print(f"Image size range: ({dataset.min_h}, {dataset.min_w}) to ({dataset.max_h}, {dataset.max_w})")
    
    # Show a sample
    sample = dataset[0]
    print(f"Sample image shape: {sample['input'].shape}")
    print(f"Sample mask shape: {sample['target'].shape}")
    print(f"Sample image dtype: {sample['input'].dtype}")
    print(f"Sample mask dtype: {sample['target'].dtype}")
    
    # Visualize first 5 samples
    print("\nVisualizing first 5 samples...")
    visualize_samples(dataset, num_samples=5)
    
    print("Example completed!")


if __name__ == "__main__":
    main() 