

import os
import random
import subprocess
import tempfile
import hashlib
import pickle
import functools
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
        
        # Try to load from cache first
        cache_path = self._get_cache_path()
        if self._load_from_cache(cache_path):
            print(f"Loaded {num_samples} images from cache: {cache_path}")
        else:
            # Pre-generate all images and masks
            print(f"Generating {num_samples} variable-sized images...")
            self.images, self.masks = self._generate_dataset()
            print("Dataset generation complete!")
            
            # Save to cache for next time
            self._save_to_cache(cache_path)
            print(f"Cached dataset to: {cache_path}")
        
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

    def _get_cache_key(self):
        """Generate a unique cache key based on dataset parameters."""
        # Include all parameters that affect the generated dataset
        cache_params = {
            'num_samples': self.num_samples,
            'min_h': self.min_h,
            'min_w': self.min_w,
            'max_h': self.max_h,
            'max_w': self.max_w,
            'num_colors': self.num_colors,
            'is_discrete': self.is_discrete,
            'num_objects': self.num_objects,
            'unique_colors': self.unique_colors,
            'fixed_background': self.fixed_background,
            'py_rng_seed': self.py_rng.getstate()[1][0]  # Get the seed from RNG state
        }
        
        # Create hash from parameters
        params_str = str(sorted(cache_params.items()))
        cache_key = hashlib.md5(params_str.encode()).hexdigest()
        return cache_key

    def _get_cache_path(self):
        """Get the cache file path for this dataset configuration."""
        # Use the same temp directory as dSprites
        temp_dir = tempfile.gettempdir()
        cache_dir = os.path.join(temp_dir, 'dsprites-dataset', 'variable_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_key = self._get_cache_key()
        cache_filename = f"variable_dsprites_{cache_key}.pkl"
        return os.path.join(cache_dir, cache_filename)

    def _save_to_cache(self, cache_path):
        """Save generated images and masks to cache."""
        try:
            cache_data = {
                'images': self.images,
                'masks': self.masks,
                'color_palette': self.color_palette,
                'cache_params': {
                    'num_samples': self.num_samples,
                    'min_h': self.min_h, 'min_w': self.min_w,
                    'max_h': self.max_h, 'max_w': self.max_w,
                    'num_colors': self.num_colors,
                    'is_discrete': self.is_discrete,
                    'num_objects': self.num_objects,
                    'unique_colors': self.unique_colors,
                    'fixed_background': self.fixed_background
                }
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _load_from_cache(self, cache_path):
        """Load images and masks from cache if available and valid."""
        if not os.path.exists(cache_path):
            return False
            
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify cache parameters match current configuration
            cached_params = cache_data['cache_params']
            current_params = {
                'num_samples': self.num_samples,
                'min_h': self.min_h, 'min_w': self.min_w,
                'max_h': self.max_h, 'max_w': self.max_w,
                'num_colors': self.num_colors,
                'is_discrete': self.is_discrete,
                'num_objects': self.num_objects,
                'unique_colors': self.unique_colors,
                'fixed_background': self.fixed_background
            }
            
            if cached_params != current_params:
                print("Cache parameters don't match current configuration, regenerating...")
                return False
            
            # Load cached data
            self.images = cache_data['images']
            self.masks = cache_data['masks']
            self.color_palette = cache_data['color_palette']
            
            return True
            
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return False

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
            'image': img,
            'target': mask
        }
    

    def get_dataloader(self, batch_size=32, max_h=None, max_w=None, padding_value=0, shuffle=True, num_workers=0):
        """
        Create a DataLoader with custom collate function for variable-sized images.
        
        Args:
            batch_size: Batch size
            max_h, max_w: Maximum dimensions to pad to. If None, uses batch maximum.
            padding_value: Value to use for padding images (e.g., 0, -1)
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader instance
        """
        # Use functools.partial to create a picklable collate function
        collate_fn = functools.partial(
            collate_variable_size, 
            max_h=max_h, 
            max_w=max_w, 
            padding_value=padding_value
        )
        
        # Enable persistent workers and pin memory for faster data loading
        use_fast_loading = num_workers > 0
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=use_fast_loading,
            persistent_workers=use_fast_loading,
            drop_last=True
        )


@dataclass
class VariableDSpritesConfig:
    """Configuration for variable-sized dSprites dataset."""
    
    num_train: int = 50000
    num_val: int = 2048
    num_test: int = 2048
    min_size: int = 32
    max_size: Optional[int] = None  # If None, same as min_size
    num_colors: int = 10
    is_discrete: bool = True
    seed: int = 42
    num_objects: Optional[int] = None
    unique_colors: bool = True
    fixed_background: bool = False
    
    def get_dataset(self, split: str = 'train') -> VariableDSpritesDataset:
        """
        Create a single variable-sized dSprites dataset from this configuration.
        
        Args:
            split: Which split to create ('train', 'val', or 'test')
        
        Returns:
            VariableDSpritesDataset: The created dataset
        """
        if split == 'train':
            num_samples = self.num_train
            seed = self.seed
        elif split == 'val':
            num_samples = self.num_val
            seed = self.seed + 1  # Different seed for validation
        elif split == 'test':
            num_samples = self.num_test
            seed = self.seed + 2  # Different seed for test
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")
        
        # Handle max_size=None case
        max_size = self.max_size if self.max_size is not None else self.min_size
        
        dataset = VariableDSpritesDataset(
            num_samples=num_samples,
            min_h=self.min_size,
            min_w=self.min_size,
            max_h=max_size,
            max_w=max_size,
            num_colors=self.num_colors,
            is_discrete=self.is_discrete,
            seed=seed,
            num_objects=self.num_objects,
            unique_colors=self.unique_colors,
            fixed_background=self.fixed_background
        )
        
        return dataset
    
    def get_datasets(self) -> dict[str, VariableDSpritesDataset]:
        """
        Create train, validation, and test datasets from this configuration.
        
        Returns:
            dict: Dictionary with keys 'train', 'val', 'test' and dataset values
        """
        return {
            'train': self.get_dataset('train'),
            'val': self.get_dataset('val'),
            'test': self.get_dataset('test')
        }


def collate_variable_size(batch, max_h=None, max_w=None, padding_value=-1):
    """
    Custom collate function for variable-sized images.
    Pads all images and masks to the same size.
    
    Args:
        batch: List of samples from dataset
        max_h, max_w: Maximum dimensions to pad to. If None, uses batch maximum.
        padding_value: Value to use for padding images (e.g., 0, -1)
        
    Returns:
        Batch dictionary with padded tensors and padding mask
    """
    # Extract images and masks
    images = [sample['image'] for sample in batch]
    masks = [sample['target'] for sample in batch]
    
    # Determine target size
    if max_h is None or max_w is None:
        # Find maximum dimensions in this batch
        batch_max_h = max(img.shape[-2] for img in images)
        batch_max_w = max(img.shape[-1] for img in images)
        target_h = max_h if max_h is not None else batch_max_h
        target_w = max_w if max_w is not None else batch_max_w
    else:
        target_h, target_w = max_h, max_w
    
    # Pad images and masks, create padding masks
    padded_images = []
    padded_masks = []
    padding_masks = []
    
    for img, mask in zip(images, masks):
        # Get current dimensions
        if img.dim() == 3:  # (C, H, W)
            _, curr_h, curr_w = img.shape
        else:  # (H, W) for discrete
            curr_h, curr_w = img.shape
        
        # Calculate padding
        pad_h = target_h - curr_h
        pad_w = target_w - curr_w
        
        # Create padding mask (True = real image, False = padding)
        padding_mask = torch.ones((target_h, target_w), dtype=torch.bool)
        if pad_h > 0 or pad_w > 0:
            # Set padding regions to False
            if pad_h > 0:
                padding_mask[curr_h:, :] = False
            if pad_w > 0:
                padding_mask[:, curr_w:] = False
        
        # Pad image
        if img.dim() == 3:  # RGB images (C, H, W)
            padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=padding_value)
        else:  # Discrete images (H, W)
            padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=padding_value)
        
        # Pad target mask (C, H, W) where C = num_colors - always pad with 0
        padded_mask = F.pad(mask, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        padded_images.append(padded_img)
        padded_masks.append(padded_mask)
        padding_masks.append(padding_mask)
    
    # Stack into batch tensors
    if padded_images[0].dim() == 3:  # RGB images
        batch_images = torch.stack(padded_images)
    else:  # Discrete images - need to add channel dimension
        batch_images = torch.stack([img.unsqueeze(0) for img in padded_images])
    
    batch_masks = torch.stack(padded_masks)
    batch_padding_masks = torch.stack(padding_masks)
    
    return {
        'image': batch_images,
        'target': batch_masks,
        'mask': batch_padding_masks
    }





def visualize_samples(dataset, num_samples=5):
    """
    Visualize the first few samples from the dataset.
    
    Args:
        dataset: VariableDSpritesDataset instance
        num_samples: Number of samples to visualize
    """
    for sample_idx in range(min(num_samples, len(dataset))):
        sample = dataset[sample_idx]
        img = sample['image']
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
    """Example usage of the dataset config and creation."""
    # Create datasets using config
    config = VariableDSpritesConfig(
        num_train=1000,  # Smaller numbers for demo
        num_val=200,
        num_test=200,
        min_size=16,
        max_size=32,
        num_colors=10,
        is_discrete=False,
        seed=44
    )
    
    # Get all datasets
    datasets = config.get_datasets()
    train_dataset = datasets['train']
    val_dataset = datasets['val']
    test_dataset = datasets['test']
    
    # Print some statistics
    print(f"\nDataset Statistics:")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of colors: {train_dataset.num_colors}")
    print(f"Image size range: ({train_dataset.min_h}, {train_dataset.min_w}) to ({train_dataset.max_h}, {train_dataset.max_w})")
    
    # Show a sample from train dataset
    sample = train_dataset[0]
    print(f"Sample image shape: {sample['image'].shape}")
    print(f"Sample mask shape: {sample['target'].shape}")
    print(f"Sample image dtype: {sample['image'].dtype}")
    print(f"Sample mask dtype: {sample['target'].dtype}")
    
    # Visualize first 5 samples from train dataset
    print("\nVisualizing first 5 samples from training set...")
    visualize_samples(train_dataset, num_samples=5)
    
    # Demonstrate DataLoader usage
    print("\nTesting DataLoader...")
    dataloader = train_dataset.get_dataloader(
        batch_size=4, 
        max_h=32, 
        max_w=32, 
        padding_value=-1,  # Use -1 for padding
        shuffle=True
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    print(f"Batch input shape: {batch['image'].shape}")
    print(f"Batch target shape: {batch['target'].shape}")
    print(f"Batch mask shape: {batch['mask'].shape}")
    print(f"Input dtype: {batch['image'].dtype}")
    print(f"Target dtype: {batch['target'].dtype}")
    print(f"Mask dtype: {batch['mask'].dtype}")
    
    # Show padding statistics
    mask = batch['mask'][0]  # First sample's padding mask
    total_pixels = mask.numel()
    real_pixels = mask.sum().item()
    padding_pixels = total_pixels - real_pixels
    print(f"Sample 0: {real_pixels}/{total_pixels} real pixels, {padding_pixels} padding pixels")
    
    # Check padding values
    # Need to expand mask to match image channels
    first_image = batch['image'][0]  # Shape: (C, H, W)
    if first_image.dim() == 3:  # RGB/multi-channel image
        expanded_mask = mask.unsqueeze(0).expand_as(first_image)  # (C, H, W)
        padded_regions = first_image[~expanded_mask]
    else:  # Single channel image
        padded_regions = first_image[~mask]
    
    if len(padded_regions) > 0:
        print(f"Padding values in input: {torch.unique(padded_regions)}")
    else:
        print("No padding in first sample")
    
    print("Example completed!")


if __name__ == "__main__":
    main() 