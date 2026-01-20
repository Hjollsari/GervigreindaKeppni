import numpy as np
import torch
from torch.utils.data import Dataset
import random

def preprocess_patch(patch):
    patch = patch.astype(np.float32)
    
    # 1. Spectral Indices (Feature Engineering)
    # S2 Bands: 2=Green, 3=Red, 7=NIR
    green, red, nir = patch[2], patch[3], patch[7]
    epsilon = 1e-8
    ndvi = (nir - red) / (nir + red + epsilon)
    ndwi = (green - nir) / (green + nir + epsilon)
    
    # Normalize Spectral Bands (Roughly 0-1)
    spectral = np.clip(patch[0:12], 0, 10000) / 3000.0
    img_features = np.concatenate([spectral, np.stack([ndvi, ndwi], axis=0)], axis=0)
    
    # 2. Terrain Features (Standardized for better gradient flow)
    # Extract center pixel for topography
    elev = patch[12, 17, 17]
    slope = patch[13, 17, 17]
    aspect = patch[14, 17, 17]
    
    # Standardizing: (Value - Mean) / Std Dev
    # Iceland average elev is ~500m, slope ~15deg
    elev_norm = (elev - 500.0) / 500.0
    slope_norm = (slope - 15.0) / 15.0
    aspect_rad = np.radians(aspect)
    
    terrain_vec = np.array([
        elev_norm, 
        slope_norm, 
        np.sin(aspect_rad), 
        np.cos(aspect_rad)
    ], dtype=np.float32)
    
    return torch.from_numpy(img_features).float(), torch.from_numpy(terrain_vec).float()

class IcelandHabitatDataset(Dataset):
    def __init__(self, patches, labels=None, augment=False):
        self.patches = patches
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        if self.augment:
            patch = np.rot90(patch, k=random.randint(0, 3), axes=(1, 2))
            if random.random() > 0.5:
                patch = np.flip(patch, axis=2)

        img, terr = preprocess_patch(patch)
        if self.labels is not None:
            return img, terr, torch.tensor(self.labels[idx], dtype=torch.long)
        return img, terr