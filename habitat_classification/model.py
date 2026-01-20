import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataset import preprocess_patch

# --- ARCHITECTURE ---
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return F.relu(x + self.conv(x))

class HabitatModel(nn.Module):
    def __init__(self, num_classes=71):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResBlock(256),
            nn.AdaptiveAvgPool2d(1) 
        )
        self.mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, img_input, terrain_input):
        x1 = self.cnn(img_input).view(img_input.size(0), -1)
        x2 = terrain_input.view(terrain_input.size(0), -1)
        x2 = self.mlp(x2)
        combined = torch.cat((x1, x2), dim=1)
        return self.classifier(combined)

# --- ENSEMBLE LOADING ---
device = torch.device("cpu")
models = []

# Load Model A (The Champion)
path_a = Path(__file__).parent / "model_A.pth"
if path_a.exists():
    try:
        m1 = HabitatModel(num_classes=71)
        m1.load_state_dict(torch.load(path_a, map_location=device))
        m1.eval()
        models.append(m1)
        print("Loaded Model A")
    except: pass

# Load Model B (The Challenger)
path_b = Path(__file__).parent / "model_B.pth"
if path_b.exists():
    try:
        m2 = HabitatModel(num_classes=71)
        m2.load_state_dict(torch.load(path_b, map_location=device))
        m2.eval()
        models.append(m2)
        print("Loaded Model B")
    except: pass

def predict(patch: np.ndarray) -> int:
    if not models: return 0
    
    img, terr = preprocess_patch(patch)
    img = img.unsqueeze(0)
    terr = terr.unsqueeze(0)
    
    total_logits = None
    
    with torch.no_grad():
        for m in models:
            logits = m(img, terr)
            if total_logits is None:
                total_logits = logits
            else:
                total_logits += logits
                
        # Average results (Voting)
        avg_logits = total_logits / len(models)
        return int(torch.argmax(avg_logits, dim=1).item())