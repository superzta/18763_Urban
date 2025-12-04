import os
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import joblib
from PIL import Image
import numpy as np

# --------------------
# 1. Define model (same as before)
# --------------------
class Identity(nn.Module):
    def forward(self, x):
        return x

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True, use_bn=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=use_bias and not use_bn)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.layers = nn.Sequential(
            LinearLayer(in_features, hidden_features, use_bias=True, use_bn=True),
            nn.ReLU(),
            LinearLayer(hidden_features, out_features, use_bias=False, use_bn=True)
        )

    def forward(self, x):
        return self.layers(x)

class PreModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ResNet18 backbone
        self.pretrained = models.resnet18(pretrained=True)

        # remove maxpool and adjust conv1
        self.pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.pretrained.maxpool = Identity()

        # Remove final FC layer
        self.pretrained.fc = Identity()

        for p in self.pretrained.parameters():
            p.requires_grad = True

        self.projector = ProjectionHead(
            in_features=512,
            hidden_features=512,
            out_features=128
        )

    def forward(self, x):
        h = self.pretrained(x)          # (B, 512)
        z = self.projector(h)           # (B, 128)
        return h, F.normalize(z, dim=1)

# --------------------
# 2. Load model + kmeans
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PreModel().to(device)
model.load_state_dict(torch.load("clustering_checkpoints/simclr_best_adam_damageroad.pth", map_location=device))
model.eval()

kmeans = joblib.load("clustering_checkpoints/kmeans_severity_damageroad.pkl")

# --------------------
# 3. Transform (must match training)
# --------------------
clean_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# --------------------
# 4. Cluster → severity mapping
# --------------------
cluster_to_severity = {
    0: "weak",
    1: "moderate",
    2: "heavy"
}

# --------------------
# 5. Loop over test/ images with timing
# --------------------
test_dir = "test_img/damageroad"
# adjust extensions if needed
image_paths = sorted(
    glob.glob(os.path.join(test_dir, "*.png"))
    + glob.glob(os.path.join(test_dir, "*.jpg"))
    + glob.glob(os.path.join(test_dir, "*.jpeg"))
)

if not image_paths:
    print(f"No images found in {test_dir}/")
else:
    times = []
    results = []

    # Optional: warm-up a couple of runs so timing is more stable on GPU
    with torch.no_grad():
        dummy = torch.randn(1, 3, 128, 128).to(device)
        for _ in range(3):
            _ = model(dummy)

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[SKIP] Failed to open {img_path}: {e}")
            continue

        tensor = clean_transform(img).unsqueeze(0).to(device)

        # ---- timing start ----
        start = time.perf_counter()

        with torch.no_grad():
            h, _ = model(tensor)
            # make sure all CUDA ops finish before stopping timer
            if device.type == "cuda":
                torch.cuda.synchronize()
            h_np = h.cpu().numpy()
            cluster_id = kmeans.predict(h_np)[0]

        end = time.perf_counter()
        # ---- timing end ----

        elapsed = end - start
        times.append(elapsed)

        severity = cluster_to_severity.get(cluster_id, f"cluster_{cluster_id}")
        results.append((os.path.basename(img_path), severity, elapsed))

        print(f"{os.path.basename(img_path)} → severity: {severity} | inference time: {elapsed*1000:.2f} ms")

    # --------------------
    # 6. Summary
    # --------------------
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print("\n=== Inference Summary ===")
        print(f"Images processed: {len(times)}")
        print(f"Average time: {avg_time*1000:.2f} ms")
        print(f"Min time:     {min_time*1000:.2f} ms")
        print(f"Max time:     {max_time*1000:.2f} ms")
