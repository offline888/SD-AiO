import os, torch, numpy as np
from PIL import Image
import torchvision.transforms as T
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import transformers
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, ckpt, dino_path):
        super().__init__()
        self.encoder = transformers.Dinov2Model.from_pretrained(dino_path)
        self.decoder = nn.Module()
        self.decoder.norm = nn.LayerNorm(768)
        self.decoder.fc = nn.Sequential(
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Linear(384, ckpt["decoder.fc.2.weight"].shape[0]),
        )
        self.load_state_dict(ckpt, strict=True)
        self.eval()

    def forward(self, x):
        out = self.encoder(pixel_values=x)
        return out.last_hidden_state[:, 0]

DATA_ROOT = "/home/yhmi/data/patches"
MODEL_PATH = "/home/yhmi/data/model/best_model.pth"
DINO_PATH = "/home/yhmi/data/model/dinov2-base"
NUM = 500
SIZE = 518

transform = T.Compose([
    T.Resize((SIZE, SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model = Model(ckpt, DINO_PATH).cuda().eval()

X, y, names = [], [], []
for label, subdir in enumerate(sorted(os.listdir(DATA_ROOT))):
    d = os.path.join(DATA_ROOT, subdir, "GT")
    if not os.path.isdir(d): continue
    imgs = sorted(os.listdir(d))[:NUM]
    feats, batch = [], []
    for f in imgs:
        batch.append(transform(Image.open(os.path.join(d, f)).convert("RGB")))
        if len(batch) == 32:
            with torch.no_grad():
                feats.append(model(torch.stack(batch).cuda()).cpu())
            batch = []
    if batch:
        with torch.no_grad():
            feats.append(model(torch.stack(batch).cuda()).cpu())
    feat = torch.cat(feats)
    X.append(feat)
    y.append(np.full(len(feat), label))
    names.append(subdir)
    print(f"{subdir}: {len(feat)}")

X = torch.cat(X).numpy()
y = np.concatenate(y)

emb = TSNE(2, perplexity=30, random_state=0, max_iter=1000).fit_transform(X)

plt.figure(figsize=(10, 8))
for i, name in enumerate(names):
    m = y == i
    plt.scatter(emb[m, 0], emb[m, 1], label=name, s=20, alpha=0.6)
plt.legend()
plt.savefig("tsne_result.png", dpi=150, bbox_inches="tight")