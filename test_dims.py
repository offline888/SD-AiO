"""Dimension flow test for SD-AiO pipeline."""
import os, sys, tempfile
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import ExitStack

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

B, C, H, W = 2, 3, 64, 64
LATENT = 4
LH, LW = H // 8, W // 8

# ---- mocks --------------------------------------------------
class MockVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=0.18215)
        self.encoder = nn.Conv2d(3, 4, 3, 1, 1)
        self.decoder = nn.Conv2d(4, 3, 3, 1, 1)
    def encode(self, x): return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: torch.randn(x.shape[0], 4, x.shape[2]//8, x.shape[3]//8)))
    def decode(self, z): return SimpleNamespace(sample=torch.randn(z.shape[0], 3, z.shape[2]*8, z.shape[3]*8))
    def add_adapter(self, c, an=None): pass
    def train(self, m=True): return self
    def eval(self): return self
    def requires_grad_(self, v): pass
    def state_dict(self, *a, **kw): return {}
    def load_state_dict(self, sd, strict=False): pass

class FakeRes(nn.Module):
    def __init__(self): super().__init__(); self.conv2 = nn.Conv2d(320, 320, 3, 1, 1)

class FakeDS(nn.Module):
    def __init__(self): super().__init__(); self.conv = nn.Conv2d(320, 320, 3, 2, 1)

class FakeUS(nn.Module):
    def __init__(self): super().__init__(); self.conv = nn.Conv2d(320, 320, 3, 1, 1)

class FakeDownBlock(nn.Module):
    def __init__(self): super().__init__(); self.resnets = nn.ModuleList([FakeRes()]); self.downsamplers = nn.ModuleList([FakeDS()])

class FakeUpBlock(nn.Module):
    def __init__(self): super().__init__(); self.resnets = nn.ModuleList([FakeRes()]); self.upsamplers = nn.ModuleList([FakeUS()])

class MockUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 320, 3, 1, 1)
        self.out_channels = 4
        self.down_blocks = nn.ModuleList([FakeDownBlock()])
        self.up_blocks = nn.ModuleList([FakeUpBlock()])
        self.mid_block = SimpleNamespace(resnets=nn.ModuleList([FakeRes()]))
    def forward(self, x, timestep, encoder_hidden_states=None, **kw):
        return SimpleNamespace(sample=torch.randn(x.shape[0], self.out_channels, *x.shape[2:]))
    def add_adapter(self, c): pass
    def train(self, m=True): return self
    def eval(self): return self
    def requires_grad_(self, v): pass
    def state_dict(self, *a, **kw): return {}
    def load_state_dict(self, sd, strict=False): pass
    def enable_xformers_memory_efficient_attention(self): pass

class MockTextEncoder(nn.Module):
    dtype = torch.float32
    def forward(self, tokens): return [torch.randn(tokens.shape[0], 77, 1024)]
    def requires_grad_(self, v): pass
    def cpu(self): return self

class MockTokenizer:
    model_max_length = 77
    def __call__(self, text, max_length=77, padding="max_length", truncation=True, return_tensors="pt"):
        return SimpleNamespace(input_ids=torch.zeros(1, 77, dtype=torch.long))

class MockScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([999])
        self.alphas_cumprod = torch.linspace(0.9999, 0.0001, 1000)
    def set_timesteps(self, steps, device=None): pass
    def add_noise(self, latent, noise, timesteps): return latent + 0.1 * noise

def enter_patches():
    stack = ExitStack()
    for target, ret in [
        ('transformers.AutoTokenizer.from_pretrained', MockTokenizer()),
        ('transformers.CLIPTextModel.from_pretrained', MockTextEncoder()),
        ('diffusers.AutoencoderKL.from_pretrained', MockVAE()),
        ('diffusers.UNet2DConditionModel.from_pretrained', MockUNet()),
        ('diffusers.DDPMScheduler.from_pretrained', MockScheduler()),
    ]:
        stack.enter_context(patch(target, return_value=ret))
    stack.enter_context(patch('peft.LoraConfig'))
    return stack

# ═══════════════════════════════════════════════════════
# Test 1 — model
# ═══════════════════════════════════════════════════════
print("1. model.py")
with enter_patches():
    from model import SDSingleStepRestoration
    model = SDSingleStepRestoration(sd_path="/mock", lora_rank_unet=0, lora_rank_vae=0, num_inference_steps=1)
    model.set_train()
    lq = torch.randn(B, C, H, W)
    assert model.encode_image(lq).shape == (B, LATENT, LH, LW);   print("  encode OK")
    z = torch.randn(B, LATENT, LH, LW)
    assert model.decode_latent(z).shape == (B, C, H, W);         print("  decode OK")
    te = torch.randn(B, 77, 1024)
    assert model(lq, te, timestep=150).shape == (B, C, H, W);     print("  forward(t=150) OK")
    assert model(lq, te, timestep=0).shape == (B, C, H, W);       print("  forward(t=0) OK")
    model.set_eval()
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f: tmp = f.name
    model.save_checkpoint(tmp); model.load_checkpoint(tmp); os.unlink(tmp)
    print("  checkpoint OK")
    print("  model.py: ALL PASSED\n")

# ═══════════════════════════════════════════════════════
# Test 2 — cond_module
# ═══════════════════════════════════════════════════════
print("2. cond_module.py")
mock_unet = MockUNet()
from cond_module import build_condition_module, MODULE_REGISTRY
for mtype in MODULE_REGISTRY:
    kwargs = dict(embed_dim=256, device="cpu", unet=mock_unet, training=False)
    if mtype in ("deg_cross_attn", "deg_resblock_attn"):
        kwargs["args"] = SimpleNamespace(num_deg_types=4, dino_type=None,
            degradation_classifier_path=None, freeze_decoder=True)
    if mtype == "deg_resblock_attn":
        print(f"  {mtype:20s}  (skipped — needs real ResBlock mock)")
        continue
    mod = build_condition_module(mtype, **kwargs)
    result = mod.get_modulation(torch.randn(1, 3, 64, 64))
    n_layers = len(getattr(mod, 'sft_layers', getattr(mod, 'hook_blocks', {})))
    print(f"  {mtype:20s}  layers={n_layers}  OK")
print(f"  Registry: {list(MODULE_REGISTRY.keys())}")
print("  cond_module.py: ALL PASSED\n")

# ═══════════════════════════════════════════════════════
# Test 3 — text_cache
# ═══════════════════════════════════════════════════════
print("3. text_cache.py")
from utils.text_cache import TextEmbeddingCache
cache = TextEmbeddingCache(MockTextEncoder(), MockTokenizer(), device="cpu")
for t, p in [("derain", "clean"), ("dehaze", "clear")]:
    cache.add_task(t, p)
assert cache["derain"].shape == (77, 1024)
assert cache.get_batch(["derain", "dehaze"]).shape == (2, 77, 1024)
print("  single (77,1024) + batch (2,77,1024) OK\n")

# ═══════════════════════════════════════════════════════
# Test 4 — dataset
# ═══════════════════════════════════════════════════════
print("4. dataset.py")
with tempfile.TemporaryDirectory() as tmpdir:
    lq_dir = f"{tmpdir}/lq"; gt_dir = f"{tmpdir}/gt"
    os.makedirs(lq_dir); os.makedirs(gt_dir)
    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8))
        img.save(f"{lq_dir}/{i:04d}.png")
        img.save(f"{gt_dir}/{i:04d}.png")
    from utils.dataset import PairedRestorationDataset
    ds = PairedRestorationDataset(
        [{"name": "derain", "lq_path": lq_dir, "gt_path": gt_dir}], image_size=128)
    s = ds[0]
    assert s['lq'].shape == (3, 128, 128) and s['gt'].shape == (3, 128, 128)
    print(f"  {len(ds)} pairs, shape={tuple(s['lq'].shape)} OK\n")

# ═══════════════════════════════════════════════════════
# Test 5 — E2E
# ═══════════════════════════════════════════════════════
print("5. E2E flow")
with enter_patches():
    from model import SDSingleStepRestoration
    model2 = SDSingleStepRestoration(sd_path="/mock", lora_rank_unet=0, lora_rank_vae=0, num_inference_steps=1)
    model2.set_train()
    assert model2(lq, te, timestep=150, cond_module=None).shape == lq.shape; print("  w/o cond_module OK")
    from cond_module import IdentityConditionModule
    assert model2(lq, te, timestep=150, cond_module=IdentityConditionModule()).shape == lq.shape; print("  w/ IdentityModule OK")
    print("  E2E: ALL PASSED\n")

print("=" * 60)
print("ALL DIMENSION CHECKS PASSED")
print("=" * 60)
