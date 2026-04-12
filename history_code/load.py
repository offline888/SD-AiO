from diffusers import Flux2KleinIRPipeline
import torch

pipeline = Flux2KleinIRPipeline.from_pretrained(
    '/home/yhmi/data/model/flux.2-klein',
    torch_dtype=torch.bfloat16,
)
