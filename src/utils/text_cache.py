"""Pre-compute and cache CLIP text embeddings per task (zero-cost training lookup)."""
import torch


class TextEmbeddingCache:

    def __init__(self, text_encoder, tokenizer, device, dtype=None):
        self._text_encoder = text_encoder
        self._tokenizer = tokenizer
        self._device = device
        self._dtype = dtype or text_encoder.dtype
        self._cache = {}

    def add_task(self, name, prompt):
        if name in self._cache:
            return
        tokens = self._tokenizer(
            prompt,
            max_length=self._tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        ).input_ids.to(self._device)
        with torch.no_grad():
            embedding = self._text_encoder(tokens)[0].detach()
        self._cache[name] = embedding.to(dtype=self._dtype).squeeze(0)

    def get_batch(self, task_names, device=None):
        if device is None:
            device = self._device
        embeddings = [self._cache[name] for name in task_names]
        stacked = torch.stack(embeddings)
        if device != self._device or self._dtype != stacked.dtype:
            stacked = stacked.to(device=device, dtype=self._dtype)
        return stacked

    def __getitem__(self, name):
        return self._cache[name]

    def __contains__(self, name):
        return name in self._cache

    def __len__(self):
        return len(self._cache)

    @property
    def task_names(self):
        return list(self._cache.keys())
