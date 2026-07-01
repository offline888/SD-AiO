import torch


class TextEmbeddingContainer:
    def __init__(self, text_encoder, tokenizer, device, dtype=None):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype or text_encoder.dtype
        self.embeddings = {}

    def add_embedding(self, task_name, prompt):
        if task_name in self.embeddings:
            return
        tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)
        with torch.no_grad():
            embedding = self.text_encoder(tokens)[0].detach()
        self.embeddings[task_name] = embedding.to(dtype=self.dtype).squeeze(0)

    def get_batch(self, task_names, device=None):
        if device is None:
            device = self.device

        embeddings = [self.embeddings[name] for name in task_names]
        stacked = torch.stack(embeddings)

        if device != self.device or self.dtype != stacked.dtype:
            stacked = stacked.to(device=device, dtype=self.dtype)
        return stacked

    def __getitem__(self, name):
        return self.embeddings[name]

    def __contains__(self, name):
        return name in self.embeddings

    def __len__(self):
        return len(self.embeddings)

    @property
    def task_names(self):
        return list(self.embeddings.keys())
