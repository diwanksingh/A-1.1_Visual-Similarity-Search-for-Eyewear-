import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import CLIPModel, CLIPProcessor

MODEL = "patrickjohncyh/fashion-clip"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("CLIP running on:", device)

model = CLIPModel.from_pretrained(MODEL).to(device)
model.eval()

processor = CLIPProcessor.from_pretrained(MODEL, use_fast=True)

@torch.no_grad()
def encode_image(img):
    if img is None:
        raise RuntimeError("embed_image() called with empty image")

    img = img.resize((224, 224))
    batch = processor(images=img, return_tensors="pt").to(device)
    vec = model.get_image_features(**batch)

    # fashion-clip may return a wrapped object instead of a raw tensor
    if not isinstance(vec, torch.Tensor):
        if hasattr(vec, "pooler_output") and vec.pooler_output is not None:
            vec = vec.pooler_output
        elif hasattr(vec, "last_hidden_state"):
            vec = vec.last_hidden_state[:, 0, :]

    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy().astype("float32")

@torch.no_grad()
def encode_text(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    vec = model.get_text_features(**inputs)

    if not isinstance(vec, torch.Tensor):
        if hasattr(vec, "pooler_output") and vec.pooler_output is not None:
            vec = vec.pooler_output
        elif hasattr(vec, "last_hidden_state"):
            vec = vec.last_hidden_state[:, 0, :]

    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy().astype("float32")