import torch
from transformers import CLIPModel, CLIPProcessor

MODEL = "openai/clip-vit-base-patch32"

# Choose device cpu or gpu prefer gpu if available
def choose_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = choose_device()
print("CLIP running on:", device)

model = CLIPModel.from_pretrained(MODEL).to(device)
model.eval()

processor = CLIPProcessor.from_pretrained(MODEL)
@torch.no_grad()
def encode_image(img):
    # quick sanity check
    if img is None:
        raise RuntimeError("embed_image() called with empty image")

    batch = processor(images=img, return_tensors="pt").to(device)
    vec = model.get_image_features(**batch)

    # normalize so we can compare vectors directly
    vec = vec / vec.norm(dim=-1, keepdim=True)

    return vec.cpu().numpy().astype("float32")

