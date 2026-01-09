import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224", use_fast = True)
print(f"Model loaded on device: {model.device}")


# load data
images = []
base_path = "../data"
for file in [file for file in os.listdir(base_path) if file.endswith(('.png', '.jpg', '.jpeg'))]:
    image = Image.open(os.path.join(base_path, file)).convert("RGB")
    images.append(image)

texts = ["a house cat prowling outside in the snow", "a pallas cat prowling outside", "a human sitting on a chair"]

# IMPORTANT: use `padding=max_length` and `max_length=64`
inputs = processor(text=texts, images=images, padding="max_length", max_length=64, return_tensors="pt").to(model.device)


# forward pass
with torch.no_grad():
    outputs = model(**inputs)


# extract the image and text embeddings - to be used in further work
image_vectors = outputs.image_embeds
text_vectors = outputs.text_embeds


# compute the similarity scores
logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image) # raw probability
# probs = torch.softmax(logits_per_image, dim=-1) # normalized probability
for i in range(len(images)):
    print(f"\nImage {i}:")
    for j in range(len(texts)):
        print(f"{probs[i][j]:.1%} that image {i} is '{texts[j]}'")
