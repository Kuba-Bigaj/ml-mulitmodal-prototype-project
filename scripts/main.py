import torch
from transformers import AutoProcessor, AutoModel
from class_embedding_generator import ClassEmbeddingGenerator

model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224", use_fast = True)
print(f"Model loaded on device: {model.device}")

generator = ClassEmbeddingGenerator(model=model, processor=processor)

embeddings = generator.generate_class_embeddings("./data")

for cname, embedding in embeddings:
    if embedding is not None:
        print(embedding.size())
    else:
        print("None")
