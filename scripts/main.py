import torch
from transformers import AutoProcessor, AutoModel
from class_embedding_generator import ClassEmbeddingGenerator
from scripts.sample_classifier import SampleClassifier

model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224", use_fast = True)
print(f"Model loaded on device: {model.device}")

sample_classifier = SampleClassifier(model=model, processor=processor)
print(sample_classifier.classify("./sample.jpg"))