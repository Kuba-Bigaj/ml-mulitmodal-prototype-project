import torch
import torchvision
import matplotlib.pyplot as plt
import random
import PIL.Image as Image
import torch
from transformers import AutoModel, AutoProcessor, SiglipVisionModel, SiglipVisionConfig


def get_image_embeddings() -> dict[int, torch.Tensor]:
    dataset = torchvision.datasets.STL10(root='../data', split='train', download=True)

    config = SiglipVisionConfig.from_pretrained("google/siglip2-base-patch16-224")
    model = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224",config=config, dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224", use_fast=True)

    print(f"SigLip2 loaded on device: {model.device}")

    images = [item[0] for item in dataset]
    embeddings = {}
    for i in range(5):
        inputs = processor(images=images[1000*i:1000*(i+1)], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        for j in range(1000):
            embeddings[1000*i + j] = outputs.pooler_output[j]

    return embeddings

def get_training_prototypes() -> dict[int, int]:
    embeddings = get_image_embeddings()
    print("Got embeddings...")
    prototypes : dict[int, int] = {}

    # for each image...
    for img_num, embedding in embeddings.items():
        if img_num % 100 == 0:
            print(f"Processing image {img_num}...")
        # ...if it doesn't have a prototype yet...
        if img_num not in prototypes.keys():
            # ...find the most similar image among the rest
            best_fit = 0
            for other_img_num, other_embedding in embeddings.items():
                if other_img_num != img_num and other_img_num not in prototypes.keys():
                    similarity = torch.cosine_similarity(embedding, other_embedding, dim=0)
                    if similarity > best_fit:
                        best_fit = similarity
                        prototypes[img_num] = other_img_num
                        prototypes[other_img_num] = img_num
                        print("Done did something")
    print("Assigned prototypes!")
    return prototypes

def get_image_prototype(image : Image.Image) -> dict[int, int]:
    pass





def main():
    prototypes = get_training_prototypes()
    print(prototypes)


if __name__ == '__main__':
    main()