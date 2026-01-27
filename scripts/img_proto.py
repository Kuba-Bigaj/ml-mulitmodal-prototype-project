import pickle
from pathlib import Path

import torchvision
import PIL.Image as Image
import torch
from networkx.algorithms.clique import max_weight_clique
from transformers import AutoModel, AutoProcessor, SiglipVisionModel, SiglipVisionConfig


def get_image_embeddings() -> dict[int, torch.Tensor]:
    dataset = torchvision.datasets.STL10(root='./data', split='train', download=True)

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
    print("Generated embeddings!")
    return embeddings

def get_prototypes() -> dict[int, int]:
    embeddings_dict = get_image_embeddings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_nums = list(embeddings_dict.keys())
    features = torch.stack([embeddings_dict[k] for k in img_nums]).to(device) # stack embeddings into a single tensor, push to GPU
    features = torch.nn.functional.normalize(features, p=2, dim=1) # L2 normalization

    sim_matrix = torch.mm(features, features.t()) # multiplication of normalized features gives cosine similarity

    sim_matrix.fill_diagonal_(-1) # prevent self-matching
    sim_matrix.cpu() # fetch to CPU

    prototypes = {}
    used_indices = set()

    for i in range(len(img_nums)): # for each image ...
        if i in used_indices: # ... if it was already used in a pair, skip it, otherwise ...
            continue

        # ... get similarities for it, ignoring already used images ...
        row = sim_matrix[i].clone()
        for used_idx in used_indices:
            row[used_idx] = -1

        # ... find the best remaining match ...
        best_match_idx = torch.argmax(row).item()

        # ... and record the prototype pair
        img_a = img_nums[i]
        img_b = img_nums[best_match_idx]

        prototypes[img_a] = img_b
        prototypes[img_b] = img_a

        used_indices.add(i)
        used_indices.add(best_match_idx)

    return prototypes

def get_stl10_class_weights_learn_difficult() -> torch.Tensor:
    difficult_cases = get_difficult_cases()

    weights = list(map(lambda x: x*(len(difficult_cases)/sum(difficult_cases)), difficult_cases))
    weights = torch.tensor(weights, dtype=torch.float32)

    return weights

def get_stl10_class_weights_learn_easy_inversion() -> torch.Tensor:
    weights = 1 / get_stl10_class_weights_learn_difficult()
    weights /= weights.mean()
    return weights

def get_stl10_class_weights_learn_easy_linear() -> torch.Tensor:
    wgts = get_stl10_class_weights_learn_difficult();
    max_wgt = torch.max(wgts)
    min_wgt = torch.min(wgts)

    easy = max_wgt - wgts + min_wgt
    easy /= easy.mean()

    return easy

def get_difficult_cases() -> list[int]:

    cache_path = Path("data") / "difficult.cache"
    if cache_path.exists():
        print("Found a cached difficult class list!")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print("Cache not found, generating difficult class list...")
    dataset = torchvision.datasets.STL10(root='./data', split='train', download=True)
    prototypes = get_prototypes()

    difficult_cases = [0] * 10

    for key in prototypes:
        key_label = dataset[key][1]
        prototype_label = dataset[prototypes[key]][1]
        if key_label != prototype_label:
            difficult_cases[key_label] += 1

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(difficult_cases, f)

    print("Generated the difficult class list and cached it!")
    return difficult_cases


def main():
    print(get_stl10_class_weights_learn_difficult())
    print(get_stl10_class_weights_learn_easy_linear())
    print(get_stl10_class_weights_learn_easy_inversion())

if __name__ == '__main__':
    main()