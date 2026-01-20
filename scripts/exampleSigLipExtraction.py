import torchvision
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

    # for i in range(5000):
    #     inputs = processor(images=images[i], return_tensors="pt").to(model.device)
    #
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #
    #     embeddings[i] = outputs.pooler_output[0]
    return embeddings

def get_training_prototypes() -> dict[int, int]:
    embeddings_dict = get_image_embeddings()
    print("Got embeddings...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Convert dict to a single Tensor [N, 768]
    # We keep track of the keys to map back later
    img_nums = list(embeddings_dict.keys())
    # Ensure embeddings are on the same device and float32 for precision
    features = torch.stack([embeddings_dict[k] for k in img_nums]).to(device)

    # 2. Normalize the embeddings (Crucial for SigLIP)
    # This turns cosine similarity into a simple dot product
    features = torch.nn.functional.normalize(features, p=2, dim=1)

    # 3. Calculate ALL similarities at once [N, N]
    # sim_matrix[i, j] is the similarity between image i and image j
    sim_matrix = torch.mm(features, features.t())

    # 4. Mask the diagonal so an image doesn't pick itself
    sim_matrix.fill_diagonal_(-1)

    sim_matrix.cpu()

    prototypes = {}
    used_indices = set()

    # 5. Greedy matching
    for i in range(len(img_nums)):
        if i in used_indices:
            continue

        # Get similarities for image 'i', ignoring already used images
        row = sim_matrix[i].clone()
        for used_idx in used_indices:
            row[used_idx] = -1

        # Find the best remaining match
        best_match_idx = torch.argmax(row).item()

        # Pair them up
        img_a = img_nums[i]
        img_b = img_nums[best_match_idx]

        prototypes[img_a] = img_b
        prototypes[img_b] = img_a

        used_indices.add(i)
        used_indices.add(best_match_idx)

    return prototypes



def get_image_prototype(image : Image.Image) -> dict[int, int]:
    pass





def main():
    prototypes = get_training_prototypes()
    print(prototypes)
    dataset = torchvision.datasets.STL10(root='../data', split='train', download=True)

    dataset[0][0].show()
    dataset[3282][0].show()

    dataset[1][0].show()
    dataset[4343][0].show()


if __name__ == '__main__':
    main()