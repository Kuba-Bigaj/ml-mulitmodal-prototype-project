import os
import torch
from PIL import Image


class ClassEmbeddingGenerator:
    """
    Generates class embeddings by processing images and texts from a given directory structure.
    """
    def __init__(self, model, processor):
        """
        Initializes the ClassEmbeddingGenerator with a model and processor.
        :param model: The pre-trained model used for generating embeddings.
        :param processor: The appropriate preprocessor for the model.
        """
        self.model = model
        self.processor = processor

    def generate_class_embeddings(self, data_path: str) -> list[tuple[str, torch.Tensor]]:
        """
        Generates class embeddings for each class directory in the given data path.
        :param data_path: Path to the root directory containing class subdirectories.
        :return: A list of tuples, each containing the class name and its corresponding embedding tensor.
        """

        embeddings = []
        with os.scandir(data_path) as root_dir:
            for entry in root_dir:
                if entry.is_dir():
                    embeddings.append(self._generate_class_embedding(entry.path))

        return embeddings

    def _generate_class_embedding(self, class_path: str) -> tuple[str, torch.Tensor]:
        images: list[Image.Image] = []
        texts: list[str] = []
        with os.scandir(class_path) as c_dir:
            for entry in c_dir:
                if entry.is_file():
                    if entry.name.endswith(('.png', '.jpg', '.jpeg')):
                        images.append(Image.open(entry.path).convert("RGB"))
                    if entry.name.endswith('.txt'):
                        with open(entry.path) as f:
                            texts.append(f.read())

        print(f"For class {os.path.basename(class_path)} scanned {len(images)} images and {len(texts)} texts.")
        if len(images) == 0 or len(texts) == 0:
            return None, None

        # IMPORTANT: use `padding=max_length` and `max_length=64`
        inputs = self.processor(text=texts, images=images, padding="max_length", max_length=64, return_tensors="pt").to(
            self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # averaging
        stack = torch.cat([outputs.image_embeds, outputs.text_embeds], dim=0)
        mean = torch.mean(stack, dim=0, keepdim=True)
        normalized = torch.nn.functional.normalize(mean, p=2, dim=-1)

        return os.path.basename(class_path), normalized