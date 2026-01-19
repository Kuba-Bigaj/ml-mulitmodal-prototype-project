import torch, PIL

from scripts.old.class_embedding_generator import ClassEmbeddingGenerator


class SampleClassifier:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        generator = ClassEmbeddingGenerator(model=model, processor=processor)
        embs = generator.generate_class_embeddings("./class_data")
        self.class_embeddings = list(filter(lambda x: x[1] is not None, embs))

    def classify(self, sample : PIL.Image.Image) -> list[tuple[str, float]]:
        embedding = self._generate_embedding(sample)
        results = []
        for c_name, c_emb in self.class_embeddings:
            similarity = torch.nn.functional.cosine_similarity(embedding, c_emb)
            results.append((c_name, similarity.item()))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _generate_embedding(self, sample: PIL.Image.Image) -> torch.Tensor:
        inputs = self.processor(text = [""],images=[sample], padding="max_length", max_length=64, return_tensors="pt").to(
            self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        print(outputs.image_embeds.size())
        return outputs.image_embeds