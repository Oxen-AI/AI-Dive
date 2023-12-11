
from transformers import ViTForImageClassification, ViTImageProcessor
from ai.dive.models.image_classification import ImageClassification

class ViT(ImageClassification):
    def __init__(self, model_name='google/vit-base-patch16-224'):
        self.model_name = model_name
        super().__init__()

    # Load model into memory
    def _build(self):
        print("Loading VIT model...")
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.model = ViTForImageClassification.from_pretrained(self.model_name)
