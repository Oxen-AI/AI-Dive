
from transformers import ViTForImageClassification, ViTImageProcessor
import os
from PIL import Image
from ai.dive.models.image_classification import ImageClassification

class ViT(ImageClassification):
    def __init__(self):
        super().__init__()

    # Load model into memory
    def _build(self):
        print("Loading VIT model...")
        model_name = 'google/vit-base-patch16-224'
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)
