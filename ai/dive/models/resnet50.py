

from transformers import AutoImageProcessor, ResNetForImageClassification
from ai.dive.models.image_classification import ImageClassification

class ResNet50(ImageClassification):
    def __init__(self):
        super().__init__()

    # Load model into memory
    def _build(self):
        print("Loading model...")
        model_name = 'microsoft/resnet-50'
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ResNetForImageClassification.from_pretrained(model_name)
