

from transformers import AutoImageProcessor, ResNetForImageClassification
from ai.dive.models.image_classification import ImageClassification

class ResNet50(ImageClassification):
    def __init__(self, model_name='microsoft/resnet-50'):
        self.model_name = model_name
        super().__init__()

    # Load model into memory
    def _build(self):
        print(f"Loading model... {self.model_name}")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = ResNetForImageClassification.from_pretrained(self.model_name)
