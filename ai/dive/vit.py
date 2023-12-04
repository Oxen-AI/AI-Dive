
from transformers import ViTForImageClassification, ViTImageProcessor
import os
from PIL import Image
from ai.dive.model import Model

class Vit(Model):
    def __init__(self):
        super().__init__()

    def build(self):
        print("Loading VIT model...")
        model_name = 'google/vit-base-patch16-224'
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)
    
    # TODO: Put in parent class that runs this function
    # populates the full file path and adds timing, os, etc.
    def process(self, data):
        filename = data['filename']
        image = Image.open(filename)
        
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")

        # Run the model forward
        outputs = self.model(**inputs)

        # Logits are the output values prior to applying an activation function like the softmax
        logits = outputs.logits

        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()

        # print("Predicted class idx:", predicted_class_idx)
        # print("predicted probability:", proba[0][predicted_class_idx])
        # print("Predicted class:", model.config.id2label[predicted_class_idx])

        # Format the output
        input_class = data['class_name']
        predicted_class = self.model.config.id2label[predicted_class_idx]
        predicted_class_options = predicted_class.split(',')
        is_correct = input_class == predicted_class
        for option in predicted_class_options:
            if option.strip().lower() == input_class.lower():
                is_correct = True
                break
        return {
            'input_class': input_class,
            'predicted_class': predicted_class,
            'predicted_class_options': predicted_class_options,
            'is_correct': is_correct
        }