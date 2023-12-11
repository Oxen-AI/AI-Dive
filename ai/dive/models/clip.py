

from transformers import CLIPProcessor, CLIPModel

from PIL import Image
from ai.dive.models.model import Model

class CLIP(Model):
    def __init__(self, labels_file=None):
        self.labels_file = labels_file
        super().__init__()

    # Load model into memory
    def _build(self):
        print("Loading model...")
        model_name = 'openai/clip-vit-large-patch14'
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.prompt = "a photo of a person's face with the emotion of"
        self.class_labels = [
            "__unknown__", # Pick a random word from the vocabulary for unknown
        ]
        if self.labels_file is not None:
            with open(self.labels_file, 'r') as f:
                self.class_labels = f.read().splitlines()

    # Function to run the model on a single example
    def _predict(self, data):
        # Read the image
        filename = data['filepath']
        image = Image.open(filename)

        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        # text = ["a photo of a dog", "a photo of a cat", "a photo of a car" ... etc]
        text = [f"{self.prompt} {label}" for label in self.class_labels]
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Run the model forward
        outputs = self.model(**inputs)

        # Logits are the output values prior to applying an activation function like the softmax
        logits = outputs.logits_per_image

        # Convert logits to probabilities
        proba = logits.softmax(-1)

        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()

        # Check if the user provided a class name
        if 'class_name' not in data:
            # if they did not, just return the predicted class and probability
            return {
                'prediction': self.class_labels[predicted_class_idx],
                'probability': proba[0][predicted_class_idx].item()
            }
        else:
            # if they did, check if the predicted class matches the input class
            input_class = data['class_name']
            predicted_class = self.class_labels[predicted_class_idx]
            predicted_class_options = predicted_class.split(',')
            is_correct = input_class == predicted_class
            for option in predicted_class_options:
                if option.strip().lower() == input_class.lower():
                    is_correct = True
                    break
            return {
                'prediction': predicted_class,
                'probability': proba[0][predicted_class_idx].item(),
                'is_correct': is_correct
            }