

from PIL import Image
from ai.dive.models.model import Model

class ImageClassification(Model):
    def __init__(self):
        super().__init__()

    # Function to run the model on a single example
    def _predict(self, data):
        if type(data) == dict:
            if 'filepath' in data:
                return self._predict_file(data)
        elif type(data) == Image.Image:
            return self._predict_image(data, {})
        
        raise Exception("Invalid data format")

    def _predict_file(self, data):
        # Read the image
        filename = data['filepath']
        image = Image.open(filename)
        return self._predict_image(image, data)

    def _predict_image(self, image, data):
        # If grayscale, convert to RGB
        if image.mode == 'L':
            image = image.convert('RGB')

        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")

        # Run the model forward
        outputs = self.model(**inputs)

        # Logits are the output values prior to applying an activation function like the softmax
        logits = outputs.logits

        # Convert logits to probabilities
        proba = logits.softmax(-1)

        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()

        # Check if the user provided a class name
        if 'class_name' not in data:
            # if they did not, just return the predicted class and probability
            return {
                'prediction': self.model.config.id2label[predicted_class_idx],
                'probability': proba[0][predicted_class_idx].item()
            }
        else:
            # if they did, check if the predicted class matches the input class
            input_class = data['class_name']
            predicted_class = self.model.config.id2label[predicted_class_idx]
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
