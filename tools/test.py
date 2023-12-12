from transformers import ViTForImageClassification, ViTImageProcessor
import cv2 
from PIL import Image

# Define the model_name you want to grab from hugging face
# https://huggingface.co/google/vit-base-patch16-224
model_name = 'google/vit-base-patch16-224'

# Load the image processor
processor = ViTImageProcessor.from_pretrained(model_name)

# Load the model
model = ViTForImageClassification.from_pretrained(model_name)

# Instantiate the video capture from the webcam
cap = cv2.VideoCapture(0) 

while(True): 

    # Capture frames in the video
    _, frame = cap.read() 

    # Convert from cv2.Mat to PIL.Image
    image = Image.fromarray(frame)  

    # Convert the PIL.Image into a pytorch tensor
    inputs = processor(images=image, return_tensors="pt")

    # Run the model on the image
    outputs = model(**inputs)

    # Get the logits (proxy for probability)
    logits = outputs.logits

    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()

    # Print the predicted class
    prediction = model.config.id2label[predicted_class_idx]
    print("Predicted class:", prediction)

    # describe the type of font 
    # to be used. 
    font = cv2.FONT_HERSHEY_SIMPLEX 

    # Use putText() method for 
    # inserting text on video 
    cv2.putText(frame,  
                prediction,
                (50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 

    # Display the resulting frame 
    cv2.imshow('video', frame) 

    # creating 'q' as the quit  
    # button for the video 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# release the cap object 
cap.release() 
# close all windows 
cv2.destroyAllWindows()