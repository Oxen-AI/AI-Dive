from ai.dive.saver import Saver
from ai.dive.diver import Diver
from ai.dive.models.clip import CLIP
from ai.dive.models.vit import ViT
from ai.dive.data.file_classification import FileClassification
from ai.dive.data.directory_classification import DirectoryClassification

import argparse
import cv2 
import os
from PIL import Image
import pandas as pd

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run model on webcam')
    # parser.add_argument('-d', '--dataset', required=True, type=str, help='Dataset to run model on')
    # parser.add_argument('-l', '--labels', required=True, type=str, help='The dynamic labels file to use')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    # parser.add_argument('-n', '--num-samples', default=-1, type=int, help='Number of samples to run model on')
    parser.add_argument('-m', '--base_model', default="google/vit-base-patch16-224", type=str, help='The base model to use')
    
    args = parser.parse_args()

    model = ViT(model_name=args.base_model)

    # Create output dir if it doesn't exist
    images_path = "images"
    if not os.path.exists(args.output):
        os.makedirs(os.path.join(args.output, images_path))

    output_data = []

    # cap = cv2.VideoCapture('sample_vid.mp4') 
    cap = cv2.VideoCapture(0) 
  
    while(True): 
        
        # Capture frames in the video
        _, frame = cap.read() 
        
        # center crop the frame to 224x224
        crop_size = 224
        height, width, channels = frame.shape
        left = int((width - crop_size) / 2)
        top = int((height - crop_size) / 2)
        right = int((width + crop_size) / 2)
        bottom = int((height + crop_size) / 2)
        frame = frame[top:bottom, left:right]
    
        # flip image
        frame = cv2.flip(frame, 1)
    
        # Convert the video into grayscale PIL image
        image = Image.fromarray(frame)
        image = image.convert('L')

        prediction = model.predict(image)
        print(prediction)
        
        # Save the image to len(output_data).jpg
        relative_path = os.path.join(images_path, f"{len(output_data)}.jpg")
        full_path = os.path.join(args.output, relative_path)
        image.save(full_path)
        # Append to the output data
        prediction["file"] = relative_path
        output_data.append(prediction)

        # describe the type of font 
        # to be used. 
        font = cv2.FONT_HERSHEY_SIMPLEX 
    
        # Use putText() method for 
        # inserting text on video 
        cv2.putText(frame,  
                    prediction["prediction"],
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
    
    # Save the csv to the output dir
    df = pd.DataFrame(output_data)
    df = df[["file", "prediction", "probability", "time"]]
    df.to_csv(os.path.join(args.output, "predictions.csv"), index=False)
    
    # release the cap object 
    cap.release() 
    # close all windows 
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    main()