from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from mtcnn import MTCNN

def getFaces(input):
    # Load the input image
    image = cv2.imread(input)

    # Initialize the MTCNN face detector
    detector = MTCNN()

    # Detect faces in the image
    faces = detector.detect_faces(image)
    
    images = []

    # Iterate over detected faces
    for face in faces:
        # Get the bounding box coordinates of the detected face
        x, y, w, h = face['box']

        # Crop the region around the face
        cropped_face = image[max(y, 0):y+h, max(x, 0):x+w]

        # Resize the cropped face while preserving aspect ratio
        resized_face = cv2.resize(cropped_face, (224, 224))
        resized_face = np.expand_dims(resized_face/255, 0)
        images.append(resized_face)
        
    return images

model = load_model('ResNet50V2_Model.h5')


labels = ['Angry', 
                  'Disgust', 
                  'Fear', 
                  'Happy', 
                  'Neutral', 
                  'Sad', 
                  'Surprise']

prediction = model.predict(getFaces('bijan3.jpeg')[0])

# Get the index of the maximum value in the prediction array
max_index = np.argmax(prediction)

# Get the corresponding label from the labels list
predicted_label = labels[max_index]

# Print the predicted label
print("Predicted emotion:", predicted_label)
