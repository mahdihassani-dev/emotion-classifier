from tensorflow.keras.models import load_model
import cv2
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
        images.append((x, y, w, h, resized_face))
        
    return images

# Load the trained model
model = load_model('ResNet50V2_Model.h5')

# Define emotions
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

inputNN = 'yosef3.jpg'

# Load the input image
input_image = cv2.imread(inputNN)

# Get faces from the input image
faces = getFaces(inputNN)

# Iterate over detected faces
for (x, y, w, h, face_image) in faces:
    # Predict emotion for the face
    prediction = model.predict(face_image)[0]

    # Get the index of the maximum value in the prediction array
    max_index = np.argmax(prediction)

    # Get the corresponding label from the labels list
    predicted_label = labels[max_index]

    # Draw rectangle around the face
    cv2.rectangle(input_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Write the emotion next to the rectangle
    cv2.putText(input_image, predicted_label, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 40, 10), 2)

# Display the input image with rectangles and emotions
cv2.imshow('Detected Faces with Emotions', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
