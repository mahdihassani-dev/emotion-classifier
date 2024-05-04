import cv2
import numpy as np
from tensorflow.keras.models import load_model

def getFaces(input):
    # Load the input image
    image = cv2.imread(input)

    # Load pre-trained Caffe model for face detection
    prototxt_path = "pretrainedFaceDetectModel/deploy.prototxt"
    model_path = "pretrainedFaceDetectModel/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Get dimensions of the image
    (h, w) = image.shape[:2]

    # Pre-process the image for face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Set the input to the network
    net.setInput(blob)

    # Perform face detection
    detections = net.forward()

    faces = []

    # Iterate over the detected faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # Compute the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face ROI
            face = image[startY:endY, startX:endX]

            # Resize the face to 224x224 (to match the input size of the emotion model)
            resized_face = cv2.resize(face, (224, 224))
            resized_face = np.expand_dims(resized_face/255, 0)

            # Append the coordinates and resized face to the list of faces
            faces.append((startX, startY, endX - startX, endY - startY, resized_face))

    return faces

# Load the trained model
emotion_model_path = 'D://RobaticTeamOfYazdUniversity//FaceProcessing//EmotionDetectionCnnModel//ResNet50V2_Model.h5'
emotion_model = load_model(emotion_model_path)

# Define emotions
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # image input of CNN model
inputNN = 'imageSamples/differentEmotions.jpg'

# Load the input image
input_image = cv2.imread(inputNN)

# Get faces from the input image
faces = getFaces(inputNN)

# Iterate over detected faces
for (x, y, w, h, face_image) in faces:
    # Predict emotion for the face
    prediction = emotion_model.predict(face_image)

    # Get the index of the maximum value in the prediction array
    max_index = np.argmax(prediction)

    # Get the corresponding label from the labels list
    predicted_label = labels[max_index]

    # Draw rectangle around the face
    cv2.rectangle(input_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Write the emotion next to the rectangle
    cv2.putText(input_image, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 40, 10), 2)

# Display the input image with rectangles and emotions
cv2.imshow('Detected Faces with Emotions', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
