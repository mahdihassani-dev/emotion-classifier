import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model for face detection using DNN
prototxt_path = "pretrainedFaceDetectModel/deploy.prototxt"
caffemodel_path = "pretrainedFaceDetectModel/res10_300x300_ssd_iter_140000_fp16.caffemodel"
face_net = cv2.dnn.readNet(prototxt_path, caffemodel_path)

def getFaces(image):
    # Get the dimensions of the image
    (h, w) = image.shape[:2]

    # Preprocess the image for face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Set the input to the pre-trained face detector
    face_net.setInput(blob)

    # Perform face detection
    detections = face_net.forward()

    # List to store detected faces
    faces = []

    # Iterate over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face ROI and resize it to 224x224
            face = image[startY:endY, startX:endX]
            face_resized = cv2.resize(face, (224, 224))
            faces.append((startX, startY, endX - startX, endY - startY, face_resized))

    return faces

# Load pre-trained model for emotion prediction
emotion_model_path = 'D://RobaticTeamOfYazdUniversity//FaceProcessing//EmotionDetectionCnnModel//ResNet50V2_Model.h5'
emotion_model = load_model(emotion_model_path)

# Define emotions
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Open webcam capture
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Get faces from the current frame
    faces = getFaces(frame)

    # Iterate over detected faces
    for (x, y, w, h, face_image) in faces:
        # Predict emotion for the face
        prediction = emotion_model.predict(np.expand_dims(face_image/255, 0))[0]

        # Get the index of the maximum value in the prediction array
        max_index = np.argmax(prediction)

        # Get the corresponding label from the labels list
        predicted_label = labels[max_index]

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

        # Write the emotion next to the rectangle
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1.4, (200, 40, 10), 2)

    # Display the frame
    cv2.imshow('Video with Emotions', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()
