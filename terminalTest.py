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
            face = cv2.resize(face, (224, 224))

            # Normalize pixel values to [0, 1]
            face = face / 255.0

            # Expand dimensions to match the shape required by the model
            face = np.expand_dims(face, axis=0)

            # Append the face to the list of faces
            faces.append(face)

    return faces

# load our model
emotion_model_path = 'D://RobaticTeamOfYazdUniversity//FaceProcessing//EmotionDetectionCnnModel//ResNet50V2_Model.h5'
emotion_model = load_model(emotion_model_path)

# Define emotions
labels = ['Angry', 
          'Disgust', 
          'Fear', 
          'Happy', 
          'Neutral', 
          'Sad', 
          'Surprise']

# image input of CNN model
input_image = 'imageSamples/differentEmotions.jpg'
counter = 1


for face in getFaces(input_image):
    # prediction in array form
    prediction = emotion_model.predict(face)

    # Get the index of the maximum value in the prediction array
    max_index = np.argmax(prediction)

    # Get the corresponding label from the labels list
    predicted_label = labels[max_index]

    # Print the predicted label
    print(f"Predicted emotion {counter} :", predicted_label)
    counter += 1
