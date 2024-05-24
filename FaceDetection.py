import cv2
import numpy as np

## Load pre-trained model for face detection using DNN
prototxt_path = 'pretrainedFaceDetectModel/deploy.prototxt'
caffemodel_path = 'pretrainedFaceDetectModel/res10_300x300_ssd_iter_140000_fp16.caffemodel'
face_net = cv2.dnn.readNet(prototxt_path, caffemodel_path)

def getImageFaces(input):
    # Load the input image
    image = cv2.imread(input)

    # Get dimensions of the image
    (h, w) = image.shape[:2]

    # Pre-process the image for face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Set the input to the network
    face_net.setInput(blob)

    # Perform face detection
    detections = face_net.forward()

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

def getFrameFaces(input):
    # Get the dimensions of the image
    (h, w) = input.shape[:2]

    # Preprocess the image for face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(input, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

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
            face = input[startY:endY, startX:endX]
            face_resized = cv2.resize(face, (224, 224))
            faces.append((startX, startY, endX - startX, endY - startY, face_resized))

    return faces
