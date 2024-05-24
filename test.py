import cv2
import numpy as np
from tensorflow.keras.models import load_model
import argparse
import os
from Utils import *
from FaceDetection import *
import time


parser = argparse.ArgumentParser()
parser.add_argument('-m', type=str, required=True, help='the input mode from image, video or webcam')
parser.add_argument('-n', type=str, help='the name of input image or video')
args = parser.parse_args()

# Define emotions
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the trained model
emotion_model_root = 'models'
emotion_model_name = 'ResNet50V2_Model.h5'
emotion_model_path = os.path.join(emotion_model_root, emotion_model_name)
emotion_model = load_model(emotion_model_path)

def imageProcessing(img_path):
    # Load the input image
    input_image = cv2.imread(img_path)
    
    # Get faces from the input image
    faces = getImageFaces(img_path)
    
    # Iterate over detected faces
    for (x, y, w, h, face_image) in faces:
        # Predict emotion for the face
        prediction = emotion_model.predict(face_image)

        # Get the index of the maximum value in the prediction array
        max_index = np.argmax(prediction)

        # Get the corresponding label from the labels list
        predicted_label = labels[max_index]

        # Draw rectangle around the face
        cv2.rectangle(input_image, (x, y), (x+w, y+h), (0, 255, 0), 8)

        # Write the emotion next to the rectangle
        cv2.putText(input_image, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 1.6, (40, 100, 255), 2)
        
        # Define the output directory and base file name
        output_dir = 'outputImage'
        base_filename = args.n  
        
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get the next available file name
        output_image_path = get_output_filename(base_filename, output_dir)
        print(output_image_path)
              
        cv2.imwrite(output_image_path, input_image)
        
    # Resize the image to fit the screen
    resized_image = resize_image_to_fit_screen(input_image, 1920, 1080)

    # Display the input image with rectangles and emotions
    cv2.imshow('Detected Faces with Emotions', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def videoProcessing(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
        
        
    # Get the video frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_dir = 'outputVideo'
    base_filename = args.n
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get the next available file name
    output_video = get_output_filename(base_filename, output_dir)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 20.0, (frame_width, frame_height))
    
    fixed_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = fixed_total_frames
    

    frame_count = 0
    start_time = time.time()
    
    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Get faces from the current frame
        faces = getFrameFaces(frame)

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
            cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 2, (200, 40, 10), 2)
            
        # Increment frame count
        frame_count += 1
        total_frames -= 1
            
        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Calculate real FPS
        real_fps = frame_count / elapsed_time
        
        duration = total_frames / real_fps


        # Write the frame to the output video
        out.write(frame)
        
        resized_frame = resize_image_to_fit_screen(frame, 1920, 1080)
        
        # Display real FPS on the frame
        cv2.putText(resized_frame, f"duration: {int(duration)}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw border for loading bar
        cv2.rectangle(resized_frame, (10, 50), (210, 70), (50, 50, 50), 2)
        
        # Display loading bar
        loading_bar_width = int(((fixed_total_frames - total_frames) / fixed_total_frames) * 200)
        cv2.rectangle(resized_frame, (10, 50), (10 + loading_bar_width, 70), (0, 255, 0), -1)

        # Display the frame
        cv2.imshow('Video with Emotions', resized_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    out.release()
    print(f"Video saved in {output_video}")
    cv2.destroyAllWindows()

def webcamProcessing():
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
        faces = getFrameFaces(frame)

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


if(args.m == 'image'):
    if vars(args)['n'] == None:
        parser.error('image mode need -n for name')
        
    sample_image_root = 'imageSamples'
    img_name = args.n
    img_path = os.path.join(sample_image_root, img_name)
    
    if not os.path.exists(img_path):
        raise ValueError(f'image "{img_name}" not exists in {sample_image_root}')
    
    imageProcessing(img_path)
elif(args.m == 'video'):
    if vars(args)['n'] == None:
        parser.error('video mode need -n for name')
        
    sample_video_root = 'videoSamples'
    video_name = args.n
    video_path = os.path.join(sample_video_root, video_name)
    
    if not os.path.exists(video_path):
        raise ValueError(f'video "{video_name} not exists in {sample_video_root}"')
    
    videoProcessing(video_path)
elif(args.m == 'webcam'):
    if vars(args)['n'] != None:
        parser.error('webcam mode does not need any extra flag')
    webcamProcessing()

    
    






