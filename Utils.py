import os
import cv2

def get_output_filename(input_filename, output_directory):
    base_name = os.path.basename(input_filename)
    output_name = "output_" + base_name
    return os.path.join(output_directory, output_name)

def resize_image_to_fit_screen(image, screen_width, screen_height):
    img_height, img_width = image.shape[:2]
    aspect_ratio = img_width / img_height

    if img_width > screen_width or img_height > screen_height:
        if img_width > screen_width:
            new_width = screen_width
            new_height = int(screen_width / aspect_ratio)
        if new_height > screen_height:
            new_height = screen_height
            new_width = int(screen_height * aspect_ratio)
    else:
        new_width, new_height = img_width, img_height
    
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image
