import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_DIRECTML_DISABLE'] = '1'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Path to the pretrained model
MODEL_PATH = r"C:\Workspace\CNN\faster_rcnn_resnet50_v1_640x640_coco17_tpu-8\saved_model"

# COCO classes we care about (COCO uses 1-based indexing)
CLASSES = {
    3: "car",
    5: "airplane",
    9: "boat"
}
print("Loading model...")
model = tf.saved_model.load(MODEL_PATH)
print("Model loaded.")

def detect(image_path):
    image = Image.open(image_path).convert("RGB") # Open the image and convert to 3 RGB channels (in case it's grayscale or has an alpha channel)
    image_np = np.array(image) # Convert the image to a NumPy array for processing. The model expects input in this format, and it allows us to manipulate the image data easily.
    input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...] # Add a batch dimension to the image tensor. The model expects input in batches, so we add an extra dimension at the beginning of the tensor to indicate that we have one image in the batch.
    
    detections = model(input_tensor) # Run the model on the input tensor to get the detections. The model processes the image and outputs a dictionary containing the detected bounding boxes, class labels, and confidence scores for each detection.
    
    boxes   = detections["detection_boxes"][0].numpy() # Extract the bounding boxes from the detections. The model outputs normalized coordinates (values between 0 and 1) for the bounding boxes, which we will later convert to pixel coordinates for visualization.
    classes = detections["detection_classes"][0].numpy().astype(int) # Extract the class labels from the detections. The model outputs class indices, which we convert to integers.
    scores  = detections["detection_scores"][0].numpy() # Extract the confidence scores from the detections. The model outputs a score for each detection, indicating how confident it is in the detection.
    
    return image_np, boxes, classes, scores

def draw_boxes(image_np, boxes, classes, scores, threshold=0.5): # Draw bounding boxes on the image for detections that exceed the confidence threshold. We loop through each detection and check if the confidence score is above the specified threshold and if the detected class is one of the classes we care about (car, airplane, boat). If both conditions are met, we draw a bounding box around the detected object and label it with the class name and confidence score.
    fig, ax = plt.subplots(1, figsize=(12, 8)) # Create a matplotlib figure and axis for displaying the image and drawing the bounding boxes. We set the figure size to make it large enough to clearly see the detections.
    ax.imshow(image_np) # Display the original image on the axis. This is the image on which we will draw the bounding boxes for the detected objects.
    
    h, w = image_np.shape[:2] # Get the height and width of the image :2 only will show h,w not rgb channels. We will use these dimensions to convert the normalized bounding box coordinates (which are between 0 and 1) to pixel coordinates for drawing the boxes on the image.
    
    for box, cls, score in zip(boxes, classes, scores):
        if score < threshold or cls not in CLASSES:
            continue
        ymin, xmin, ymax, xmax = box # The model outputs bounding boxes in the format [ymin, xmin, ymax, xmax], where the coordinates are normalized (between 0 and 1). We unpack these values into separate variables for easier processing.
        x      = xmin * w # Convert the normalized xmin coordinate to pixel coordinates by multiplying it by the width of the image. This gives us the x-coordinate of the top-left corner of the bounding box in pixels.
        y      = ymin * h # Convert the normalized ymin coordinate to pixel coordinates by multiplying it by the height of the image. This gives us the y-coordinate of the top-left corner of the bounding box in pixels.
        width  = (xmax - xmin) * w # Calculate the width of the bounding box in pixels by taking the difference between xmax and xmin (which gives us the width in normalized coordinates) and multiplying it by the width of the image.
        height = (ymax - ymin) * h # Calculate the height of the bounding box in pixels by taking the difference between ymax and ymin (which gives us the height in normalized coordinates) and multiplying it by the height of the image.
        
        rect = patches.Rectangle( # Create a rectangle patch for the bounding box using the calculated pixel coordinates and dimensions. We specify the top-left corner (x, y), the width and height of the rectangle, and set the edge color to red and the face color to none (transparent) so that only the outline of the box is visible.
            (x, y), width, height, # The rectangle is defined by its top-left corner (x, y) and its width and height. This will allow us to draw a box around the detected object in the image.
            linewidth=2, edgecolor="red", facecolor="none" # We set the line width of the rectangle to 2 pixels, the edge color to red, and the face color to none (transparent) so that only the outline of the box is visible. This makes it easier to see the detected objects without obscuring them with a filled box.
        )
        ax.add_patch(rect) # Add the rectangle patch to the axis so that it will be drawn on the image. This will display the bounding box around the detected object when we show the plot.
        ax.text(x, y - 5, f"{CLASSES[cls]} {score:.0%}", # Add a text label above the bounding box to indicate the class name and confidence score. We position the text slightly above the top-left corner of the bounding box (x, y - 5) and format the label to show the class name and the confidence score as a percentage with no decimal places.
                color="red", fontsize=12, fontweight="bold") # We set the color of the text to red, the font size to 12, and the font weight to bold to make it easily readable against the image background.
    
    plt.axis("off") # Hide the axis for a cleaner look when displaying the image with detections. This removes the x and y axis ticks and labels, allowing us to focus on the image and the detected objects without any distractions from the axes.
    plt.show()

image_np, boxes, classes, scores = detect(r"C:\Workspace\CNN\test.jpg")
draw_boxes(image_np, boxes, classes, scores)
