import torch
import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from ultralytics import YOLO  # Ensure YOLO library is installed

# Load the pre-trained YOLO model (classification)
yolo_model = YOLO("yolov8n-cls.pt")  # Do NOT change this to a detection model

def classify_objects(image, bounding_boxes):
    """
    Uses YOLO to classify objects inside given bounding boxes.
    Returns a list of labels corresponding to each bounding box.
    """
    labels = []
    for (x, y, w, h) in bounding_boxes:
        roi = image[y:y+h, x:x+w]  # Crop the region of interest
        if roi.size == 0:
            labels.append("unknown")  # Skip empty ROIs
            continue

        # Run YOLO classification
        results = yolo_model(roi)
        if results:
            detected_label = results[0].names[results[0].probs.top1]
        else:
            detected_label = "unknown"

        labels.append(detected_label)
    return labels

def process_images(image_folder, output_dir, min_area=10000, hue_range=10, peak_threshold_ratio=0.01):
    annotations = []
    
    output_contours_colored = "output_contours_colored"
    output_contours_black = "output_contours_black"
    output_annotations = "output_annotations"
    
    os.makedirs(output_contours_colored, exist_ok=True)
    os.makedirs(output_contours_black, exist_ok=True)
    os.makedirs(output_annotations, exist_ok=True)
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            print(f"Processing {image_path}...")
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading {image_path}")
                continue
            
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue_channel = hsv[:, :, 0]
            hist = cv2.calcHist([hue_channel], [0], None, [180], [0, 180]).flatten()
            
            peak_threshold = peak_threshold_ratio * np.max(hist)
            peaks, _ = find_peaks(hist, height=peak_threshold)
            
            masks = []
            red_lower_peaks = peaks[peaks < hue_range]
            red_upper_peaks = peaks[peaks > (179 - hue_range)]
            if red_lower_peaks.size > 0 or red_upper_peaks.size > 0:
                mask_lower = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([hue_range, 255, 255]))
                mask_upper = cv2.inRange(hsv, np.array([179 - hue_range, 50, 50]), np.array([179, 255, 255]))
                mask_red = cv2.bitwise_or(mask_lower, mask_upper)
                masks.append(('Red', mask_red))
            
            for peak in peaks:
                if peak < hue_range or peak > (179 - hue_range):
                    continue
                lower_bound = np.array([max(0, peak - hue_range), 50, 50])
                upper_bound = np.array([min(179, peak + hue_range), 255, 255])
                mask = cv2.inRange(hsv, lower_bound, upper_bound)
                masks.append((f'Hue {peak}', mask))
            
            all_bounding_boxes = []
            
            # Create Contour Images (NO bounding boxes at this stage)
            gry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blr = cv2.GaussianBlur(gry, (9, 9), 0)
            cny = cv2.Canny(blr, 50, 200)
            contours, _ = cv2.findContours(cny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
            image_with_contours = image.copy()
            cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), thickness=1)
            
            black_background = np.zeros_like(image)
            cv2.drawContours(black_background, contours, -1, (255, 255, 255), thickness=1)
            
            output_contours_colored_path = os.path.join(output_contours_colored, filename)
            cv2.imwrite(output_contours_colored_path, image_with_contours)  # NO bounding boxes
            
            output_contours_black_path = os.path.join(output_contours_black, filename)
            cv2.imwrite(output_contours_black_path, black_background)  # NO bounding boxes
            
            # NOW create bounding boxes
            for label, mask in masks:
                kernel = np.ones((5, 5), np.uint8)
                mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w * h > min_area:
                        all_bounding_boxes.append((x, y, w, h))

            # Classify detected objects using YOLO
            object_labels = classify_objects(image, all_bounding_boxes)

            # Draw bounding boxes and labels
            for (x, y, w, h), label in zip(all_bounding_boxes, object_labels):
                color = np.random.randint(0, 255, 3).tolist()
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save annotation in COCO format
            annotation = {
                "image": image_path,
                "annotations": [{"bbox": [x, y, w, h], "label": label} 
                                for (x, y, w, h), label in zip(all_bounding_boxes, object_labels)]
            }
            annotations.append(annotation)

            output_image_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_image_path, image)  # Output with bounding boxes and labels
    
    with open(os.path.join(output_annotations, "annotations.json"), "w") as f:
        json.dump(annotations, f, indent=4)
    
    print("Processing completed. Annotations with labels saved.")

# Define input and output paths
image_folder = "nature_images"  # Change this to your actual image folder path
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Run processing
process_images(image_folder, output_dir)
