# Contour-extraction-and-colour-filtering.

## 📌 Overview  
This project processes natural scene images to detect objects, extract their contours, classify them using pre trained models(YOLOv8 in this case), and save annotations in COCO JSON format. It combines HSV-based color segmentation with contour detection for object recognition.  

---

## 🟢 1. Loading Images  
- Images are loaded from the `nature_images/` directory.  
- Supports .jpg, .jpeg, and .png formats.  
- Skips images that fail to load.  

---

## 🟠 2. Convert to HSV & Detect Dominant Colors  
- Converts images from BGR to HSV (Hue, Saturation, Value) color space.  
- Extracts the Hue channel and calculates a histogram of color intensities.  
- Uses peak detection to find dominant colors.  
- Special handling for red color (spans across 0-10 and 170-180 in HSV).  
- Creates binary masks for detected hues.  

---

## 🟡 3. Extract Contours & Apply Edge Detection  
- Converts image to grayscale and applies Gaussian blur to remove noise.  
- Uses Canny edge detection to identify object boundaries.  
- Extracts contours and saves two versions:  
  - Contours drawn on original image (`output_contours_colored/`).  
  - Contours on a black background (`output_contours_black/`).  

---

## 🟣 4. Detect Objects Using Color Masks  
- Applies morphological transformations to clean up detected regions.  
- Extracts bounding boxes around detected objects.  
- Filters out small objects (area threshold = 10,000 pixels).  

---

## 🔵 5. Classify Objects with YOLOv8  
- Loads a YOLOv8 classification model (`yolov8n-cls.pt`).  
- Extracts Regions of Interest (ROI) from detected objects.  
- Runs classification on each ROI and assigns labels.  

---

## 🔴 6. Draw Bounding Boxes & Save Images  
- Draws bounding boxes and labels on detected objects.  
- Saves processed images to `output_images/`.  

---

## 🟠 7. Generate COCO JSON Annotations  
- Stores detected objects in COCO JSON format (`output_annotations/annotations.json`).  

---

## ⚠️ Limitations & Future Improvements  
🔹 YOLOv8 is a classification model, NOT optimized for object detection.  
For better results, use Detectron2 (Mask R-CNN, Faster R-CNN, etc.).  

💡 Why Not Used Here?  
I initially planned to use Detectron2/other LLMs, but I was unable to set up the Conda environment due to some issues. If resolved, switching to Detectron2 or a more advanced model would significantly improve detection/label accuracy.  
