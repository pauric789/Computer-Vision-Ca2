import os
import random

import cv2
from ultralytics import YOLO

# FIX: Import the actual class name
from tracker import ObjectTracker 

# --- Configuration ---
# Set the path to your input video
video_path = "vid1.mp4" 

# --- Initialization ---
# 1. Video Capture
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# 2. YOLO Model Initialization
model = YOLO("yolov8n.pt") # Ensure this model file is present

# 3. Tracker Initialization
tracker = ObjectTracker()

# 4. Color Palette for Tracking IDs
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5

# --- Main Processing Loop ---
while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break # Exit loop if no frame is read (end of video)
    
    # 1. Run Detection
    results = model(frame, verbose=False) # verbose=False suppresses logging output

    # 2. Prepare Detections for Tracker (CORRECTED FORMAT)
    detections = []
    
    # Loop through all detected objects in the current frame
    for result in results:
        # result.boxes.data.tolist() gives: x1, y1, x2, y2, score, class_id
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            
            # Filter detections by the confidence score
            if score > detection_threshold:
                
                # 1. Calculate TLWH bounding box
                # TLWH: [x_top_left, y_top_left, w, h]
                w = x2 - x1
                h = y2 - y1
                bbox_tlwh = [x1, y1, w, h]

                # 2. FIX: DeepSORT-Realtime expects the detection as a tuple:
                # ([x, y, w, h], confidence, class_id)
                # Ensure class_id is an integer or string label.
                detections.append((bbox_tlwh, score, int(class_id)))

    # 3. Update Tracker
    # The update method returns a list of tracked objects (bbox_tlwh, track_id, class_label)
    tracked_objects = tracker.update(frame, detections)

    # 4. Draw Tracking Bounding Boxes
    # The format is (bbox_tlwh, track_id, class_label)
    for bbox_tlwh, track_id, class_label in tracked_objects:
        
        # Get coordinates in TLWH format
        x, y, w, h = bbox_tlwh
        
        # Convert coordinates to integer for drawing
        x1, y1, w, h = map(int, [x, y, w, h])
        
        # Convert TLWH (x, y, w, h) to TLBR (x1, y1, x2, y2) for cv2.rectangle
        x2 = x1 + w
        y2 = y1 + h

        # FIX APPLIED HERE: Convert track_id to int()
        color_index = int(track_id) % len(colors)
        color = colors[color_index]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Add the track ID and label
        text_label = f"ID: {track_id} | {class_label}"
        cv2.putText(frame, text_label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
    # 5. Display the Frame in a Window
    cv2.imshow('YOLO Object Tracking', frame)

    # 6. Handle User Input
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()

print("Video processing finished.")