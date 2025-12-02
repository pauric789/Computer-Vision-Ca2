# import my libraries
import random
import cv2
from ultralytics import YOLO


from tracker import ObjectTracker 


# read in my video
cap = cv2.VideoCapture("vid1.mp4") 


# load in my pre-trained YOLO model
model = YOLO("yolov8n.pt") 

# initialize my object tracker
tracker = ObjectTracker()

# create unique rgb for different track IDs
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# set detection threshold
detection_threshold = 0.5

# loop through video frames
while True:
    # Read the next frame
    ret, frame = cap.read()

    # if the video ends break the loop
    if not ret:
        break 
    
    # read the frame into YOLO model
    results = model(frame) 

    # create a list to hold detections 
    detections = []
    
    # Loop through all detected objects in the frame
    for result in results:
        
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            
            # Filter detections by the confidence score
            if score > detection_threshold:
                
                # Calculate TLWH bounding box
                w = x2 - x1
                h = y2 - y1
                bbox_tlwh = [x1, y1, w, h]

                # Append detection into the list
                detections.append((bbox_tlwh, score, int(class_id)))

    # update the tracker to return the tracked objects
    tracked_objects = tracker.update(frame, detections)

    # get the bounding box and id for each person
    for bbox_tlwh, track_id in tracked_objects:
        
        # Get coordinates in TLWH format
        x, y, w, h = bbox_tlwh
        
        # Convert coordinates to integer for drawing
        x1, y1, w, h = map(int, [x, y, w, h])
        
        # Convert the TLWH  to TLBR for drawing
        x2 = x1 + w
        y2 = y1 + h

        # get the colour based on track id
        color_index = int(track_id) % len(colors)
        color = colors[color_index]
        # draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Add the track ID and label
        text_label = f"ID: {track_id}"
        cv2.putText(frame, text_label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        """""
         # draw counts (current and cumulative unique)
        current_count = len(tracked_objects)
        total_count = len(seen_ids)
        count_text = f"Current: {current_count}  Total: {total_count}"
        # draw outline for visibility
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
        """""
        
    # display the video 
    cv2.imshow("Pauric's security system", frame)

    # break loop when 'ESC' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

# destroy all windows
cv2.destroyAllWindows()

