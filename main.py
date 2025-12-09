# import my libraries
import random
import cv2
import time 
from ultralytics import YOLO


from tracker import ObjectTracker 


# --- Group Entry Constants ---
GROUP_ENTRY_THRESHOLD_COUNT = 4  # Number of new IDs to trigger a flag
GROUP_ENTRY_TIME_WINDOW = 2.0    # Time window in seconds
# -----------------------------

# --- Loitering Constants ---
LOITERING_TIME_THRESHOLD = 8.0 # Time in seconds (8 seconds)
LOITERING_DISTANCE_THRESHOLD = 20.0 # Max pixel distance change to be considered stationary
# ---------------------------

# --- YOLO CONSTANT ---
PERSON_CLASS_ID = 0 # Class ID for "person" in the COCO dataset
# ---------------------

# --- Startup Constants ---
STARTUP_FRAME_SKIP = 30 # Skip group detection for the first 30 frames
# -------------------------


# read in my video
cap = cv2.VideoCapture("group.mp4") 


# load in my pre-trained YOLO model
model = YOLO("yolov8n.pt") 

# initialize my object tracker
tracker = ObjectTracker()

# create unique rgb for different track IDs
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# set detection threshold
detection_threshold = 0.5

# --- Tracking Variables ---
seen_ids = set()
id_entry_time = {} 
recent_new_ids = [] 
group_entry_flagged = False
flag_display_start_time = 0
frame_count = 0 
# Stores data for loitering detection: {track_id: [last_x, last_y, stationary_start_time]}
loitering_data = {} 
# Stores IDs of objects currently flagged for loitering
loitering_flagged = set()
# --------------------------

# Helper function to get the IDs that are new in the current frame
def get_newly_tracked_ids(tracked_objects, seen_ids):
    current_ids = {track_id for _, track_id in tracked_objects}
    new_ids = current_ids - seen_ids
    return new_ids

# loop through video frames
while True:
    # Read the next frame
    ret, frame = cap.read()
    current_time = time.time()

    # if the video ends break the loop
    if not ret:
        break 
    
    frame_count += 1
    
    # read the frame into YOLO model
    results = model(frame) 

    # create a list to hold detections 
    detections = []
    
    # Loop through all detected objects in the frame
    for result in results:
        
        for r in result.boxes.data.tolist():
            # r: [x1, y1, x2, y2, score, class_id]
            x1, y1, x2, y2, score, class_id = r
            
            # FILTER: Only track if it's a PERSON AND score > threshold
            if int(class_id) == PERSON_CLASS_ID and score > detection_threshold:
                
                # Calculate TLWH bounding box
                w = x2 - x1
                h = y2 - y1
                bbox_tlwh = [x1, y1, w, h]

                # Append detection into the list
                detections.append((bbox_tlwh, score, int(class_id)))

    # update the tracker to return the tracked objects
    tracked_objects = tracker.update(frame, detections)
    
    
    # --- Group Entry Detection Logic ---
    # (Existing logic from previous modification)
    
    # ONLY check for group entry after the initial stabilization period
    if frame_count > STARTUP_FRAME_SKIP:
        
        new_ids = get_newly_tracked_ids(tracked_objects, seen_ids)
        
        for track_id in new_ids:
            seen_ids.add(track_id)
            id_entry_time[track_id] = current_time 
        
        recent_new_ids = [
            id for id in recent_new_ids 
            if current_time - id_entry_time.get(id, current_time) <= GROUP_ENTRY_TIME_WINDOW
        ]
        
        for track_id in new_ids:
            if track_id not in recent_new_ids:
                recent_new_ids.append(track_id)
                
        if len(recent_new_ids) >= GROUP_ENTRY_THRESHOLD_COUNT and not group_entry_flagged:
            group_entry_flagged = True
            flag_display_start_time = current_time
            print(f"!!! ALERT: Group Entry Detected - {len(recent_new_ids)} new IDs in {GROUP_ENTRY_TIME_WINDOW}s !!!")

    # Manage group flag display timeout 
    if group_entry_flagged and (current_time - flag_display_start_time > 5.0): 
        group_entry_flagged = False
        recent_new_ids = [] 
    
    # -----------------------------------
    
    # --- Loitering Detection Logic ---
    current_tracked_ids = {track_id for _, track_id in tracked_objects}
    
    # Prune loitering_data for IDs that have left the frame
    ids_to_remove = [id for id in loitering_data if id not in current_tracked_ids]
    for id in ids_to_remove:
        loitering_data.pop(id, None)
        loitering_flagged.discard(id)

    for bbox_tlwh, track_id in tracked_objects:
        x, y, w, h = bbox_tlwh
        
        # Calculate Centroid (center point)
        center_x = x + w / 2
        center_y = y + h / 2
        current_centroid = (center_x, center_y)

        if track_id not in loitering_data:
            # First time seeing this ID or resetting the loitering tracker
            loitering_data[track_id] = [center_x, center_y, current_time]
            loitering_flagged.discard(track_id)

        else:
            last_x, last_y, stationary_start_time = loitering_data[track_id]
            
            # Calculate distance moved since the last recorded stationary point
            distance = ((center_x - last_x)**2 + (center_y - last_y)**2)**0.5

            if distance < LOITERING_DISTANCE_THRESHOLD:
                # The object is stationary 
                
                # Check if the stationary time exceeds the threshold
                if (current_time - stationary_start_time) >= LOITERING_TIME_THRESHOLD:
                    if track_id not in loitering_flagged:
                        loitering_flagged.add(track_id)
                        print(f"!!! ALERT: Loitering detected for ID {track_id} !!!")
                # If distance is small, stationary_start_time remains the original start time
                
            else:
                # The object has moved significantly, reset the stationary tracking
                loitering_data[track_id] = [center_x, center_y, current_time]
                loitering_flagged.discard(track_id)
    # ---------------------------------
    
    # get the bounding box and id for each person
    for bbox_tlwh, track_id in tracked_objects:
        
        # Ensure we add ALL IDs to seen_ids even during the skip period for the Total Count
        if track_id not in seen_ids:
             seen_ids.add(track_id)
        
        # Get coordinates in TLWH format
        x, y, w, h = bbox_tlwh
        x1, y1, w, h = map(int, [x, y, w, h])
        x2 = x1 + w
        y2 = y1 + h

        # get the colour based on track id
        color_index = int(track_id) % len(colors)
        color = colors[color_index]
        
        # --- Drawing Logic: Loitering has highest priority for visual warning ---
        if track_id in loitering_flagged:
            color = (255, 255, 0) # Cyan/Yellow for Loitering
            text_label = f"ID: {track_id} (LOITERING!)"
        elif track_id in recent_new_ids:
            color = (0, 0, 255) # Red for Group Entry
            text_label = f"ID: {track_id}"
        else:
            text_label = f"ID: {track_id}"
        # ----------------------------------------------------------------------

        # draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Add the track ID and label
        cv2.putText(frame, text_label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
    # Draw Counts 
    current_count = len(tracked_objects)
    total_count = len(seen_ids)
    count_text = f"Current: {current_count} Total Unique: {total_count}"
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw the Group Entry Flag
    if group_entry_flagged:
        flag_text = "!!! GROUP ENTRY FLAGGED !!!"
        # Draw a thick red warning box/text at the top
        cv2.rectangle(frame, (0, 50), (frame.shape[1], 100), (0, 0, 255), -1)
        cv2.putText(frame, flag_text, (50, 85), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
    # display the video 
    cv2.imshow("Pauric's security system", frame)

    # break loop when 'ESC' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

# destroy all windows
cap.release()
cv2.destroyAllWindows()