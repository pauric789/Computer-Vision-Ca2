#Import my libraries
import random
import cv2
import time
import math
from collections import deque
from ultralytics import YOLO
# Import my DEEPSORT tracker
from tracker import ObjectTracker  

# My threshold for securtiy flags

# Group entry 
# 4 people entering within 2 seconds
group_entry = 4
time_window = 2.0
# Skip initial frames so it not flagged at the start of the video 
skip_start_frames = 30  


# Loitering
# Person staying within 20 pixels for over 8 seconds
loitering_time = 8.0
loitering_distance = 20.0

#Direction chnages
# Person changing direction by over 90 degrees within 15 seconds
history_length = 15
angle_change = 90
min_distance = 15.0
flag_duration = 3.0

# Show only people in the COCO dataset
person_id_class = 0


# read in my video
cap = cv2.VideoCapture("loitering.mp4")

# load in my pre-trained YOLO model
model = YOLO("yolov8n.pt")

# initialize my object tracker from DeepSORT file
tracker = ObjectTracker()

# create unique rgb for different track IDs 100 different colors
colours = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(100)]

# set detection threshold
detection_threshold = 0.5

# My security flag parameters
# seen id set
seen_ids = set()
# Dictionary to hold entry times for IDs
id_entry_time = {}
# List to hold recent ID for group entry
recent_new_ids = []
# Set the group entry to false 
group_entry_flagged = False
# Set the start to 0
flag_display_start_time = 0
# Set the frame count to 0
frame_count = 0
# Loitering Dictionary for tracking positions and times
loitering_data = {}     
# Set for flagged loitering IDs
loitering_flagged = set()
# Dictionary for path history
path_history = {}      
# Dictionary for direction change flags  
direction_flagged = {}  

# A Helper function to get newly tracked IDs
# It feeds in and compares current tracked IDs with seen_ids set
def get_newly_tracked_ids(tracked_objects, seen_ids):
    # Dictionary of currently tracked IDs
    current_ids = {track_id for _, track_id in tracked_objects}
    # New Ids are in current Ids but not in seen_ids
    new_ids = current_ids - seen_ids
    # Returns the new Ids
    return new_ids

# Main loop for processing the video frames and apply the seurity logic
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    # Get the current time
    current_time = time.time()
    # Break the loop if no frame is returned
    if not ret:
        break
    # Increment frame count
    frame_count += 1

    # Run the YOLO model on the frame
    results = model(frame)

    # Create list of detections for the tracker
    detections = []
    # Loop over the results and filter for people only
    for result in results:
        # Convert the YOLO results to the format required by the tracker
        for r in result.boxes.data.tolist():
            # Give the values from the YOLO model their own variables
            x1, y1, x2, y2, score, class_id = r
            # Make sure the YOLO model only detects people and meets the detection threshold
            if int(class_id) == person_id_class and score > detection_threshold:
                # Convert to tlwh format for the DEEPSORT tracker
                w = x2 - x1
                h = y2 - y1
                # Create bbox in tlwh format
                bbox_tlwh = [x1, y1, w, h]
                # Append to detections list
                detections.append((bbox_tlwh, float(score), int(class_id)))

    # Update tracker with current frame detections
    tracked_objects = tracker.update(frame, detections)  

    # Group Entry Methodology 
    # Skip the initial frames to avoid false positives
    if frame_count > skip_start_frames:
        # Get newly tracked IDs
        new_ids = get_newly_tracked_ids(tracked_objects, seen_ids)

        # For loop to add new IDs to seen_ids and record their entry time
        for track_id in new_ids:
            seen_ids.add(track_id)
            id_entry_time[track_id] = current_time
        
        # List to get the filtered recent IDs
        filtered_ids = []

        # For loop to filter recent IDs based on time window
        for obj_id in recent_new_ids:
            # Get the time this ID first appeared
            entry_time = id_entry_time.get(obj_id, current_time)

            # Get the time since entry 
            time_since_entry = current_time - entry_time

            # Keep the ID only if it's within the time window
            if time_since_entry <= time_window:
                # Add to filtered list
                filtered_ids.append(obj_id)

        # Replace the old list with the filtered one
        recent_new_ids = filtered_ids
 
        # For loop to add new IDs to recent_new_ids
        for track_id in new_ids:
            # Add only if not already in the recent_new_ids
            if track_id not in recent_new_ids:
                # Add to recent_new_ids
                recent_new_ids.append(track_id)

        # If the number of recent new IDs exceeds the group entry threshold flag it
        if len(recent_new_ids) >= group_entry:
            # Change the group entry flag to true
            group_entry_flagged = True
            # Record the time the flag was raised
            flag_display_start_time = current_time
            

    # After 3 seconds remove the group entry flag and reset everything
    if group_entry_flagged and (current_time - flag_display_start_time > 3.0):
        group_entry_flagged = False
        recent_new_ids = []

    # End of Group Entry Methodology

    # Loitering Detection Methodology
    # Create set for current tracked IDs
    current_tracked_ids = set()

    # Loop throught tracked objects
    for tracked_object in tracked_objects:
        # Get the bounding box and track ID from the tracked object
        bounding_box, track_id = tracked_object
        # Add the track ID to the set 
        current_tracked_ids.add(track_id)


    # List to store IDs that are no longer visible
    ids_that_left = []

    # For loop for tracked IDs in loitering data
    for track_id in loitering_data:
        # If the track ID is not in the current tracked IDs
        if track_id not in current_tracked_ids:
            # Add to the list of IDs that have left
            ids_that_left.append(track_id)

    # For loop for IDs that have left
    for track_id in ids_that_left:
        # if track ID in loitering 
        if track_id in loitering_data:
            # Remove from loitering data
            del loitering_data[track_id]

        # Remove loitering alert flag 
        loitering_flagged.discard(track_id)

    # Loop through all tracked objects in the current frame
    for bbox_tlwh, track_id in tracked_objects:
        # Assign bounding box values to variables
        x, y, w, h = bbox_tlwh
        
        # Calculate the center coordinates of the bounding box
        center_x = x + w / 2
        center_y = y + h / 2

        # If the track ID is not already in loitering data
        if track_id not in loitering_data:
            # Add the track ID with its current position and time
            loitering_data[track_id] = [center_x, center_y, current_time]

            # Remove from loitering flagged set if it was flagged before
            loitering_flagged.discard(track_id)

        else: # If the track ID is being tracked for loitering
            
            # Get its last known position and stationary strating time
            last_x, last_y, stationary_start_time = loitering_data[track_id]

            # Calculate how far the object has moved since last check with euclidean distance
            distance_moved = math.hypot(center_x - last_x, center_y - last_y)

            # Check if object is stationary
            if distance_moved < loitering_distance:
                # Check if person has been standing still long enough to be loitering
                time_stationary = current_time - stationary_start_time
                # If the time stationary is more than the loitering time 
                if time_stationary >= loitering_time:
                    # Flag the object as loitering if not already flagged
                    if track_id not in loitering_flagged:
                        # Add to loitering flagged set
                        loitering_flagged.add(track_id)

            else: # If person has moved
                
                # Reset its stationary tracking
                loitering_data[track_id] = [center_x, center_y, current_time]

                # Remove from loitering flagged set if it was flagged before
                loitering_flagged.discard(track_id)

    # End of loitering methodology
    
    # Direction change methodology

    # List of IDs to remove 
    ids_to_remove = []
    # Loop for track IDs in path history
    for track_id in path_history:
        # If the track ID is not in the current tracked IDs
        if track_id not in current_tracked_ids:
            # Add to the list of IDs to remove
            ids_to_remove.append(track_id)

    # For loop to remove IDs no longer tracked
    for track_id in ids_to_remove:
        # If in path history
        if track_id in path_history:
            # Pop (remove) from path history
            path_history.pop(track_id)

        # If in direction flagged
        if track_id in direction_flagged:
            # remove from direction flagged
            direction_flagged.pop(track_id)
    # for bounding boxes and track IDs in tracked objects
    for bbox_tlwh, track_id in tracked_objects:
        # Assign bounding box values to variables
        x, y, w, h = bbox_tlwh
        # Calculate the centroid of the bounding box
        cx = x + w / 2
        cy = y + h / 2
        # Current centroid position
        current_centroid = (cx, cy)

        # if track id not in path history
        if track_id not in path_history:
            # create a deque for track ID 
            path_history[track_id] = deque(history_length)
            # Append the current centroid
            path_history[track_id].append(current_centroid)

        #-------------------------- needs

        # Check for direction change if we have enough history (at least 3 points)
        if len(path_history[track_id]) >= 3:
            # Get the last three points: P1 (older), P2 (middle), P3 (newest)
            p1, p2, p3 = list(path_history[track_id])[-3:]
            p1, p2, p3 = map(lambda p: list(p), [p1, p2, p3]) # convert tuples to lists for numpy

            # Create vectors V1 (P1->P2) and V2 (P2->P3)
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])

            # Check if movement is significant enough to calculate a stable angle
            dist_v1 = math.hypot(*v1)
            dist_v2 = math.hypot(*v2)

            if dist_v1 > direction_change_min_distance and dist_v2 > direction_change_min_distance:
                # Calculate the angle between the vectors
                dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                angle_rad = math.acos(dot_product / (dist_v1 * dist_v2))
                angle_deg = math.degrees(angle_rad)

                if angle_deg > DIRECTION_CHANGE_ANGLE_THRESHOLD:
                    direction_flagged[track_id] = current_time
        #---------------------------
                    
    # End of direction change methodology
    

    # Draw the tracking info and flags on the frame
    for bbox_tlwh, track_id in tracked_objects:
        # if the track ID is new
        if track_id not in seen_ids:
            # Add to the seen IDs set
            seen_ids.add(track_id)
        # Assign bounding box values to variables
        x, y, w, h = bbox_tlwh
        # Convert to integers
        x1, y1, w, h = map(int, [x, y, w, h])
        # Assign bottom right coordinates
        x2 = x1 + w
        # Assign top left coordinates
        y2 = y1 + h
        # Pick a colour for the ID
        colour_index = int(track_id) % len(colours)
        # Get the colour
        base_colour = colours[colour_index]
        # Assign the colour
        colour = base_colour  

        # Check if direction change flag is still active
        if track_id in direction_flagged and (current_time - direction_flagged[track_id] > flag_duration):
            direction_flagged.pop(track_id, None)

        # Add the flag to the ID
        # flag for loitering
        if track_id in loitering_flagged:
            color = (255, 255, 0)  
            text_label = f"ID: {track_id} (LOITERING!)"
        # flag for direction change
        elif track_id in direction_flagged:
            color = (0, 165, 255)  # Orange for direction change
            text_label = f"ID: {track_id} (DIRECTION!)"
        # Normal ID when no flags
        else:
            text_label = f"ID: {track_id}"

        # Draw bounding box for the person
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Put text label above box 
        ((text_w, text_h), _) = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        # Get top left and bottom right for the text
        text_bg_pt1 = (x1, y1 - text_h - 10)
        text_bg_pt2 = (x1 + text_w + 6, y1)
        # Draw rectangle background for text
        cv2.rectangle(frame, text_bg_pt1, text_bg_pt2, (0, 0, 0), -1)
        # Draw the text label
        cv2.putText(frame, text_label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw the path history and an arrow showing current direction
        if track_id in path_history and len(path_history[track_id]) > 1:
            # Draw the path trail
            path_points = [tuple(map(int, p)) for p in path_history[track_id]]
            for i in range(1, len(path_points)):
                cv2.line(frame, path_points[i-1], path_points[i], color, 2)

            # Draw an arrow for the most recent movement
            if len(path_points) > 1:
                start_point = path_points[-2]
                end_point = path_points[-1]
                cv2.arrowedLine(frame, start_point, end_point, (255, 255, 255), 2, tipLength=0.4)

    # Draw the counts
    # Thr current count
    current_count = len(tracked_objects)
    # The total unique count
    total_count = len(seen_ids)
    # Text to display the counts
    count_text = f"Current: {current_count}   Total Unique: {total_count}"
    # Draw text for counts
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw the Group Entry Flag
    # If group entry flag is raised
    if group_entry_flagged:
        # Text for the flag
        flag_text = "!!! GROUP ENTRY FLAGGED !!!"
        # Draw rectangle for flag
        cv2.rectangle(frame, (0, 50), (frame.shape[1], 100), (0, 0, 255), -1)
        # Add in the text
        cv2.putText(frame, flag_text, (50, 85), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # display the video
    cv2.imshow("Pauric's security system", frame)

    # break loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the window
cv2.destroyAllWindows()