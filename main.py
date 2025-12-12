#
import random
import cv2
import time
import math
from collections import deque
from ultralytics import YOLO

from tracker import ObjectTracker  # your DeepSort wrapper

# --- Group Entry Constants ---
GROUP_ENTRY_THRESHOLD_COUNT = 4  # Number of new IDs to trigger a flag
GROUP_ENTRY_TIME_WINDOW = 2.0    # Time window in seconds
# -----------------------------

# --- Loitering Constants ---
LOITERING_TIME_THRESHOLD = 8.0       # Time in seconds (8 seconds)
LOITERING_DISTANCE_THRESHOLD = 20.0  # Max pixel distance change to be considered stationary
# ---------------------------

# --- Direction Change Constants ---
PATH_HISTORY_LENGTH = 15               # Number of recent positions to store.
DIRECTION_CHANGE_ANGLE_THRESHOLD = 90  # Angle in degrees to trigger a flag (e.g., > 90 for a sharp turn).
DIRECTION_CHANGE_MIN_DISTANCE = 15.0   # Minimum distance (in pixels) between points in history to calculate a reliable angle. This filters out jitter.
DIRECTION_FLAG_DURATION = 3.0          # How many seconds the direction change flag stays active.
# ----------------------------------

# --- YOLO CONSTANT ---
PERSON_CLASS_ID = 0  # Class ID for "person" in the COCO dataset
# ---------------------

# --- Startup Constants ---
STARTUP_FRAME_SKIP = 30  # Skip group detection for the first 30 frames
# -------------------------

# read in my video
cap = cv2.VideoCapture("loitering.mp4")

# load in my pre-trained YOLO model
model = YOLO("yolov8n.pt")

# initialize my object tracker
tracker = ObjectTracker()

# create unique rgb for different track IDs
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(100)]

# set detection threshold
detection_threshold = 0.5

# --- Tracking Variables ---
seen_ids = set()
id_entry_time = {}
recent_new_ids = []
group_entry_flagged = False
flag_display_start_time = 0
frame_count = 0

# Loitering detection state
loitering_data = {}     # track_id : [last_x, last_y, stationary_start_time]
loitering_flagged = set()

# Direction change detection state
path_history = {}        # {track_id: deque([(x, y), ...])}
direction_flagged = {}   # {track_id: flag_activation_time}
# --------------------------

# Helper function to get the IDs that are new in the current frame
def get_newly_tracked_ids(tracked_objects, seen_ids):
    current_ids = {track_id for _, track_id in tracked_objects}
    new_ids = current_ids - seen_ids
    return new_ids

# Main loop
while True:
    ret, frame = cap.read()
    current_time = time.time()

    if not ret:
        break

    frame_count += 1

    # Run YOLO inference on the frame
    results = model(frame)

    # Gather detections in TLWH format expected by DeepSORT wrapper: (bbox_tlwh, score, class_id)
    detections = []
    for result in results:
        for r in result.boxes.data.tolist():
            # r: [x1, y1, x2, y2, score, class_id]
            x1, y1, x2, y2, score, class_id = r
            if int(class_id) == PERSON_CLASS_ID and score > detection_threshold:
                w = x2 - x1
                h = y2 - y1
                bbox_tlwh = [x1, y1, w, h]
                detections.append((bbox_tlwh, float(score), int(class_id)))

    # Update tracker
    tracked_objects = tracker.update(frame, detections)  # returns list of (bbox_tlwh, track_id)

    # --- Group Entry Detection ---
    if frame_count > STARTUP_FRAME_SKIP:
        new_ids = get_newly_tracked_ids(tracked_objects, seen_ids)

        for track_id in new_ids:
            seen_ids.add(track_id)
            id_entry_time[track_id] = current_time

        # purge recent_new_ids outside the GROUP_ENTRY_TIME_WINDOW
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

    # --- Loitering Detection ---
    current_tracked_ids = {track_id for _, track_id in tracked_objects}

    # Prune loitering_data for IDs that have left the frame
    ids_to_remove = [tid for tid in loitering_data if tid not in current_tracked_ids]
    for tid in ids_to_remove:
        loitering_data.pop(tid, None)
        loitering_flagged.discard(tid)

    for bbox_tlwh, track_id in tracked_objects:
        x, y, w, h = bbox_tlwh
        center_x = x + w / 2
        center_y = y + h / 2

        if track_id not in loitering_data:
            loitering_data[track_id] = [center_x, center_y, current_time]
            loitering_flagged.discard(track_id)
        else:
            last_x, last_y, stationary_start_time = loitering_data[track_id]
            distance = math.hypot(center_x - last_x, center_y - last_y)

            if distance < LOITERING_DISTANCE_THRESHOLD:
                if (current_time - stationary_start_time) >= LOITERING_TIME_THRESHOLD:
                    if track_id not in loitering_flagged:
                        loitering_flagged.add(track_id)
                        print(f"!!! ALERT: Loitering detected for ID {track_id} !!!")
                # else: still within stationary period; do nothing
            else:
                # Reset stationary tracking
                loitering_data[track_id] = [center_x, center_y, current_time]
                loitering_flagged.discard(track_id)
    # ---------------------------------

    # --- Direction Change Detection ---
    # Prune path and flag data for IDs that have left the frame
    dir_ids_to_remove = [tid for tid in path_history if tid not in current_tracked_ids]
    for tid in dir_ids_to_remove:
        path_history.pop(tid, None)
        direction_flagged.pop(tid, None)

    for bbox_tlwh, track_id in tracked_objects:
        x, y, w, h = bbox_tlwh
        cx = x + w / 2
        cy = y + h / 2
        current_centroid = (cx, cy)

        # Add current position to path history
        if track_id not in path_history:
            path_history[track_id] = deque(maxlen=PATH_HISTORY_LENGTH)
        path_history[track_id].append(current_centroid)

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

            if dist_v1 > DIRECTION_CHANGE_MIN_DISTANCE and dist_v2 > DIRECTION_CHANGE_MIN_DISTANCE:
                # Calculate the angle between the vectors
                dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                angle_rad = math.acos(dot_product / (dist_v1 * dist_v2))
                angle_deg = math.degrees(angle_rad)

                if angle_deg > DIRECTION_CHANGE_ANGLE_THRESHOLD:
                    direction_flagged[track_id] = current_time
                    print(f"!!! ALERT: Sharp direction change for ID {track_id} ({angle_deg:.1f}°) !!!")
    # -----------------------------------

    # Draw tracked boxes, labels, arrows and counts
    for bbox_tlwh, track_id in tracked_objects:
        # Ensure we add ALL IDs to seen_ids even during the skip period for the Total Count
        if track_id not in seen_ids:
            seen_ids.add(track_id)

        x, y, w, h = bbox_tlwh
        x1, y1, w, h = map(int, [x, y, w, h])
        x2 = x1 + w
        y2 = y1 + h

        color_index = int(track_id) % len(colors)
        base_color = colors[color_index]
        color = base_color  # default

        # Check if direction change flag is still active
        if track_id in direction_flagged and (current_time - direction_flagged[track_id] > DIRECTION_FLAG_DURATION):
            direction_flagged.pop(track_id, None)

        # Decide label and color priority: Loitering > Direction-change > Group-entry > normal
        if track_id in loitering_flagged:
            color = (255, 255, 0)  # Cyan/Yellow for Loitering
            text_label = f"ID: {track_id} (LOITERING!)"
        elif track_id in direction_flagged:
            color = (0, 165, 255)  # Orange for direction change
            text_label = f"ID: {track_id} (DIRECTION!)"
        elif track_id in recent_new_ids:
            color = (0, 0, 255)  # Red for Group Entry
            text_label = f"ID: {track_id}"
        else:
            text_label = f"ID: {track_id}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Put text label above box (with a background for readability)
        ((text_w, text_h), _) = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_bg_pt1 = (x1, y1 - text_h - 10)
        text_bg_pt2 = (x1 + text_w + 6, y1)
        cv2.rectangle(frame, text_bg_pt1, text_bg_pt2, (0, 0, 0), -1)
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

    # Draw Counts
    current_count = len(tracked_objects)
    total_count = len(seen_ids)
    count_text = f"Current: {current_count}   Total Unique: {total_count}"
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw the Group Entry Flag
    if group_entry_flagged:
        flag_text = "!!! GROUP ENTRY FLAGGED !!!"
        cv2.rectangle(frame, (0, 50), (frame.shape[1], 100), (0, 0, 255), -1)
        cv2.putText(frame, flag_text, (50, 85), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # display the video
    cv2.imshow("Pauric's security system", frame)

    # break loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()