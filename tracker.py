import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    """
    A wrapper class for the DeepSort object tracker.
    """
    
    # Initialize the DeepSort tracker
    # 'mobilenet' is a good balance for speed and accuracy for the embedding model.
    tracker = DeepSort(
        max_age=30,             # Maximum number of frames a track can be 'lost' for before deletion
        nn_budget=None,         # Number of features to keep for each track
        embedder_model_name='mobilenet', # The model used for appearance feature extraction
        n_init=3                # Number of frames to confirm a track
    )
    
    def __init__(self):
        print("DeepSort ObjectTracker initialized.")

    def update(self, frame, detections):
        """
        Updates the tracker with new detections and returns a list of tracked objects.

        Args:
            frame (np.ndarray): The current video frame (image).
            detections (list): A list of detections in the format:
                                [ (bbox_tlbr, confidence, class_label), ... ]
                                where bbox_tlbr is [x_min, y_min, x_max, y_max]
        
        Returns:
            list: A list of confirmed tracked objects, each containing:
                  (bbox_tlwh, track_id, class_label)
        """
        
        # NOTE: DeepSort-Realtime requires detections in TLBR format ([x_min, y_min, x_max, y_max])
        # and usually handles feature extraction internally, so we pass the frame as well.
        
        # 1. Update the tracker with the detections
        # The 'frame' argument is used by the internal embedder for feature extraction.
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        tracked_objects = []
        
        # 2. Process and filter the tracks
        for track in tracks:
            # Skip unconfirmed tracks (those that haven't been seen enough times)
            # and deleted tracks (those that have been missing for too long).
            if not track.is_confirmed():
                continue

            # Get the bounding box in TLWH format (Top-Left, Width, Height)
            # This is a common output format for drawing.
            bbox_tlwh = track.to_tlwh() 
            track_id = track.track_id
            
            # The original class label is often stored in the track metadata
            class_label = track.get_det_class() 
            
            tracked_objects.append((bbox_tlwh, track_id, class_label))
            
        return tracked_objects

# --- Helper function for demonstration ---
def draw_tracks(frame, tracked_objects):
    """Draws bounding boxes and IDs on the frame."""
    for bbox, track_id, class_label in tracked_objects:
        x, y, w, h = [int(i) for i in bbox]
        
        # Draw Bounding Box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw ID and Label
        text = f"ID: {track_id} | {class_label}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame