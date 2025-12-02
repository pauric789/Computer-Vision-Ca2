from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    
    # Initialize the DeepSort tracker
    tracker = DeepSort(
        max_age=30,             # Maximum number of frames a track can be missing before deletion
        nn_budget=None,         # Number of features to keep for each track
        embedder_model_name='mobilenet', # The model used for appearance feature extraction
        n_init=3                # Number of frames to confirm a track
    )
    
  
    # method to update the tracker with new detections
    def update(self, frame, detections):
        
        
        # udate the tracks with the new detections
        tracks = self.tracker.update_tracks(detections, frame=frame)
        # create a list to hold the people
        tracked_objects = []
        
        # for loop through the tracks
        for track in tracks:
            

            # Get the bounding box in TLWH format 
            bbox_tlwh = track.to_tlwh() 
            # get the track ID
            track_id = track.track_id
            
            
            # append the bbox and track id to the list
            tracked_objects.append((bbox_tlwh, track_id))
        # return the list of bbox and track ids  
        return tracked_objects

