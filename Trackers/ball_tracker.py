import cv2
from ultralytics import YOLO
import os
from utils import video_utils, object_utils
import numpy as np
import pickle

class BallTracker:
    def __init__(self, model_path: str = "models\\yolo5_last.pt"):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading YOLO model from {model_path}: {e}")
    def track(self, frames: list, load_cache:bool = False, cache_path: str = "cache\\ball_stub.pkl"):
        if load_cache is True and os.path.exists(cache_path):
            with open(cache_path, "rb") as file:
                location = pickle.load(file)
            return location

        location = []
        for frame in frames:
            try:
                results = self.model.predict(frame)
                xyxy = np.array(results[0].boxes.xyxy.squeeze())
                if xyxy.shape != (4,):
                    location.append([np.nan, np.nan, np.nan, np.nan])
                else:
                    location.append(xyxy)
            except Exception as e:
                print(f"Error handling a frame : {e}")
                location.append([np.nan, np.nan, np.nan, np.nan])
        with open(cache_path, "wb") as file:
            pickle.dump(location, file)
        return location
    def draw_box(self, frames: list, locations:list):
        locations = object_utils.interpolate_ball_location(locations)
        output_frames = []
        for i, (x1, y1, x2, y2) in enumerate(locations):
            cv2.putText(frames[i], "Ball", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,color=(10, 20, 200), thickness=2)
            cv2.rectangle(frames[i], (int(x1), int(y1)), (int(x2) , int(y2)), (10, 20, 200), 2)
            output_frames.append(frames[i])
        return output_frames
