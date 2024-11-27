import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np


def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames), fps


def save_frames(output_video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps,
                          (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (8, 30), cv2.FONT_HERSHEY_COMPLEX, 1.1, (150, 150, 10), 2)
        out.write(frame)
    out.release()
