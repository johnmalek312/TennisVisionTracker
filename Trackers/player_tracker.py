import cv2
from ultralytics import YOLO
import os
import pickle
import torch


class PlayerTracker:
    def __init__(self, original_h, original_w, model_path: str = "models/yolov8x.pt"):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {e}")
        self.original_h, self.original_w = original_h, original_w

    def track(self, frames: list, load_cache: bool = False, cache_path: str = "cache/player_stub.pkl") -> list:
        if load_cache and os.path.exists(cache_path):
            with open(cache_path, "rb") as file:
                location = pickle.load(file)
            return location

        location = [{} for _ in range(len(frames))]
        for i, frame in enumerate(frames):
            results = self.model.track(frame, persist=True)
            for result in results:
                people = torch.where(result.boxes.cls == 0)[0].tolist()
                for person in people:
                    location[i][int(result.boxes.id[person].item())] = result.boxes.xyxy[person].squeeze().numpy()
        with open(cache_path, "wb") as file:
            pickle.dump(location, file)
        return location

    def draw_box(self, frames, locations, players):
        locations = [{key: d[key] for key in players if key in d} for d in locations]

        output_frames = []
        for i, people in enumerate(locations):
            for person in people:
                id = person
                person = people[person]
                if person[0]:
                    x1, y1, x2, y2 = map(int, person)
                    cv2.putText(frames[i], f"PLayer {id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                color=(10, 20, 200), thickness=2)
                    cv2.rectangle(frames[i], (x1, y1), (x2, y2), (10, 20, 200), 2)
            output_frames.append(frames[i])
        return output_frames

    def calculate_scores(self, location, keypoints_subset):
        temp_score = {}
        for pid, person in location.items():
            px = (person[0] + person[2]) / 2
            py = (person[1] + person[3]) / 2
            tscore = sum(
                ((kx - px) ** 2 + (ky - py) ** 2) for kx, ky in keypoints_subset
            ) / (len(keypoints_subset) + 1e-10)
            temp_score[pid] = tscore
        return temp_score

    def transform_keypoints(self, keypoints):
        """Scale keypoints"""
        x_key = keypoints[::2] / 224 * self.original_w
        y_key = keypoints[1::2] / 224 * self.original_h
        return x_key, y_key

    def choose_player(self, locations, keypoints):
        fplayer_points = {2, 5, 10, 13, 11, 7, 3}
        log = True
        player_score = {}
        for location, keypoint in zip(locations, keypoints):
            x_key, y_key = self.transform_keypoints(keypoint)
            x_list1 = [x_key[i] for i in range(len(x_key)) if i in fplayer_points]
            y_list1 = [y_key[i] for i in range(len(y_key)) if i in fplayer_points]
            scores = self.calculate_scores(location, list(zip(x_list1, y_list1)))
            closest = min(scores, key=scores.get)
            player_score[closest] = player_score.get(closest, 0) + 1
        fplayer = max(player_score, key=player_score.get)

        player_score = {}
        for location, keypoint in zip(locations, keypoints):
            x_key, y_key = self.transform_keypoints(keypoint)
            x_list2 = [x_key[i] for i in range(len(x_key)) if i not in fplayer_points]
            y_list2 = [y_key[i] for i in range(len(y_key)) if i not in fplayer_points]
            scores = self.calculate_scores(location, list(zip(x_list2, y_list2)))
            closest = min(scores, key=scores.get)
            player_score[closest] = player_score.get(closest, 0) + 1
        splayer = max(player_score, key=player_score.get)
        return [fplayer, splayer]
