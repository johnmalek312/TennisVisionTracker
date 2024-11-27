import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import v2
import cv2
import os

class CourtLineModel(nn.Module):
    def __init__(self, pretrain="models/keypoints_model.pth"):
        super().__init__()
        if not os.path.exists(pretrain):
            raise FileNotFoundError(f"Model file not found: {pretrain}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = v2.Compose(
            [v2.ToTensor(), v2.Resize((224, 224)), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.network = torchvision.models.resnet50(pretrained=True)
        self.network.fc = nn.Linear(self.network.fc.in_features, 28)
        self.network.load_state_dict(torch.load(pretrain, map_location=self.device))

    def forward(self, X):
        if len(X) == 0:
            raise ValueError("Empty input frames.")
        transformed_frames = []
        for frame in X:
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame.")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame_tensor = self.transform(frame_rgb)  # Apply transformations
            transformed_frames.append(frame_tensor)
        transformed_frames = torch.stack(transformed_frames, dim=0).to(self.device)
        output = self.network(transformed_frames)
        return output

    def draw_keypoints(self, frames):
        if len(frames) == 0:
            raise ValueError("Empty input frames.")
        self.eval()
        with torch.inference_mode():
            points = self.forward(frames)
        output = []
        for frame, point in zip(frames, points):
            output.append(_draw_keypoints(frame, point, frame.shape[0], frame.shape[1]))
        return output, points.cpu().numpy()


def _draw_keypoints(frame, points, original_h, original_w):
    xs = points[::2]
    ys = points[1::2]
    for i, (x, y) in enumerate(zip(xs, ys)):
        x, y = int(x.item() * original_w / 224), int(y.item() * original_h / 224)
        cv2.circle(frame, (x, y), 2, (20, 20, 255), 2)
        cv2.putText(frame, str(i), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (10, 10, 255), 1)
    return frame
