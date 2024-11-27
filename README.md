# Tennis Match Tracker

This project uses computer vision and deep learning to track tennis players and the ball within match footage. It combines YOLO-based models for ball detection and player tracking, along with a ResNet50-based model to detect court keypoints. The system processes video frames, tracks player and ball locations, and visualizes the results with bounding boxes and keypoints.

## Features:
- **Ball Tracking**: Detect and track the ball using YOLOv5.
- **Player Tracking**: Track players using a YOLOv8 model.
- **Court Keypoint Detection**: Detect court lines and key points using a ResNet50-based model.
- **Frame Annotation**: Visualize player and ball locations with bounding boxes and court keypoints.
- **Output**: Annotated video saved for further analysis.

## Requirements:
- Python 3.x
- PyTorch
- OpenCV
- torchvision
- other dependencies (see `requirements.txt`)

## Installation:
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/tennis-match-tracker.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the required pre-trained models:
   - [YOLOv5 model](https://huggingface.co/bountyhunterxx/Tennis-match-models/resolve/main/yolo5_last.pt)
   - [ResNet50 Court Model](https://huggingface.co/bountyhunterxx/Tennis-match-models/resolve/main/keypoints_model.pth)
   - [YOLOv8 model for player tracking](https://huggingface.co/bountyhunterxx/Tennis-match-models/resolve/main/yolov8x.pt)

## Usage:
1. Place your input video in the `data/` folder.
2. Place your models in `models/` folder.
3. Run the script:
   ```
   python main.py
   ```
4. The output video will be saved as `video.avi`.

## Notes:
- Ensure that the model paths are correctly set in the script.
- The input video should be in `.mp4` format.
