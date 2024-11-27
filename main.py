import torch
from utils import video_utils
from Trackers.ball_tracker import BallTracker
from Trackers.player_tracker import PlayerTracker
from Trackers.courtline_detector import CourtLineModel

out_path = "video.avi"

if __name__ == '__main__':
    print("Started...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    frames, fps = video_utils.get_frames("data/input_video.mp4")
    # init
    ball_track = BallTracker("models/yolo5_last.pt")
    court_model = CourtLineModel("models/keypoints_model.pth").to(device)
    player_tracker = PlayerTracker(frames.shape[1], frames.shape[2], "models/yolov8x.pt")

    # process
    out_frames, points = court_model.draw_keypoints(frames)

    # track
    ball_location = ball_track.track(frames, True)
    player_location = player_tracker.track(frames=frames, load_cache=True)
    players = player_tracker.choose_player(player_location, points)

    # draw and save
    out_frames = ball_track.draw_box(out_frames, ball_location)
    out_frames = player_tracker.draw_box(out_frames, player_location, players)
    video_utils.save_frames(out_frames, out_path, fps)
    print("Saved!")
