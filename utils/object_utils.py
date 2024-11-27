import pandas as pd


def interpolate_ball_location(ball_location: list):
    df = pd.DataFrame(ball_location, columns=["x1", "y1", "x2", "y2"]).interpolate().bfill()
    return df.values.tolist()


def interpolate_player_location(player_location: list):
    df = pd.DataFrame(player_location, columns=["x1", "y1", "x2", "y2"]).interpolate()
    return df.bfill().values.tolist()
