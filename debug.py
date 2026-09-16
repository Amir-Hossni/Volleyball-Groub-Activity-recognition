import numpy as np
from collections import defaultdict


def audit_player_id_spatial_consistency(dataset):
    """
    Audit whether player_ID has a consistent spatial meaning
    across different clips.

    Uses the actual structure of VolleyballDataset:

        sample["player_tracks"][player_id][frame_id]["box"]

    where ["box"] is a BoxInfo object and:
        BoxInfo.box = (x1, y1, x2, y2)
    """

    # ---------------------------------------------------------
    # Collect one mean spatial position for each player_ID
    # in each clip.
    # ---------------------------------------------------------

    player_positions = defaultdict(list)

    for sample in dataset.samples:

        video_id = sample["video_id"]
        clip_id = sample["clip_id"]

        player_tracks = sample["player_tracks"]

        for player_id in range(12):

            track = player_tracks[player_id]

            if not track:
                continue

            centers = []

            for frame_id, item in track.items():

                box_info = item["box"]

                x1, y1, x2, y2 = box_info.box

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                centers.append((cx, cy))

            if not centers:
                continue

            centers = np.asarray(
                centers,
                dtype=np.float32
            )

            # Mean position of this player inside this clip
            mean_cx = centers[:, 0].mean()
            mean_cy = centers[:, 1].mean()

            player_positions[player_id].append(
                {
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "cx": mean_cx,
                    "cy": mean_cy,
                }
            )

    # ---------------------------------------------------------
    # Report cross-clip statistics
    # ---------------------------------------------------------

    print("=" * 90)
    print("PLAYER_ID CROSS-CLIP SPATIAL CONSISTENCY")
    print("=" * 90)

    results = {}

    for player_id in range(12):

        positions = player_positions[player_id]

        if not positions:
            continue

        xy = np.asarray(
            [
                [p["cx"], p["cy"]]
                for p in positions
            ],
            dtype=np.float32,
        )

        mean_xy = xy.mean(axis=0)

        std_x = xy[:, 0].std()
        std_y = xy[:, 1].std()

        distances = np.linalg.norm(
            xy - mean_xy,
            axis=1,
        )

        mean_distance = distances.mean()

        results[player_id] = {
            "num_clips": len(positions),
            "mean_cx": float(mean_xy[0]),
            "mean_cy": float(mean_xy[1]),
            "std_x": float(std_x),
            "std_y": float(std_y),
            "mean_distance": float(mean_distance),
        }

        print(
            f"Player {player_id:2d} | "
            f"clips={len(positions):4d} | "
            f"mean=({mean_xy[0]:7.2f}, {mean_xy[1]:7.2f}) | "
            f"std=({std_x:7.2f}, {std_y:7.2f}) | "
            f"mean_dist={mean_distance:7.2f}"
        )

    return results