from collections import defaultdict
import numpy as np


def audit_player_id_spatial_consistency(dataset, name="Dataset"):
    """
    Audit whether player_ID has a stable spatial meaning across clips.

    For every sample:
        - compute the mean normalized (x, y) position of each player_ID
          across the 9 temporal frames.

    Then across all samples:
        - calculate mean/std position for each player_ID
        - calculate spatial spread
        - calculate pairwise distance between player-ID centers
    """

    # player_id -> list of one representative (x, y) position per sample
    player_positions = defaultdict(list)

    total_samples = len(dataset)

    for idx in range(total_samples):

        sample = dataset.samples[idx]

        frame_boxes = sample["frame_boxes"]

        # Collect positions for each player inside this sample
        sample_positions = defaultdict(list)

        for frame_id, boxes in frame_boxes.items():

            for box_info in boxes:

                player_id = box_info.player_ID

                x1, y1, x2, y2 = box_info.box

                # Bounding-box center
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Normalize to [0, 1]
                # Volleyball frames: 1280 x 720
                cx_norm = cx / 1280.0
                cy_norm = cy / 720.0

                sample_positions[player_id].append(
                    (cx_norm, cy_norm)
                )

        # One representative position for each player
        # in this sample
        for player_id, positions in sample_positions.items():

            positions = np.asarray(positions)

            mean_position = positions.mean(axis=0)

            player_positions[player_id].append(
                mean_position
            )

    # =========================================================
    # Per-player statistics
    # =========================================================

    print("\n" + "=" * 90)
    print(f"PLAYER ID SPATIAL CONSISTENCY AUDIT — {name}")
    print("=" * 90)

    print(f"Samples analyzed: {total_samples}")

    print("\nPer-player spatial distribution:")
    print("-" * 90)

    player_means = {}

    for player_id in range(12):

        positions = player_positions[player_id]

        if len(positions) == 0:

            print(
                f"ID {player_id:2d}: "
                f"NO DATA"
            )

            continue

        positions = np.asarray(positions)

        mean_x = positions[:, 0].mean()
        mean_y = positions[:, 1].mean()

        std_x = positions[:, 0].std()
        std_y = positions[:, 1].std()

        min_x = positions[:, 0].min()
        max_x = positions[:, 0].max()

        min_y = positions[:, 1].min()
        max_y = positions[:, 1].max()

        player_means[player_id] = np.array(
            [mean_x, mean_y]
        )

        print(
            f"ID {player_id:2d} | "
            f"N={len(positions):4d} | "
            f"mean=({mean_x:.3f}, {mean_y:.3f}) | "
            f"std=({std_x:.3f}, {std_y:.3f}) | "
            f"range_x=({min_x:.3f}, {max_x:.3f}) | "
            f"range_y=({min_y:.3f}, {max_y:.3f})"
        )

    # =========================================================
    # Pairwise distance between player-ID mean positions
    # =========================================================

    print("\n" + "=" * 90)
    print("PAIRWISE DISTANCE BETWEEN PLAYER-ID MEAN POSITIONS")
    print("=" * 90)

    ids = sorted(player_means.keys())

    for i in range(len(ids)):

        for j in range(i + 1, len(ids)):

            p1 = player_means[ids[i]]
            p2 = player_means[ids[j]]

            distance = np.linalg.norm(p1 - p2)

            print(
                f"ID {ids[i]:2d} <-> ID {ids[j]:2d} "
                f"| distance={distance:.3f}"
            )

    # =========================================================
    # Spatial region statistics
    # =========================================================

    print("\n" + "=" * 90)
    print("SPATIAL REGION DISTRIBUTION PER PLAYER ID")
    print("=" * 90)

    print(
        "Regions: "
        "LEFT (<0.33), CENTER (0.33-0.66), RIGHT (>0.66)"
    )

    for player_id in range(12):

        positions = player_positions[player_id]

        if len(positions) == 0:
            continue

        positions = np.asarray(positions)

        x_values = positions[:, 0]

        left = np.sum(x_values < 0.33)
        center = np.sum(
            (x_values >= 0.33) &
            (x_values <= 0.66)
        )
        right = np.sum(x_values > 0.66)

        total = len(x_values)

        print(
            f"ID {player_id:2d} | "
            f"LEFT={left / total:.2%} | "
            f"CENTER={center / total:.2%} | "
            f"RIGHT={right / total:.2%}"
        )


# =============================================================
# RUN — NO TRAINING
# =============================================================



