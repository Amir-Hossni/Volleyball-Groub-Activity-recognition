"""
Dataset audit for the real Volleyball data (run on Kaggle).

Uses the project's own loaders so the numbers describe exactly what
VolleyballDataset returns.

    python audit_dataset.py            # stats for train + val, visuals for 3 samples
    python audit_dataset.py --visual 6 # more visual samples

Outputs:
    printed statistics
    /kaggle/working/audit/*.png
"""
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image

from Data.dataset import VolleyballDataset
from Data.preprocessing import prepare_model
from Data.volleyball_annot_loader import load_video_annot
from Data.boxinfo import BoxInfo

OUT = Path("/kaggle/working/audit")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def raw_lines_per_player(tracking_file):
    """Raw annotation lines per player before the [5:-6] slice."""
    per_player = defaultdict(list)
    with open(tracking_file) as f:
        for line in f:
            b = BoxInfo(line)
            per_player[b.player_ID].append(b.frame_ID)
    return per_player


def audit_split(name, ds, videos_path, annot_root):
    print(f"\n{'=' * 70}\n{name}: {len(ds)} samples\n{'=' * 70}")
    idx_to_name = {}
    for k, v in ds.scene_to_idx.items():
        idx_to_name.setdefault(v, k)

    per_video = Counter(s["video_id"] for s in ds.samples)
    per_class = Counter(s["scene_label"] for s in ds.samples)
    print("videos:", len(per_video), sorted(per_video, key=int))
    print("samples/video: min", min(per_video.values()), "max", max(per_video.values()))
    print("samples/class:", {idx_to_name[c]: per_class[c] for c in sorted(per_class)})

    raw_label_strings = Counter()
    T_hist, target_idx_hist, offsets_hist = Counter(), Counter(), Counter()
    players_per_frame, raw_len_hist, ids_over_11 = Counter(), Counter(), 0
    lost, generated, total_boxes = 0, 0, 0
    inverted, zero_area, out_of_bounds = [], [], []
    player_full_track = Counter()
    max_jump = []  # per player: max centre displacement between consecutive frames / box height
    clip_windows = defaultdict(list)

    for s in ds.samples:
        vid, cid, fb = s["video_id"], s["clip_id"], s["frame_boxes"]
        target = int(cid)
        frames = sorted(fb)
        T_hist[len(frames)] += 1
        offsets_hist[tuple(f - target for f in frames)] += 1
        target_idx_hist[frames.index(target) if target in frames else "absent"] += 1
        clip_windows[vid].append((frames[0], frames[-1], cid))

        tracking_file = Path(annot_root) / vid / cid / f"{cid}.txt"
        for pid, lines in raw_lines_per_player(tracking_file).items():
            if pid > 11:
                ids_over_11 += 1
            else:
                raw_len_hist[len(lines)] += 1

        W, H = Image.open(Path(videos_path) / vid / cid / f"{frames[0]}.jpg").size
        tracks = defaultdict(dict)
        for f in frames:
            players_per_frame[len(fb[f])] += 1
            for b in fb[f]:
                total_boxes += 1
                lost += b.lost
                generated += b.generated
                x1, y1, x2, y2 = b.box
                if x2 < x1 or y2 < y1:
                    inverted.append((vid, cid, f, b.player_ID, b.box))
                elif x2 == x1 or y2 == y1:
                    zero_area.append((vid, cid, f, b.player_ID, b.box))
                if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                    out_of_bounds.append((vid, cid, f, b.player_ID, b.box, (W, H)))
                tracks[b.player_ID][f] = b.box
        for pid, tr in tracks.items():
            player_full_track[len(tr) == len(frames)] += 1
            fs = sorted(tr)
            jumps = []
            for a, c in zip(fs, fs[1:]):
                (ax1, ay1, ax2, ay2), (cx1, cy1, cx2, cy2) = tr[a], tr[c]
                d = (((ax1 + ax2) - (cx1 + cx2)) ** 2 + ((ay1 + ay2) - (cy1 + cy2)) ** 2) ** 0.5 / 2
                jumps.append(d / max(ay2 - ay1, 1))
            if jumps:
                max_jump.append((max(jumps), vid, cid, pid))

    ann_cache = {}
    for s in ds.samples:
        vid = s["video_id"]
        if vid not in ann_cache:
            ann_cache[vid] = load_video_annot(Path(videos_path) / vid / "annotations.txt")
        raw_label_strings[ann_cache[vid][s["clip_id"]]] += 1

    overlaps = 0
    for vid, w in clip_windows.items():
        w.sort()
        overlaps += sum(1 for a, b in zip(w, w[1:]) if b[0] <= a[1])

    print("raw label strings:", dict(raw_label_strings))
    print("T (frames per clip):", dict(T_hist))
    print("frame offsets vs clip id (target):", {k: v for k, v in offsets_hist.most_common(5)})
    print("target index inside window:", dict(target_idx_hist))
    print("raw lines per player (before slice):", dict(sorted(raw_len_hist.items())))
    print("player-track entries with id > 11 (dropped):", ids_over_11)
    print("players per frame:", dict(sorted(players_per_frame.items())))
    miss = sum((12 - k) * v for k, v in players_per_frame.items())
    print(f"missing player slots: {miss} / {12 * sum(players_per_frame.values())} "
          f"({100 * miss / (12 * sum(players_per_frame.values())):.2f}%)")
    print("present players with all 9 frames:", dict(player_full_track))
    print(f"boxes: {total_boxes} | lost=1: {lost} | generated=1: {generated}")
    print("inverted boxes:", len(inverted), inverted[:5])
    print("zero-area boxes:", len(zero_area), zero_area[:5])
    print("out-of-bounds boxes:", len(out_of_bounds), out_of_bounds[:5])
    print("clip windows overlapping another clip of same video:", overlaps)
    max_jump.sort(reverse=True)
    print("largest per-frame centre jump (in box heights) — identity-swap candidates:")
    for j in max_jump[:10]:
        print(f"   {j[0]:.2f}  video {j[1]} clip {j[2]} player {j[3]}")
    return set(per_video)


def visualise(ds, index, videos_path):
    s = ds.samples[index]
    vid, cid, fb = s["video_id"], s["clip_id"], s["frame_boxes"]
    frames = sorted(fb)
    label = next(k for k, v in ds.scene_to_idx.items() if v == s["scene_label"])
    item = ds[index]
    imgs, mask = item["images"], item["mask"]

    # 1) Target frame with all slot boxes, IDs and action labels
    target = int(cid)
    img = Image.open(Path(videos_path) / vid / cid / f"{target}.jpg")
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(img)
    for b in fb.get(target, []):
        x1, y1, x2, y2 = b.box
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", lw=1.5))
        ax.text(x1, y1 - 4, f"{b.player_ID}:{b.category}", color="yellow", fontsize=8, backgroundcolor="black")
    ax.set_title(f"video {vid} clip {cid} target frame {target} | group label: {label}")
    ax.axis("off")
    fig.savefig(OUT / f"{vid}_{cid}_target_boxes.png", dpi=90, bbox_inches="tight")
    plt.close(fig)

    # 2) Dataset tensor grid: rows = time, cols = slot. Same physical player must fill each column.
    fig, axes = plt.subplots(len(frames), 12, figsize=(18, 1.7 * len(frames)))
    for t in range(len(frames)):
        for p in range(12):
            a = axes[t, p]
            a.imshow((imgs[t, p] * STD + MEAN).clamp(0, 1).permute(1, 2, 0))
            a.set_xticks([]); a.set_yticks([])
            if not mask[t, p]:
                a.set_facecolor("red"); a.text(0.5, 0.5, "PAD", color="red", ha="center", transform=a.transAxes)
            if t == 0:
                a.set_title(f"slot {p}", fontsize=8)
            if p == 0:
                a.set_ylabel(f"t{frames[t] - target:+d}\n{frames[t]}", fontsize=7)
    fig.suptitle(f"video {vid} clip {cid} | {label} | rows=time cols=player slot | mask sum={int(mask.sum())}")
    fig.savefig(OUT / f"{vid}_{cid}_slot_grid.png", dpi=70, bbox_inches="tight")
    plt.close(fig)

    # 3) Full 9-frame strip
    fig, axes = plt.subplots(1, len(frames), figsize=(3 * len(frames), 2.2))
    for a, f in zip(axes, frames):
        a.imshow(Image.open(Path(videos_path) / vid / cid / f"{f}.jpg"))
        a.set_title(f"{f} (t{f - target:+d})", fontsize=8); a.axis("off")
    fig.savefig(OUT / f"{vid}_{cid}_frames.png", dpi=70, bbox_inches="tight")
    plt.close(fig)
    print(f"visuals written for video {vid} clip {cid} ({label})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open("config.yaml"))["Data"]
    root = Path(cfg["DATA_ROOT"])
    videos_path = root / cfg["PATHS"]["VIDEOS_PATH"]
    annot_root = root / cfg["PATHS"]["TRACKING_ANNOTATION_PATH"]
    common = dict(videos_path=videos_path, annot_root=annot_root,
                  scene_to_idx=cfg["CATEGORIES"]["SCENE_TO_IDX"],
                  player_to_idx=cfg["CATEGORIES"]["PLAYER_TO_IDX"],
                  mode="clip_frames_players", transform=prepare_model(image_level=False))

    train = VolleyballDataset(split_ids=cfg["SPLIT"]["TRAIN_IDS"], **common)
    val = VolleyballDataset(split_ids=cfg["SPLIT"]["VAL_IDS"], **common)
    tv = audit_split("TRAIN", train, videos_path, annot_root)
    vv = audit_split("VAL", val, videos_path, annot_root)
    print("\nvideo overlap train/val:", tv & vv)
    all_videos = {p.name for p in videos_path.iterdir() if p.is_dir()}
    print("videos on disk not in train/val:", sorted(all_videos - tv - vv, key=int))

    rng = random.Random(0)
    for i in rng.sample(range(len(val)), args.visual):
        visualise(val, i, videos_path)


if __name__ == "__main__":
    main()
