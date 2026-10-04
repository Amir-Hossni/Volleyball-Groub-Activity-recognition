# Volleyball Group Activity Recognition

A PyTorch re-implementation of the baseline ladder from **Ibrahim et al., _A Hierarchical Deep Temporal Model for Group Activity Recognition_ (CVPR 2016)** on the Volleyball dataset, with a ResNet-50 backbone in place of the original AlexNet.

The goal is to recognise what a **team** is doing (pass, set, spike, winpoint, on the left or right side) from a short video clip, using two levels of supervision: an action label for every player and an activity label for the whole scene.

> **Status:** B1, B3, B4 and B5 are complete. B6, B7 and B8 are in progress. Every reported number is on the **validation** split; the test split has not been evaluated yet.

### At a glance

| Baseline | Idea | Val Accuracy | Val Macro-F1 |
|---|---|---|---|
| B1 | Single frame → fine-tuned ResNet-50 | 71.96% | 0.737 |
| B3 | Player crops → person-action CNN → max-pool → MLP | 75.47% | 0.728 |
| B4 | 9 frames → B1 backbone → LSTM | **76.44%** | **0.780** |
| B5 | Per-player LSTM → concat players → MLP | 64.65% | 0.611 |
| B6 – B8 | Two-stage hierarchical models | 🚧 | 🚧 |

Details and caveats are in [Results](#6-results).

---

## Table of Contents

1. [The Paper](#1-the-paper)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Data Pipeline](#4-data-pipeline)
5. [Baselines](#5-baselines)
6. [Results](#6-results)
7. [Engineering Notes](#7-engineering-notes)
8. [How to Run](#8-how-to-run)
9. [Roadmap](#9-roadmap)
10. [References](#10-references)

---

## 1. The Paper

**M. S. Ibrahim, S. Muralidharan, Z. Deng, A. Vahdat, G. Mori.**
*A Hierarchical Deep Temporal Model for Group Activity Recognition.* CVPR 2016.
[Paper (PDF)](https://www.cs.sfu.ca/~mori/research/papers/ibrahim-cvpr16.pdf) · [Official Caffe code and dataset](https://github.com/mostafa-saad/deep-activity-rec)

### 1.1 Core idea

A group activity is defined by what individual people do and how their actions evolve over time. The model therefore works in two stages:

1. **Person level.** For each tracked person, a CNN extracts appearance features from the person's bounding box at every time step, and an LSTM (**LSTM 1**) models how that person's action evolves.
2. **Group level.** At each time step, the person representations are pooled into one frame-level vector, and a second LSTM (**LSTM 2**) models the dynamics of the whole group.

```text
person tracklets ─► CNN ─► LSTM 1 ─┐
                     │             ├─► concat ─► max-pool over people ─► FC ─► LSTM 2 ─► softmax
                     └─────────────┘   (per time step)
```

Pooling (Eq. 7–8 in the paper):

```text
P_t^k = x_t^k ⊕ h_t^k            x = CNN (fc7) feature, h = LSTM 1 output, ⊕ = concat
Z_t   = P_t^1 ⋄ P_t^2 ⋄ … ⋄ P_t^K  ⋄ = max pooling over the K people at time t
```

### 1.2 Implementation details (Section 3.3)

| Component | Paper setting |
|---|---|
| Person CNN | AlexNet (ImageNet pre-trained), fine-tuned end-to-end with LSTM 1 on person tracklets |
| LSTM 1 | 9 time steps, 3000 hidden units, softmax over person actions |
| Pooling | Max pooling over all people (sum and average were tried and were consistently worse) |
| Group stage | 3000-node fully connected layer → 9-step LSTM with 500 units → softmax over group activities |
| Optimisation | Fixed learning rate 1e-5, momentum 0.9, same protocol for every model |
| Tracking | Danelljan et al. tracker (Dlib) |

### 1.3 Baselines defined in the paper (Section 4.1)

| ID | Name | Definition |
|---|---|---|
| B1 | Image Classification | AlexNet fine-tuned for group activity on a single frame |
| B2 | Person Classification | ImageNet AlexNet on each person, fc7 pooled over people, softmax on group activity |
| B3 | Fine-tuned Person Classification | As B2, but the person CNN is first fine-tuned on person actions, then frozen |
| B4 | Temporal Model with Image Features | Whole-frame fc7 features fed to an LSTM |
| B5 | Temporal Model with Person Features | fc7 features pooled over people, fed to an LSTM |
| B6 | Two-stage Model without LSTM 1 | Full model with only the fine-tuned person CNN at the person level |
| B7 | Two-stage Model without LSTM 2 | Full model without the group-level LSTM |

### 1.4 Results reported in the paper

These numbers are for the **original 2016 release** of the Volleyball dataset (15 videos, 1,525 annotated frames, 7 person actions, 6 group activities). They are not directly comparable with this project, which uses the later 55-video, 8-activity release.

| Method | Collective Activity | Volleyball (2016) |
|---|---|---|
| B1 Image Classification | 63.0 | 46.7 |
| B2 Person Classification | 61.8 | 33.1 |
| B3 Fine-tuned Person Classification | 66.3 | 35.2 |
| B4 Temporal Model with Image Features | 64.2 | 37.4 |
| B5 Temporal Model with Person Features | 62.2 | 45.9 |
| B6 Two-stage Model without LSTM 1 | 70.1 | 48.8 |
| B7 Two-stage Model without LSTM 2 | 76.8 | 49.7 |
| Two-stage Hierarchical Model | **81.5** | **51.1** |

The paper's conclusions: temporal modelling helps, explicitly modelling people helps, and the person-level LSTM (B7) contributes more than the group-level LSTM (B6). On volleyball, most errors come from left/right confusion.

---

## 2. Dataset

### 2.1 Overview

This project uses the extended Volleyball dataset release.

| Property | Value |
|---|---|
| Videos | 55 YouTube volleyball matches |
| Annotated clips | 4,830 (train 2,152 · val 1,341 · test 1,337) |
| Group activities | 8 (left/right × pass, set, spike, winpoint) |
| Person actions | 9 |
| Players per frame | up to 12 (6 per team), tracked with bounding boxes |
| Clip | annotated target frame `t`, with surrounding frames on disk |
| Tracking annotation | 20 frames per player (`t-10 … t+9`) |

### 2.2 Labels

**Group activities (8)**

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | `l_pass` | 1 | `r_pass` |
| 2 | `l_spike` | 3 | `r_spike` |
| 4 | `l_set` | 5 | `r_set` |
| 6 | `l_winpoint` | 7 | `r_winpoint` |

**Person actions (9):** `blocking`, `digging`, `falling`, `jumping`, `moving`, `setting`, `spiking`, `standing`, `waiting`.

> The raw annotation files spell some activities with a hyphen (`l-pass`, `r-pass`, `l-spike`) and others with an underscore (`r_spike`, `l_set`, …). `SCENE_TO_IDX` in `config.yaml` therefore has 16 keys mapping to the 8 class IDs. These aliases are intentional, and every target the model sees is in `0…7`.

### 2.3 Annotation format

`volleyball_tracking_annotation/<video>/<clip>/<clip>.txt` has one line per player per frame:

```text
<player_id> <x1> <y1> <x2> <y2> <frame_id> <lost> <grouping> <generated> <action>
```

`videos/<video>/annotations.txt` has one line per clip; the first two tokens are `<target_frame>.jpg <group_activity>`.

### 2.4 Split

The standard video-level split is used, so no video appears in more than one split.

| Split | Videos | Clips |
|---|---|---|
| Train | 1 3 6 7 10 13 15 16 18 22 23 31 32 36 38 39 40 41 42 48 50 52 53 54 | 2,152 |
| Validation | 0 2 8 12 17 19 24 26 27 28 30 33 46 49 51 | 1,341 |
| Test | 4 5 9 11 14 20 21 25 29 34 35 37 43 44 45 47 | 1,337 (not evaluated yet) |

### 2.5 Statistics (measured on the real data)

**Clips per group activity**

| Class | Train | Val |
|---|---|---|
| l_pass | 378 | 222 |
| r_pass | 353 | 238 |
| l_spike | 289 | 174 |
| r_spike | 261 | 189 |
| l_set | 304 | 160 |
| r_set | 283 | 169 |
| l_winpoint | 161 | 104 |
| r_winpoint | 123 | 85 |

**Players and boxes**

| | Train | Val |
|---|---|---|
| Player tracks (all complete over the 9-frame window) | 25,703 | 15,981 |
| Player crops (tracks × 9 frames) | 231,327 | 143,829 |
| Missing player slots | 0.47% | 0.69% |
| Clips with all 12 players | 95% | 94% |
| Boxes flagged `lost=1` | 0.79% | 0.74% |

The class mix and missing-player rate are almost identical between train and validation, so there is no meaningful distribution shift between the splits.

---

## 3. Project Structure

```text
.
├── config.yaml                 # dataset paths, label maps, video split
├── main.py                     # builds datasets, models and trainers; runs one experiment
├── debug.py                    # standalone debugging / diagnostic script
├── audit_dataset.py            # dataset statistics and visual checks on the real data
├── Data/
│   ├── boxinfo.py              # parses one tracking-annotation line
│   ├── volleyball_annot_loader.py  # tracking / clip annotation loaders
│   ├── dataset.py              # VolleyballDataset with one mode per baseline
│   └── preprocessing.py        # image transforms (image-level vs. crop-level)
├── Models/
│   ├── Baseline1/model_B1.py   # SceneClassifierB1
│   ├── Baseline2/model_B2.py   # B2Model
│   ├── Baseline3/model_B3.py   # PersonClassifierB3, GroupClassifierB3
│   ├── Baseline4/model_B4.py   # TemporalImageClassifierB4
│   ├── Baseline5/model_B5.py   # PersonTemporalB5, GroupTemporalClassifierB5
│   └── Baseline6/model_B6.py   # B6GroupActivityClassifier
├── engine/
│   ├── trainer.py              # training loop, AMP, checkpointing, early stopping
│   ├── evaluator.py            # validation loop
│   ├── adapters.py             # batch → (inputs, targets[, mask])
│   └── sampler.py              # class-balanced WeightedRandomSampler
├── utlis/                      # metrics, checkpoints, early stopping, logging, TensorBoard
└── logs/                       # text logs and TensorBoard events per baseline
```

---

## 4. Data Pipeline

### 4.1 Temporal window

`load_tracking_annot` keeps the middle 9 of the 20 annotated frames per player (`[5:]` then `[:-6]`):

```text
raw:      t-10 … t-6 | t-5 t-4 t-3 t-2 t-1  t  t+1 t+2 t+3 | t+4 … t+9
kept:                  └──────────────── 9 frames ──────────┘
target t is at index 5 of the window
```

Boxes are stored under their own `frame_id`, so a crop is always taken from the frame it was annotated on. A player missing from a frame simply has no box there.

### 4.2 Dataset modes

One `VolleyballDataset` class serves every baseline through its `mode` argument.

| Mode | Used by | One sample | Output |
|---|---|---|---|
| `single_frame` | B1 | middle frame of the window | `image (3,224,224)` |
| `person` | B3 stage A (reference sampling) | one player crop, all 9 frames | `image (3,224,224)`, `player_label` |
| `person_frames` | B3 stage A (current) | all players of one frame, all 9 frames | `images (12,3,224,224)`, `player_labels (12,)` |
| `person_grouped` | B3 stage B | all players of one frame (`t-5`) | `images (12,3,224,224)`, `player_labels (12,)` |
| `clip_frames` | B4 | 9 full frames | `frames (9,3,224,224)` |
| `person_temporal` | B5 | 12 player tracks × 9 frames | `images (12,9,3,224,224)`, `player_labels (12,)` |
| `clip_frames_players` | B6 | 9 frames × 12 players | `images (9,12,3,224,224)`, `mask (9,12)` |

`person_frames` produces exactly the same crops and labels as `person` (verified tensor by tensor). The difference is that it decodes each frame once for all of its players, which is about 7× faster to load.

**Player handling.** Player slots follow the tracking `player_id`, so slot `p` is the same person in every frame. Missing players are zero tensors, with label `-1` (removed by `flatten_person_batch`) or `mask = False`.

### 4.3 Preprocessing

`prepare_model(image_level=...)` in `Data/preprocessing.py`:

| Setting | Transform | Used for |
|---|---|---|
| `image_level=True` | Resize 256 → CenterCrop 224 → ImageNet normalise | whole frames (B1, B4) |
| `image_level=False` | Resize 224 → ImageNet normalise | player crops (B3, B5, B6) |

Player crops are already tight boxes, so they must not be centre-cropped again. A frozen backbone also has to receive exactly the transform it was trained with.

---

## 5. Baselines

All models use an ImageNet-pre-trained **ResNet-50** (2048-d features after global average pooling) instead of AlexNet fc7. Training uses AdamW, mixed precision, `CrossEntropyLoss`, and selects the checkpoint with the best validation accuracy.

### B1: Image Classification

```text
frame (3,224,224) ─► ResNet-50 (fine-tuned) ─► FC ─► 8 classes
```

| Setting | Value |
|---|---|
| Input | middle frame of the 9-frame window (index 4, frame `t-1`), whole image |
| Transform | `image_level=True` |
| Training | full fine-tuning, AdamW lr 1e-4, batch 16 |

### B3: Fine-tuned Person Classification

**Stage A: person action classifier**

```text
player crop (3,224,224) ─► ResNet-50 (fine-tuned) ─► FC ─► 9 actions
```

**Stage B: group classifier**

```text
12 player crops ─► ResNet-50 (stage A) ─► (12,2048) ─► max-pool over players ─► MLP ─► 8 classes
```

| Setting | Stage A | Stage B |
|---|---|---|
| Input | player crops, all 9 frames (`person_frames`) | 12 players of one frame (`person_grouped`) |
| Transform | `image_level=False` | `image_level=False` |
| Head | Linear 2048 → 9 | MLP 2048 → 4096 → 2048 → 8, dropout 0.2 |
| Training | AdamW lr 1e-4, 5 epochs | AdamW lr 1e-4, batch 16 |
| Backbone | fine-tuned | **fine-tuned end-to-end** (the paper keeps it frozen) |

> **Stage A was retrained.** The first stage-A model was trained on a single frame per clip (`t-5`, 25.7k crops) and selected at epoch 2 of a run that overfit right afterwards. It was strong on `standing` but weak on the actions that define group activities (recall: digging 0.12, jumping 0.00, setting 0.49, spiking 0.50). It was retrained on all 9 frames (231k crops), matching the paper, which fine-tunes the person CNN on full tracklets.

### B4: Temporal Model with Image Features

```text
9 frames ─► ResNet-50 (from B1) ─► (9,2048) ─► LSTM 512 ─► last hidden ─► MLP ─► 8 classes
```

| Setting | Value |
|---|---|
| Input | 9 full frames (`clip_frames`) |
| Backbone | initialised from the trained B1 model, trained end-to-end |
| Head | LSTM(2048 → 512) → Dropout → Linear 512 → 256 → ReLU → Dropout → Linear 256 → 8 |
| Training | AdamW lr 1e-4, weight decay 1e-4, batch 16 |

### B5: Temporal Model with Person Features

> **Definition used here.** This project follows the course specification: an LSTM runs over each player's track, then the player representations are combined. The paper's B5 instead pools fc7 features over people first and runs a single LSTM on the pooled sequence.

**Stage A: person temporal model**

```text
player track (9,3,224,224) ─► frozen B3 ResNet-50 ─► (9,2048) ─► LSTM 512 ─► last hidden ─► Linear ─► 9 actions
```

**Stage B: group classifier**

```text
12 tracks ─► frozen stage A ─► (12,512) ─► mask missing players ─► concat (6144) ─► MLP ─► 8 classes
```

| Setting | Stage A | Stage B |
|---|---|---|
| Input | 12 player tracks × 9 frames (`person_temporal`) | same |
| Frozen | B3 backbone | all of stage A |
| Head | Linear 512 → 9 | MLP 6144 → 2048 → 1024 → 512 → 8, LayerNorm, dropout 0.5 |
| Training | AdamW lr 3e-4, wd 1e-4, cosine schedule, batch 16 | AdamW lr 1e-4, wd 1e-4, cosine, class-balanced sampler, batch 16 |

Stage B concatenates the 12 player vectors in track-ID order instead of max-pooling them.

> B5 was trained on the **first** stage-A B3 backbone (single frame `t-5`). Its numbers will be re-run on the retrained backbone.

---

## 6. Results

Validation split (15 videos, 1,341 clips). Each row reports the epoch with the best validation accuracy.

### 6.1 Group activity (8 classes)

| Baseline | Val Accuracy | Val Macro-F1 | Best epoch |
|---|---|---|---|
| B1: Image Classification | 71.96% | 0.737 | 29 |
| B3: Fine-tuned Person Classification (stage B) | 75.47% | 0.728 | 16 |
| B4: Temporal Model with Image Features | **76.44%** | **0.780** | 33 |
| B5: Temporal Model with Person Features (stage B) | 64.65% | 0.611 | 2 |
| B6: Two-stage without LSTM 1 | 🚧 in progress | | |
| B7: Two-stage without LSTM 2 | 🚧 | | |
| B8: Two-stage with team pooling | 🚧 | | |

### 6.2 Person actions (9 classes)

| Model | Val Accuracy | Val Macro-F1 | Notes |
|---|---|---|---|
| B3 stage A, first run (frame `t-5` only) | 75.12% | 0.480 | epoch 2 of 12; validation loss rose after it |
| B3 stage A, retrained (all 9 frames) | ≈ 77% | TBD | backbone used from B6 onwards |
| B5 stage A (LSTM over each track) | 76.13% | 0.515 | epoch 1 of 11 |

### 6.3 Observations

- **Temporal context helps at the image level.** B4 improves on B1 by about 4.5 points.
- **Person crops help.** B3 stage B, with a single frame, already beats B1.
- **Person-action accuracy hides class imbalance.** Player crops are dominated by `standing`, so accuracy stays near 75% even when rare, decisive actions are poorly recognised. Macro-F1 is the more informative number.
- **B5 overfits quickly.** Training accuracy passes 80% while validation stalls around 62–65%. It was trained on the first, weaker B3 backbone.
- Accuracy is computed on the validation split that was also used to choose the checkpoint, so these numbers are optimistic. Final numbers will be reported on the test split.

---

## 7. Engineering Notes

These problems were found and fixed while auditing the pipeline. They are kept here because they apply to every later baseline.

| Issue | Effect | Resolution |
|---|---|---|
| Player crops passed through `image_level=True` while the frozen B3 backbone was trained with `image_level=False` | Features computed on centre-cropped, rescaled players that the backbone never saw in training | Crops always use `image_level=False` |
| B3 stage A trained on one frame per clip (`t-5`) | 9× less data; weak recognition of rare actions | Retrained on all 9 frames (`person_frames`) |
| `mode="person"` decodes a full 1280×720 frame for every single crop | Hours per epoch | `person_frames` decodes each frame once for all its players |
| Weighted sampler built on `scene_label` reused for person-action training | Balances the wrong label | No sampler for person-level training |

Verified on the real data: 9 frames per clip with the target at index 5, complete player tracks, correct masks, correct label mapping, a clean video-level split, and no train/validation leakage.

---

## 8. How to Run

### 8.1 Requirements

```text
python 3.10+
torch, torchvision
scikit-learn, torchmetrics
pyyaml, pillow, matplotlib, tqdm, tensorboard
```

### 8.2 Configure

Edit the dataset paths in `config.yaml`:

```yaml
Data:
  DATA_ROOT: "/kaggle/input/datasets/ahmedmohamed365/volleyball"
  PATHS:
    TRACKING_ANNOTATION_PATH: "volleyball_tracking_annotation/volleyball_tracking_annotation"
    VIDEOS_PATH: "volleyball_/videos"
```

### 8.3 Train

`main.py` builds every model and trainer; the experiment is chosen in its `__main__` block, together with the dataset `mode` that the baseline needs (see [Section 4.2](#42-dataset-modes)).

```python
if __name__ == "__main__":
    trainer_Baseline6.fit(train_loader, val_loader)
```

```bash
python main.py
```

Checkpoints are written to `/kaggle/working/` and TensorBoard logs to `runs/<experiment>`:

```bash
tensorboard --logdir runs
```

### 8.4 Audit the dataset

```bash
python audit_dataset.py
```

This prints split statistics (frames per clip, window offsets, players per frame, missing players, invalid boxes, label distributions) and saves visual checks of player slots over time.

---

## 9. Roadmap

- [x] B1: Image Classification
- [x] B3: Fine-tuned Person Classification
- [x] B4: Temporal Model with Image Features
- [x] B5: Temporal Model with Person Features
- [x] B3 stage A retrained on all 9 frames
- [ ] B6: Two-stage Model without LSTM 1
- [ ] B7: Two-stage Model without LSTM 2
- [ ] B8: Two-stage Model with team-level pooling
- [ ] Re-run B5 on the retrained B3 backbone
- [ ] Evaluate every baseline on the test split

<!--
Template for the next baselines (copy into Section 5 and fill in):

### B6: Two-stage Model without LSTM 1

```text
(9,12) crops ─► frozen B3 ─► (9,12,2048) ─► max-pool over players ─► (9,2048) ─► FC 3000 ─► LSTM 500 ─► classifier ─► 8
```

| Setting | Value |
|---|---|
| Input | `clip_frames_players` |
| Backbone | frozen B3 stage A (retrained) |
| Head | |
| Training | |

Then add a row to Section 6.1 and tick the box above.
-->

---

## 10. References

1. M. S. Ibrahim, S. Muralidharan, Z. Deng, A. Vahdat, G. Mori. *A Hierarchical Deep Temporal Model for Group Activity Recognition.* CVPR 2016.
2. M. S. Ibrahim et al. *Hierarchical Deep Temporal Models for Group Activity Recognition.* Journal extension, [arXiv:1607.02643](https://arxiv.org/abs/1607.02643) (adds team-level pooling).
3. [mostafa-saad/deep-activity-rec](https://github.com/mostafa-saad/deep-activity-rec): official implementation and dataset.
4. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
