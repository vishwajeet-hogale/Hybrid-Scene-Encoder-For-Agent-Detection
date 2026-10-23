# How this project was built

A LiDAR-based 3D object detector for autonomous driving, trained on nuScenes-mini.
This document walks through every step, from the raw data to the final demo video.
Each section starts with the short version and then goes deep.

---

## Part 1 — The big picture

**What the model does:** it takes one spin of the car's roof LiDAR (about 34,000 3D points) and outputs a
3D box for every vehicle, pedestrian and cyclist around the car.

Each box is 7 numbers:

| Numbers | Meaning |
|---|---|
| x, y, z | where the object's center is (meters, relative to our car) |
| w, l, h | how big the object is (width, length, height in meters) |
| yaw | which way the object is facing (angle around the vertical axis) |

**How it works, in one line per stage:**

```
LiDAR points ──► group into 0.5 m vertical columns ("pillars") ──► tiny PointNet per pillar
             ──► paint the pillar features onto a top-down image (bird's-eye view, BEV)
             ──► 2D CNN over that image
             ──► "heatmap" head: a peak at every object center  +  "regression" head: box size/height/heading
             ──► pick the peaks ──► 3D boxes
```

**The steps we went through:**

1. Get the data (nuScenes-mini) and understand its format.
2. Understand the coordinate frames (where is "x"?), because every point and box has to be in the same frame.
3. Turn raw data into training samples.
4. Build the model (v1).
5. Make it actually run (fix bugs, make it 11× faster).
6. Train it and measure it properly.
7. Visualize what it predicts (the demo video).
8. Diagnose the weaknesses and build an improved model (v2).
9. Compare v1 vs v2.
10. Find out what the improved model still gets wrong (overfitting, heading, false positives).
11. Add the cameras: LiDAR-camera fusion (BEVFusion-style).
12. Try to fix heading (orientation), and find out why it doesn't generalize.

---

## Part 2 — Step by step

### Step 1 — The data: nuScenes-mini

**Short version:** nuScenes is a self-driving dataset recorded in Boston and Singapore. "Mini" is a free
10-scene sample of it (~4 GB). Each scene is a 20-second drive with every sensor recorded and every object
labeled with a 3D box.

**Deep dive**

*What's on disk* (`nuscenes-mini/`):

| Folder | Contents | Size |
|---|---|---|
| `samples/` | Sensor data at **keyframes** (2 Hz) — the frames that have labels | 0.7 GB |
| `sweeps/` | Sensor data **between** keyframes (LiDAR at 20 Hz, cameras at 12 Hz) — no labels | 4.3 GB |
| `maps/` | Map rasters | 5.5 MB |
| `v1.0-mini/` | 13 JSON "tables" that describe everything | 32 MB |

The car has 12 sensors: 1 roof LiDAR (`LIDAR_TOP`), 6 cameras covering 360°, and 5 radars.

*The JSON tables are a relational database.* Every record has a `token` (a unique ID), and records point to
each other through `*_token` fields:

```
scene ──► sample (keyframe) ──► sample_data (one sensor reading: a .bin / .jpg file)
                │                     ├──► ego_pose          (where the car was at that moment)
                │                     └──► calibrated_sensor (where the sensor is mounted on the car) ──► sensor
                └──► sample_annotation (one 3D box) ──► instance (the same object across frames) ──► category
```

| Table | Count (mini) | One record is… |
|---|---|---|
| `scene` | 10 | a 20 s drive |
| `sample` | 404 | a labeled keyframe |
| `sample_data` | 31,206 | one sensor reading (file path, timestamp) |
| `ego_pose` | 31,206 | the car's pose in the world at one timestamp |
| `calibrated_sensor` | 120 | one sensor's mounting (12 sensors × 10 scenes) |
| `sensor` | 12 | a sensor slot (CAM_FRONT, LIDAR_TOP, …) |
| `sample_annotation` | 18,538 | a 3D box in one keyframe |
| `instance` | 911 | one physical object tracked over time |
| `category` | 23 | a class name (`vehicle.car`, `human.pedestrian.adult`, …) |
| `attribute` | 8 | a state (`vehicle.moving`, `cycle.with_rider`, …) |
| `visibility` | 4 | how visible the object is (0–40 %, …) |
| `log`, `map` | 8, 4 | recording session, map files |

*Checking the download:* we loaded the tables with the `nuscenes-devkit` and confirmed that all 31,206 files
referenced by `sample_data` exist on disk.

*Splits:* the devkit defines `mini_train` (8 scenes → 323 keyframes) and `mini_val` (2 scenes → 81 keyframes).
The validation scenes (`scene-0103`, `scene-0916`) are never trained on — that's what the demo video shows.

---

### Step 2 — Coordinate frames (the most important concept)

**Short version:** the same point has different coordinates depending on "relative to what". The LiDAR
reports points relative to itself, labels are stored relative to the world map, and the camera sees in its own
frame. Before training, everything must be put into one frame — we use the **ego frame** (relative to our car).

**Deep dive**

*The four frames:*

| Frame | Origin | Axes | Moves? |
|---|---|---|---|
| **Global (map)** | a fixed corner of the city map | x = east, y = north, z = up | never |
| **Ego (car)** | center of the rear axle, on the ground | x = forward, y = left, z = up | with the car |
| **LiDAR** | the LiDAR head (~1.84 m up) | x = right, y = forward, z = up (rotated −90° vs ego!) | fixed to car |
| **Camera** | the lens | x = right, y = down, z = forward (OpenCV convention) | fixed to car |

*Pose* = position + orientation ("where is it, which way is it facing"). Up to 6 numbers:
x, y, z + roll, pitch, yaw. A pose can be written as a 4×4 matrix (rotation in the top-left 3×3, translation in
the right column), and moving a point between frames is a matrix multiplication.

*Where the transforms come from:*

| Transform | Table | Changes over time? |
|---|---|---|
| sensor → ego ("extrinsics") | `calibrated_sensor` (`translation`, `rotation`) | no (fixed per scene) |
| ego → global ("ego pose") | `ego_pose` (`translation`, `rotation`) | yes — one per sensor reading |
| camera → pixels ("intrinsics") | `calibrated_sensor.camera_intrinsic` (3×3 K) | no |

Rotations are stored as **quaternions** `[w, x, y, z]` (4 numbers: `w = cos(θ/2)`, `(x,y,z) = sin(θ/2)·axis`).
They're compact and interpolate smoothly; for actually rotating points we convert them to 3×3 **rotation
matrices** (`Quaternion(q).rotation_matrix`). Example: CAM_FRONT's rotation matrix has columns
(0,−1,0), (0,0,−1), (1,0,0) — i.e. camera-x = ego-right, camera-y = ego-down, camera-z = ego-forward.

*Chains we use:*

```
LiDAR point → ego:        p_ego = T_lidar→ego · p_lidar

Label box → ego:          box_ego = inverse(T_ego→global at t_lidar) · box_global

Box (ego) → camera pixel: ego(t_lidar) ──T_ego→global(t_lidar)──► global
                                       ──inverse T_ego→global(t_cam)──► ego(t_cam)
                                       ──inverse T_cam→ego──► camera ──K, divide by depth──► (u, v)
```

The detour through **global** exists because the LiDAR and the camera fire at slightly different times and the
car moves in between (at 10 m/s, 25 ms = 25 cm). Global is the only frame that doesn't move.

*Two facts about nuScenes labels:* boxes are stored in the **global** frame, and their **pitch and roll are
exactly 0** (we checked all 18,538). That's why the model predicts a 4-DoF pose (x, y, z, yaw) + 3 sizes, not a
full 6-DoF pose — there's no ground truth for roll/pitch.

---

### Step 3 — Building a training sample

**Short version:** for each keyframe, load the LiDAR points, move them into the ego frame, load the labeled
boxes, move them into the ego frame too, and keep only the classes we care about.

**Deep dive** — `src/data/nuscenes_dataset.py`

1. **Points.** `load_lidar_points_ego()` reads the `.bin` file (x, y, z, intensity, ring index → we keep the
   first 4), builds `T_lidar→ego` from `calibrated_sensor`, and applies it.
2. **Boxes.** For every `sample_annotation`: build a devkit `Box` from translation / size (w, l, h) / rotation,
   subtract the ego translation and apply the inverse ego rotation (global → ego), then read the yaw.
3. **Labels.** nuScenes' 23 categories are mapped to 3 classes:

   | Class | nuScenes categories |
   |---|---|
   | 0 vehicle | `vehicle.*` (car, truck, bus, trailer, construction, emergency) |
   | 1 pedestrian | `human.pedestrian.*` |
   | 2 cyclist | `vehicle.bicycle`, `vehicle.motorcycle` |

   Everything else (animals, cones, barriers, debris) is dropped.
4. **Batching.** Point clouds have different sizes, so `src/data/collate.py` keeps them as a Python list rather
   than stacking them.
5. **Augmentation (added in v2, train split only)** — see Step 8.

`src/data/nuscenes_datamodule.py` wraps this in a PyTorch Lightning `DataModule` that builds the train/val
datasets from the split scene lists. Config lives in `configs/` and is loaded with Hydra.

---

### Step 4 — The model (v1)

**Short version:** a simplified **PointPillars** encoder (LiDAR → top-down feature image) followed by a
**CenterPoint-style** head (find object centers as heatmap peaks, then read box parameters at each peak).
About 0.56 M parameters.

**Deep dive**

*4a. Pillarization* (`src/models/lidar/pillars.py`). The area around the car — x from −10 to 70 m, y from −40 to
40 m — is divided into a 160 × 160 grid of 0.5 m cells. Each cell is a vertical column, a "pillar". Every point is
dropped into its pillar; we keep at most 6,000 non-empty pillars and 24 points per pillar (zero-padded), giving a
dense tensor `[P, 24, 4]` plus each pillar's grid coordinates.

*4b. Pillar Feature Net* (`pointpillars_encoder.py`). A tiny PointNet: the same small MLP is applied to every
point, then a max-pool over the points of each pillar → one 64-dim vector per pillar. Max-pooling makes it
independent of point order.

*4c. Scatter to BEV.* Each pillar vector is written back into its grid cell, producing a 64-channel 160 × 160
"pseudo-image" — a bird's-eye view of the scene that ordinary 2D convolutions can process.

*4d. BEV backbone.* v1: three 3×3 conv layers at full resolution → 128 channels.

*4e. CenterHead* (`src/models/heads/center_head.py`). Two small conv branches over the BEV features:

- **Heatmap** `[3, 160, 160]`: one channel per class; trained to output a peak of 1.0 at each object center.
  Its bias starts at −2.19 so initial predictions are ~0.1 (otherwise the loss explodes at the start).
- **Regression** `[8, 160, 160]`: at each center cell — `dx, dy` (sub-cell offset of the true center),
  `z`, `log w, log l, log h`, `sin yaw, cos yaw`. Yaw is encoded as sin/cos so −179° and +179° are close
  (a raw angle would jump by 358°).

*4f. Training targets* (`src/models/heads/targets.py`). For every GT box: draw a 2D Gaussian blob on its class's
heatmap at the center cell, and write the 8 regression values into that one cell (with a mask marking it as
positive).

*4g. Losses* (`losses.py`).
- Heatmap: CenterNet **focal loss** — positives are pushed to 1, and negatives near a center are penalized less
  (they're "almost right"), which handles the huge imbalance (a few dozen centers vs. 25,600 cells).
- Regression: **L1**, only at positive cells.

*4h. Decoding* (`src/vis/decode.py`). sigmoid → keep only cells that are the max of their 3×3 neighborhood
(peak NMS) → top-K → score threshold → convert offsets/log-sizes/sin-cos back into metric boxes in the ego
frame.

---

### Step 5 — Making it run

**Short version:** the code initially "worked" but trained ~25 s per batch (≈ 6 hours for 20 epochs) and had
two silent bugs. After fixes: 2.2 s per batch.

**Deep dive**

| Problem | Symptom | Fix |
|---|---|---|
| Wrong splits | Config used `train`/`val` (full-dataset scene lists); only 6/4 of them happen to be in mini | Use `mini_train` / `mini_val` |
| Cyclists never labeled | `vehicle.bicycle` matched the `vehicle.` rule first → class 2 had 0 examples | Check bicycle/motorcycle before the generic vehicle rule (now 403 cyclists in train) |
| Slow pillarization | A Python loop over every pillar (~2,000 per cloud × 8 per batch), 2.5 s per cloud on the GPU | Vectorized: sort points by pillar, compute each point's rank within its pillar, write all points in one indexed assignment. Verified bit-identical to the loop. |
| Slow epochs | DataLoader workers re-spawned every epoch, each re-copying the whole dataset index (macOS uses "spawn") | `persistent_workers: true` |

---

### Step 6 — Training v1 and measuring it

**Short version:** loss went down, but loss isn't the goal — detecting objects is. We wrote a proper
detection metric and found the model plateaued at epoch ~9 and then started overfitting.

**Deep dive**

*Metric* (`src/eval/center_ap.py`). nuScenes-style **Average Precision with center-distance matching**: a
prediction is correct if its center is within 2 m of an unmatched GT center of the same class; precision is
averaged over recall levels 0.1…1.0. mAP = mean over the 3 classes. Computed on all 81 val keyframes.

*v1 results:*

| Checkpoint | mAP | Vehicle AP | Pedestrian AP | Cyclist AP |
|---|---|---|---|---|
| epoch 3 | 0.093 | 0.187 | 0.092 | 0.000 |
| **epoch 9 (best)** | **0.123** | 0.280 | 0.088 | 0.000 |
| epoch 12 | 0.118 | 0.283 | 0.069 | 0.004 |

*Overfitting:* train loss kept falling (8.8 → 2.4) while val loss bottomed at ~5.2 around epoch 9 and rose
to 5.55 by epoch 19. Only 8 training scenes — the model starts memorizing them.

Note: these APs are pessimistic compared to official nuScenes numbers — we count every GT box in the grid,
including ones with zero LiDAR points, which the official evaluation filters out.

---

### Step 7 — Visualizing the predictions

**Short version:** `render_demo.py` runs the model on a held-out scene and renders a video with three
views: the front camera with 3D boxes projected into it, a bird's-eye view with boxes, and the raw heatmap.

**Deep dive**

- **10 Hz, not 2 Hz.** Labels exist only at 2 Hz keyframes, but the model doesn't need labels to predict, so we
  run it on every 2nd LiDAR sweep (20 Hz → 10 Hz) for smooth video.
- **Camera panel.** For each LiDAR sweep, pick the CAM_FRONT image with the closest timestamp. Each predicted
  box's 8 corners go through the ego(t_lidar) → global → ego(t_cam) → camera → pixel chain from Step 2.
  Boxes with any corner behind the camera are skipped.
- **BEV panel.** LiDAR points (brightness = height) with boxes, a heading tick, and the ego car.
- **Heatmap panel.** The raw sigmoid heatmap *before* peak selection, colored by the strongest class, √-scaled
  with a 0.05 noise floor. White rings mark peaks that became boxes. This shows *what the model is looking
  at* — e.g. faint "pedestrian" activations on walls and poles are typical single-sweep false positives.
- **Circle NMS.** A car sometimes produced 2–3 peaks a few cells apart → stacked duplicate boxes. CenterPoint's
  fix: keep the highest-scoring center and drop same-class centers within a class radius
  (vehicle 2 m, pedestrian 0.5 m, cyclist 1 m).
- **Output.** Frames → `ffmpeg` → H.264 MP4 + palette-optimized GIF. Full-scene GIFs are huge (~58 MB); for a
  website, an `<video autoplay loop muted playsinline>` MP4 looks identical at a fraction of the size.

---

### Step 8 — The improved model (v2)

**Short version:** six changes, each aimed at a weakness we measured: a backbone that sees more context,
richer point features, better-scaled regression targets, CenterPoint's Gaussian sizing, data augmentation
against overfitting, and training that keeps the best model by mAP. 2.58 M parameters.

**Deep dive**

| # | Change | Why |
|---|---|---|
| 1 | **Multi-scale backbone**: stages at stride 1/2/4 (64/128/256 channels, 3 convs each), each upsampled back to 160 × 160, concatenated, fused by a 1×1 conv | v1's receptive field was 9 cells ≈ 4.5 m — about one car length. Each output cell now sees ~39 cells ≈ 20 m of context. |
| 2 | **9-feature pillars**: each point gets x, y, z, intensity **+** offset from its pillar's point centroid (3) **+** offset from the pillar's cell center (2); BatchNorm in the PFN | With only absolute coordinates the network must learn local shape separately at every location. Relative offsets give shape directly (original PointPillars design). |
| 3 | **Normalized regression**: z and log-sizes are predicted relative to per-class means computed on `mini_train` (vehicle 2.08 × 5.08 × 1.86 m, pedestrian 0.69 × 0.74 × 1.76 m, cyclist 0.68 × 1.79 × 1.49 m); regression loss weight 0.25 | Raw z and log-size values are much larger than the 0–1 offsets and dominated the L1 loss (v1 reg loss 3.3 > heatmap loss). |
| 4 | **Size-adaptive Gaussian** (CenterNet radius formula, min radius 2 cells) instead of a fixed σ = 1.5 cells | Matches CenterPoint's nuScenes setup. At 0.5 m cells this makes blobs *sharper* for most objects (σ ≈ 0.83) and wider only for buses — so neighboring pedestrians no longer merge into one blob. (We initially expected it to enlarge car blobs; computing the radii showed the opposite.) |
| 5 | **Augmentation** on train only: random left-right flip, rotation ±22.5°, scale 0.95–1.05, applied to points and boxes together. No front-back flip, because the grid is asymmetric (−10…70 m). | Fights the overfitting from Step 6 by showing the network new geometry every epoch. |
| 6 | **Training setup**: AdamW + one-cycle LR (peak 1e-3), val mAP computed every epoch, checkpoint kept by **best val mAP** (plus `last.ckpt`), 24 epochs | v1 kept only the last checkpoint, so overfitting decided which model we got. |

*Verification before the long run:*

- **Target round-trip test:** build targets from real (augmented) GT boxes, feed them to the decoder as a
  "perfect" prediction → every in-grid box came back with 0.0000 error in center, z, size, yaw and class.
  This rules out sign/index/prior bugs in the target ↔ decode pair.
- **Augmentation test:** count LiDAR points inside each GT box before and after 5 random augmentations → the
  counts were identical (478 points), so points and boxes move together.
- **Speed:** 2.3 s per training step (same as v1 despite the 4.6× bigger model).

*The memory leak (and the fix).* The first v2 run grew to **21 GB of RAM** on a 16 GB Mac by epoch 9 and
spent its time swapping (~1 minute per batch). Measuring instead of guessing:

- 40 bare training steps showed **flat** memory → the training step itself was fine.
- What differed in the real run was validation, which decoded detections on the GPU with a *different number
  of boxes every batch*. PyTorch's Apple-GPU backend (MPS) compiles and caches kernels per tensor shape, so
  constantly changing shapes are a known way to grow memory.
- Fix — the standard PointPillars split between CPU and GPU:
  - **pillarization on CPU** (≈ 8 ms per cloud, faster than on the GPU), every sample **padded to a fixed
    3,000 pillars** (the maximum in the whole dataset is 2,807); the scatter to BEV sends padding to a
    discarded "trash" cell, so every GPU tensor has the same shape every step;
  - **target building and decoding on CPU** (small, variable-size work).
- Checks: the new encoder matches the old one to 3e-7 in eval mode (including when the pillar cap truncates);
  memory stayed at 200–600 MB through 6 test epochs with full validation; and steps got faster (1.5 s vs 2.3 s).
- A second MPS-only bug surfaced on the way: the BEV tensor came out of `permute().reshape()` as a strided
  view, and BatchNorm's backward pass on MPS fails on that layout. One `.contiguous()` fixed it.

The full 24-epoch run then took ~16 minutes with memory never above 400 MB.

---

### Step 9 — Results: v1 vs v2

Same validation set (81 keyframes from the 2 held-out scenes), same metric (center-distance AP, 2 m):

| | **v2 (best, epoch 20)** | v1 (best, epoch 9) | Change |
|---|---|---|---|
| **mAP** | **0.196** | 0.123 | **+59 %** |
| Vehicle AP | **0.430** | 0.280 | +54 % |
| Pedestrian AP | **0.149** | 0.088 | +69 % |
| Cyclist AP | 0.010 | 0.000 | still ≈ 0 |

*How training behaved:*

- **No more overfitting.** v1: train loss 2.4 vs val loss 5.5 at the end. v2: 2.7 vs 2.8 — augmentation works.
- **One-cycle shape.** mAP was noisy and roughly level with v1 up to epoch ~10 (peak learning rate), then
  climbed steadily as the learning rate decayed: 0.107 (ep 10) → 0.173 (ep 12) → 0.196 (ep 20).
- **Cyclists stay near zero:** 403 training examples, thin and sparse in single-sweep LiDAR.

*Choosing the score threshold for the demo* (precision / recall on val, center distance 2 m):

| Threshold | Vehicle P / R / F1 | Pedestrian P / R / F1 |
|---|---|---|
| 0.15 | 0.57 / 0.46 / **0.51** | 0.14 / 0.50 / 0.22 |
| 0.20 | **0.72** / 0.38 / 0.49 | 0.21 / 0.40 / 0.27 |
| 0.25 | 0.84 / 0.31 / 0.46 | 0.28 / 0.31 / **0.29** |
| 0.30 | 0.91 / 0.27 / 0.41 | **0.36** / 0.23 / 0.28 |
| 0.40 | 0.96 / 0.18 / 0.30 | 0.41 / 0.09 / 0.15 |
| 0.50 | 0.99 / 0.09 / 0.17 | 0.60 / 0.04 / 0.08 |

The final demo uses per-class thresholds — **vehicle 0.3, pedestrian/cyclist 0.4** — because false boxes look
worse in a video than a few missed ones (≈ 7 cars per frame at 91 % precision). A very high threshold such as
0.85 would show nothing: this model's highest score anywhere on val is 0.83 (vehicles) and 0.70 (pedestrians).
CenterNet heatmap peaks are not calibrated probabilities; with Gaussian targets and focal loss, peaks of a
well-trained model typically sit around 0.3–0.7.

---

### Step 10 — What the improved model still gets wrong

**Short version:** more training only memorizes the 8 training scenes; heading (yaw) is the weakest output; and
most false positives are thin vertical clutter (poles, walls, trees) that a single LiDAR sweep can't tell apart
from people.

**Deep dive**

*Longer training / fine-tuning → overfitting.* Fine-tuning the best model for 30 more epochs (weights only, fresh
one-cycle schedule with peak LR 3e-4, `train.py init_ckpt=...`) gave:

| | Train mAP (8 seen scenes) | Val mAP (2 unseen scenes) |
|---|---|---|
| 24-epoch model | 0.473 | 0.196 |
| + 30-epoch fine-tune | **0.669** | 0.199 |

Much better on the scenes it trains on, no better on unseen ones — the model is **data-limited**, not
training-limited. (A plain "resume with more epochs" doesn't work either: the one-cycle schedule had already
decayed to ~1e-6.)

*Heading.* For correctly-centered vehicles on val the mean yaw error is 77° (median 68°) — barely better than
chance (90°). Reading the yaw **at the exact ground-truth center cell** is just as bad (83°), so this is not
caused by peaks landing one cell off; the yaw regression itself is weak (even on training scenes, median 23°).
Likely reasons: yaw is 2 of 8 regression channels, the regression is weighted 0.25 and supervised at a single
cell per object, and front/back often look alike in sparse LiDAR. mAP never noticed, because center-distance
matching ignores orientation (the official nuScenes AOE metric measures it separately).

*False positives* (val, demo thresholds), classified by what is actually there using **all** nuScenes
annotations, including categories the model doesn't predict:

| Class | Precision | Main cause | Second |
|---|---|---|---|
| Vehicle | 0.72 | 87 % — nothing annotated (walls, poles, trees, signs) | 6 % — on pedestrians |
| Pedestrian | 0.36 | 71 % — nothing annotated | 21 % — on cars |
| Cyclist | 0.06 | 40 % — on pedestrians, 40 % — nothing annotated | 20 % — on cars |

Duplicates are only 0.3–2.4 %, so NMS works; the extra centers are real heatmap activations on clutter. A
threshold can't fix pedestrians: poles and people get similar scores.

*Next steps, by expected gain:* multi-sweep input (≈10× denser points: poles become clean cylinders, people show
limbs, and heading becomes visible); camera fusion (color and texture separate poles from people trivially);
more data (v1.0-trainval); and for heading specifically, supervising the regression on the 3×3 cells around each
center with a higher yaw weight.

*What would help next* (in rough order of expected gain):

1. **More data** — v1.0-trainval has 700 training scenes vs. 8 here; this is the biggest lever by far.
2. **Multi-sweep input** — stack ~10 past LiDAR sweeps (moved into the current ego frame) with a time channel:
   denser points for small/far objects (cyclists, pedestrians).
3. **IoU-based evaluation and the official nuScenes metrics** (mAP over 0.5/1/2/4 m, NDS), with GT boxes that
   contain zero LiDAR points filtered out as the official eval does.

---

### Step 11 — Camera fusion (BEVFusion-style)

**Short version:** a camera branch turns all six images into a bird's-eye-view feature map on the same 160 × 160
grid as the LiDAR, a small fuser merges the two, and the existing backbone and head run on the result. Image
features come from a frozen DINOv2 model; the depth needed to place them in 3D is guided by the LiDAR. It
consistently improves vehicle detection (+0.037 AP), but it did not fix the clutter false positives.

**Deep dive**

*Architecture* — the `encoders` dict and `fusion` slot of `HybridSceneEncoder` were built for this:

```
6 cameras ─► frozen DINOv2 ViT-S/14 (cached) ─► 18×32×384 per camera ─┐
                                                                       ├─► depth net (+ sparse LiDAR depth) ─► lift ─► splat ─► camera BEV [B,32,160,160] ─┐
LiDAR points projected into each camera ──────────────────────────────┘                                                                                 ├─► fuser ─► multi-scale backbone ─► CenterHead
LiDAR ─► pillars ─► PFN ─► scatter ─────────────────────────────────────────────────────────────────────────────────► LiDAR BEV  [B,64,160,160] ─┘
```

1. **Image features** (`src/data/camera.py`, `cache_camera_features.py`). Images are resized 1600×900 → 448×252
   and run through a **frozen** DINOv2 ViT-S/14, giving an 18 × 32 grid of 384-d features per camera (one cell
   = 50 × 50 original pixels). Frozen because 8 training scenes would make a trainable image backbone memorize
   them; computed once for all 2,424 keyframe images (1 GB, ~17 min) so training stays fast.
2. **Camera geometry.** For each camera: camera → ego(t_cam) → global → ego(t_lidar) — the same chain as in
   Step 2 — plus the intrinsics rescaled to feature-grid units. The training augmentation (flip/rotate/scale) is
   now also returned as a 4×4 matrix and applied to this transform, so both sensors stay aligned.
3. **LiDAR-guided depth** (the BEVDepth idea). LiDAR points are projected into each camera's feature grid
   (nearest depth per cell). The depth net sees that sparse depth as an input and is trained to predict it
   (cross-entropy over 48 depth bins from 1 to 61 m) — turning the hardest part of camera 3D (depth) into a
   mostly-supervised problem.
4. **Lift and splat** (`src/models/camera/lss_encoder.py`, Lift-Splat-Shoot). Each cell's feature is multiplied
   by its depth distribution, giving a feature at 48 points along the pixel's ray; those points are moved into
   the ego frame and summed into the BEV grid with one fixed-size `index_add` (same trick as the LiDAR scatter,
   so MPS shapes never change).
5. **Fuser** (`src/models/fusion.py`): `out = lidar_bev + f(concat(lidar_bev, camera_bev))`, with the last conv of
   `f` initialized to zero. A fusion model initialized from the LiDAR-only checkpoint therefore starts out
   computing **exactly** the LiDAR-only prediction and learns how much camera information to add.

*Checks before training:* augmentation matrix matches the augmented points (1e-5); LiDAR depth back-projected
through the camera geometry lands a median 1.3 % of its depth from a real LiDAR point (within the 2 % feature
cell); the splat matches an independent NumPy count exactly; the fused model initialized from LiDAR weights
gives bit-identical outputs; memory flat in a real run. The camera branch adds only 0.27 M trainable parameters.

*Training:* initialized from the best LiDAR model; new modules at LR 1e-3, the pretrained LiDAR path at 1e-4
(`model.pretrained_lr_scale=0.1`) — the first attempt with one LR for everything knocked the pretrained weights
out of their optimum (mAP fell to 0.115). A **control run** used the same fine-tune without cameras, so any gain
can be attributed to the cameras rather than extra training.

*Results* (val, averaged over the second half of training, 16 epochs, to beat 2-scene noise):

| | mAP | Vehicle AP | Pedestrian AP |
|---|---|---|---|
| **LiDAR + camera** | **0.194 ± 0.008** | **0.455 ± 0.008** | 0.116 ± 0.015 |
| LiDAR-only control | 0.181 ± 0.007 | 0.418 ± 0.013 | 0.119 ± 0.013 |

Best single epochs: fusion 0.204 (vehicle 0.467), control 0.195.

*What it did and didn't do:*

- ✅ **More cars found**: +0.037 vehicle AP, consistently; maximum vehicle recall 0.27 → 0.35.
- ❌ **No gain at high precision**: at matched precision (~0.92) recall is the same (0.25 vs 0.27) — the extra
  cars are low-confidence ones, so a high-threshold demo looks similar.
- ❌ **Clutter false positives not reduced**, pedestrians and heading unchanged.

Why: each camera feature cell is 50 px wide (≈ 0.8 m at 20 m) — coarser than a pole — and lifting spreads each
feature along its ray (visible as orange streaks along the road in the demo's heatmap panel). With 8 training
scenes the fuser mostly learned the strong, easy signal: more evidence of cars. The natural next step for clutter
is **PointPainting**: per-point labels from a segmentation model with an explicit "pole" class, read at each
LiDAR point's exact pixel.

```bash
.venv/bin/python cache_camera_features.py                      # once: DINOv2 features for all keyframe images
.venv/bin/python train.py use_camera=true init_ckpt=<lidar best.ckpt> model.lr=1e-3 model.pretrained_lr_scale=0.1
```

---

### Step 12 — Heading: what was tried, and why it doesn't generalize

**Short version:** three changes aimed at heading did nothing on unseen scenes. A diagnostic showed why: the
model predicts heading well on the scenes it trained on (median error under 10°) but not on new ones — it
memorizes headings instead of reading them from the point shape. That's a data problem, not a bug or a model-size
problem.

**Deep dive**

*What was tried* (all available as options in `configs/config.yaml`, off by default):

- `reg_neighbor_weight: 0.5` — supervise the box regression at the 3×3 cells around each center (9× more heading
  supervision; a peak one cell off still reads trained values). Neighbor dx/dy are relative to each cell, so every
  supervised cell decodes to the exact box (checked: max error 2e-7).
- `yaw_weight: 2.0` — double weight for the sin/cos channels in the L1 loss.
- `heading_weight: 0.5` — a direct heading loss `1 − cos(Δyaw)` on the normalized (sin, cos) prediction:
  0 = correct, 1 = 90° off, 2 = 180° off.
- A new validation metric, `val/yaw_err_<class>`: mean heading error of matched detections (a simple version of
  nuScenes' AOE), logged every epoch.

*Result* — fine-tuning the fusion model with and without these changes (24 epochs each):

| | Vehicle heading error (val) | mAP |
|---|---|---|
| Starting fusion model | 78° | 0.204 |
| With heading changes | 76–83° over training | ~0.19–0.20 |
| Control (no changes) | 75–82° over training | ~0.19–0.21 |

*Diagnosis* — same model, same metric, training vs held-out scenes, against trivial baselines:

| Matched cars | Model (mean / median) | "always 0°" | "closer of 0°/180°" |
|---|---|---|---|
| Training scenes | **26° / 9.8°** | 99° | 42° |
| Val scenes | **76° / 64°** | 88° | 42° |

So targets, augmentation and decoding are correct (the model learns heading where it has data), and capacity is
not the issue (more channels would only make memorizing easier). On unseen scenes the model is worse than the
trivial "along or against the road" guess. Fixes that address generalization: more data, multi-sweep input
(denser points show the car's shape), and full 360° rotation augmentation (today ±22.5°), so headings can't be
memorized per location.

*Loss reshaping didn't help either.* A further run switched the box regression to MSE (`model.reg_loss=mse`) and
normalized every loss term by its running average with weights summing to 1 (`model.loss_norm=true`; heatmap 0.29,
regression 0.43, heading 0.14, depth 0.14). On training scenes the regression and heading losses fell further
(heading 0.54 → 0.24), but on val it tied with the control (mAP 0.191 vs 0.191, vehicle heading 82° vs 79°), and
with the heatmap's smaller share vehicle scores dropped. Both options remain available; defaults stay L1 and fixed
weights. A practical lesson from this run: DataLoader workers each started their own BLAS thread pool, which under
memory pressure busy-waited inside a numpy matmul for half an hour — `train.py` now pins BLAS/OpenMP to one thread
per process (and steps got faster, 2.2 s → 1.4 s).

---

## Part 3 — Reproducing everything

```bash
# environment: Python 3.11 venv with torch, lightning, hydra-core, nuscenes-devkit, opencv, pillow, matplotlib
# data: nuscenes-mini/ in the repo root (v1.0-mini tables + samples/ sweeps/ maps/)

# train (Hydra config in configs/; outputs in outputs/lightning_logs/version_N/)
.venv/bin/python train.py

# long runs that survive closing the terminal: tmux + caffeinate (keeps the Mac awake on AC power)
tmux new -s train "caffeinate -is .venv/bin/python train.py"
#   detach: Ctrl-b d      re-attach: tmux attach -t train
#   closing the laptop lid still sleeps the Mac unless:  sudo pmset -a disablesleep 1   (undo with 0)

# render the demo for a held-out scene (full-scene MP4 + 8 s GIF; per-class score thresholds 0.2,0.3,0.3)
.venv/bin/python render_demo.py --ckpt outputs/lightning_logs/version_N/checkpoints/<best>.ckpt \
    --scene scene-0103 --out outputs/demo
```

---

## Glossary

| Term | Meaning |
|---|---|
| **BEV** | Bird's-eye view: a top-down 2D grid of the scene |
| **Pillar** | A vertical column of space over one BEV grid cell |
| **Ego** | Our own car |
| **Keyframe / sample** | A 2 Hz frame that has human labels |
| **Sweep** | An unlabeled sensor reading between keyframes |
| **Extrinsics** | Where a sensor is mounted on the car (sensor → ego transform) |
| **Intrinsics** | The camera's lens model (3D camera coordinates → pixels) |
| **Pose** | Position + orientation; 4-DoF = x, y, z, yaw; 6-DoF adds roll and pitch |
| **Quaternion** | A 4-number rotation representation `[w, x, y, z]` |
| **Heatmap / center** | Per-class score map; peaks mark object centers (anchor-free detection) |
| **Focal loss** | A loss that down-weights easy negatives to cope with class imbalance |
| **NMS** | Non-maximum suppression: removing duplicate detections of the same object |
| **AP / mAP** | Average precision per class / mean over classes |
| **Receptive field** | How much of the input one output cell can "see" |
| **Overfitting** | Getting better on training data while getting worse on unseen data |
