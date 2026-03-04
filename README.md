# Structure-from-motion
This repository is an implementation of a Structure from Motion Pipeline. This was given as a course assignment for COL780 - Computer Vision under Prof. Chetan Arora in IIT Delhi.

Task 5:

Perfect. Let’s do this cleanly and systematically.

You are now moving from *building SfM* → to *evaluating against a professional SfM system*.

We will follow a structured plan.

---

# ✅ TASK 5 — Complete Step-by-Step Plan (macOS M2 + COLMAP)

We will:

1. Install COLMAP
2. Extract frames
3. Run sparse reconstruction
4. Export model
5. Parse outputs in Python
6. Build descriptor → 3D map
7. Build KD-Tree
8. Compute metrics
9. Compare with your custom pipeline

---

# 🔷 STEP 1 — Install COLMAP

```bash
brew install colmap
```

Verify:

```bash
colmap -h
```

---

# 🔷 STEP 2 — Extract Frames

COLMAP works on images, not video.

If you used interval = 120 in your pipeline:

```bash
mkdir images

ffmpeg -i split_a_truck-004.mp4 \
       -vf "select=not(mod(n\,120))" \
       -vsync vfr images/frame_%05d.jpg
```

If ffmpeg missing:

```bash
brew install ffmpeg
```

---

# 🔷 STEP 3 — Compute Intrinsics

In Python:

```python
import cv2

img = cv2.imread("images/frame_00001.jpg")
H, W = img.shape[:2]

fx = 0.7 * W
fy = 0.7 * W
cx = W / 2
cy = H / 2

print(f"{fx},{fy},{cx},{cy}")
```

Save these values.

---

# 🔷 STEP 4 — Run COLMAP Feature Extraction

```bash
colmap feature_extractor \
    --database_path database.db \
    --image_path images \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_params "fx,fy,cx,cy"
```

Replace with actual numbers.

---

# 🔷 STEP 5 — Feature Matching

```bash
colmap exhaustive_matcher \
    --database_path database.db
```

Since few frames, exhaustive is fine.

---

# 🔷 STEP 6 — Run Sparse Mapping

```bash
mkdir sparse

colmap mapper \
    --database_path database.db \
    --image_path images \
    --output_path sparse \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 0
```

This disables intrinsic refinement for fairness.

---

# 🔷 STEP 7 — Convert Model to TXT

```bash
colmap model_converter \
    --input_path sparse/0 \
    --output_path sparse_txt \
    --output_type TXT
```

Now you have:

```
sparse_txt/
    cameras.txt
    images.txt
    points3D.txt
```

---

# 🔷 STEP 8 — What You Must Extract

From:

### cameras.txt

* Intrinsics

### images.txt

* Camera poses (quaternion + translation)

### points3D.txt

* 3D coordinates
* Track list (which image sees which 3D point)

---

# 🔥 IMPORTANT PART — Descriptor Extraction

COLMAP stores descriptors inside `database.db`.

You must:

1. Open database.db (SQLite)
2. Extract descriptors from table `descriptors`
3. Link them via keypoint IDs to 3D points using tracks

This gives:

```
descriptor → 3D point
```

mapping.

---

# 🔷 STEP 9 — Build KD-Tree

In Python:

```python
from sklearn.neighbors import KDTree

tree = KDTree(descriptor_matrix)
```

Each row corresponds to a 3D point index.

This is your localization map for Task 6.

---

# 🔷 STEP 10 — Compute Metrics (Same as Task 4)

For all 3 scenes:

### (a) Mean reprojection error

Use same function as before.

### (b) Chamfer distance

Zero-center and normalize both clouds:

```python
X = X - X.mean(axis=0)
X /= np.max(np.linalg.norm(X, axis=1))
```

Do same for ground truth.

Then compute symmetric nearest-neighbor distance.

### (c) Visualization

Plot:

* Custom reconstruction
* COLMAP reconstruction
* Ground truth

Different colors.

---

# 🔷 STEP 11 — Comparison Section

You must compare:

### 1️⃣ Pose Comparison

For same frame:

[
R_{diff} = R_{colmap} R_{custom}^T
]

Angle:

[
\theta = \cos^{-1}((trace(R_{diff}) - 1)/2)
]

Translation direction difference:

[
\cos^{-1} \left( \frac{t_1 \cdot t_2}{||t_1|| ||t_2||} \right)
]

---

### 2️⃣ Reprojection Error

Compare means.

---

### 3️⃣ Map Density

Number of 3D points.

---

# 🔥 Expected Outcome

COLMAP should:

* Have more 3D points
* Lower reprojection error
* Better Chamfer score
* More stable trajectory

---

# 🎯 Before We Proceed

Have you:

* Installed COLMAP?
* Extracted frames?
* Run sparse reconstruction?

Tell me where you are currently so I guide you next step precisely.
