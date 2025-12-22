# Wand Calibration Algorithm Documentation

This document describes the wand calibration algorithm implemented in `wand_calibrator.py`.

## Overview

The wand calibrator performs multi-camera extrinsic and intrinsic calibration using a 2-point wand of known length (typically 10mm). The algorithm uses bundle adjustment to jointly optimize camera parameters and 3D point positions.

---

## Input Data Format

### CSV Format
```
frame,cam_id,pt_idx,x,y,radius,status
0,0,0,640.5,320.1,15.2,Raw
0,0,1,680.3,325.7,14.8,Raw
0,1,0,620.1,318.5,15.0,Raw
...
```

- `frame`: Frame ID (integer)
- `cam_id`: Camera ID (0, 1, 2, ...)
- `pt_idx`: Point index (0 or 1 for 2-point wand)
- `x, y`: Detected center coordinates in pixels
- `radius`: Detected circle radius (not used in calibration)
- `status`: "Raw" or "Filtered_Small"/"Filtered_Large"

### Valid Frame Requirements
A frame is valid for calibration if:
- Both wand points (pt_idx=0 and pt_idx=1) are detected
- At least 2 cameras observe the frame

---

## Camera Parameter Vector (11 parameters per camera)

```python
X[i*11 : i*11+11] = [r0, r1, r2, t0, t1, t2, f, cx, cy, k1, k2]
```

| Index | Parameter | Description | Typical Initial Value |
|-------|-----------|-------------|----------------------|
| 0-2   | rvec      | Rodrigues rotation vector | From 8-point algorithm |
| 3-5   | tvec      | Translation vector (mm) | From 8-point algorithm |
| 6     | f         | Focal length (pixels) | User-provided (e.g., 60000) |
| 7     | cx        | Principal point x (pixels) | image_width / 2 |
| 8     | cy        | Principal point y (pixels) | image_height / 2 |
| 9     | k1        | Radial distortion coef 1 | 0.0 |
| 10    | k2        | Radial distortion coef 2 | 0.0 |

---

## Algorithm Architecture

The calibration process is divided into three high-level **Phases**, each utilizing a shared **4-Stage Optimization Pipeline**.

### Phase 1: Primary Initialization
**Goal**: Establish a robust global coordinate system using the best available camera pair.
1.  **Pair Selection**: Automatically finds the camera pair with the most shared wand observations.
2.  **Geometric Initialization**: Uses OpenCV's 8-Point Algorithm (`findEssentialMat`, `recoverPose`) to estimate relative pose. Scales translation using the known wand length.
3.  **Primary Optimization**: Runs the **4-Stage Pipeline** (see below) on *only* the primary pair to refine their relationship and initial focal lengths.

### Phase 2: Secondary Initialization (Incremental)
**Goal**: Register remaining cameras into the global coordinate system one by one.
1.  **PnP Registration**: For each additional camera, `solvePnP` is used to estimate its pose relative to the 3D points initialized in Phase 1.
2.  **Per-Camera Optimization**:
    *   Ideally, we refine the single camera's intrinsics (f, k1) and extrinsics while keeping 3D points fixed.
    *   Includes logic to optimize Distortion Coefficients based on user selection (0, 1, or 2 coeffs).
    *   **Logic**: Minimizes reprojection error for this specific camera.

### Phase 3: Global Bundle Adjustment (BA)
**Goal**: Jointly optimize all cameras and points for maximum accuracy.
1.  **N-View Triangulation**: Re-calculates all 3D points using **SVD-based N-View Triangulation** (using all visible cameras, not just 2). This eliminates bias towards the primary pair.
2.  **Global Optimization**: Runs the **4-Stage Pipeline** on the full system (All Cameras + All Points).

---

## 4-Stage Optimization Pipeline

This pipeline is the core optimization engine used in Phase 1 and Phase 3. It applies constraints progressively to ensure convergence.

### Stage 1: Lock Intrinsics
*   **Goal**: Fix coarse geometric errors (Rotation/Translation) without distorting focal lengths.
*   **Constraints**:
    *   Focal length `f`: Locked to initial guess (±1.0 px).
    *   Principal point `cx, cy`: Locked to image center.
    *   Distortion `k`: Locked to 0.

### Stage 2: Refine All (Intrinsics Unlocked)
*   **Goal**: Allow focal lengths and positions to settle naturally.
*   **Constraints**:
    *   `f`: Unlocked (must be > 100).
    *   `cx, cy`: Still locked to center (to prevent drift).
    *   **Distortion**: Unlocked if requested by user.

### Stage 3: Triangulation Constraint (N-View)
*   **Goal**: Enforce strict geometric consistency.
*   **Method**: 3D points are **NOT** optimization variables. Instead, they are *computed* from camera parameters at every iteration using N-View SVD.
*   **Residual**:
    *   Minimizes `(Triangulated_Distance - Physical_Wand_Length)`.
    *   Minimizes Reprojection Error of the triangulated points.

### Stage 4: Fine-Tune
*   **Goal**: Final polish.
*   **Changes**:
    *   `cx, cy`: Released (allowed to drift ±50 px).
    *   Tolerances: Tightened (`ftol=1e-6`).

---

## Precalibration Check

A specialized 'fast-path' feature designed for data cleaning:
*   **Method**: Runs a single global optimization (similar to Stage 1/2) without the expensive multi-stage process.
*   **Purpose**: Quickly estimates 3D points to calculate Reprojection Errors and Wand Length Errors.
*   **Workflow**: User identifies frames with high errors -> Marks them for deletion -> Re-runs Precalibration to verify improvement.

---

## Technical Details

### N-View Triangulation (SVD)
Unlike standard `cv2.triangulatePoints` (which uses only 2 views), Phase 3 uses a custom SVD implementation to triangulate points seen by $N$ cameras ($N \ge 2$).
*   Constructs matrix $A$ where each row represents a projection equation $u_i \times (P_i X) = 0$.
*   Solves $AX = 0$ via SVD to find the 3D point $X$ that best satisfies all camera observations.
*   Result: More robust 3D points and uniform error distribution across all cameras.

---

## Post-Optimization Processing

### Coordinate System Alignment

After optimization, the world origin is shifted to the centroid of all 3D points:
```python
centroid = np.mean(points_3d, axis=0)

# Shift 3D points
points_3d_aligned = points_3d - centroid

# Update camera translations to compensate
for each camera:
    tvec_new = tvec_old - R @ centroid
```

This ensures:
- 3D points are centered around origin
- Camera positions reflect distance from the point cloud

---

## Output Format

Camera parameters are exported to text files:
```
# Camera Model: (PINHOLE/POLYNOMIAL)
PINHOLE
# Camera Calibration Error: 
None
# Pose Calibration Error: 
None
# Image Size: (n_row,n_col)
800,1280
# Camera Matrix: 
fx,0.0,cx
0.0,fy,cy
0.0,0.0,1.0
# Distortion Coefficients: 
k1,k2,0.0,0.0,0.0
# Rotation Vector: 
r0,r1,r2
# Rotation Matrix: 
R[0,0],R[0,1],R[0,2]
R[1,0],R[1,1],R[1,2]
R[2,0],R[2,1],R[2,2]
# Inverse of Rotation Matrix: 
R^T[...]
# Translation Vector: 
t0,t1,t2
# Inverse of Translation Vector: 
-R^T @ t
```

---

## Typical Performance

| Metric | Before Optimization | After Optimization |
|--------|--------------------|--------------------|
| Reprojection RMS | 6-10 px | 0.2-0.5 px |
| Wand Length RMS | 2-3 mm | 0.01-0.05 mm |
| Total Iterations | - | 50-100 |

---

## Key Dependencies

- `scipy.optimize.least_squares`: Core optimizer
- `cv2.findEssentialMat`: 8-point algorithm
- `cv2.recoverPose`: Essential matrix decomposition
- `cv2.triangulatePoints`: 3D point triangulation

---

## Related Files

- `wand_calibrator.py`: Core calibration algorithm
- `view.py`: GUI integration
- `debug_calibration_script.py`: Standalone test script

---

## Author Notes

This implementation is inspired by MATLAB easyWand5 but uses a different optimization approach:
- easyWand uses constrained optimization with hard wand length constraint
- This implementation uses weighted soft constraint with triangulation-based refinement
- The 4-stage approach provides stability and typically achieves comparable accuracy
