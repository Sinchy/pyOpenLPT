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

## Algorithm Stages

### Stage 0: Geometric Initialization (8-Point Algorithm)

**Purpose**: Establish initial relative camera poses without prior knowledge of camera intrinsics.

**Input**: 
- User-provided initial focal length `f0`
- All valid wand point observations

**Process**:

1. **Camera Pair Selection**
   - Find the camera pair with the most common observations
   - This pair will be used as the base reference

2. **Point Collection for 8-Point Algorithm**
   ```python
   for each frame observed by both cameras:
       pts1.append([wand_point_A_in_cam1, wand_point_B_in_cam1])  # 2 points/frame
       pts2.append([wand_point_A_in_cam2, wand_point_B_in_cam2])
   ```
   - Uses BOTH wand points from each frame
   - Total points = num_frames × 2

3. **Essential Matrix Computation**
   ```python
   # Normalize pixel coordinates using initial focal length
   K = [[f0, 0, cx], [0, f0, cy], [0, 0, 1]]
   pts1_norm = cv2.undistortPoints(pts1, K, None)
   pts2_norm = cv2.undistortPoints(pts2, K, None)
   
   # Compute Essential Matrix (8-point algorithm with RANSAC)
   E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, 
                                   focal=1.0, pp=(0,0),
                                   method=cv2.RANSAC, 
                                   prob=0.999, 
                                   threshold=1e-3)
   ```

4. **Pose Recovery**
   ```python
   # Decompose E into R and t (4 possible solutions, pick correct one)
   _, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm)
   ```
   - Returns relative pose: Camera 2 relative to Camera 1
   - t is unit vector (scale unknown)

5. **Scale Recovery from Wand Length**
   ```python
   # Triangulate points with unit-scale t
   P1 = K @ [I | 0]
   P2 = K @ [R | t]
   points_3d = cv2.triangulatePoints(P1, P2, pts1, pts2)
   
   # Compute median wand length in current scale
   wand_lengths = [||pt_A - pt_B|| for each frame]
   median_length = np.median(wand_lengths)
   
   # Compute scale factor
   scale = wand_length_mm / median_length  # e.g., 10mm / 0.0105 = 952
   
   # Apply scale to translation
   t_scaled = t * scale
   ```

6. **Convert to Rodrigues Rotation**
   ```python
   rvec, _ = cv2.Rodrigues(R)
   ```

**Output**:
- Camera 0: rvec=[0,0,0], tvec=[0,0,0] (reference)
- Camera 1: rvec, tvec (relative to camera 0)
- Initial 3D points for all frames

---

### Stage 1: Lock Intrinsics, Optimize Geometry

**Purpose**: Refine geometric relationships while keeping intrinsic parameters fixed.

**Optimized Variables**: 
- Extrinsic parameters (rvec, tvec) for all cameras
- 3D point positions

**Fixed Variables**: f, cx, cy, k1, k2

**Bounds**:
```python
# Intrinsic parameters: locked to initial values ±epsilon
f:  [f0 - 1.0, f0 + 1.0]      # Almost fixed
cx: [cx0 - 0.5, cx0 + 0.5]    # Almost fixed
cy: [cy0 - 0.5, cy0 + 0.5]    # Almost fixed
k1: [-1e-4, 1e-4]             # Almost zero
k2: [-1e-6, 1e-6]             # Almost zero

# 3D point Z coordinates
Z:  [10.0, +inf]              # Points must be in front of camera
```

**Optimizer Settings**:
```python
ftol=1e-4, method='trf', loss='huber', f_scale=1.0
```

---

### Stage 2: Refine All Parameters (cx, cy Fixed at Center)

**Purpose**: Joint optimization of camera parameters and 3D points.

**Optimized Variables**: 
- All camera parameters (11 × N_cameras)
- All 3D points (3 × N_points)

**Key Design Decision**: cx, cy are fixed at image center
- Prevents parameter coupling issues
- Matches MATLAB easyWand approach
- Allows other parameters to compensate

**Bounds**:
```python
# Focal length
f:  [100.0, +inf]             # f > 100

# Principal point: FIXED at image center
cx: [width/2 - 0.01, width/2 + 0.01]   # Effectively fixed
cy: [height/2 - 0.01, height/2 + 0.01] # Effectively fixed

# 3D point Z coordinates
Z:  [10.0, +inf]
```

**Residual Function**:
```python
residuals = []
for each frame:
    # Wand length constraint
    dist = ||pt3d_A - pt3d_B||
    residuals.append((dist - wand_length) * WAND_WEIGHT)  # WAND_WEIGHT = 1.0
    
    # Reprojection error for each camera
    for each camera:
        projected_A = project(pt3d_A, cam_params)
        projected_B = project(pt3d_B, cam_params)
        residuals.extend(projected_A - observed_A)  # 2 values (x, y)
        residuals.extend(projected_B - observed_B)  # 2 values (x, y)
```

**Optimizer Settings**:
```python
ftol=1e-4, xtol=1e-4, gtol=1e-6
method='trf', loss='soft_l1', f_scale=1.0
jac_sparsity=A  # Sparse Jacobian for efficiency
```

**Sparse Jacobian Structure**:
- Exploits sparsity: each residual depends only on relevant camera params and point
- Significantly speeds up optimization for large datasets

---

### Stage 3: Triangulation-Based Refinement (cx, cy Fixed)

**Purpose**: Ensure geometric consistency by re-triangulating 3D points at each iteration.

**Key Difference from Stage 2**:
- 3D points are NOT optimization variables
- 3D points are COMPUTED from camera parameters via triangulation
- This enforces proper triangulation geometry

**Optimized Variables**: 
- Camera parameters only (11 × N_cameras)

**3D Points**: Re-triangulated at each iteration
```python
def _triangulate_frame(cam_params, observations):
    # Build projection matrices from current cam_params
    P1 = build_projection_matrix(cam_params[cam1])
    P2 = build_projection_matrix(cam_params[cam2])
    
    # Triangulate
    pt3d_A = cv2.triangulatePoints(P1, P2, obs_A_cam1, obs_A_cam2)
    pt3d_B = cv2.triangulatePoints(P1, P2, obs_B_cam1, obs_B_cam2)
    
    return pt3d_A, pt3d_B
```

**Residual Function**:
```python
def _residuals_triangulation(cam_params):
    residuals = []
    
    for each frame:
        # Re-triangulate with current camera params
        pt3d_A, pt3d_B = triangulate_frame(cam_params, observations)
        
        # Wand length constraint (STRONGER WEIGHT)
        dist = ||pt3d_A - pt3d_B||
        residuals.append((dist - wand_length) * 10.0)  # Weight = 10
        
        # Reprojection to ALL cameras
        for each camera:
            residuals.extend(project(pt3d_A, cam) - observed_A)
            residuals.extend(project(pt3d_B, cam) - observed_B)
    
    return residuals
```

**Bounds**:
```python
f:  [100.0, +inf]
cx: [width/2 - 0.01, width/2 + 0.01]   # Fixed at center
cy: [height/2 - 0.01, height/2 + 0.01] # Fixed at center
```

**Optimizer Settings**:
```python
ftol=1e-5, xtol=1e-5, gtol=1e-6
method='trf', loss='huber', f_scale=1.0
```

---

### Stage 4: Fine-Tune with Released cx, cy

**Purpose**: Allow small adjustments to principal point for final refinement.

**Optimized Variables**: 
- Camera parameters only (same as Stage 3)

**Key Change**: cx, cy bounds are relaxed
```python
cx: [width/2 - 50, width/2 + 50]   # Allow ±50 pixels
cy: [height/2 - 50, height/2 + 50] # Allow ±50 pixels
```

**Residual Function**: Same as Stage 3 (triangulation-based)

**Optimizer Settings**:
```python
ftol=1e-6, xtol=1e-6, gtol=1e-6  # Finest tolerance
method='trf', loss='huber', f_scale=1.0
```

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
