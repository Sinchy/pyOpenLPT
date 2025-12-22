import cv2
import numpy as np

class GridDetector:
    def __init__(self):
        pass

    @staticmethod
    def detect(image_path, min_area=15, max_area=10000, min_circ=0.5, min_conv=0.7, min_inertia=0.4, blob_color=255, smart_fill=False):
        """
        Detects grid points in the given image.
        
        Args:
            image_path (str): Path to the image file.
            min_area (int): Minimum blob area.
            max_area (int): Maximum blob area.
            min_circ (float): Minimum circularity (0-1).
            min_conv (float): Minimum convexity (0-1).
            min_inertia (float): Minimum inertia ratio (0-1).
            blob_color (int): 255 for light blobs on dark, 0 for dark blobs on light.
            smart_fill (bool): If True, attempts to infer missing grid points based on lattice structure.
            
        Returns:
            tuple: (keypoints, vis_img)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # 1. Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 2. Setup Detector Parameters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = blob_color
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area
        params.filterByCircularity = True
        params.minCircularity = min_circ
        params.filterByConvexity = True
        params.minConvexity = min_conv
        params.filterByInertia = True
        params.minInertiaRatio = min_inertia

        # 3. Detect
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = list(detector.detect(enhanced))
        
        # 4. Smart Fill Logic
        inferred_keypoints = []
        if smart_fill and len(keypoints) > 4:
            inferred_keypoints = GridDetector._fill_missing_points(keypoints, img, blob_color)

        # 5. Visualization Image
        vis_img = img.copy()
        
        # Draw detected (Green)
        vis_img = cv2.drawKeypoints(vis_img, keypoints, np.array([]), (0, 255, 0), 
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                                  
        # Draw inferred (Blue)
        if inferred_keypoints:
             vis_img = cv2.drawKeypoints(vis_img, inferred_keypoints, np.array([]), (255, 0, 0), 
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
             keypoints.extend(inferred_keypoints)
        
        return keypoints, vis_img

    @staticmethod
    def _fill_missing_points(keypoints, img, blob_color):
        """
        Infers missing points in a regular grid using lattice basis vectors.
        Validates inferred points by checking pixel intensity against observed points.
        Returns ONLY the newly inferred keypoints.
        """
        if not keypoints:
            return []
            
        points = np.array([kp.pt for kp in keypoints])
        
        # 0. Intensity Stats of Valid Points
        # Convert to gray if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        valid_intensities = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]:
                valid_intensities.append(gray[y, x])
        
        if not valid_intensities:
            return []
            
        median_intensity = np.median(valid_intensities)
        intensity_thresh = 40 # Tolerance
        
        # 1. Find Nearest Neighbors relative spacing
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=5).fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # Collect all neighbor vectors
        vectors = []
        for i in range(len(points)):
            p1 = points[i]
            for j in range(1, 5): # skip self
                p2 = points[indices[i][j]]
                vec = p2 - p1
                vectors.append(vec)
        vectors = np.array(vectors)
        
        # 2. Cluster vectors to find two primary basis vectors
        # Simple clustering: magnitude and angle
        norms = np.linalg.norm(vectors, axis=1)
        median_dist = np.median(norms)
        
        # Filter vectors close to median distance (grid pitch)
        valid_mask = (norms > median_dist * 0.7) & (norms < median_dist * 1.3)
        valid_vectors = vectors[valid_mask]
        
        if len(valid_vectors) < 2:
            return []
            
        # Fix for Windows KMeans memory leak warning
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
            
        # K-Means to find 4 main directions (up, down, left, right) -> reduce to 2 basis
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=4, n_init=10).fit(valid_vectors)
        centers = kmeans.cluster_centers_
        
        # Select two basis vectors that are roughly orthogonal
        v1 = centers[0]
        best_v2 = None
        best_ortho = 0
        
        for i in range(1, 4):
            v = centers[i]
            # Check orthogonality (dot product close to 0)
            v1_u = v1 / np.linalg.norm(v1)
            v_u = v / np.linalg.norm(v)
            cross = np.abs(np.cross(v1_u, v_u)) # 2D cross product is scalar z-comp
            if cross > 0.5: # At least 30 deg apart
                 # Pick the one that forms expected grid axes
                 best_v2 = v
                 break
        
        if best_v2 is None:
            return [] # Could not determine grid
            
        v2 = best_v2
        
        # 3. Grid Indexing
        origin = points[0] # Pick arbitrary origin
        basis_matrix = np.column_stack((v1, v2))
        inv_basis = np.linalg.pinv(basis_matrix)
        
        grid_indices = []
        point_map = {} # (u, v) -> pt
        
        for pt in points:
            rel_pos = pt - origin
            uv = inv_basis @ rel_pos
            u, v = int(round(uv[0])), int(round(uv[1]))
            grid_indices.append((u, v))
            point_map[(u, v)] = pt
            
        # 4. Fill Gaps
        us = [i[0] for i in grid_indices]
        vs = [i[1] for i in grid_indices]
        min_u, max_u = min(us), max(us)
        min_v, max_v = min(vs), max(vs)
        
        new_kps = []
        avg_size = np.median([kp.size for kp in keypoints])
        
        h, w = gray.shape
        
        for u in range(min_u, max_u + 1):
            for v in range(min_v, max_v + 1):
                if (u, v) not in point_map:
                    # Predict position
                    pred_pos = origin + u * v1 + v * v2
                    px, py = int(pred_pos[0]), int(pred_pos[1])
                    
                    # Check bounds
                    if 10 < px < w - 10 and 10 < py < h - 10:
                        # Validation: Intensity Check
                        val = gray[py, px]
                        
                        # Check similarity to median valid intensity
                        # If bright blobs (255): value should be high (close to median)
                        # If dark blobs (0): value should be low (close to median)
                        # We use a simple difference threshold
                        if abs(int(val) - int(median_intensity)) < intensity_thresh:
                            kp = cv2.KeyPoint(x=float(px), y=float(py), size=float(avg_size))
                            new_kps.append(kp)
                        
        return new_kps

    @staticmethod
    def detect_template(image_path, template, threshold=0.7, smart_fill=False, center_offset=(0, 0), search_mask=None):
        """
        Detects grid points using Template Matching.
        Useful for oblique/distorted views where blobs are non-circular.
        
        Args:
            image_path (str): Path to image.
            template (np.array): Template image (BGR or Gray).
            threshold (float): Correlation threshold (0-1).
            smart_fill (bool): Whether to infer missing points.
            center_offset (tuple): (dx, dy) offset of the dot center from the template geometric center (scale 1.0).
            search_mask (np.arrayType): Optional uint8 mask (255=ROI). Points outside are ignored.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Convert to Gray
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
            
        if len(template.shape) == 3:
            templ_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            templ_gray = template
            
        w, h = templ_gray.shape[::-1]
        
        # Scales to search: 1.0 is standard. 
        # If we suspect a larger center point, we can add scales like 1.5, 2.0.
        scales = [1.0, 1.5, 1.8]
        
        all_keypoints = []
        
        for scale in scales:
            if scale == 1.0:
                t = templ_gray
            else:
                # Resize template
                nw = int(w * scale)
                nh = int(h * scale)
                t = cv2.resize(templ_gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
            
            # 1. Match Template
            if t.shape[0] > img_gray.shape[0] or t.shape[1] > img_gray.shape[1]:
                continue
                
            res = cv2.matchTemplate(img_gray, t, cv2.TM_CCOEFF_NORMED)
            
            # 2. Find Peaks
            k_size = min(t.shape[0], t.shape[1]) // 2
            if k_size % 2 == 0: k_size += 1
            if k_size < 3: k_size = 3
            
            from cv2 import dilate
            kernel = np.ones((k_size, k_size), np.uint8)
            dilated = dilate(res, kernel)
            
            local_max = (res == dilated) & (res > threshold)
            ys, xs = np.where(local_max)
            
            # Offset to center
            # Geometric center of the *scaled* template (use float division for accuracy)
            off_x = t.shape[1] / 2.0
            off_y = t.shape[0] / 2.0
            
            # Apply sub-pixel bias (scaled) - note: bias_x/y are already in the correct direction
            bias_x = center_offset[0] * scale
            bias_y = center_offset[1] * scale
            
            for x, y in zip(xs, ys):
                cx = x + off_x + bias_x
                cy = y + off_y + bias_y
                
                # Check Search ROI
                if search_mask is not None:
                    # Check integer coords
                    ix, iy = int(cx), int(cy)
                    if 0 <= iy < search_mask.shape[0] and 0 <= ix < search_mask.shape[1]:
                        if search_mask[iy, ix] == 0:
                            continue # Skip
                    else:
                        continue # Out of bounds
                        
                # KeyPoint size = current template scale size
                kp_size = (t.shape[1] + t.shape[0]) / 2.0
                kp = cv2.KeyPoint(x=float(cx), y=float(cy), size=float(kp_size))
                all_keypoints.append(kp)

        # Merge duplicates (NMS)
        keypoints = []
        if all_keypoints:
            pts = np.array([kp.pt for kp in all_keypoints])
            # Use a radius check. Radius ~ w/2
            min_dist = w / 2.0
            
            # Simple greedy NMS
            used = [False] * len(all_keypoints)
            for i in range(len(all_keypoints)):
                if used[i]: continue
                
                kp_main = all_keypoints[i]
                keypoints.append(kp_main)
                used[i] = True
                
                for j in range(i+1, len(all_keypoints)):
                    if used[j]: continue
                    kp_other = all_keypoints[j]
                    dist = np.hypot(kp_main.pt[0] - kp_other.pt[0], kp_main.pt[1] - kp_other.pt[1])
                    if dist < min_dist:
                        used[j] = True # Suppress neighbor
        
        # 3. Smart Fill Logic
        inferred_keypoints = []
        if smart_fill and len(keypoints) > 4:
            inferred_keypoints = GridDetector._fill_missing_points(keypoints, img, 0)    
            # 3. Smart Fill Logic
            # Merge smart fill with current
            pass
            
        # 4. Visualization
        vis_img = img.copy()
        
        # Custom Draw: Cross "+" with size = template width / 4 (Smaller as requested)
        marker_size = int(w / 4)
        if marker_size < 2: marker_size = 2
        
        # Helper to draw cross
        def draw_cross(img, pt, color, size):
            x, y = int(pt[0]), int(pt[1])
            cv2.line(img, (x - size, y), (x + size, y), color, 2)
            cv2.line(img, (x, y - size), (x, y + size), color, 2)
            
        # Draw detected (Green)
        for kp in keypoints:
            draw_cross(vis_img, kp.pt, (0, 255, 0), marker_size)
            
        # Draw inferred (Blue)
        for kp in inferred_keypoints:
             draw_cross(vis_img, kp.pt, (255, 0, 0), marker_size)
             keypoints.append(kp)
             
        return keypoints, vis_img
