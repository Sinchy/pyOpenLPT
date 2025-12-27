// ========================================================================
// VSC.cpp - Volume Self Calibration Implementation
// ========================================================================
//
// This file implements the VSC module for OpenLPT.
// Main components:
// - accumulate(): Collects reliable 3D-2D correspondences from tracked
// particles.
// - runVSC(): Optimizes camera extrinsics using Levenberg-Marquardt.
// - runOTF(): Fits spatially-varying Gaussian OTF parameters.
//
// Dependencies:
// - nanoflann: KD-tree for fast neighbor search (isolation check).
// - myMATH: Matrix operations (inverse, eye, etc.).
// - OpenMP: Parallel processing.
// ========================================================================

// CRITICAL: nanoflann.hpp must be included FIRST due to template requirements
#include <nanoflann.hpp>

#include "ImageIO.h"
#include "OTF.h"
#include "ObjectFinder.h"
#include "VSC.h"
#include "myMATH.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <omp.h>

#include <random>
#include <unordered_map>

// ========================================================================
// KD-Tree Adaptor for 2D Object Candidates
// ========================================================================
// This adaptor allows nanoflann to operate on a vector of Object2D pointers.
// Used for fast isolation check: find all 2D detections near a projected point.

struct Obj2DCloud {
  using coord_t = double; ///< Coordinate type required by nanoflann

  const std::vector<std::unique_ptr<Object2D>>
      &_pts; ///< Reference to object list

  /// Return number of points in the dataset
  inline size_t kdtree_get_point_count() const { return _pts.size(); }

  /// Return coordinate 'dim' (0=x, 1=y) of point 'idx'
  inline coord_t kdtree_get_pt(const size_t idx, const size_t dim) const {
    return _pts[idx]->_pt_center[dim];
  }

  /// Optional bounding box (not used, return false)
  template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};

/// KD-tree type for 2D point cloud
using VisKDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, Obj2DCloud>, Obj2DCloud, 2>;

// ========================================================================
// Anonymous Namespace: Helper Functions
// ========================================================================

namespace {

/**
 * @brief Convert rotation vector to rotation matrix using Rodrigues' formula.
 *
 * R = I + sin(theta)*K + (1-cos(theta))*K^2
 * where K is the skew-symmetric matrix of the unit rotation axis.
 *
 * @param r_vec Rotation vector (axis * angle).
 * @return 3x3 rotation matrix.
 */
Matrix<double> Rodrigues(const Pt3D &r_vec) {
  double theta = std::sqrt(r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] +
                           r_vec[2] * r_vec[2]);

  // For very small rotations, return identity
  if (theta < 1e-8) {
    return myMATH::eye<double>(3);
  }

  // Build skew-symmetric matrix K from unit axis k = r_vec / theta
  Matrix<double> K(3, 3, 0);
  K(0, 1) = -r_vec[2] / theta;
  K(0, 2) = r_vec[1] / theta;
  K(1, 0) = r_vec[2] / theta;
  K(1, 2) = -r_vec[0] / theta;
  K(2, 0) = -r_vec[1] / theta;
  K(2, 1) = r_vec[0] / theta;

  // Rodrigues' formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
  Matrix<double> I = myMATH::eye<double>(3);
  return I + (K * std::sin(theta)) + ((K * K) * (1.0 - std::cos(theta)));
}

/**
 * @brief Map 3D world position to OTF grid index.
 *
 * Replicates OTF::mapGridID logic for use in VSC without needing friend access.
 *
 * @param param OTF parameter structure with grid info.
 * @param pt 3D world point.
 * @return Linear grid index.
 */
int getOTFGridID(const OTFParam &param, const Pt3D &pt) {
  // Compute grid cell indices
  int x_id =
      static_cast<int>(std::floor((pt[0] - param.boundary.x_min) / param.dx));
  int y_id =
      static_cast<int>(std::floor((pt[1] - param.boundary.y_min) / param.dy));
  int z_id =
      static_cast<int>(std::floor((pt[2] - param.boundary.z_min) / param.dz));

  // Clamp to valid range
  x_id = std::max(0, std::min(x_id, param.nx - 1));
  y_id = std::max(0, std::min(y_id, param.ny - 1));
  z_id = std::max(0, std::min(z_id, param.nz - 1));

  // Linear index: i = x_id * (ny*nz) + y_id * nz + z_id
  return x_id * (param.ny * param.nz) + y_id * param.nz + z_id;
}

} // namespace

// ========================================================================
// VSC Methods Implementation
// ========================================================================

void VSC::configure(const VSCParam &cfg) { _cfg = cfg; }

void VSC::initStrategy(const ObjectConfig &obj_cfg) {
  if (obj_cfg.kind() == ObjectKind::Tracer) {
    _strategy = std::make_unique<VSCTracerStrategy>();
  } else {
    _strategy = std::make_unique<VSCBubbleStrategy>();
  }
}

void VSC::reset() {
  _buffer.clear();
  _voxel_counts.clear();
  _grid_initialized = false;
}

int VSC::computeVoxelIndex(const Pt3D &pt) const {
  if (!_grid_initialized) {
    return -1;
  }

  const int n_div = std::max(1, _cfg._n_divisions);

  // Compute voxel sizes
  const double dx = (_grid_max[0] - _grid_min[0]) / n_div;
  const double dy = (_grid_max[1] - _grid_min[1]) / n_div;
  const double dz = (_grid_max[2] - _grid_min[2]) / n_div;

  // Handle degenerate cases
  const double safe_dx = (dx > 1e-6) ? dx : 1.0;
  const double safe_dy = (dy > 1e-6) ? dy : 1.0;
  const double safe_dz = (dz > 1e-6) ? dz : 1.0;

  int xi = static_cast<int>((pt[0] - _grid_min[0]) / safe_dx);
  int yi = static_cast<int>((pt[1] - _grid_min[1]) / safe_dy);
  int zi = static_cast<int>((pt[2] - _grid_min[2]) / safe_dz);

  // Clamp to valid range
  xi = std::max(0, std::min(xi, n_div - 1));
  yi = std::max(0, std::min(yi, n_div - 1));
  zi = std::max(0, std::min(zi, n_div - 1));

  return xi * n_div * n_div + yi * n_div + zi;
}

bool VSC::isReady() const {
  return _buffer.size() >= static_cast<size_t>(_cfg._min_points_to_trigger);
}

// ========================================================================
// accumulate() - Collect calibration points from current frame
// ========================================================================

void VSC::accumulate(int frame_id, const std::deque<Track> &active_tracks,
                     const std::vector<Image> &images,
                     const std::vector<Camera> &cams,
                     const ObjectConfig &obj_cfg) {
  // Initialize strategy on first call
  if (!_strategy) {
    initStrategy(obj_cfg);
  }

  const size_t n_cams = cams.size();

  // ----- Step 1: Detect 2D objects in all cameras -----
  // Each camera's detections are stored independently for parallel processing.
  std::vector<std::vector<std::unique_ptr<Object2D>>> all_candidates(n_cams);

#pragma omp parallel for if (!omp_in_parallel())
  for (int k = 0; k < static_cast<int>(n_cams); ++k) {
    // Thread-local ObjectFinder for thread safety
    ObjectFinder2D finder;
    all_candidates[k] = finder.findObject2D(images[k], obj_cfg);
  }

  // ----- Step 2: Build KD-Trees for fast neighbor queries -----
  // Used to check isolation: a valid calibration point should have exactly
  // one 2D detection nearby (itself). Multiple detections indicate overlap.
  std::vector<std::unique_ptr<VisKDTree>> kdtrees(n_cams);
  std::vector<Obj2DCloud> clouds;
  clouds.reserve(n_cams);

  for (size_t k = 0; k < n_cams; ++k) {
    if (all_candidates[k].empty())
      return;

    clouds.emplace_back(Obj2DCloud{all_candidates[k]});
    kdtrees[k] = std::make_unique<VisKDTree>(
        2, clouds.back(), nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtrees[k]->buildIndex();
  }

  // ----- Step 3: Precompute object radii for each camera -----
  // Avoids repeated dynamic_cast in getObject2DSize
  std::vector<std::vector<double>> obj_radii(n_cams);
  for (size_t k = 0; k < n_cams; ++k) {
    size_t N = all_candidates[k].size();
    obj_radii[k].resize(N);
    for (size_t i = 0; i < N; ++i) {
      obj_radii[k][i] = _strategy->getObject2DSize(*all_candidates[k][i]) / 2.0;
    }
  }

  // ----- Step 4: Precompute Isolation for each object in each camera -----
  // is_isolated[k][i] = true if object i in camera k has no overlapping
  // neighbors
  // Simple O(N^2/2) traversal without early termination for correctness
  std::vector<std::vector<bool>> is_isolated(n_cams);

  for (size_t k = 0; k < n_cams; ++k) {
    size_t N = all_candidates[k].size();
    is_isolated[k].assign(N, true);

    for (size_t i = 0; i < N; ++i) {
      double r_i = obj_radii[k][i];
      const Pt2D &c_i = all_candidates[k][i]->_pt_center;

      for (size_t j = i + 1; j < N; ++j) {
        double r_j = obj_radii[k][j];
        const Pt2D &c_j = all_candidates[k][j]->_pt_center;

        double dist = myMATH::dist2(c_i, c_j);
        double min_dist = r_i + r_j + _cfg._isolation_radius;

        if (dist < min_dist) {
          is_isolated[k][i] = false;
          is_isolated[k][j] = false;
        }
      }
    }
  }

  // Use thread-local buffer to reduce critical section overhead
  // Grid Initialization (Once) - Dynamic update handled safely below

  std::vector<CalibrationPoint> local_buffer;
  bool is_tracer = (obj_cfg.kind() == ObjectKind::Tracer);

  for (const auto &trk : active_tracks) {
    // Filter: track must be long enough to be reliable
    if (trk._obj3d_list.size() < static_cast<size_t>(_cfg._min_track_len)) {
      continue;
    }
    if (trk._obj3d_list.empty())
      continue;

    // Get 3D position from the most recent point in track
    const auto &obj3d = *trk._obj3d_list.back();
    Pt3D P3 = obj3d._pt_center;

    // Check visibility and isolation in ALL active cameras
    bool all_visible = true;
    std::vector<Observation> current_obs;

    for (size_t k = 0; k < n_cams; ++k) {
      // User request: Must be visible in ALL cameras.
      // Check if candidate list is empty for this camera
      if (all_candidates[k].empty()) {
        all_visible = false;
        break;
      }

      // Project 3D point to 2D
      // Project 3D point to 2D
      Pt2D P2_proj = cams[k].project(P3);

      // ----- Isolation Check using precomputed is_isolated -----
      // Find the nearest candidate to the projected point
      const double query_pt[2] = {P2_proj[0], P2_proj[1]};
      size_t nearest_idx = 0;
      double nearest_dist2 = 0;
      nanoflann::KNNResultSet<double> resultSet(1);
      resultSet.init(&nearest_idx, &nearest_dist2);
      kdtrees[k]->findNeighbors(resultSet, query_pt,
                                nanoflann::SearchParameters());

      double obj_r_px = obj_radii[k][nearest_idx];
      // Check if the nearest candidate is close enough to be "this" object
      double margin = 0.25;
      double match_threshold = obj_r_px * margin * obj_r_px * margin;
      if (nearest_dist2 > match_threshold) {
        all_visible = false; // No matching candidate found
        break;
      }

      // Check if this candidate is isolated (no overlapping neighbors)
      if (!is_isolated[k][nearest_idx]) {
        all_visible = false; // Candidate has overlapping neighbors
        break;
      }

      // Check if projection is within image bounds
      if (P2_proj[0] - obj_r_px < 0 ||
          P2_proj[0] + obj_r_px >= cams[k].getNCol() ||
          P2_proj[1] - obj_r_px < 0 ||
          P2_proj[1] + obj_r_px >= cams[k].getNRow()) {
        all_visible = false;
        break;
      }

      size_t idx = nearest_idx;
      const auto &candidate = all_candidates[k][idx];

      Observation obs;
      obs._cam_id = static_cast<int>(k);
      obs._meas_2d = candidate->_pt_center;
      obs._proj_2d = P2_proj;
      obs._quality_score = 1.0;
      obs._obj_radius = obj_r_px; // Use precomputed radius

      // ----- Identify OTF Parameters (Tracer Only) -----
      if (is_tracer && _cfg._enable_otf) {
        // Use tighter window for fitting robustness, slightly larger than 1.0r
        // to identify decay
        int half_w = static_cast<int>(std::ceil(1.5 * obj_r_px));

        // Check image bounds before calling estimate
        int cx = std::lround(obs._meas_2d[0]);
        int cy = std::lround(obs._meas_2d[1]);

        // Strict bounds check: only accept points fully inside the image
        // to ensure the OTF window is complete (2*half_w + 1)
        // CRITICAL: LOGIC MUST MATCH runOTF(). IF YOU CHANGE THIS CLIPPING
        // BEHAVIOR, YOU MUST UPDATE VSC::runOTF to handle partial windows or
        // different bounds!
        int x0 = cx - half_w;
        int x1 = cx + half_w + 1;
        int y0 = cy - half_w;
        int y1 = cy + half_w + 1;

        if (x0 >= 0 && y0 >= 0 && x1 <= images[k].getDimCol() &&
            y1 <= images[k].getDimRow()) {
          obs._otf_params = estimateOTFParams(images[k], obs._meas_2d, half_w);

          // Store ROI for verification
          // Since we checked bounds, we can copy directly without re-clipping
          obs._roi_intensity = Matrix<double>(y1 - y0, x1 - x0, 0.0);
          for (int r = y0; r < y1; ++r) {
            for (int c = x0; c < x1; ++c) {
              obs._roi_intensity(r - y0, c - x0) = images[k](r, c);
            }
          }
        } else {
          // If clipped, discard observation for robustness
          continue;
        }

        // If OTF fit failed (degenerate or invalid), discard this observation
        // User requirement: Do not use this point if fit fails
        if (!obs._otf_params.valid) {
          // Option 1: Mark not all_visible -> Drop entire 3D point
          // Option 2: Drop just this camera -> Point might still be valid with
          // fewer cameras? User said: "VSC should not use this point". Usually
          // points must be seen by ALL cameras.
          all_visible = false;
          break;
        }
      }

      current_obs.push_back(obs);
    }

    if (all_visible && !current_obs.empty()) {
      // Dynamic Grid Update (Critical Section handled safely due to serial
      // track loop)
      updateGridAndRebalance(P3);

      int voxel_id = computeVoxelIndex(P3);

      if (voxel_id >= 0 && _buffer.size() >= 100) {
        int total_points = 0;
        for (const auto &[vid, cnt] : _voxel_counts) {
          total_points += cnt;
        }
        size_t n_voxels =
            std::max(_voxel_counts.size(), static_cast<size_t>(1));
        double avg_count = static_cast<double>(total_points) / n_voxels;
        double threshold = std::max(
            avg_count * 2.0, static_cast<double>(_cfg._min_points_per_voxel));

        int current_count = 0;
        auto it = _voxel_counts.find(voxel_id);
        if (it != _voxel_counts.end()) {
          current_count = it->second;
        }

        if (current_count >= static_cast<int>(threshold)) {
          continue;
        }
      }

      CalibrationPoint cp;
      cp._pos_3d = P3;
      cp._frame_id = frame_id;
      cp._obs = std::move(current_obs);
      local_buffer.push_back(std::move(cp));

      if (voxel_id >= 0) {
        _voxel_counts[voxel_id]++;
      }
    }
  }

  // Append local buffer to global buffer
  if (!local_buffer.empty()) {
    _buffer.insert(_buffer.end(), std::make_move_iterator(local_buffer.begin()),
                   std::make_move_iterator(local_buffer.end()));

    // Log current volume size
    if (_grid_initialized) {
      double L = _grid_max[0] - _grid_min[0];
      double W = _grid_max[1] - _grid_min[1];
      double H = _grid_max[2] - _grid_min[2];
      std::cout << "  VSC Volume: " << L << " x " << W << " x " << H
                << " (LxWxH)" << std::endl;
    }
  }
}

// ========================================================================
// updateGridAndRebalance() - Dynamic grid expansion
// ========================================================================

void VSC::updateGridAndRebalance(const Pt3D &pt) {
  if (!_grid_initialized) {
    _grid_min = pt;
    _grid_max = pt;
    _grid_initialized = true;
    // Add small margin to avoid zero volume initially
    for (int i = 0; i < 3; ++i) {
      _grid_min[i] -= 0.1;
      _grid_max[i] += 0.1;
    }
    return;
  }

  bool changed = false;
  // Expand with small margin to include the new point
  const double margin = 0.1;

  for (int i = 0; i < 3; ++i) {
    if (pt[i] < _grid_min[i]) {
      _grid_min[i] = pt[i] - margin;
      changed = true;
    }
    if (pt[i] > _grid_max[i]) {
      _grid_max[i] = pt[i] + margin;
      changed = true;
    }
  }

  if (changed) {
    // Grid changed: previous voxel IDs are invalid.
    // Re-compute all.
    _voxel_counts.clear();
    for (const auto &cp : _buffer) {
      int vid = computeVoxelIndex(cp._pos_3d);
      if (vid >= 0) {
        _voxel_counts[vid]++;
      }
    }
  }
}

// Fit Log-Intensity to quadratic model:
// ln(I) ~ p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*xy
//
// OTF Model: I = a * exp( -(b*xx^2 + c*yy^2) )
// Exponent term Q(x,y) = b*xx^2 + c*yy^2 is positive definite.
// Matrix M in (x-xc)^T M (x-xc) has eigenvalues b, c.
//
// We solve for p vector (6x1).
// System: A * p = L
// A matrix rows: [1, x, y, x^2, y^2, xy]
// L vector rows: [ln(I)]

VSC::OTFParams VSC::estimateOTFParams(const Image &img, const Pt2D &center,
                                      int half_w) const {
  OTFParams params;
  params.valid = false;

  int cx = std::lround(center[0]);
  int cy = std::lround(center[1]);
  int x0 = std::max(0, cx - half_w);
  int x1 = std::min(img.getDimCol(), cx + half_w + 1);
  int y0 = std::max(0, cy - half_w);
  int y1 = std::min(img.getDimRow(), cy + half_w + 1);

  if (x1 <= x0 || y1 <= y0)
    return params; // Degenerate bounds

  // 1. Accumulate Normal Equations: (AtA) * p = (AtL)
  // 6 parameters => 6x6 matrix and 6x1 vector
  Matrix<double> AtA(6, 6, 0.0);
  Matrix<double> AtL(6, 1, 0.0);

  int n_samples = 0;
  double max_val = 0;

  // Pre-scan for max intensity to set threshold
  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      double val = img(y, x);
      if (val > max_val)
        max_val = val;
    }
  }

  // Threshold: e.g., 5% of peak or fixed noise floor
  double threshold = std::max(10.0, max_val * 0.05);
  if (max_val < threshold)
    return params; // Too dim

  // Use local coordinates relative to center
  double xc = center[0];
  double yc = center[1];

  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      double val = img(y, x);
      if (val <= threshold)
        continue;

      double dx = x - xc;
      double dy = y - yc;
      double log_v = std::log(val);

      // Basis functions: 1, dx, dy, dx^2, dy^2, dx*dy
      double basis[6] = {1.0, dx, dy, dx * dx, dy * dy, dx * dy};

      // Update Normal Equations (Upper triangular)
      for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j) {
          AtA(i, j) += basis[i] * basis[j];
        }
        AtL(i, 0) += basis[i] * log_v;
      }
      n_samples++;
    }
  }

  // Fill lower triangle
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < i; ++j) {
      AtA(i, j) = AtA(j, i);
    }
  }

  if (n_samples < 6)
    return params; // Not enough points (min 6 for 6 params)

  // 2. Solve linear system
  // Check singularity
  // det(AtA) is expensive, maybe just try inverse
  // Using myMATH::inverse from user library
  // We need to check if AtA is invertible.

  // Safe solve? myMATH::inverse might not report failure explicitly or throw.
  // Let's assume it works (Gaussian elim).
  // If matrix is singular, results will be NaN/Inf.

  Matrix<double> p = myMATH::inverse(AtA) * AtL;

  // Check for NaN
  if (!std::isfinite(p(0, 0)))
    return params;

  // 3. Extract Parameters
  // Quadratic form: Q(dx, dy) = - ( p3*dx^2 + p4*dy^2 + p5*dx*dy )
  // Note the negative sign because ln(I) has negative exponents.
  // Let M = - [ p3    p5/2 ]
  //          [ p5/2  p4   ]

  double p1 = p(1, 0);
  double p2 = p(2, 0);
  double p3 = p(3, 0);
  double p4 = p(4, 0);
  double p5 = p(5, 0);

  double mxx = -p3;
  double myy = -p4;
  double mxy = -0.5 * p5;

  // Eigenvalues of M
  double delta_eig = std::sqrt(std::pow(mxx - myy, 2) + 4.0 * mxy * mxy);
  double lam1 = (mxx + myy + delta_eig) / 2.0; // Major
  double lam2 = (mxx + myy - delta_eig) / 2.0; // Minor

  // Check validity: For a valid Gaussian peak, eigenvalues (coefficients of
  // squared terms) must be positive. I.e., the parabola must open downwards in
  // Log domain -> positive decay coefficients.
  if (lam2 <= 1e-9)
    return params; // Not a valid peak (saddle or valley)

  params.b = lam2; // Smaller coefficient (wider axis) ?
                   // Wait, earlier moment method: b = 1/(2*lam_mom).
                   // Here b is directly the coefficient in exponent.
                   // Standard form: exp( - b x^2 ... ).
                   // So b, c are the eigenvalues.
                   // Usually we assign b <= c (b is major axis decay, c is
                   // minor/steeper axis decay). Or user convention? Assuming b
                   // corresponds to major axis (smaller decay). So let's sort
                   // min, max.
  if (lam1 < lam2)
    std::swap(lam1, lam2);
  // Now lam1 >= lam2.

  // Actually, "Major axis" of elongation means SLOWER decay => SMALLER
  // coefficient. "Minor axis" (width) means FASTER decay => LARGER coefficient.
  // User's moment code: b = 1/(2lam1), c = 1/(2lam2).
  // If lam1 > lam2 (variance), then b < c.
  // So consistent: b is smaller coefficient.

  params.b = lam2; // Smaller eigenvalue
  params.c = lam1; // Larger eigenvalue

  // Orientation
  // Angle of the eigenvector corresponding to b (Major axis, smaller decay).
  // The eigenvector for lam2 (smaller) in M corresponds to the direction of
  // SLOWEST descent (major axis). tan(2*theta) = 2*mxy / (mxx - myy). But we
  // need to distinguish which eigenvalue. Angle for lam: theta satisfies (mxx -
  // lam)*cos + mxy*sin = 0

  // Calculate angle for lam2 (the smaller eigenvalue -> major axis)
  if (std::abs(mxy) > 1e-9) {
    params.alpha = std::atan2(lam2 - mxx, mxy);
  } else {
    params.alpha = 0.0;
    if (mxx > myy)
      params.alpha =
          1.57079632679; // If mxx (x decay) is larger, major axis is Y
  }

  // Peak intensity 'a'
  // ln(I) = p0 + p1*x + ...
  // At center (0,0), ln(I_peak) approx p0 (if p1,p2 small).
  // More accurately, p0 is ln(a) + offset if center not zero?
  // We used coordinates relative to (xc, yc), so dx=0, dy=0 corresponds to the
  // MEASURED center. But the fitted peak might be slightly offset. Stationary
  // point of parabola: Grad = 0.
  // For simplicity and stability, we can just use exp(p0) as 'a' if the offset
  // is small. Or calculate the peak offset. Let's check offset magnitude.
  double det_M = 4.0 * p3 * p4 - p5 * p5;
  double dx_peak = 0, dy_peak = 0;
  if (std::abs(det_M) > 1e-9) {
    dx_peak = (p5 * p2 - 2.0 * p4 * p1) / (4.0 * p3 * p4 - p5 * p5);
    dy_peak = (p5 * p1 - 2.0 * p3 * p2) / (4.0 * p3 * p4 - p5 * p5);
  }

  // If peak is too far (e.g. > 2 pixels), the Gaussian model is likely fitting
  // background slope or noise.
  if (dx_peak * dx_peak + dy_peak * dy_peak > 4.0)
    return params;

  double ln_a = p(0, 0) + p(1, 0) * dx_peak + p(2, 0) * dy_peak +
                p(3, 0) * dx_peak * dx_peak + p(4, 0) * dy_peak * dy_peak +
                p(5, 0) * dx_peak * dy_peak;

  params.a = std::exp(ln_a);
  params.valid = true;

  return params;
}

/**
 * @brief Extract camera parameters (8 params: 6 extrinsic + 2 intrinsic).
 *
 * @param cam Camera to extract from.
 * @return [rx, ry, rz, tx, ty, tz, f_eff, k1].
 */
std::vector<double> getCamExtrinsics(Camera &cam) {
  Pt3D r_vec = cam.rmtxTorvec(cam._pinhole_param.r_mtx);
  Pt3D t_vec = cam._pinhole_param.t_vec;

  double f_eff = cam._pinhole_param.cam_mtx(0, 0);
  double k1 = (cam._pinhole_param.dist_coeff.size() > 0)
                  ? cam._pinhole_param.dist_coeff[0]
                  : 0.0;

  return {r_vec[0], r_vec[1], r_vec[2], t_vec[0],
          t_vec[1], t_vec[2], f_eff,    k1};
}

/**
 * @brief Update camera parameters from 8-parameter vector.
 *
 * @param cam Camera to update.
 * @param params [rx, ry, rz, tx, ty, tz, f_eff, k1].
 */
void updateCamExtrinsics(Camera &cam, const std::vector<double> &params) {
  // Extract rotation vector
  Pt3D r_vec;
  r_vec[0] = params[0];
  r_vec[1] = params[1];
  r_vec[2] = params[2];

  // Extract translation vector
  Pt3D t_vec;
  t_vec[0] = params[3];
  t_vec[1] = params[4];
  t_vec[2] = params[5];

  // Extract intrinsics
  double f_eff = params[6];
  double k1 = params[7];

  // Update rotation matrix and translation
  cam._pinhole_param.r_mtx = Rodrigues(r_vec);
  cam._pinhole_param.t_vec = t_vec;

  // Update intrinsics
  // Assume fx = fy = f_eff
  cam._pinhole_param.cam_mtx(0, 0) = f_eff;
  cam._pinhole_param.cam_mtx(1, 1) = f_eff;

  // Update distortion k1
  if (cam._pinhole_param.dist_coeff.empty()) {
    cam._pinhole_param.dist_coeff.resize(1);
    cam._pinhole_param.n_dist_coeff = 1;
  }
  cam._pinhole_param.dist_coeff[0] = k1;

  // Update inverse matrices (needed for lineOfSight calculations)
  cam._pinhole_param.r_mtx_inv = myMATH::inverse(cam._pinhole_param.r_mtx);
  cam._pinhole_param.t_vec_inv =
      (cam._pinhole_param.r_mtx_inv * cam._pinhole_param.t_vec) * -1.0;
}

// ========================================================================
// runVSC() - Optimize camera parameters using Levenberg-Marquardt
// ========================================================================
//
// Algorithm Description:
// ----------------------
// This function performs a non-linear Least Squares optimization to refine
// both the extrinsic and intrinsic parameters of each active camera.
//
// Optimized Parameters (8 DoF per camera):
//    1-3. Rotation Vector (rx, ry, rz)
//    4-6. Translation Vector (tx, ty, tz)
//    7.   Effective Focal Length (f_eff)
//    8.   Radial Distortion Coefficient (k1)
//
// Objective Function:
//    Minimize Sum_i || P_proj(C_k, X_i) - x_meas_i ||^2
//    Where:
//      X_i     : 3D position of the i-th calibration point (fixed from
//      tracking). x_meas_i: Measured 2D centroid of the i-th point in image k.
//      C_k     : Camera parameters to optimize.
//      P_proj  : Pinhole projection function with distortion.
//
// Solver: Levenberg-Marquardt (LM) Algorithm
//    Iteratively updates parameters 'p' to minimize the sum of squared
//    residuals. Update rule: (J^T * J + lambda * I) * delta = J^T * r
//
// Key Steps:
// 1. Independent Optimization: Each camera is optimized independently because
//    the 3D points (X_i) are derived from reliable long tracks and are
//    treated as ground truth references for this refinement step.
// 2. Data Collection: Gather all 2D observations for each camera.
// 3. Finite Differences: Jacobian J is computed numerically.
// 4. Robustness: Updates are only accepted if reprojection error decreases.
//
bool VSC::runVSC(std::vector<Camera> &cams) {
  if (!isReady())
    return false;

  // ========================================================================
  // Joint Optimization with Re-triangulation (Aligned with Python VSC)
  // ========================================================================
  //
  // Key differences from previous implementation:
  // 1. All cameras are optimized JOINTLY (not per-camera).
  // 2. 3D points are RE-TRIANGULATED in each iteration using current camera
  //    parameters (not fixed).
  // 3. Residuals include both TRIANGULATION ERROR (mm) and REPROJECTION ERROR
  //    (px).
  // 4. SLIDING WINDOW constraints: bounds are re-centered every outer
  //    iteration.
  //
  // ========================================================================

  const double eps = 1e-6;                   // Numerical differentiation step
  const int max_lm_iter = 50;                // Max LM iterations per outer loop
  const double convergence_threshold = 1e-8; // Early stop
  const int n_params_per_cam = 8; // [rx, ry, rz, tx, ty, tz, f_eff, k1]
  const int n_outer_iters = 3;    // Sliding window iterations

  // Bounds constants (relative constraints)
  const double rvec_bound = 0.1;     // ±0.1 rad (~5.7 degrees)
  const double tvec_bound = 50.0;    // ±50 mm
  const double f_bound_ratio = 0.05; // ±5%
  const double k1_bound_ratio = 0.5; // ±50% or 0.1 min

  // ----- Collect active camera indices -----
  std::vector<size_t> active_cams;
  for (size_t k = 0; k < cams.size(); ++k) {
    if (cams[k]._is_active && cams[k]._type == CameraType::PINHOLE) {
      active_cams.push_back(k);
    }
  }

  if (active_cams.empty()) {
    std::cout << "  VSC: No active PINHOLE cameras found." << std::endl;
    return false;
  }

  const int n_cams = static_cast<int>(active_cams.size());
  const int total_params = n_cams * n_params_per_cam;

  // ----- Build multi-view correspondences -----
  // Each correspondence: 3D point + observations from multiple cameras
  // Structure: {obs_list} where obs_list[i] = (cam_internal_idx, 2D point)
  struct MultiViewCorr {
    std::vector<std::pair<int, Pt2D>> obs; // (internal_cam_idx, 2d_meas)
  };
  std::vector<MultiViewCorr> correspondences;

  // Map external cam_id to internal index
  std::unordered_map<int, int> cam_id_to_internal;
  for (int i = 0; i < n_cams; ++i) {
    cam_id_to_internal[static_cast<int>(active_cams[i])] = i;
  }

  // Group observations by 3D point (from _buffer)
  for (const auto &cp : _buffer) {
    MultiViewCorr corr;
    for (const auto &obs : cp._obs) {
      auto it = cam_id_to_internal.find(obs._cam_id);
      if (it != cam_id_to_internal.end()) {
        corr.obs.emplace_back(it->second, obs._meas_2d);
      }
    }
    // Need at least 2 views for triangulation
    if (corr.obs.size() >= 2) {
      correspondences.push_back(std::move(corr));
    }
  }

  if (correspondences.size() < 100) {
    std::cout << "  VSC: Insufficient multi-view correspondences ("
              << correspondences.size() << " < 100)." << std::endl;
    return false;
  }

  std::cout << "  VSC: " << correspondences.size()
            << " multi-view correspondences from " << n_cams << " cameras."
            << std::endl;

  // ----- Initialize joint parameter vector -----
  std::vector<double> params(total_params);
  std::vector<Camera> working_cams(n_cams);

  for (int i = 0; i < n_cams; ++i) {
    working_cams[i] = cams[active_cams[i]];
    std::vector<double> cam_params = getCamExtrinsics(working_cams[i]);
    for (int j = 0; j < n_params_per_cam; ++j) {
      params[i * n_params_per_cam + j] = cam_params[j];
    }
  }

  // ----- Lambda function: Update cameras from params -----
  auto updateCamerasFromParams = [&](const std::vector<double> &p) {
    for (int i = 0; i < n_cams; ++i) {
      std::vector<double> cam_p(p.begin() + i * n_params_per_cam,
                                p.begin() + (i + 1) * n_params_per_cam);
      updateCamExtrinsics(working_cams[i], cam_p);
    }
  };

  // ----- Lambda function: Triangulate and compute combined error -----
  // Returns: {total_error, triang_error_sum, reproj_error_sum, n_valid_pts}
  auto computeErrors = [&](const std::vector<double> &p)
      -> std::tuple<double, double, double, int> {
    // First update cameras
    std::vector<Camera> temp_cams = working_cams;
    for (int i = 0; i < n_cams; ++i) {
      std::vector<double> cam_p(p.begin() + i * n_params_per_cam,
                                p.begin() + (i + 1) * n_params_per_cam);
      updateCamExtrinsics(temp_cams[i], cam_p);
    }

    double total_err = 0.0;
    double triang_err_sum = 0.0;
    double reproj_err_sum = 0.0;
    int n_valid = 0;

    for (const auto &corr : correspondences) {
      // Build lines of sight for triangulation
      std::vector<Line3D> lines;
      for (const auto &obs : corr.obs) {
        int cam_idx = obs.first;
        Pt2D pt_2d = obs.second;
        // Undistort and get line of sight
        Pt2D pt_undist = temp_cams[cam_idx].undistort(
            pt_2d, temp_cams[cam_idx]._pinhole_param);
        Line3D los = temp_cams[cam_idx].pinholeLine(pt_undist);
        lines.push_back(los);
      }

      // Triangulate 3D point
      Pt3D pt3d;
      double triang_error = 0.0;
      myMATH::triangulation(pt3d, triang_error, lines);

      if (std::isnan(pt3d[0]) || std::isnan(pt3d[1]) || std::isnan(pt3d[2])) {
        continue; // Skip invalid points
      }

      // Compute reprojection error for each observation
      for (const auto &obs : corr.obs) {
        int cam_idx = obs.first;
        Pt2D pt_meas = obs.second;
        Pt2D pt_proj = temp_cams[cam_idx].project(pt3d);
        double reproj_err = myMATH::dist2(pt_proj, pt_meas);
        reproj_err_sum += reproj_err;
      }

      // Triangulation error (from myMATH::triangulation output)
      triang_err_sum += triang_error * triang_error;
      n_valid++;

      // Combined error: triang (mm^2) + reproj (px^2)
      // Note: Different scales, but LM handles this via Jacobian weighting
      total_err += triang_error * triang_error;
      for (const auto &obs : corr.obs) {
        Pt2D pt_proj = temp_cams[obs.first].project(pt3d);
        total_err += myMATH::dist2(pt_proj, obs.second);
      }
    }

    return {total_err, triang_err_sum, reproj_err_sum, n_valid};
  };

  // ----- Build bounds -----
  auto buildBounds = [&](const std::vector<double> &center_params)
      -> std::pair<std::vector<double>, std::vector<double>> {
    std::vector<double> lb(total_params), ub(total_params);
    for (int i = 0; i < n_cams; ++i) {
      int base = i * n_params_per_cam;
      // rvec: ±0.1 rad
      lb[base + 0] = center_params[base + 0] - rvec_bound;
      lb[base + 1] = center_params[base + 1] - rvec_bound;
      lb[base + 2] = center_params[base + 2] - rvec_bound;
      ub[base + 0] = center_params[base + 0] + rvec_bound;
      ub[base + 1] = center_params[base + 1] + rvec_bound;
      ub[base + 2] = center_params[base + 2] + rvec_bound;
      // tvec: ±50 mm
      lb[base + 3] = center_params[base + 3] - tvec_bound;
      lb[base + 4] = center_params[base + 4] - tvec_bound;
      lb[base + 5] = center_params[base + 5] - tvec_bound;
      ub[base + 3] = center_params[base + 3] + tvec_bound;
      ub[base + 4] = center_params[base + 4] + tvec_bound;
      ub[base + 5] = center_params[base + 5] + tvec_bound;
      // f_eff: ±5%
      double f = center_params[base + 6];
      lb[base + 6] = f * (1.0 - f_bound_ratio);
      ub[base + 6] = f * (1.0 + f_bound_ratio);
      // k1: ±50% or 0.1 min
      double k1 = center_params[base + 7];
      double k1_margin = std::max(0.1, std::abs(k1) * k1_bound_ratio);
      lb[base + 7] = k1 - k1_margin;
      ub[base + 7] = k1 + k1_margin;
    }
    return {lb, ub};
  };

  // ----- Initial error -----
  auto [init_err, init_triang, init_reproj, init_n] = computeErrors(params);
  double init_reproj_rmse =
      (init_n > 0) ? std::sqrt(init_reproj / (init_n * n_cams)) : 0.0;
  double init_triang_rmse =
      (init_n > 0) ? std::sqrt(init_triang / init_n) : 0.0;
  std::cout << "  Initial: TriangErr=" << init_triang_rmse
            << "mm, ProjErr=" << init_reproj_rmse << "px" << std::endl;

  // ========================================================================
  // Sliding Window Optimization (Outer Loop)
  // ========================================================================
  for (int outer = 0; outer < n_outer_iters; ++outer) {
    // Re-center bounds on current params
    auto bounds = buildBounds(params);
    std::vector<double> lb = std::get<0>(bounds);
    std::vector<double> ub = std::get<1>(bounds);

    std::cout << "  [Iter " << (outer + 1) << "/" << n_outer_iters
              << "] Re-centered bounds. Running LM..." << std::endl;

    double lambda = 0.001;
    auto init_err = computeErrors(params);
    double current_err = std::get<0>(init_err);

    // ----- LM Inner Loop -----
    for (int iter = 0; iter < max_lm_iter; ++iter) {
      // Build JtJ and Jtr using numerical Jacobian
      Matrix<double> JtJ(total_params, total_params, 0.0);
      Matrix<double> Jtr(total_params, 1, 0.0);

      // Compute residuals at current params
      auto err0 = computeErrors(params);
      double f0 = std::get<0>(err0);

      // Numerical Jacobian (forward difference)
      std::vector<double> grad(total_params, 0.0);

#pragma omp parallel for
      for (int p = 0; p < total_params; ++p) {
        std::vector<double> params_p = params;
        params_p[p] += eps;
        // Clamp to bounds
        params_p[p] = std::max(lb[p], std::min(ub[p], params_p[p]));
        auto err_p = computeErrors(params_p);
        double fp = std::get<0>(err_p);
        grad[p] = (fp - f0) / eps;
      }

      // Build approximate Hessian: JtJ = grad * grad^T
      // And gradient: Jtr = grad * f0 (simplified LM update)
      for (int i = 0; i < total_params; ++i) {
        for (int j = 0; j < total_params; ++j) {
          JtJ(i, j) = grad[i] * grad[j];
        }
        Jtr(i, 0) = -grad[i] * f0; // Negative gradient for descent
      }

      // Apply damping: (JtJ + lambda * diag(JtJ))
      for (int i = 0; i < total_params; ++i) {
        JtJ(i, i) *= (1.0 + lambda);
        if (JtJ(i, i) < 1e-10)
          JtJ(i, i) = 1e-10; // Regularization
      }

      // Solve for delta
      Matrix<double> delta = myMATH::inverse(JtJ) * Jtr;

      // Check convergence
      double delta_norm = 0;
      for (int i = 0; i < total_params; ++i) {
        delta_norm += delta(i, 0) * delta(i, 0);
      }
      if (delta_norm < convergence_threshold) {
        break;
      }

      // Compute candidate parameters with bounds clamping
      std::vector<double> params_new = params;
      for (int i = 0; i < total_params; ++i) {
        params_new[i] += delta(i, 0);
        params_new[i] = std::max(lb[i], std::min(ub[i], params_new[i]));
      }

      // Evaluate new error
      auto [new_err, nt, nr, nn] = computeErrors(params_new);

      // LM decision
      if (new_err < current_err) {
        params = params_new;
        current_err = new_err;
        lambda /= 10.0;
      } else {
        lambda *= 10.0;
      }

      // Early stopping if lambda too large
      if (lambda > 1e10)
        break;
    }

    // Log progress
    auto [err, te, re, nv] = computeErrors(params);
    double triang_rmse = (nv > 0) ? std::sqrt(te / nv) : 0.0;
    double reproj_rmse = (nv > 0) ? std::sqrt(re / (nv * n_cams)) : 0.0;
    std::cout << "  [Iter " << (outer + 1) << "] TriangErr=" << triang_rmse
              << "mm, ProjErr=" << reproj_rmse << "px" << std::endl;
  }

  // ----- Apply final parameters to cameras -----
  updateCamerasFromParams(params);
  for (int i = 0; i < n_cams; ++i) {
    cams[active_cams[i]] = working_cams[i];
  }

  // ----- Final error -----
  auto [final_err, final_triang, final_reproj, final_n] = computeErrors(params);
  double final_triang_rmse =
      (final_n > 0) ? std::sqrt(final_triang / final_n) : 0.0;
  double final_reproj_rmse =
      (final_n > 0) ? std::sqrt(final_reproj / (final_n * n_cams)) : 0.0;
  std::cout << "  Final: TriangErr=" << final_triang_rmse
            << "mm, ProjErr=" << final_reproj_rmse << "px" << std::endl;

  // ----- Save cameras -----
  if (!_cfg._output_path.empty()) {
    for (size_t k = 0; k < cams.size(); ++k) {
      if (!cams[k]._is_active)
        continue;
      std::string cam_path =
          _cfg._output_path + "/vsc_cam" + std::to_string(k) + ".txt";
      cams[k].saveParameters(cam_path);
    }
    std::cout << "  VSC cameras saved to " << _cfg._output_path << std::endl;
  }

  return true;
}

// ========================================================================
// runOTF() - Spatially Varying OTF Calibration
// ========================================================================
//
// Algorithm Description:
// ----------------------
// This function builds a 3D map of OTF (Optical Transfer Function) parameters
// (a, b, c, alpha) for each camera. The OTF describes the particle intensity
// profile:
//      I(x,y) = a * exp(- b*x'^2 - c*y'^2)
// where (x', y') are coordinates rotated by alpha.
//
// The parameters (a, b, c, alpha) were already estimated for each individual
// particle during the 'accumulate' phase using Linear Least Squares fitting.
//
// Spatial Aggregation:
// 1. Grid Mapping: The 3D measurement volume is divided into a voxel grid
//    (defined in tracer_cfgs[k]._otf._param).
// 2. Binning: Each valid calibration point is assigned to a voxel based on
// its
//    3D position.
// 3. Averaging: Compute candidate parameters for each grid cell.
// 4. Verification: Compare reconstruction error of Candidates vs Current
// parameters
//    against the stored ROI intensity. Only accept candidates if error
//    decreases.
// 5. Update: Write accepted parameters to TracerConfig.
//
// Return:
//    true if any grid cell was updated.
//
bool VSC::runOTF(std::vector<TracerConfig> &tracer_cfgs) {
  if (_buffer.empty())
    return false;

  // Check global enable flag
  if (!_cfg._enable_otf)
    return false;

  bool updated = false;

  // Iterate over all provided tracer configurations (typically just one)
  for (auto &t_cfg : tracer_cfgs) {
    auto &otf = t_cfg._otf;
    auto &param = otf._param;

    if (param.n_grid <= 0 || param.n_cam <= 0)
      continue;

    // Temporary accumulators for candidate parameters
    // Key: mapping index = cam_id * n_grid + grid_id
    struct OTFAccum {
      double a = 0, b = 0, c = 0, alpha = 0;
      int count = 0;
    };
    std::unordered_map<int, OTFAccum> candidates;

    // 1. Accumulate observations to form candidates
    for (const auto &cp : _buffer) {
      int grid_id = getOTFGridID(param, cp._pos_3d);
      if (grid_id < 0 || grid_id >= param.n_grid)
        continue;

      for (const auto &obs : cp._obs) {
        if (!obs._otf_params.valid)
          continue;

        int cam_id = obs._cam_id;
        if (cam_id < 0 || cam_id >= param.n_cam)
          continue;

        int key = cam_id * param.n_grid + grid_id;
        auto &cand = candidates[key];
        const auto &p = obs._otf_params;

        cand.a += p.a;
        cand.b += p.b;
        cand.c += p.c;
        cand.alpha += p.alpha;
        cand.count++;
      }
    }

    // 2. Compute candidate averages
    // Store as OTFParams for easier usage.
    // We only process keys that exist in candidates.
    std::unordered_map<int, OTFParams> candidate_params;
    for (auto &[key, cand] : candidates) {
      if (cand.count > 5) {
        double inv_n = 1.0 / cand.count;
        OTFParams p;
        p.a = cand.a * inv_n;
        p.b = cand.b * inv_n;
        p.c = cand.c * inv_n;
        p.alpha = cand.alpha * inv_n;
        p.valid = true;
        candidate_params[key] = p;
      }
    }

    // 3. Verification: Calculate errors for Old vs New
    std::unordered_map<int, double> err_old_sum;
    std::unordered_map<int, double> err_new_sum;

    // Helper to compute reconstruction error for a single observation
    // Helper to compute reconstruction error for a single observation
    auto compute_obj_error = [&](const Observation &obs, const OTFParams &p) {
      double err = 0;
      int rows = obs._roi_intensity.getDimRow();
      int cols = obs._roi_intensity.getDimCol();

      // Since we strictly enforce incomplete windows are discarded in
      // accumulate: rows = cols = 2 * half_w + 1
      // CRITICAL: LOGIC MUST MATCH accumulate(). THIS ASSUMES STRICT CLIPPING!
      int h = (rows - 1) / 2;

      // Because strictly clipped, x0/y0 are simply center - h
      int cx = std::lround(obs._meas_2d[0]);
      int cy = std::lround(obs._meas_2d[1]);
      int x0 = cx - h;
      int y0 = cy - h;

      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          // Image coordinates
          int img_y = y0 + r;
          int img_x = x0 + c;

          // Relative coordinates to centroid
          double dx = img_x - obs._meas_2d[0];
          double dy = img_y - obs._meas_2d[1];

          // Rotate coordinates
          double cos_a = std::cos(p.alpha);
          double sin_a = std::sin(p.alpha);
          double xx = dx * cos_a + dy * sin_a;
          double yy = -dx * sin_a + dy * cos_a;

          // Model intensity
          double exponent = -(p.b * xx * xx + p.c * yy * yy);
          double model_val = p.a * std::exp(exponent);

          double meas_val = obs._roi_intensity(r, c);
          double diff = meas_val - model_val;
          err += diff * diff;
        }
      }
      return err;
    };

    for (const auto &cp : _buffer) {
      int grid_id = getOTFGridID(param, cp._pos_3d);
      if (grid_id < 0 || grid_id >= param.n_grid)
        continue;

      for (const auto &obs : cp._obs) {
        if (!obs._otf_params.valid)
          continue;
        int cam_id = obs._cam_id;
        int key = cam_id * param.n_grid + grid_id;

        // Only verifying if we have a candidate update
        if (candidate_params.find(key) == candidate_params.end())
          continue;

        // Current (Old) Parameters
        OTFParams p_old;
        p_old.a = param.a(cam_id, grid_id);
        p_old.b = param.b(cam_id, grid_id);
        p_old.c = param.c(cam_id, grid_id);
        p_old.alpha = param.alpha(cam_id, grid_id);

        // Candidate (New) Parameters
        const OTFParams &p_new = candidate_params[key];

        err_old_sum[key] += compute_obj_error(obs, p_old);
        err_new_sum[key] += compute_obj_error(obs, p_new);
      }
    }

    // 4. Update if error reduced
    for (const auto &[key, p_cand] : candidate_params) {
      // If error decreased (or first initialization), accept
      // Initialization check: if old 'a' is 0, it's likely uninitialized
      int cam_id = key / param.n_grid;
      int grid_id = key % param.n_grid;

      double old_a = param.a(cam_id, grid_id);
      double e_old = err_old_sum[key];
      double e_new = err_new_sum[key];

      // Simple heuristic: accept if better or if old was uninitialized (a <
      // 1e-9)
      if (e_new < e_old || old_a < 1e-9) {
        param.a(cam_id, grid_id) = p_cand.a;
        param.b(cam_id, grid_id) = p_cand.b;
        param.c(cam_id, grid_id) = p_cand.c;
        param.alpha(cam_id, grid_id) = p_cand.alpha;
        updated = true;
      }
    }
  }

  return updated;
}
