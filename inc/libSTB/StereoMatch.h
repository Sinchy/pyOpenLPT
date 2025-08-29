
/**
 * StereoMatch
 * Pipeline: build (first m active cams) -> check (remaining active cams)
 *           -> prune (global disjoint selection) -> triangulate (final Object3D).
 *
 * Notes:
 * - A 2D observation may be reused across different matches until the prune step.
 * - Per-call inputs are bound at the start of match() via internal pointers.
 *   Do not call match() concurrently on the same instance.
 */
#pragma once
#include <vector>
#include <memory>
#include <limits>
#include <algorithm>
#include <functional>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <unordered_set>
#include <stdexcept>
#include <mutex>  // for std::mutex

#include "Config.h"
#include "Matrix.h"
#include "ObjectInfo.h"
#include "Camera.h"      // Camera class, used for projection and line of sight
#include "myMath.h"      // Pt2D, Pt3D, Line2D, Line3D
#include "STBCommons.h"

// Forward declarations of types used here
class IDMap;

class StereoMatch {
public:
    //Constructor
    explicit StereoMatch(const std::vector<Camera>&                       cams,
                            const std::vector<std::vector<std::unique_ptr<Object2D>>>& obj2d_list,
                            const ObjectConfig&                                        obj_cfg); // All reference variable should be initialized first

    // Main entry
    std::vector<std::unique_ptr<Object3D>> match() const;

private:
    // ---- bound per-call inputs (read-only during a match) ----
    const std::vector<Camera>&                _cams;
    const ObjectConfig&                       _obj_cfg;
    std::vector<std::vector<const Object2D*>> _obj2d_list;

    // ---- shared accelerators usable by build/check during a match ----
    std::vector<std::unique_ptr<IDMap>> _idmaps; // one per camera (nullptr if inactive)
    // _idmaps must be built in the contructor, because match() const cannot change any member variable due to "const".

    // ---- main pipeline pieces ----
    void buildMatch(std::vector<int>& match_candidate_id,
                    std::vector<std::vector<int>>& build_candidates) const;

    // Enumerate candidates on a specific camera, given the 3D lines of sight.
    void enumerateCandidatesOnCam(const std::vector<Line3D>&           los3d,
                                           int                         target_cam,
                                           std::vector<int>&           out_candidates) const;

    bool checkMatch(std::vector<int>& candidate_ids) const;

    std::vector<std::vector<int>>
    pruneMatch(const std::vector<std::vector<int>>& match_candidates) const;

    std::vector<std::unique_ptr<Object3D>>
    triangulateMatch(const std::vector<std::vector<int>>& selected_matches) const;

    // ---- camera selection helpers ----
    int selectSecondCameraByLineLength(const Line3D& los_ref,
                                       const std::vector<int>& remaining_cams) const;

    int selectNextCameraByMaxPairAngle(const std::vector<Line3D>& los3d,
                                       const std::vector<int>&     remaining_cams) const;

    // ---- LOS→image line helpers ----
    Line2D makeLine2DFromLOS3D(int cam_id, const Line3D& los) const;

    void buildLinesOnCam(const std::vector<Line3D>& los3d,
                         int cam_id,
                         std::vector<Line2D>& out_lines) const;

    // ---- early checks & tolerances (decl only; you已有实现或后续实现) ----
    double computeMinParallaxDeg(const std::vector<Line3D>& los3d) const;
    double calTriangulateTol(double final_tol_3d_mm,
                             int  k_selected, int k_target,
                             double min_parallax_deg) const;
    bool TriangulationCheckWithTol(const std::vector<Line3D>& los3d,
                                   double tol_3d_mm) const;
    bool objectEarlyCheck(const std::vector<int>& cams_used,
                          const std::vector<int>& ids_on_used) const;
    bool tracerEarlyCheck(const std::vector<int>& cams_in_path,
                                   const std::vector<int>& ids_in_path) const;
    bool bubbleEarlyCheck(const std::vector<int>& cams_in_path,
                                   const std::vector<int>& ids_in_path) const;
};


/**
 * @brief Sparse grid (cell buckets) over an image for fast 2D candidate lookup.
 *        Cells are axis-aligned squares of size _cell_px (in pixels).
 *        Each cell stores the indices (pids) of 2D observations.
 */
// IDMap.hpp

class IDMap {
public:
    struct RowSpan { int x_min, x_max; }; // invalid if x_min > x_max

    IDMap(int img_rows_px, int img_cols_px, int cell_px);

    // Rebuild the buckets for this camera using a contiguous array of 2D objects.
    // 'objs' must outlive the IDMap while you iterate.
    void rebuild(const std::vector<const Object2D*>& objs);

    // Compute per-row intersection of K LOS strips; result in CELL indices.
    // 'spans' is resized to rowsCell(), each entry holds [cx_min, cx_max] (inclusive).
    void computeStripIntersection(const std::vector<Line2D>& lines_px,
                                  double tol_px,
                                  std::vector<RowSpan>& spans) const;

    // Enumerate points ONLY inside the final per-row spans, with precise point-to-line checks.
    // Geometry check: for each point q, require distance_to_every_line(q) <= tol_px.
    // Dedup is handled internally (a point may appear in multiple cells).
    void visitPointsInRowSpans(const std::vector<RowSpan>& spans,
                               const std::vector<Line2D>&  lines_px,
                               double                      tol_px,
                               std::vector<int>&           out_indices) const;

    // ---- small helpers (inline) ----
    inline int rowsCell() const { return _rows_cell; }
    inline int colsCell() const { return _cols_cell; }
    inline int cellSizePx() const { return _cell_px; }

private:
    // Row-major bucket index
    inline int idx(int cy, int cx) const { return cy * _cols_cell + cx; }

    // Normalize a 2D vector defensively (used inside .cpp)
    static Pt2D normalized(const Pt2D& v);

private:
    int _img_rows_px = 0;
    int _img_cols_px = 0;
    int _cell_px     = 1;
    int _rows_cell   = 0;
    int _cols_cell   = 0;

    // Buckets: one vector<int> per cell (row-major)
    std::vector<std::vector<int>> _buckets;

    // Pointer to external storage of Object2D* for coordinate access during enumeration
    const std::vector<const Object2D*>* _objs = nullptr;
};


// ---------------- MatchPruner Options ----------------

// ---------------- Options ----------------
struct MatchPrunerOptions {
    // During 1↔2 growth, we remove one chosen match candidate Y to free its 2D points,
    // then search new match candidates Z that touch those freed 2D points. To cap branching,
    // we only consider the best K match candidates per freed 2D point (ranked by lower error).
    int topK_per_point = 12;

    // Number of improvement passes (each pass runs 1↔2 growth then 1↔1 quality swaps).
    int max_passes = 2;
};

// -------------- Main class ---------------
class MatchPruner {
public:
    // Construct a pruner bound to current cameras and 2D points.
    //  - cams        : list of cameras (used for LOS and triangulation).
    //  - obj2d_list  : per-camera list of const Object2D* (for 2D coordinates).
    //  - opts        : pruning options (branch cap, improvement passes).
    MatchPruner(const std::vector<Camera>& cams,
                const std::vector<std::vector<const Object2D*>>& obj2d_list,
                MatchPrunerOptions opts = {});

    // Select a disjoint subset from match_candidates.
    //  - match_candidates : vector of match candidates; each candidate is a vector<int> of size n_cams.
    //                       ids[c] = -1 if this candidate does not use camera c; otherwise the 2D index.
    // Return:
    //  - a subset of candidates (same layout) that are mutually disjoint on 2D points,
    //    chosen by: maximize count, then minimize total triangulation error.
    std::vector<std::vector<int>>
    prune(const std::vector<std::vector<int>>& match_candidates);

private:
    // ----------- internal types -----------
    struct CandidateInfo {
        // Stores info for a match candidate
        int idx = -1;                      // index in input array (for external reference only)
        std::vector<int> ids;              // size == n_cams, -1 for unused camera

        // point_flats
        // -----------
        // Flat indices of all 2D points used by this match candidate.
        // Each flat index f encodes (cam, pid) via:
        //    f = cam_offsets[cam] + pid
        //
        // Rationale:
        //  - No packing/unpacking overhead.
        //  - Direct indexing into _owner[] and _point_to_matchCandidates[].
        //  - Uniquely identifies a 2D point across all cameras.
        std::vector<size_t> point_flats;

        double error_mm = 0.0;             // triangulation error (mm)
        int scarcity = 0;                  // min degree among its 2D points (filled later)
    };

    // ----------- helpers (members hold shared state to avoid long parameter lists) -----------

    // Build CandidateInfo for all match candidates: point_flats + triangulation error.
    void buildCandidateInfos(const std::vector<std::vector<int>>& match_candidates);

    // Precompute flattening for (cam,pid) -> flat index [0, total_points).
    void buildFlatIndexMap();

    // Build reverse index: flat 2D point -> list of match candidate indices that use it.
    // And compute per-point degrees + fill per-candidate scarcity.
    void buildReverseIndexAndScarcity();

    // greedySelect()
    // --------------
    // Initial selection phase: greedily pick a maximal set of non-conflicting match candidates.
    //
    // Process:
    //   1) Sort all match candidates by:
    //        (a) scarcity  ↑  (min degree among its 2D points, rarer points first)
    //        (b) error_mm  ↑  (lower triangulation error first)
    //        (c) views     ↓  (more 2D points first)
    //        (d) idx       ↑  (original order for stability)
    //   2) Iterate in sorted order; for each candidate:
    //        - If all its 2D points are currently free (canPlace),
    //          mark those points as owned by this candidate (place)
    //          and add it to the chosen set.
    //   3) Result is a maximal disjoint set with priority given to covering
    //      rare 2D points and minimizing error.
    //
    // This forms the initial solution before local improvement passes (improveSelect).
    void greedySelect();

    // improveSelect()
    // ---------------
    // Local improvement phase: iteratively refine the chosen set from greedySelect()
    // using two types of local moves:
    //
    //   (A) 1↔2 Growth:
    //       Try to replace one chosen match candidate Y with two match candidates X and Z
    //       (both unchosen) that do not conflict with each other. This increases
    //       the total count of chosen matches. To limit branching, only consider
    //       the top-K lowest-error match candidates for each freed 2D point.
    //
    //   (B) 1↔1 Quality Swap:
    //       If an unchosen match candidate X conflicts with exactly one chosen match candidate Y
    //       and has strictly lower error, replace Y with X (count unchanged, error reduced).
    //
    // Process:
    //   - Repeat (A) then (B) over all unchosen match candidates, up to max_passes times.
    //   - If a full pass makes no changes, terminate early (converged).
    //
    // This step prioritizes increasing the number of matches, and then lowering the
    // total triangulation error without reducing the match count.
    void improveSelect();

    // ----- micro helpers -----

    // Build flat list and LOS list for a match candidate ids vector.
    void buildFlatsAndLOS(const std::vector<int>& cand_ids,
                          std::vector<size_t>& out_flats,
                          std::vector<Line3D>& out_los) const;

    // Triangulation error (mm) from LOS list.
    static double triErrorMM(const std::vector<Line3D>& los3d);

    // Occupancy management helpers
    // ----------------------------
    // These three functions maintain the `_owner` array, which tracks the current
    // assignment of each flat 2D point index:
    //
    //   _owner[f] = -1        → 2D point f is free (unassigned)
    //   _owner[f] = j         → 2D point f is owned by chosen match candidate with _C-index j
    //
    // Functions:
    //   - canPlace(ci):
    //       Returns true if all 2D points used by match candidate `ci` are currently free.
    //
    //   - place(ci, j):
    //       Marks all 2D points used by `ci` as owned by internal index `j` (index into _C).
    //
    //   - unplace(ci):
    //       Marks all 2D points used by `ci` as free (-1).
    //
    // Used in both greedy selection and local improvements to enforce disjointness.
    bool canPlace(const CandidateInfo& ci) const;
    void place(const CandidateInfo& ci, int internal_index);
    void unplace(const CandidateInfo& ci);

    // collectConflicts()
    // ------------------
    // Given a match candidate X, find all currently chosen match candidates that conflict with X.
    // Two match candidates conflict if they share at least one 2D point.
    //
    // Parameters:
    //   - X   : the match candidate to test
    //   - out : output vector of conflicting _C indices (deduplicated)
    //
    // Mechanism:
    //   - For each flat 2D point f in X, look up _owner[f]:
    //         -1  → free (no conflict)
    //         >=0 → index j of a chosen match candidate in _C
    //   - Insert each found j into `out` if not already present.
    //
    // Notes:
    //   - If `out` is empty: X has no conflict and can be placed directly.
    //   - If `out` has exactly one j: eligible for 1↔2 growth or 1↔1 swap logic.
    //   - If `out` has more than one j: current implementation skips for speed.
    void collectConflicts(const CandidateInfo& X, std::vector<int>& out) const;

    // From a set of freed 2D points, collect Top-K match candidate indices touching them (rank by error).
    void gatherTopKOnFreed(const std::vector<size_t>& freed_flats, int K,
                           std::vector<int>& out_sorted_unique) const;

private:
    // ----------- bound inputs -----------
    const std::vector<Camera>&                          _cams;
    const std::vector<std::vector<const Object2D*>>&    _obj2d;
    MatchPrunerOptions                                   _opt;

    // ----------- derived indexes / caches -----------
    // flatten (cam,pid) -> flat index
    std::vector<size_t> _cam_offsets;   // size n_cams+1, prefix sums of per-cam 2D point counts
    size_t              _n_points_total = 0;

    // candidate info
    std::vector<CandidateInfo> _C;      // internal list of match candidates kept after validation

    // reverse index: per flat 2D point -> match candidate indices that use it
    std::vector<std::vector<int>> _point_to_matchCandidates; // size _n_points_total
    std::vector<int>              _point_degrees;            // size _n_points_total

    // current solution state
    std::vector<int>  _owner;       // size _n_points_total; -1 if free, else internal _C index
    std::vector<uint8_t> _chosen;   // size _C.size(); mark chosen (0/1)
    std::vector<int>  _chosen_list; // list of chosen internal indices into _C

    // LOS cache (optional but enabled): for each flat 2D point, precompute its Line3D once.
    std::vector<Line3D> _los_cache; // size == _n_points_total
    bool                _los_cached = false;

    // scratch buffers to avoid frequent realloc
    mutable std::vector<Line3D>    _scratch_los;
    mutable std::vector<size_t>    _scratch_flats;
};