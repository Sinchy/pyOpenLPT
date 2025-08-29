#include <iterator>
#include "StereoMatch.h"
#include "omp.h"

static std::vector<std::vector<const Object2D*>>
makeView(const std::vector<std::vector<std::unique_ptr<Object2D>>>& src2d)
{
    // convert the input obj2d_list (unique_ptr) to view-only constant pointer
    // which can make coding easier 
    std::vector<std::vector<const Object2D*>> view(src2d.size());
    for (size_t c = 0; c < src2d.size(); ++c) {
        auto& dst = view[c];
        dst.reserve(src2d[c].size());
        for (auto const& up : src2d[c])
            dst.push_back(up.get());
    }
    return view;
}

StereoMatch::StereoMatch(const std::vector<Camera>&                           cams,
                            const std::vector<std::vector<std::unique_ptr<Object2D>>>& obj2d_list,
                            const ObjectConfig&                                        obj_cfg) 
    : _cams(cams)      // 引用成员必须在这里初始化
    , _obj_cfg(obj_cfg)
    , _obj2d_list(makeView(obj2d_list))
{
    // pre-check
    const int n_cams = static_cast<int>(_cams.size());
    if (n_cams <= 0)
        throw std::runtime_error("preCheck: cams is empty.");
    if (static_cast<int>(_obj2d_list.size()) != n_cams)
        throw std::runtime_error("preCheck: cams and obj2d_list sizes must match.");

    int n_active = 0;
    for (int i = 0; i < n_cams; ++i)
        if (_cams[i]._is_active) ++n_active;

    if (n_active < 2)
        throw std::runtime_error("preCheck: need at least 2 active cameras.");

    int match_cam_count = _obj_cfg._sm_param.match_cam_count;
    if (match_cam_count < 2 || match_cam_count > n_active)
        throw std::runtime_error("preCheck: match_cam_count must be within [2, #active].");

    // ---- (Re)build IDMaps for all active cams once (used by buildMatch/checkMatch). ----
    _idmaps.clear();
    _idmaps.resize(n_cams);
    for (int c = 0; c < n_cams; ++c) {
        if (!_cams[c]._is_active) continue;
        // create IDMap for this camera
        _idmaps[c] = std::make_unique<IDMap>(_cams[c].getNRow(), _cams[c].getNCol(), _obj_cfg._sm_param.idmap_cell_px);

        // build IDMap buckets
        _idmaps[c]->rebuild(_obj2d_list[c]);
    }
} // All reference variable should be initialized first


std::vector<std::unique_ptr<Object3D>> StereoMatch::match() const
{
    const int n_cams = static_cast<int>(_cams.size());

    // ---- Choose reference camera: among active cams, pick the one with the fewest 2D observations. ----
    int ref_cam = -1;
    {
        size_t best = std::numeric_limits<size_t>::max();
        for (int i = 0; i < n_cams; ++i) {
            if (!_cams[i]._is_active) continue;
            const size_t sz = _obj2d_list[i].size();
            if (sz < best) { best = sz; ref_cam = i; }
        }
        if (ref_cam < 0) {
            throw std::runtime_error("StereoMatch::match: no active camera found.");
        }
    }

    // ---- Stage A + B: build match candidates per reference observation, then check on remaining cams. ----
    const auto& ref_obs = _obj2d_list[ref_cam];
    std::vector<std::vector<int>> match_candidates;
    match_candidates.clear();
    //match_candidates.reserve(estimate_total_candidates);

    // Prepare per-thread buckets (no sharing between threads)
    const int T = std::max(1, _obj_cfg._n_thread);
    std::vector<std::vector<std::vector<int>>> thread_buckets(T);

    #pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();
        auto& bucket = thread_buckets[tid];              // thread-local bucket (by index)

        #pragma omp for schedule(dynamic, 8)
        for (int id_ref = 0; id_ref < static_cast<int>(ref_obs.size()); ++id_ref) {
            // One candidate initialized with -1 everywhere and ref_cam filled.
            std::vector<int> match_candidate_id(n_cams, -1);
            match_candidate_id[ref_cam] = id_ref;

            // Build across the first m active cams (DFS inside; uses new selection rules).
            std::vector<std::vector<int>> build_candidates;
            buildMatch(match_candidate_id, build_candidates); // must be thread-safe

            // Check on remaining active cams (existence / object-specific light checks).
            std::vector<std::vector<int>> selected;      // per-iter local
            selected.reserve(build_candidates.size());
            for (auto& cand : build_candidates) {
                if (!checkMatch(cand)) continue;
                selected.emplace_back(std::move(cand));
            }

            // Move this thread's selected candidates into its own bucket (no lock)
            if (!selected.empty()) {
                bucket.insert(bucket.end(),
                    std::make_move_iterator(selected.begin()),
                    std::make_move_iterator(selected.end()));
            }
        }
    } // parallel region ends

    // Flatten buckets sequentially (single-thread). 不需要预估；若想快一点，这里可先统计总数再一次性 reserve。
    match_candidates.clear();
    // 可选优化：size_t total = 0; for (auto& b : thread_buckets) total += b.size(); match_candidates.reserve(total);
    for (auto& b : thread_buckets) {
        match_candidates.insert(match_candidates.end(),
            std::make_move_iterator(b.begin()),
            std::make_move_iterator(b.end()));
    }
    std::sort(match_candidates.begin(), match_candidates.end()); // make sure the sequence the same everytime running the code

    // ---- Global pruning (disjoint usage of 2D points, pick best by triangulation error, etc.). ----
    std::vector<std::vector<int>> selected = pruneMatch(match_candidates);

    // ---- Final triangulation and Object3D construction (all active cams participate). ----
    std::vector<std::unique_ptr<Object3D>> out = triangulateMatch(selected);

    return out;
}

// Build across the first m active cameras (iterative DFS).
// Input : match_candidate_id -> size == n_cams; only ref_cam position is set (>=0), others -1
// Output: build_candidates   -> each element size == n_cams; used cams filled with 2D ids, others -1
void StereoMatch::buildMatch(std::vector<int>& match_candidate_id,
                             std::vector<std::vector<int>>& build_candidates) const
{
    build_candidates.clear();

    const int n_cams = static_cast<int>(_cams.size());

    // ---- Extract (ref_cam, ref_id) from the incoming pattern (match() already guarantees it exists) ----
    int ref_cam = -1, ref_id = -1;
    for (int c = 0; c < n_cams; ++c) {
        if (!_cams[c]._is_active) continue;
        if (match_candidate_id[c] >= 0) { ref_cam = c; ref_id = match_candidate_id[c]; break; }
    }

    // ---- Collect active cams & pool (exclude ref) in one pass ----
    std::vector<int> active_cams; active_cams.reserve(n_cams);
    std::vector<int> pool_cams;   pool_cams.reserve(n_cams);
    for (int c = 0; c < n_cams; ++c) {
        if (!_cams[c]._is_active) continue;
        active_cams.push_back(c);
        if (c != ref_cam) pool_cams.push_back(c);
    }

    const int build_cam_count = std::min(std::max(2, _obj_cfg._sm_param.match_cam_count),
                                         static_cast<int>(active_cams.size()));

    // ---- DFS frame definition ----
    struct Frame {
        // Chosen path so far (parallel arrays)
        std::vector<int>    chosen_cams;   // camera ids on this path (order = build order)
        std::vector<int>    chosen_ids;    // 2D indices aligned with chosen_cams
        std::vector<Line3D> los3d;         // LOS for each chosen pair (same order)

        // Remaining build cameras to choose from
        std::vector<int>    rem_cams;

        // The camera we are currently expanding and its candidate list
        int                 target_cam = -1;
        std::vector<int>    candidates;    // candidate 2D ids on target_cam
        size_t              cand_idx  = 0; // next candidate index to try
    };

    std::vector<Frame> stack;
    stack.reserve(build_cam_count + 4); // small headroom

    // ---- Initialize root frame from (ref_cam, ref_id) ----
    Frame root;
    root.chosen_cams = { ref_cam };
    root.chosen_ids  = { ref_id  };
    root.rem_cams    = pool_cams;

    {
        const Pt2D& p = _obj2d_list[ref_cam][ref_id]->_pt_center;
        root.los3d.push_back(_cams[ref_cam].lineOfSight(p));
    }

    // Select the 2nd build camera by "shortest projected LOS segment", then enumerate its candidates.
    if (!root.rem_cams.empty()) {
        root.target_cam = selectSecondCameraByLineLength(root.los3d.front(), root.rem_cams);
        if (root.target_cam >= 0) {
            enumerateCandidatesOnCam(root.los3d, root.target_cam, root.candidates);
            root.cand_idx = 0;
        }
    }
    if (root.target_cam < 0 || root.candidates.empty()) return; // cannot grow from this ref

    // ---- Push the root frame to start DFS ----
    stack.push_back(std::move(root));

    // =========================
    // Iterative DFS main loop
    // =========================
    //
    // Stack mechanics:
    // - stack.back() is the current frame to expand.
    // - Try candidates of frame.target_cam one by one:
    //     * For each candidate → extend path (build a child "nxt"),
    //       run quick triangulation + object early checks.
    //     * If checks fail → CONTINUE (stay on current frame, try next candidate).
    //     * If checks pass:
    //         - If chosen_cams reached build_cam_count → push terminal child;
    //           it will be recorded (materialized) at the next loop head, then POP (backtrack).
    //         - Else select next target_cam (max pairwise 2D angle), enumerate candidates,
    //           then PUSH child (go deeper).
    // - When current frame runs out of candidates → POP (backtrack to its parent).
    //
    /*
        ===============================================================
        DFS (stack-based) – concise, plain-language version
        Focus: per-camera / per-2D iteration with PUSH/POP
        ---------------------------------------------------------------

        [Start]
        |
        [Initialize: candidate=-1 per camera; clear scratch; clear stack]
        |
        [Push root frame (first camera, its 2D list, save scratch sizes)]
        |
        +------------------- While stack is not empty -------------------+
        | [Top = stack.back()]                                           |
        |   |                                                            |
        |   +-- Have we tried all 2D ids for Top.camera? -- Yes --> [POP]|
        |   |      (restore scratch sizes; clear this camera;            |
        |   |       pop the frame; go back to while)                     |
        |   |                                                            |
        |   +-- No --> [Take next 2D id for Top.camera]                  |
        |                |                                               |
        |                [Extend locally: set candidate[Top.camera]=id;  |
        |                 append LOS/flat to scratch]                    |
        |                |                                               |
        |                +-- Do quick checks pass? -- No -->             |
        |                |     [Undo local extend (clear and pop)        |
        |                |      try next 2D in the same frame]           |
        |                |                                               |
        |                +-- Yes --> Do we still need more cameras?      |
        |                           |                                    |
        |                           +-- Yes -->                          |
        |                           |    [PUSH child frame               |
        |                           |     (next camera, its 2D list,     |
        |                           |      save current scratch sizes);  |
        |                           |     descend and continue while]    |
        |                           |                                    |
        |                           +-- No  -->                          |
        |                                [Emit full candidate;           |
        |                                 Undo local extend (stay here); |
        |                                 try next 2D in this frame]     |
        +----------------------------------------------------------------+

        Legend:
        - PUSH: push a new frame (chosen camera, its 2D list, and saved scratch sizes).
        - POP : pop current frame, restore scratch to saved sizes, clear that camera’s slot.
        - “Undo local extend” = revert only the latest (camera,2D) choice without popping the frame.
        ===============================================================
        */

    while (!stack.empty()) {
        Frame& fr = stack.back();

        // Terminal frame: reached build_cam_count cameras on this path
        if (static_cast<int>(fr.chosen_cams.size()) == build_cam_count) {
            // Materialize a cam-aligned result starting from the incoming pattern.
            std::vector<int> ids_norm = match_candidate_id; // ref already set; others -1
            for (size_t i = 0; i < fr.chosen_cams.size(); ++i) {
                ids_norm[ fr.chosen_cams[i] ] = fr.chosen_ids[i];
            }
            build_candidates.push_back(std::move(ids_norm));

            // Done with this terminal node → POP to try siblings at the upper level.
            stack.pop_back();
            continue;
        }

        // already loop all candidates on current target → POP and backtrack
        if (fr.cand_idx >= fr.candidates.size()) {
            stack.pop_back();
            continue;
        }

        // Take the next candidate on current target_cam
        const int pid = fr.candidates[fr.cand_idx++]; // advance for the next loop
        const int cam = fr.target_cam;

        // Prepare child frame (nxt). Push only if sanity checks pass.
        Frame nxt;
        nxt.chosen_cams = fr.chosen_cams;
        nxt.chosen_ids  = fr.chosen_ids;
        nxt.los3d       = fr.los3d;
        nxt.rem_cams    = fr.rem_cams;

        // Extend path with (cam, pid)
        nxt.chosen_cams.push_back(cam);
        nxt.chosen_ids.push_back(pid);

        {
            const Pt2D& p = _obj2d_list[cam][pid]->_pt_center;
            nxt.los3d.push_back(_cams[cam].lineOfSight(p));
        }

        // Remove this cam from the remaining pool
        if (auto it = std::find(nxt.rem_cams.begin(), nxt.rem_cams.end(), cam); it != nxt.rem_cams.end())
            nxt.rem_cams.erase(it);

        // ---- Early pruning BEFORE pushing the child frame ----
        // Quick triangulation check with unified tolerance (>=2 LOS here)
        {
            const int    k           = static_cast<int>(nxt.los3d.size());
            const double th_min_deg  = computeMinParallaxDeg(nxt.los3d);
            const double tol_quickmm = calTriangulateTol(_obj_cfg._sm_param.tol_3d_mm, k, build_cam_count, th_min_deg);
            if (!TriangulationCheckWithTol(nxt.los3d, tol_quickmm)) {
                // Reject this candidate; stay on current frame to try its next candidate.
                continue;
            }
        }
        // Object-specific cheap check (e.g., bubble radius coherence)
        if (!objectEarlyCheck(nxt.chosen_cams, nxt.chosen_ids)) {
            // Reject this candidate; try the next one in current frame.
            continue;
        }

        // Reached build count after adding this cam → push terminal child.
        if (static_cast<int>(nxt.chosen_cams.size()) == build_cam_count) {
            nxt.target_cam = -1; nxt.candidates.clear(); nxt.cand_idx = 0;
            stack.push_back(std::move(nxt)); // will be materialized and popped at loop head
            continue;
        }

        // Select NEXT build camera (we already have >=2 LOS here):
        nxt.target_cam = selectNextCameraByMaxPairAngle(nxt.los3d, nxt.rem_cams);
        if (nxt.target_cam < 0) {
            // Dead end for this candidate; try the next one in current frame.
            continue;
        }

        // Enumerate candidates ONLY for the selected next camera.
        enumerateCandidatesOnCam(nxt.los3d, nxt.target_cam, nxt.candidates);
        if (nxt.candidates.empty()) {
            // Dead end for this candidate; try the next one in current frame.
            continue;
        }
        nxt.cand_idx = 0;

        // All early checks passed and we have next-camera candidates → PUSH child, go deeper.
        stack.push_back(std::move(nxt));
    }
}

// Enumerate candidate 2D indices on `target_cam` that lie within the intersection
// of all projected strips. Geometry checks & dedup are done inside IDMap::visitPointsInRowSpans.
void StereoMatch::enumerateCandidatesOnCam(const std::vector<Line3D>& los3d,
                                           int                         target_cam,
                                           std::vector<int>&           out_candidates) const
{
    out_candidates.clear();

    if (target_cam < 0 || target_cam >= static_cast<int>(_idmaps.size())) return;
    IDMap* idm = _idmaps[target_cam].get();
    if (!idm || los3d.empty()) return;

    // 1) Project LOS -> 2D image lines on this camera
    std::vector<Line2D> lines_px;
    buildLinesOnCam(los3d, target_cam, lines_px);
    if (lines_px.empty()) return;

    // 2) Compute per-row strip intersection in CELL indices
    std::vector<IDMap::RowSpan> spans;
    idm->computeStripIntersection(lines_px, _obj_cfg._sm_param.tol_2d_px, spans);

    bool any_valid = false;
    for (const auto& s : spans) { if (s.x_min <= s.x_max) { any_valid = true; break; } }
    if (!any_valid) return;

    // 3) Visit-and-collect with precise distance test and dedup done by IDMap
    idm->visitPointsInRowSpans(spans, lines_px, _obj_cfg._sm_param.tol_2d_px, out_candidates);

    // Optional: deterministic order if you prefer stable outputs
    // std::sort(out_candidates.begin(), out_candidates.end());
}

// Verify: for every remaining active (check) camera, the intersection of all
// projected strips (from already chosen LOS) contains at least one 2D point.
// candidate_ids: size == n_cams, chosen build cams have non-negative ids, others -1.
bool StereoMatch::checkMatch(std::vector<int>& candidate_ids) const
{
    const int n_cams = static_cast<int>(_cams.size());

    // ---- 1) Gather LOS from chosen (build) cameras ----
    std::vector<Line3D> los3d;
    los3d.reserve(_obj_cfg._sm_param.match_cam_count);
    for (int cam = 0; cam < n_cams; ++cam) {
        const int pid = candidate_ids[cam];
        if (pid < 0) continue; // not chosen on this cam
        const Pt2D& q = _obj2d_list[cam][pid]->_pt_center;
        los3d.push_back(_cams[cam].lineOfSight(q));
    }
    if (los3d.size() < 2) {
        // Should not happen if build stage produced a valid candidate,
        // but guard against malformed input.
        return false;
    }

    // ---- 2) For every active camera that is NOT chosen yet (check cameras) ----
    for (int cam = 0; cam < n_cams; ++cam) {
        if (!_cams[cam]._is_active)    continue; // inactive -> ignore
        if (candidate_ids[cam] >= 0)     continue; // already used in build -> skip

        // IDMap must exist for this camera
        IDMap* idm = (cam < static_cast<int>(_idmaps.size())) ? _idmaps[cam].get() : nullptr;
        if (!idm) return false; // defensive: no map to verify against

        // ---- 3) Project all LOS onto this camera as 2D lines ----
        std::vector<Line2D> lines_px;
        buildLinesOnCam(los3d, cam, lines_px);
        if (lines_px.empty()) return false;

        // ---- 4) Compute per-row strip intersection (CELL space) ----
        std::vector<IDMap::RowSpan> spans;
        idm->computeStripIntersection(lines_px, _obj_cfg._sm_param.tol_2d_px, spans);

        bool any_valid = false;
        for (const auto& s : spans) {
            if (s.x_min <= s.x_max) { any_valid = true; break; }
        }
        if (!any_valid) return false; // empty intersection on this check cam

        // ---- 5) Visit points inside intersection with precise geometry & dedup (inside IDMap) ----
        std::vector<int> inliers;
        idm->visitPointsInRowSpans(spans, lines_px, _obj_cfg._sm_param.tol_2d_px, inliers);

        // Optional: add object-specific quick filter here (e.g., bubble radius coherence per-cam).
        // if (!inliers.empty() && !objectCheckOnCheckCam(cam, inliers, candidate_ids)) { return false; }

        if (inliers.empty()) return false; // this check cam has no supporting observation
    }

    // ---- 6) All check cameras have ≥1 supporting point ----
    return true;
}

// Select a disjoint, high-quality subset from the raw match candidates.
// - Uses MatchPruner (greedy + local 1↔2, 1↔1 improvements) with triangulation error only.
// - Assumes _cams/_obj2d_list are already bound by StereoMatch::match() before calling.
std::vector<std::vector<int>>
StereoMatch::pruneMatch(const std::vector<std::vector<int>>& match_candidates) const
{
    // Basic guard: match() should have bound these; keep a defensive check.
    if (match_candidates.empty()) return {};

    // Tunables: keep small and conservative; expose if you want later.
    MatchPrunerOptions opt;
    opt.topK_per_point = 12; // cap branching during 1↔2 growth
    opt.max_passes           = 2;  // growth + quality passes

    // Run pruner.
    MatchPruner pruner(_cams, _obj2d_list, opt);
    return pruner.prune(match_candidates);
}

// Triangulate all selected matches, drop those with error > tol_3d,
// and build Object3D (Tracer3D / Bubble3D) with cloned 2D observations (camera order preserved).
std::vector<std::unique_ptr<Object3D>>
StereoMatch::triangulateMatch(const std::vector<std::vector<int>>& selected_matches) const
{
    const auto& cams  = _cams;
    const auto& obs2d = _obj2d_list;
    const double tol3d = _obj_cfg._sm_param.tol_3d_mm; // [mm]

    std::vector<std::unique_ptr<Object3D>> out;
    out.reserve(selected_matches.size());

    std::vector<Line3D> los;
    Pt3D  pt_world;
    double err = 0.0;

    for (const auto& ids : selected_matches) {
        // 1) Build LOS from this match 
        los.clear();
        const int n_cams = static_cast<int>(cams.size());
        los.reserve(n_cams);
        for (int c = 0; c < n_cams; ++c) {
            const int pid = ids[c];
            if (pid < 0) continue; 
            const Pt2D& q = obs2d[c][pid]->_pt_center;
            los.push_back(cams[c].lineOfSight(q));
        }
        if (los.size() < 2) continue; 

        // 2) Triangulation
        try {
            myMATH::triangulation(pt_world, err, los);
        } catch (...) {
            continue; 
        }

        // 3) Final tolerance gate
        if (err > tol3d) continue;

        // 4) create object 2D list
        std::vector<std::unique_ptr<Object2D>> obj2d_list(n_cams);
        for (int c = 0; c < n_cams; ++c) {
            const int pid = ids[c];
            if (pid < 0) continue;
            obj2d_list[c] = obs2d[c][pid]->clone();
        }

        // 5) create object
        CreateArgs a;
        a._pt_center       = pt_world;
        a._obj2d_ready     = std::move(obj2d_list);
        a._cams            = &cams;                 // only needed if Bubble radius should be computed
        a._compute_bubble_radius = (_obj_cfg.kind() == ObjectKind::Bubble);

        auto obj3d = _obj_cfg.creatObject3D(std::move(a)); // create object according to _obj_cfg
        if (!obj3d) continue;
        obj3d->_is_tracked = false;

        out.emplace_back(std::move(obj3d));
    }

    return out;
}


// ---- helper: make a 2D line on camera 'cam_id' from a 3D LOS ----
inline Line2D StereoMatch::makeLine2DFromLOS3D(int cam_id, const Line3D& los) const
{
    const Pt2D a = _cams[cam_id].project(los.pt);
    const Pt2D b = _cams[cam_id].project(Pt3D{
        los.pt[0] + los.unit_vector[0],
        los.pt[1] + los.unit_vector[1],
        los.pt[2] + los.unit_vector[2]
    });
    Line2D L;
    L.pt = a;
    L.unit_vector = myMATH::createUnitVector(a, b); // assumed normalized
    return L;
}

// ---- helper: project a set of LOS to 2D lines on a camera (re-uses output buffer) ----
inline void StereoMatch::buildLinesOnCam(const std::vector<Line3D>& los3d,
                                         int cam_id,
                                         std::vector<Line2D>& out_lines) const
{
    out_lines.clear();
    out_lines.reserve(los3d.size());
    for (const auto& L3 : los3d) {
        out_lines.push_back(makeLine2DFromLOS3D(cam_id, L3));
    }
}

// ---- early checks & tolerances (decl only; you已有实现或后续实现) ----
// Compute minimal parallax angle among current LOS set (degrees).
double StereoMatch::computeMinParallaxDeg(const std::vector<Line3D>& los) const {
    if (los.size() < 2) return 180.0;
    double min_deg = 180.0;
    for (size_t i = 0; i < los.size(); ++i) {
        const Pt3D& a = los[i].unit_vector;
        for (size_t j = i + 1; j < los.size(); ++j) {
            const Pt3D& b = los[j].unit_vector;
            double dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
            dot = std::max(-1.0, std::min(1.0, dot));
            double deg = std::acos(dot) * 180.0 / M_PI;
            if (deg < min_deg) min_deg = deg;
        }
    }
    return min_deg;
}

// Compute quick-stage tolerance based on depth (k), target build count (m) and min parallax.
double StereoMatch::calTriangulateTol(double tol_final_mm, int k, int m, double min_parallax_deg) const
{
    // Depth factor ~ sqrt(m / k): information accumulation (variance ~ 1/k, stdev ~ 1/sqrt(k))
    const int kk = std::max(k, 2);
    const int mm = std::max(m, 2);
    const double alpha_depth = std::sqrt(double(mm) / double(kk));

    // Angle factor ~ 1/sin(theta), clamped to [1, gamma_max]
    const double theta_ref_deg = 6.0;   // tweakable
    const double theta_min_cap = 1.0;   // avoid degeneracy
    const double gamma_max     = 2.0;   // upper clamp

    constexpr auto deg2rad = [](double deg) constexpr {
    return deg * M_PI / 180.0;
    };

    const double th = std::max(min_parallax_deg, theta_min_cap);
    const double num = std::sin(deg2rad(theta_ref_deg));
    const double den = std::max(1e-6, std::sin(deg2rad(th)));
    const double alpha_angle = std::max(1.0, std::min(gamma_max, num / den));

    return tol_final_mm * alpha_depth * alpha_angle;
}

// Unified triangulation check with a given tolerance (mm).
bool StereoMatch::TriangulationCheckWithTol(const std::vector<Line3D>& los3d,
                                            double tol_3d_mm) const
{
    if (los3d.size() < 2) return true; // need >=2 LOS to triangulate
    Pt3D pt; double err = 0.0;
    try { myMATH::triangulation(pt, err, los3d); }
    catch (...) { return false; }
    return err <= tol_3d_mm;
}

// Object-specific lightweight check hook (build stage). Router + stubs.
bool StereoMatch::objectEarlyCheck(const std::vector<int>& cams_in_path,
                                   const std::vector<int>& ids_in_path) const
{
    switch (_obj_cfg.kind()) {
        case ObjectKind::Tracer: return tracerEarlyCheck(cams_in_path, ids_in_path);
        case ObjectKind::Bubble: return bubbleEarlyCheck(cams_in_path, ids_in_path);
        default:                 return true;
    }
}

// Tracer: no extra early rule in build stage.
bool StereoMatch::tracerEarlyCheck(const std::vector<int>&,
                                   const std::vector<int>&) const
{
    return true;
}

// bubble: check radius consistency
bool StereoMatch::bubbleEarlyCheck(const std::vector<int>& cams_in_path,
                                   const std::vector<int>& ids_in_path) const
{
    const size_t k = std::min(cams_in_path.size(), ids_in_path.size());
    if (k < 2) return true;

    // 1) Collect LOS / cameras / 2D objects (aligned by index)
    std::vector<Line3D>               los;           los.reserve(k);
    std::vector<const Camera*>        cams;          cams.reserve(k);
    std::vector<const Object2D*>      obj2d_by_cam;  obj2d_by_cam.reserve(k);

    for (size_t i = 0; i < k; ++i) {
        const int cam = cams_in_path[i], pid = ids_in_path[i];
        if (cam < 0 || pid < 0) continue;
        const Object2D* base = _obj2d_list[cam][pid];
        if (!base) continue;

        const Pt2D& q = base->_pt_center;                 // same domain as calibration
        los.push_back(_cams[cam].lineOfSight(q));
        cams.push_back(&_cams[cam]);
        obj2d_by_cam.push_back(base);
    }
    if (los.size() < 2) return true;

    // 2) Triangulate 3D center with ALL LOS
    Pt3D   X_world{};
    double tri_err = 0.0;                                  // world-length units
    myMATH::triangulation(X_world, tri_err, los);
    if (!std::isfinite(tri_err) || tri_err < 0.0) return true; // permissive

    // 3) Build tol_3d from triangulation error with parallax amplification
    //    tol_3d_eff ≈ tri_err / sin(theta_min), clamp theta to avoid blow-up.
    const double theta_min_deg = computeMinParallaxDeg(los);       // you already have this
    const double min_deg = 10.0;                                   // TODO: _cfg.min_parallax_deg if available
    const double theta_use_deg = std::max(theta_min_deg, min_deg);
    const double sin_theta = std::max(std::sin(theta_use_deg * M_PI / 180.0), 1e-3);
    const double tol_3d_eff = tri_err / sin_theta;                  // world-length units

    // 4) 2D reprojection tolerance from calibration (pixels)
    const double tol_2d_px = _obj_cfg._sm_param.tol_2d_px;                        // <=0 to disable if you want

    // 5) Radius consistency gate (PINHOLE effective; POLYNOMIAL ignored for now)
    return Bubble::checkRadiusConsistency(X_world, cams, obj2d_by_cam,
                                          tol_2d_px, tol_3d_eff);
}



// Select the 2nd build camera by minimizing the visible segment length of the single projected LOS.
int StereoMatch::selectSecondCameraByLineLength(const Line3D& los_ref,
                                                const std::vector<int>& remaining_cams) const
{
    if (remaining_cams.empty()) return -1;

    auto segLenOnCam = [&](int cam) -> double {
        Line2D L = makeLine2DFromLOS3D(cam, los_ref);
        const double px = L.pt[0], py = L.pt[1];
        const double ux = L.unit_vector[0], uy = L.unit_vector[1];

        const double W = static_cast<double>(_cams[cam].getNCol());
        const double H = static_cast<double>(_cams[cam].getNRow());

        auto in = [](double v, double lo, double hi){ return v >= lo && v <= hi; };
        auto push_if = [&](double x,double y,std::vector<Pt2D>& v){ if(in(x,0.0,W)&&in(y,0.0,H)) v.emplace_back(x,y); };

        std::vector<Pt2D> pts; pts.reserve(4);
        if (std::fabs(ux) > 1e-12) {
            double t = (0.0 - px)/ux;  push_if(0.0, py + t*uy, pts);
            t = (W   - px)/ux;         push_if(W,   py + t*uy, pts);
        }
        if (std::fabs(uy) > 1e-12) {
            double t = (0.0 - py)/uy;  push_if(px + t*ux, 0.0, pts);
            t = (H   - py)/uy;         push_if(px + t*ux, H,   pts);
        }
        // de-dup corners
        const double eps = 1e-9;
        std::vector<Pt2D> uniq; uniq.reserve(pts.size());
        for (auto& q: pts) {
            bool dup = false;
            for (auto& p: uniq) if (std::fabs(p[0]-q[0])<eps && std::fabs(p[1]-q[1])<eps) { dup=true; break; }
            if (!dup) uniq.push_back(q);
        }
        if (uniq.size() < 2) return 0.0;
        const double dx = uniq[0][0]-uniq[1][0], dy = uniq[0][1]-uniq[1][1];
        return std::sqrt(dx*dx + dy*dy);
    };

    int best_cam = -1;
    double best_len = std::numeric_limits<double>::infinity();
    for (int cam : remaining_cams) {
        const double L = segLenOnCam(cam);
        if (L < best_len - 1e-9) { best_len = L; best_cam = cam; }
    }
    return best_cam;
}

// Select next build camera (>=2 LOS already selected) by maximizing the maximum pairwise 2D angle.
// Score = max_{i<j} |sin(theta_ij)| = |u_i x u_j|; ties keep the first encountered.
int StereoMatch::selectNextCameraByMaxPairAngle(const std::vector<Line3D>& los3d,
                                                const std::vector<int>&     remaining_cams) const
{
    if (remaining_cams.empty()) return -1;

    int    best_cam   = -1;
    double best_score = -1.0;

    std::vector<Line2D> lines_px; // reused buffer

    for (int cam : remaining_cams) {
        buildLinesOnCam(los3d, cam, lines_px);

        double score = 0.0;
        if (lines_px.size() >= 2) {
            for (size_t i = 0; i < lines_px.size(); ++i) {
                const Pt2D& ui = lines_px[i].unit_vector; // unit
                for (size_t j = i + 1; j < lines_px.size(); ++j) {
                    const Pt2D& uj = lines_px[j].unit_vector;
                    const double s = std::fabs(ui[0]*uj[1] - ui[1]*uj[0]);
                    if (s > score) score = s;
                    if (score >= 1.0 - 1e-12) break; // ~90°, cannot do better
                }
                if (score >= 1.0 - 1e-12) break;
            }
        }
        if (score > best_score) { best_score = score; best_cam = cam; }
    }
    return best_cam;
}


IDMap::IDMap(int img_rows_px, int img_cols_px, int cell_px)
    : _img_rows_px(img_rows_px),
      _img_cols_px(img_cols_px),
      _cell_px(std::max(1, cell_px))
{
    _rows_cell = std::max(1, _img_rows_px / _cell_px + (_img_rows_px % _cell_px ? 1 : 0));
    _cols_cell = std::max(1, _img_cols_px / _cell_px + (_img_cols_px % _cell_px ? 1 : 0));
    _buckets.assign(_rows_cell * _cols_cell, {});
}

void IDMap::rebuild(const std::vector<const Object2D*>& objs)
{
    _objs = &objs;
    // clear buckets
    for (auto& v : _buckets) v.clear();

    const int n = static_cast<int>(objs.size());
    for (int pid = 0; pid < n; ++pid) {
        const Object2D* o = objs[pid];
        int cx = static_cast<int>(std::floor(o->_pt_center[0] / _cell_px));
        int cy = static_cast<int>(std::floor(o->_pt_center[1] / _cell_px));
        if (cx < 0 || cy < 0 || cx >= _cols_cell || cy >= _rows_cell) continue;
        _buckets[idx(cy, cx)].push_back(pid);
    }
}

static inline double hypot2(double x, double y) { return x*x + y*y; }

Pt2D IDMap::normalized(const Pt2D& v)
{
    const double nx = v[0], ny = v[1];
    const double n2 = nx*nx + ny*ny;
    if (n2 <= 1e-24) {
        // fallback to +X if zero-length (should not happen if callers ensure unit)
        return Pt2D{1.0, 0.0};
    }
    const double inv = 1.0 / std::sqrt(n2);
    return Pt2D{ nx * inv, ny * inv };
}

void IDMap::computeStripIntersection(const std::vector<Line2D>& lines_px,
                                     double tol_px,
                                     std::vector<RowSpan>& spans) const
{
    spans.assign(_rows_cell, RowSpan{ 0, _cols_cell - 1 });

    const double hw = 0.5 * _cell_px;   // half cell size (px)
    const double hh = 0.5 * _cell_px;
    const double to_index = 1.0 / _cell_px;
    const double eps = 1e-12;
    const double tol_px_grow = tol_px + 1e-6; // tiny safety for boundaries

    for (const Line2D& L : lines_px) {
        // Line normal form: n·x + d = 0
        const Pt2D u = normalized(L.unit_vector);   // if guaranteed unit, you may skip
        const Pt2D n{ -u[1], u[0] };                // unit normal
        const double d = -(n[0]*L.pt[0] + n[1]*L.pt[1]);

        for (int cy = 0; cy < _rows_cell; ++cy) {
            RowSpan& row = spans[cy];
            if (row.x_min > row.x_max) continue; // already invalid

            const double y_c = (cy + 0.5) * _cell_px;
            const double C   = n[1] * y_c + d;  // note: |n * c + d| is the distance to the line from point c(cx, cy) 
            const double proj_extent = std::fabs(n[0]) * hw + std::fabs(n[1]) * hh;
            const double T   = tol_px_grow + proj_extent;

            int a = 0, b = _cols_cell - 1;

            if (std::fabs(n[0]) < eps) {
                // Horizontal strip (independent of x): whole row if |C| <= T, else invalidate row
                if (std::fabs(C) > T) { row.x_min = 1; row.x_max = 0; continue; }
                // else keep [0, cols-1]
            } else {
                // Allowed center-x (pixels): x ∈ [(-T-C)/n_x, (T-C)/n_x]
                double x_left  = (-T - C) / n[0];
                double x_right = ( T - C) / n[0];
                if (x_left > x_right) std::swap(x_left, x_right);

                // Map to cell index interval: center of cx is (cx+0.5)*cell_px
                a = static_cast<int>(std::ceil( x_left  * to_index - 0.5 ));
                b = static_cast<int>(std::floor(x_right * to_index - 0.5 ));

                a = std::max(0, a);
                b = std::min(_cols_cell - 1, b);

                if (a > b) { row.x_min = 1; row.x_max = 0; continue; }
            }

            // Intersect with existing span
            row.x_min = std::max(row.x_min, a);
            row.x_max = std::min(row.x_max, b);
        }
    }
}

void IDMap::visitPointsInRowSpans(const std::vector<RowSpan>& spans,
                                  const std::vector<Line2D>&  lines_px,
                                  double                      tol_px,
                                  std::vector<int>&           out_indices) const
{
    out_indices.clear();

    if (!_objs || lines_px.empty() || spans.empty()) return;

    // Precompute unit directions (required by cross-product distance)
    std::vector<Pt2D> U(lines_px.size());
    for (size_t i = 0; i < lines_px.size(); ++i)
        U[i] = normalized(lines_px[i].unit_vector);

    const double tol2 = tol_px * tol_px;
    const auto& objs  = *_objs;

    // Dedup bitmap: a point may appear in multiple cells along the row-span
    std::vector<unsigned char> seen(objs.size(), 0);

    // Iterate every row of cells
    const int rowsC = _rows_cell;
    for (int cy = 0; cy < rowsC; ++cy) {
        const RowSpan& row = spans[cy];
        if (row.x_min > row.x_max) continue; // invalid span -> nothing on this row

        // Iterate each cell in the inclusive [x_min, x_max] range
        for (int cx = row.x_min; cx <= row.x_max; ++cx) {
            const auto& cell = _buckets[idx(cy, cx)];
            for (int pid : cell) {
                if (pid < 0 || static_cast<size_t>(pid) >= objs.size()) continue;
                if (seen[pid]) continue;

                const Pt2D& q = objs[pid]->_pt_center;

                // Precise distance check w.r.t. all lines: |(q - p) x u| <= tol
                for (size_t i = 0; i < lines_px.size(); ++i) {
                    const Pt2D& Lp = lines_px[i].pt;
                    const Pt2D& u  = U[i];
                    const double dx = q[0] - Lp[0];
                    const double dy = q[1] - Lp[1];
                    const double z  = dy * u[0] - dx * u[1]; // signed 2D cross (perp distance if u is unit)
                    if (z*z > tol2) goto reject_point;
                }

                // All distances within tol -> accept once
                seen[pid] = 1;
                out_indices.push_back(pid);

            reject_point:;
            }
        }
    }
}

// ---------------- ctor ----------------

// ---------------- ctor ----------------
MatchPruner::MatchPruner(const std::vector<Camera>& cams,
                         const std::vector<std::vector<const Object2D*>>& obj2d_list,
                         MatchPrunerOptions opts)
: _cams(cams), _obj2d(obj2d_list), _opt(opts)
{
    if (_cams.size() != _obj2d.size())
        throw std::runtime_error("MatchPruner: cams and obj2d_list size mismatch.");
    buildFlatIndexMap();         // fills _cam_offsets, _n_points_total
    // Build LOS cache once (can be rebuilt if inputs change)
    _los_cache.resize(_n_points_total);
    const int n_cams = int(_cams.size());
    for (int cam = 0; cam < n_cams; ++cam) {
        for (size_t pid = 0; pid < _obj2d[cam].size(); ++pid) {
            const size_t f = _cam_offsets[size_t(cam)] + pid;
            const Pt2D& q = _obj2d[cam][pid]->_pt_center;
            _los_cache[f] = _cams[cam].lineOfSight(q);
        }
    }
    _los_cached = true;
}

// ------------- public entry ------------
std::vector<std::vector<int>>
MatchPruner::prune(const std::vector<std::vector<int>>& match_candidates)
{
    if (match_candidates.empty()) return {};

    // 0) Candidate infos (flat 2D points + tri error) with validation
    buildCandidateInfos(match_candidates);

    if (_C.empty()) return {};

    // 1) Reverse index + scarcity (depends on _C, _cam_offsets)
    buildReverseIndexAndScarcity();

    // 2) Greedy packing (maximize count first)
    greedySelect();

    // 3) Local improvements (grow then quality swap)
    improveSelect();

    // 4) Materialize output
    std::vector<std::vector<int>> out;
    out.reserve(_chosen_list.size());
    for (int j : _chosen_list) out.push_back(_C[size_t(j)].ids);
    return out;
}

// ------------- pipeline steps -------------
void MatchPruner::buildCandidateInfos(const std::vector<std::vector<int>>& match_candidates)
{
    const int n_cams = int(_cams.size());
    _C.clear();
    _C.reserve(match_candidates.size());
    _scratch_los.clear();  _scratch_los.reserve(8);
    _scratch_flats.clear(); _scratch_flats.reserve(8);

    for (size_t i = 0; i < match_candidates.size(); ++i) {
        const auto& ids = match_candidates[i];
        if (int(ids.size()) != n_cams)
            throw std::runtime_error("MatchPruner: candidate size != n_cams.");

        int used = 0;
        for (int c = 0; c < n_cams; ++c) {
            const int pid = ids[c];
            if (pid < 0) continue;
            if (size_t(pid) >= _obj2d[c].size() || _obj2d[c][pid] == nullptr)
                throw std::runtime_error("MatchPruner: candidate has out-of-range or null 2D index.");
            ++used;
        }
        if (used < 2) continue; // Drop degenerate candidates (need >=2 views to triangulate)

        CandidateInfo ci;
        ci.idx = int(i);
        ci.ids = ids;

        _scratch_flats.clear();
        _scratch_los.clear();
        buildFlatsAndLOS(ids, _scratch_flats, _scratch_los);
        ci.point_flats = _scratch_flats;

        ci.error_mm = triErrorMM(_scratch_los);
        if (!std::isfinite(ci.error_mm)) continue; // Drop invalid error

        // Defensive dedup (normally unnecessary)
        std::sort(ci.point_flats.begin(), ci.point_flats.end());
        ci.point_flats.erase(std::unique(ci.point_flats.begin(), ci.point_flats.end()), ci.point_flats.end());

        _C.push_back(std::move(ci));
    }
}

void MatchPruner::buildFlatIndexMap()
{
    const int n_cams = int(_obj2d.size());
    _cam_offsets.assign(size_t(n_cams) + 1, 0);
    for (int c = 0; c < n_cams; ++c)
        _cam_offsets[size_t(c)+1] = _cam_offsets[size_t(c)] + _obj2d[c].size();
    _n_points_total = _cam_offsets.back();
}

void MatchPruner::buildReverseIndexAndScarcity()
{
    // reverse index
    _point_to_matchCandidates.assign(_n_points_total, {});
    for (size_t j = 0; j < _C.size(); ++j) {
        const CandidateInfo& ci = _C[j];
        for (size_t f : ci.point_flats) _point_to_matchCandidates[f].push_back(int(j)); // store internal index j
    }

    // degrees
    _point_degrees.resize(_n_points_total);
    for (size_t f = 0; f < _n_points_total; ++f)
        _point_degrees[f] = int(_point_to_matchCandidates[f].size());

    // scarcity = min degree among its 2D points
    for (CandidateInfo& ci : _C) {
        int sc = std::numeric_limits<int>::max();
        for (size_t f : ci.point_flats) sc = std::min(sc, _point_degrees[f]);
        ci.scarcity = (sc == std::numeric_limits<int>::max()) ? 0 : sc;
    }
}

void MatchPruner::greedySelect()
{
    // Greedy selection flow:
    //
    //   All candidates
    //       |
    //       v
    //   Sort by (scarcity ↑, error_mm ↑, views ↓, idx ↑)
    //       |
    //       v
    //   For each candidate in order:
    //       |
    //       +-- canPlace? (all 2D points free) --- No --> skip
    //       |                                      |
    //       Yes                                    |
    //       |                                      |
    //   place() mark points owned                  |
    //       |                                      |
    //   chosen_list.push_back                      |
    //       |                                      |
    //      next candidate <-----------------------+
    //       |
    //       v
    //   Initial chosen set (maximal, disjoint)

    const size_t M = _C.size();
    _owner.assign(_n_points_total, -1);
    _chosen.assign(M, 0);
    _chosen_list.clear(); _chosen_list.reserve(M/10 + 8);

    // Order: scarcity ↑, error ↑, views ↓, idx ↑
    std::vector<int> ord(M);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(),
              [&](int a, int b){
                  const auto& A = _C[size_t(a)];
                  const auto& B = _C[size_t(b)];
                  if (A.scarcity != B.scarcity) return A.scarcity < B.scarcity;
                  if (A.error_mm != B.error_mm) return A.error_mm < B.error_mm;
                  if (A.point_flats.size() != B.point_flats.size()) return A.point_flats.size() > B.point_flats.size();
                  return A.idx < B.idx;
              });

    for (int j : ord) {
        const CandidateInfo& ci = _C[size_t(j)];
        if (canPlace(ci)) {
            place(ci, j);            // store internal index j
            _chosen[size_t(j)] = 1;
            _chosen_list.push_back(j);
        }
    }
}

void MatchPruner::improveSelect()
{
    // Local improvement flow (improveSelect):
    //
    //  Repeat (A)+(B) over all unchosen X, up to max_passes times:
    //    |
    //    v
    //   (A) 1↔2 Growth:
    //         For each unchosen X:
    //           conflicts = collectConflicts(X)
    //           |
    //           +-- No conflict --> place X directly
    //           |
    //           +-- >1 conflict  --> skip
    //           |
    //           +-- Exactly 1 conflict Y:
    //                 freed_points = points(Y) \ points(X)
    //                 |
    //                 +-- freed_points empty:
    //                 |       if error(X) < error(Y): swap X for Y (1↔1 quality swap)
    //                 |
    //                 +-- freed_points non-empty:
    //                         pool = gatherTopKOnFreed(freed_points, topK)
    //                         unplace(Y)
    //                         if canPlace(X):
    //                             place(X)
    //                             find Z in pool that canPlace()
    //                             if found: place(Z), accept growth (replace Y with X+Z)
    //                             else: rollback X, restore Y
    //
    //    |
    //    v
    //   (B) 1↔1 Quality Swap (again):
    //         For each unchosen X:
    //           conflicts = collectConflicts(X)
    //           if exactly 1 conflict Y and error(X) < error(Y):
    //               swap X for Y
    //
    //    |
    //    +-- No change this pass? --> break

    std::vector<int> conflicts; conflicts.reserve(8);
    std::vector<size_t> freed;  freed.reserve(8);
    std::vector<int> pool;      pool.reserve(size_t(_opt.topK_per_point) * 2);

    for (int pass = 0; pass < _opt.max_passes; ++pass) {
        bool any_change = false;

        // (A) 1↔2 growth
        for (size_t xj = 0; xj < _C.size(); ++xj) {
            if (_chosen[xj]) continue;           // only unchosen X
            const CandidateInfo& X = _C[xj];

            collectConflicts(X, conflicts);
            if (conflicts.empty()) {
                // Rare after greedy; accept directly
                if (canPlace(X)) {
                    place(X, int(xj));
                    _chosen[xj] = 1;
                    _chosen_list.push_back(int(xj));
                    any_change = true;
                }
                continue;
            }
            if (conflicts.size() != 1) continue; // keep it simple and fast

            const int yj = conflicts[0];
            const CandidateInfo& Y = _C[size_t(yj)];

            // Freed 2D points = Points(Y) \ Points(X)
            freed.clear();
            for (size_t fy : Y.point_flats) {
                bool inX = false;
                for (size_t fx : X.point_flats) { if (fx == fy) { inX = true; break; } }
                if (!inX) freed.push_back(fy);
            }

            if (freed.empty()) {
                // 1↔1 quality swap (inline):
                // -------------------------
                // Case: X conflicts with exactly one chosen Y, and no 2D points are freed
                //       (X and Y occupy essentially the same set of 2D points).
                // Here, no growth is possible (cannot replace Y with X+Z), so if X has lower
                // triangulation error than Y, we swap X in for Y directly.
                // This is an opportunistic quality improvement done inside the 1↔2 loop.
                if (X.error_mm + 1e-12 < Y.error_mm) {
                    unplace(Y);
                    if (canPlace(X)) {
                        place(X, int(xj));
                        _chosen[yj] = 0;
                        _chosen[xj] = 1;
                        for (int& v : _chosen_list) if (v == yj) { v = int(xj); break; }
                        any_change = true;
                    } else {
                        place(Y, yj); // rollback
                    }
                }
                continue;
            }

            // 1↔2: remove Y; try X + Z from Top-K touching freed 2D points
            gatherTopKOnFreed(freed, _opt.topK_per_point, pool);

            bool grew = false;
            unplace(Y);

            if (canPlace(X)) {
                place(X, int(xj));

                for (int zj : pool) {
                    if (zj == yj || zj == int(xj) || _chosen[size_t(zj)]) continue;
                    const CandidateInfo& Z = _C[size_t(zj)];
                    if (!canPlace(Z)) continue;

                    // Accept growth: replace Y with X+Z
                    place(Z, zj);
                    _chosen[zj] = 1;
                    _chosen[xj] = 1;

                    for (auto it = _chosen_list.begin(); it != _chosen_list.end(); ++it) {
                        if (*it == yj) { _chosen_list.erase(it); break; }
                    }
                    _chosen_list.push_back(int(xj));
                    _chosen_list.push_back(zj);
                    any_change = true;
                    grew = true;
                    break;
                }

                if (!grew) {
                    unplace(X);
                    place(Y, yj);
                }
            } else {
                place(Y, yj);
            }
        }

        // (B) 1↔1 quality swap (full pass):
        // ---------------------------------
        // After all 1↔2 growth attempts are done, the chosen set may have changed,
        // unlocking new opportunities for 1↔1 quality swaps.
        // This pass systematically scans all unchosen X again:
        //   - If X conflicts with exactly one chosen Y
        //   - And X has strictly lower triangulation error than Y
        //   → Replace Y with X (same count, lower total error).
        // This ensures that after maximizing the count in (A), we also minimize
        // total error without reducing the match count.
        for (size_t xj = 0; xj < _C.size(); ++xj) {
            if (_chosen[xj]) continue;
            const CandidateInfo& X = _C[xj];
            collectConflicts(X, conflicts);
            if (conflicts.size() != 1) continue;

            const int yj = conflicts[0];
            const CandidateInfo& Y = _C[size_t(yj)];
            if (!(X.error_mm + 1e-12 < Y.error_mm)) continue;

            unplace(Y);
            if (canPlace(X)) {
                place(X, int(xj));
                _chosen[yj] = 0;
                _chosen[xj] = 1;
                for (int& v : _chosen_list) if (v == yj) { v = int(xj); break; }
                any_change = true;
            } else {
                place(Y, yj);
            }
        }

        if (!any_change) break; // converged
    }
}

// ------------- micro helpers -------------
void MatchPruner::buildFlatsAndLOS(const std::vector<int>& cand_ids,
                                   std::vector<size_t>& out_flats,
                                   std::vector<Line3D>& out_los) const
{
    out_flats.clear();
    out_los.clear();
    const int n_cams = int(_cams.size());

    for (int c = 0; c < n_cams; ++c) {
        const int pid = cand_ids[c];
        if (pid < 0) continue;
        const size_t f = _cam_offsets[size_t(c)] + size_t(pid);
        out_flats.push_back(f);
        out_los.push_back(_los_cached ? _los_cache[f] : _cams[c].lineOfSight(_obj2d[c][pid]->_pt_center));
    }

    // Defensive dedup
    std::sort(out_flats.begin(), out_flats.end());
    out_flats.erase(std::unique(out_flats.begin(), out_flats.end()), out_flats.end());
}

double MatchPruner::triErrorMM(const std::vector<Line3D>& los3d)
{
    if (los3d.size() < 2) return std::numeric_limits<double>::infinity();
    Pt3D pt; double err = 0.0;
    myMATH::triangulation(pt, err, los3d);
    return err;
}

bool MatchPruner::canPlace(const CandidateInfo& ci) const
{
    for (size_t f : ci.point_flats) if (_owner[f] != -1) return false;
    return true;
}

void MatchPruner::place(const CandidateInfo& ci, int internal_index)
{
    for (size_t f : ci.point_flats) _owner[f] = internal_index; // store _C index
}

void MatchPruner::unplace(const CandidateInfo& ci)
{
    for (size_t f : ci.point_flats) _owner[f] = -1;
}

void MatchPruner::collectConflicts(const CandidateInfo& X, std::vector<int>& out) const
{
    out.clear();
    // X uses ~m 2D points; dedup linearly (m very small, ~2–4)
    for (size_t f : X.point_flats) {
        const int j = _owner[f];
        if (j < 0) continue;
        bool dup = false;
        for (int v : out) if (v == j) { dup = true; break; }
        if (!dup) out.push_back(j);
    }
}

void MatchPruner::gatherTopKOnFreed(const std::vector<size_t>& freed_flats, int K,
                                    std::vector<int>& out_sorted_unique) const
{
    out_sorted_unique.clear();
    std::unordered_set<int> U;
    U.reserve(size_t(freed_flats.size()) * size_t(K) + 16);

    // For each freed 2D point, locally take best-K by error, then merge & dedup.
    for (size_t f : freed_flats) {
        const auto& matchList = _point_to_matchCandidates[f];
        if ((int)matchList.size() <= K) {
            for (int j : matchList) U.insert(j);
        } else {
            std::vector<int> tmp = matchList;
            std::nth_element(tmp.begin(), tmp.begin()+K, tmp.end(),
                [&](int a, int b){ return _C[size_t(a)].error_mm < _C[size_t(b)].error_mm; });
            tmp.resize(K);
            for (int j : tmp) U.insert(j);
        }
    }

    out_sorted_unique.assign(U.begin(), U.end());
    std::sort(out_sorted_unique.begin(), out_sorted_unique.end(),
              [&](int a, int b){ return _C[size_t(a)].error_mm < _C[size_t(b)].error_mm; });
    if ((int)out_sorted_unique.size() > K) out_sorted_unique.resize(K);
}