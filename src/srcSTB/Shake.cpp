#include <algorithm>
#include <cassert>
#include <omp.h>
#include <cmath>    // ★ floor, ceil, exp, cos, sin, fabs, round
#include <limits>   // ★ std::numeric_limits

#include "nanoflann.hpp"
#include "Shake.h"
#include "BubbleResize.h"
#include "CircleIdentifier.h"
#include "BubbleRefImg.h"

#define CORR_INIT -100

// ---------------------------------------------------
Shake::Shake(const std::vector<Camera>& cams, const ObjectConfig& obj_cfg):
        _cams(cams), _obj_cfg(obj_cfg) 

{
    // Ensure strategy exists (construct elsewhere or plug in before calling)
    // Resolve by ObjectConfig::kind() only
    switch (_obj_cfg.kind()) {
        case ObjectKind::Tracer:
            _strategy = std::make_unique<TracerShakeStrategy>(_cams, _obj_cfg);
            break;
        case ObjectKind::Bubble:
            _strategy = std::make_unique<BubbleShakeStrategy>(_cams, _obj_cfg);
            break;
        default:
            THROW_FATAL(ErrorCode::UnsupportedType, "Shake::ensureStrategy: unsupported ObjectKind in ObjectConfig.");
    }
}


std::vector<ObjFlag>
Shake::runShake(std::vector<std::unique_ptr<Object3D>>& objs,
                const std::vector<Image>& img_orig)
{
    // Basic sanity checks
    const int n_cam = static_cast<int>(_cams.size());
    assert(n_cam == static_cast<int>(img_orig.size()) && "cams/img_orig size mismatch");

    const size_t n_obj = objs.size();
    std::vector<ObjFlag> flags(n_obj, ObjFlag::None);

    _img_res_list.clear();
    // Allocate residuals and per-object scores
    _img_res_list.assign(n_cam, Image{});
    _score_list.assign(n_obj, 1.0);

    // Shake schedule
    double delta           = _obj_cfg._shake_param._shake_width;   // initial Δ (mm)
    double dmin            = _obj_cfg._shake_param._shakewidth_min;
    const int n_loop       = _obj_cfg._shake_param._n_shake_loop;

    dmin = (dmin > 0) ? dmin : delta/20;

    // 1) Ensure each object has up-to-date 2D projections for all cameras.
    //    NOTE: Replace the call below with your actual signature if different.
    for (auto& up : objs) {
        if (!up) continue;
        up->projectObject2D(_cams); 
    }

    for (int it = 0; it < n_loop; ++it) {
        // 1) Build residual images for this iteration (Jacobi-style baseline)
        //    Residual fusion must follow strategy->fuseMode(): Overwrite or Min.
        calResidueImage(objs, img_orig);

        // OpenMP parallel region (fallback to serial if OpenMP is not enabled)
        const int n_thread = _obj_cfg._n_thread;
        #pragma omp parallel num_threads(n_thread)
        {
            #pragma omp for schedule(dynamic, 8)
            for (std::ptrdiff_t i_obj = 0; i_obj < static_cast<std::ptrdiff_t>(n_obj); ++i_obj) {
                size_t i = static_cast<size_t>(i_obj);
                if (!objs[i]) continue;

                // 2.1 Build per-object Aug (ROI = residual ROI + this object's projection).
                std::vector<ROIInfo> roi = buildROIInfo(*objs[i], img_orig);

                // 2.2 Shake one object with current Δ (search {-Δ,0,+Δ} per axis, quad fit, refine).
                //     Implementation detail: inside, only consider cameras that are:
                //     active && ROI non-empty && selected by strategy->selectShakeCam(...).

                // first, we need to determine which camera is used for shaking
                std::vector<bool> shake_cam = _strategy->selectShakeCam(*objs[i], roi, img_orig);
                (void) shakeOneObject(*objs[i], roi, delta, shake_cam);

                // 2.3 Compute object score (cross-camera aggregation inside your calObjectScore).
                double score = calObjectScore(*objs[i], roi);
                _score_list[i] *= score;
            }

        } // end parallel region

        // 4) Δ schedule (halve each loop; clamp to delta_min)
        delta *= 0.5;
        if (delta < dmin) delta = dmin;
    }

    // 5) Post-processing — mark repeated
    std::vector<bool> is_repeated = markRepeatedObj(objs);
    for (size_t i = 0; i < is_repeated.size(); ++i) {
        if (is_repeated[i]) flags[i] |= ObjFlag::Repeated;
    }

    // 6) Post-processing — mark ghosts
    //    Here we threshold by relative-to-mean rule: score < score_min * mean(score)
    double sum = 0.0;
    size_t cnt = 0;
    for (size_t i = 0; i < _score_list.size(); ++i) {
        if (is_repeated[i]) continue;
        sum += _score_list[i]; ++cnt;
    }
    const double mean_score = (cnt ? sum / cnt : 0.0);
    const double percent_ghost    = _obj_cfg._shake_param._thred_ghost; 

    for (size_t i = 0; i < n_obj; ++i) {
        if (!objs[i]) continue;
        if (_score_list[i] < mean_score * percent_ghost) {
            flags[i] |= ObjFlag::Ghost;
        }
    }

    return flags;
}


void Shake::calResidueImage(const std::vector<std::unique_ptr<Object3D>>& objs,
                            const std::vector<Image>& img_orig,
                            bool non_negative, const std::vector<ObjFlag>* flags)
{
    const int n_cam = static_cast<int>(_cams.size());
    assert(n_cam == static_cast<int>(img_orig.size()));
    const bool use_mask = (flags != nullptr);
    if (use_mask) {
        REQUIRE(objs.size() == flags->size(), ErrorCode::InvalidArgument,
                "The number of objects doesn't match the number of object flags!");
    }

    // 2) Copy originals to residual buffers (one per camera)
    _img_res_list = img_orig;  // deep copy; if your Image doesn't support operator= as deep copy, use explicit copy

    // 4) Parallelize across cameras (each thread owns one residual image)
    //    You can also parallelize rows inside the camera loop if Images are big.
    if (_obj_cfg._n_thread > 0) omp_set_num_threads(_obj_cfg._n_thread);
    #pragma omp parallel for schedule(static,1)
    for (int k = 0; k < n_cam; ++k) {
        // 4.1 Skip inactive cameras, but keep slot alignment (residual stays as original)
        if (!_cams[k]._is_active) continue;

        Image&       res  = _img_res_list[k]; // get the reference
        const Image& orig = img_orig[k];

        // 4.2 For each object, subtract its projection over its ROI only
        for (size_t id_obj = 0; id_obj < objs.size(); ++id_obj) {
            if (use_mask) {
                if ((*flags)[id_obj] != ObjFlag::None) continue;  // skip ghost and repeated objects (flagged objects)
            }
            const Object3D* obj = objs[id_obj].get();

            // --- Obtain ROI center from object's 2D projection; size from strategy ---
            Pt2D pt_center = obj->_obj2d_list[k]->_pt_center;
            double cx = pt_center[0], cy = pt_center[1];

            // Strategy returns (dx, dy) = half width/height (in pixels) for the ROI
            // If your calROISize returns std::vector<double>, read [0],[1]; or change to a struct.
            const auto sz = _strategy->calROISize(*obj, k);
            double dx = sz.dx, dy = sz.dy;

            // Compute and clamp ROI bounds to the image
            // projection size: one object size
            const PixelRange roi = calROIBound(k, cx, cy, dx, dy);
            if (roi.row_max < roi.row_min || roi.col_max < roi.col_min) continue; // empty ROI

            // 4.2.1 Iterate pixels in ROI and fuse projection into residual
            //       NOTE: replace res.at(r,c) / orig.at(r,c) / projection accessor with your actual image API.
            for (int r = roi.row_min; r < roi.row_max; ++r) {
                for (int c = roi.col_min; c < roi.col_max; ++c) {
                    const double p = _strategy->project2DInt(*obj, k, r, c);
                    if (p == 0.0) continue;             // cheap skip
                    // "Min" fusion for all types:
                    // residual := min(current residual, orig - projection_of_this_object)
                    double& rr = res(r, c);
                    const double o = orig(r, c);
                    const double cand = o - p;
                    if (cand < rr) rr = cand;
                    if (non_negative && rr < 0) rr = 0.0;
                }
            }
        } // end for each object
    } // end per-camera loop
}

PixelRange Shake::calROIBound(int id_cam, double cx, double cy, double dx, double dy) const
{
    const int H = _cams[id_cam].getNRow();           
    const int W = _cams[id_cam].getNCol();

    if (H <= 0 || W <= 0 || dx <= 0.0 || dy <= 0.0) {
        return PixelRange{1, 0, 1, 0};  // empty range
    }

    // compute in double, then cast once to int after floor/ceil
    const double rmin_d = std::floor(cy - dy);
    const double rmax_d = std::ceil (cy + dy + 1);
    const double cmin_d = std::floor(cx - dx);
    const double cmax_d = std::ceil (cx + dx + 1);

    int rmin = static_cast<int>(std::max(0.0,          rmin_d));
    int cmin = static_cast<int>(std::max(0.0,          cmin_d));
    int rmax = static_cast<int>(std::min<double>(H,    rmax_d));
    int cmax = static_cast<int>(std::min<double>(W,    cmax_d));

    if (rmax < rmin || cmax < cmin) {
        return PixelRange{1, 0, 1, 0};
    }
    return PixelRange{rmin, rmax, cmin, cmax};  // [min, max)
}

std::vector<ROIInfo> Shake::buildROIInfo(const Object3D& obj,
                            const std::vector<Image>& img_orig) const
{
    const int n_cam = static_cast<int>(_cams.size());
    std::vector<ROIInfo> roi_info;
    roi_info.resize(n_cam);

    for (int k = 0; k < n_cam; ++k) {
        // keep slot alignment
        if (!_cams[k]._is_active ||
            k >= static_cast<int>(obj._obj2d_list.size()) ||
            !obj._obj2d_list[k]) {
            continue; // do not initialize ROI info
        }

        const Pt2D& ctr = obj._obj2d_list[k]->_pt_center;   // (x=col, y=row)

        // ROI half size from strategy (假设已改成 ROISize {dx,dy})
        const auto sz = _strategy->calROISize(obj, k);
        const double dx = std::max(0.0, sz.dx);
        const double dy = std::max(0.0, sz.dy);

        double ratio_region = _obj_cfg._shake_param._ratio_augimg;
        const PixelRange roi = calROIBound(k, ctr[0], ctr[1], dx * ratio_region , dy * ratio_region);
        roi_info[k]._ROI_range = roi;

        if (roi.row_max < roi.row_min || roi.col_max < roi.col_min) {
            continue;
        }

        // allocate Aug & corr map
        roi_info[k].allocAugImg();
        roi_info[k].allocCorrMap();
        
        // creating augmented image
        // project back the object but within the range of project (one object size)
        const PixelRange range_project = calROIBound(k, ctr[0], ctr[1], dx, dy);
        for (int r = roi.row_min; r < roi.row_max; ++r)
            for (int c = roi.col_min; c < roi.col_max; ++c) {
                double aug = _img_res_list[k](r, c);
                if (r >= range_project.row_min && r < range_project.row_max &&
                    c >= range_project.col_min && c < range_project.col_max)
                    aug += _strategy->project2DInt(obj, k, r, c);
                
                roi_info[k].aug_img(r, c) = std::max(0.0, aug); // augmented image should be non-negative
        }

    }
    return roi_info;
}

// shake one object, input delta is the shake width, return the score for updating, obj is also updated
double Shake::shakeOneObject(Object3D& obj, std::vector<ROIInfo>& ROI_info, double delta, const std::vector<bool>& shake_cam) const
{
    std::vector<double> delta_list = { -delta, 0.0, +delta };
    std::vector<double> array_list(4, 0.0);          // x0,x1,x2,x*
    std::vector<double> array_list_fit(3, 0.0);      // for polyfit (x0,x1,x2)
    std::vector<double> coeff(3, 0.0);               // quad coeffs
    std::vector<double> residue_list(4, 0.0);        // f0,f1,f2,f*
    std::vector<double> residue_list_fit(3, 0.0);    // for polyfit

    // shaking on x,y,z direction
    double residue = 0.0;
    std::unique_ptr<Object3D> obj3d_temp;

    switch (_obj_cfg.kind())
    {
    case ObjectKind::Tracer:
        auto& tr = static_cast<Tracer3D&>(obj);
        obj3d_temp = std::make_unique<Tracer3D>(tr._pt_center, tr._r2d_px);
        break;
    case ObjectKind::Bubble:
        auto& bb = static_cast<Bubble3D&>(obj);
        obj3d_temp = std::make_unique<Bubble3D>(bb._pt_center, bb._r3d);
        break;
    default:
        THROW_FATAL(ErrorCode::UnsupportedType, "Shake: unsupported ObjectKind to shake.");
    }

    for (int i = 0; i < 3; i ++)
    {
        for (int j = 0; j < 3; j ++)
        {
            array_list[j] = obj._pt_center[i] + delta_list[j];
            array_list_fit[j] = array_list[j];

            obj3d_temp->_pt_center[i] = array_list[j];
            
            // update 2D information
            obj3d_temp->projectObject2D(_cams); 

            residue_list[j] = _strategy->calShakeResidue(*obj3d_temp, ROI_info, shake_cam);
            residue_list_fit[j] = residue_list[j];
        }
        
        // residue = coeff[0] + coeff[1] * x + coeff[2] * x^2
        myMATH::polyfit(coeff, array_list_fit, residue_list_fit, 2);

        // 计算区间内的顶点 x*
        bool has_star = false;
        if (coeff[2] != 0.0) {
            array_list[3] = - coeff[1] / (2.0 * coeff[2]);
            if (array_list[3] > array_list[0] && array_list[3] < array_list[2]) {
                obj3d_temp->_pt_center[i] = array_list[3];
                obj3d_temp->projectObject2D(_cams);
                residue_list[3] = _strategy->calShakeResidue(*obj3d_temp, ROI_info, shake_cam);
                has_star = true;
            } else {
                residue_list[3] = std::numeric_limits<double>::infinity(); // safer than "+1"
            }
        } else {
            residue_list[3] = std::numeric_limits<double>::infinity();
        }

        // 选最小残差的位置（含 x* 如有效）
        int min_id = 0;
        double min_val = residue_list[0];
        int t_max = (has_star ? 4 : 3);
        for (int t = 1; t < t_max; ++t) {
            if (residue_list[t] < min_val) { min_val = residue_list[t]; min_id = t; }
        }
        
        // update obj
        obj3d_temp->_pt_center[i] = array_list[min_id];
        obj._pt_center[i] = array_list[min_id];
        residue = residue_list[min_id];
    }

    // TODO: remove camera with residue > thredshold, and redo shakeoneobject

    // update 2D information for next loop of shaking
    obj.projectObject2D(_cams);

    return residue;
}

// calculate score based on the intensity
double Shake::calObjectScore(Object3D& obj, const std::vector<ROIInfo>& ROI_info) const
{
    const int n_cam = static_cast<int>(_cams.size());
    constexpr double kTiny = 1e-12;   // avoid 0
    constexpr double kDown = 0.1;     // if camera number is less than 2, return this

    // per-camera
    std::vector<PixelRange> score_rect(n_cam); // region for evaluating the intensity
    std::vector<int>        use_cam(n_cam, 0); // evaluate cameras only without the highest intensity
    std::vector<double>     peak(n_cam, 0.0);

    int n_used = 0;
    for (int k = 0; k < n_cam; ++k) {
        if (!_cams[k]._is_active) continue;

        // region: one object size
        const auto sz = _strategy->calROISize(obj, k);
        const double dx = std::max(0.0, sz.dx);
        const double dy = std::max(0.0, sz.dy);
        if (dx <= 0.0 || dy <= 0.0) continue;

        const Pt2D& ctr = obj._obj2d_list[k]->_pt_center; // (x=col, y=row)
        score_rect[k]   = calROIBound(k, ctr[0], ctr[1], dx, dy);

        const bool empty = (score_rect[k].row_max < score_rect[k].row_min) || (score_rect[k].col_max < score_rect[k].col_min);
        if (empty) continue; // if no region found on camera, skip this camera

        use_cam[k] = 1; ++n_used;

        // get the highest peak intensity for this camera
        for (int r = score_rect[k].row_min; r < score_rect[k].row_max; ++r) {
            for (int c = score_rect[k].col_min; c < score_rect[k].col_max; ++c) {
                const double val = ROI_info[k].inRange(r, c)
                                   ? ROI_info[k].aug_img(r, c)    
                                   : _img_res_list[k](r, c);     // residual may contain negative value, but it doesn't matter here
                if (val > peak[k]) peak[k] = val;
            }
        }
    }

    // if less than 2 camera
    if (n_used < 2) return kDown;

    // remove one of the camera with highest intensity, if camera number > 2
    if (n_used > 2) {
        int    max_id = -1;
        double max_v  = -std::numeric_limits<double>::infinity();
        for (int k = 0; k < n_cam; ++k) {
            if (use_cam[k] && peak[k] > max_v) { max_v = peak[k]; max_id = k; }
        }
        if (max_id >= 0) { use_cam[max_id] = 0; --n_used; }
    }

    // calculate intensity ratio for each camera r_k = |sum_measured / sum_model|，and use geometric mean
    double ratio = 1.0;    // for geometric mean it should be 1.0

    for (int k = 0; k < n_cam; ++k) {
        if (!use_cam[k]) continue;
        const auto& R = score_rect[k];

        double numer = 0.0;    // sum of measured intensity
        double denom = 0.0;    // sum of predicted intensity

        for (int r = R.row_min; r <= R.row_max; ++r) {
            for (int c = R.col_min; c <= R.col_max; ++c) {
                // measured：ROI 内用 Aug，外用 residual（可能为负）
                double meas = ROI_info[k].inRange(r, c)
                                    ? ROI_info[k].aug_img(r, c)
                                    : _img_res_list[k](r, c);
                meas = std::max(0.0, meas); // since residual can be negative due to overlap particle, we set it 0
                numer += meas;

                // predict intensity, using same projection model as in calResidualImage
                const double pred = _strategy->project2DInt(obj, k, r, c);
                denom += pred;
            }
        }

        const double rk = std::abs(numer / denom);   // 单相机亮度比值（≥0）
        ratio *= rk;
    }

    return ratio; 
}

// mark repeated objects
std::vector<bool> Shake::markRepeatedObj(const std::vector<std::unique_ptr<Object3D>>& objs)
{
    int n_obj = objs.size();
    std::vector<bool> is_repeated;
    is_repeated.resize(n_obj, false);
    int n_repeated = 0;

    // Build KD tree for fast neighbor search
    using KDTreeObj3d = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, Obj3dCloud>,
        Obj3dCloud,
        3 // dimensionality
    >;
    Obj3dCloud obj3d_cloud(objs);
    KDTreeObj3d tree_obj3d(3, obj3d_cloud, {10});
    tree_obj3d.buildIndex();

    // Remove repeated tracks
    double tol_3d = _obj_cfg._sm_param.tol_3d_mm;
    double repeat_thres_2 = tol_3d * tol_3d;
    for (int i = 0; i < n_obj - 1; i ++)
    {
        if (is_repeated[i])
        {
            continue;
        }

        std::vector<nanoflann::ResultItem<size_t, double>> indices_dists;
        nanoflann::RadiusResultSet<double, size_t> resultSet(repeat_thres_2, indices_dists);
        tree_obj3d.findNeighbors(resultSet, objs[i]->_pt_center.data(), nanoflann::SearchParameters());
        
        for (int j = 1; j < resultSet.size(); j ++)
        {
            is_repeated[indices_dists[j].first] = 1;
            n_repeated ++;
        }
    }
    return is_repeated;
}


//------------------------------ROIINfo----------------------------//
// allocate augmented image for cam id_cam
void ROIInfo::allocAugImg()
{
    const int roi_h = _ROI_range.getNumOfRow();
    const int roi_w = _ROI_range.getNumOfCol();
    _ROI_augimg     = Image(roi_h, roi_w, 0.0);
}

void ROIInfo::allocCorrMap()
{
    const int roi_h = _ROI_range.getNumOfRow();
    const int roi_w = _ROI_range.getNumOfCol();
    _ROI_corrmap     = Image(roi_h, roi_w, CORR_INIT); // set value to #define CORR_INIT -100
}

bool ROIInfo::inRange(int row, int col) const
{
    if (_ROI_range.getNumOfRow() <= 0 || _ROI_range.getNumOfCol() <= 0) 
        return false;

    // 包含判断（左闭右开）
    if (row < _ROI_range.row_min || row >= _ROI_range.row_max ||
        col < _ROI_range.col_min || col >= _ROI_range.col_max) {
        return false;
    }

    return true;
}

bool ROIInfo::mapToLocal(int row, int col, int& i, int& j) const
{
    if (!inRange(row, col)) return false;

    // 全局(row,col) → 局部(i,j)
    i = row - _ROI_range.row_min;
    j = col - _ROI_range.col_min;
    return true;
}

double& ROIInfo::aug_img(int row, int col)
{
    int i = -1, j = -1;
    const bool inside = mapToLocal(row, col, i, j);
    assert(inside && "aug_img(): (row,col) is outside ROI for this camera or ROI is empty");

    return _ROI_augimg(i, j);
}

const double& ROIInfo::aug_img(int row, int col) const
{
    int i = -1, j = -1;
    const bool inside = mapToLocal(row, col, i, j);
    assert(inside && "aug_img() const: (row,col) is outside ROI for this camera or ROI is empty");

    return _ROI_augimg(i, j);
}

double& ROIInfo::corr_map(int row, int col)
{
    int i = -1, j = -1;
    const bool inside = mapToLocal(row, col, i, j);
    assert(inside && "corr_map(): (row,col) is outside ROI for this camera or ROI is empty");

    return _ROI_corrmap(i, j);
}

const double& ROIInfo::corr_map(int row, int col) const
{
    int i = -1, j = -1;
    const bool inside = mapToLocal(row, col, i, j);
    assert(inside && "corr_map() const: (row,col) is outside ROI for this camera or ROI is empty");

    return _ROI_corrmap(i, j);
}

//----------------------------TracerShakeStrategy---------------------------
double TracerShakeStrategy::gaussIntensity(int x, int y, Pt2D const& pt2d, std::vector<double> const& otf_param) const
{
    double dx = x - pt2d[0];
    double dy = y - pt2d[1];
    double xx =  dx * std::cos(otf_param[3]) + dy * std::sin(otf_param[3]);
    double yy = -dx * std::sin(otf_param[3]) + dy * std::cos(otf_param[3]);
    double value = otf_param[0] * std::exp(- otf_param[1] * (xx*xx) - otf_param[2] * (yy*yy));
    return std::max(0.0, value);
}

double TracerShakeStrategy::project2DInt(const Object3D& obj, int id_cam, int row, int col) const 
{
    const auto& tr_cfg = static_cast<const TracerConfig&>(_obj_cfg);

    std::vector<double> otf_param = tr_cfg._otf.getOTFParam(id_cam, obj._pt_center);

    return gaussIntensity(col, row, obj._obj2d_list[id_cam]->_pt_center, otf_param);
}

ShakeStrategy::ROISize TracerShakeStrategy::calROISize(const Object3D& obj, int id_cam) const 
{
    ROISize roi_size;
    Object2D* obj2d = obj._obj2d_list[id_cam].get(); // must get the raw pointer from the unique_ptr because unique_ptr cannot be casted directly
    auto* tr = static_cast<Tracer2D*>(obj2d); 
    roi_size.dx = tr->_r_px;
    roi_size.dy = tr->_r_px;

    return roi_size;
}

double TracerShakeStrategy::calShakeResidue(const Object3D& obj_candidate, std::vector<ROIInfo>& ROI_info, const std::vector<bool>& shake_cam) const
{
    // For Tracer: per-camera MSE on the fixed ROI, then arithmetic mean over usable cameras.
    // Usable camera = ROI non-empty (cameras already filtered upstream by _is_active when building ROI).
    // Read measured intensity from ROI Aug via roi_info.aug_img(cam, row, col) using FULL image indices.
    // Predicted intensity comes from project2DInt(obj_candidate, cam, row, col) (≥0; 3σ cutoff inside).
    // If no usable camera exists, return a large sentinel cost.

    const int n_cam = static_cast<int>(_cams.size());
    int cams_used = 0;
    double cost_acc = 0.0;

    for (int cam = 0; cam < n_cam; ++cam) {
        if (!_cams[cam]._is_active) continue;
        if (!shake_cam[cam]) continue;
        const PixelRange& ROI_range = ROI_info[cam]._ROI_range;
        const int nrows = ROI_range.getNumOfRow();
        const int ncols = ROI_range.getNumOfCol();

        // Skip empty ROI
        if (nrows <= 0 || ncols <= 0) continue;

        double sse = 0.0;
        std::size_t pix = 0;
        
        // this is used for seting range of calculating intensity
        Object2D* obj2d = obj_candidate._obj2d_list[cam].get();
        auto* tr = static_cast<Tracer2D*>(obj2d);
        Pt2D obj2d_center = tr->_pt_center;
        double r_px = tr->_r_px;
        int pred_row_min = std::floor(obj2d_center[1] - r_px), pred_row_max = std::ceil(obj2d_center[1] + r_px + 1);
        int pred_col_min = std::floor(obj2d_center[0] - r_px), pred_col_max = std::ceil(obj2d_center[0] + r_px + 1);

        // Iterate over the ROI in FULL image coordinates (left-closed, right-open: [min, max))
        for (int row = ROI_range.row_min; row < ROI_range.row_max; ++row) {
            for (int col = ROI_range.col_min; col < ROI_range.col_max; ++col) {
                // Measured Aug value from ROI (the accessor maps full (row,col) to local (i,j) with checks)
                const double meas = ROI_info[cam].aug_img(row, col);

                double pred = 0;
                // Forward model prediction for this pixel (Gaussian PSF; returns 0 outside object size)
                if (row >= pred_row_min && row < pred_row_max &&
                    col >= pred_col_min && col < pred_col_max)
                    pred = project2DInt(obj_candidate, cam, row, col);

                const double d = meas - pred;
                sse += d * d;
                ++pix;
            }
        }

        // Guard against degenerate ROIs (should not happen if ranges are valid)
        if (pix > 0) {
            // Per-camera MSE to avoid scale drift with different ROI sizes
            const double mse = sse / static_cast<double>(pix);
            cost_acc += mse;
            ++cams_used;
        }
    }

    // Aggregate across cameras: arithmetic mean; if none usable, return a large penalty
    return (cams_used > 0) 
           ? (cost_acc / static_cast<double>(cams_used)) 
           : std::numeric_limits<double>::infinity();
}

//----------------------------BubbleShakeStrategy---------------------------

double BubbleShakeStrategy::project2DInt(const Object3D& obj, int id_cam, int row, int col) const 
{
    Object2D* obj2d = obj._obj2d_list[id_cam].get();
    auto* bb2d = static_cast<Bubble2D*>(obj2d);
    double xc = bb2d->_pt_center[0], yc = bb2d->_pt_center[1];
    double r_px = bb2d->_r_px;

    double dist = (xc - col) * (xc - col) + (yc - row) * (yc - row);
    double int_val = 0.0;
    if (dist < r_px * r_px)
    {
        int_val = _cams[id_cam]._max_intensity;
    }

    return int_val;
}

ShakeStrategy::ROISize BubbleShakeStrategy::calROISize(const Object3D& obj, int id_cam) const
{
    Object2D* obj2d = obj._obj2d_list[id_cam].get();
    auto* bb2d = static_cast<Bubble2D*>(obj2d);
    double r_px = bb2d->_r_px;
    ROISize roi_size;
    roi_size.dx = r_px; roi_size.dy = r_px;
    return roi_size;
}

std::vector<bool> BubbleShakeStrategy::selectShakeCam(const Object3D& obj, const std::vector<ROIInfo>& roi_info, const std::vector<Image>& imgOrig) const
{
    const int n_cam = _cams.size();
    std::vector<bool> shake_cam;
    shake_cam.assign(n_cam, true);

    for (int cam = 0; cam < n_cam; ++cam) {
        if (!_cams[cam]._is_active) {shake_cam[cam] = false; continue;}

        const PixelRange& region = roi_info[cam]._ROI_range;

        // 1) ROI validity (half-open [min, max))
        const int n_row = region.getNumOfRow();
        const int n_col = region.getNumOfCol();
        if (n_row <= 0 || n_col <= 0) {
            shake_cam[cam] = false;
            continue;
        }

        // 2) Extract original subimage (ROI)
        Image imgOrig_sub(n_row, n_col, 0.0);
        for (int i = 0; i < n_row; ++i) {
            for (int j = 0; j < n_col; ++j) {
                imgOrig_sub(i, j) = imgOrig[cam](region.row_min + i, region.col_min + j);
            }
        }

        // 3) Get bubble 2D info on this camera
        const Object2D* base = obj._obj2d_list[cam].get();
        const auto* bb2d = dynamic_cast<const Bubble2D*>(base);
        const double r_px = bb2d->_r_px;

        // 4) Run circle detection (small-radius branch uses centered square crop + 2x upsampling)
        std::vector<Pt2D>   centers;
        std::vector<double> radii;
        std::vector<double> metrics;
        double rmin = 2.0, rmax = 0.0, sense = 0.95;

        if (r_px < 5.0) {
            // 4.a) center-crop to a square: size = min(n_row, n_col)
            const int npix = std::min(n_row, n_col);
            Image img_ref(npix, npix, 0.0);

            const int r0 = (n_row - npix) / 2;
            const int c0 = (n_col - npix) / 2;
            for (int y = 0; y < npix; ++y)
                for (int x = 0; x < npix; ++x)
                    img_ref(y, x) = imgOrig_sub(r0 + y, c0 + x);

            // 4.b) 2x upsample
            const int img_size = 2 * npix;
            BubbleResize bb_resizer;
            const Image& img_up = bb_resizer.ResizeBubble(img_ref, img_size, _cams[cam]._max_intensity);

            // 4.c) detect circles on upsampled image
            CircleIdentifier circle_id(img_up);
            rmin = 2.0;
            rmax = std::ceil(r_px) * 2.0 + 6.0;      // keep your original heuristic
            sense = 0.95;
            metrics = circle_id.BubbleCenterAndSizeByCircle(centers, radii, rmin, rmax, sense);

            // 4.d) restore to original scale
            for (size_t i = 0; i < centers.size(); ++i) {
                centers[i] *= 0.5;
                radii[i]   *= 0.5;
            }
        } else {
            // large-radius branch: detect directly on ROI subimage
            CircleIdentifier circle_id(imgOrig_sub);
            rmin = 2.0;
            rmax = std::ceil(r_px) + 3.0;            // keep your original heuristic
            sense = 0.95;
            metrics = circle_id.BubbleCenterAndSizeByCircle(centers, radii, rmin, rmax, sense);
        }

        // 5) Decide validity
        const int n_bb = static_cast<int>(radii.size());
        bool use_cam = false;

        if (n_bb == 0) {
            // Fallback: consider the case of "filled by a big bubble".
            // Keep your original mean-intensity gate using reference intensity.
            double int_orig = 0.0;
            for (int i = 0; i < n_row; ++i)
                for (int j = 0; j < n_col; ++j)
                    int_orig += imgOrig_sub(i, j);
            int_orig /= static_cast<double>(n_row * n_col);

            auto& bb_cfg = static_cast<const BubbleConfig&>(_obj_cfg);
            const BubbleRefImg& bb_refimg = bb_cfg._bb_ref_img;
            const double intRef = bb_refimg.getIntRef(cam); 
            if (int_orig < intRef * 1.2 && int_orig > intRef * 0.8) {
                use_cam = true; // within [0.8, 1.2] * intRef
            }
        } else {
            // At least one detected bubble close to expected radius and with sufficient metric
            for (int ci = 0; ci < n_bb; ++ci) {
                const double cr = radii[ci];
                const bool ok = (std::fabs(cr - r_px) < std::min(0.3 * r_px, 2.0)) &&
                                (metrics[ci] > 0.1);
                if (ok) { use_cam = true; break; }
            }
        }

        shake_cam[cam] = use_cam;
    }

    return shake_cam;
}

double BubbleShakeStrategy::getImgCorr(ROIInfo& roi_info, const int x, const int y, const Image& ref_img) const
{
    auto computeCorr = [&]() -> double {
        const double cx = static_cast<double>(x - roi_info._ROI_range.col_min);
        const double cy = static_cast<double>(y - roi_info._ROI_range.row_min);
        return myMATH::imgCrossCorrAtPt(roi_info.getAugImg(), ref_img, cx, cy);
    };

    double corr = 0.0;
    if (roi_info.inRange(y, x)){
        // if inside the correlation map, then we can update or get from corr_map
        if (roi_info.corr_map(y, x) < -1) {
            // location on augmented image
            corr = computeCorr();
            roi_info.corr_map(y, x) = corr; //update corr_map
        } else {
            corr = roi_info.corr_map(y, x);
        }
    } else {
        // if outside the map, we have to calculate it and can't update to corr_map
        // location on augmented image
        corr = computeCorr();
    }
    
    return corr;
}

double BubbleShakeStrategy::calShakeResidue(const Object3D& obj_candidate, std::vector<ROIInfo>& roi_info, const std::vector<bool>& shake_cam) const
{
    const int n_cam = static_cast<int>(_cams.size());
    int cams_used = 0;
    double residue = 0;

    for (int cam = 0; cam < n_cam; ++cam) {
        if (!_cams[cam]._is_active) continue;
        if (!shake_cam[cam]) continue;

        Object2D* obj2d = obj_candidate._obj2d_list[cam].get();
        const auto* bb2d = static_cast<const Bubble2D*>(obj2d);
        double xc = bb2d->_pt_center[0], yc = bb2d->_pt_center[1];
        double r_px = bb2d->_r_px;
        const PixelRange& range_corrmap = roi_info[cam]._ROI_range;

        // the candidate bubble can be partly within the augmented image
        // however, if the bubble is total out of the augmented image, then residue = 1
        // ---- Early-out: bubble completely outside augmented image ----
        // Check overlap between the ref patch (centered at (xc,yc)) and the ROI image.
        // Use ROI-local center for the test.
        const double cx_local = xc - range_corrmap.col_min;
        const double cy_local = yc - range_corrmap.row_min;
        const double safe_factor = 0.8;
        const double half_w   = r_px * safe_factor; // geometric half-size (nearest-neighbor NCC)
        const double half_h   = r_px * safe_factor;

        const bool no_overlap =
            (cx_local <= -half_w) || (cx_local >= range_corrmap.getNumOfCol() + half_w) ||
            (cy_local <= -half_h) || (cy_local >= range_corrmap.getNumOfRow() + half_h);

        if (no_overlap) continue; //skip this camera
        
        // calcualte correlation of 4 pixel around xc, yc and do interpolation for sub-pixel accuracy 
        int x_low = std::floor(xc); int x_high = x_low + 1;
        int y_low = std::floor(yc); int y_high = y_low + 1;

        // get reference image 
        const int r_int = std::round(r_px); 
        int npix = r_int * 2 + 1; // guarantee there is only a whole center pixel on ref_img
        const auto& bb_cfg = static_cast<const BubbleConfig&>(_obj_cfg); // to get the bubble reference image
        BubbleResize bb_resizer;
        const Image& ref_img = bb_resizer.ResizeBubble(bb_cfg._bb_ref_img[cam], npix, _cams[cam]._max_intensity);
        
        // calculate cross-correlation
        std::vector<double> corr_interp(4,0);
        corr_interp[0] = getImgCorr(roi_info[cam], x_low, y_low, ref_img); 
        corr_interp[1] = getImgCorr(roi_info[cam], x_high, y_low, ref_img); 
        corr_interp[2] = getImgCorr(roi_info[cam], x_high, y_high, ref_img);
        corr_interp[3] = getImgCorr(roi_info[cam], x_low, y_high, ref_img);

        // bilinear interpolation
        AxisLimit grid_limit(x_low, x_high, y_low, y_high, 0,0);
        std::vector<double> center = {xc, yc};
        double res = 1 - myMATH::bilinearInterp(
            grid_limit, corr_interp, center
        );
        residue += res;
        cams_used ++;
    }
    
    residue = cams_used ? residue/cams_used : 2;
    return residue;
}