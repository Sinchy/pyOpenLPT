#include "ObjectInfo.h"

bool Object3D::isReconstructable(const std::vector<Camera>& cam_list) 
{
    int n_outrange = 0;
    projectObject2D(cam_list);

    double x, y;
    int n_cam = cam_list.size();
    for (int i = 0; i < n_cam; i ++)
    {
        x = _obj2d_list[i]->_pt_center[0];
        y = _obj2d_list[i]->_pt_center[1];

        if (x < 0 || x >= cam_list[i].getNCol() ||
            y < 0 || y >= cam_list[i].getNCol())
        {
            n_outrange ++;

            // if less than 2 camera can see it, then it is not reconstructable
            if (n_cam - n_outrange < 2)
            {
                return false;
            }
        }
    }

    return true;

}

void Object3D::projectObject2D(std::vector<Camera> const& cam_list)
{
    int n_cam = cam_list.size(); 
    _obj2d_list.clear();
    _obj2d_list.reserve(n_cam); // reserve space for n_cam elements

    for (int i = 0; i < n_cam; i ++)
    {
        auto obj2d = create2DObject(); // create a 2D object, this will call the derived class's create2DObject
        obj2d->_pt_center = cam_list[i].project(_pt_center); // _pt2d_list[i] is a pointer to Object2D
        _obj2d_list.push_back(std::move(obj2d)); // move the object to the vector
    }
    // project additional 2D information
    // for example, bubbles need to project the radius
    additional2DProjection(cam_list);
}

void Tracer3D::additional2DProjection(std::vector<Camera> const& cam_list)
{
    // This function can be overridden by derived classes to add additional 2D projections
    for (int i = 0; i < _obj2d_list.size(); i ++)
    {
        auto* tr2D = static_cast<Tracer2D*>(_obj2d_list[i].get()); // Check if the pointer is of type Tracer2D
        if (tr2D)
        {
            tr2D->_r_px = _r2d_px; // this updates the radius for each 2D tracer
        }

    }
}

void Bubble3D::additional2DProjection(std::vector<Camera> const& cam_list)
{
    // This function can be overridden by derived classes to add additional 2D projections
    for (int i = 0; i < _obj2d_list.size(); i ++)
    {
        auto* bb2D = static_cast<Bubble2D*>(_obj2d_list[i].get()); // Check if the pointer is of type Bubble2D
        if (bb2D)
        {
            bb2D->_r_px = Bubble::cal2DRadius(cam_list[i], _pt_center, _r3d);
        }
    }
}

void Tracer3D::saveObject3D(std::ostream& out) const
{
    // 3D center
    out << _pt_center[0] << "," << _pt_center[1] << "," << _pt_center[2];

    // 2D projections: fixed-width (2 cols per camera)
    const size_t n_cam = _obj2d_list.size();
    for (size_t i = 0; i < n_cam; ++i)
    {
        if (auto* tr2d = dynamic_cast<Tracer2D*>(_obj2d_list[i].get())) {
            out << "," << tr2d->_pt_center[0] << "," << tr2d->_pt_center[1];
        } else {
            // keep column count constant even if dynamic type is unexpected
            out << ",,";
        }
    }

    out << "\n";
}

void Bubble3D::saveObject3D(std::ostream& output) const
{
    // 3D center and radius
    output << _pt_center[0] << "," << _pt_center[1] << "," << _pt_center[2]
           << "," << _r3d;

    // 2D projections: fixed-width (3 cols per camera)
    const size_t n_cam = _obj2d_list.size();
    for (size_t i = 0; i < n_cam; ++i)
    {
        if (auto* bb2d = dynamic_cast<Bubble2D*>(_obj2d_list[i].get())) {
            output << "," << bb2d->_pt_center[0] << "," << bb2d->_pt_center[1]
                   << "," << bb2d->_r_px;
        } else {
            // placeholders to keep the line width constant
            output << ",,,";
        }
    }

    output << "\n";
}

// Tracer3D.cpp
void Tracer3D::loadObject3D(std::istream& in)
{
    std::string sx, sy, sz;
    REQUIRE(read_csv_field(in, sx) && read_csv_field(in, sy) && read_csv_field(in, sz),
            ErrorCode::IOfailure, "Tracer3D::loadObject3D: insufficient 3D columns");

    _pt_center[0] = std::stod(sx);
    _pt_center[1] = std::stod(sy);
    _pt_center[2] = std::stod(sz);
    // 2D will be generated later via projectObject2D().
}

// Bubble3D.cpp
void Bubble3D::loadObject3D(std::istream& in)
{
    std::string sx, sy, sz, sr;
    REQUIRE(read_csv_field(in, sx) && read_csv_field(in, sy) && read_csv_field(in, sz) && read_csv_field(in, sr),
            ErrorCode::IOfailure, "Bubble3D::loadObject3D: insufficient 3D/R3D columns");
    
    _pt_center[0] = std::stod(sx);
    _pt_center[1] = std::stod(sy);
    _pt_center[2] = std::stod(sz);
    _r3d          = std::stod(sr);
    // 2D will be generated later via projectObject2D().
}

namespace Bubble {

// Exact 3D radius from one view (pinhole), pixel-domain.
// R = d * k / sqrt(1 + k^2), where k = r_px / f_px, d = ||X - C||.
double calRadiusFromOneCam(const Camera& cam,
                           const Pt3D&   X,
                           double        r_px)
{
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    if (!(r_px > 0.0)) return NaN;

    switch (cam._type) {
    case CameraType::PINHOLE:
    {
        // --- Access pinhole parameters (pixels intrinsics; C = camera center in world) ---
        // TODO: adjust accessor to your actual Camera API
        const double fx = cam._pinhole_param.cam_mtx(0, 0);         // pixels
        const double fy = cam._pinhole_param.cam_mtx(1, 1);         // pixels
        if (!(fx > 0.0) || !(fy > 0.0)) return NaN;

        // Use geometric mean for generality (handles non-square pixels robustly).
        const double f_px = std::sqrt(fx * fy);

        // Camera center in world coordinates (per your struct: t_vec_inv)
        const Pt3D& C = cam._pinhole_param.t_vec_inv;               // world units, same as X/R

        // Distance d = ||X - C|| (same units as desired R)
        Pt3D dX = X - C;
        const double d  = dX.norm();
        if (!(d > 0.0)) return NaN;

        // Exact inversion from pinhole geometry
        const double k     = r_px / f_px;
        const double denom = std::sqrt(1.0 + k*k);
        if (!(denom > 0.0)) return NaN;

        return d * k / denom;
    }

    case CameraType::POLYNOMIAL:
        // TODO: polynomial — use Jacobian at X: R ≈ r_px / sigma_max(J(X))
        return NaN;

    case CameraType::PINPLATE:
    default:
        // TODO: not implemented
        return NaN;
    }
}

// Forward projection: predict image radius r_px from 3D radius R.
// Pinhole (exact): r_px = f_px * R / sqrt(d^2 - R^2),
//   where f_px is pixel focal length, d = ||X - C||.
// Returns NaN if unsupported or inputs invalid.
double cal2DRadius(const Camera& cam,
                   const Pt3D&   X,
                   double        R)
{
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    if (!(R > 0.0) || !std::isfinite(R)) return NaN;

    switch (cam._type) {
    case CameraType::PINHOLE:
    {
        // --- Access pinhole parameters (pixel intrinsics; camera center C in world) ---
        // TODO: adjust accessor to your actual Camera API
        const double fx = cam._pinhole_param.cam_mtx(0, 0);         // pixels
        const double fy = cam._pinhole_param.cam_mtx(1, 1);         // pixels
        if (!(fx > 0.0) || !(fy > 0.0)) return NaN;

        // Use geometric mean for generality (handles non-square pixels robustly).
        const double f_px = std::sqrt(fx * fy);

        // Camera center in world coordinates
        const Pt3D& C = cam._pinhole_param.t_vec_inv;

        // Distance from camera to bubble center
        const double d = (X - C).norm();
        if (!std::isfinite(d) || !(d > R)) {
            // invalid or degenerate: object at/inside the camera center sphere of radius R
            return NaN;
        }

        // Exact forward formula
        const double denom_sq = d*d - R*R;
        if (!(denom_sq > 0.0)) return NaN; // extra guard
        const double r_px = f_px * R / std::sqrt(denom_sq);

        return std::isfinite(r_px) ? r_px : NaN;
    }

    case CameraType::POLYNOMIAL:
        // TODO: polynomial — use local Jacobian at X: r_px ≈ sigma_max(J(X)) * R
        return NaN;

    case CameraType::PINPLATE:
    default:
        // TODO: not implemented
        return NaN;
    }
}

// Multi-view 3D radius estimate for a final accepted match.
// Inputs:
//   cams          : per-view cameras (aligned with obj2d_by_cam by index)
//   X             : 3D bubble center (world coords)
//   obj2d_by_cam  : per-view Object2D pointers; nullptr if that camera has no 2D
// Behavior:
//   - For each view that has a 2D bubble, read r_px from Bubble2D,
//     call calRadiusFromOneCam(cam_i, X, r_px), and collect valid R_i.
//   - Aggregate R_i with the median (robust).
// Returns NaN if no usable views.
double calRadiusFromCams(const std::vector<Camera>&                     cams,
                         const Pt3D&                                    X,
                         const std::vector<std::unique_ptr<Object2D>>&  obj2d_list)
{
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    if (cams.empty() || cams.size() != obj2d_list.size())
        return NaN;

    std::vector<double> Rs;
    Rs.reserve(cams.size());

    for (size_t i = 0; i < cams.size(); ++i) {
        const Camera&   cam  = cams[i];
        const Object2D* base = obj2d_list[i].get();
        if (!base) continue; // no camera or no 2D for this view

        // Container is homogeneous (Bubble2D) per your guarantee → static_cast is fine.
        const Bubble2D* bb = static_cast<const Bubble2D*>(base);
        const double r_px = bb->_r_px;    
        if (!(r_px > 0.0) || !std::isfinite(r_px)) continue;

        const double Ri = calRadiusFromOneCam(cam, X, r_px); // dispatches by cam._type
        if (std::isfinite(Ri) && (Ri > 0.0)) {
            Rs.push_back(Ri);
        }
    }

    if (Rs.empty()) return NaN;

    // Robust aggregation: median of R_i
    const size_t n   = Rs.size();
    const size_t mid = n / 2;
    std::nth_element(Rs.begin(), Rs.begin() + mid, Rs.end());
    double median = Rs[mid];
    if ((n % 2) == 0) {
        const auto max_lhs = std::max_element(Rs.begin(), Rs.begin() + mid);
        median = 0.5 * (median + *max_lhs);
    }

    return median;
}

// Cross-view consistency gate using per-view radius inversion from Object2D list.
// Policy: if fewer than 2 usable views → return true (permissive).
// Dynamic threshold is derived from tol_2d/tol_3d via RMS combination.
bool checkRadiusConsistency(const Pt3D&                          X,
                            const std::vector<const Camera*>&    cams,
                            const std::vector<const Object2D*>&  obj2d_by_cam,
                            double                               tol_2d_px,  // <=0 to disable
                            double                               tol_3d)     // <=0 to disable
{
    if (cams.empty() || cams.size() != obj2d_by_cam.size())
        return true; // permissive on size mismatch

    double Rmin =  std::numeric_limits<double>::infinity();
    double Rmax = -std::numeric_limits<double>::infinity();
    size_t n_ok = 0;

    // Track worst-case relative error bound across views to derive a dynamic spread gate.
    double eps_max = 0.0;
    bool   eps_has_data = false;

    for (size_t i = 0; i < cams.size(); ++i) {
        const Camera*   cam  = cams[i];
        const Object2D* base = obj2d_by_cam[i];
        if (!cam || !base) continue;

        // Container is homogeneous (Bubble2D) per your guarantee → static_cast is fine.
        const Bubble2D* bb = static_cast<const Bubble2D*>(base);
        const double r_px = bb->_r_px;                    // TODO: replace with accessor if available
        if (!(r_px > 0.0) || !std::isfinite(r_px)) continue;

        // Per-view 3D radius via model-dispatched inversion (PINHOLE implemented).
        const double Ri = calRadiusFromOneCam(*cam, X, r_px);
        if (std::isfinite(Ri) && (Ri > 0.0)) {
            Rmin = std::min(Rmin, Ri);
            Rmax = std::max(Rmax, Ri);
            ++n_ok;

            // --- Build dynamic threshold contributions (PINHOLE only for now) ---
            if (cam->_type == CameraType::PINHOLE) {
                // Access intrinsics (pixels) and camera center in world
                const double fx = cam->_pinhole_param.cam_mtx(0,0);
                const double fy = cam->_pinhole_param.cam_mtx(1,1);
                if (fx > 0.0 && fy > 0.0) {
                    const double f_px = std::sqrt(fx * fy); // or use fx if that's your convention
                    const double d_i  = (X - cam->_pinhole_param.t_vec_inv).norm();
                    if (std::isfinite(d_i) && d_i > 0.0) {
                        const double k = r_px / f_px;

                        // Relative error components:
                        //   δR/R ≈ (δr/r)/(1+k^2)  (from exact pinhole derivative wrt r)
                        //   δR/R ≈ δd/d            (sensitivity wrt distance)
                        double eps_r = 0.0, eps_d = 0.0;
                        if (tol_2d_px > 0.0 && r_px > 0.0) {
                            eps_r = (tol_2d_px / r_px) / (1.0 + k*k);
                        }
                        if (tol_3d > 0.0) {
                            eps_d = tol_3d / d_i;
                        }

                        // Combine contributions:
                        // Recommended default: RMS (assume independence)
                        const double eps_i = std::hypot(eps_r, eps_d); // sqrt(eps_r^2 + eps_d^2)
                        if (std::isfinite(eps_i)) {
                            eps_max = std::max(eps_max, eps_i);
                            eps_has_data = true;
                        }
                    }
                }
            }
        }
    }

    if (n_ok < 2) return true;                 // not enough evidence → permissive
    if (!(Rmin > 0.0) || !(Rmax > 0.0)) return true;

    const double spread = Rmax / Rmin;

    // Dynamic gate from tolerances if we have data; otherwise use a conservative default.
    double max_spread = 1.35;                  // fallback default
    if (eps_has_data && (tol_2d_px > 0.0 || tol_3d > 0.0)) {
        // Keep ε within a sane range for the (1+ε)/(1-ε) mapping
        double eps = std::min(std::max(eps_max, 0.0), 0.49); // clamp to avoid blow-up
        max_spread = (1.0 + eps) / (1.0 - eps);
        // avoid being unrealistically tight for tiny eps
        if (max_spread < 1.05) max_spread = 1.05;
    }

    return (spread <= max_spread);
}



} // namespace bubble
