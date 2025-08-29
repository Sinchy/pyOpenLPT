#ifndef LIBSTB_CONFIG_H
#define LIBSTB_CONFIG_H

#include <iostream>
#include <fstream>
#include <vector>
#include <optional>
#include <string>

#include "ImageIO.h"
#include "ObjectInfo.h"
#include "Camera.h"
#include "PredField.h"
#include "OTF.h"
#include "BubbleRefImg.h"

class BasicSetting {
public:
    bool readConfig(const std::string& config_file);

    int _n_cam;
    int _frame_start = 0;
    int _frame_end;
    int _fps;
    int _n_thread;
    std::vector<Camera> _cam_list; 
    AxisLimit _axis_limit;
    double _voxel_to_mm;
    std::string _output_path;
    std::vector<std::string> _image_file_paths;
    std::vector<std::string> _object_types;
    std::vector<std::string> _object_config_paths;
    bool _load_track = false;
    int _load_track_frame = _frame_start;
    std::string _load_track_path;
};

// IPR parameters definition
struct IPRParam
{
    int n_cam_reduced = 1; // number of cameras to reduce
    int n_loop_ipr = 4; // 
    int n_loop_ipr_reduced = 2; // number of shake times after reducing cameras
    int n_obj2d_process_max = 200000; // maximum number of 2D objects to process at once

};

// predictive field parameter
struct PFParam
{
    // X,Y,Z limits of view area
    AxisLimit limit;
    // # of grids in X,Y,Z >=2
    int nx, ny, nz;
    
    // radius of search sphere
    //  usually set: r <= 0.5*min(dx,dy,dz)
    //  it is better to make sure displacement < 0.5 * r
    double r;
    
    // number of bins for displacement statistics
    int nBin_x = 11; // >=2, if <2, set to 2
    int nBin_y = 11; // >=2, if <2, set to 2
    int nBin_z = 11; // >=2, if <2, set to 2

    // smoothing param
    bool is_smooth = true;
    double sigma_x = 1.0;
    double sigma_y = 1.0;
    double sigma_z = 1.0;
};

// StereoMatch configuration (minimal)
struct SMParam {
    int    match_cam_count   = 3;   // number of cams to build per candidate (>=2, <= #active)
    int    idmap_cell_px     = 4;   // cell size in pixels for IDMap bucketing
    double tol_2d_px         = 1.0; // strip half-width in pixels (used inside IDMap)
    double tol_3d_mm         = 2.4e-2; // final triangulation tolerance (mm)
    // ... add other knobs if needed
};

// Shaking configuration
struct ShakeParam {
    double _shake_width;
    int _n_shake_loop = 4; // number of iterations for shaking
    double _thred_ghost = 0.1; // threshold for ghost object removal
    double _shakewidth_min; // the minimum shake width during shaking loops
    int _ratio_augimg = 2; // the ratio of augmented image size to the object size
}; 

// STB configuration
struct STBParam {
        // Initial Phase
    double _radius_search_obj; // mm, radius to search for objects in the initial phase
    int _n_initial_frames = 4; //should be a number larger than 4
        // Convergence Phase
    double _radius_search_track; // mm, radius to search for neighbors in the convergence phase
};

// object configuration
enum class ObjectKind { Tracer, Bubble };

// This is used for creating an object
struct CreateArgs {
    const Object3D* _proto = nullptr;                 // optional: copy common/type-specific fields
    std::optional<Pt3D> _pt_center;                   // optional: override center

    std::vector<std::unique_ptr<Object2D>> _obj2d_ready; // optional: prebuilt 2D list (will be moved)

    // Renamed hints per your request
    std::optional<double> _r2d_px_tracer;             // for Tracer
    std::optional<double> _r3d_bubble;                // for Bubble

    // Optional for Bubble radius computation
    const std::vector<Camera>* _cams = nullptr;
    bool _compute_bubble_radius = false;
};

// This is a base class for different object configurations
class ObjectConfig {
public:
    // common configuration parameters for all types of objects
    int _n_thread = 1; // number of threads
    // STB parameters
    STBParam _stb_param;

    // Shake
    ShakeParam _shake_param;

    // Predict Field
    PFParam _pf_param; // parameters for the predicted field

    // IPR
    IPRParam _ipr_param; // IPR parameters

    // Stereomatch
    SMParam _sm_param;

    // reading common configuration parameters
    std::pair<std::vector<std::string>, int> readCommonConfig(const std::string& filepath, BasicSetting& settings);

    virtual ~ObjectConfig() = default;
    // read specific configuration parameters for the object type
    virtual bool readConfig(const std::string& filepath, BasicSetting& settings) = 0;

    // use for object type identification
    virtual ObjectKind kind() const = 0; 

    // creat object based on object configuration
    virtual std::unique_ptr<Object3D> creatObject3D(CreateArgs args) const = 0;
};

// Tracer configuration
class TracerConfig : public ObjectConfig {
public:
    // Object Info
    int _min_obj_int; // minimum object intensity
    int _radius_obj; // radius of the object in pixels;

    OTF _otf; // OTF parameters for tracer projection

    bool readConfig(const std::string& filepath, BasicSetting& settings) override;

    ObjectKind kind() const override { return ObjectKind::Tracer; }

    std::unique_ptr<Object3D> creatObject3D(CreateArgs args) const override;
};

// Bubble configuration
class BubbleConfig : public ObjectConfig {
public:
    int _radius_min; // minimum bubble radius in pixels
    int _radius_max; // maximum bubble radius in pixels
    double _sense = 0.85; // sensitivity for bubble detection

    BubbleRefImg _bb_ref_img; // bubble reference image for shaking
    
    bool readConfig(const std::string& filepath, BasicSetting& settings) override;

    ObjectKind kind() const override { return ObjectKind::Bubble; }

    std::unique_ptr<Object3D> creatObject3D(CreateArgs args) const override;
};

#endif // LIBSTB_CONFIG_H