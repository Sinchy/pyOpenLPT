
#include "ObjectFinder.h"

std::vector<std::unique_ptr<Object2D>>
ObjectFinder2D::findObject2D(Image const& img, ObjectConfig const& obj_cfg)
{
    switch (obj_cfg.kind())
    {
    case ObjectKind::Tracer:
        return findTracer2D(img, static_cast<TracerConfig const&>(obj_cfg));

    case ObjectKind::Bubble:
        return findBubble2D(img, static_cast<BubbleConfig const&>(obj_cfg));

    default:
        // for unsupported object types, return an empty vector
        return {};
    }
}

/**
 * @brief Detect tracer particles in a 2D image and return them as Object2D pointers.
 *
 * This function scans the input image for local intensity maxima above a minimum threshold
 * and refines their positions to sub-pixel accuracy using a 3-point logarithmic parabola fit
 * in both x and y directions. Each detected tracer is returned as a Tracer2D object stored
 * in a unique_ptr<Object2D>.
 *
 * @param img   The input image.
 * @param cfg   TracerConfig containing detection parameters:
 *              - _radius_obj: expected particle radius in pixels
 *              - _min_obj_int: minimum intensity threshold
 * @return      A vector of unique_ptr<Object2D> pointing to detected Tracer2D objects.
 */
std::vector<std::unique_ptr<Object2D>>
ObjectFinder2D::findTracer2D(Image const& img, TracerConfig const& cfg)
{
    const int rows = img.getDimRow();
    const int cols = img.getDimCol();
    const double r_px = cfg._radius_obj;
    const double min_intensity = cfg._min_obj_int;

    // Estimate max possible number of particles based on density and image size
    constexpr double particle_density = 0.125; // estimated particles per (2*r)^2 area
    size_t estimated_count = static_cast<size_t>(
        (rows * cols) * particle_density / ((2.0 * r_px) * (2.0 * r_px))
    );

    std::vector<std::unique_ptr<Object2D>> out;
    out.reserve(estimated_count);

    auto safe_ln = [](double v) {
        const double vv = (v < LOGSMALLNUMBER) ? LOGSMALLNUMBER : v;
        return std::log(vv);
    };

    for (int row = 1; row < rows - 1; ++row)
    {
        for (int col = 1; col < cols - 1; ++col)
        {
            const double centerI = static_cast<double>(img(row, col));
            if (centerI < min_intensity) continue;
            if (!myMATH::isLocalMax(img, row, col)) continue;

            const int x1 = col - 1, x2 = col, x3 = col + 1;
            const int y1 = row - 1, y2 = row, y3 = row + 1;

            // --- X direction fit ---
            const double ln_z1x = safe_ln(static_cast<double>(img(y2, x1)));
            const double ln_z2  = safe_ln(centerI); // center pixel
            const double ln_z3x = safe_ln(static_cast<double>(img(y2, x3)));

            const double num_x =   ln_z1x * ((x2 * x2) - (x3 * x3))
                                 - ln_z2  * ((x1 * x1) - (x3 * x3))
                                 + ln_z3x * ((x1 * x1) - (x2 * x2));
            const double den_x =   ln_z1x * (x3 - x2)
                                 - ln_z3x * (x1 - x2)
                                 + ln_z2  * (x1 - x3);
            if (den_x == 0.0) continue;
            const double xc = -0.5 * (num_x / den_x);
            if (!std::isfinite(xc)) continue;

            // --- Y direction fit ---
            const double ln_z1y = safe_ln(static_cast<double>(img(y1, x2)));
            const double ln_z3y = safe_ln(static_cast<double>(img(y3, x2)));

            const double num_y =   ln_z1y * ((y2 * y2) - (y3 * y3))
                                 - ln_z2  * ((y1 * y1) - (y3 * y3))
                                 + ln_z3y * ((y1 * y1) - (y2 * y2));
            const double den_y =   ln_z1y * (y3 - y2)
                                 - ln_z3y * (y1 - y2)
                                 + ln_z2  * (y1 - y3);
            if (den_y == 0.0) continue;
            const double yc = -0.5 * (num_y / den_y);
            if (!std::isfinite(yc)) continue;

            auto tracer = std::make_unique<Tracer2D>();
            tracer->_r_px = r_px;
            tracer->_pt_center[0] = xc;
            tracer->_pt_center[1] = yc;

            out.emplace_back(std::move(tracer));
        }
    }

    // Free unused reserved space
    out.shrink_to_fit();
    return out;
}


/**
 * @brief Detect bubbles via circular fitting and return as Object2D pointers.
 *
 * This function uses CircleIdentifier to locate bubble centers and radii within
 * the specified radius range. Each detected bubble is wrapped as a Bubble2D and
 * returned through unique_ptr<Object2D> to preserve polymorphism.
 *
 * @param img  Input image.
 * @param cfg  BubbleConfig containing detection parameters:
 *             - _radius_min, _radius_max: allowed radius range in pixels
 *             - _sense: detector sensitivity (higher -> more detections)
 * @return     Vector of unique_ptr<Object2D> pointing to Bubble2D objects.
 */
std::vector<std::unique_ptr<Object2D>>
ObjectFinder2D::findBubble2D(Image const& img, BubbleConfig const& cfg)
{
    // Basic parameter sanity checks (optional but helpful)
    if (cfg._radius_min > cfg._radius_max) {
        // Swap or early return; here we choose early return
        return {};
    }

    CircleIdentifier circle_id(img);

    std::vector<Pt2D> center;
    std::vector<double> radius;

    const double sense = cfg._sense;  // use config, do not hardcode

    circle_id.BubbleCenterAndSizeByCircle(
        center, radius, cfg._radius_min, cfg._radius_max, sense
    );

    std::vector<std::unique_ptr<Object2D>> out;
    out.reserve(center.size());

    for (size_t i = 0; i < center.size(); ++i)
    {
        // Preserve subclass type by constructing Bubble2D and storing as base pointer
        out.emplace_back(std::make_unique<Bubble2D>(center[i], radius[i]));
    }

    // Optional: release extra capacity (usually small)
    out.shrink_to_fit();
    return out;
}
