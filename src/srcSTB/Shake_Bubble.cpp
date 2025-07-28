#ifndef SHAKE_BUBBLE_CPP
#define SHAKE_BUBBLE_CPP

#include "Shake.h"
#define CORR_INIT -100
#define RESIDUE_THRES 0.7 // 1 - 0.3

// Bubbles //

void Shake::runShake(std::vector<Bubble3D>& bb3d_list, std::vector<Image> const& imgOrig_list, std::vector<Image> const& imgRef_list, bool tri_only)
{
    shakeBubbles(bb3d_list, imgOrig_list, imgRef_list, tri_only);
}

void Shake::shakeBubbles(std::vector<Bubble3D>& bb3d_list, std::vector<Image> const& imgOrig_list, std::vector<Image> const& imgRef_list, bool tri_only)
{
    // update tr2d position
    int n_bb3d = bb3d_list.size();

    if (_n_thread > 0)
    {
        omp_set_num_threads(_n_thread);
    }
    #pragma omp parallel for
    for (int i = 0; i < n_bb3d; i ++)
    {
        bb3d_list[i].projectObject2D(_cam_list.useid_list, _cam_list.cam_list);
    }

    // Initialize lists
    _imgRes_list.clear();
    _is_ghost.resize(n_bb3d, 0);
    _n_ghost = 0;

    // Initialize residue image
    int cam_id;
    for (int id = 0; id < _n_cam_use; id ++)
    {
        cam_id = _cam_list.useid_list[id];
        _imgRes_list.push_back(imgOrig_list[cam_id]);
    }

    // if only do triangulation, then skip the following steps
    if (tri_only)
    {
        calResImg(bb3d_list, imgRef_list, imgOrig_list);
        absResImg();
        return;
    }

    // Calculate average intensity of reference image list
    std::vector<double> intRef_list(imgRef_list.size(), 0);
    for (int i = 0; i < imgRef_list.size(); i++)
    {
        int n_row_ref = imgRef_list[i].getDimRow();
        int n_col_ref = imgRef_list[i].getDimCol();
        int n_sum = 0;
        double xc = (n_col_ref - 1) / 2.0;
        double yc = (n_row_ref - 1) / 2.0;
        double r = n_col_ref / 2.0;
        for (int row = 0; row < n_row_ref; row++)
        {
            for (int col = 0; col < n_col_ref; col++)
            {
                double dist = std::sqrt(std::pow(row - yc, 2) + std::pow(col - xc, 2));
                if (dist < r) {
                    intRef_list[i] += imgRef_list[i](row, col);
                    n_sum++;
                }
            }
        }
        if (n_sum > 0) {
            intRef_list[i] /= n_sum;
        }
    }


    // Initialize score list
    _score_list.resize(n_bb3d);
    std::fill(_score_list.begin(), _score_list.end(), 1);

    double delta;
    for (int loop = 0; loop < _n_loop; loop ++)
    {
        // update residue img
        calResImg(bb3d_list, imgRef_list, imgOrig_list);

        // update shake width
        if (loop < 1)
        {
            delta = _shake_width;
        }
        // else if (loop < 32)
        else if (loop < 5)
        {
            delta = _shake_width / std::pow(2, loop-1);  
            // delta = _shake_width / std::pow(10, loop);   
        }
        else 
        {
            delta = _shake_width / 20;
        }

        // shake each tracer
        if (_n_thread != 0)
        {
            omp_set_num_threads(_n_thread);
        }
        #pragma omp parallel for
        for (int i = 0; i < n_bb3d; i ++)
        {
            if (_score_list[i] > _SCORE_MISMATCH + 1)
            {
                _score_list[i] = shakeOneBubble(bb3d_list[i], imgRef_list, intRef_list, imgOrig_list, delta, _score_list[i]);
            }
        }
        
    }

    // remove ghost bubbles
    findGhost(bb3d_list);

    // update residue image
    calResImg(bb3d_list, imgRef_list, imgOrig_list);

    // calculate absolute intensity value
    absResImg();
}

void Shake::calResImg(std::vector<Bubble3D> const& bb3d_list, std::vector<Image> const& imgRef_list, std::vector<Image> const& imgOrig_list)
{
    int n_bb3d = bb3d_list.size();
    int cam_id, row_min, row_max, col_min, col_max, img_size;
    double xc, yc, r_px;
    double dist, int_val, residue;
    BubbleResize bb_resizer;

    // initialize residue image
    for (int id = 0; id < _n_cam_use; id ++)
    {
        cam_id = _cam_list.useid_list[id];
        _imgRes_list[id] = imgOrig_list[cam_id];
    }

    // remove all bubbles
    double ratio_region = 1;
    for (int i = 0; i < n_bb3d; i ++)
    {
        if (_is_ghost[i])
        {
            continue;
        }

        for (int id = 0; id < _n_cam_use; id ++)
        {
            cam_id = _cam_list.useid_list[id];

            xc = bb3d_list[i]._bb2d_list[id]._pt_center[0];
            yc = bb3d_list[i]._bb2d_list[id]._pt_center[1];
            r_px = bb3d_list[i]._bb2d_list[id]._r_px;  
            PixelRange res_region = findRegion(
                id, 
                yc, // row_id
                xc, // col_id
                r_px * ratio_region
            );
            
            row_min = res_region.row_min;
            row_max = res_region.row_max;
            col_min = res_region.col_min;
            col_max = res_region.col_max;

            for (int row = row_min; row < row_max; row ++)
            {
                for (int col = col_min; col < col_max; col ++)
                {
                    dist = (xc - col) * (xc - col) + (yc - row) * (yc - row);
                    if (dist < r_px * r_px)
                    {
                        int_val = _cam_list.intensity_max[cam_id];
                    }
                    else
                    {
                        int_val = 0;
                    }
                    residue = imgOrig_list[cam_id](row, col) - int_val;
                    
                    if (residue < _imgRes_list[id](row, col))
                    {
                        _imgRes_list[id](row, col) = residue;
                    }
                    // _imgRes_list[id](row, col) = residue;
                }
            }
        }
    }
}

double Shake::shakeOneBubble(Bubble3D& bb3d, std::vector<Image> const& imgRef_list, std::vector<double> const& intRef_list, std::vector<Image> const& imgOrig_list, double delta, double score_old)
{
    ImgAugList imgAug_list; // augmented image list
    std::vector<Image> corr_map_list; // cross-correlation map for each camera
    
    // check if the camera is valid for shaking 
    //  and calculate the augmented image
    int ratio_region = 2;
    std::vector<int> cam_useid_mismatch;
    for (int id = 0; id < _n_cam_use; id ++)
    {
        int cam_id = _cam_list.useid_list[id];

        // Get the augmented image range   
        double xc = bb3d._bb2d_list[id]._pt_center[0];
        double yc = bb3d._bb2d_list[id]._pt_center[1];
        double r_px = bb3d._bb2d_list[id]._r_px;
        PixelRange region = findRegion(id, yc, xc, r_px * ratio_region
        );

        // check if this cam is valid for shaking
        bool is_valid = isCamValidForShaking(
            cam_id, region, imgRef_list[cam_id], intRef_list[cam_id], imgOrig_list[cam_id], bb3d._bb2d_list[id]
        );
        if (!is_valid)
        {
            cam_useid_mismatch.push_back(id);
            continue;
        }

        // create a particle reproj image (I_p) matrix in the pixel range
        int n_row = region.getNumOfRow();
        int n_col = region.getNumOfCol();
        Image aug_img(n_row, n_col, 0);

        int i = 0;
        for (int row = region.row_min; row < region.row_max; row ++)
        {
            int j = 0;
            for (int col = region.col_min; col < region.col_max; col ++)
            {                
                double dist = std::sqrt(
                    (xc - col) * (xc - col) + 
                    (yc - row) * (yc - row)
                );
                double value = dist < r_px ? _cam_list.intensity_max[cam_id] : 0;

                // Creating a particle augmented residual image: (res+p)
                aug_img(i, j) = std::max( 
                    0.0,
                    std::min(double(_cam_list.intensity_max[cam_id]), 
                             _imgRes_list[id](row,col) + value)
                );
                j++;
            }
            i++;
        }

        imgAug_list.region_list.push_back(region);
        imgAug_list.img_list.push_back(aug_img);
        corr_map_list.push_back(Image(n_row, n_col, CORR_INIT));
    }

    int n_cam_mismatch = cam_useid_mismatch.size();
    if (n_cam_mismatch > _n_cam_use - 2) {
        return _SCORE_MISMATCH;
    }

    // Update the particle position, imgAug and search range
    double residue = updateBubble(bb3d, cam_useid_mismatch, imgRef_list, imgAug_list, corr_map_list, delta);
    if (residue < _SCORE_MISMATCH + 1) {
        return _SCORE_MISMATCH;
    }

    // update the bubble score
    // sum up the score of all cam 
    // to delete ghost bubble 
    double bb_score = calBubbleScore(
        bb3d,
        imgAug_list, 
        cam_useid_mismatch, 
        score_old
    );
    return bb_score;
}

bool Shake::isCamValidForShaking(int cam_id, PixelRange const& region, Image const& imgRef, double intRef, Image const& imgOrig, Bubble2D const& bb2d)
{
    // check if the camera is valid for shaking
    // 1. check if the region is valid
    int n_row = region.getNumOfRow();
    int n_col = region.getNumOfCol();
    if (n_row <= 0 || n_col <= 0)
    {
        return false;
    }

    // 2. get original sub-img 
    Image imgOrig_sub(n_row, n_col, 0);
    for (int i = 0; i < n_row; i++)
    {
        for (int j = 0; j < n_col; j++)
        {
            imgOrig_sub(i, j) = imgOrig(region.row_min + i, region.col_min + j);
        }
    }

    // 3. find bubble in the original img
    double r_px = bb2d._r_px;
    double rmin, rmax, sense;
    std::vector<Pt2D> center;
    std::vector<double> radius;
    std::vector<double> metrics;
    if (r_px < 5) {
        // double the resolution if the bubble radius is small
        // 1. get original img
        int npix = std::min(n_row, n_col);
        Image img_ref(npix, npix, 0);
        for (int x_id = 0; x_id < npix; x_id++)
        {
            for (int y_id = 0; y_id < npix; y_id++)
            {
                img_ref(y_id, x_id) = imgOrig_sub(y_id, x_id);
            }
        }
        
        // 2. double resolution
        int img_size = 2 * npix;
        Image img(img_size, img_size, 0);
        BubbleResize bb_resizer;
        bb_resizer.ResizeBubble(img, img_ref, npix*2, _cam_list.intensity_max[cam_id]);

        // 3. find circles in the original sub-img 
        CircleIdentifier circle_id(img);
        rmin = 2;
        rmax = std::ceil(r_px) * 2 + 6;
        sense = 0.95;
        metrics = circle_id.BubbleCenterAndSizeByCircle(center, radius, rmin, rmax, sense);

        // 4. restore the original size, position will not be used at latter steps 
        for (size_t i = 0; i < center.size(); i++)
        {
            center[i] *= 0.5;
            radius[i] *= 0.5;
        }

    } else {
        CircleIdentifier circle_id(imgOrig_sub);
        rmin = 2;
        rmax = std::ceil(r_px) + 3;
        sense = 0.95;
        metrics = circle_id.BubbleCenterAndSizeByCircle(center, radius, rmin, rmax, sense);
    }

    // 3. check if the found bubbles is valid for shaking
    int n_bb = radius.size();
    if (n_bb == 0) {
        // check if the original img is fully occupied by a big bubble. if so, and the avg int is in a suitable range, then it is valid

        // 1. calculate avg intensity of original img
        double int_orig = 0;
        for (int i = 0; i < n_row; i++)
        {
            for (int j = 0; j < n_col; j++)
            {
                int_orig += imgOrig_sub(i, j);
            }
        }
        int_orig /= (n_row * n_col);

        // TODO: check if the range is good for 8 bit, 16 bit
        // orig: [1.5, 0.5]
        // if (int_orig > intRef * 1.5 || 
        //     int_orig < intRef * 0.5)
        if (int_orig > intRef * 1.2 || 
            int_orig < intRef * 0.8)
        {
            return false;
        } 
    } else {
        // make sure there is at least one found bubble is valid
        bool is_valid = false;
        for (int ci = 0; ci < n_bb; ci ++) {
            double cr = radius[ci];
            bool judge = std::fabs(cr - r_px) < 0.3 * r_px && std::fabs(cr - r_px) < 2 && metrics[ci] > 0.1;
            if (judge) {
                is_valid = true;
                break;
            }
        }
        if (!is_valid) {
            return false;
        }
    }

    return true;
}

double Shake::updateBubble(Bubble3D& bb3d, std::vector<int> & cam_useid_mismatch, std::vector<Image> const& imgRef_list, ImgAugList& imgAug_list, std::vector<Image>& corr_map_list, double delta)
{
    std::vector<double> delta_list(3);
    delta_list[0] = - delta;
    delta_list[1] = 0;
    delta_list[2] = delta;

    std::vector<double> array_list(4, 0);
    std::vector<double> array_list_fit(3, 0);
    std::vector<double> coeff(3, 0);
    std::vector<double> residue_list(4, 0);
    std::vector<double> residue_list_fit(3, 0);

    // record history of residue for each camera
    int n_cam_match = corr_map_list.size();
    std::vector<std::vector<double>> residue_cam_list(n_cam_match);
    std::pair<double, std::vector<double>> residue_pair;

    // shaking on x,y,z direction
    int min_id;
    double residue = 0.0;
    Bubble3D bb3d_temp(bb3d);

    for (int i = 0; i < 3; i ++)
    {
        for (int j = 0; j < 3; j ++)
        {
            array_list[j] = bb3d._pt_center[i] + delta_list[j];
            array_list_fit[j] = array_list[j];

            bb3d_temp._pt_center[i] = array_list[j];
            
            // update bb3d 2d match
            bb3d_temp.projectObject2D(_cam_list.useid_list, _cam_list.cam_list); 

            residue_pair = calBubbleResidue(
                corr_map_list,
                bb3d_temp,
                cam_useid_mismatch,
                imgAug_list,
                imgRef_list
            );
            residue_list[j] = residue_pair.first;
            residue_list_fit[j] = residue_list[j];

            // save residue for each camera
            for (int id = 0; id < n_cam_match; id ++)
            {
                residue_cam_list[id].push_back(residue_pair.second[id]);
            }
        }
        
        // residue = coeff[0] + coeff[1] * x + coeff[2] * x^2
        myMATH::polyfit(coeff, array_list_fit, residue_list_fit, 2);

        if (coeff[2] == 0)
        {
            if (coeff[1] == 0)
            {
                min_id = 1;
            }
            else if (coeff[1] > 0)
            {
                min_id = 0;
            }
            else
            {
                min_id = 2;
            }
        }
        else 
        {
            array_list[3] = - coeff[1] / (2 * coeff[2]);
            if (array_list[3]>array_list[0] && array_list[3]<array_list[2])
            {
                bb3d_temp._pt_center[i] = array_list[3];

                // update bb3d 2d match
                bb3d_temp.projectObject2D(_cam_list.useid_list, _cam_list.cam_list);
                residue_pair = calBubbleResidue(
                    corr_map_list,
                    bb3d_temp,
                    cam_useid_mismatch,
                    imgAug_list,
                    imgRef_list
                );
                residue_list[3] = residue_pair.first;
                
                // save residue for each camera
                for (int id = 0; id < n_cam_match; id ++)
                {
                    residue_cam_list[id].push_back(residue_pair.second[id]);
                }
            }
            else
            {
                residue_list[3] = residue_list[0] + residue_list[1] + residue_list[2] + 1; // set a maximum residue value
            }

            min_id = std::min_element(residue_list.begin(), residue_list.end()) - residue_list.begin();
        }
        
        // update bb3d 2d match
        bb3d_temp._pt_center[i] = array_list[min_id];
        // bb3d._pt_center[i] = array_list[min_id];
        residue = residue_list[min_id];
    }

    // check the history of residue for each cam 
    // if min(residue) is still larger than threshold, then ignore this cam and redo update
    int n_mismatch = cam_useid_mismatch.size();
    std::vector<int> cam_useid_mismatch_add;
    for (int id_use = 0; id_use < _n_cam_use; id_use ++)
    {
        if (myMATH::ismember(id_use, cam_useid_mismatch)) {
            continue; 
        }

        double residue_min = *std::min_element(residue_cam_list[id_use].begin(), residue_cam_list[id_use].end());
        if (residue_min > RESIDUE_THRES)
        {
            cam_useid_mismatch_add.push_back(_cam_list.useid_list[id_use]);         
        }
    }
    int n_add_mismatch = cam_useid_mismatch_add.size();

    if (n_add_mismatch > 0)
    {
        if (n_add_mismatch + n_mismatch > _n_cam_use - 2) 
        {
            return _SCORE_MISMATCH;
        } 
        else 
        {
            // update cam_useid_mismatch, corr_map_list
            ImgAugList imgAug_list_new;
            std::vector<Image> corr_map_list_new;
            int id = 0;
            for (int id_use = 0; id_use < _n_cam_use; id_use ++)
            {
                if (myMATH::ismember(id_use, cam_useid_mismatch)) {
                    continue; 
                }
                
                if (!myMATH::ismember(_cam_list.useid_list[id_use], cam_useid_mismatch_add)) {
                    imgAug_list_new.region_list.push_back(imgAug_list.region_list[id]);
                    imgAug_list_new.img_list.push_back(imgAug_list.img_list[id]);
                    corr_map_list_new.push_back(corr_map_list[id]);
                }

                id ++;
            }
            imgAug_list = imgAug_list_new;
            corr_map_list = corr_map_list_new;
            cam_useid_mismatch.insert(
                cam_useid_mismatch.end(), 
                cam_useid_mismatch_add.begin(), 
                cam_useid_mismatch_add.end()
            );

            // redo updateBubble with updated cam_useid_mismatch
            return updateBubble(bb3d, cam_useid_mismatch, imgRef_list, imgAug_list, corr_map_list, delta);
        }
    }

    // update the bubble 3d position
    bb3d._pt_center = bb3d_temp._pt_center;
    bb3d.projectObject2D(_cam_list.useid_list, _cam_list.cam_list);

    return residue;
}

// residue = 1 - corr: [0, 2], smaller is better
std::pair<double, std::vector<double>> Shake::calBubbleResidue(std::vector<Image>& corr_map_list, Bubble3D const& bb3d, std::vector<int> const& cam_useid_mismatch, ImgAugList const& imgAug_list, std::vector<Image> const& imgRef_list)
{
    int n_cam_match = _n_cam_use - cam_useid_mismatch.size();
    std::vector<double> residue_list(n_cam_match, 0);
    double residue = 0;

    int id = 0;
    for (int id_use = 0; id_use < _n_cam_use; id_use ++) {
        if (myMATH::ismember(id_use, cam_useid_mismatch)) {
            continue; 
        }

        int cam_id = _cam_list.useid_list[id_use];
        double xc = bb3d._bb2d_list[id_use]._pt_center[0];
        double yc = bb3d._bb2d_list[id_use]._pt_center[1];
        double r_px = bb3d._bb2d_list[id_use]._r_px;

        int n_row = imgAug_list.img_list[id].getDimRow();
        int n_col = imgAug_list.img_list[id].getDimCol();
        
        // calculate 1-corr as residue 
        int x_low = std::floor(xc); int x_high = x_low + 1;
        int y_low = std::floor(yc); int y_high = y_low + 1;

        if (x_low >= 0 && x_high < n_col && 
            y_low >= 0 && y_high < n_row) {
            // both x and y are within image boundaries
            AxisLimit grid_limit(
                x_low, x_high, y_low, y_high, 0,0
            );

            std::vector<double> center = {xc, yc};
            std::vector<double> corr_interp(4,0);

            corr_interp[0] = getCorrInterp(
                corr_map_list[id], 
                x_low, y_low, r_px, 
                imgAug_list.img_list[id], 
                imgAug_list.region_list[id], 
                imgRef_list[cam_id], 
                _cam_list.intensity_max[cam_id]
            );
            corr_interp[1] = getCorrInterp(
                corr_map_list[id], 
                x_high, y_low, r_px, 
                imgAug_list.img_list[id], 
                imgAug_list.region_list[id], 
                imgRef_list[cam_id], 
                _cam_list.intensity_max[cam_id]
            );
            corr_interp[2] = getCorrInterp(
                corr_map_list[id], 
                x_high, y_high, r_px, 
                imgAug_list.img_list[id], 
                imgAug_list.region_list[id], 
                imgRef_list[cam_id], 
                _cam_list.intensity_max[cam_id]
            );
            corr_interp[3] = getCorrInterp(
                corr_map_list[id], 
                x_low, y_high, r_px, 
                imgAug_list.img_list[id], 
                imgAug_list.region_list[id], 
                imgRef_list[cam_id], 
                _cam_list.intensity_max[cam_id]
            );

            // bilinear interpolation
            residue_list[id] = 1 - myMATH::bilinearInterp(
                grid_limit, corr_interp, center
            );
            residue += residue_list[id];
        } else {
            residue_list[id] = 1 - imgCrossCorr(
                imgAug_list.img_list[id], 
                imgAug_list.region_list[id], 
                imgRef_list[cam_id], 
                _cam_list.intensity_max[cam_id], 
                xc, yc, r_px
            );
            residue += residue_list[id];
        }

        id ++;
    }
    
    residue /= n_cam_match;
    return std::pair<double, std::vector<double>>(residue, residue_list);
}

double Shake::imgCrossCorr(Image const& imgAug, PixelRange const& region, Image const& imgRef, double intMax, double center_x, double center_y, double r)
{
    // Calculate the cross-correlation between the augmented image and the reference image
    int n_row = imgAug.getDimRow();
    int n_col = imgAug.getDimCol();
    double xc = center_x - region.col_min; // start from region.col_min
    double yc = center_y - region.row_min; // start from region.row_min

    // int r_int = std::floor(r);
    // int x_min = std::max(0, int(std::round(xc) - r_int));
    // int x_max = std::min(n_col, int(std::round(xc) + r_int) + 1);
    // int y_min = std::max(0, int(std::round(yc) - r_int));
    // int y_max = std::min(n_row, int(std::round(yc) + r_int) + 1);

    // // get reference image 
    // int npix = r_int * 2 + 1;
    // int center_ref = r_int;
    // Image img_ref(npix, npix, 0);
    // BubbleResize bb_resizer;
    // bb_resizer.ResizeBubble(img_ref, imgRef, npix, intMax);

    int r_int = std::round(r);
    int x_min = std::max(0, int(std::round(xc - r_int)));
    int x_max = std::min(n_col, int(std::round(xc + r_int)) + 1);
    int y_min = std::max(0, int(std::round(yc - r_int)));
    int y_max = std::min(n_row, int(std::round(yc + r_int)) + 1);

    // get reference image 
    int npix = r_int * 2 + 1;
    int center_ref = r_int;
    Image img_ref(npix, npix, 0);
    BubbleResize bb_resizer;
    bb_resizer.ResizeBubble(img_ref, imgRef, npix, intMax);

    // calculate avg_int 
    double int_aug_avg = 0;
    double int_ref_avg = 0;
    int n_sum = 0;
    for (int x_id = x_min; x_id < x_max; x_id++)
    {
        for (int y_id = y_min; y_id < y_max; y_id++)
        {
            int_aug_avg += imgAug(y_id, x_id);
            int dx = std::round(center_ref + x_id - xc);
            int dy = std::round(center_ref + y_id - yc);
            int_ref_avg += img_ref(dy, dx);
            n_sum ++;
        }
    }
    int_aug_avg /= n_sum;
    int_ref_avg /= n_sum;

    // calculate cross-correlation
    double corr = 0;
    double int_aug_var = 0;
    double int_ref_var = 0;
    for (int x_id = x_min; x_id < x_max; x_id++)
    {
        for (int y_id = y_min; y_id < y_max; y_id++)
        {
            int dx = std::round(center_ref + x_id - xc);
            int dy = std::round(center_ref + y_id - yc);
            corr += (imgAug(y_id, x_id) - int_aug_avg) * (img_ref(dy, dx) - int_ref_avg);
            int_aug_var += (imgAug(y_id, x_id) - int_aug_avg) * (imgAug(y_id, x_id) - int_aug_avg);
            int_ref_var += (img_ref(dy, dx) - int_ref_avg) * (img_ref(dy, dx) - int_ref_avg);
        }
    }
    int_aug_var = std::max(int_aug_var, 0.0);
    int_ref_var = std::max(int_ref_var, 0.0);

    if (int_aug_var > SMALLNUMBER && int_ref_var > SMALLNUMBER)
    {
        corr /= (std::sqrt(int_aug_var) * std::sqrt(int_ref_var));
    } 
    else if (int_aug_var < SMALLNUMBER && int_ref_var < SMALLNUMBER)
    {
        corr = 1.0;
    } 
    else 
    {
        corr = 0.0;
    }
    
    corr = std::clamp(corr, -1.0, 1.0);
    return corr;
}

double Shake::getCorrInterp(Image& corr_map, int x, int y, double r_px, Image const& imgAug, PixelRange const& region, Image const& imgRef, double intMax) {
    double corr = 0.0;
    if (corr_map(y, x) < CORR_INIT + 1) {
        corr = imgCrossCorr(
            imgAug, region, imgRef, intMax, x, y, r_px
        );
        corr_map(y, x) = corr;
    } else {
        corr = corr_map(y, x);
    }
    return corr;
}

double Shake::calBubbleScore(Bubble3D const& bb3d, ImgAugList const& imgAug_list, std::vector<int> const& cam_useid_mismatch, double score)
{
    int n_cam_match = _n_cam_use - cam_useid_mismatch.size();
    if (n_cam_match < 2) {
        return _SCORE_MISMATCH;
    }

    // get range for calculating score (radius = r_px)
    std::vector<PixelRange> score_region(n_cam_match);
    int id = 0;
    for (int id_use = 0; id_use < _n_cam_use; id_use ++) {
        if (myMATH::ismember(id_use, cam_useid_mismatch)) {
            continue; 
        }
        score_region[id] = findRegion(
            id_use, 
            bb3d._bb2d_list[id_use]._pt_center[1], // row
            bb3d._bb2d_list[id_use]._pt_center[0], // col
            bb3d._bb2d_list[id_use]._r_px
        );
        id ++;
    }

    // calculate the score on each cam
    std::vector<double> ratio_int(n_cam_match, 0); // intensity ratio
    id = 0;
    for (int id_use = 0; id_use < _n_cam_use; id_use ++) {
        if (myMATH::ismember(id_use, cam_useid_mismatch)) {
            continue; 
        }

        int cam_id = _cam_list.useid_list[id_use];
        double xc = bb3d._bb2d_list[id_use]._pt_center[0];
        double yc = bb3d._bb2d_list[id_use]._pt_center[1];
        double r_px = bb3d._bb2d_list[id_use]._r_px;
        double r_px2 = r_px * r_px;

        int x_min = score_region[id].col_min;
        int x_max = score_region[id].col_max;
        int y_min = score_region[id].row_min;
        int y_max = score_region[id].row_max;
        int x_min_orig = imgAug_list.region_list[id].col_min;
        int y_min_orig = imgAug_list.region_list[id].row_min;
        int n_row_orig = imgAug_list.region_list[id].getNumOfRow();
        int n_col_orig = imgAug_list.region_list[id].getNumOfCol();

        double num = 0.0, denum = 0.0;
        for (int x_id = x_min; x_id < x_max; x_id++) {
            for (int y_id = y_min; y_id < y_max; y_id++) {
                double dist2 = std::pow(x_id - xc, 2) + std::pow(y_id - yc, 2);
                if (dist2 < r_px2) {
                    int x_id_orig = x_id - x_min_orig;
                    int y_id_orig = y_id - y_min_orig;
                    bool judge = (x_id_orig >= 0 && x_id_orig < n_col_orig && y_id_orig >= 0 && y_id_orig < n_row_orig);
                    if (judge) {
                        num += imgAug_list.img_list[id](y_id_orig, x_id_orig);
                        denum += _cam_list.intensity_max[cam_id];
                    }
                }
            }
        }
        ratio_int[id] = num / denum;
        id ++;
    }

    // calculate new score
    double score_new = 1;
    int n_mul = 0;
    for (int i = 0; i < n_cam_match; i ++) {
        if (ratio_int[i] < 2) {
            score_new *= ratio_int[i];
            n_mul ++;
        }
    }

    if (n_mul == 0) {
        score_new = score; 
        std::cerr << "Shake::calBubbleScore Warning: all cameras have high intensity ratio" << std::endl;
    } else {
        score_new = std::pow(score_new, 1.0 / n_mul) * score;
        if (score_new > 1) {
            std::cerr << "Shake::calBubbleScore Warning: score exceeds 1: " << score_new << std::endl;
            score_new = 1; 
        }
    }

    return score_new;
}

void Shake::findGhost(std::vector<Bubble3D>& bb3d_list)
{
    int n_bb3d = bb3d_list.size();

    // find particles that are close to each other
    checkRepeatedObj(bb3d_list, _tol_3d);

    // remove outliers
    double sum = 0;
    int n_mean = 0;
    double mean = 0;
    for (int i = 0; i < n_bb3d; i ++)
    {
        if (_score_list[i] > 0 && !_is_repeated[i])
        {
            sum += _score_list[i];
            n_mean ++;
        }
    }
    mean = sum / n_mean;

    // std::cout << "Shake score mean: " << mean << std::endl;

    // remove ghost tracers if the score is less than _min_score*mean
    _n_ghost = 0;
    for (int i = 0; i < n_bb3d; i ++)
    {
        if (_score_list[i] < _score_min * mean || _is_repeated[i])
        {
            _is_ghost[i] = 1;
            _n_ghost ++;
        }
        else
        {
            _is_ghost[i] = 0;
        }
    }

    #ifdef DEBUG
    std::cout << "\tShake::findGhost: find " << _n_ghost << " ghost bubbles, including " << _n_repeated << " repeated bubbles." << std::endl;
    #endif
}

void Shake::checkRepeatedObj(std::vector<Bubble3D> const& bb3d_list, double tol_3d)
{
    int n_bb3d = bb3d_list.size();
    _is_repeated.resize(n_bb3d, 0);
    _n_repeated = 0;

    // Remove repeated tracks
    double repeat_thres_2 = tol_3d * tol_3d;
    for (int i = 0; i < n_bb3d-1; i ++) {
        if (_is_repeated[i]) {
            continue;
        }
        
        for (int j = i + 1; j < n_bb3d; j ++) {
            double dist2 = myMATH::dist2(
                bb3d_list[i]._pt_center, 
                bb3d_list[j]._pt_center
            );
            if (dist2 < std::pow(bb3d_list[i]._r3d + bb3d_list[j]._r3d + tol_3d, 2)) {
                // if the distance is smaller than the threshold, then they are repeated
                _is_repeated[j] = 1;
                _n_repeated ++;
            }
        }
    }
}

#endif 