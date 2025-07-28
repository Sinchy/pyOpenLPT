#include "BubbleRefImg.h"

bool BubbleRefImg::GetBubbleRefImg(std::vector<Image>& img_out, std::vector<Bubble3D> const& bb3d_list, std::vector<std::vector<Bubble2D>> const& bb2d_list_all, std::vector<Image> const& img_input, double r_thres, int n_bb_thres) {
    bool is_valid = true;
    int n_bb3d = bb3d_list.size();

    std::vector<int> is_select(n_bb3d, 1);
    std::vector<int> id_select;
    for (int i = 0; i < n_bb3d; i++) {
        for (int j = 0; j < _n_cam_used; j++) {
            if (bb3d_list[i]._bb2d_list[j]._r_px <= r_thres) {
                is_select[i] = 0;
                break;
            }
        }
        if (is_select[i]) {
            id_select.push_back(i);
        }
    }
    int n_select = id_select.size();

    if (n_select > n_bb_thres) {
        // obtain the maximum bubble diameter 
        std::vector<double> dia_ref(_n_cam_used, 0);
        for (int i = 0; i < _n_cam_used; i++) {
            double dia_max = 0;
            // #pragma omp parallel for reduction(max:dia_max)
            for (int j = 0; j < n_select; j++) {
                int id = id_select[j];
                dia_max = std::max(dia_max, 2 * bb3d_list[id]._bb2d_list[i]._r_px);
            }
            dia_ref[i] = dia_max;
        }

        // initialize the bubble reference image
        for (int i = 0; i < _n_cam_used; i++) {
            int npix = std::round(dia_ref[i]);
            img_out.push_back(Image(npix, npix, 0.0)); 
        }

        // generate bubble reference images
        for (int i = 0; i < _n_cam_used; i++) {
            int npix = std::round(dia_ref[i]);
            std::vector<Image> bb_img_i(n_select, Image(npix, npix, 0.0));
            std::vector<double> max_intensity(n_select, 0.0);
            std::vector<int> is_satisfy(n_select, 0);      
            
            int cam_id_real = _cam_list.useid_list[i];
            int nrow = img_input[cam_id_real].getDimRow();
            int ncol = img_input[cam_id_real].getDimCol();
            // TODO: not sure what is the best way to normalize the intensity
            double intensity_max = _cam_list.intensity_max[cam_id_real];
            // double intensity_max = 0;
            // for (int row_id = 0; row_id < nrow; row_id++) {
            //     for (int col_id = 0; col_id < ncol; col_id++) {
            //         intensity_max = std::max(intensity_max, img_input[cam_id_real](row_id, col_id));
            //     }
            // }

            #pragma omp parallel for 
            for (int j = 0; j < n_select; j++) {
                int id = id_select[j];
                double r = bb3d_list[id]._bb2d_list[i]._r_px;
                bool is_overlap = false;
                bool is_itself = false;

                for (int k = 0; k < bb2d_list_all[i].size(); k ++) {
                    double dist = myMATH::dist(bb2d_list_all[i][k]._pt_center, bb3d_list[id]._bb2d_list[i]._pt_center);

                    if (dist < 1 && !is_itself) {
                        is_itself = true;
                        continue;
                    } else if (dist < bb2d_list_all[i][k]._r_px + r - bb2d_list_all[i][k]._r_px * 0.1) {
                        is_overlap = true;
                        break;
                    }
                }
                if (is_overlap) continue;

                double xc = bb3d_list[id]._bb2d_list[i]._pt_center[0];
                double yc = bb3d_list[id]._bb2d_list[i]._pt_center[1];

                // int x_min = std::ceil(xc - r);
                // int x_max = std::floor(xc + r) + 1;
                // int y_min = std::ceil(yc - r);
                // int y_max = std::floor(yc + r) + 1;
                int x_min = std::round(xc - r);
                int x_max = std::round(xc + r) + 1;
                int y_min = std::round(yc - r);
                int y_max = std::round(yc + r) + 1;

                // check if out of the image range
                if (x_min < 0 || x_max > ncol || 
                    y_min < 0 || y_max > nrow) {
                    continue;
                }

                int dx = x_max - x_min;
                int dy = y_max - y_min;
                int img_size = std::min(dx, dy);
                Image bb_img_ij(img_size, img_size, 0.0);
                double max_int = 0;
                for (int x_id = 0; x_id < img_size; x_id++) {
                    for (int y_id = 0; y_id < img_size; y_id++) {
                        double dx_real = x_min + x_id - xc;
                        double dy_real = y_min + y_id - yc;
                        bool is_outside = dx_real * dx_real + dy_real * dy_real > (r+1) * (r+1);

                        if (is_outside) {
                            bb_img_ij(y_id, x_id) = 0;
                        } else {
                            double val = img_input[cam_id_real](y_id + y_min, x_id + x_min);
                            bb_img_ij(y_id, x_id) = val;
                            max_int = std::max(max_int, val);
                        }
                    }
                }
                max_intensity[j] = max_int;
                is_satisfy[j] = 1;

                // resize image to given size
                BubbleResize bb_resizer;
                bb_resizer.ResizeBubble(bb_img_i[j], bb_img_ij, npix, intensity_max);

                // // for debug
                // std::string file = "D:\\My Code\\Tracking Code\\OpenLPT 0.3\\OpenLPT\\test\\inputs\\test_BubbleRefImg\\debug\\cam" + std::to_string(cam_id_real) + "_bb_" + std::to_string(j) + ".csv";
                // bb_img_ij.write(file);
            }
            
            double mean_int = 0;
            int n_mean_int = 0;
            for (int j = 0; j < n_select; j ++) {
                if (is_satisfy[j]) {
                    mean_int += max_intensity[j];
                    n_mean_int ++;
                }
            }   
            if (n_mean_int > 0) {
                mean_int /= n_mean_int;
            } else {
                is_valid = false;
                return is_valid;
            }

            // calculate average resize image 
            for (int x_id = 0; x_id < npix; x_id ++) {
                for (int y_id = 0; y_id < npix; y_id ++) {
                    int n_avg = 0;
                    for (int k = 0; k < n_select; k++) {
                        if (max_intensity[k] > mean_int * 0.8) {
                            img_out[i](y_id, x_id) += bb_img_i[k](y_id, x_id);
                            n_avg++; 
                        }
                    }
                    img_out[i](y_id, x_id) /= n_avg; 
                }
            }
        }
    } else {
        is_valid = false;
    }

    return is_valid;
}