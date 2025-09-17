#include "PredField.h"
#include "Config.h" // for ObjectConfig

// Directly set displacement field
PredField::PredField (ObjectConfig& obj_cfg, Matrix<double> const& disp_field) : _obj_cfg(obj_cfg)
{
    setGrid();
    
    if (disp_field.getDimRow() != _nGrid_tot || disp_field.getDimCol() != 3)
    {
        THROW_FATAL_CTX(ErrorCode::InvalidArgument, 
                        "The input displacement field has wrong dimension.",
                        "Expected: (" + std::to_string(_nGrid_tot) + ",3), Got: (" 
                        + std::to_string(disp_field.getDimRow()) + "," + std::to_string(disp_field.getDimCol()) + ").");
    }

    _disp_field = disp_field;
}

void PredField::calPredField(const std::vector<std::unique_ptr<Object3D>>& obj3d_list_prev, 
                       const std::vector<std::unique_ptr<Object3D>>& obj3d_list_curr)
{
    setGrid();

    _disp_field = Matrix<double>(_nGrid_tot,3,0);

    // calculate displacement field
    PFParam& param = _obj_cfg._pf_param;
    // loop for each grid
    const int n_thread = _obj_cfg._n_thread;
    #pragma omp parallel num_threads(n_thread)
    {
        #pragma omp for
        for (int x_id = 0; x_id < param.nx; x_id ++)
        {
            for (int y_id = 0; y_id < param.ny; y_id ++)
            {
                for (int z_id = 0; z_id < param.nz; z_id ++)
                {
                    std::vector<int> xyz_id = {x_id, y_id, z_id};

                    int grid_id = mapGridID(xyz_id[0],xyz_id[1],xyz_id[2]);

                    // find obj that is close to the grid
                    
                    std::vector<int> objID_list_prev, objID_list_curr;
                    findNeighbor(objID_list_prev, xyz_id, obj3d_list_prev);
                    findNeighbor(objID_list_curr, xyz_id, obj3d_list_curr);

                    // if not find obj in one frame, then stop
                    if (objID_list_curr.empty() || objID_list_prev.empty())
                    {
                        _disp_field(grid_id, 0) = 0;
                        _disp_field(grid_id, 1) = 0;
                        _disp_field(grid_id, 2) = 0;
                        continue;
                    }

                    // getting the displacement vectors (if there are particles in the search volume for both the frames)
                    std::vector<std::vector<double>> disp_list; // all dx,dy,dz
                    std::vector<double> disp(3,0); // dx,dy,dz
                    int id_curr, id_prev;
                    
                    for (int j = 0; j < objID_list_curr.size(); j ++) 
                    {
                        id_curr = objID_list_curr[j];
                        for (int k = 0; k < objID_list_prev.size(); k ++) 
                        {  
                            id_prev = objID_list_prev[k];
                            // TODO: add a filter to get objects with similar features

                            disp[0] = obj3d_list_curr[id_curr]->_pt_center[0] 
                                    - obj3d_list_prev[id_prev]->_pt_center[0];

                            disp[1] = obj3d_list_curr[id_curr]->_pt_center[1] 
                                    - obj3d_list_prev[id_prev]->_pt_center[1];

                            disp[2] = obj3d_list_curr[id_curr]->_pt_center[2] 
                                    - obj3d_list_prev[id_prev]->_pt_center[2];

                            disp_list.push_back(disp);
                        }
                    }
                    
                    // update displacement pdf
                    std::vector<double> disp_pdf(_nBin_tot, 0);
                    updateDispPDF(disp_pdf, disp_list);

                    // find the peak location of the displacement pdf
                    std::vector<double> disp_opt = findPDFPeakLoc(disp_pdf);

                    _disp_field(grid_id, 0) = disp_opt[0];
                    _disp_field(grid_id, 1) = disp_opt[1];
                    _disp_field(grid_id, 2) = disp_opt[2];
                }
            }
        }
    }

    if (param.is_smooth)
    {
        smoothDispField(param.sigma_x, param.sigma_y, param.sigma_z);
    }
}



Pt3D PredField::getDisp (Pt3D const& pt3d) const
{
    Pt3D disp;
    const PFParam& param = _obj_cfg._pf_param;
    // find out the limits of interpolation cube 
    double pt_x = std::min(
        std::max(pt3d[0], param.limit.x_min),
        param.limit.x_max
    );
    double pt_y = std::min(
        std::max(pt3d[1], param.limit.y_min),
        param.limit.y_max
    );
    double pt_z = std::min(
        std::max(pt3d[2], param.limit.z_min),
        param.limit.z_max
    );

    int x_id = std::max(
        0, 
        (int) std::floor((pt_x - param.limit.x_min) / _dx)
    );
    int y_id = std::max(
        0, 
        (int) std::floor((pt_y - param.limit.y_min) / _dy)
    );
    int z_id = std::max(
        0, 
        (int) std::floor((pt_z - param.limit.z_min) / _dz)
    );

    x_id = std::min(x_id, param.nx-2);
    y_id = std::min(y_id, param.ny-2);
    z_id = std::min(z_id, param.nz-2);

    AxisLimit grid_limit;

    grid_limit.x_min = _grid_x[x_id];
    grid_limit.x_max = _grid_x[x_id + 1];

    grid_limit.y_min = _grid_y[y_id];
    grid_limit.y_max = _grid_y[y_id + 1];

    grid_limit.z_min = _grid_z[z_id];
    grid_limit.z_max = _grid_z[z_id + 1];

    int i_000 = mapGridID(x_id  , y_id  , z_id  );
    int i_100 = mapGridID(x_id+1, y_id  , z_id  );
    int i_101 = mapGridID(x_id+1, y_id  , z_id+1);
    int i_001 = mapGridID(x_id  , y_id  , z_id+1);
    int i_010 = mapGridID(x_id  , y_id+1, z_id  );
    int i_110 = mapGridID(x_id+1, y_id+1, z_id  );
    int i_111 = mapGridID(x_id+1, y_id+1, z_id+1);
    int i_011 = mapGridID(x_id  , y_id+1, z_id+1);

    std::vector<double> pt_vec = {pt_x, pt_y, pt_z};

    for (int j = 0; j < 3; j ++)
    {
        std::vector<double> field = {
            _disp_field(i_000, j),
            _disp_field(i_100, j),
            _disp_field(i_101, j),
            _disp_field(i_001, j),
            _disp_field(i_010, j),
            _disp_field(i_110, j),
            _disp_field(i_111, j),
            _disp_field(i_011, j)
        };
        disp[j] = myMATH::triLinearInterp(grid_limit, field, pt_vec);
    }

    return disp;
}


void PredField::saveDispField (std::string const& file)
{
    _disp_field.write(file);
}


void PredField::setGrid()
{
    PFParam& param = _obj_cfg._pf_param;
    _nGrid_tot = param.nx * param.ny * param.nz;
    _dx = (param.limit.x_max - param.limit.x_min) / (param.nx-1);
    _dy = (param.limit.y_max - param.limit.y_min) / (param.ny-1);
    _dz = (param.limit.z_max - param.limit.z_min) / (param.nz-1);

    _grid_x = myMATH::linspace(param.limit.x_min, param.limit.x_max, param.nx);
    _grid_y = myMATH::linspace(param.limit.y_min, param.limit.y_max, param.ny);
    _grid_z = myMATH::linspace(param.limit.z_min, param.limit.z_max, param.nz);

    // set parameters for displacement statistics
    _nBin_tot = param.nBin_x * param.nBin_y * param.nBin_z;

    _m_xyz.resize(3);
    _c_xyz.resize(3);

    _m_xyz[0] = (param.nBin_x - 1) / (4 * param.r);
    _c_xyz[0] = (param.nBin_x - 1) / 2;

    _m_xyz[1] = (param.nBin_y - 1) / (4 * param.r);
    _c_xyz[1] = (param.nBin_y - 1) / 2;

    _m_xyz[2] = (param.nBin_z - 1) / (4 * param.r);
    _c_xyz[2] = (param.nBin_z - 1) / 2;
}


void PredField::findNeighbor (std::vector<int>& pt_list_id, std::vector<int> const& xyz_id, const std::vector<std::unique_ptr<Object3D>>& obj3d_list)
{
    double x = _grid_x[xyz_id[0]];
    double y = _grid_y[xyz_id[1]];
    double z = _grid_z[xyz_id[2]];

    double dx, dy, dz, distsqr;
    // double rsqr = _param.r*_param.r;

    PFParam& param = _obj_cfg._pf_param;

    for (int i = 0; i < obj3d_list.size(); i++)
    {
        
        dx = obj3d_list[i]->_pt_center[0] - x;
        dy = obj3d_list[i]->_pt_center[1] - y;
        dz = obj3d_list[i]->_pt_center[2] - z;
    
        if (dx<param.r && dx>-param.r &&
            dy<param.r && dy>-param.r &&
            dz<param.r && dz>-param.r)
        {
            pt_list_id.push_back(i);
        }
    }
}


int PredField::mapGridID (int x_id, int y_id, int z_id) const
{
    PFParam& param = _obj_cfg._pf_param;
    return x_id * param.ny * param.nz + y_id * param.nz + z_id;
}


int PredField::mapBinID (int xBin_id, int yBin_id, int zBin_id) const
{
    PFParam& param = _obj_cfg._pf_param;
    return xBin_id * param.nBin_y * param.nBin_z + yBin_id * param.nBin_z + zBin_id;
}


void PredField::updateDispPDF(std::vector<double>& disp_pdf, std::vector<std::vector<double>> const& disp_list)
{
    // initialize disp_pdf
    std::fill(disp_pdf.begin(), disp_pdf.end(), 0);

    PFParam& param = _obj_cfg._pf_param;

    for (int id = 0; id < disp_list.size(); id ++)
    {
        double xBin = _m_xyz[0] * disp_list[id][0] + _c_xyz[0];
        double yBin = _m_xyz[1] * disp_list[id][1] + _c_xyz[1];
        double zBin = _m_xyz[2] * disp_list[id][2] + _c_xyz[2];

        int xBin_id = std::round(xBin);
        int yBin_id = std::round(yBin);
        int zBin_id = std::round(zBin);

        for (int i = std::max(0, xBin_id - 1); 
             i <= std::min(param.nBin_x - 1, xBin_id + 1); 
             i ++) 
        {
            double xx = i - xBin;
            for (int j = std::max(0, yBin_id - 1); 
                 j <= std::min(param.nBin_y - 1, yBin_id + 1); 
                 j ++) 
            {
                double yy = j - yBin;
                for (int k = std::max(0, zBin_id - 1); 
                     k <= std::min(param.nBin_z - 1, zBin_id + 1); 
                     k ++) 
                {
                    double zz = k - zBin;
                    int bin_id = mapBinID(i,j,k);
                    disp_pdf[bin_id] += std::exp(-(xx * xx + yy * yy + zz * zz));
                }
            }
        }
    }
}


std::vector<double> PredField::findPDFPeakLoc(std::vector<double> const& disp_pdf)
{
    std::vector<double> dist_opt(3,0);

    PFParam& param = _obj_cfg._pf_param;
    std::vector<int> peak_id = findPDFPeakID(disp_pdf);

    // finding the peak using 1D Gaussian in x, y & z direction
    int x1, y1, z1, x2, y2, z2, x3, y3, z3;
    for (int i = 0; i < 3; i ++)
    {
        const int nbin_i = (i==0 ? param.nBin_x : (i==1 ? param.nBin_y : param.nBin_z));
        if (peak_id[i] < 0)
        {
            dist_opt[i] = 0.0;
        }
        else if (peak_id[i] == 0)
        {
            // dist_opt[i] = 0.0; // no need to calculate, it is possible that there is only one bin 
            dist_opt[i] = -2 * param.r;
        }
        else if (peak_id[i] == nbin_i - 1)
        {
            dist_opt[i] = 2 * param.r;
        }
        else
        {
            x1 = peak_id[0] - 1 * (i==0);
            y1 = peak_id[1] - 1 * (i==1);
            z1 = peak_id[2] - 1 * (i==2);

            x2 = peak_id[0];
            y2 = peak_id[1];
            z2 = peak_id[2];

            x3 = peak_id[0] + 1 * (i==0);
            y3 = peak_id[1] + 1 * (i==1);
            z3 = peak_id[2] + 1 * (i==2);

            double loc = fitPeakBinLocGauss(
                peak_id[i]-1, disp_pdf[mapBinID(x1,y1,z1)],
                peak_id[i],   disp_pdf[mapBinID(x2,y2,z2)],
                peak_id[i]+1, disp_pdf[mapBinID(x3,y3,z3)]
            );

            dist_opt[i] = (loc - _c_xyz[i]) / _m_xyz[i];
        }
    }

    return dist_opt;
}


std::vector<int> PredField::findPDFPeakID(std::vector<double> const& disp_pdf)
{
    std::vector<int> peak_id(3,-1);
    PFParam& param = _obj_cfg._pf_param;
    int bind_id = std::distance(disp_pdf.begin(), std::max_element(disp_pdf.begin(), disp_pdf.end()));
    if (disp_pdf[bind_id] > 1e-1)
    {
        peak_id[0] = bind_id / (param.nBin_y * param.nBin_z);
        peak_id[1] = (bind_id - peak_id[0] * param.nBin_y * param.nBin_z) / param.nBin_z;
        peak_id[2] = bind_id % param.nBin_z;
    }
    else
    {
        peak_id[0] = -1;
        peak_id[1] = -1;
        peak_id[2] = -1;
    }
    return peak_id;
}


double PredField::fitPeakBinLocGauss (double y1, double v1, double y2, double v2, double y3, double v3)
{
    double ln_z1, ln_z2, ln_z3;

    ln_z1 = v1 < LOGSMALLNUMBER ? std::log(LOGSMALLNUMBER) : std::log(v1);
    ln_z2 = v2 < LOGSMALLNUMBER ? std::log(LOGSMALLNUMBER) : std::log(v2);
    ln_z3 = v3 < LOGSMALLNUMBER ? std::log(LOGSMALLNUMBER) : std::log(v3);

    double yc = -0.5 * (  (ln_z1 * double((y2 * y2) - (y3 * y3))) 
                        - (ln_z2 * double((y1 * y1) - (y3 * y3))) 
                        + (ln_z3 * double((y1 * y1) - (y2 * y2))) ) 
                     / (  (ln_z1 * double(y3 - y2)) 
                        - (ln_z3 * double(y1 - y2)) 
                        + (ln_z2 * double(y1 - y3)) );

    if (!std::isfinite(yc))
    {
        yc = y2;
    }

    return yc;
}


void PredField::applyGaussian (Matrix<double>& field, int nx, int ny, int nz, std::vector<double> const& kernel, int radius, int axis)
{
    Matrix<double> temp(field); // Copy input data

    // along x-direction
    #pragma omp parallel for collapse(2)
    for (int x = 0; x < nx; x++) 
    {
        for (int y = 0; y < ny; y++) 
        {
            for (int z = 0; z < nz; z++) 
            {
                double sum = 0.0;
                // double weight = 0.0;
                for (int k = -radius; k <= radius; k++) 
                {
                    int idx = x + k;
                    if (idx >= 0 && idx < nx) 
                    {
                        int src = mapGridID(idx, y, z);
                        sum += temp(src,axis) * kernel[k + radius];
                    }
                }
                int dst = mapGridID(x, y, z);
                field(dst,axis) = sum;
            }
        }
    }
    temp = field; // update temp

    // along y-direction
    #pragma omp parallel for collapse(2)
    for (int x = 0; x < nx; x++) 
    {
        for (int y = 0; y < ny; y++) 
        {
            for (int z = 0; z < nz; z++) 
            {
                double sum = 0.0;
                for (int k = -radius; k <= radius; k++) 
                {
                    int idx = y + k;
                    if (idx >= 0 && idx < ny) 
                    {
                        int src = mapGridID(x, idx, z);
                        sum += temp(src,axis) * kernel[k + radius];
                    }
                }
                int dst = mapGridID(x, y, z);
                field(dst,axis) = sum;
            }
        }
    }
    temp = field; // update temp

    // along z-direction
    #pragma omp parallel for collapse(2)
    for (int x = 0; x < nx; x++) 
    {
        for (int y = 0; y < ny; y++) 
        {
            for (int z = 0; z < nz; z++) 
            {
                double sum = 0.0;
                // double weight = 0.0;
                for (int k = -radius; k <= radius; k++) 
                {
                    int idx = z + k;
                    if (idx >= 0 && idx < nz) 
                    {
                        int src = mapGridID(x, y, idx);
                        sum += temp(src,axis) * kernel[k + radius];
                    }
                }
                int dst = mapGridID(x, y, z);
                field(dst,axis) = sum;
            }
        }
    }
}


void PredField::smoothDispField(double sigma_x, double sigma_y, double sigma_z)
{
    PFParam& param = _obj_cfg._pf_param;
    // Typically 3σ for good results
    int radius_x = std::round(3 * sigma_x);
    int radius_y = std::round(3 * sigma_y);
    int radius_z = std::round(3 * sigma_z);  
    std::vector<double> kernel_x = myMATH::createGaussianKernel(radius_x, sigma_x);
    std::vector<double> kernel_y = myMATH::createGaussianKernel(radius_y, sigma_y);
    std::vector<double> kernel_z = myMATH::createGaussianKernel(radius_z, sigma_z);

    // ux
    applyGaussian(_disp_field, param.nx, param.ny, param.nz, kernel_x, radius_x, 0); 
    // uy
    applyGaussian(_disp_field, param.nx, param.ny, param.nz, kernel_y, radius_y, 1); 
    // uz
    applyGaussian(_disp_field, param.nx, param.ny, param.nz, kernel_z, radius_z, 2); 
}
