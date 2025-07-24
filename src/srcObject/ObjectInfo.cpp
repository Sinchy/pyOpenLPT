#include "ObjectInfo.h"

void Tracer3D::addTracer2D(Tracer2D const& tracer2d, int cam_id)
{
    _camid_list.push_back(cam_id);
    _tr2d_list.push_back(tracer2d);
    _n_2d ++;
}

void Tracer3D::addTracer2D(std::vector<Tracer2D> const& tracer2d_list, std::vector<int> const& camid_list)
{
    if (tracer2d_list.size() != camid_list.size())
    {
        std::cerr << "Tracer3D::addTracer2D: tracer2d_list.size() != camid_list.size()" << std::endl;
        throw error_size;
    }

    _camid_list.insert(_camid_list.end(), camid_list.begin(), camid_list.end());
    _tr2d_list.insert(_tr2d_list.end(), tracer2d_list.begin(), tracer2d_list.end());
    _n_2d += camid_list.size();
}

void Tracer3D::removeTracer2D(int cam_id)
{
    for (int i = 0; i < _n_2d; i ++)
    {
        if (_camid_list[i] == cam_id)
        {
            _camid_list.erase(_camid_list.begin() + i);
            _tr2d_list.erase(_tr2d_list.begin() + i);
            _n_2d --;
            break;
        }
    }
}

void Tracer3D::removeTracer2D(std::vector<int> const& camid_list)
{
    for (int i = 0; i < camid_list.size(); i ++)
    {
        removeTracer2D(camid_list[i]);
    }
}

void Tracer3D::clearTracer2D()
{
    _camid_list.clear();
    _tr2d_list.clear();
    _n_2d = 0;
}

void Tracer3D::updateTracer2D(Tracer2D const& tracer2d, int cam_id)
{
    for (int i = 0; i < _n_2d; i ++)
    {
        if (_camid_list[i] == cam_id)
        {
            _tr2d_list[i] = tracer2d;
            break;
        }
    }
}

void Tracer3D::updateTracer2D(std::vector<Tracer2D> const& tracer2d_list, std::vector<int> const& camid_list)
{
    if (tracer2d_list.size() != camid_list.size())
    {
        std::cerr << "Tracer3D::updateTracer2D: tracer2d_list.size() != camid_list.size()" << std::endl;
        throw error_size;
    }

    _tr2d_list = tracer2d_list;
    _camid_list = camid_list;
    _n_2d = camid_list.size();
}

void Tracer3D::projectObject2D(std::vector<int> const& camid_list, std::vector<Camera> const& cam_list_all)
{
    _n_2d = camid_list.size();
    _camid_list = camid_list;
    _tr2d_list.resize(_n_2d);

    int cam_id;
    for (int i = 0; i < _n_2d; i ++)
    {
        cam_id = _camid_list[i];
        _tr2d_list[i]._pt_center = cam_list_all[cam_id].project(_pt_center);
        _tr2d_list[i]._r_px = _r2d_px;
    }
}

void Tracer3D::getTracer2D(Tracer2D& tracer2d, int cam_id)
{
    for (int i = 0; i < _camid_list.size(); i ++)
    {
        if (_camid_list[i] == cam_id)
        {
            tracer2d = _tr2d_list[i];
            break;
        }
    }
}

void Tracer3D::saveObject3D(std::ofstream& output, int n_cam_all) const
{
    output << _pt_center[0] << "," << _pt_center[1] << "," << _pt_center[2] << "," << _error << "," << _n_2d;
    
    std::vector<double> pt2d_list(n_cam_all*2, IMGPTINIT);
    for (int i = 0; i < _n_2d; i ++)
    {
        pt2d_list[_camid_list[i]*2] = _tr2d_list[i]._pt_center[0];
        pt2d_list[_camid_list[i]*2+1] = _tr2d_list[i]._pt_center[1];
    }

    for (int i = 0; i < n_cam_all; i ++)
    {
        output << "," << pt2d_list[i*2] << "," << pt2d_list[i*2+1];
    }

    output << "\n";
}



// Bubble3D class
void Bubble3D::addBubble2D(Bubble2D const& bb2d, int cam_id)
{
    _camid_list.push_back(cam_id);
    _bb2d_list.push_back(bb2d);
    _n_2d ++;
}

void Bubble3D::addBubble2D(std::vector<Bubble2D> const& bb2d_list, std::vector<int> const& camid_list)
{
    if (bb2d_list.size() != camid_list.size())
    {
        std::cerr << "Bubble3D::addBubble2D: bb2d_list.size() != camid_list.size()" << std::endl;
        throw error_size;
    }

    _camid_list.insert(_camid_list.end(), camid_list.begin(), camid_list.end());
    _bb2d_list.insert(_bb2d_list.end(), bb2d_list.begin(), bb2d_list.end());
    _n_2d += camid_list.size();
}

void Bubble3D::removeBubble2D(int cam_id)
{
    for (int i = 0; i < _n_2d; i ++)
    {
        if (_camid_list[i] == cam_id)
        {
            _camid_list.erase(_camid_list.begin() + i);
            _bb2d_list.erase(_bb2d_list.begin() + i);
            _n_2d --;
            break;
        }
    }
}

void Bubble3D::removeBubble2D(std::vector<int> const& camid_list)
{
    for (int i = 0; i < camid_list.size(); i ++)
    {
        removeBubble2D(camid_list[i]);
    }
}

void Bubble3D::clearBubble2D()
{
    _camid_list.clear();
    _bb2d_list.clear();
    _n_2d = 0;
}

void Bubble3D::updateBubble2D(Bubble2D const& bb2d, int cam_id)
{
    for (int i = 0; i < _n_2d; i ++)
    {
        if (_camid_list[i] == cam_id)
        {
            _bb2d_list[i] = bb2d;
            break;
        }
    }
}

void Bubble3D::updateBubble2D(std::vector<Bubble2D> const& bb2d_list, std::vector<int> const& camid_list)
{
    if (bb2d_list.size() != camid_list.size())
    {
        std::cerr << "Bubble3D::updateBubble2D: bb2d_list.size() != camid_list.size()" << std::endl;
        throw error_size;
    }

    _bb2d_list = bb2d_list;
    _camid_list = camid_list;
    _n_2d = camid_list.size();
}

void Bubble3D::projectObject2D(std::vector<int> const& camid_list, std::vector<Camera> const& cam_list_all)
{
    _camid_list = camid_list;
    _n_2d = camid_list.size();
    for (int i = 0; i < _n_2d; i ++)
    {
        int cam_id = _camid_list[i];
        _bb2d_list[i]._pt_center = cam_list_all[cam_id].project(_pt_center);
    }
    updateR2D(cam_list_all);
}

void Bubble3D::setRadius2D(std::vector<double> r_px_list)
{
    if (r_px_list.size() != _n_2d)
    {
        std::cerr << "Bubble3D::setRadius2D: r_px_list.size() != _n_2d" << std::endl;
        throw error_size;
    }

    for (int i = 0; i < _n_2d; i ++)
    {
        _bb2d_list[i]._r_px = r_px_list[i];
    }
}

void Bubble3D::getBubble2D(Bubble2D& bb2d, int cam_id)
{
    for (int i = 0; i < _camid_list.size(); i ++)
    {
        if (_camid_list[i] == cam_id)
        {
            bb2d = _bb2d_list[i];
            break;
        }
    }
}

bool Bubble3D::updateR3D(std::vector<Camera> const& cam_list_all, double ratio_thres, double tol3d) 
{
    if (_n_2d < 2) {
        std::cerr << "Bubble3D::updateR3D: not enough 2D bubbles to update 3D radius" << std::endl;
        return false;
    }

    // Calculate the radius from the 2D bubbles
    std::vector<double> r3d_list(_n_2d, 0);
    _r3d = 0;
    Pt3D pt3d;
    for (int i = 0; i < _n_2d; i ++) {
        // transform 3d position to cam coordinate
        int cam_id = _camid_list[i];
        pt3d = cam_list_all[cam_id]._pinhole_param.r_mtx * _pt_center + cam_list_all[cam_id]._pinhole_param.t_vec;
        
        double dist2 = pt3d[0]*pt3d[0] + pt3d[1]*pt3d[1] + pt3d[2]*pt3d[2];
        dist2 = std::max(0.0, dist2);
        double r2 = _bb2d_list[i]._r_px * _bb2d_list[i]._r_px; 
        double f2 = std::pow(cam_list_all[cam_id]._pinhole_param.cam_mtx(0,0), 2);

        r3d_list[i] = std::sqrt(dist2 * r2 / (f2 + r2));
        _r3d += r3d_list[i];
    }
    _r3d /= _n_2d;

    for (int i = 0; i < _n_2d; i ++) {
        double r3d_diff = std::fabs(r3d_list[i] - _r3d);
        if (r3d_diff > ratio_thres * _r3d || 
            r3d_diff > tol3d) {
            return false;
        }
    }

    return true;
}

void Bubble3D::updateR2D(int cam_id, std::vector<Camera> const& cam_list_all) 
{
    for (int i = 0; i < _n_2d; i ++) {
        if (_camid_list[i] == cam_id) {
            Pt3D pt3d = cam_list_all[cam_id]._pinhole_param.r_mtx * _pt_center + cam_list_all[cam_id]._pinhole_param.t_vec;
            double dist2 = pt3d[0]*pt3d[0] + pt3d[1]*pt3d[1] + pt3d[2]*pt3d[2];
            dist2 = std::max(0.0, dist2);
            double f = cam_list_all[cam_id]._pinhole_param.cam_mtx(0,0);
            _bb2d_list[i]._r_px = f * _r3d / std::sqrt(dist2 - _r3d * _r3d);
        }
        break;
    }
}

void Bubble3D::updateR2D(std::vector<Camera> const& cam_list_all) 
{
    Pt3D pt3d;
    for (int i = 0; i < _n_2d; i ++) {
        int cam_id = _camid_list[i];
        pt3d = cam_list_all[cam_id]._pinhole_param.r_mtx * _pt_center + cam_list_all[cam_id]._pinhole_param.t_vec;
        double dist2 = pt3d[0]*pt3d[0] + pt3d[1]*pt3d[1] + pt3d[2]*pt3d[2];
        dist2 = std::max(0.0, dist2);
        double f = cam_list_all[cam_id]._pinhole_param.cam_mtx(0,0);
        _bb2d_list[i]._r_px = f * _r3d / std::sqrt(dist2 - _r3d * _r3d);
    }
}

void Bubble3D::saveObject3D(std::ofstream& output, int n_cam_all) const
{
    output << _pt_center[0] << "," << _pt_center[1] << "," << _pt_center[2] << "," << _error << "," << _r3d << "," << _n_2d;
    
    std::vector<double> pt2d_list(n_cam_all*3, IMGPTINIT);
    for (int i = 0; i < _n_2d; i ++)
    {
        pt2d_list[_camid_list[i]*3] = _bb2d_list[i]._pt_center[0];
        pt2d_list[_camid_list[i]*3+1] = _bb2d_list[i]._pt_center[1];
        pt2d_list[_camid_list[i]*3+2] = _bb2d_list[i]._r_px; // store radius in px
    }

    for (int i = 0; i < n_cam_all; i ++)
    {
        output << "," << pt2d_list[i*3] << "," << pt2d_list[i*3+1] << "," << pt2d_list[i*3+2];
    }

    output << "\n";
}

