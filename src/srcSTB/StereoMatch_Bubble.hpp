#ifndef STEREOMATCH_BUBBLE_HPP
#define STEREOMATCH_BUBBLE_HPP

#include "StereoMatch.h"
#define R3D_RATIO_THRES 0.05

//              //
// Bubble match //
//              //

// get list of matched bubble_id list
void StereoMatch::bubbleMatch (std::vector<std::vector<Bubble2D>> const& obj2d_list)
{
    if (_n_cam_use < 2)
    {
        std::cerr << "StereoMatch::bubbleMatch error at line " << __LINE__ << ": \n"
                  << "Need at least 2 cameras for matching" 
                  << std::endl;
        throw error_size;
    }
    if (int(obj2d_list.size()) != _n_cam_use)
    {
        std::cerr << "StereoMatch::bubbleMatch error at line " << __LINE__ << ":\n"
                  << "The size of bubble list is: " 
                  << obj2d_list.size() 
                  << ", is different from the size of camera list: "
                  << _n_cam_use << "."
                  << std::endl;
        throw error_size;
    }

    if (_param.n_thread != 0)
    {
        omp_set_num_threads(_param.n_thread);
    }
    #pragma omp parallel
    {
        // for 1st used camera, draw a line of sight through each particle on its image plane.
        // project these lines of sight onto the image planes of 2nd camera.
        // particles within mindist_2D of these lines are candidate matches between first 2 cams.
        // then project 2 line of sights from each particle pair of 1st & 2nd cam onto 3rd cam.
        // particles within torlerance are candidate matches from 3rd cam.
        // repeat similarly for subsequent cams
        #pragma omp for 
        for (int obj_id = 0; obj_id < obj2d_list[0].size(); obj_id ++)
        {
            std::deque<std::vector<int>> objID_match_list; // match list for the particle i in the first camera

            std::vector<int> objID_match; 
            objID_match.push_back(obj_id);

            std::deque<double> error_list;

            findBubbleMatch (
                1,
                objID_match,
                objID_match_list,
                error_list,
                obj2d_list
            );

            #pragma omp critical
            {
                if (error_list.size() != objID_match_list.size())
                {
                    std::cerr << "StereoMatch::bubbleMatch error at line " << __LINE__ << ": \n"
                              << "error_list.size = " << error_list.size()
                              << ", objID_match_list.size = " << objID_match_list.size() << "\n";
                    throw error_size;
                }
                for (int i = 0; i < objID_match_list.size(); i ++)
                {
                    if (objID_match_list[i].size() != _n_cam_use)
                    {
                        continue;
                    }

                    int n_error_size_pre = _error_list.size();
                    int n_match_size_pre = _objID_match_list.size();

                    _error_list.push_back(error_list[i]);
                    _objID_match_list.push_back(objID_match_list[i]);  
                    
                    int n_error_size = _error_list.size();
                    int n_match_size = _objID_match_list.size();
                    if (n_error_size == n_error_size_pre || n_match_size == n_match_size_pre)
                    {
                        std::cerr << "StereoMatch::bubbleMatch error at line " << __LINE__ << ":\n"
                                  << "_error_list.size != _object_id_match_list.size; "
                                  << "_error_list.size = " << n_error_size
                                  << "," << n_error_size_pre << ", _object_id_match_list.size = " << n_match_size << "," << n_match_size_pre
                                  << "\n";
                        throw error_size;
                    }
                }
            }
        }
    }
    
    if (_error_list.size() != _objID_match_list.size())
    {
        std::cerr << "StereoMatch::bubbleMatch error at line " << __LINE__ << ":\n"
                  << "_error_list.size != _objID_match_list.size; "
                  << "_error_list.size = " << _error_list.size()
                  << ", _object_id_match_list.size = " << _objID_match_list.size()
                  << "\n";
        throw error_size;
    }

    _n_before_del = _objID_match_list.size();
    #ifdef DEBUG
    std::cout << "\tFinish stereomatching: " 
              << "n_before_del = " << _n_before_del << "."
              << std::endl;
    #endif
}

// remove ghost bubble
void StereoMatch::removeGhostBubble (std::vector<Bubble3D>& obj3d_list, std::vector<std::vector<Bubble2D>> const& obj2d_list)
{
    int n_match = _objID_match_list.size();
    if (n_match == 0)
    {
        return;
    }

    std::vector<int> match_score(n_match, 0);

    // update match_score  
    int n_obj2d, obj2d_id, opt_matchID;
    for (int i = 0; i < _n_cam_use; i ++)
    {
        n_obj2d = obj2d_list[i].size();

        // optMatchID_map: 
        //  save the corresponding match id with smallest error for each obj2d 
        std::vector<int> optMatchID_map(n_obj2d, -1);
        
        // update optMatchID_map
        for (int j = 0; j < n_match; j ++)
        {
            obj2d_id = _objID_match_list[j][i];
            opt_matchID = optMatchID_map[obj2d_id];

            if (opt_matchID == -1) 
            {
                optMatchID_map[obj2d_id] = j;
            }
            else 
            {
                if (_error_list[opt_matchID] > _error_list[j])
                {
                    // replace by a match id with smaller error 
                    optMatchID_map[obj2d_id] = j;
                }
            }
        }


        // update the score
        for (int j = 0; j < n_obj2d; j ++)
        {
            if (optMatchID_map[j] != -1)
            {
                opt_matchID = optMatchID_map[j];
                match_score[opt_matchID] += 1;
            }
        }

    }

    // sort match_score and error_list
    //  match_score: small to large 
    //  error_list: large to small
    std::vector<int> matchID_list(n_match);
    for (int i = 0; i < n_match; i ++)
    {
        matchID_list[i] = i;
    }
    std::vector<double>& error_list(_error_list);
    auto comparator = [match_score, error_list](size_t i, size_t j) {
        if (match_score[i] < match_score[j])
        {
            return true;
        }
        else if (match_score[i] == match_score[j])
        {
            return error_list[i] > error_list[j];
        }
        else
        {
            return false;
        }
    };
    std::sort(matchID_list.begin(), matchID_list.end(), comparator);

    // create map for record whether a obj2d is used
    std::vector<std::vector<bool>> is_obj2d_use;
    for (int i = 0; i < _n_cam_use; i ++)
    {
        n_obj2d = obj2d_list[i].size();
        is_obj2d_use.push_back(std::vector<bool>(n_obj2d, false));
    }

    // select the optimal match based on the sortID
    //  from the highest score and smallest error to the lowest score and largest error
    //  from the last one to the first one
    std::vector<bool> is_select(n_match, true);
    int match_id;
    bool is_use;
    for (int i = n_match-1; i > -1; i --)
    {
        match_id = matchID_list[i];

        for (int j = 0; j < _n_cam_use; j ++)
        {
            obj2d_id = _objID_match_list[match_id][j];
            is_use = is_obj2d_use[j][obj2d_id];
            if (is_use)
            {
                is_select[match_id] = false;
                break;
            }
        }

        if (is_select[match_id])
        {
            for (int j = 0; j < _n_cam_use; j ++)
            {
                obj2d_id = _objID_match_list[match_id][j];
                is_obj2d_use[j][obj2d_id] = true;
            }
        }
    }
    
    // get new match list 
    Bubble3D obj3d;
    obj3d._camid_list = _cam_list.useid_list;
    obj3d._n_2d = _n_cam_use;
    obj3d._bb2d_list.resize(_n_cam_use);
    std::vector<Line3D> sight3D_list(_n_cam_use);

    int cam_id;
    std::vector<std::vector<int>> objID_match_list_new;
    std::vector<double> error_list_new;
    for (int i = 0; i < n_match; i ++)
    {
        if (is_select[i])
        {
            for (int id = 0; id < _n_cam_use; id ++)
            {
                obj2d_id = _objID_match_list[i][id];
                cam_id = _cam_list.useid_list[id];

                obj3d._bb2d_list[id]._pt_center = obj2d_list[id][obj2d_id]._pt_center;
                obj3d._bb2d_list[id]._r_px = obj2d_list[id][obj2d_id]._r_px;
                sight3D_list[id] = _cam_list.cam_list[cam_id].lineOfSight(obj3d._bb2d_list[id]._pt_center);
            }
            myMATH::triangulation(obj3d._pt_center, obj3d._error, sight3D_list);

            // update 3d radius
            obj3d.updateR3D(_cam_list.cam_list, R3D_RATIO_THRES, _param.tor_3d);

            obj3d_list.push_back(obj3d);

            if (_param.is_update_inner_var)
            {
                objID_match_list_new.push_back(_objID_match_list[i]);
                error_list_new.push_back(_error_list[i]);
            }
        }
    }
    if (_param.is_update_inner_var)
    {
        _objID_match_list = objID_match_list_new;
        _error_list = error_list_new;
    }

    // print info
    _n_after_del = obj3d_list.size();
    _n_del = _n_before_del - _n_after_del;
    std::cout << "\tFinish deleting gohst match: "
              << "n_del = " << _n_del << ", "
              << "n_after_del = " << _n_after_del << "."
              << std::endl;
}

// fill bubble info
void StereoMatch::fillBubbleInfo (std::vector<Bubble3D>& obj3d_list, std::vector<std::vector<Bubble2D>> const& obj2d_list)
{
    Bubble3D obj3d;
    obj3d._camid_list = _cam_list.useid_list;
    obj3d._n_2d = _n_cam_use;
    obj3d._bb2d_list.resize(_n_cam_use);

    std::vector<Line3D> sight3D_list(_n_cam_use);

    int obj2d_id;
    int cam_id;
    for (int i = 0; i < _objID_match_list.size(); i ++)
    {
        for (int id = 0; id < _n_cam_use; id ++)
        {
            obj2d_id = _objID_match_list[i][id];
            cam_id = _cam_list.useid_list[id];

            obj3d._bb2d_list[id]._pt_center = obj2d_list[id][obj2d_id]._pt_center;
            obj3d._bb2d_list[id]._r_px = obj2d_list[id][obj2d_id]._r_px;
            sight3D_list[id] = _cam_list.cam_list[cam_id].lineOfSight(obj3d._bb2d_list[id]._pt_center);
        }

        myMATH::triangulation(obj3d._pt_center, obj3d._error, sight3D_list);

        // update 3d radius
        obj3d.updateR3D(_cam_list.cam_list, R3D_RATIO_THRES, _param.tor_3d);

        obj3d_list.push_back(obj3d);
    }
}

// save bubble info
void StereoMatch::saveBubbleInfo (std::string path, std::vector<Bubble3D> const& obj3d_list)
{
    std::ofstream file;
    file.open(path, std::ios::out);

    file << "WorldX,WorldY,WorldZ,Error,R3D,Ncam";

    int n_cam_all = _cam_list.cam_list.size();
    for (int i = 0; i < n_cam_all; i ++)
    {
        file << "," << "cam" << i << "_" << "x(col)" 
             << "," << "cam" << i << "_" << "y(row)"
             << "," << "cam" << i << "_" << "r(px)";
    }
    file << "\n";

    file.precision(SAVEPRECISION);

    for (int i = 0; i < obj3d_list.size(); i ++)
    {
        obj3d_list[i].saveObject3D(file, n_cam_all);
    }

    file.close();

}

// auxiliary functions for bubbleMatch //

// recursively find matches for bubble
void StereoMatch::findBubbleMatch (
    int id, // id = 1 => 2nd used cam 
    std::vector<int> const& objID_match,
    std::deque<std::vector<int>>& objID_match_list,
    std::deque<double>& error_list,
    std::vector<std::vector<Bubble2D>> const& obj2d_list
)
{
    if (id < 1)
    {
        std::cerr << "StereoMatch::findBubbleMatch error at line " << __LINE__ << ":\n"
                  << "id = " << id << " < 1"
                  << std::endl;
        throw error_size;
    }

    int camID_curr = _cam_list.useid_list[id];
    int n_row = _cam_list.cam_list[camID_curr].getNRow();
    int n_col = _cam_list.cam_list[camID_curr].getNCol();

    // Line of sight
    std::vector<Line2D> sight2D_list;
    std::vector<Line3D> sight3D_list;
    Line2D sight2D;
    Line3D sight3D;
    Pt2D pt2d_1;
    Pt2D pt2d_2;
    bool is_parallel;
    Pt2D unit2d;
    for (int i = 0; i < id; i ++)
    {       
        // project from cam_prev onto cam_curr
        int camID_prev = _cam_list.useid_list[i];

        // get 3d light of sight from cam_prev
        sight3D = _cam_list.cam_list[camID_prev].lineOfSight(obj2d_list[i][objID_match[i]]._pt_center);
        sight3D_list.push_back(sight3D);

        // project 3d light of sight onto cam_curr (3d line => 2d line)
        pt2d_1 = _cam_list.cam_list[camID_curr].project(sight3D.pt);
        pt2d_2 = _cam_list.cam_list[camID_curr].project(sight3D.pt + sight3D.unit_vector);
        unit2d = myMATH::createUnitVector(pt2d_1, pt2d_2);
        sight2D.pt = pt2d_1;
        sight2D.unit_vector = unit2d;
        sight2D_list.push_back(sight2D);
    }
    sight3D_list.push_back(sight3D);


    //                       //
    // id = 1 (2nd used cam) //
    //                       //
    Pt3D pt3d;
    if (id == 1)  
    {
        // find obj candidates around the line of sight         
        // ObjectMarkerAroundLine   

        // calculate the parallel border lines: represented by the ref points
        Line2D sight2D_plus;
        Line2D sight2D_minus;
        // calculate the perpendicular unit vecotr
        unit2d[0] = sight2D.unit_vector[1];
        unit2d[1] = -sight2D.unit_vector[0];
        // calculate the ref points
        pt2d_1 = sight2D.pt + unit2d * _param.tor_2d;
        pt2d_2 = sight2D.pt - unit2d * _param.tor_2d;
        sight2D_plus.pt = pt2d_1;
        sight2D_plus.unit_vector = sight2D.unit_vector;
        sight2D_minus.pt = pt2d_2;
        sight2D_minus.unit_vector = sight2D.unit_vector;

        // Search the candidate within the torlerance near a line
        // Determine the border of the torlerance by iterating each x pixel or y pixel depending on the slope.
        // x_pixel (col id), y_pixel (row id)
        //  if the |slope| > 1, iterate every y_pixel (row)
        //  else, iterate every x_pixel (col) 
        Line2D sight2D_axis;
        if (std::fabs(sight2D.unit_vector[1]) > std::fabs(sight2D.unit_vector[0]))
        {
            sight2D_axis.unit_vector = Pt2D(1,0);
            int x_pixel_1, x_pixel_2, min, max;   

            int y_start = 0;
            int y_end = n_row;
            for (int y_pixel = y_start; y_pixel < y_end; y_pixel ++)
            {
                sight2D_axis.pt[0] = 0;
                sight2D_axis.pt[1] = y_pixel;

                // Get x_pixel (col id) from two lines 
                // quit if the two lines are parallel
                is_parallel = myMATH::crossPoint(pt2d_1, sight2D_axis, sight2D_plus);
                if (is_parallel)
                {
                    return;
                }
                is_parallel = myMATH::crossPoint(pt2d_2, sight2D_axis, sight2D_minus);
                if (is_parallel)
                {
                    return;
                }

                x_pixel_1 = pt2d_1[0];
                x_pixel_2 = pt2d_2[0];
                if (x_pixel_1 > x_pixel_2)
                {
                    max = x_pixel_1 + 1; 
                    min = x_pixel_2;
                }
                else 
                {
                    max = x_pixel_2 + 1; 
                    min = x_pixel_1;
                }
                min = std::max(0, min);
                min = std::min(n_col-1, min);
                max = std::max(1, max);
                max = std::min(n_col, max);

                for (int col = min; col < max; col ++)
                {
                    iterOnObjIDMap (
                        id, 
                        y_pixel, col,
                        sight2D_list,
                        sight3D_list,
                        objID_match, 
                        objID_match_list,
                        error_list,
                        obj2d_list
                    );
                }
            }
        }
        else 
        {
            sight2D_axis.unit_vector = Pt2D(0,1);
            int y_pixel_1, y_pixel_2, min, max;

            int x_start = 0;
            int x_end = n_col;
            for (int x_pixel = x_start; x_pixel < x_end; x_pixel ++)
            {
                sight2D_axis.pt[0] = x_pixel;
                sight2D_axis.pt[1] = 0;

                // Get y_pixel (row id) from two lines 
                // quit if the two lines are parallel
                is_parallel = myMATH::crossPoint(pt2d_1, sight2D_axis, sight2D_plus);
                if (is_parallel)
                {
                    return;
                }
                is_parallel = myMATH::crossPoint(pt2d_2, sight2D_axis, sight2D_minus);
                if (is_parallel)
                {
                    return;
                }

                y_pixel_1 = pt2d_1[1];
                y_pixel_2 = pt2d_2[1];
                if (y_pixel_1 > y_pixel_2)
                {
                    max = y_pixel_1 + 1; 
                    min = y_pixel_2;
                }
                else 
                {
                    max = y_pixel_2 + 1; 
                    min = y_pixel_1;
                }
                min = std::max(0, min);
                min = std::min(n_row-1, min);
                max = std::max(0, max);
                max = std::min(n_row, max);

                for (int row = min; row < max; row ++)
                {
                    iterOnObjIDMap (
                        id, 
                        row, x_pixel,
                        sight2D_list,
                        sight3D_list,
                        objID_match, 
                        objID_match_list,
                        error_list,
                        obj2d_list
                    );
                }
            }
        }

    }
    //                                  //
    // cam_cur_id = 2 ~ _check_cam_id-1 //
    //                                  //
    else
    {
        // find search region
        std::pair<PixelRange, bool> search_output = findSearchRegion(id, sight2D_list);
        if (!search_output.second)
        {
            return;
        }

        // iterate every pixel in the search region
        PixelRange search_region = search_output.first;
        for (int i = search_region.row_min; i < search_region.row_max; i ++)
        {
            for (int j = search_region.col_min; j < search_region.col_max; j ++)
            {
                // judge whether the distances between the candidate
                // and all the lines are all within the range 
                iterOnObjIDMap (
                    id, 
                    i, j,
                    sight2D_list,
                    sight3D_list,
                    objID_match, 
                    objID_match_list,
                    error_list,
                    obj2d_list
                );
            }
        }
    }
}


void StereoMatch::iterOnObjIDMap (
    int id, 
    int row_id, int col_id,
    std::vector<Line2D> const& sight2D_list,
    std::vector<Line3D>& sight3D_list,
    std::vector<int> const& objID_match, 
    std::deque<std::vector<int>>& objID_match_list,
    std::deque<double>& error_list,
    std::vector<std::vector<Bubble2D>> const& obj2d_list
)
{
    int camID_curr = _cam_list.useid_list[id];
    Pt3D pt3d;
    double tor_2d_sqr = _param.tor_2d * _param.tor_2d;

    for (
        int k = 0; 
        k < _objID_map_list[id](row_id, col_id).size(); 
        k ++
    )
    {
        int obj_id = _objID_map_list[id](row_id, col_id)[k];

        if (obj_id == -1)
        {
            break;
        }

        bool in_range = true;
        for (int m = 0; m < id; m ++)
        {
            double dist2 = myMATH::dist2(obj2d_list[id][obj_id]._pt_center, sight2D_list[m]);
            if (dist2 > tor_2d_sqr)
            {
                in_range = false;
                break;
            }
        } 

        // reproject onto the previous cam, then is within error line
        if (in_range && checkReProject(id, obj_id, objID_match, obj2d_list))
        {
            std::vector<int> objID_match_new = objID_match;
            objID_match_new.push_back(obj_id);

            // if there is still other cameras to check
            if (id < _param.check_id - 1) 
            {
                int next_id = id + 1;

                // move onto the next camera and search candidates
                findBubbleMatch (
                    next_id,
                    objID_match_new, 
                    objID_match_list,
                    error_list,
                    obj2d_list                  
                );
            }
            // if the current camera is the last one, then finalize the match.
            else
            {
                // test 3d distance by triangulation  
                sight3D_list[id] = _cam_list.cam_list[camID_curr].lineOfSight(obj2d_list[id][obj_id]._pt_center);
                double error_3d = 0.0;
                myMATH::triangulation(pt3d, error_3d, sight3D_list);

                if (error_3d > _param.tor_3d)
                {
                    continue;
                }
                else 
                {
                    if (id < _n_cam_use - 1)
                    {
                        int next_id = id + 1;

                        checkBubbleMatch (
                            next_id,
                            pt3d,
                            objID_match_new,
                            objID_match_list,
                            error_list,
                            obj2d_list
                        );
                    }
                    else 
                    {
                        // check if the radius is within the tolerance
                        Bubble3D obj3d(pt3d);
                        obj3d._camid_list = _cam_list.useid_list;
                        obj3d._n_2d = _n_cam_use;
                        for (int i = 0; i < _n_cam_use; i ++)
                        {
                            obj3d._bb2d_list.push_back(
                                obj2d_list[i][objID_match_new[i]]
                            );
                        }
                        bool is_valid = obj3d.updateR3D(_cam_list.cam_list, R3D_RATIO_THRES, _param.tor_3d);
                        
                        if (is_valid)
                        {
                            objID_match_list.push_back(objID_match_new);
                            error_list.push_back(error_3d);
                        }
                    }
                }
            }
        }

    }  
}

bool StereoMatch::checkReProject (
    int id, 
    int obj_id,
    std::vector<int> const& obj_id_match, 
    std::vector<std::vector<Bubble2D>> const& obj2d_list
)
{
    int camID_curr = _cam_list.useid_list[id];
    Line3D sight3D = _cam_list.cam_list[camID_curr].lineOfSight(obj2d_list[id][obj_id]._pt_center);

    Line2D sight2D;
    Pt2D pt2d_1;
    Pt2D pt2d_2;
    double dist2 = 0;
    double tor_2d_sqr = _param.tor_2d * _param.tor_2d;

    for (int i = 0; i < id; i ++)
    {
        int cam_id = _cam_list.useid_list[i];

        pt2d_1 = _cam_list.cam_list[cam_id].project(sight3D.pt);
        pt2d_2 = _cam_list.cam_list[cam_id].project(sight3D.pt + sight3D.unit_vector);
        sight2D.pt = pt2d_1;
        sight2D.unit_vector = myMATH::createUnitVector(pt2d_1, pt2d_2);

        dist2 = myMATH::dist2(obj2d_list[i][obj_id_match[i]]._pt_center, sight2D);

        if (dist2 > tor_2d_sqr)
        {
            return false;
        }
    }

    return true;
}


void StereoMatch::checkBubbleMatch(
    int id, 
    Pt3D const& pt3d,
    std::vector<int> const& objID_match, 
    std::deque<std::vector<int>>& objID_match_list,
    std::deque<double>& error_list,
    std::vector<std::vector<Bubble2D>> const& obj2d_list
)
{
    if (id < 1)
    {
        std::cerr << "StereoMatch::checkBubbleMatch error at line " << __LINE__ << ":\n"
                  << "id = " << id << " < 1"
                  << std::endl;
        throw error_size;
    }

    int camID_curr = _cam_list.useid_list[id];
    int n_row = _cam_list.cam_list[camID_curr].getNRow();
    int n_col = _cam_list.cam_list[camID_curr].getNCol();

    // Line of sight
    std::vector<Line2D> sight2D_list;
    std::vector<Line3D> sight3D_list;
    Line2D sight2D;
    Line3D sight3D;
    Pt2D pt2d_1;
    Pt2D pt2d_2;
    Pt2D unit2d;
    for (int i = 0; i < id; i ++)
    {       
        // project from cam_prev onto cam_curr
        int camID_prev = _cam_list.useid_list[i];

        // get 3d light of sight from cam_prev
        sight3D = _cam_list.cam_list[camID_prev].lineOfSight(obj2d_list[i][objID_match[i]]._pt_center);
        sight3D_list.push_back(sight3D);

        // project 3d light of sight onto cam_curr (3d line => 2d line)
        pt2d_1 = _cam_list.cam_list[camID_curr].project(sight3D.pt);
        pt2d_2 = _cam_list.cam_list[camID_curr].project(sight3D.pt + sight3D.unit_vector);
        unit2d = myMATH::createUnitVector(pt2d_1, pt2d_2);
        sight2D.pt = pt2d_1;
        sight2D.unit_vector = unit2d;
        sight2D_list.push_back(sight2D);
    }
    sight3D_list.push_back(sight3D);

    // find search region
    //  directly project pt_world onto the image plane of the current camera
    //  then find the search region
    Pt2D pt2d = _cam_list.cam_list[camID_curr].project(pt3d);

    int row_min = pt2d[1] - _param.check_radius;
    int row_max = pt2d[1] + _param.check_radius + 1;
    int col_min = pt2d[0] - _param.check_radius;
    int col_max = pt2d[0] + _param.check_radius + 1;

    row_min = std::max(0, row_min);
    row_max = std::min(n_row, row_max);
    col_min = std::max(0, col_min);
    col_max = std::min(n_col, col_max);

    // if the search region is out of the image plane, then return
    if (row_min >= row_max || col_min >= col_max)
    {
        return;
    }

    for (int i = row_min; i < row_max; i ++)
    {
        for (int j = col_min; j < col_max; j ++)
        {
            iterOnObjIDMap (
                id, 
                i, j,
                sight2D_list,
                sight3D_list,
                objID_match, 
                objID_match_list,
                error_list,
                obj2d_list
            );
        }
    }
}



#endif