#include "test.h"

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <variant>
#include <omp.h>
#include <time.h>

#include "Matrix.h"
#include "Camera.h"
#include "ObjectInfo.h"
#include "nanoflann.hpp"

struct Tr3dCloud 
{
    std::vector<Tracer3D> const& _tr3d_list;  // 3D points
    Tr3dCloud(std::vector<Tracer3D> const& tr3d_list) : _tr3d_list(tr3d_list) {}

    // Must define the interface required by nanoflann
    inline size_t kdtree_get_point_count() const { return _tr3d_list.size(); }
    inline float kdtree_get_pt(const size_t idx, int dim) const { return _tr3d_list[idx]._pt_center[dim]; }

    // Bounding box (not needed for standard KD-tree queries)
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

bool test_function_1 ()
{
    // load 3d tracer
    std::vector<std::vector<Tracer3D>> tr3d_list_all;
    Tracer3D tr3d;
    for (int i = 0; i < 2; i ++)
    {
        std::vector<Tracer3D> tr3d_list;
        
        std::ifstream file("../test/inputs/test_nanoflann/pt3d_" + std::to_string(i+1) + ".csv");
        std::string line;
        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string token;
            std::vector<std::string> tokens;
            while (std::getline(ss, token, ','))
            {
                tokens.push_back(token);
            }
            if (tokens.size() != 3)
            {
                std::cerr << "Error: Invalid line format in file: " << line << std::endl;
                continue;
            }
            tr3d._pt_center[0] = std::stod(tokens[0]);
            tr3d._pt_center[1] = std::stod(tokens[1]);
            tr3d._pt_center[2] = std::stod(tokens[2]);
            tr3d_list.push_back(tr3d);
        }
        file.close();

        tr3d_list_all.push_back(tr3d_list);
    }

    // use nanoflann to find cloest neighbor
    // create a kd-tree
    Tr3dCloud cloud(tr3d_list_all[0]);
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, Tr3dCloud>,
        Tr3dCloud,
        3 // dimensionality
    >;
    KDTree index(3, cloud, {10 /* max leaf */});
    index.buildIndex();

    // query the kd-tree for the closest point to a given point
    int npts = tr3d_list_all[1].size();
    std::vector<size_t> idx_list(npts);
    std::vector<double> dist_list(npts);
    clock_t t_start, t_end;
    t_start = clock();
    #pragma omp parallel for
    for (int i = 0; i < npts; i++)
    {
        size_t ret_index = 0;
        double out_dist_sqr = 0;
        nanoflann::KNNResultSet<double> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);
        index.findNeighbors(resultSet, tr3d_list_all[1][i]._pt_center.data(), nanoflann::SearchParameters());
        idx_list[i] = ret_index;
        dist_list[i] = std::sqrt(out_dist_sqr);
    }
    t_end = clock();
    std::cout << "Time taken to find nearest neighbors: " << static_cast<double>(t_end - t_start) / CLOCKS_PER_SEC << " seconds" << std::endl;

    // save the result to a file
    std::ofstream file_out("../test/results/test_nanoflann/idx.csv");
    for (int i = 0; i < npts; i++)
    {
        file_out << idx_list[i] << std::endl;
    }
    file_out.close();
    file_out.open("../test/results/test_nanoflann/dist.csv");
    for (int i = 0; i < npts; i++)
    {
        file_out << dist_list[i] << std::endl;
    }
    file_out.close();


    // load expected result
    std::vector<size_t> idx_list_sol(npts);
    std::ifstream file("../test/inputs/test_nanoflann/idx.csv");
    std::string line;
    size_t i = 0;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        ss >> idx_list_sol[i];
        i ++;
    }
    file.close();

    std::vector<double> dist_list_sol(npts);
    file.open("../test/inputs/test_nanoflann/dist.csv");
    i = 0;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        ss >> dist_list_sol[i];
        i ++;
    }
    file.close();

    // compare the result with the expected result
    bool is_equal = true;
    for (int i = 0; i < npts; i++)
    {
        if (idx_list[i] != idx_list_sol[i])
        {
            std::cout << "Error: idx_list[" << i << "] = " << idx_list[i] << ", expected " << idx_list_sol[i] << std::endl;
            is_equal = false;
            break;
        }
        if (std::abs(dist_list[i] - dist_list_sol[i]) > 1e-6)
        {
            std::cout << "Error: dist_list[" << i << "] = " << dist_list[i] << ", expected " << dist_list_sol[i] << std::endl;
            is_equal = false;
            break;
        }
    }
    
    return is_equal;
}

int main()
{
    fs::create_directories("../test/results/test_nanoflann/");

    test_function_1();

    return 0;
}