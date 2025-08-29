#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <ctime>    // clock_t, clock, CLOCKS_PER_SEC
#include <cstdlib>  // EXIT_SUCCESS/EXIT_FAILURE

#include "STBCommons.h"
#include "ImageIO.h"
#include "Matrix.h"
#include "Camera.h"
#include "ObjectInfo.h"
#include "STB.h"


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: OpenLPT <config_file_path>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string config_path = argv[1];

    // Read basic configuration from file
    BasicSetting basic_settings;
    if (!basic_settings.readConfig(config_path)) {
        std::cerr << "Error: Failed to read basic configuration from file: " << config_path << std::endl;
        return EXIT_FAILURE;
    }

    try {
        // Create STB objects based on object types found in config
        std::vector<STB> stb_objects;

        for (size_t i = 0; i < basic_settings._object_types.size(); ++i) {
            const std::string& type = basic_settings._object_types[i];
            const std::string& obj_config_path = basic_settings._object_config_paths[i];
            stb_objects.emplace_back(STB(basic_settings, type, obj_config_path));
        }

        // load previous tracks if needed
        if (basic_settings._load_track) {
            for (size_t i = 0; i < basic_settings._object_types.size(); ++i) { 
                stb_objects[i].loadTracksAll(basic_settings._load_track_path, basic_settings._load_track_frame);
            }
        }

        // --- Prepare image IO ---
        // 运行前做一致性检查：相机数量 vs 图像路径数量
        if (basic_settings._cam_list.size() != basic_settings._image_file_paths.size()) {
            std::cerr << "Error: #cams (" << basic_settings._cam_list.size()
                      << ") != #image paths (" << basic_settings._image_file_paths.size() << ")\n";
            return EXIT_FAILURE;
        }

        std::vector<ImageIO> imgio_list;
        imgio_list.reserve(basic_settings._image_file_paths.size());
        for (const auto& path : basic_settings._image_file_paths) {
            ImageIO io;
            io.loadImgPath("", path);
            imgio_list.push_back(io);
        }

        // image_list 与 imgio_list 保持相同尺寸，避免越界
        std::vector<Image> image_list(imgio_list.size());

        std::cout << "**************" << std::endl;
        std::cout << "OpenLPT start!" << std::endl;
        std::cout << "**************\n" << std::endl;

        // Process frames
        int frame_start = basic_settings._frame_start;
        int frame_end = basic_settings._frame_end;
        int num_cams = static_cast<int>(imgio_list.size());

        clock_t start = clock();
        for (int frame_id = frame_start; frame_id <= frame_end; ++frame_id) {
            for (int i = 0; i < num_cams; ++i) {
                image_list[i] = imgio_list[i].loadImg(frame_id);
            }

            for (auto& stb : stb_objects) {            
                stb.processFrame(frame_id, image_list);
            }
        }
        clock_t end = clock();
        std::cout << "\nTotal time for STB: " << double(end - start) / CLOCKS_PER_SEC << "s\n" << std::endl;
        std::cout << "***************" << std::endl;
        std::cout << "OpenLPT finish!" << std::endl;
        std::cout << "***************" << std::endl;
    }
    catch (const FatalError& e) {   // error defined in error.hpp
        std::cerr << "Program aborted due to error: " << e.what() << std::endl;
        return EXIT_FAILURE;   
    }
    catch (const std::exception& e) { // error from C++
        std::cerr << "Unhandled std exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "Unknown error occurred!" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
