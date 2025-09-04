cmake_minimum_required(VERSION 3.15...3.27)

# ---- Project ----
project(OpenLPT VERSION 0.1.0 LANGUAGES C CXX)

# Require C++20 project-wide (targets may override if needed)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for IDEs (clangd, VSCode, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type for single-config generators (Makefiles/Ninja)
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Toggle: build Python bindings via pybind11 (OFF = native C++ build)
option(PYOPENLPT "Build Python bindings (pybind11)" OFF)

# OpenMP: detect here; actual linking is per-target in module files
find_package(OpenMP REQUIRED COMPONENTS CXX)
if (OpenMP_CXX_FOUND)
  message(STATUS "OpenMP CXX FOUND")
endif()

# --- Core C++ targets (always needed so the py module can link them) ---
if (EXISTS "${PROJECT_SOURCE_DIR}/cmake/openLPT.cmake")
  include("${PROJECT_SOURCE_DIR}/cmake/openLPT.cmake")
else()
  message(FATAL_ERROR "openLPT.cmake not found under ${PROJECT_SOURCE_DIR}/cmake")
endif()

# -------- Build switch --------
if (PYOPENLPT)
  # (keep your Python + pybind setup)
  find_package(pybind11 REQUIRED)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  include("${PROJECT_SOURCE_DIR}/cmake/bindings.cmake")
else()
  include(GNUInstallDirs)
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/inc/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  if (TARGET openlpt_includes)
    install(TARGETS openlpt_includes EXPORT OpenLPTTargets)
  endif()

  option(OPENLPT_BUILD_TESTS "Build OpenLPT tests" OFF)
  if (OPENLPT_BUILD_TESTS AND EXISTS "${PROJECT_SOURCE_DIR}/cmake/ctest.cmake")
    include("${PROJECT_SOURCE_DIR}/cmake/ctest.cmake")
  endif()

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/OpenLPTConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
  )
  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/OpenLPTConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/OpenLPTConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenLPT
  )
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/OpenLPTConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/OpenLPTConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenLPT
  )
  install(EXPORT OpenLPTTargets
    NAMESPACE OpenLPT::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenLPT
  )
endif()


