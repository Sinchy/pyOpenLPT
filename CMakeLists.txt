cmake_minimum_required(VERSION 3.15...3.28)

# --------------------------
# Project
# --------------------------
project(OpenLPT VERSION 0.1.0 LANGUAGES C CXX)

# Global C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type for single-config generators
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Toggle: build Python bindings via pybind11
option(PYOPENLPT "Build Python bindings (pybind11)" OFF)

# Optional: OpenMP detection (link per-target inside module cmake files)
find_package(OpenMP QUIET COMPONENTS CXX)
if (OpenMP_CXX_FOUND)
  message(STATUS "OpenMP CXX found")
endif()

# --------------------------
# Core C++ libraries
# --------------------------
# Expect your existing library definitions here
if (EXISTS "${PROJECT_SOURCE_DIR}/cmake/openLPT.cmake")
  include("${PROJECT_SOURCE_DIR}/cmake/openLPT.cmake")
else()
  message(FATAL_ERROR "Missing cmake/openLPT.cmake")
endif()

# --------------------------
# Python bindings (pybind11)
# --------------------------
if (PYOPENLPT)
  # Python interpreter & development (headers, libs for embedding)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

  # 用字符串缓存而非布尔 option，避免被强制成 ON/OFF
  set(OPENLPT_PYBIND11_PROVIDER "auto" CACHE STRING "pybind11 provider: auto|pip|extern|system")
  set_property(CACHE OPENLPT_PYBIND11_PROVIDER PROPERTY STRINGS auto pip extern system)


  if (OPENLPT_PYBIND11_PROVIDER STREQUAL "extern")
    add_subdirectory(extern/pybind11)
  elseif (OPENLPT_PYBIND11_PROVIDER STREQUAL "pip" OR OPENLPT_PYBIND11_PROVIDER STREQUAL "auto")
    find_package(pybind11 CONFIG QUIET)
    if (NOT pybind11_FOUND)
      execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                      OUTPUT_VARIABLE _pybind11_cmakedir OUTPUT_STRIP_TRAILING_WHITESPACE)
      if (_pybind11_cmakedir)
        set(pybind11_DIR "${_pybind11_cmakedir}" CACHE PATH "pip pybind11 cmakedir")
        find_package(pybind11 CONFIG REQUIRED)
      elseif(OPENLPT_PYBIND11_PROVIDER STREQUAL "auto" AND EXISTS "${PROJECT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt")
        add_subdirectory(extern/pybind11)
      else()
        message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
      endif()
    endif()
  else()
    find_package(pybind11 REQUIRED CONFIG)  # system
  endif()


  # Build the Python module (defined in cmake/bindings.cmake)
  if (EXISTS "${PROJECT_SOURCE_DIR}/cmake/bindings.cmake")
    include("${PROJECT_SOURCE_DIR}/cmake/bindings.cmake")
  else()
    message(FATAL_ERROR "Missing cmake/bindings.cmake")
  endif()

# --------------------------
# Native C++ install/export
# --------------------------
else()
  include(GNUInstallDirs)

  # Install public headers
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/inc/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  # Ensure interface include target is part of export set if other libs link to it
  if (TARGET openlpt_includes)
    install(TARGETS openlpt_includes EXPORT OpenLPTTargets)
  endif()

  # (Optional) tests
  option(OPENLPT_BUILD_TESTS "Build OpenLPT tests" OFF)
  if (OPENLPT_BUILD_TESTS AND EXISTS "${PROJECT_SOURCE_DIR}/cmake/ctest.cmake")
    include("${PROJECT_SOURCE_DIR}/cmake/ctest.cmake")
  endif()

  # Package config export
  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/OpenLPTConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
  )

  # If you have a template config, keep it; otherwise a minimal one can be added later.
  if (EXISTS "${PROJECT_SOURCE_DIR}/cmake/OpenLPTConfig.cmake.in")
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
  else()
    # If you don't maintain a config template, at least install the version file
    install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/OpenLPTConfigVersion.cmake"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenLPT
    )
  endif()

  # Export targets created in openLPT.cmake
  install(EXPORT OpenLPTTargets
    NAMESPACE OpenLPT::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenLPT
  )
endif()
