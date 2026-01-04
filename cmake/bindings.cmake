# ===========================
#  OpenLPT Python Bindings
# ===========================
cmake_minimum_required(VERSION 3.15)

# OpenMP may be needed by linked libs; keep this quiet if not present
find_package(OpenMP QUIET)

# Directory containing binding translation units
set(OPENLPT_PY_DIR "${PROJECT_SOURCE_DIR}/src/pybind_OpenLPT")

# Add or remove binding files here as your module grows
set(PY_SOURCES
  "${PROJECT_SOURCE_DIR}/src/main.cpp"
  "${OPENLPT_PY_DIR}/pyConfig.cpp"
  "${OPENLPT_PY_DIR}/pyOpenLPT.cpp"
  "${OPENLPT_PY_DIR}/pyCamera.cpp"
  "${OPENLPT_PY_DIR}/pyImageIO.cpp"
  "${OPENLPT_PY_DIR}/pyObjectInfo.cpp"
  "${OPENLPT_PY_DIR}/pyObjectFinder.cpp"
  "${OPENLPT_PY_DIR}/pyPredField.cpp"
  "${OPENLPT_PY_DIR}/pyStereoMatch.cpp"
  "${OPENLPT_PY_DIR}/pyShake.cpp"
  "${OPENLPT_PY_DIR}/pyIPR.cpp"
  "${OPENLPT_PY_DIR}/pyOTF.cpp"
  "${OPENLPT_PY_DIR}/pyBubbleRefImg.cpp"
  "${OPENLPT_PY_DIR}/pyBubbleResize.cpp"
  "${OPENLPT_PY_DIR}/pyTrack.cpp"
  "${OPENLPT_PY_DIR}/pyMatrix.cpp"
  "${OPENLPT_PY_DIR}/pymyMath.cpp"
  "${OPENLPT_PY_DIR}/pySTB.cpp"
  "${OPENLPT_PY_DIR}/pySTBCommons.cpp"
  "${OPENLPT_PY_DIR}/pyCircleIdentifier.cpp"
)

# Python module name (import name)
set(OPENLPT_PYMODULE_NAME "pyopenlpt")

# Build the Python extension (.pyd/.so)
pybind11_add_module(${OPENLPT_PYMODULE_NAME} MODULE ${PY_SOURCES})

if (MSVC)
  target_compile_options(${OPENLPT_PYMODULE_NAME} PRIVATE /std:c++latest)
endif()

# Make headers available to the bindings
target_include_directories(${OPENLPT_PYMODULE_NAME} PRIVATE
  "${PROJECT_SOURCE_DIR}/inc"
)

# Link against your existing C++ targets (adjust to your actual libs)
target_link_libraries(${OPENLPT_PYMODULE_NAME} PRIVATE
  STB 
  Config
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

# Compile options
target_compile_features(${OPENLPT_PYMODULE_NAME} PRIVATE cxx_std_20)
target_compile_definitions(${OPENLPT_PYMODULE_NAME} PRIVATE
    NOMINMAX
    OPENLPT_EXPOSE_PRIVATE # Expose private members for debugging 
)


# Match Python's /MD runtime on MSVC
if (MSVC)
  set_property(TARGET ${OPENLPT_PYMODULE_NAME} PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# Optional diagnostics (remove after first success)
get_target_property(_inc ${OPENLPT_PYMODULE_NAME} INCLUDE_DIRECTORIES)
message(STATUS "[PY] ${OPENLPT_PYMODULE_NAME} include dirs = ${_inc}")
