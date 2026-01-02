# OpenLPT - Open-source Lagrangian Particle Tracking GUI

[![GitHub Stars](https://img.shields.io/github/stars/JHU-NI-LAB/OpenLPT_GUI?style=social)](https://github.com/JHU-NI-LAB/OpenLPT_GUI)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**OpenLPT** is a powerful, user-friendly open-source software for **3D Lagrangian Particle Tracking (LPT)**, designed for experimental fluid dynamics and flow visualization. Developed by the **Ni Research Lab at Johns Hopkins University (JHU)**, it provides a comprehensive GUI-based workflow for high-precision particle tracking and reconstruction.

---

### 🚀 Key Capabilities
*   **3D Particle Tracking**: Robust Lagrangian tracking (LPT) and Shake-the-Box (STB) methods.
*   **Multi-Camera Calibration**: Easy-to-use tools for wand and plate calibration (intrinsic & extrinsic parameters).
*   **Cross-Platform**: Full support for **Windows**, **macOS**, and **Linux**.
*   **Performance**: High-performance C++ core with Python Python bindings for flexibility and speed.

**Keywords**: *Lagrangian Particle Tracking (LPT), Shake-the-Box (STB), 3D Flow Visualization, PIV, Particle Reconstruction, Multi-camera Calibration, Experimental Fluid Dynamics, JHU Ni Research Lab.*

---

## Quick Start

Look how easy it is to use:

### 1. Graphical User Interface (GUI)
```bash
# Activate environment and launch the interactive GUI
conda activate OpenLPT
python GUI.py
```

### 2. Command Line Interface (CLI)
```bash 
# Run STB tracking directly
${code_path}/build/Release/OpenLPT.exe {your configuration file path}
```

### 3. Python API
```python
import pyopenlpt as lpt

# Redirect std::cout to python console
redirector = lpt.PythonStreamRedirector() 

config_file = 'path/to/config.txt'
lpt.run(config_file)
```

---

## Demo

<video src="gui/demo.mp4" width="100%" controls></video>

---

## Features
- User-friendly interface in python
- Lagrangian particle tracking for multiple objects (point-like particles, spherical particles, etc.)
- Support stereomatching with multiple cameras (at least 2)
- Include multiple test cases for users to test and understand the code
- Better structure for adding new functions


## Installation

### Method 1: One-Click Installation (Recommended)

We provide automated scripts that set up everything for you (including Conda, environment, and dependencies).

1.  **Download the code**:
    ```bash
    git clone https://github.com/JHU-NI-LAB/OpenLPT_GUI.git
    cd OpenLPT_GUI
    ```

2.  **Run the Installer**:

    -   **Windows**: 
        Double-click `install_windows.bat`
        *(Or run in terminal: `install_windows.bat`)*

    -   **macOS**: 
        Run in terminal:
        ```bash
        bash install_mac.command
        ```

3.  **Run the GUI**:
    After installation, simply run:
    ```bash
    python GUI.py
    ```

<details>
<summary><h3>Method 2: Manual Installation (Click to expand)</h3></summary>

If you prefer to set up the environment manually:

1.  **Prerequisites**:
    - [Miniforge](https://github.com/conda-forge/miniforge) or [Anaconda](https://www.anaconda.com/)
    - C++ Compiler (Visual Studio 2022 for Windows, Clang for macOS/Linux)

2.  **Create Environment and Install**:

    ```bash
    # Create environment
    conda create -n OpenLPT python=3.10
    conda activate OpenLPT

    # Install dependencies
    mamba install -c conda-forge --file requirements.txt

    # Build and install the package
    pip install . --no-build-isolation
    ```

#### Troubleshooting

| Problem | Solution |
| :--- | :--- |
| **Windows**: Build fails | Install VS Build Tools and Win11 SDK |
| **macOS**: `omp.h` not found | See **macOS OpenMP Fix** section below |
| **macOS**: Architecture | `python -c "import platform; print(platform.machine())"` |
| **Linux**: Permissions | Use `chmod +x` or `sudo` |
| **All**: Stale cache | Delete `build/` folder and retry |
| **Windows**: Unicode Path | `install_windows.bat` handles this automatically |

#### macOS OpenMP Fix

If you get `fatal error: 'omp.h' file not found`:

```bash
export CC="$CONDA_PREFIX/bin/clang"
export CXX="$CONDA_PREFIX/bin/clang++"
export CPPFLAGS="-I$CONDA_PREFIX/include"
export LDFLAGS="-L$CONDA_PREFIX/lib -lomp"
pip install . --no-build-isolation
```

</details>

---

## Samples and Tests

Please see the sample format of configuration files, camera files and image files in `/test/test_STB` or `/test/test_STB_Bubble`.

To run the sample:
1. Open OpenLPT GUI.
2. Load the project configuration from the sample folders.
3. Click 'Run tracking'.

---

## Citation

If you use **OpenLPT** in your research, please cite our publications:

- Tan, S., Salibindla, A., Masuk, A.U.M. and Ni, R., 2020. **Introducing OpenLPT: new method of removing ghost particles and high-concentration particle shadow tracking**. *Experiments in Fluids*, 61(2), p.47.
- Tan, S., Zhong, S. and Ni, R., 2023. **3D Lagrangian tracking of polydispersed bubbles at high image densities**. *Experiments in Fluids*, 64(4), p.85.

## License

This repository contains a mix of original code and **MATLAB Coder-generated** files under a MathWorks **Academic License**.

### ⚠️ Restricted Paths (MATLAB Coder-generated)
The following paths contain code generated by MATLAB Coder and are **NOT** covered by the general MIT/Open-source license of this repository:

- `/src/srcObject/BubbleCenterAndSizeByCircle`
- `/src/srcObject/BubbleCenterAndSizeByCircle/CircleIdentifier.cpp`
- `/src/srcObject/BubbleResize`
- `/inc/libObject/BubbleCenterAndSizeByCircle`
- `/inc/libObject/CircleIdentifier.h`
- `/inc/libObject/BubbleResize`

### 📜 Terms and Conditions
For the paths listed above:
- **ACADEMIC INTERNAL OPERATIONS ONLY**: Usage is restricted to teaching, academic research, and course requirements. 
- **NO Commercial Use**: Government, commercial, or other organizational use is **NOT permitted**.
- **Header Preservation**: Do not modify or remove the "Academic License" header comments in these files.
- **No Sublicensing**: These files are not sublicensed under this repository's open-source license.
- **Redistribution**: If you redistribute this repository, you must keep the original Academic License headers in the generated files.
- **Modification**: If you need to modify the generated code, you must hold a valid MathWorks MATLAB Coder license.

All **other** files in this repository are original work and are distributed under the **MIT License** (see `LICENSE`).

---

## Contact & Contribution

- **Issues**: Please report bugs or request features via [GitHub Issues](https://github.com/JHU-NI-LAB/OpenLPT_GUI/issues).
- **Contact**: For questions, please contact szhong12@jhu.edu or tanshiyong84@gmail.com.
- **Organization**: Ni Research Lab, Johns Hopkins University.
- **Support**: If you find this tool helpful, please give us a ⭐ on GitHub!
