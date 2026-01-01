#!/bin/bash
echo "=========================================="
echo "      OpenLPT One-Click Installer"
echo "=========================================="
echo ""
echo "This script will:"
echo "1. Create a Conda environment named 'OpenLPT'"
echo "2. Install all dependencies"
echo "3. Install the OpenLPT package"
echo ""
read -p "Press enter to continue..."

cd "$(dirname "$0")"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo ""
    echo "[!] Conda/Mamba not found."
    echo "[!] Should I automatically download and install Miniforge3 (Minimal Conda + Mamba)?"
    echo "    Target: $HOME/miniforge3"
    echo ""
    read -p "Press ENTER to install Miniforge3 (or Ctrl+C to cancel)..."
    
    echo ""
    echo "[0/3] Installing Miniforge3..."
    ARCH=$(uname -m)
    INSTALLER="Miniforge3-MacOSX-${ARCH}.sh"
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}"
    
    echo "Downloading ${URL}..."
    curl -L -O "$URL"
    
    echo "Installing to $HOME/miniforge3..."
    bash "$INSTALLER" -b -p "$HOME/miniforge3"
    rm "$INSTALLER"
    
    # Initialize for future shells
    "$HOME/miniforge3/bin/conda" init zsh
    "$HOME/miniforge3/bin/conda" init bash
    
    # Activate for current script
    source "$HOME/miniforge3/bin/activate"
    
    echo "Miniforge3 installed successfully!"
fi

echo ""
echo "[1/3] Creating Conda Environment 'OpenLPT'..."
conda create -n OpenLPT python=3.10 -y

echo ""
echo "[2/3] Activating Environment..."
# Try to find conda hook
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate OpenLPT
else
    echo "[Warning] Could not find conda.sh. Assuming environment is correct or manual activation needed."
fi

echo ""
echo "[3/3] Installing Dependencies..."
mamba install -c conda-forge --file requirements.txt -y
if [ $? -ne 0 ]; then
    echo "[Error] Mamba install failed."
    echo "NOTE: This project requires mamba."
    read -p "Press enter to exit"
    exit 1
fi

echo ""
echo "[3.5/4] Configuring Compiler Flags for macOS (OpenMP)..."
# 1. Force a clean build to ensure CMake picks up new flags
rm -rf build/
rm -rf _skbuild/  # just in case

# 2. Set environment variables that CMake respects automatically
export CPPFLAGS="-I$CONDA_PREFIX/include $CPPFLAGS"
export CXXFLAGS="-I$CONDA_PREFIX/include $CXXFLAGS"
export CFLAGS="-I$CONDA_PREFIX/include $CFLAGS"
export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib $LDFLAGS"

echo "Using Include Path: $CONDA_PREFIX/include"

echo ""
echo "[4/4] Installing OpenLPT..."
pip install . --no-build-isolation
if [ $? -ne 0 ]; then
    echo "[Error] Pip install failed."
    read -p "Press enter to exit"
    exit 1
fi

echo ""
echo "=========================================="
echo "      Installation Complete!"
echo "=========================================="
echo "You can now run the GUI using:"
echo "conda activate OpenLPT"
echo "python gui/main.py"
echo ""
read -p "Press enter to close..."
