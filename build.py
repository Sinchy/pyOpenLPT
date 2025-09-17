# build.py — configure & build the Python extension "openlpt" with MSVC + pip pybind11
import os, sys, subprocess
from pathlib import Path

ROOT  = Path(__file__).resolve().parent
BUILD = ROOT / "build-py"
CFG   = os.environ.get("CONFIG", "Release")  # or "Debug"
GEN   = os.environ.get("CMAKE_GENERATOR", "Visual Studio 17 2022")
ARCH  = os.environ.get("CMAKE_ARCH", "x64")

def run(cmd, cwd=None):
    print(">>", " ".join(map(str, cmd)))
    p = subprocess.run(cmd, cwd=cwd, text=True)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

def main():
    # 确保已安装 pybind11（不装就提示）
    try:
        import pybind11
        pybind11_dir = pybind11.get_cmake_dir()
    except Exception as e:
        print("pybind11 not installed. Please run: python -m pip install pybind11")
        raise

    BUILD.mkdir(exist_ok=True)

    cfg_cmd = [
        "cmake", "-S", str(ROOT), "-B", str(BUILD),
        "-G", GEN, "-A", ARCH,
        "-DPYOPENLPT=ON",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-Dpybind11_DIR={pybind11_dir}",   
        "-DOPENLPT_PYBIND11_PROVIDER=pip",
    ]
    run(cfg_cmd)

    # run(["cmake", "--build", str(BUILD), "--config", CFG, "--target", "openlpt", "--", "/m"])
    run(["cmake", "--build", str(BUILD), "--config", CFG, "--target", "pyopenlpt", "OpenLPT", "--", "/m"])

    print(f"\nBuild done. Module at: {BUILD / CFG}\n")

if __name__ == "__main__":
    main()
