# setup.py — build the CMake-based extension "openlpt" using pip-installed pybind11
import os, sys, platform, subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# 关键：作为“构建时依赖”，确保 pip 会在构建前装好 pybind11
# 更推荐在 pyproject.toml 里声明（见文末备忘）；这里只是运行时兜底 import。
try:
    import pybind11
    PYBIND11_DIR = pybind11.get_cmake_dir()
except Exception as e:
    print("ERROR: pybind11 is required. Install with: python -m pip install pybind11")
    raise

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())

class CMakeBuild(build_ext):
    def run(self):
        # [Windows] Robust Visual Studio Detection
        # CMake 3.x+ often fails to find "Build Tools" (vs_buildtools.exe) installations automatically.
        # However, if we are running inside a Developer Command Prompt (vcvarsall.bat),
        # we can rely on NMake or just standard CMake detection without forcing the generator.
        if platform.system() == "Windows":
             # Optional: Verify environment if needed, but usually vcvars handles it
             pass

        subprocess.check_call(["cmake", "--version"])
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DPYOPENLPT=ON",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={PYBIND11_DIR}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            "-DOPENLPT_PYBIND11_PROVIDER=pip",
        ]
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]

            # Helper for Visual Studio Generator parallel builds
            # NMake doesn't support /m, but Ninja uses -j.
            # We check the generator env var (set by install_windows.bat) or CMake default.
            generator = os.environ.get("CMAKE_GENERATOR", "")
            
            if "Visual Studio" in generator:
                 build_args += ["--", "/m"]
            elif "NMake" in generator:
                 # NMake is serial, no extra flags suitable here
                 pass 
            elif "Ninja" in generator:
                 build_args += ["--", "-j"]
            else:
                 # Fallback: Assume VS if not specified (default CMake behavior on Windows)
                 # But safer to NOT assume parallel if unknown
                 build_args += ["--", "/m"]

        else:
            cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}", "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"]
            build_args += ["--", "-j"]

        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        print(f"[setup.py] CMake Args: {cmake_args}")
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)

from _version import __version__

setup(
    name="openlpt",
    version=__version__,
    description="OpenLPT Python bindings",
    author="Shiyong Tan, Shijie Zhong",
    author_email="szhong12@jhu.edu",
    ext_modules=[CMakeExtension("pyopenlpt", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    install_requires=[
        "numpy>=1.16.0",
        "pandas>=1.0.0",
        "pybind11>=2.10"  # 运行时/编译时都需要
    ],
    packages=[],
    include_package_data=True,
)
