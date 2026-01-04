# setup.py — build the CMake-based extension "openlpt" using pip-installed pybind11
import os, sys, platform, subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# 关键：作为“构建时依赖”，确保 pip 会在构建前装好 pybind11
# 现在我们已经在 pyproject.toml 里显式声明了 pybind11，
# 所以 setup.py 顶层不再需要强制 import，可以等真正 build 时再加载。

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
        try:
            import pybind11
            pybind11_dir = pybind11.get_cmake_dir()
        except ImportError:
            print("ERROR: pybind11 is required at build time.")
            sys.exit(1)

        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DPYOPENLPT=ON",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11_dir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            "-DOPENLPT_PYBIND11_PROVIDER=pip",
        ]
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]

            # Helper for Visual Studio Generator parallel builds
            # If CMAKE_GENERATOR is set (e.g. by install_windows.bat), pass it explicitly via -G
            generator = os.environ.get("CMAKE_GENERATOR", "")
            if generator:
                 print(f"[setup.py] Using generator from env: {generator}")
                 cmake_args += ["-G", generator]
            
            if "Visual Studio" in generator:
                 build_args += ["--", "/m"]
            elif "NMake" in generator:
                 # NMake is serial
                 pass 
            elif "Ninja" in generator:
                 # Ninja auto-detects parallelism, no need for -j
                 pass
            else:
                 # Fallback: Assume VS if not specified (default CMake behavior on Windows)
                 if not generator:
                      build_args += ["--", "/m"]

        else:
            cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}", "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"]
            
            # [macOS] Help CMake find Homebrew's libomp
            if platform.system() == "Darwin":
                try:
                    brew_prefix = subprocess.check_output(["brew", "--prefix"], text=True).strip()
                    libomp_dir = Path(brew_prefix) / "opt" / "libomp"
                    if libomp_dir.exists():
                        cmake_args += [f"-DOpenMP_ROOT={libomp_dir}"]
                except Exception:
                    # Fail silently if brew is not available (though it should be on GitHub Actions)
                    pass

        build_temp = Path(self.build_temp).resolve()
        
        # [NEW] Auto-clean on generator mismatch
        if (build_temp / "CMakeCache.txt").exists():
            try:
                with open(build_temp / "CMakeCache.txt", 'r') as f:
                    cache_content = f.read()
                
                # Check for cached generator (e.g., CMAKE_GENERATOR:INTERNAL=NMake Makefiles)
                import re
                m = re.search(r'CMAKE_GENERATOR:INTERNAL=(.*)', cache_content)
                cached_gen = m.group(1).strip() if m else None
                
                # Compare with current generator
                current_gen = os.environ.get("CMAKE_GENERATOR", "")
                
                if cached_gen and current_gen and cached_gen != current_gen:
                    print(f"[setup.py] Generator mismatch detected ({cached_gen} != {current_gen}). Cleaning {build_temp}...")
                    import shutil
                    shutil.rmtree(build_temp)
            except Exception as e:
                print(f"[setup.py] Warning: Failed to check CMakeCache.txt: {e}")

        build_temp.mkdir(parents=True, exist_ok=True)

        print(f"[setup.py] CMake Args: {cmake_args}")
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)
# 获取版本号 (避开 import 以兼容构建环境)
version_info = {}
with open("_version.py", encoding="utf-8") as f:
    exec(f.read(), version_info)
__version__ = version_info["__version__"]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    ext_modules=[CMakeExtension("pyopenlpt", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
)
