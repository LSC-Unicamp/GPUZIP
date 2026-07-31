import os
import subprocess
import sys
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = Path(__file__).parent.resolve()
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        build_args = []
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            f"-DVERSION_INFO={self.distribution.get_version()}",
        ]

        # Get possible CMAKE_ARGS in the environment
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="gpuzipy",
    version="0.0.1",
    author="Thiago Maltempi",
    author_email="maltempi@ic.unicamp.br",
    description="Python bindings for GPUZIP's GPU compression layer (Bitcomp, cuZFP, cuSZp).",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/LSC-Unicamp/GPUZIP",
    project_urls={
        "Source": "https://github.com/LSC-Unicamp/GPUZIP",
        "Documentation": "https://github.com/LSC-Unicamp/GPUZIP/blob/main/docs/PythonExamples.md",
        "Bug Tracker": "https://github.com/LSC-Unicamp/GPUZIP/issues",
    },
    license="MIT",
    license_files=["LICENSE"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    ext_modules=[CMakeExtension("gpuzipy")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    extras_require={
        "test": ["pytest>=6.0"],
        "example": ["cupy>=13.0.0"],
    },
    python_requires=">=3.7",
)
