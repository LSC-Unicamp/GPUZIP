## Installing cuZFP

To use cuZFP, you need to install it. You can either use our Docker container (`maltempi/awave-dev:ompc`) or install it manually by following the steps below:

```sh
# Shallow clone the repository
git clone --branch 1.0.1 --depth 1 git@github.com:LLNL/zfp.git

# Navigate to the zfp directory
cd zfp

# Create and navigate to the build directory
mkdir build
cd build

# Configure the build with CUDA support
cmake -DZFP_WITH_CUDA=1 ..

# Compile the code
make
```

After installation, ensure that the `CMakeLists.txt` file in your GPUZIP project points to the correct path where cuZFP is installed. Our example uses the `/opt/zfp/include` directory.

### Additional Information:
- [Using CUDA with ZFP](https://zfp.readthedocs.io/en/release1.0.1/execution.html#using-cuda)
- [ZFP Installation Guide](https://zfp.readthedocs.io/en/release1.0.1/installation.html#c.ZFP_WITH_CUDA)