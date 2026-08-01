FROM cupy/cupy:v13.6.0

# git is required by GPUZIPy's build: CMake fetches Thrust/ZFP/cuSZp via FetchContent's
# GIT_REPOSITORY, which shells out to `git clone`.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Build context is the repo root (see README.md), so this grabs src/ and GPUZIPy/ together.
WORKDIR /GPUZIP
COPY . /GPUZIP

WORKDIR /GPUZIP/GPUZIPy
RUN pip install .
RUN pip install -r example/requirements.txt

CMD ["python3", "example/main.py"]
