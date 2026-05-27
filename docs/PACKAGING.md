# Packaging SNNCuda

SNNCuda installs as a CMake package with an exported target:

```cmake
find_package(SNNCuda CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE SNNCuda::snncuda)
```

For downstream project structure and API boundaries, see
[SNNCuda Library User Guide](LIBRARY_USER_GUIDE.md).

## Install

```bash
cmake -S . -B build-install -DCMAKE_BUILD_TYPE=Release
cmake --build build-install -j
cmake --install build-install --prefix /opt/snncuda
```

For a shared library suitable for dynamic loading from Python:

```bash
cmake -S . -B build-shared -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build-shared -j
cmake --install build-shared --prefix /opt/snncuda
```

## Consume From CMake

If SNNCuda is installed in a non-standard prefix, point CMake at it:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/snncuda
cmake --build build -j
```

Minimal consumer:

```cpp
#include "snncuda/core/Version.h"

#include <iostream>

int main() {
    std::cout << snncuda::core::version() << '\n';
}
```

```cmake
cmake_minimum_required(VERSION 3.22)
project(Consumer LANGUAGES CXX)

find_package(SNNCuda CONFIG REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE SNNCuda::snncuda)
```

## Python Wrapper

The `python/` directory contains a pure-Python `ctypes` wrapper around the C ABI.
It intentionally avoids requiring pybind11 for basic integration.

```bash
cmake -S . -B build-shared -DBUILD_SHARED_LIBS=ON
cmake --build build-shared -j
cmake --install build-shared --prefix /opt/snncuda

cd python
python -m pip install .
export SNNCUDA_LIBRARY=/opt/snncuda/lib/libsnncuda.so
python -c "import snncuda; print(snncuda.version(), snncuda.cuda_available())"
```

The current Python API exposes version and CUDA availability. Larger bindings can
be layered on top of the stable C ABI without affecting the C++ package surface.

If Conda's `libstdc++` is older than the compiler runtime used to build SNNCuda,
use system Python or update Conda's C++ runtime. A typical symptom is a missing
`GLIBCXX_3.4.xx` symbol while loading `libsnncuda.so`.
