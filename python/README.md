# SNNCuda Python Wrapper

This package is a small `ctypes` wrapper around an installed SNNCuda shared
library. Build and install SNNCuda with `BUILD_SHARED_LIBS=ON`, then point
`SNNCUDA_LIBRARY` at the installed shared library if it is not on your platform
library path.

```bash
cmake -S .. -B ../build-shared -DBUILD_SHARED_LIBS=ON
cmake --build ../build-shared -j
cmake --install ../build-shared --prefix /tmp/snncuda

export SNNCUDA_LIBRARY=/tmp/snncuda/lib/libsnncuda.so
python -m pip install .
python -c "import snncuda; print(snncuda.version(), snncuda.cuda_available())"
```

If Conda's `libstdc++` is older than the compiler runtime used to build
SNNCuda, use system Python or update Conda's C++ runtime. A typical symptom is
an error mentioning a missing `GLIBCXX_3.4.xx` symbol while loading
`libsnncuda.so`.
