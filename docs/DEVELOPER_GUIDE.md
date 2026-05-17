# Developer Guide

## Configure

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
```

## Build

```bash
cmake --build build -j
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

Parser and connectome coverage lives in `tests/parser_tests.cpp` with committed fixtures under
`tests/fixtures/`. Those tests verify that Native JSON, SONATA JSON, SONATA HDF5, NeuroML,
and HOC all normalize into `NetworkIR` and flatten into the same `Connectome` structure.

## Optional CUDA

CUDA is controlled by `SNNCUDA_ENABLE_CUDA`.

```bash
cmake -S . -B build-cuda -DSNNCUDA_ENABLE_CUDA=ON
```

If CMake cannot find a CUDA compiler, the project falls back to CPU-only.

## Adding Code

- Public APIs go in `include/snncuda`.
- Implementations go in `src`.
- Tests go in `tests`.
- Examples go in `examples`.
- Design notes go in `docs`.
