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

Benchmarks and experiments live in `tests/` as executable validation artifacts.
They are not part of the public library API.

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

Keep application-specific code out of the library. If a feature is only useful
for one downstream project, put it in that project. Add it to SNNCuda only when
it is a reusable runtime, parser, backend, storage, learning, or diagnostics
capability.

## Packaging Checks

Before changing public headers, install/export logic, or dependency handling,
verify an out-of-tree consumer:

```bash
cmake -S . -B build-install -DCMAKE_BUILD_TYPE=Release
cmake --build build-install -j
cmake --install build-install --prefix /tmp/snncuda-install

cmake -S examples/cmake_consumer -B /tmp/snncuda-consumer \
  -DCMAKE_PREFIX_PATH=/tmp/snncuda-install
cmake --build /tmp/snncuda-consumer -j
/tmp/snncuda-consumer/snncuda_consumer
```
