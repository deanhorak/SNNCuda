# SNNCuda

SNNCuda is a new C++ spiking neural network codebase inspired by lessons from SNNFrame, but designed as a fresh implementation. The project targets a portable CPU-first runtime with optional CUDA acceleration when a CUDA toolkit and device are available.

## Current Status

This repository is in initial scaffolding. The current code provides:

- CMake-based C++20 project structure
- Optional CUDA backend detection
- Optional HDF5-backed SONATA node/edge loading
- Starter CPU backend
- Core hierarchy, learning, temporal-pattern, storage, and parser interfaces
- LRU-backed neuron state cache for the future CUDA residency model
- Minimal neuron and simulation-clock primitives
- Example executable
- Smoke tests
- GitHub Actions CI
- Documentation and contribution templates

## Design Direction

The initial project structure keeps room for the ideas that proved useful in SNNFrame:

- declarative experiment configuration
- separable CPU and accelerator backends
- measurable simulation boundaries
- stage-wise diagnostics before tuning complex decision logic
- clean experiment artifacts and benchmark reproducibility
- spike-triggered neuron execution with explicit state residency and eviction

The old implementation is not copied. Runtime, storage, network construction, CUDA kernels, and experiment harnesses are expected to be redesigned deliberately.

## Requirements

- CMake 3.22 or newer
- C++20 compiler
- Optional: NVIDIA CUDA Toolkit

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA is enabled automatically when `SNNCUDA_ENABLE_CUDA=ON` and CMake finds a CUDA compiler.

```bash
cmake -S . -B build -DSNNCUDA_ENABLE_CUDA=ON
```

To force CPU-only:

```bash
cmake -S . -B build -DSNNCUDA_ENABLE_CUDA=OFF
```

## Run

```bash
./build/snncuda
./build/examples/simple_spike
```

## Repository Layout

```text
include/snncuda/    Public headers
src/                Library and CLI implementation
examples/           Small runnable examples
tests/              Smoke and unit tests
configs/            Declarative configuration examples
docs/               Design and developer documentation
scripts/            Utility scripts
cmake/              CMake helper modules
.github/            GitHub workflows and templates
```

## Documentation

- [Developer Guide](docs/DEVELOPER_GUIDE.md)
- [Architecture Notes](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [Decision Log](docs/DECISIONS.md)

## License

MIT License. See [LICENSE](LICENSE).
