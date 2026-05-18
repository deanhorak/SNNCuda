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

## Learned Temporal Patterns

SNNCuda treats spike timing as part of the signal, not just spike count. Each neuron carries a small temporal-pattern state alongside its membrane and dendritic compartment state. When a spike reaches the synapse/dendrite processing stage, the target neuron records the spike's offset from the start of its current temporal window. The window length, similarity threshold, and maximum learned patterns come from the neuron's declarative parameters.

The first repeated-enough spike sequence in a window becomes a learned reference pattern. Later spike sequences are compared against the learned reference using cosine similarity over their offset vectors. If the similarity crosses the neuron's threshold, the neuron records a temporal match. This gives the runtime a direct way to distinguish different spike-time codes even when the same neurons participate.

The CPU runtime stores this state in `NeuronState::temporal_pattern`. The CUDA resident runtime keeps a compact per-neuron version on device: observed offsets, one learned reference pattern, learned-pattern count, match count, and last-match state. The CUDA implementation currently uses a fixed offset capacity of 8 and one learned reference pattern per resident neuron. This is enough for the current experiments and provides a concrete baseline before expanding to multiple reference patterns per neuron.

Temporal recognition is exercised by the learning consistency experiments. The CPU experiment verifies that repeated spike intervals are learned and later recognized. The CUDA consistency experiment runs the same learning and propagation path on the GPU and verifies that temporal match counts and learned-pattern counts agree with the CPU reference.

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
