# SNNCuda

SNNCuda is a new C++ spiking neural network codebase inspired by lessons from SNNFrame, but designed as a fresh implementation. The project targets a portable CPU-first runtime with optional CUDA acceleration when a CUDA toolkit and device are available.

## Current Status

SNNCuda is now usable as an installable C++ library with a CPU reference runtime,
optional CUDA acceleration, declarative connectome loading, STDP learning,
synapse/dendrite/receptor processing, temporal spike-code recognition, and
benchmark coverage. The current code provides:

- Installable CMake package exporting `SNNCuda::snncuda`
- Optional CUDA backend with CPU/CUDA learning consistency tests
- Optional HDF5-backed SONATA node/edge loading
- Native JSON, SONATA, NeuroML, and HOC parsers that normalize into `NetworkIR`
- Flattened `Connectome` runtime representation
- Timing-wheel spike scheduler
- CPU reference propagation through synapse/dendrite/receptor processing
- CUDA receptor/STDP/dendrite kernels with debug metrics
- STDP with runtime plasticity enable/disable
- Synapse-level emitted spike-code patterns and recognition metrics
- Neuron-level temporal pattern learning and recognition
- LRU-backed neuron state cache for resident-state paging semantics
- C ABI for basic dynamic loading and Python interop
- Pure-Python `ctypes` wrapper for version/CUDA availability checks
- Structural parser tests, propagation tests, learning experiments, and accuracy benchmarks

## Design Direction

The library keeps the SNNFrame ideas that proved useful while avoiding
experiment-specific baggage:

- declarative experiment configuration
- separable CPU and accelerator backends
- measurable simulation boundaries
- stage-wise diagnostics before tuning complex decision logic
- clean experiment artifacts and benchmark reproducibility outside the public API
- spike-triggered neuron execution with explicit state residency and eviction

The old implementation is not copied. Runtime, storage, network construction,
CUDA kernels, and benchmark harnesses are deliberately implemented in this
codebase.

## Learned Temporal Patterns

SNNCuda treats spike timing as part of the signal, not just spike count. Each neuron carries a small temporal-pattern state alongside its membrane and dendritic compartment state. When a spike reaches the synapse/dendrite processing stage, the target neuron records the spike's offset from the start of its current temporal window. The window length, similarity threshold, and maximum learned patterns come from the neuron's declarative parameters.

The first repeated-enough spike sequence in a window becomes a learned reference pattern. Later spike sequences are compared against the learned reference using cosine similarity over their offset vectors. If the similarity crosses the neuron's threshold, the neuron records a temporal match. This gives the runtime a direct way to distinguish different spike-time codes even when the same neurons participate.

The CPU runtime stores this state in `NeuronState::temporal_pattern`. The CUDA resident runtime keeps a compact per-neuron version on device: observed offsets, one learned reference pattern, learned-pattern count, match count, and last-match state. The CUDA implementation currently uses a fixed offset capacity of 8 and one learned reference pattern per resident neuron. This is enough for the current experiments and provides a concrete baseline before expanding to multiple reference patterns per neuron.

Temporal recognition is exercised by the learning consistency experiments. The CPU experiment verifies that repeated spike intervals are learned and later recognized. The CUDA consistency experiment runs the same learning and propagation path on the GPU and verifies that temporal match counts and learned-pattern counts agree with the CPU reference.

## Synapse Spike Codes

When a neuron fires, each outgoing synapse can emit a synapse-specific temporal
code instead of a single spike. The code is configured as `spike_code_offsets`,
and downstream delivery schedules one event per offset:

```text
delivery_tick = firing_tick + synapse.delay_ticks + spike_code_offset
```

This makes the emitted spike train a property of the synapse. The receiving
synapse tracks learned code patterns separately from the target neuron's
temporal-pattern state, so the runtime can measure whether the synapse
recognizes repeated upstream identity codes. CPU and CUDA paths both support
code expansion and recognition.

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

## Install And Use As A Library

```bash
cmake -S . -B build-install -DCMAKE_BUILD_TYPE=Release
cmake --build build-install -j
cmake --install build-install --prefix /opt/snncuda
```

Then consume it from another CMake project:

```cmake
find_package(SNNCuda CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE SNNCuda::snncuda)
```

For Python, build a shared library and use the wrapper in `python/`:

```bash
cmake -S . -B build-shared -DBUILD_SHARED_LIBS=ON
cmake --build build-shared -j
cmake --install build-shared --prefix /opt/snncuda
cd python
python -m pip install .
export SNNCUDA_LIBRARY=/opt/snncuda/lib/libsnncuda.so
python -c "import snncuda; print(snncuda.version(), snncuda.cuda_available())"
```

See [Packaging SNNCuda](docs/PACKAGING.md) for a complete consumer example.
For downstream application setup, see the [Library User Guide](docs/LIBRARY_USER_GUIDE.md).

## Run

```bash
./build/snncuda
./build/examples/simple_spike
```

## Repository Layout

```text
include/snncuda/    Public headers
src/                Library, C ABI, and CLI implementation
examples/           Small runnable examples and CMake consumer sample
tests/              Tests, benchmarks, and experiments; not part of the public API
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
- [Packaging SNNCuda](docs/PACKAGING.md)
- [Library User Guide](docs/LIBRARY_USER_GUIDE.md)

## License

MIT License. See [LICENSE](LICENSE).
