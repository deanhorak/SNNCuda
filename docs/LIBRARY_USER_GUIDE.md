# SNNCuda Library User Guide

This guide is for projects that want to use SNNCuda as a dependency without
copying experiments, benchmarks, or application code into the library.

## Recommended Integration Model

Use SNNCuda as an installed CMake package:

```cmake
find_package(SNNCuda CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE SNNCuda::snncuda)
```

Keep your project-specific code in your own repository:

- stimulus generation
- datasets and adapters
- training scripts
- evaluation metrics
- application pipelines
- domain-specific experiments

Keep SNNCuda focused on reusable runtime pieces:

- public headers under `include/snncuda`
- runtime implementation under `src`
- declarative loading
- CPU/CUDA backends
- learning and temporal-code primitives
- packaging and compatibility tests

The `tests/` directory in this repository contains validation experiments and
benchmarks. Treat those as examples of how to test the substrate, not as APIs
to link from downstream projects.

## Install SNNCuda

CPU-only or auto-detected CUDA:

```bash
cmake -S . -B build-install -DCMAKE_BUILD_TYPE=Release
cmake --build build-install -j
cmake --install build-install --prefix /opt/snncuda
```

Force CPU-only:

```bash
cmake -S . -B build-install \
  -DCMAKE_BUILD_TYPE=Release \
  -DSNNCUDA_ENABLE_CUDA=OFF
cmake --build build-install -j
cmake --install build-install --prefix /opt/snncuda
```

Build a shared library for dynamic loading or Python:

```bash
cmake -S . -B build-shared \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON
cmake --build build-shared -j
cmake --install build-shared --prefix /opt/snncuda
```

## Minimal C++ Consumer

Project layout:

```text
my_app/
  CMakeLists.txt
  main.cpp
```

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.22)
project(MySNNApp LANGUAGES CXX)

find_package(SNNCuda CONFIG REQUIRED)

add_executable(my_snn_app main.cpp)
target_link_libraries(my_snn_app PRIVATE SNNCuda::snncuda)
target_compile_features(my_snn_app PRIVATE cxx_std_20)
```

`main.cpp`:

```cpp
#include "snncuda/core/Version.h"

#include <iostream>

int main() {
    std::cout << "SNNCuda " << snncuda::core::version()
              << " cuda_available=" << snncuda::core::cuda_available()
              << '\n';
}
```

Build:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/snncuda
cmake --build build -j
```

The repository also contains a complete sample under
`examples/cmake_consumer`.

## Building A Network In A Consumer

Downstream projects can construct `NetworkIR` directly, load one of the
supported declarative formats, or operate on a flattened `Connectome`.

Typical flow:

```cpp
#include "snncuda/declarative/Connectome.h"
#include "snncuda/runtime/NetworkPropagator.h"

using namespace snncuda;

declarative::NetworkIR ir;
// Fill neuron parameters, populations, and projections.

const auto connectome = declarative::ConnectomeBuilder{}.build(ir);
runtime::MemoryNeuronStateStore store;
runtime::NetworkPropagator propagator(connectome, store, 4096);

const auto input = connectome.populations.at("input").at(0);
propagator.inject(input, 0, 1.0F);
propagator.run_until(100);
```

For file-based loading, use `DeclarativeLoader` with Native JSON, SONATA,
NeuroML, or HOC. All supported formats normalize into `NetworkIR` and then
flatten into `Connectome`.

## Stable Public Surface

Prefer these headers in downstream projects:

- `snncuda/core/Version.h`
- `snncuda/declarative/NetworkIR.h`
- `snncuda/declarative/Connectome.h`
- `snncuda/declarative/DeclarativeLoader.h`
- `snncuda/runtime/NetworkPropagator.h`
- `snncuda/runtime/NeuronStateCache.h`
- `snncuda/runtime/SynapseDendriteProcessor.h`
- `snncuda/learning/STDP.h`
- `snncuda/snn/SpikeEvent.h`
- `snncuda/snn/Synapse.h`
- `snncuda/snn/TemporalPattern.h`
- `snncuda/backends/CudaBackend.h`

Avoid depending on files under `src/` or `tests/`; those are implementation and
validation details.

## Persistent CUDA Inference

For dataset inference, prefer `backends::CudaInferenceSession` over calling
`CudaBackend::run_resident_propagation` for every sample. The compatibility API
still works, but it packs and uploads CUDA state for each call. A session uploads
the immutable `Connectome` buffers once, then reuses device allocations for many
samples.

```cpp
#include "snncuda/backends/CudaBackend.h"

using namespace snncuda;

declarative::Connectome connectome = /* build or load once */;
backends::CudaInferenceSession session(connectome);

backends::CudaInferenceOptions options;
options.max_steps = 8;
options.reset_state_between_samples = true; // stateless inference by default
options.readout_neurons = class_neuron_ids;

std::vector<backends::CudaInferenceSample> samples;
samples.push_back({
    .inputs = {
        {.neuron = pixel_17},
        {.neuron = pixel_311},
    },
});

const auto result = session.run_batch(samples, options);
for (const auto& sample : result.samples) {
    // sample.readout_spike_counts is ordered like options.readout_neurons.
}
```

`CudaInferenceSample` uses `CudaWeightedInput`:

```cpp
struct CudaWeightedInput {
    core::NeuronId neuron;
    float weight{1.0F};
    std::uint32_t tick{0};
};
```

The current CUDA session treats each input as an externally fired neuron at its
requested tick. Use `weight = 1.0F` for active-neuron inference. The weight field
is part of the public shape so weighted external current injection can be added
without changing downstream call sites.

Each `CudaInferenceSampleResult` returns per-sample readout spike counts,
delivered spike count, fired spike count, and `CudaDebugMetrics`. `run_batch`
also reports total batch elapsed time. Stateless inference resets mutable
neuron/synapse/runtime buffers between samples. Set
`reset_state_between_samples = false` to preserve neuron and synapse state while
still clearing per-sample scheduler and metrics buffers.

## Python

The `python/` package is a small `ctypes` wrapper around the C ABI. It currently
exposes:

- `snncuda.version()`
- `snncuda.cuda_available()`

Build SNNCuda as a shared library and point `SNNCUDA_LIBRARY` at it:

```bash
cmake -S . -B build-shared -DBUILD_SHARED_LIBS=ON
cmake --build build-shared -j
cmake --install build-shared --prefix /opt/snncuda

cd python
python -m pip install .
export SNNCUDA_LIBRARY=/opt/snncuda/lib/libsnncuda.so
python -c "import snncuda; print(snncuda.version(), snncuda.cuda_available())"
```

If the shared library was built with CUDA enabled and the process can see the
NVIDIA driver/device, `snncuda.cuda_available()` returns `True`.

The Python wrapper is intentionally minimal. Broader simulation bindings should
be added through a stable C ABI first so C++, C, and Python consumers stay
compatible.

## Keeping Applications Separate

To keep SNNCuda stable as a library:

- Put app-specific training loops in the consumer project.
- Put dataset loaders and format adapters that are not general SNN formats in
  the consumer project.
- Use SNNCuda tests to validate runtime behavior, not to ship application logic.
- Add library features only when they are reusable across more than one
  application.
- Add new public APIs under `include/snncuda` and document them here.

This separation lets multiple downstream projects share the same runtime without
turning the core library into a collection of one-off experiments.
