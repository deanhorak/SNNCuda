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
samples. Stateless batches execute with a device-side batch dimension, so
samples are processed together instead of one host-driven CUDA run per sample.

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

The CUDA session applies each input as external soma current at its requested
tick. Inputs with the same neuron and tick accumulate. The neuron is added to the
fired queue only when the accumulated external input crosses that neuron's
threshold. Use `weight = 1.0F` for active-neuron inference with input neurons
whose threshold is `1.0F`.

Each `CudaInferenceSampleResult` returns per-sample readout spike counts,
readout scores when available, delivered spike count, fired spike count, and
`CudaDebugMetrics`. `run_batch` also reports total batch elapsed time. Stateless
inference resets mutable neuron/synapse/runtime buffers between samples. Set
`reset_state_between_samples = false` to preserve neuron and synapse state while
still clearing per-sample scheduler and metrics buffers. Stateful mode currently
uses the sequential persistent-session path because preserving state creates an
ordered dependency between samples.

For fixed-weight feedforward classifiers, use `run_feedforward_batch` instead
of the spiking path. It skips pixel-neuron simulation, temporal pattern state,
STDP scans, dendritic decay, scheduler queues, and per-tick host synchronization.
The CUDA kernel directly accumulates:

```text
readout_score[class] += input_weight[pixel] * synapse_weight[pixel -> class]
```

Results use `options.readout_neurons` order. `readout_scores` contains the raw
class currents, and `readout_spike_counts` is thresholded against each readout
neuron's configured threshold.

```cpp
backends::CudaInferenceOptions readout_options;
readout_options.readout_neurons = class_neuron_ids;

std::vector<backends::CudaInferenceSample> images;
images.push_back({
    .inputs = {
        {.neuron = pixel_17, .weight = 0.42F},
        {.neuron = pixel_311, .weight = 0.91F},
    },
});

const auto scores = session.run_feedforward_batch(images, readout_options);
const auto& class_scores = scores.samples.front().readout_scores;
```

Use `run_recurrent_feedforward_batch` when a fixed-weight classifier also has
readout-to-feature feedback edges and you want a small number of recurrent
settling iterations without entering the full spike scheduler:

```cpp
backends::CudaRecurrentFeedforwardOptions recurrent_options;
recurrent_options.iterations = 2;
recurrent_options.hidden_neurons = pixel_neuron_ids;
recurrent_options.readout_neurons = class_neuron_ids;
recurrent_options.feedback_decay = 0.5F;
recurrent_options.top_k_feedback_readouts = 3;
recurrent_options.top_k_hidden = 128;
recurrent_options.use_scores_not_spikes = true;

const auto settled = session.run_recurrent_feedforward_batch(images, recurrent_options);
```

The recurrent feedforward path starts from weighted input features at recurrent
step zero, scores readouts through fixed synapse weights, selects readouts by
top-k or threshold, propagates selected readout feedback to hidden neurons,
optionally keeps only the top-k hidden features, and re-scores readouts. Unlike
the single-pass feedforward path, recurrent feedforward explicitly honors each
projection's `delay_ticks`: a hidden-to-readout or readout-to-hidden contribution
is delivered at `current_step + delay_ticks`. `iterations` controls the number
of feedback rounds after the initial delayed readout pass. Final
`readout_scores` are accumulated over the delayed readout event timeline and
then thresholded into `readout_spike_counts`. Feedback uses scores when
`use_scores_not_spikes` is true, otherwise selected readouts contribute a binary
`1.0F` gate.

For latency-sensitive models, STDP experiments, or temporal coding, enable the
full spike-timing path:

```cpp
backends::CudaRecurrentFeedforwardOptions timing_options;
timing_options.iterations = 2;
timing_options.readout_neurons = class_neuron_ids;
timing_options.use_full_spike_timing = true;
timing_options.max_timing_steps = 32;

const auto timed = session.run_recurrent_feedforward_batch(images, timing_options);
```

With `use_full_spike_timing` enabled, `run_recurrent_feedforward_batch` delegates
to the resident CUDA spiking scheduler instead of the compact recurrent score
timeline. Weighted external inputs are applied at their requested ticks, neuron
thresholds determine spikes, synapse events are delivered by `delay_ticks`, and
the normal dendritic, temporal, and plasticity accounting paths run. Results are
reported as readout spike counts plus `CudaDebugMetrics`; `readout_scores` are
left empty because this mode is a timed spike simulation rather than direct
projection scoring. If `max_timing_steps` is zero, the session derives a timing
window from `iterations` and the largest projection delay.

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
