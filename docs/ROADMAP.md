# Roadmap

## Phase 0: Project Foundation

- Repository structure: complete
- CMake build: complete
- Optional CUDA detection: complete
- Starter CPU backend: complete
- Smoke tests: complete
- Documentation and GitHub templates: complete

## Phase 1: Runtime Primitives

- Spike event representation: complete
- Synapse model: complete
- Deterministic scheduler: complete
- CPU reference simulation loop: complete
- LRU-backed neuron-state residency semantics: complete
- Spike-triggered neuron execution semantics: complete
- Synapse/dendrite/receptor processing: complete baseline
- STDP learning: complete baseline
- Synapse-specific spike-code emission and recognition: complete baseline

## Phase 2: Configuration

- Native JSON schema foundation: complete
- Config validation through `NetworkIR`: complete
- Declarative connectome construction: complete
- SONATA HDF5 node/edge loading: complete when HDF5 is available
- NeuroML structural parsing: complete baseline
- HOC structural extraction: complete baseline
- Reproducible run metadata: future

## Phase 3: CUDA Backend

- Device capability discovery: complete
- Memory layout design: complete baseline
- Receptor/STDP/dendrite kernels: complete baseline
- CPU/GPU equivalence tests: complete baseline
- Spike-batch scheduling that preserves event semantics: complete baseline
- Debug metrics: complete baseline
- Resident neuron-state cache on device memory: future
- Host/device paging policy for cold neuron states: future

## Phase 4: Experiments

- Minimal synthetic SNN tasks: complete baseline
- Spike-code accuracy benchmark: complete baseline
- Network learning accuracy experiment: complete baseline
- Throughput benchmark: complete baseline
- Vision-inspired adapters: keep in downstream projects unless generalized
- Diagnostics for propagation, separability, and performance: ongoing

## Phase 5: Library Stabilization

- Preserve installable CMake package compatibility.
- Expand C ABI only for stable cross-language surfaces.
- Keep experiments isolated from public APIs.
- Add semantic versioning and release artifacts.
- Add downstream consumer CI.
