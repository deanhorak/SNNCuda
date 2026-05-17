# Roadmap

## Phase 0: Project Foundation

- Repository structure
- CMake build
- Optional CUDA detection
- Starter CPU backend
- Smoke tests
- Documentation and GitHub templates

## Phase 1: Runtime Primitives

- Spike event representation
- Synapse model
- Population containers
- Deterministic scheduler
- CPU reference simulation loop
- LRU-backed neuron-state residency
- Spike-triggered neuron execution semantics

## Phase 2: Configuration

- Native JSON schema foundation
- Config validation through `NetworkIR`
- Declarative connectome construction
- Reproducible run metadata
- SONATA HDF5 node/edge loading
- NeuroML parser dependency selection
- HOC structural extraction strategy

## Phase 3: CUDA Backend

- Device capability discovery
- Memory layout design
- Kernel prototypes
- CPU/GPU equivalence tests
- Resident neuron-state cache on device memory
- Host/device paging policy for cold neuron states
- Spike-batch scheduling that preserves event semantics

## Phase 4: Experiments

- Minimal synthetic SNN tasks
- Vision-inspired adapters
- Benchmark harness
- Diagnostics for propagation, separability, and performance
