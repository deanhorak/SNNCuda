# Architecture Notes

SNNCuda is a fresh C++ implementation of a spiking neural network runtime with
optional CUDA acceleration. It is now structured as a reusable library, with
experiments and benchmarks kept in `tests/` so downstream applications can link
the runtime without inheriting project-specific workflows.

## Initial Boundaries

- `snn`: neuron, synapse, population, and spike primitives
- `runtime`: simulation timing, event scheduling, execution loops
- `backends`: CPU and CUDA execution implementations
- `config`: declarative configuration loading
- `tests`: reproducible validation, benchmark, and research harnesses
- `diagnostics`: activity, separability, and performance measurement
- `hierarchy`: brain-scale structural organization
- `learning`: STDP and future learning rules
- `storage`: persistent object and neuron-state storage

## Principles

- CPU behavior is the reference implementation.
- CUDA paths must match CPU semantics before being optimized.
- Experiment surfaces should be isolated from reusable runtime code.
- Diagnostic visibility comes before complex learning or decision tuning.

## CUDA Neuron Residency Model

The intended runtime model is spike-triggered execution:

1. A spike event identifies a target neuron.
2. The runtime asks a resident neuron-state cache for the target state.
3. If the state is resident, it is used directly.
4. If the state is not resident, it is loaded from backing storage.
5. If the resident cache is full, the least recently used neuron state is evicted and saved.
6. The CUDA backend wakes work for the target neuron and applies the spike computation.

The current `NeuronStateCache` is a CPU/reference abstraction for this policy.
It is intentionally separate from CUDA kernels so eviction semantics can be
tested before device memory management is optimized. CUDA kernels currently run
resident propagation over compact device arrays; host/device cold-state paging
is still a future optimization.

## Spike Delivery Queue

Spike delivery uses a circular timing wheel rather than a heap. Each simulation tick maps to a bucket by `delivery_tick % wheel_slots`, and each bucket stores the spike events for that temporal slot. Popping due spikes advances a monotonic cursor and drains all buckets up to the requested tick.

This is the default because fixed-timestep SNN simulations schedule large numbers of near-future spikes. A timing wheel gives O(1) amortized insertion and avoids heap churn in dense traffic. The implementation still stores absolute delivery ticks inside each bucket, so events farther in the future than one wheel rotation are retained until their real tick arrives.

The CUDA path translates fired neurons and scheduled synapse events into compact
device arrays. `CudaInferenceSession` also provides a separate fixed-weight
feedforward readout path for classifier-style networks where weighted inputs
project directly to readout neurons. That path bypasses pixel-neuron firing,
temporal/STDP machinery, dendritic state, and per-tick scheduler queues, and
accumulates readout scores as `input_weight * synapse_weight` on the device. A
recurrent feedforward variant repeats this direct projection with readout
top-k/threshold gating and readout-to-feature feedback on a compact recurrent
event timeline that honors per-projection `delay_ticks`, still avoiding the full
spiking scheduler. When `CudaRecurrentFeedforwardOptions::use_full_spike_timing`
is enabled, the same public API switches to the resident CUDA spiking scheduler:
weighted inputs are injected on explicit ticks, projection delays schedule real
synaptic events, and readout results are gathered from timed spikes instead of
direct projection scores.
Future paging work should add:

- group due spikes by target neuron or resident-state page
- load cold neuron state through the LRU residency layer
- launch or enqueue device work for active target sets
- preserve absolute tick ordering for STDP and temporal pattern windows

## Propagation Plumbing

`NetworkPropagator` is the current CPU/reference propagation path. It consumes a flattened
`Connectome`, initializes neuron state from connectome neuron parameters, accepts external
spike injections, processes due spikes through the timing wheel, and schedules downstream
events through outgoing synapses whenever a neuron fires.

This is intentionally small and deterministic. CUDA kernels should match this behavior before
introducing batching, device-resident state pages, or parallel delivery optimizations.

## Synapse/Dendrite/Receptor Stage

Spikes are not applied directly as "weight into soma membrane." They enter a
synapse/dendrite processing stage:

1. receptor dynamics convert synapse weight into signed current
2. current is integrated into soma, basal, apical, or inhibitory compartments
3. compartments decay by receptor-specific time constants
4. membrane potential is computed from the compartment state
5. firing resets the neuron and schedules downstream synapse events
6. STDP updates run on pre/post spike timing when plasticity is enabled

CPU and CUDA paths both implement this stage.

## Synapse Spike Codes

Each synapse can define `spike_code_offsets`. When the source neuron fires, the
runtime schedules one spike for each offset. This makes temporal identity a
synapse property rather than only a neuron property. The receiving synapse keeps
its own learned code-pattern state and match counters, while the target neuron
continues to maintain neuron-level temporal-pattern state.

The CUDA implementation supports code expansion and synapse-code recognition
with a fixed capacity of 8 offsets per code.

## Core Framework Direction

The codebase now has reusable library types for:

- brain hierarchy: `Brain -> Hemisphere -> Lobe -> Region -> Nucleus -> Column -> Layer -> Cluster -> Neuron`
- canonical cortical columns with named `L1`, `L2/3`, `L4`, `L5`, and `L6` layer groups
- spike events and retrograde events
- STDP enable/disable and causal weight updates
- temporal pattern matching
- synapse-specific emitted spike codes
- object storage and LRU state caching
- parser registry and declarative loader interfaces
- installable CMake package and minimal C ABI

The SNNFrame concepts are carried forward as boundaries and data models, not as experiment-specific code.

## SNNFrame Lessons To Preserve

- Keep declarative configuration as a first-class path.
- Treat bilateral or multi-stage pipelines as measurable compositions.
- Freeze learning during evaluation unless an experiment explicitly studies online adaptation.
- Protect known baselines from speculative changes.

## Deferred Implementation Choices

- RocksDB is represented by a storage boundary; the default build does not
  require RocksDB.
- Native JSON, SONATA, NeuroML, and HOC now normalize into `NetworkIR`, then flatten into `Connectome`.
- SONATA support parses the JSON circuit config, SNNCuda/SNNFrame extension sections, and direct HDF5 node/edge files when HDF5 is available.
- CUDA kernels should start by implementing CPU-equivalent neuron-state updates, then add batching and memory-residency optimization.

## Declarative Loading And Connectome

All supported formats target the same two-stage in-memory representation:

1. `NetworkIR`: hierarchical, format-neutral network description with neuron parameter sets and projections.
2. `Connectome`: flattened neuron and synapse lists plus population indexes for runtime delivery.

Current parser coverage:

- Native JSON: `.snnf.json` and `.snncuda.json`, including flat network configs, hierarchy configs, column templates, reusable neuron params, and projections.
- SONATA: `circuit_config.json` and `.sonata.json`, reading `snncuda` or `snnframe` extension blocks from the JSON config plus HDF5 `/nodes/<population>/node_id`, `/nodes/<population>/node_type_id`, `/edges/<population>/source_node_id`, `/edges/<population>/target_node_id`, `/edges/<population>/weight`, and `/edges/<population>/delay`.
- NeuroML: `.nml` and `.neuroml`, extracting cells, populations, and projections from common NeuroML v2 XML structure plus `snnfw:` properties.
- HOC: `.hoc`, extracting template parameters, loop-instantiated populations, and `NetCon` projections for structural imports.

The parser fixture battery is intentionally small but structural: each fixture must produce a valid
tree-shaped `NetworkIR`, then a flattened connectome with expected populations, neuron counts,
synapse counts, weights, delays, and source/target population mappings.
