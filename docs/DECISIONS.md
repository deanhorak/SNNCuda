# Decision Log

This file records design decisions that should remain visible as the project evolves.

## 2026-05-15: Start As A Fresh Codebase

SNNCuda is loosely inspired by SNNFrame concepts, but it is not a port. The first scaffold establishes repository hygiene, CMake, optional CUDA detection, smoke tests, and documentation before choosing runtime architecture.

## 2026-05-15: CPU First, CUDA Optional

The CPU backend is the semantic reference. CUDA support is optional at configure time and should only accelerate behavior already covered by CPU tests.

## 2026-05-15: Carry Forward Boundaries, Not Experiment Baggage

The new codebase keeps the SNNFrame ideas that are still structurally useful: hierarchy, cortical layers, STDP, temporal patterns, persistent storage, and declarative loading. It does not carry over Retina benchmark implementation details, CIFAR experiment detours, or decision-stage tuning paths.

## 2026-05-15: Model GPU Residency Explicitly

The CUDA runtime will treat resident neuron state as a cache. Spike arrival wakes computation for the target neuron, and state is loaded or evicted through an LRU policy when the number of active neurons exceeds resident GPU capacity. A CPU/reference `NeuronStateCache` exists first so paging semantics can be tested independent of kernels.
