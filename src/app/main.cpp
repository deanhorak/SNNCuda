#include "snncuda/backends/CpuBackend.h"
#include "snncuda/core/Version.h"
#include "snncuda/runtime/SimulationClock.h"
#include "snncuda/snn/Neuron.h"

#include <iostream>

int main() {
    snncuda::backends::CpuBackend cpu;
    snncuda::runtime::SimulationClock clock;
    snncuda::snn::Neuron neuron{1.0};

    neuron.receive(1.25);
    neuron.fire();
    clock.advance();

    std::cout << "SNNCuda " << snncuda::core::version() << '\n';
    std::cout << "CPU backend: " << (cpu.available() ? "available" : "unavailable") << '\n';
    std::cout << "CUDA compiled: " << (snncuda::core::cuda_available() ? "yes" : "no") << '\n';
    std::cout << "Simulation time: " << clock.time_ms() << " ms\n";
    std::cout << "Neuron spikes: " << neuron.spike_count() << '\n';

    return 0;
}

