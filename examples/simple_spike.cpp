#include "snncuda/runtime/SimulationClock.h"
#include "snncuda/snn/Neuron.h"

#include <iostream>

int main() {
    snncuda::runtime::SimulationClock clock{0.5};
    snncuda::snn::Neuron neuron{1.0};

    for (int i = 0; i < 4; ++i) {
        neuron.receive(0.3);
        neuron.fire();
        clock.advance();
    }

    std::cout << "time_ms=" << clock.time_ms() << '\n';
    std::cout << "spikes=" << neuron.spike_count() << '\n';
    std::cout << "membrane=" << neuron.membrane_potential() << '\n';

    return neuron.spike_count() == 1 ? 0 : 1;
}

