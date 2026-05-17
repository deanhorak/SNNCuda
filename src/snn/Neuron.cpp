#include "snncuda/snn/Neuron.h"

namespace snncuda::snn {

Neuron::Neuron(double threshold)
    : threshold_(threshold) {
}

void Neuron::receive(double weight) {
    membrane_potential_ += weight;
}

bool Neuron::should_fire() const noexcept {
    return membrane_potential_ >= threshold_;
}

void Neuron::reset() noexcept {
    membrane_potential_ = 0.0;
}

double Neuron::membrane_potential() const noexcept {
    return membrane_potential_;
}

double Neuron::threshold() const noexcept {
    return threshold_;
}

std::size_t Neuron::spike_count() const noexcept {
    return spike_count_;
}

void Neuron::fire() {
    if (should_fire()) {
        ++spike_count_;
        reset();
    }
}

} // namespace snncuda::snn

