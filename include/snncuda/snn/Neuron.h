#pragma once

#include <cstddef>
#include <vector>

namespace snncuda::snn {

class Neuron {
public:
    explicit Neuron(double threshold = 1.0);

    void receive(double weight);
    [[nodiscard]] bool should_fire() const noexcept;
    void reset() noexcept;

    [[nodiscard]] double membrane_potential() const noexcept;
    [[nodiscard]] double threshold() const noexcept;
    [[nodiscard]] std::size_t spike_count() const noexcept;

    void fire();

private:
    double membrane_potential_{0.0};
    double threshold_{1.0};
    std::size_t spike_count_{0};
};

using NeuronPopulation = std::vector<Neuron>;

} // namespace snncuda::snn

