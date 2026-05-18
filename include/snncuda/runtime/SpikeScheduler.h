#pragma once

#include "snncuda/snn/SpikeEvent.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace snncuda::runtime {

class SpikeScheduler {
public:
    explicit SpikeScheduler(std::size_t wheel_slots = 4096);

    void schedule(snn::SpikeEvent event);
    [[nodiscard]] std::vector<snn::SpikeEvent> pop_due(std::uint64_t tick);
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::size_t pending_count() const noexcept;
    [[nodiscard]] std::size_t wheel_slots() const noexcept;

private:
    using Bucket = std::vector<snn::SpikeEvent>;

    [[nodiscard]] std::size_t slot_for(std::uint64_t tick) const noexcept;

    std::vector<Bucket> wheel_;
    std::uint64_t next_pop_tick_{0};
    std::size_t pending_count_{0};
};

} // namespace snncuda::runtime
