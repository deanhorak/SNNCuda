#include "snncuda/backends/CpuBackend.h"

namespace snncuda::backends {

std::string_view CpuBackend::name() const noexcept {
    return "cpu";
}

bool CpuBackend::available() const noexcept {
    return true;
}

} // namespace snncuda::backends

