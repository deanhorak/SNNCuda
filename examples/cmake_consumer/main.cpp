#include "snncuda/core/Version.h"

#include <iostream>

int main() {
    std::cout << "SNNCuda " << snncuda::core::version()
              << " cuda_available=" << snncuda::core::cuda_available() << '\n';
    return 0;
}
