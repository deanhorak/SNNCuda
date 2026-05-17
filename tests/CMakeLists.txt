add_executable(snncuda_smoke_tests
    smoke_tests.cpp
)

target_link_libraries(snncuda_smoke_tests PRIVATE SNNCuda::snncuda)
snncuda_apply_common_options(snncuda_smoke_tests)

add_test(NAME snncuda_smoke_tests COMMAND snncuda_smoke_tests)

add_executable(snncuda_parser_tests
    parser_tests.cpp
)

target_link_libraries(snncuda_parser_tests PRIVATE SNNCuda::snncuda)
target_compile_definitions(snncuda_parser_tests
    PRIVATE
        SNNCUDA_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)
snncuda_apply_common_options(snncuda_parser_tests)

add_test(NAME snncuda_parser_tests COMMAND snncuda_parser_tests)

add_executable(snncuda_propagation_tests
    propagation_tests.cpp
)

target_link_libraries(snncuda_propagation_tests PRIVATE SNNCuda::snncuda)
snncuda_apply_common_options(snncuda_propagation_tests)

add_test(NAME snncuda_propagation_tests COMMAND snncuda_propagation_tests)

add_executable(snncuda_spike_throughput
    spike_throughput_benchmark.cpp
)

target_link_libraries(snncuda_spike_throughput PRIVATE SNNCuda::snncuda)
snncuda_apply_common_options(snncuda_spike_throughput)
