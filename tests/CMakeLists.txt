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

add_executable(snncuda_synapse_dendrite_tests
    synapse_dendrite_tests.cpp
)

target_link_libraries(snncuda_synapse_dendrite_tests PRIVATE SNNCuda::snncuda)
snncuda_apply_common_options(snncuda_synapse_dendrite_tests)

add_test(NAME snncuda_synapse_dendrite_tests COMMAND snncuda_synapse_dendrite_tests)

add_executable(snncuda_learning_temporal_experiment
    learning_temporal_experiment.cpp
)

target_link_libraries(snncuda_learning_temporal_experiment PRIVATE SNNCuda::snncuda)
snncuda_apply_common_options(snncuda_learning_temporal_experiment)

add_test(NAME snncuda_learning_temporal_experiment COMMAND snncuda_learning_temporal_experiment)

if(SNNCUDA_HAS_CUDA)
    add_executable(snncuda_cuda_synapse_dendrite_tests
        cuda_synapse_dendrite_tests.cpp
    )

    target_link_libraries(snncuda_cuda_synapse_dendrite_tests PRIVATE SNNCuda::snncuda)
    snncuda_apply_common_options(snncuda_cuda_synapse_dendrite_tests)

    add_test(NAME snncuda_cuda_synapse_dendrite_tests COMMAND snncuda_cuda_synapse_dendrite_tests)

    add_executable(snncuda_cuda_learning_consistency_experiment
        cuda_learning_consistency_experiment.cpp
    )

    target_link_libraries(snncuda_cuda_learning_consistency_experiment PRIVATE SNNCuda::snncuda)
    snncuda_apply_common_options(snncuda_cuda_learning_consistency_experiment)

    add_test(
        NAME snncuda_cuda_learning_consistency_experiment
        COMMAND snncuda_cuda_learning_consistency_experiment
    )
endif()

add_executable(snncuda_spike_throughput
    spike_throughput_benchmark.cpp
)

target_link_libraries(snncuda_spike_throughput PRIVATE SNNCuda::snncuda)
snncuda_apply_common_options(snncuda_spike_throughput)
