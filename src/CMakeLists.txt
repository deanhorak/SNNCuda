add_library(snncuda
    c/SNNCudaC.cpp
    core/Version.cpp
    declarative/Connectome.cpp
    declarative/DeclarativeLoader.cpp
    declarative/FormatParser.cpp
    declarative/NetworkIR.cpp
    declarative/parsers/BuiltinParsers.cpp
    hierarchy/CorticalMicrocircuit.cpp
    hierarchy/Hierarchy.cpp
    learning/STDP.cpp
    runtime/NetworkPropagator.cpp
    runtime/SimulationClock.cpp
    runtime/NeuronStateCache.cpp
    runtime/SpikeScheduler.cpp
    runtime/SynapseDendriteProcessor.cpp
    snn/Neuron.cpp
    snn/TemporalPattern.cpp
    backends/cpu/CpuBackend.cpp
    storage/MemoryObjectStore.cpp
    storage/RocksDbObjectStore.cpp
)

add_library(SNNCuda::snncuda ALIAS snncuda)

target_include_directories(snncuda
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(snncuda
    PUBLIC
        SNNCUDA_VERSION="${PROJECT_VERSION}"
        SNNCUDA_HAS_CUDA=$<BOOL:${SNNCUDA_HAS_CUDA}>
        SNNCUDA_HAS_HDF5=$<BOOL:${SNNCUDA_HAS_HDF5}>
)

snncuda_apply_common_options(snncuda)

if(SNNCUDA_HAS_CUDA)
    target_sources(snncuda PRIVATE backends/cuda/CudaBackend.cu)
    target_link_libraries(snncuda PUBLIC CUDA::cudart)
    target_compile_options(snncuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
    set_source_files_properties(backends/cuda/CudaBackend.cu PROPERTIES LANGUAGE CUDA)
endif()

if(SNNCUDA_HAS_HDF5)
    if(HDF5_PC_FOUND)
        target_include_directories(snncuda PUBLIC ${HDF5_PC_INCLUDE_DIRS})
        target_link_directories(snncuda PUBLIC ${HDF5_PC_LIBRARY_DIRS})
        target_link_libraries(snncuda PUBLIC ${HDF5_PC_LIBRARIES})
    else()
        target_include_directories(snncuda PUBLIC ${HDF5_INCLUDE_DIRS})
        target_link_libraries(snncuda PUBLIC ${HDF5_LIBRARIES})
    endif()
endif()

add_executable(snncuda_cli app/main.cpp)
target_link_libraries(snncuda_cli PRIVATE SNNCuda::snncuda)
snncuda_apply_common_options(snncuda_cli)

set_target_properties(snncuda_cli PROPERTIES
    OUTPUT_NAME snncuda
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

install(TARGETS snncuda snncuda_cli
    EXPORT SNNCudaTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
