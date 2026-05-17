cmake_minimum_required(VERSION 3.22)

project(SNNCuda
    VERSION 0.1.0
    DESCRIPTION "A modern C++ spiking neural network framework with optional CUDA acceleration"
    LANGUAGES CXX)

option(SNNCUDA_BUILD_EXAMPLES "Build example programs" ON)
option(SNNCUDA_BUILD_TESTS "Build tests" ON)
option(SNNCUDA_ENABLE_CUDA "Enable CUDA acceleration when a CUDA toolkit is available" ON)
option(SNNCUDA_ENABLE_HDF5 "Enable SONATA HDF5 node/edge loading when HDF5 is available" ON)
option(SNNCUDA_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(SNNCudaCompilerOptions)

if(SNNCUDA_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit QUIET)
        set(SNNCUDA_HAS_CUDA ON)
        message(STATUS "CUDA support enabled")
    else()
        set(SNNCUDA_HAS_CUDA OFF)
        message(STATUS "CUDA toolkit not found; building CPU backend only")
    endif()
else()
    set(SNNCUDA_HAS_CUDA OFF)
    message(STATUS "CUDA support disabled")
endif()

set(SNNCUDA_HAS_HDF5 OFF)
if(SNNCUDA_ENABLE_HDF5)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(HDF5_PC QUIET hdf5)
    endif()

    if(HDF5_PC_FOUND)
        set(SNNCUDA_HAS_HDF5 ON)
        message(STATUS "HDF5 support enabled via pkg-config")
    else()
        find_package(HDF5 COMPONENTS C QUIET)
        if(HDF5_FOUND)
            set(SNNCUDA_HAS_HDF5 ON)
            message(STATUS "HDF5 support enabled via CMake")
        else()
            message(STATUS "HDF5 not found; SONATA HDF5 node/edge loading disabled")
        endif()
    endif()
else()
    message(STATUS "HDF5 support disabled")
endif()

add_subdirectory(src)

if(SNNCUDA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(SNNCUDA_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()

install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
