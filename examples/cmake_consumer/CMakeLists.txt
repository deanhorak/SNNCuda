cmake_minimum_required(VERSION 3.22)

project(SNNCudaConsumer LANGUAGES CXX)

find_package(SNNCuda CONFIG REQUIRED)

add_executable(snncuda_consumer main.cpp)
target_link_libraries(snncuda_consumer PRIVATE SNNCuda::snncuda)
