add_executable(simple_spike simple_spike.cpp)
target_link_libraries(simple_spike PRIVATE SNNCuda::snncuda)
snncuda_apply_common_options(simple_spike)

