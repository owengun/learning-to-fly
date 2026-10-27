cmake_minimum_required(VERSION 3.16)
project(learning_to_fly_adaptive_curriculum_v5)

set(CMAKE_CXX_STANDARD 17)

# Checkpoints and held-out evaluation load HDF5 actor files, so a build without
# HDF5 would produce an unusable final package. Fail at configuration time with
# a clear dependency error instead of failing later during compilation.
find_package(HDF5 REQUIRED)
set(RL_TOOLS_ENABLE_HDF5 ON)

find_package(Protobuf)
if(Protobuf_FOUND)
    set(RL_TOOLS_ENABLE_TENSORBOARD ON)
endif()

set(RL_TOOLS_ENABLE_JSON OFF)
set(RL_TOOLS_ENABLE_BOOST_BEAST OFF)

add_library(learning_to_fly_experiment INTERFACE)
target_include_directories(
    learning_to_fly_experiment
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/rl_tools
    ${CMAKE_CURRENT_BINARY_DIR}/rl_tools
    EXCLUDE_FROM_ALL
)

add_executable(adaptive_curriculum_v5_training adaptive_curriculum_v5.cpp)
add_executable(adaptive_curriculum_v5_force_only ablation_force_only.cpp)
add_executable(adaptive_curriculum_v5_mass_only ablation_mass_only.cpp)
add_executable(adaptive_curriculum_v5_force_mass_fixed_reward ablation_force_mass_fixed_reward.cpp)
add_executable(heldout_stress_evaluation heldout_stress_evaluation.cpp)
foreach(target
    adaptive_curriculum_v5_training
    adaptive_curriculum_v5_force_only
    adaptive_curriculum_v5_mass_only
    adaptive_curriculum_v5_force_mass_fixed_reward
)
    target_link_libraries(${target} PRIVATE rl_tools learning_to_fly_experiment)
endforeach()
target_link_libraries(
    heldout_stress_evaluation
    PRIVATE
    rl_tools
    learning_to_fly_experiment
)
