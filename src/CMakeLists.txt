
if(RL_TOOLS_ENABLE_JSON AND RL_TOOLS_ENABLE_BOOST_BEAST)
find_package(Boost REQUIRED)
message(STATUS "Boost version: ${Boost_VERSION}")
add_executable(ui ui/ui.cpp)
target_link_libraries(
        ui
        PRIVATE
        rl_tools
        learning_to_fly
        Boost::boost
)
endif()

add_executable(training_headless training.cpp)
target_link_libraries(
        training_headless
        PRIVATE
        rl_tools
        learning_to_fly
)
add_executable(training_benchmark training.cpp)
target_link_libraries(
        training_benchmark
        PRIVATE
        rl_tools
        learning_to_fly
)
target_compile_definitions(training_benchmark PRIVATE LEARNING_TO_FLY_IN_SECONDS_BENCHMARK)

if(RL_TOOLS_ENABLE_HDF5)
add_executable(ablation_study ablation_study.cpp)
target_link_libraries(
        ablation_study
        PRIVATE
        rl_tools
        learning_to_fly
)
endif()

if(LEARNING_TO_FLY_ENABLE_OLD_UI)
add_executable(evaluate_actor evaluate_actor.cpp)
target_link_libraries(
        evaluate_actor
        PRIVATE
        rl_tools
        learning_to_fly
)
endif()


if(RL_TOOLS_BACKEND_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()
add_executable(training_mlp_matched training.cpp)
target_link_libraries(training_mlp_matched PRIVATE rl_tools learning_to_fly)
target_compile_definitions(training_mlp_matched PRIVATE LEARNING_TO_FLY_ACTOR_MATCHED_MLP)

add_executable(test_matched_mlp_actor ../experiments/rnn/test_matched_mlp_actor.cpp)
target_compile_features(test_matched_mlp_actor PRIVATE cxx_std_17)

add_executable(training_mlp training.cpp)
target_link_libraries(training_mlp PRIVATE rl_tools learning_to_fly)
target_compile_definitions(training_mlp PRIVATE LEARNING_TO_FLY_ACTOR_MLP)
foreach(actor IN ITEMS lstm gru)
    string(TOUPPER ${actor} actor_define)
    add_executable(training_${actor} training.cpp)
    target_link_libraries(training_${actor} PRIVATE rl_tools learning_to_fly)
    target_compile_definitions(training_${actor} PRIVATE LEARNING_TO_FLY_ACTOR_${actor_define})
endforeach()

add_executable(test_gated_actor ../experiments/rnn/test_gated_actor.cpp)
target_compile_features(test_gated_actor PRIVATE cxx_std_17)

add_executable(test_original_mlp_checkpoint test_original_mlp_checkpoint.cpp)
target_link_libraries(test_original_mlp_checkpoint PRIVATE rl_tools learning_to_fly)
target_compile_definitions(test_original_mlp_checkpoint PRIVATE LEARNING_TO_FLY_ACTOR_MLP)

add_executable(training_gru_history training.cpp)
target_link_libraries(training_gru_history PRIVATE rl_tools learning_to_fly)
target_compile_definitions(training_gru_history PRIVATE LEARNING_TO_FLY_ACTOR_GRU_HISTORY)

add_executable(test_action_history ../experiments/rnn/test_action_history.cpp)
target_link_libraries(test_action_history PRIVATE rl_tools learning_to_fly)

add_executable(training_gru_adam training.cpp)
target_link_libraries(training_gru_adam PRIVATE rl_tools learning_to_fly)
target_compile_definitions(training_gru_adam PRIVATE LEARNING_TO_FLY_ACTOR_GRU_HISTORY LEARNING_TO_FLY_ACTOR_GRU_ADAM)

add_executable(test_actor_adam ../experiments/rnn/test_actor_adam.cpp)
target_link_libraries(test_actor_adam PRIVATE rl_tools learning_to_fly)
target_compile_definitions(test_actor_adam PRIVATE LEARNING_TO_FLY_ACTOR_GRU_HISTORY LEARNING_TO_FLY_ACTOR_GRU_ADAM)

add_executable(training_gru_full training.cpp)
target_link_libraries(training_gru_full PRIVATE rl_tools learning_to_fly)
target_compile_definitions(training_gru_full PRIVATE
        LEARNING_TO_FLY_ACTOR_GRU_HISTORY LEARNING_TO_FLY_ACTOR_GRU_ADAM LEARNING_TO_FLY_ACTOR_GRU_FULL)

add_executable(test_full_observation ../experiments/rnn/test_full_observation.cpp)
target_link_libraries(test_full_observation PRIVATE rl_tools learning_to_fly)
target_compile_definitions(test_full_observation PRIVATE
        LEARNING_TO_FLY_ACTOR_GRU_HISTORY LEARNING_TO_FLY_ACTOR_GRU_ADAM LEARNING_TO_FLY_ACTOR_GRU_FULL)
