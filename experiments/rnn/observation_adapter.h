
#ifndef LEARNING_TO_FLY_RNN_OBSERVATION_ADAPTER_H
#define LEARNING_TO_FLY_RNN_OBSERVATION_ADAPTER_H

#include "rnn_actor.h"
#include <array>
#include <cstddef>
#include <rl_tools/operations/cpu_mux.h>

namespace learning_to_fly::rnn {

template <std::size_t OBS_DIM>
RNNActor<18, 32, 4>::Input
extract_observation(
    const std::array<float, OBS_DIM>& observation
) {
    static_assert(OBS_DIM >= 18,
                  "Observation must contain at least 18 values");

    RNNActor<18, 32, 4>::Input input{};

    for (std::size_t i = 0; i < 18; ++i) {
        input[i] = observation[i];
    }

    return input;
}
// Convert an RLtools observation matrix to RNN input.
// MATRIX is expected to have at least 18 columns.
template <typename MATRIX>
RNNActor<18, 32, 4>::Input
extract_rltools_observation(const MATRIX& observation) {
    RNNActor<18, 32, 4>::Input input{};

    for (std::size_t i = 0; i < 18; ++i) {
        input[i] = rl_tools::get(observation, 0, i);
    }

    return input;
}

} // namespace learning_to_fly::rnn

#endif
