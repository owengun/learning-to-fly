
#pragma once

#include <array>
#include <cstddef>

#include <rl_tools/operations/cpu_mux.h>

namespace learning_to_fly::rnn {

    // Extract one row of dLoss/dAction from an RLtools matrix.
    // The input must already contain the actor-loss gradient,
    // including TD3's negative sign and batch normalization.
    template <typename MATRIX, std::size_t ACTION_DIM = 4>
    std::array<float, ACTION_DIM>
    extract_action_gradient(
        const MATRIX& gradients,
        std::size_t batch_index
    ) {
        std::array<float, ACTION_DIM> result{};

        for (std::size_t i = 0; i < ACTION_DIM; ++i) {
            result[i] = static_cast<float>(
                RL_TOOLS_NAMESPACE_WRAPPER::rl_tools::get(
                    gradients, batch_index, i
                )
            );
        }

        return result;
    }

}
