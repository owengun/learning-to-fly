
#pragma once

#include <rl_tools/operations/cpu_mux.h>
#include <array>
#include <cstddef>

namespace learning_to_fly::rnn {

template <std::size_t SEQ_LEN,
          std::size_t CAPACITY,
          typename REPLAY_BUFFER>
bool extract_sequence(
    const REPLAY_BUFFER& rb,
    std::size_t end_index,
    std::array<std::array<float, 18>, SEQ_LEN>& result
) {
    static_assert(SEQ_LEN > 0);
    static_assert(CAPACITY >= SEQ_LEN);

    namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;

    // Number of valid transitions.
    const std::size_t count =
        rb.full ? CAPACITY : static_cast<std::size_t>(rb.position);

    if (count < SEQ_LEN || end_index >= count ||
        end_index < SEQ_LEN - 1) {
        return false;
    }

    // In a full circular buffer, position points to
    // the next write slot, which is also the oldest slot.
    const std::size_t oldest =
        rb.full ? static_cast<std::size_t>(rb.position) : 0;

    const std::size_t first = end_index - (SEQ_LEN - 1);

    // Reject sequences that cross an episode boundary.
    // A terminal flag on the final transition is allowed.
    for (std::size_t t = 0; t + 1 < SEQ_LEN; ++t) {
        const std::size_t index =
            (oldest + first + t) % CAPACITY;

        if (rlt::get(rb.terminated, index, 0) ||
            rlt::get(rb.truncated, index, 0)) {
            return false;
        }
    }

    // Read observations in chronological order.
    for (std::size_t t = 0; t < SEQ_LEN; ++t) {
        const std::size_t index =
            (oldest + first + t) % CAPACITY;

        for (std::size_t j = 0; j < 18; ++j) {
            result[t][j] = static_cast<float>(
                rlt::get(rb.observations, index, j)
            );
        }
    }

    return true;
}

} // namespace learning_to_fly::rnn
