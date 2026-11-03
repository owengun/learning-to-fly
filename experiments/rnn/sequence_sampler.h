
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

template <
    std::size_t SEQ_LEN,
    std::size_t CAPACITY,
    typename DEVICE,
    typename REPLAY_BUFFER,
    typename BATCH,
    typename RNG
>
bool gather_aligned_sequence_batch(
    DEVICE& device,
    const REPLAY_BUFFER& rb,
    BATCH& batch,
    std::array<
        std::array<std::array<float, 18>, SEQ_LEN>,
        BATCH::BATCH_SIZE
    >& current_sequences,
    std::array<
        std::array<std::array<float, 18>, SEQ_LEN>,
        BATCH::BATCH_SIZE
    >& next_sequences,
    RNG& rng
) {
    namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;

    const std::size_t count =
        rb.full ? CAPACITY : static_cast<std::size_t>(rb.position);

    if (count < SEQ_LEN) {
        return false;
    }

    constexpr std::size_t BATCH_SIZE = BATCH::BATCH_SIZE;

    for (std::size_t row_i = 0; row_i < BATCH_SIZE; ++row_i) {
        bool found = false;
        std::size_t end_index = 0;

        for (std::size_t attempt = 0; attempt < 128; ++attempt) {
            auto candidate = rlt::random::uniform_int_distribution(
                typename DEVICE::SPEC::RANDOM(),
                static_cast<typename DEVICE::index_t>(SEQ_LEN - 1),
                static_cast<typename DEVICE::index_t>(count - 1),
                rng
            );

            if (extract_sequence<SEQ_LEN, CAPACITY>(
                    rb, candidate, current_sequences[row_i])) {
                end_index = candidate;
                found = true;
                break;
            }
        }

        if (!found) {
            return false;
        }

        const std::size_t oldest =
            rb.full ? static_cast<std::size_t>(rb.position) : 0;

        const std::size_t physical =
            (oldest + end_index) % CAPACITY;

        // Next sequence = shift current sequence,
        // then append next observation of this transition.
        for (std::size_t t = 0; t + 1 < SEQ_LEN; ++t) {
            next_sequences[row_i][t] =
                current_sequences[row_i][t + 1];
        }

        for (std::size_t j = 0; j < 18; ++j) {
            next_sequences[row_i][SEQ_LEN - 1][j] =
                static_cast<float>(
                    rlt::get(rb.next_observations, physical, j)
                );
        }

        // Transfer the exact same transition into TD3 batch.
        auto obs_source =
            rlt::row(device, rb.observations, physical);
        auto obs_target =
            rlt::row(device, batch.observations, row_i);
        rlt::copy(device, device, obs_source, obs_target);

        auto priv_source =
            rlt::row(device, rb.observations_privileged, physical);
        auto priv_target =
            rlt::row(device, batch.observations_privileged, row_i);
        rlt::copy(device, device, priv_source, priv_target);

        auto action_source =
            rlt::row(device, rb.actions, physical);
        auto action_target =
            rlt::row(device, batch.actions, row_i);
        rlt::copy(device, device, action_source, action_target);

        auto next_source =
            rlt::row(device, rb.next_observations, physical);
        auto next_target =
            rlt::row(device, batch.next_observations, row_i);
        rlt::copy(device, device, next_source, next_target);

        auto next_priv_source =
            rlt::row(device, rb.next_observations_privileged, physical);
        auto next_priv_target =
            rlt::row(device, batch.next_observations_privileged, row_i);
        rlt::copy(device, device, next_priv_source, next_priv_target);

        rlt::set(
            batch.rewards, 0, row_i,
            rlt::get(rb.rewards, physical, 0)
        );
        rlt::set(
            batch.terminated, 0, row_i,
            rlt::get(rb.terminated, physical, 0)
        );
        rlt::set(
            batch.truncated, 0, row_i,
            rlt::get(rb.truncated, physical, 0)
        );
    }

    return true;
}

} // namespace learning_to_fly::rnn
