
#pragma once

#include "rnn_actor.h"
#include "sequence_sampler.h"

#include <tuple>
#include <array>
#include <cstddef>

namespace learning_to_fly::rnn {

// Experimental RNN Actor update using the existing TD3 Critic.
//
// This function:
// 1. Samples chronological observation sequences.
// 2. Computes RNN actions at the final time step.
// 3. Evaluates these actions using the existing MLP Critic.
// 4. Obtains dLoss/dAction from the Critic.
// 5. Updates the RNN using BPTT.
//
// IMPORTANT:
// The RNN's environment hidden state is preserved.

template <std::size_t SEQ_LEN = 8, typename TS, typename RNN>
bool train_rnn_actor_td3(
    TS& ts,
    RNN& rnn,
    float learning_rate = 0.0001f
) {
    namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;

    using CONFIG = typename TS::SPEC;
    using TI = typename CONFIG::TI;

    using RB_SPEC =
        typename decltype(
            ts.off_policy_runner
        )::REPLAY_BUFFER_SPEC;

    constexpr std::size_t CAPACITY =
        RB_SPEC::CAPACITY;

    constexpr std::size_t BATCH_SIZE =
        CONFIG::ACTOR_CRITIC_TYPE::SPEC::
            PARAMETERS::ACTOR_BATCH_SIZE;

    constexpr std::size_t ACTION_DIM =
        CONFIG::ENVIRONMENT::ACTION_DIM;

    constexpr std::size_t CRITIC_OBS_DIM =
        CONFIG::CRITIC_TYPE::INPUT_DIM - ACTION_DIM;

    static_assert(ACTION_DIM == 4);

    static_assert(
        std::tuple_size<typename RNN::Input>::value == 18,
        "RNN must accept 18 input features"
    );

    static_assert(
        std::tuple_size<typename RNN::Output>::value ==
            ACTION_DIM,
        "RNN output dimension must match action dimension"
    );

    static_assert(
        CONFIG::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS == 1,
        "This version supports one environment"
    );

    // --------------------------------------------
    // 1. Access the real Replay Buffer
    // --------------------------------------------

    const auto& rb =
        ts.off_policy_runner.replay_buffers[0];

    const std::size_t count =
        rb.full
            ? CAPACITY
            : static_cast<std::size_t>(rb.position);

    if (count < SEQ_LEN) {
        return false;
    }

    auto& buffers = ts.actor_training_buffers;

    using Sequence =
        std::array<typename RNN::Input, SEQ_LEN>;

    std::array<Sequence, BATCH_SIZE> sequences{};

    // --------------------------------------------
    // 2. Preserve environment hidden state
    // --------------------------------------------

    // This hidden state belongs to the actual
    // ongoing drone episode.
    //
    // BPTT training will temporarily reset and
    // modify rnn.hidden, so we must restore it.
    const auto environment_hidden = rnn.hidden;

    // --------------------------------------------
    // 3. Sample chronological sequences
    // --------------------------------------------

    for (std::size_t row = 0; row < BATCH_SIZE; ++row) {

        bool found = false;
        std::size_t end_index = 0;

        for (
            std::size_t attempt = 0;
            attempt < 64;
            ++attempt
        ) {

            const auto candidate =
                rlt::random::uniform_int_distribution(
                    typename CONFIG::DEVICE::SPEC::RANDOM(),
                    static_cast<TI>(SEQ_LEN - 1),
                    static_cast<TI>(count - 1),
                    ts.rng
                );

            if (
                extract_sequence<SEQ_LEN, CAPACITY>(
                    rb,
                    candidate,
                    sequences[row]
                )
            ) {
                end_index = candidate;
                found = true;
                break;
            }
        }

        // No valid sequence found.
        if (!found) {
            rnn.hidden = environment_hidden;
            return false;
        }

        // ----------------------------------------
        // 4. Find corresponding Critic observation
        // ----------------------------------------

        const std::size_t oldest =
            rb.full
                ? static_cast<std::size_t>(rb.position)
                : 0;

        const std::size_t physical_index =
            (oldest + end_index) % CAPACITY;

        for (
            std::size_t j = 0;
            j < CRITIC_OBS_DIM;
            ++j
        ) {

            rlt::set(
                buffers.observations,
                row,
                j,
                rlt::get(
                    rb.observations_privileged,
                    physical_index,
                    j
                )
            );
        }

        // ----------------------------------------
        // 5. Compute RNN action
        // ----------------------------------------

        // Training sequences are independent.
        // Each begins with zero hidden state.
        rnn.reset();

        typename RNN::Output action{};

        for (const auto& observation : sequences[row]) {
            action = rnn.forward(observation);
        }

        for (
            std::size_t j = 0;
            j < ACTION_DIM;
            ++j
        ) {

            rlt::set(
                buffers.actions,
                row,
                j,
                action[j]
            );
        }
    }

    // --------------------------------------------
    // 6. Evaluate RNN actions using MLP Critic
    // --------------------------------------------

    auto& critic =
        ts.actor_critic.critic_1;

    rlt::forward(
        ts.device,
        critic,
        buffers.state_action_value_input,
        buffers.state_action_value
    );

    // --------------------------------------------
    // 7. Compute TD3 Actor loss gradient
    // --------------------------------------------

    // Actor loss:
    //
    // L = -mean(Q(s, RNN(s)))
    //
    // dL/dQ = -1 / BATCH_SIZE

    rlt::set_all(
        ts.device,
        buffers.d_output,
        static_cast<typename CONFIG::T>(
            -1.0 / BATCH_SIZE
        )
    );

    rlt::backward_input(
        ts.device,
        critic,
        buffers.d_output,
        buffers.d_critic_input,
        ts.critic_buffers[0]
    );

    // --------------------------------------------
    // 8. Backpropagate Critic gradients into RNN
    // --------------------------------------------

    typename RNN::Gradients gradients{};
    for (
        std::size_t row = 0;
        row < BATCH_SIZE;
        ++row
    ) {

        typename RNN::Output action_gradient{};

        for (
            std::size_t j = 0;
            j < ACTION_DIM;
            ++j
        ) {

            action_gradient[j] =
                static_cast<float>(
                    rlt::get(
                        buffers.d_critic_input,
                        row,
                        CRITIC_OBS_DIM + j
                    )
                );
        }

        rnn.accumulate_sequence_gradient(
            sequences[row],
            action_gradient,
            gradients
        );
    }

    rnn.apply_gradients(gradients, learning_rate);

    // --------------------------------------------
    // 9. Restore environment hidden state
    // --------------------------------------------

    // Training must not replace the ongoing
    // episode's recurrent memory.
    rnn.hidden = environment_hidden;

    return true;
}

} // namespace learning_to_fly::rnn
