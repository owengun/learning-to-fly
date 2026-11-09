
#pragma once

#include "rnn_actor.h"
#include "observation_adapter.h"
#ifdef LEARNING_TO_FLY_ACTOR_GRU_HISTORY
#include "action_history_sampler.h"
#endif

namespace learning_to_fly::rnn {

// Replace the MLP policy during environment interaction.
// The Critic architecture is not changed.
template <typename TS, typename RNN>
void rnn_environment_step(TS& ts, RNN& actor) {
    namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;

    auto& runner = ts.off_policy_runner;

    // Before a new episode, discard the previous hidden state.
    // Initial runner.truncated is also true.
    const bool episode_start = rlt::get(runner.truncated, 0, 0);
    if (episode_start) {
        actor.reset();
    }

    // Prepare environment and obtain current observations.
    rlt::rl::components::off_policy_runner::prologue(
        ts.device, runner, ts.rng
    );

    // One environment, as in the current project configuration.
#ifdef LEARNING_TO_FLY_ACTOR_GRU_HISTORY
    // Epilogue left the previous action after exploration and clipping here.
    auto input = history_input(runner.buffers.observations, 0, runner.buffers.actions, 0,
                               episode_start);
#else
    auto input = extract_rltools_observation(runner.buffers.observations);
#endif

    // RNN generates four motor actions.
#ifdef LEARNING_TO_FLY_ACTOR_GRU_ADAM
    require_finite(input, "Non-finite online GRU observation");
#endif
    auto action = actor.forward(input);
#ifdef LEARNING_TO_FLY_ACTOR_GRU_ADAM
    require_finite(actor.hidden.h, "Non-finite online GRU hidden state");
    require_finite(action, "Non-finite online GRU action");
#endif

    for (std::size_t i = 0; i < 4; ++i) {
        rlt::set(
            runner.buffers.actions,
            0,
            i,
            action[i]
        );
    }

    // Apply the RNN actions to the real simulator,
    // storing transitions in the existing Replay Buffer.
    rlt::rl::components::off_policy_runner::epilogue(
        ts.device, runner, ts.rng
    );
#ifdef LEARNING_TO_FLY_ACTOR_GRU_ADAM
    const auto& rb = runner.replay_buffers[0];
    const auto row = (rb.position + rb.CAPACITY - 1) % rb.CAPACITY;
    const auto& next = rlt::get(rb.next_states, row, 0);
    require_finite(next.position, "Non-finite simulator position");
    require_finite(next.orientation, "Non-finite simulator orientation");
    require_finite(next.linear_velocity, "Non-finite simulator velocity");
    require_finite(next.angular_velocity, "Non-finite simulator angular velocity");
    require_finite(next.rpm, "Non-finite simulator rotor speed");
    if (!std::isfinite(rlt::get(rb.rewards, row, 0)))
        throw std::runtime_error("Non-finite replay reward");
#endif
}

} // namespace learning_to_fly::rnn
