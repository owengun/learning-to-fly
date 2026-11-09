
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
    auto action = actor.forward(input);

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
}

} // namespace learning_to_fly::rnn
