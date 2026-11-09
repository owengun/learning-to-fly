#pragma once

#include "action_history_sampler.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace learning_to_fly::rnn {

template <typename TS>
bool gather_history_critic(TS& ts) {
    namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;
    using CONFIG = typename TS::SPEC;
    constexpr std::size_t batch_size = CONFIG::TD3_PARAMETERS::CRITIC_BATCH_SIZE;
    const auto& rb = ts.off_policy_runner.replay_buffers[0];
    const auto& actor = ts.rnn_actor_target;
    HistoryWindow window;
    for (std::size_t row = 0; row < batch_size;) {
        if (!sample_history_window(ts.device, rb, std::min(HISTORY_LEARN, batch_size - row),
                                   ts.rng, window)) return false;
        auto state = burn_in(actor, window);
        for (std::size_t t = 0; t < window.learn; ++t, ++row) {
            const auto physical = window.transitions[t];
            copy_replay_transition(ts.device, rb, ts.critic_batch, physical, row);
            state = actor.advance(window.inputs[window.burn + t], state).state;
            const auto action = next_history_action(actor, rb, physical, state);
            for (std::size_t j = 0; j < action.size(); ++j)
                rlt::set(ts.critic_training_buffers.next_actions, row, j, action[j]);
        }
    }
    return true;
}

template <typename TS>
bool train_history_actor(TS& ts, float learning_rate) {
    namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;
    using CONFIG = typename TS::SPEC;
    using Actor = std::remove_reference_t<decltype(ts.rnn_actor)>;
    constexpr std::size_t batch_size = CONFIG::TD3_PARAMETERS::ACTOR_BATCH_SIZE;
    constexpr std::size_t action_dim = CONFIG::ENVIRONMENT::ACTION_DIM;
    constexpr std::size_t obs_dim = CONFIG::CRITIC_TYPE::INPUT_DIM - action_dim;
    auto& actor = ts.rnn_actor;
    const auto& rb = ts.off_policy_runner.replay_buffers[0];
    auto& buffers = ts.actor_training_buffers;
    struct Sample {
        HistoryWindow window;
        typename Actor::State initial;
    };
    std::vector<Sample> samples;
    samples.reserve(batch_size / HISTORY_LEARN + 1);
    std::size_t burn_steps = 0;
    for (std::size_t row = 0; row < batch_size;) {
        auto& sample = samples.emplace_back();
        auto& window = sample.window;
        if (!sample_history_window(ts.device, rb, std::min(HISTORY_LEARN, batch_size - row),
                                   ts.rng, window)) return false;
        sample.initial = burn_in(actor, window);
        burn_steps += window.burn;
        auto state = sample.initial;
        for (std::size_t t = 0; t < window.learn; ++t, ++row) {
            const auto physical = window.transitions[t];
            for (std::size_t j = 0; j < obs_dim; ++j)
                rlt::set(buffers.observations, row, j, rlt::get(rb.observations_privileged, physical, j));
            state = actor.advance(window.inputs[window.burn + t], state).state;
            const auto action = actor.action(state.h);
            for (std::size_t j = 0; j < action_dim; ++j)
                rlt::set(buffers.actions, row, j, action[j]);
        }
    }

    auto& critic = ts.actor_critic.critic_1;
    rlt::forward(ts.device, critic, buffers.state_action_value_input, buffers.state_action_value);
    rlt::set_all(ts.device, buffers.d_output, static_cast<typename CONFIG::T>(-1.0 / batch_size));
    rlt::backward_input(ts.device, critic, buffers.d_output, buffers.d_critic_input,
                        ts.critic_buffers[0]);

    typename Actor::Gradients gradients{};
    std::size_t row = 0;
    for (const auto& sample : samples) {
        const auto& window = sample.window;
        std::array<typename Actor::Output, HISTORY_LEARN> upstream{};
        for (std::size_t t = 0; t < window.learn; ++t, ++row)
            for (std::size_t j = 0; j < action_dim; ++j)
                upstream[t][j] = rlt::get(buffers.d_critic_input, row, obs_dim + j);
        actor.accumulate_window_gradient(window.inputs.data() + window.burn, window.learn,
                                         sample.initial, upstream, gradients);
    }
    const double grad_norm = actor.gradient_norm(gradients);
    if (!std::isfinite(grad_norm)) throw std::runtime_error("Non-finite GRU actor gradient");
    if (ts.step % 1000 == 0 || ts.step == CONFIG::N_WARMUP_STEPS_ACTOR +
                                         CONFIG::TD3_PARAMETERS::ACTOR_TRAINING_INTERVAL) {
        double q_sum = 0, q_abs_max = 0;
        for (std::size_t i = 0; i < batch_size; ++i) {
            const double q = rlt::get(buffers.state_action_value, i, 0);
            q_sum += q;
            q_abs_max = std::max(q_abs_max, std::abs(q));
        }
        std::cout << "[GRU Diagnostics] step,q_mean,q_abs_max,grad_l2,mean_burn "
                  << ts.step << ',' << q_sum / batch_size << ',' << q_abs_max << ','
                  << grad_norm << ',' << static_cast<double>(burn_steps) / samples.size() << '\n';
    }
    actor.apply_gradients(gradients, learning_rate);
    return true;
}

} // namespace learning_to_fly::rnn
