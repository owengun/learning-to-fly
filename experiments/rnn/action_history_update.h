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
    using Actor = std::remove_cv_t<std::remove_reference_t<decltype(actor)>>;
    HistoryWindowFor<typename Actor::Input> window;
    for (std::size_t row = 0; row < batch_size;) {
        if (!sample_history_window(ts.device, rb, std::min(HISTORY_LEARN, batch_size - row),
                                   ts.rng, window)) return false;
        auto state = burn_in(actor, window);
        for (std::size_t t = 0; t < window.learn; ++t, ++row) {
            const auto physical = window.transitions[t];
            copy_replay_transition(ts.device, rb, ts.critic_batch, physical, row);
            state = actor.advance(window.inputs[window.burn + t], state).state;
            const auto action = next_history_action(actor, rb, physical, state);
#ifdef LEARNING_TO_FLY_ACTOR_SEQUENCE_ADAM
            require_finite(action, "Non-finite target GRU action");
#endif
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
        HistoryWindowFor<typename Actor::Input> window;
        typename Actor::State initial;
    };
    std::vector<Sample> samples;
    samples.reserve(batch_size / HISTORY_LEARN + 1);
    std::size_t burn_steps = 0;
#ifdef LEARNING_TO_FLY_ACTOR_SEQUENCE_ADAM
    const auto& dynamics = ts.off_policy_runner.envs[0].parameters.dynamics;
    double update_gate_sum = 0, reset_gate_sum = 0;
    double action_sum = 0, applied_sum = 0, thrust_sum = 0;
#ifdef LEARNING_TO_FLY_ACTOR_MATCHED_RECURRENT
    std::array<double, 4> gate_sum{};
    double hidden_squared = 0, cell_squared = 0, hidden_saturated = 0;
#endif
#endif
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
            const auto frame = actor.advance(window.inputs[window.burn + t], state);
            state = frame.state;
            const auto action = actor.action(state.h);
#ifdef LEARNING_TO_FLY_ACTOR_SEQUENCE_ADAM
            require_finite(action, "Non-finite replay GRU action");
            for (const auto& gate : frame.gate) require_finite(gate, "Non-finite recurrent gate");
            require_finite(state.h, "Non-finite replay hidden state");
            require_finite(state.cell, "Non-finite replay cell state");
            for (double value : frame.gate[0]) update_gate_sum += value;
            if constexpr (Actor::gate_count > 1)
                for (double value : frame.gate[1]) reset_gate_sum += value;
#ifdef LEARNING_TO_FLY_ACTOR_MATCHED_RECURRENT
            for (std::size_t k = 0; k < Actor::gate_count; ++k)
                for (double value : frame.gate[k]) gate_sum[k] += value;
            for (double value : state.h) {
                hidden_squared += value * value;
                hidden_saturated += std::abs(value) > 0.95;
            }
            for (double value : state.cell) cell_squared += value * value;
#endif
            const auto& next = rlt::get(rb.next_states, physical, 0);
            double thrust = 0;
            for (std::size_t j = 0; j < action_dim; ++j) {
                action_sum += action[j];
                applied_sum += rlt::get(rb.actions, physical, j);
                const double rpm = next.rpm[j];
                thrust += dynamics.thrust_constants[0] + dynamics.thrust_constants[1] * rpm +
                          dynamics.thrust_constants[2] * rpm * rpm;
            }
            const double qx = next.orientation[1], qy = next.orientation[2];
            thrust_sum += thrust * (1 - 2 * (qx * qx + qy * qy)) /
                          (dynamics.mass * std::abs(dynamics.gravity[2]));
#endif
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
    const bool log_update = ts.step % 1000 == 0 || ts.step == CONFIG::N_WARMUP_STEPS_ACTOR +
                                                 CONFIG::TD3_PARAMETERS::ACTOR_TRAINING_INTERVAL;
#ifdef LEARNING_TO_FLY_ACTOR_SEQUENCE_ADAM
    require_finite(std::array{update_gate_sum, reset_gate_sum, action_sum, applied_sum, thrust_sum},
                   "Non-finite GRU replay diagnostics");
#else
    if (!std::isfinite(grad_norm)) throw std::runtime_error("Non-finite GRU actor gradient");
    if (log_update) {
#endif
        double q_sum = 0, q_abs_max = 0;
        for (std::size_t i = 0; i < batch_size; ++i) {
            const double q = rlt::get(buffers.state_action_value, i, 0);
#ifdef LEARNING_TO_FLY_ACTOR_SEQUENCE_ADAM
            if (!std::isfinite(q)) throw std::runtime_error("Non-finite GRU actor Q");
#endif
            q_sum += q;
            q_abs_max = std::max(q_abs_max, std::abs(q));
        }
#ifdef LEARNING_TO_FLY_ACTOR_SEQUENCE_ADAM
    const auto update = ts.history_adam.apply(actor, gradients);
    if (log_update) {
        constexpr auto gate_values = batch_size * std::tuple_size_v<typename Actor::Hidden>;
#ifdef LEARNING_TO_FLY_ACTOR_MATCHED_RECURRENT
        std::cout << "[Recurrent Adam Diagnostics] " << ts.step << ',' << ts.history_adam.updates
                  << ',' << q_sum / batch_size << ',' << q_abs_max << ',' << grad_norm
                  << ',' << update.update_l2 << ',' << update.parameter_l2;
        for (double value : gate_sum) std::cout << ',' << value / gate_values;
        std::cout << ',' << std::sqrt(hidden_squared / batch_size)
                  << ',' << std::sqrt(cell_squared / batch_size) << ',' << hidden_saturated / gate_values;
#else
        std::cout << "[GRU Adam Diagnostics] " << ts.step << ',' << ts.history_adam.updates
                  << ',' << q_sum / batch_size << ',' << q_abs_max << ',' << grad_norm
                  << ',' << update.update_l2 << ',' << update.parameter_l2
                  << ',' << update_gate_sum / gate_values << ',' << reset_gate_sum / gate_values;
#endif
        std::cout << ',' << action_sum / (batch_size * action_dim)
                  << ',' << applied_sum / (batch_size * action_dim) << ',' << thrust_sum / batch_size
                  << ',' << static_cast<double>(burn_steps) / samples.size();
#ifdef LEARNING_TO_FLY_ACTOR_SECOND_CURRICULUM_LR
        std::cout << ',' << ts.history_adam.alpha;
#endif
        std::cout << '\n';
    }
#else
        std::cout << "[GRU Diagnostics] step,q_mean,q_abs_max,grad_l2,mean_burn "
                  << ts.step << ',' << q_sum / batch_size << ',' << q_abs_max << ','
                  << grad_norm << ',' << static_cast<double>(burn_steps) / samples.size() << '\n';
    }
    actor.apply_gradients(gradients, learning_rate);
#endif
    return true;
}

} // namespace learning_to_fly::rnn
