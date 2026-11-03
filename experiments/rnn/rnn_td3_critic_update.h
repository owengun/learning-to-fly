
#pragma once

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>

// Critic training with target actions already supplied
// by an external RNN target actor.
//
// The two critics remain the original MLP models.

namespace learning_to_fly::rnn {

template <
    typename DEVICE,
    typename ACTOR_CRITIC,
    typename CRITIC,
    typename BATCH,
    typename OPTIMIZER,
    typename CRITIC_BUFFERS,
    typename TRAINING_BUFFERS
>
void train_critic_with_rnn_targets(
    DEVICE& device,
    ACTOR_CRITIC& actor_critic,
    CRITIC& critic,
    BATCH& batch,
    OPTIMIZER& optimizer,
    CRITIC_BUFFERS& critic_buffers,
    TRAINING_BUFFERS& training_buffers
) {
    namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;

    // The caller must first populate:
    // training_buffers.next_actions
    // using the TARGET RNN on correctly aligned
    // next-observation sequences.

    rlt::zero_gradient(device, critic);

    // Same TD3 target-policy smoothing as baseline.
    rlt::noisy_next_actions(device, training_buffers);

    // The critic still receives privileged observations.
    rlt::copy(
        device,
        device,
        batch.next_observations_privileged,
        training_buffers.next_observations
    );

    // First target MLP Critic.
    rlt::evaluate(
        device,
        actor_critic.critic_target_1,
        training_buffers.next_state_action_value_input,
        training_buffers.next_state_action_value_critic_1,
        critic_buffers
    );

    // Second target MLP Critic.
    rlt::evaluate(
        device,
        actor_critic.critic_target_2,
        training_buffers.next_state_action_value_input,
        training_buffers.next_state_action_value_critic_2,
        critic_buffers
    );

    // Original TD3 target calculation.
    rlt::target_actions(
        device,
        actor_critic,
        batch,
        training_buffers
    );

    // Train the original MLP Critic.
    rlt::forward(
        device,
        critic,
        batch.observations_and_actions
    );

    rlt::nn::loss_functions::mse::gradient(
        device,
        rlt::output(critic),
        training_buffers.target_action_value,
        training_buffers.d_output
    );

    rlt::backward(
        device,
        critic,
        batch.observations_and_actions,
        training_buffers.d_output,
        critic_buffers
    );

    rlt::step(
        device,
        optimizer,
        critic
    );
}

} // namespace learning_to_fly::rnn
