
#include "rnn_actor.h"
#include <array>
#include <cmath>
#include <iostream>
using learning_to_fly::rnn::RNNActor;

int main() {
    RNNActor<18, 32, 4> actor;
    actor.initialize(42);

    constexpr std::size_t SEQ_LEN = 4;
    std::array<RNNActor<18, 32, 4>::Input, SEQ_LEN> inputs{};

    inputs[0][0] = 1.0f;
    inputs[1][1] = 0.5f;
    inputs[2][2] = -0.3f;
    inputs[3][3] = 0.2f;

    // Action before the update.
    actor.reset();
    std::array<float, 4> before{};
    for (const auto& input : inputs) {
        before = actor.forward(input);
    }

    // dLoss/dAction: increase the loss when action[0] increases.
    // Gradient descent should therefore decrease action[0].
    RNNActor<18, 32, 4>::Output gradient{};
    gradient[0] = 1.0f;

    auto second_inputs = inputs;
    second_inputs[0][0] = -0.5f;
    auto second_gradient = gradient;
    second_gradient[0] = -0.25f;
    second_gradient[1] = 0.5f;

    auto batch_actor = actor;
    decltype(actor)::Gradients batch_gradients{};
    batch_actor.accumulate_sequence_gradient(inputs, gradient, batch_gradients);
    batch_actor.accumulate_sequence_gradient(
        second_inputs, second_gradient, batch_gradients
    );
    if (batch_actor.W_x != actor.W_x || batch_actor.W_h != actor.W_h ||
        batch_actor.W_out != actor.W_out || batch_actor.b_h != actor.b_h ||
        batch_actor.b_out != actor.b_out) {
        std::cerr << "Accumulation changed weights before the batch update\n";
        return 1;
    }

    auto loss = [&](auto model) {
        auto sequence_loss = [&](const auto& sequence, const auto& grad) {
            model.reset();
            decltype(actor)::Output action{};
            for (const auto& input : sequence) {
                action = model.forward(input);
            }
            float value = 0;
            for (std::size_t i = 0; i < action.size(); ++i) {
                value += action[i] * grad[i];
            }
            return value;
        };
        return sequence_loss(inputs, gradient) +
               sequence_loss(second_inputs, second_gradient);
    };

    constexpr float epsilon = 0.001f;
    auto plus = actor;
    auto minus = actor;
    plus.W_h[0][0] += epsilon;
    minus.W_h[0][0] -= epsilon;
    const float expected = (loss(plus) - loss(minus)) / (2 * epsilon);
    batch_actor.apply_gradients(batch_gradients, 0.1f);
    const float actual = (actor.W_h[0][0] - batch_actor.W_h[0][0]) / 0.1f;
    if (!std::isfinite(actual) || std::abs(expected - actual) > 0.00005f) {
        std::cerr << "Batch recurrent gradient disagrees with finite differences\n";
        return 1;
    }

    actor.train_sequence_with_gradient(
        inputs, gradient, 0.001f
    );

    // Evaluate the same sequence after the update.
    actor.reset();
    std::array<float, 4> after{};
    for (const auto& input : inputs) {
        after = actor.forward(input);
    }

    std::cout << "Before: " << before[0] << '\n';
    std::cout << "After:  " << after[0] << '\n';

    if (!std::isfinite(after[0]) ||
        !(after[0] < before[0])) {
        std::cerr << "External gradient test FAILED\n";
        return 1;
    }

    std::cout << "External gradient test PASSED\n";
    return 0;
}

