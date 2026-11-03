
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

