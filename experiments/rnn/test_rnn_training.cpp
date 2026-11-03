
#include "rnn_actor.h"
#include <array>
#include <iostream>
#include <random>

int main() {
    using Actor = learning_to_fly::rnn::RNNActor<18, 32, 4>;

    Actor actor;
    actor.initialize(42);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    constexpr int SEQ_LEN = 4;
    constexpr int EPOCHS = 20000;

    float first_loss = 0.0f;
    float last_loss = 0.0f;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::array<Actor::Input, SEQ_LEN> inputs{};

        // Only the first time step contains the signal.
        float signal = dist(rng);
        inputs[0][0] = signal;

        // The remaining time steps contain no signal.
        Actor::Output target{};
        target[0] = signal;

        float loss = actor.train_sequence(inputs, target, 0.005f);

        if (epoch == 0)
            first_loss = loss;

        if (epoch == EPOCHS - 1)
            last_loss = loss;

        if (epoch % 500 == 0)
            std::cout << "Epoch " << epoch
                      << " Loss: " << loss << "\n";
    }

    std::cout << "First loss: " << first_loss << "\n";
    std::cout << "Last loss: " << last_loss << "\n";

    // Test whether the trained RNN remembers an earlier signal.
    actor.reset();

    Actor::Input input{};
    input[0] = 0.7f;

    actor.forward(input);

    input.fill(0.0f);

    actor.forward(input);
    actor.forward(input);
    auto result = actor.forward(input);

    std::cout << "Expected output: 0.7\n";
    std::cout << "Predicted output: " << result[0] << "\n";

    return 0;
}
