
#include "observation_adapter.h"
#include <array>
#include <cmath>
#include <iostream>

int main() {
    // Simulate the 146-dimensional observation format.
    std::array<float, 146> observation{};

    for (std::size_t i = 0; i < 146; ++i) {
        observation[i] = static_cast<float>(i);
    }

    // Extract the first 18 dimensions.
    auto input = learning_to_fly::rnn::extract_observation(
        observation
    );

    bool correct = true;

    for (std::size_t i = 0; i < 18; ++i) {
        if (input[i] != static_cast<float>(i)) {
            correct = false;
        }
    }

    std::cout << "Original dimension: "
              << observation.size() << "\n";

    std::cout << "RNN input dimension: "
              << input.size() << "\n";

    std::cout << "First value: "
              << input[0] << "\n";

    std::cout << "Last value: "
              << input[17] << "\n";

    if (correct) {
        std::cout << "Observation adapter test PASSED\n";
        return 0;
    }

    std::cout << "Observation adapter test FAILED\n";
    return 1;
}
