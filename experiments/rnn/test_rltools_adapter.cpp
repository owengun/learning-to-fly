
#include <rl_tools/operations/cpu_mux.h>
#include "observation_adapter.h"

#include <iostream>

namespace rlt = rl_tools;

int main() {
    using DEVICE = rlt::devices::DefaultCPU;
    DEVICE device;

    using MatrixSpec = rlt::matrix::Specification<
        float, DEVICE::index_t, 1, 146>;

    rlt::MatrixDynamic<MatrixSpec> observation{};
    rlt::malloc(device, observation);

    for (DEVICE::index_t i = 0; i < 146; ++i) {
        rlt::set(observation, 0, i, static_cast<float>(i));
    }

    auto input =
        learning_to_fly::rnn::extract_rltools_observation(
            observation
        );

    bool correct = true;

    for (std::size_t i = 0; i < 18; ++i) {
        if (input[i] != static_cast<float>(i)) {
            correct = false;
        }
    }

    std::cout << "RLtools matrix columns: 146\n";
    std::cout << "RNN input dimension: " << input.size() << "\n";
    std::cout << "First value: " << input[0] << "\n";
    std::cout << "Last value: " << input[17] << "\n";

    rlt::free(device, observation);
    if (correct) {
        std::cout << "RLtools adapter test PASSED\n";
        return 0;
    }

    std::cout << "RLtools adapter test FAILED\n";
    return 1;
}
