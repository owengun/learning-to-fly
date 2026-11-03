
#include <rl_tools/operations/cpu_mux.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;

#include "critic_gradient_adapter.h"

#include <cmath>
#include <iostream>

int main() {
    using DEVICE = rlt::devices::DefaultCPU;

    DEVICE device;

    using SPEC = rlt::matrix::Specification<
        float,
        DEVICE::index_t,
        2,
        4
    >;

    rlt::MatrixDynamic<SPEC> gradients;

    rlt::malloc(device, gradients);

    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
            rlt::set(
                gradients,
                row,
                col,
                static_cast<float>(row * 4 + col + 1)
            );
        }
    }

    auto result =
        learning_to_fly::rnn::extract_action_gradient(
            gradients, 1
        );

    std::cout << "Extracted gradients: ";

    for (float value : result) {
        std::cout << value << " ";
    }

    std::cout << "\n";

    const bool passed =
        result[0] == 5.0f &&
        result[1] == 6.0f &&
        result[2] == 7.0f &&
        result[3] == 8.0f;

    rlt::free(device, gradients);

    if (!passed) {
        std::cerr << "Critic gradient adapter FAILED\n";
        return 1;
    }

    std::cout << "Critic gradient adapter PASSED\n";
    return 0;
}
