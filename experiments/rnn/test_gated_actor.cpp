#include "gated_actor.h"

#include <iostream>

template <bool LSTM>
bool check_actor() {
    using Actor = learning_to_fly::rnn::GatedActor<LSTM, 2, 3, 2>;
    Actor actor;
    actor.initialize(17);
    const std::array<typename Actor::Input, 3> sequence{{{0.5f, -0.2f}, {0.3f, 0.1f}, {-0.4f, 0.8f}}};
    const typename Actor::Output upstream{0.7f, -0.3f};
    auto forward = [&](Actor model) {
        model.reset();
        typename Actor::Output output{};
        for (const auto& input : sequence) output = model.forward(input);
        return output;
    };
    const auto output = forward(actor);
    actor.forward(sequence[0]);
    actor.forward(sequence[1]);
    if (actor.forward(sequence[2]) != output || forward(actor) != output ||
        actor.forward(sequence[2]) == output) return false;

    typename Actor::Gradients gradients{};
    actor.accumulate_sequence_gradient(sequence, upstream, gradients);
    auto loss = [&](Actor model) {
        const auto a = forward(model);
        return a[0] * upstream[0] + a[1] * upstream[1];
    };
    constexpr float eps = 0.002f;
    for (std::size_t k = 0; k < Actor::gate_count; ++k) {
        auto plus = actor;
        auto minus = actor;
        plus.gates[k].recurrent[0][1] += eps;
        minus.gates[k].recurrent[0][1] -= eps;
        const float numerical = (loss(plus) - loss(minus)) / (2 * eps);
        const float analytical = gradients.gates[k].recurrent[0][1];
        if (!std::isfinite(analytical) || std::abs(numerical - analytical) > 0.00001f)
            return false;
    }
    actor.apply_gradients(gradients, 0.01f);
    return loss(actor) < output[0] * upstream[0] + output[1] * upstream[1];
}

int main() {
    if (!check_actor<true>()) {
        std::cerr << "Gated actor gradient or state reset failed\n";
        return 1;
    }
    std::cout << "LSTM forward, reset and BPTT passed\n";
}
