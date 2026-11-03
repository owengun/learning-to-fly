#include "matched_mlp_actor.h"

#include <iostream>

int main() {
    using Actor = learning_to_fly::rnn::MatchedMLPActor;
    Actor actor;
    actor.initialize(7);
    std::array<Actor::Input, 2> sequence{};
    sequence[0][0] = 0.8f;
    sequence[1][0] = -0.4f;
    const Actor::Output gradient{0.7f, -0.2f};
    const auto action = learning_to_fly::rnn::forward_sequence(actor, sequence);
    if (action != actor.forward(sequence.back())) return 1;

    Actor::Gradients accumulated{};
    actor.accumulate_sequence_gradient(sequence, gradient, accumulated);
    auto loss = [&](const Actor& candidate) {
        const auto output = candidate.forward(sequence.back());
        return output[0] * gradient[0] + output[1] * gradient[1];
    };
    auto plus = actor;
    auto minus = actor;
    constexpr float epsilon = 0.001f;
    plus.parameters.first.weights[0][0] += epsilon;
    minus.parameters.first.weights[0][0] -= epsilon;
    const float expected = (loss(plus) - loss(minus)) / (2 * epsilon);
    const float actual = accumulated.first.weights[0][0];
    if (!std::isfinite(actual) || std::abs(actual - expected) > 0.00001f) return 1;
    const auto before = loss(actor);
    actor.apply_gradients(accumulated, 0.01f);
    if (!(loss(actor) < before)) return 1;
    std::cout << "Matched MLP forward and finite-difference gradient passed\n";
}
