#pragma once

#include "rnn_actor.h"

namespace learning_to_fly::rnn {

// The paper's two-hidden-layer MLP topology under the recurrent experiment's
// input, replay, optimizer, and evaluation protocol.
struct MatchedMLPActor {
    static constexpr bool recurrent = false;
    static constexpr const char* name = "mlp-matched";
    using Input = std::array<float, 18>;
    using Hidden = std::array<float, 64>;
    using Output = std::array<float, 4>;

    template <typename In, typename Out>
    struct Layer {
        std::array<In, std::tuple_size_v<Out>> weights{};
        Out bias{};

        Out forward(const In& input) const {
            auto output = bias;
            for (std::size_t i = 0; i < output.size(); ++i) {
                for (std::size_t j = 0; j < input.size(); ++j)
                    output[i] += weights[i][j] * input[j];
                output[i] = std::tanh(output[i]);
            }
            return output;
        }

        In backward(const In& input, const Out& output, const Out& upstream,
                    Layer& gradient) const {
            In downstream{};
            for (std::size_t i = 0; i < output.size(); ++i) {
                const float delta = upstream[i] * (1 - output[i] * output[i]);
                gradient.bias[i] += delta;
                for (std::size_t j = 0; j < input.size(); ++j) {
                    gradient.weights[i][j] += delta * input[j];
                    downstream[j] += delta * weights[i][j];
                }
            }
            return downstream;
        }

        void apply(const Layer& gradient, float rate) {
            for (std::size_t i = 0; i < bias.size(); ++i) {
                bias[i] -= rate * gradient.bias[i];
                for (std::size_t j = 0; j < weights[i].size(); ++j)
                    weights[i][j] -= rate * gradient.weights[i][j];
            }
        }
    };

    struct Gradients {
        Layer<Input, Hidden> first;
        Layer<Hidden, Hidden> second;
        Layer<Hidden, Output> output;
    };

    Gradients parameters{};
    std::array<float, 0> hidden{};
    void reset() {}

    template <typename Actor, typename Visit>
    static void visit_parameters(Actor& actor, Visit visit) {
        auto visit_layer = [&](auto& layer) {
            for (auto& row : layer.weights)
                for (auto& weight : row) visit(weight);
            for (auto& bias : layer.bias) visit(bias);
        };
        visit_layer(actor.parameters.first);
        visit_layer(actor.parameters.second);
        visit_layer(actor.parameters.output);
    }

    void initialize(unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        visit_parameters(*this, [&](float& value) { value = dist(rng); });
        parameters.first.bias.fill(0);
        parameters.second.bias.fill(0);
        parameters.output.bias.fill(0);
    }

    Output forward(const Input& input) const {
        return parameters.output.forward(
            parameters.second.forward(parameters.first.forward(input)));
    }

    template <std::size_t N>
    void accumulate_sequence_gradient(const std::array<Input, N>& sequence,
                                      const Output& upstream, Gradients& gradients) const {
        static_assert(N > 0);
        const auto& input = sequence.back();
        const auto first = parameters.first.forward(input);
        const auto second = parameters.second.forward(first);
        const auto output = parameters.output.forward(second);
        const auto d_second = parameters.output.backward(second, output, upstream, gradients.output);
        const auto d_first = parameters.second.backward(first, second, d_second, gradients.second);
        parameters.first.backward(input, first, d_first, gradients.first);
    }

    void apply_gradients(const Gradients& gradients, float rate) {
        parameters.first.apply(gradients.first, rate);
        parameters.second.apply(gradients.second, rate);
        parameters.output.apply(gradients.output, rate);
    }
};
}
