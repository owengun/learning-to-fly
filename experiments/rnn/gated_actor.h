#pragma once

#include "rnn_actor.h"

namespace learning_to_fly::rnn {

template <bool LSTM, std::size_t I = 18, std::size_t H = 32, std::size_t O = 4>
struct GatedActor {
    using Base = RNNActor<I, H, O>;
    using Input = typename Base::Input;
    using Hidden = typename Base::Hidden;
    using Output = typename Base::Output;
    static constexpr bool recurrent = true;
    static constexpr const char* name = LSTM ? "lstm" : "gru";
    static constexpr std::size_t gate_count = LSTM ? 4 : 3;

    struct Gate {
        decltype(Base::W_x) input{};
        decltype(Base::W_h) recurrent{};
        Hidden bias{};

        Hidden affine(const Input& x, const Hidden& h) const {
            auto result = bias;
            for (std::size_t i = 0; i < H; ++i) {
                for (std::size_t j = 0; j < I; ++j)
                    result[i] += input[i][j] * x[j];
                for (std::size_t j = 0; j < H; ++j)
                    result[i] += recurrent[i][j] * h[j];
            }
            return result;
        }

        void backward(const Input& x, const Hidden& h, const Hidden& delta,
                      Gate& gradient, Hidden& dh) const {
            for (std::size_t i = 0; i < H; ++i) {
                gradient.bias[i] += delta[i];
                for (std::size_t j = 0; j < I; ++j)
                    gradient.input[i][j] += delta[i] * x[j];
                for (std::size_t j = 0; j < H; ++j) {
                    gradient.recurrent[i][j] += delta[i] * h[j];
                    dh[j] += recurrent[i][j] * delta[i];
                }
            }
        }
    };

    struct State { Hidden h{}, cell{}; };
    struct Frame {
        State state;
        std::array<Hidden, gate_count> gate;
        Hidden reset_hidden{};
    };
    struct Gradients {
        std::array<Gate, gate_count> gates{};
        decltype(Base::W_out) output{};
        Output bias{};
    };

    std::array<Gate, gate_count> gates{};
    decltype(Base::W_out) W_out{};
    Output b_out{};
    State hidden{};

    void reset() { hidden = {}; }

    template <typename Actor, typename Visit>
    static void visit_parameters(Actor& actor, Visit visit) {
        for (auto& gate : actor.gates) {
            for (auto& row : gate.input)
                for (auto& value : row) visit(value);
            for (auto& row : gate.recurrent)
                for (auto& value : row) visit(value);
            for (auto& value : gate.bias) visit(value);
        }
        for (auto& row : actor.W_out)
            for (auto& value : row) visit(value);
        for (auto& value : actor.b_out) visit(value);
    }

    void initialize(unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        visit_parameters(*this, [&](float& value) { value = dist(rng); });
        for (auto& gate : gates) gate.bias.fill(0);
        if constexpr (LSTM) gates[1].bias.fill(1);
        b_out.fill(0);
        reset();
    }

    Frame advance(const Input& input, const State& previous) const {
        Frame frame{};
        auto& g = frame.gate;
        constexpr std::size_t sigmoid_gates = LSTM ? 3 : 2;
        for (std::size_t k = 0; k < sigmoid_gates; ++k) {
            g[k] = gates[k].affine(input, previous.h);
            for (auto& value : g[k]) value = 1 / (1 + std::exp(-value));
        }
        if constexpr (!LSTM) {
            for (std::size_t j = 0; j < H; ++j)
                frame.reset_hidden[j] = g[1][j] * previous.h[j];
        }
        auto& candidate = g.back();
        candidate = gates.back().affine(input, LSTM ? previous.h : frame.reset_hidden);
        for (std::size_t j = 0; j < H; ++j) {
            candidate[j] = std::tanh(candidate[j]);
            if constexpr (LSTM) {
                frame.state.cell[j] = g[1][j] * previous.cell[j] + g[0][j] * candidate[j];
                frame.state.h[j] = g[2][j] * std::tanh(frame.state.cell[j]);
            } else {
                frame.state.h[j] = g[0][j] * previous.h[j] + (1 - g[0][j]) * candidate[j];
            }
        }
        return frame;
    }

    Output action(const Hidden& h) const {
        auto output = b_out;
        for (std::size_t i = 0; i < O; ++i) {
            for (std::size_t j = 0; j < H; ++j) output[i] += W_out[i][j] * h[j];
            output[i] = std::tanh(output[i]);
        }
        return output;
    }

    Output forward(const Input& input) {
        hidden = advance(input, hidden).state;
        return action(hidden.h);
    }

    template <std::size_t N>
    void accumulate_sequence_gradient(const std::array<Input, N>& inputs,
                                      const Output& gradient, Gradients& gradients) const {
        static_assert(N > 0);
        std::array<Frame, N + 1> frames{};
        for (std::size_t t = 0; t < N; ++t)
            frames[t + 1] = advance(inputs[t], frames[t].state);

        const auto& last = frames.back().state.h;
        const auto output = action(last);
        Hidden dh{}, dc{};
        for (std::size_t i = 0; i < O; ++i) {
            float delta = gradient[i] * (1 - output[i] * output[i]);
            gradients.bias[i] += delta;
            for (std::size_t j = 0; j < H; ++j) {
                gradients.output[i][j] += delta * last[j];
                dh[j] += delta * W_out[i][j];
            }
        }

        for (std::size_t t = N; t > 0; --t) {
            const auto& frame = frames[t];
            const auto& previous = frames[t - 1].state;
            const auto& g = frame.gate;
            const auto& candidate = g.back();
            std::array<Hidden, gate_count> delta{};
            Hidden previous_dh{};
            for (std::size_t j = 0; j < H; ++j) {
                if constexpr (LSTM) {
                    const float cell = std::tanh(frame.state.cell[j]);
                    dc[j] += dh[j] * g[2][j] * (1 - cell * cell);
                    delta[0][j] = dc[j] * candidate[j];
                    delta[1][j] = dc[j] * previous.cell[j];
                    delta[2][j] = dh[j] * cell;
                    delta.back()[j] = dc[j] * g[0][j];
                    dc[j] *= g[1][j];
                } else {
                    delta[0][j] = dh[j] * (previous.h[j] - candidate[j]);
                    delta.back()[j] = dh[j] * (1 - g[0][j]);
                    previous_dh[j] = dh[j] * g[0][j];
                }
                delta.back()[j] *= 1 - candidate[j] * candidate[j];
            }
            if constexpr (LSTM) {
                gates.back().backward(inputs[t - 1], previous.h, delta.back(),
                                      gradients.gates.back(), previous_dh);
            } else {
                Hidden dreset{};
                gates.back().backward(inputs[t - 1], frame.reset_hidden, delta.back(),
                                      gradients.gates.back(), dreset);
                for (std::size_t j = 0; j < H; ++j) {
                    delta[1][j] = dreset[j] * previous.h[j];
                    previous_dh[j] += dreset[j] * g[1][j];
                }
            }
            for (std::size_t k = 0; k + 1 < gate_count; ++k) {
                for (std::size_t j = 0; j < H; ++j)
                    delta[k][j] *= g[k][j] * (1 - g[k][j]);
                gates[k].backward(inputs[t - 1], previous.h, delta[k],
                                  gradients.gates[k], previous_dh);
            }
            dh = previous_dh;
        }
    }

    void apply_gradients(const Gradients& gradients, float rate) {
        for (std::size_t k = 0; k < gate_count; ++k) {
            for (std::size_t i = 0; i < H; ++i) {
                gates[k].bias[i] -= rate * gradients.gates[k].bias[i];
                for (std::size_t j = 0; j < I; ++j)
                    gates[k].input[i][j] -= rate * gradients.gates[k].input[i][j];
                for (std::size_t j = 0; j < H; ++j)
                    gates[k].recurrent[i][j] -= rate * gradients.gates[k].recurrent[i][j];
            }
        }
        for (std::size_t i = 0; i < O; ++i) {
            b_out[i] -= rate * gradients.bias[i];
            for (std::size_t j = 0; j < H; ++j)
                W_out[i][j] -= rate * gradients.output[i][j];
        }
    }
};
}
