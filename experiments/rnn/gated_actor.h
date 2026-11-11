#pragma once

#include "rnn_actor.h"

namespace learning_to_fly::rnn {

template <bool LSTM, std::size_t I = 18, std::size_t H = 32, std::size_t O = 4, bool VANILLA = false>
struct GatedActor {
    static_assert(!(LSTM && VANILLA));
    using Base = RNNActor<I, H, O>;
    using Input = typename Base::Input;
    using Hidden = typename Base::Hidden;
    using Output = typename Base::Output;
    static constexpr bool recurrent = true;
    static constexpr const char* name = VANILLA ? "rnn-full" : (LSTM ? (I == 146 ? "lstm-full" : "lstm") :
        (I == 146 ? "gru-full" : (I == 22 ? "gru-history" : "gru")));
    static constexpr std::size_t gate_count = VANILLA ? 1 : (LSTM ? 4 : 3);
    static constexpr std::size_t parameter_count = gate_count * H * (I + H + 1) + O * (H + 1);

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
        if constexpr (VANILLA) {
            Base reference;
            reference.initialize(seed);
            gates[0].input = reference.W_x;
            gates[0].recurrent = reference.W_h;
            gates[0].bias = reference.b_h;
            W_out = reference.W_out;
            b_out = reference.b_out;
            reset();
            return;
        }
        if constexpr (!LSTM && !VANILLA && I == 22) {
            // Match the 18-feature reference at initialization.
            GatedActor<false, 18, H, O> reference;
            reference.initialize(seed);
            for (std::size_t k = 0; k < gate_count; ++k) {
                gates[k].recurrent = reference.gates[k].recurrent;
                gates[k].bias = reference.gates[k].bias;
                for (std::size_t i = 0; i < H; ++i) {
                    gates[k].input[i].fill(0);
                    for (std::size_t j = 0; j < 18; ++j)
                        gates[k].input[i][j] = reference.gates[k].input[i][j];
                }
            }
            W_out = reference.W_out;
            b_out = reference.b_out;
            reset();
            return;
        }
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
        if constexpr (VANILLA) {
            g[0] = gates[0].affine(input, previous.h);
            for (std::size_t j = 0; j < H; ++j)
                frame.state.h[j] = g[0][j] = std::tanh(g[0][j]);
        } else {
            constexpr std::size_t sigmoid_gates = LSTM ? 3 : 2;
            for (std::size_t k = 0; k < sigmoid_gates; ++k) {
                g[k] = gates[k].affine(input, previous.h);
                for (auto& value : g[k]) value = 1 / (1 + std::exp(-value));
            }
            if constexpr (!LSTM && !VANILLA) {
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
        std::array<Output, N> upstream{};
        upstream.back() = gradient;
        accumulate_window_gradient(inputs.data(), N, State{}, upstream, gradients);
    }

    // initial_state comes from the burn-in prefix and is held fixed during BPTT.
    template <std::size_t N>
    void accumulate_window_gradient(const Input* inputs, std::size_t count,
                                    const State& initial_state,
                                    const std::array<Output, N>& upstream,
                                    Gradients& gradients) const {
        std::array<Frame, N + 1> frames{};
        frames[0].state = initial_state;
        for (std::size_t t = 0; t < count; ++t)
            frames[t + 1] = advance(inputs[t], frames[t].state);

        Hidden dh{}, dc{};
        for (std::size_t t = count; t > 0; --t) {
            const auto& h = frames[t].state.h;
            const auto output = action(h);
            for (std::size_t i = 0; i < O; ++i) {
                const float delta = upstream[t - 1][i] * (1 - output[i] * output[i]);
                gradients.bias[i] += delta;
                for (std::size_t j = 0; j < H; ++j) {
                    gradients.output[i][j] += delta * h[j];
                    dh[j] += delta * W_out[i][j];
                }
            }
            const auto& frame = frames[t];
            const auto& previous = frames[t - 1].state;
            const auto& g = frame.gate;
            const auto& candidate = g.back();
            std::array<Hidden, gate_count> delta{};
            Hidden previous_dh{};
            for (std::size_t j = 0; j < H; ++j) {
                if constexpr (VANILLA) {
                    delta.back()[j] = dh[j];
                } else if constexpr (LSTM) {
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
            if constexpr (LSTM || VANILLA) {
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

    static double gradient_norm(const Gradients& gradients) {
        double squared = 0;
        auto add = [&](const auto& values) {
            for (double value : values) squared += value * value;
        };
        for (const auto& gate : gradients.gates) {
            for (const auto& row : gate.input) add(row);
            for (const auto& row : gate.recurrent) add(row);
            add(gate.bias);
        }
        for (const auto& row : gradients.output) add(row);
        add(gradients.bias);
        return std::sqrt(squared);
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
