
#ifndef LEARNING_TO_FLY_RNN_ACTOR_H
#define LEARNING_TO_FLY_RNN_ACTOR_H

#include <array>
#include <cmath>
#include <cstddef>
#include <random>

namespace learning_to_fly::rnn {

template <std::size_t INPUT_DIM = 18,
          std::size_t HIDDEN_DIM = 32,
          std::size_t OUTPUT_DIM = 4>
struct RNNActor {
    using Input = std::array<float, INPUT_DIM>;
    using Hidden = std::array<float, HIDDEN_DIM>;
    using Output = std::array<float, OUTPUT_DIM>;

    std::array<std::array<float, INPUT_DIM>, HIDDEN_DIM> W_x{};
    std::array<std::array<float, HIDDEN_DIM>, HIDDEN_DIM> W_h{};
    std::array<float, HIDDEN_DIM> b_h{};

    std::array<std::array<float, HIDDEN_DIM>, OUTPUT_DIM> W_out{};
    std::array<float, OUTPUT_DIM> b_out{};

    static constexpr bool recurrent = true;
    static constexpr const char* name = "vanilla";

    template <typename Actor, typename Visit>
    static void visit_parameters(Actor& actor, Visit visit) {
        for (auto& row : actor.W_x)
            for (auto& value : row) visit(value);
        for (auto& row : actor.W_h)
            for (auto& value : row) visit(value);
        for (auto& value : actor.b_h) visit(value);
        for (auto& row : actor.W_out)
            for (auto& value : row) visit(value);
        for (auto& value : actor.b_out) visit(value);
    }

    Hidden hidden{};

    void reset() {
        hidden.fill(0.0f);
    }

    // Initialize trainable weights.
    void initialize(unsigned int seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

        for (auto& row : W_x)
            for (auto& w : row)
                w = dist(rng);

        for (auto& row : W_h)
            for (auto& w : row)
                w = dist(rng);

        for (auto& row : W_out)
            for (auto& w : row)
                w = dist(rng);

        b_h.fill(0.0f);
        b_out.fill(0.0f);
        reset();
    }

    // Forward pass with recurrent hidden state.
    Output forward(const Input& input) {
        Hidden next_hidden{};

        for (std::size_t i = 0; i < HIDDEN_DIM; ++i) {
            float value = b_h[i];

            for (std::size_t j = 0; j < INPUT_DIM; ++j)
                value += W_x[i][j] * input[j];

            for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
                value += W_h[i][j] * hidden[j];

            next_hidden[i] = std::tanh(value);
        }

        hidden = next_hidden;

        Output output{};

        for (std::size_t i = 0; i < OUTPUT_DIM; ++i) {
            float value = b_out[i];

            for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
                value += W_out[i][j] * hidden[j];

            output[i] = std::tanh(value);
        }

        return output;
    }

    // Train on a sequence using BPTT.
    // The target is the desired output at the final time step.
    template <std::size_t SEQ_LEN>
    float train_sequence(
        const std::array<Input, SEQ_LEN>& inputs,
        const Output& target,
        float learning_rate = 0.01f
    ) {
        static_assert(SEQ_LEN > 0, "Sequence must not be empty");

        // Forward through time, starting a new sequence.
        reset();

        std::array<Hidden, SEQ_LEN + 1> states{};
        Output output{};

        for (std::size_t t = 0; t < SEQ_LEN; ++t) {
            output = forward(inputs[t]);
            states[t + 1] = hidden;
        }

        // Mean squared error and output gradient.
        float loss = 0.0f;
        Output delta_out{};

        for (std::size_t i = 0; i < OUTPUT_DIM; ++i) {
            float error = output[i] - target[i];
            loss += error * error / OUTPUT_DIM;

            delta_out[i] =
                (2.0f / OUTPUT_DIM) * error *
                (1.0f - output[i] * output[i]);
        }

        // Accumulate gradients across the full sequence.
        decltype(W_x) gWx{};
        decltype(W_h) gWh{};
        decltype(b_h) gbh{};
        decltype(W_out) gWout{};
        decltype(b_out) gbout{};

        Hidden dh{};

        for (std::size_t i = 0; i < OUTPUT_DIM; ++i) {
            gbout[i] = delta_out[i];

            for (std::size_t j = 0; j < HIDDEN_DIM; ++j) {
                gWout[i][j] =
                    delta_out[i] * states[SEQ_LEN][j];

                dh[j] += W_out[i][j] * delta_out[i];
            }
        }

        // Backpropagation through time.
        for (std::size_t t = SEQ_LEN; t > 0; --t) {
            Hidden delta_h{};
            Hidden dh_previous{};

            for (std::size_t i = 0; i < HIDDEN_DIM; ++i) {
                float h = states[t][i];
                delta_h[i] = dh[i] * (1.0f - h * h);

                gbh[i] += delta_h[i];

                for (std::size_t j = 0; j < INPUT_DIM; ++j)
                    gWx[i][j] +=
                        delta_h[i] * inputs[t - 1][j];

                for (std::size_t j = 0; j < HIDDEN_DIM; ++j) {
                    gWh[i][j] +=
                        delta_h[i] * states[t - 1][j];

                    dh_previous[j] +=
                        W_h[i][j] * delta_h[i];
                }
            }

            dh = dh_previous;
        }

        // SGD parameter update.
        for (std::size_t i = 0; i < HIDDEN_DIM; ++i) {
            b_h[i] -= learning_rate * gbh[i];

            for (std::size_t j = 0; j < INPUT_DIM; ++j)
                W_x[i][j] -= learning_rate * gWx[i][j];

            for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
                W_h[i][j] -= learning_rate * gWh[i][j];
        }

        for (std::size_t i = 0; i < OUTPUT_DIM; ++i) {
            b_out[i] -= learning_rate * gbout[i];

            for (std::size_t j = 0; j < HIDDEN_DIM; ++j)
                W_out[i][j] -= learning_rate * gWout[i][j];
        }

        return loss;
    }
    
    // Train the RNN using an external gradient dLoss/dAction.
    // This allows a critic to supply the actor's policy gradient.
    struct Gradients {
        decltype(W_x) gWx{};
        decltype(W_h) gWh{};
        decltype(b_h) gbh{};
        decltype(W_out) gWout{};
        decltype(b_out) gbout{};
    };

    template <std::size_t SEQ_LEN>
    void accumulate_sequence_gradient(
        const std::array<Input, SEQ_LEN>& inputs,
        const Output& action_gradient,
        Gradients& gradients
    ) {
        static_assert(SEQ_LEN > 0, "Sequence must not be empty");

        // Start with zero hidden state.
        reset();

        std::array<Hidden, SEQ_LEN + 1> states{};
        Output output{};

        // Forward through the entire sequence.
        for (std::size_t t = 0; t < SEQ_LEN; ++t) {
            output = forward(inputs[t]);
            states[t + 1] = hidden;
        }

        // Convert dLoss/dAction into dLoss/dPreActivation.
        Output delta_out{};

        for (std::size_t i = 0; i < OUTPUT_DIM; ++i) {
            delta_out[i] =
                action_gradient[i] *
                (1.0f - output[i] * output[i]);
        }

        Hidden dh{};

        // Output layer backward.
        for (std::size_t i = 0; i < OUTPUT_DIM; ++i) {
            gradients.gbout[i] += delta_out[i];

            for (std::size_t j = 0; j < HIDDEN_DIM; ++j) {
                gradients.gWout[i][j] +=
                    delta_out[i] * states[SEQ_LEN][j];

                dh[j] +=
                    W_out[i][j] * delta_out[i];
            }
        }

        // Backpropagation through time.
        for (std::size_t t = SEQ_LEN; t > 0; --t) {
            Hidden delta_h{};
            Hidden dh_previous{};

            for (std::size_t i = 0; i < HIDDEN_DIM; ++i) {
                float h = states[t][i];

                delta_h[i] =
                    dh[i] * (1.0f - h * h);

                gradients.gbh[i] += delta_h[i];

                for (std::size_t j = 0; j < INPUT_DIM; ++j) {
                    gradients.gWx[i][j] +=
                        delta_h[i] * inputs[t - 1][j];
                }

                for (std::size_t j = 0; j < HIDDEN_DIM; ++j) {
                    gradients.gWh[i][j] +=
                        delta_h[i] * states[t - 1][j];

                    dh_previous[j] +=
                        W_h[i][j] * delta_h[i];
                }
            }

            dh = dh_previous;
        }

    }

    void apply_gradients(const Gradients& gradients, float learning_rate) {
        for (std::size_t i = 0; i < HIDDEN_DIM; ++i) {
            b_h[i] -= learning_rate * gradients.gbh[i];

            for (std::size_t j = 0; j < INPUT_DIM; ++j) {
                W_x[i][j] -= learning_rate * gradients.gWx[i][j];
            }

            for (std::size_t j = 0; j < HIDDEN_DIM; ++j) {
                W_h[i][j] -= learning_rate * gradients.gWh[i][j];
            }
        }

        for (std::size_t i = 0; i < OUTPUT_DIM; ++i) {
            b_out[i] -= learning_rate * gradients.gbout[i];

            for (std::size_t j = 0; j < HIDDEN_DIM; ++j) {
                W_out[i][j] -=
                    learning_rate * gradients.gWout[i][j];
            }
        }
    }

    template <std::size_t SEQ_LEN>
    void train_sequence_with_gradient(
        const std::array<Input, SEQ_LEN>& inputs,
        const Output& action_gradient,
        float learning_rate = 0.001f
    ) {
        Gradients gradients{};
        accumulate_sequence_gradient(inputs, action_gradient, gradients);
        apply_gradients(gradients, learning_rate);
    }

};

template <typename Actor, typename Sequence>
typename Actor::Output forward_sequence(Actor& actor, const Sequence& sequence) {
    actor.reset();
    if constexpr (Actor::recurrent) {
        typename Actor::Output output{};
        for (const auto& observation : sequence) output = actor.forward(observation);
        return output;
    } else {
        return actor.forward(sequence.back());
    }
}

} // namespace learning_to_fly::rnn

#endif
