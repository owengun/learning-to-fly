#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace learning_to_fly::rnn {

template <typename Values>
void require_finite(const Values& values, const char* message) {
    for (const auto value : values)
        if (!std::isfinite(value)) throw std::runtime_error(message);
}

template <typename Actor, typename Parameters>
struct ActorAdam {
    using Gradients = typename Actor::Gradients;
    using Scalar = typename Parameters::T;
    static constexpr std::size_t H = std::tuple_size_v<typename Actor::Hidden>;
    static constexpr std::size_t I = std::tuple_size_v<typename Actor::Input>;
    static constexpr std::size_t O = std::tuple_size_v<typename Actor::Output>;
    static constexpr std::size_t size = Actor::gate_count * H * (I + H + 1) + O * (H + 1);
    using Vector = std::array<Scalar, size>;
    Vector first{}, second{}, last_gradient{};
    std::size_t updates = 0;

    struct Step {
        double update_l2 = 0;
        double parameter_l2 = 0;
    };

    template <typename Visitor>
    static void visit(Actor& actor, const Gradients& gradients, Visitor visitor) {
        auto tensor = [&](auto& parameters, const auto& derivative, Scalar decay, Scalar factor) {
            for (std::size_t i = 0; i < parameters.size(); ++i)
                visitor(parameters[i], derivative[i], decay, factor);
        };
        for (std::size_t k = 0; k < Actor::gate_count; ++k) {
            auto& gate = actor.gates[k];
            const auto& grad = gradients.gates[k];
            for (std::size_t i = 0; i < H; ++i)
                tensor(gate.input[i], grad.input[i], Parameters::WEIGHT_DECAY_INPUT, 1);
            for (std::size_t i = 0; i < H; ++i)
                tensor(gate.recurrent[i], grad.recurrent[i], Parameters::WEIGHT_DECAY, 1);
            tensor(gate.bias, grad.bias, 0, Parameters::BIAS_LR_FACTOR);
        }
        for (std::size_t i = 0; i < O; ++i)
            tensor(actor.W_out[i], gradients.output[i], Parameters::WEIGHT_DECAY_OUTPUT, 1);
        tensor(actor.b_out, gradients.bias, 0, Parameters::BIAS_LR_FACTOR);
    }

    Step apply(Actor& actor, const Gradients& gradients) {
        Actor candidate = actor;
        std::size_t index = 0;
        visit(candidate, gradients, [&](Scalar&, Scalar raw, Scalar, Scalar) {
            last_gradient[index++] = raw;
        });
        require_finite(last_gradient, "Non-finite Adam actor gradient");

        Vector next_first = first, next_second = second;
        const Scalar age = static_cast<Scalar>(updates + 1);
        const Scalar first_correction = 1 / (1 - std::pow(Parameters::BETA_1, age));
        const Scalar second_correction = 1 / (1 - std::pow(Parameters::BETA_2, age));
        double update_squared = 0, parameter_squared = 0;
        index = 0;
        visit(candidate, gradients, [&](Scalar& value, Scalar raw, Scalar decay, Scalar bias_factor) {
            auto& m = next_first[index];
            auto& v = next_second[index++];
            m = Parameters::BETA_1 * m + (1 - Parameters::BETA_1) * raw;
            v = Parameters::BETA_2 * v + (1 - Parameters::BETA_2) * raw * raw;
            Scalar change = Parameters::ALPHA * first_correction * m /
                            (std::sqrt(v * second_correction) + Parameters::EPSILON);
            if (bias_factor > 1) change *= bias_factor;
            // Match RLtools CPU Adam: weight decay is outside alpha and only on weights.
            change += value * decay / 2;
            const Scalar next_value = value - change;
            if (!std::isfinite(m) || !std::isfinite(v) || !std::isfinite(next_value))
                throw std::runtime_error("Non-finite Adam actor parameter or moment");
            const double actual = static_cast<double>(next_value) - value;
            update_squared += actual * actual;
            parameter_squared += static_cast<double>(next_value) * next_value;
            value = next_value;
        });
        // Commit the whole update only after checking every parameter.
        actor = candidate;
        first = next_first;
        second = next_second;
        ++updates;
        return {std::sqrt(update_squared), std::sqrt(parameter_squared)};
    }

    bool save_diagnostic_state(const std::filesystem::path& path) const {
        std::ofstream stream(path);
        stream << std::setprecision(std::numeric_limits<Scalar>::max_digits10);
        stream << "parameter,first_moment,second_moment,last_raw_gradient,completed_updates\n";
        for (std::size_t i = 0; i < size; ++i)
            stream << i << ',' << first[i] << ',' << second[i] << ','
                   << last_gradient[i] << ',' << updates << '\n';
        stream.flush();
        return stream.good();
    }
};

} // namespace learning_to_fly::rnn
