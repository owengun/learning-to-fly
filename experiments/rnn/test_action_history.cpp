#include "action_history_sampler.h"
#include "gated_actor.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace rnn = learning_to_fly::rnn;
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;
using Device = rlt::devices::DefaultCPU;
using TI = Device::index_t;

struct Replay {
    static constexpr TI CAPACITY = 64;
    template <typename T, TI Columns>
    using Matrix = rlt::MatrixStatic<rlt::matrix::Specification<T, TI, CAPACITY, Columns>>;
    Matrix<float, 18> observations, next_observations;
    Matrix<float, 4> actions;
    Matrix<bool, 1> terminated, truncated;
    TI position = CAPACITY;
    bool full = false;
};

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    Replay rb{};
    Device device;
    rlt::malloc(device, rb.observations);
    rlt::malloc(device, rb.next_observations);
    rlt::malloc(device, rb.actions);
    rlt::malloc(device, rb.terminated);
    rlt::malloc(device, rb.truncated);
    for (TI i = 0; i < rb.CAPACITY; ++i) {
        for (TI j = 0; j < 18; ++j) {
            rlt::set(rb.observations, i, j, static_cast<float>(i + j) / 100);
            rlt::set(rb.next_observations, i, j, static_cast<float>(i + j + 9) / 100);
        }
        for (TI j = 0; j < 4; ++j)
            rlt::set(rb.actions, i, j, -static_cast<float>(i + j) / 100);
        rlt::set(rb.terminated, i, 0, false);
        rlt::set(rb.truncated, i, 0, false);
    }

    rnn::HistoryWindow window;
    require(rnn::extract_history_window(rb, 45, 8, window), "full window");
    require(window.burn == 32 && window.learn == 8, "32 + 8 window dimensions");
    require(window.inputs[0][18] == rlt::get(rb.actions, 5, 0) &&
            window.inputs[32][18] == rlt::get(rb.actions, 37, 0),
            "history must use recorded previous action");

    using Actor = rnn::GatedActor<false, 22, 4>;
    Actor actor;
    actor.initialize(8);
    rnn::GatedActor<false, 18, 4> base;
    base.initialize(8);
    decltype(base)::Input state_only{};
    std::copy_n(window.inputs[0].begin(), 18, state_only.begin());
    require(actor.forward(window.inputs[0]) == base.forward(state_only),
            "initialization must preserve old policy");
    actor.reset();
    const auto initial = rnn::burn_in(actor, window);
    for (std::size_t t = 0; t < window.burn; ++t) actor.forward(window.inputs[t]);
    require(actor.hidden.h == initial.h, "burn-in must equal step-by-step recurrence");

    auto state = initial;
    for (std::size_t t = 0; t < window.learn; ++t) {
        const auto& input = window.inputs[window.burn + t];
        state = actor.advance(input, state).state;
        require(actor.forward(input) == actor.action(state.h), "online/replay action agreement");
    }
    const auto row = window.transitions[window.learn - 1];
    auto target = actor;
    const auto target_next = rnn::next_history_action(target, rb, row, state);
    auto next = window.inputs[window.burn + window.learn - 1];
    for (TI j = 0; j < 18; ++j) next[j] = rlt::get(rb.next_observations, row, j);
    for (TI j = 0; j < 4; ++j) next[18 + j] = rlt::get(rb.actions, row, j);
    target.hidden = state;
    require(target_next == target.forward(next), "target must extend current history once");

    std::array<Actor::Output, 8> upstream{};
    upstream[0][1] = -0.5f;
    upstream[3][1] = 0.4f;
    upstream.back()[0] = 0.8f;
    Actor::Gradients gradient{};
    const auto* learning = window.inputs.data() + window.burn;
    actor.accumulate_window_gradient(learning, window.learn, initial, upstream, gradient);
    auto loss = [&](const Actor& candidate) {
        auto previous = initial;
        float sum = 0;
        for (std::size_t t = 0; t < window.learn; ++t) {
            previous = candidate.advance(learning[t], previous).state;
            const auto action = candidate.action(previous.h);
            for (std::size_t i = 0; i < action.size(); ++i) sum += action[i] * upstream[t][i];
        }
        return sum;
    };
    constexpr float epsilon = 0.001f;
    auto plus = actor, minus = actor;
    plus.gates[2].input[0][18] += epsilon;
    minus.gates[2].input[0][18] -= epsilon;
    const float numerical = (loss(plus) - loss(minus)) / (2 * epsilon);
    require(std::abs(numerical - gradient.gates[2].input[0][18]) < 0.00002f &&
            std::abs(numerical) > 0.0001f, "finite difference for history weight and multi-step loss");

    rlt::set(rb.truncated, 45, 0, true);
    require(rnn::extract_history_window(rb, 45, 8, window) && window.learn == 8,
            "truncation endpoint is valid");
    require(rnn::extract_history_window(rb, 48, 8, window) && window.burn == 0 &&
            window.learn == 3 && window.transitions[0] == 46 && window.inputs[0][18] == 0,
            "time limit resets action and hidden history");
    rlt::set(rb.terminated, 46, 0, true);
    require(rnn::extract_history_window(rb, 47, 8, window) && window.learn == 1 &&
            window.transitions[0] == 47 && window.inputs[0][18] == 0,
            "termination resets action and hidden history");
    rlt::set(rb.truncated, 45, 0, false);
    rlt::set(rb.terminated, 46, 0, false);
    rb.full = true;
    rb.position = 60;
    require(!rnn::extract_history_window(rb, 0, 8, window), "overwritten predecessor");
    require(rnn::extract_history_window(rb, 44, 8, window) && window.burn == 32 &&
            window.inputs[0][18] == rlt::get(rb.actions, 0, 0),
            "window respects circular replay order");
    std::cout << "GRU history, burn-in, target alignment and multi-step BPTT passed\n";
}
