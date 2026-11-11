#include "../../src/training.h"
#include <stdexcept>

namespace rnn = learning_to_fly::rnn;
using Config = learning_to_fly::config::Config<learning_to_fly::config::DEFAULT_ABLATION_SPEC>;
using Actor = decltype(learning_to_fly::TrainingState<Config>::rnn_actor);

void require(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}

template <typename Model>
void gradients() {
    Model actor;
    actor.initialize(17);
    std::array<typename Model::Input, 8> inputs{};
    std::array<typename Model::Output, 8> upstream{};
    for (std::size_t t = 0; t < inputs.size(); ++t) {
        for (std::size_t j = 0; j < inputs[t].size(); ++j)
            inputs[t][j] = 0.2f * std::sin(float(t * 3 + j));
        for (std::size_t j = 0; j < upstream[t].size(); ++j)
            upstream[t][j] = 0.3f * std::cos(float(t + j));
    }
    typename Model::State initial{};
    initial.h.fill(0.05f);
    initial.cell.fill(0.13f);
    actor.hidden = initial;
    typename Model::Gradients gradient{};
    actor.accumulate_window_gradient(inputs.data(), inputs.size(), initial, upstream, gradient);
    require(actor.hidden.h == initial.h && actor.hidden.cell == initial.cell,
            "offline BPTT must preserve live episode state");
    auto loss = [&](const Model& candidate) {
        auto state = initial;
        double result = 0;
        for (std::size_t t = 0; t < inputs.size(); ++t) {
            state = candidate.advance(inputs[t], state).state;
            const auto action = candidate.action(state.h);
            for (std::size_t j = 0; j < action.size(); ++j) result += action[j] * upstream[t][j];
        }
        return result;
    };
    std::vector<float> analytical;
    rnn::ActorAdam<Model, typename Config::OPTIMIZER::PARAMETERS>::visit(actor, gradient,
        [&](float&, float value, float, float) { analytical.push_back(value); });
    std::size_t index = 0;
    Model::visit_parameters(actor, [&](float& weight) {
        // Exhaust the small cell and sample every gate/tensor in the actual full-size cell.
        const bool selected = Model::parameter_count < 200 || index % 97 == 0 || index + 1 == Model::parameter_count;
        if (selected) {
            const float original = weight;
            constexpr float epsilon = 0.002f;
            weight = original + epsilon;
            const double plus = loss(actor);
            weight = original - epsilon;
            const double minus = loss(actor);
            weight = original;
            const double numerical = (plus - minus) / (2 * epsilon);
            require(std::abs(numerical - analytical[index]) < 0.00008 + 0.004 * std::abs(numerical),
                    "multi-step finite difference with a fixed nonzero burn-in state");
        }
        ++index;
    });
    // Explicitly probe the last action-history column in every gate.
    for (std::size_t k = 0; k < Model::gate_count; ++k) {
        auto& weight = actor.gates[k].input[0].back();
        const float original = weight;
        constexpr float epsilon = 0.002f;
        weight += epsilon;
        const double plus = loss(actor);
        weight = original - epsilon;
        const double minus = loss(actor);
        weight = original;
        require(std::abs((plus - minus) / (2 * epsilon) - gradient.gates[k].input[0].back()) < 0.00008,
                "finite difference includes the newest action-history feature");
    }
    actor.reset();
    require(actor.hidden.h == typename Model::Hidden{} && actor.hidden.cell == typename Model::Hidden{},
            "episode reset clears both recurrent states");
}

struct Replay {
    using TI = Config::TI;
    static constexpr TI CAPACITY = 64;
    template <typename T, TI C>
    using Matrix = rlt::MatrixStatic<rlt::matrix::Specification<T, TI, CAPACITY, C>>;
    Matrix<float, 146> observations{}, next_observations{};
    Matrix<bool, 1> terminated{}, truncated{};
    TI position = CAPACITY;
    bool full = false;
};

int main() {
#ifdef LEARNING_TO_FLY_ACTOR_RNN_FULL
    gradients<rnn::GatedActor<false, 3, 3, 2, true>>();
    // Vanilla equations and initialization must match the original tanh RNN.
    Actor current;
    rnn::RNNActor<146, 64> original;
    current.initialize(19);
    original.initialize(19);
    Actor::Input input{};
    input[145] = 0.4f;
    for (int i = 0; i < 5; ++i)
        require(current.forward(input) == original.forward(input), "vanilla recurrence matches original equations");
#else
    gradients<rnn::GatedActor<true, 3, 3, 2>>();
    static_assert(rnn::GatedActor<true, 146, 20>::parameter_count == 13444);
    static_assert(rnn::GatedActor<true, 146, 21>::parameter_count == 14200);
#endif
    gradients<Actor>();
    rlt::devices::DefaultCPU device;
    Replay replay;
    rlt::malloc(device, replay.observations);
    rlt::malloc(device, replay.next_observations);
    rlt::malloc(device, replay.terminated);
    rlt::malloc(device, replay.truncated);
    for (std::size_t i = 0; i < replay.CAPACITY; ++i) {
        for (std::size_t j = 0; j < 146; ++j) {
            rlt::set(replay.observations, i, j, float(i + j) / 200);
            rlt::set(replay.next_observations, i, j, float(i + j + 7) / 200);
        }
        rlt::set(replay.terminated, i, 0, false);
        rlt::set(replay.truncated, i, 0, false);
    }
    rnn::HistoryWindowFor<Actor::Input> window;
    require(rnn::extract_history_window(replay, 45, 8, window), "full replay window");
    require(window.burn == 32 && window.learn == 8, "same 32/8 sequence protocol");
    Actor actor;
    actor.initialize(8);
    auto state = rnn::burn_in(actor, window);
    for (std::size_t t = 0; t < window.learn; ++t) {
        state = actor.advance(window.inputs[window.burn + t], state).state;
        const auto row = window.transitions[t];
        auto expected = actor;
        expected.hidden = state;
        const auto next = rnn::full_observation(replay.next_observations, row);
        require(rnn::next_history_action(actor, replay, row, state) == expected.forward(next),
                "target extends each current prefix exactly once with matching next observation");
    }
    rlt::set(replay.truncated, 45, 0, true);
    require(rnn::extract_history_window(replay, 48, 8, window) && window.burn == 0 &&
            window.learn == 3 && window.transitions[0] == 46, "time-limit boundary");
    rlt::set(replay.terminated, 46, 0, true);
    require(rnn::extract_history_window(replay, 47, 8, window) && window.burn == 0 &&
            window.learn == 1 && window.transitions[0] == 47, "true termination boundary");
    rlt::set(replay.truncated, 45, 0, false);
    rlt::set(replay.terminated, 46, 0, false);
    replay.full = true;
    replay.position = 60;
    require(!rnn::extract_history_window(replay, 0, 8, window), "reject overwritten prefix");
    require(rnn::extract_history_window(replay, 44, 8, window) && window.burn == 32 &&
            window.inputs[0][145] == rlt::get(replay.observations, 1, 145), "circular replay chronology");
    auto target = actor;
    target.hidden.h.fill(0.4f);
    target.hidden.cell.fill(0.3f);
    actor.gates[0].input[0][145] += 1;
    const float before = target.gates[0].input[0][145];
    rnn::update_target(target, actor, 0.75f);
    require(std::abs(target.gates[0].input[0][145] - (0.75f * before + 0.25f * actor.gates[0].input[0][145])) < 1e-7,
            "target Polyak alignment includes history weights");
    require(target.hidden.h[0] == 0.4f && target.hidden.cell[0] == 0.3f, "target update preserves live state");
    const auto saved = target.gates[0].input;
    const std::filesystem::path path = std::filesystem::path("checkpoint-test") / "invalid.txt";
    std::filesystem::create_directories(path.parent_path());
    { std::ofstream file(path); file << "recurrent-actor-v1 " << Actor::name << "\n1\n"; }
    require(!rnn::load_actor(target, path) && target.gates[0].input == saved,
            "reject truncated checkpoint without mutation");
    std::cout << "Full recurrent gradients, episode boundaries, target alignment and checkpoint rejection passed\n";
}
