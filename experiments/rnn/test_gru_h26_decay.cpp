#include "../../src/training.h"
#include <cmath>
#include <stdexcept>

namespace rnn = learning_to_fly::rnn;
using Config = learning_to_fly::config::Config<learning_to_fly::config::DEFAULT_ABLATION_SPEC>;
using Training = learning_to_fly::TrainingState<Config>;
using Actor = decltype(Training::rnn_actor);
using Parameters = typename Config::OPTIMIZER::PARAMETERS;

void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

int main() {
    Training state;
    learning_to_fly::init(state, 0);
    for (int i = 0; i < 80; ++i) learning_to_fly::step(state);

    const float critic_alpha = state.critic_optimizers[0].alpha;
    state.step = 100000;
    learning_to_fly::step(state);
    require(state.history_adam.alpha == Parameters::ALPHA && state.history_adam.updates == 1,
            "first curriculum must keep Actor alpha");
    state.step = 199980;
    learning_to_fly::step(state);
    require(state.history_adam.alpha == Parameters::ALPHA && state.history_adam.updates == 2,
            "last Actor update before boundary must use base alpha");

    const auto before = state.rnn_actor.b_out;
    state.step = 200000;
    learning_to_fly::step(state);
    require(state.history_adam.alpha == Parameters::ALPHA * 0.1f &&
            state.history_adam.updates == 3, "first post-curriculum Actor update must use decayed alpha");
    require(state.critic_optimizers[0].alpha == critic_alpha &&
            state.critic_optimizers[1].alpha == critic_alpha, "Critic alpha must stay fixed");

    Actor saved{};
    require(rnn::load_actor(saved, std::filesystem::path("checkpoints") / Actor::name / "actor_step_200000.txt"),
            "load the boundary checkpoint");
    require(saved.b_out == before, "step 200000 checkpoint must precede the update");

    bool has_nonzero_update = false;
    const auto& optimizer = state.history_adam;
    for (std::size_t j = 0; j < before.size(); ++j) {
        const auto index = optimizer.size - before.size() + j;
        const double first = optimizer.first[index] / (1 - std::pow(Parameters::BETA_1, 3.f));
        const double second = optimizer.second[index] / (1 - std::pow(Parameters::BETA_2, 3.f));
        const double expected = optimizer.alpha * first / (std::sqrt(second) + Parameters::EPSILON);
        has_nonzero_update |= std::abs(expected) > 1e-6;
        require(std::abs((before[j] - state.rnn_actor.b_out[j]) - expected) < 1e-7,
                "boundary update must apply decayed alpha without resetting moments");
    }
    require(has_nonzero_update, "boundary test requires a nonzero bias update");
    state.step = 200020;
    learning_to_fly::step(state);
    require(state.history_adam.alpha == Parameters::ALPHA * 0.1f &&
            state.history_adam.updates == 4, "decay must persist on subsequent Actor updates");
    learning_to_fly::destroy(state);
    std::cout << "GRU h26 LR decay boundary and checkpoint order passed\n";
}
