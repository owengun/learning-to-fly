#include "../../src/training.h"
#include "actor_adam.h"
#include "actor_checkpoint.h"
#include <type_traits>

namespace rnn = learning_to_fly::rnn;
using Config = learning_to_fly::config::Config<learning_to_fly::config::DEFAULT_ABLATION_SPEC>;
using State = learning_to_fly::TrainingState<Config>;
using Model = decltype(State::rnn_actor);
using Target = decltype(State::rnn_actor_target);
using Adam = decltype(State::history_adam);
using Parameters = typename Config::OPTIMIZER::PARAMETERS;
static_assert(std::is_same_v<Model, Target>);
static_assert(Adam::size == 5412);
static_assert(Parameters::ALPHA == 1e-3f && Parameters::BETA_1 == 0.9f &&
              Parameters::BETA_2 == 0.999f && Parameters::EPSILON == 1e-8f);
static_assert(Parameters::WEIGHT_DECAY == 1e-4f && Parameters::WEIGHT_DECAY_INPUT == 1e-4f &&
              Parameters::WEIGHT_DECAY_OUTPUT == 1e-4f && Parameters::BIAS_LR_FACTOR == 1);

void check(double actual, double expected, double tolerance, const char* message) {
    if (!(std::abs(actual - expected) <= tolerance)) throw std::runtime_error(message);
}

int main() {
    Model actor{};
    Adam optimizer;
    Model::Gradients gradient{};
    actor.gates[0].input[0][0] = 2;
    actor.gates[0].recurrent[0][0] = 3;
    actor.W_out[0][0] = -4;
    actor.b_out[0] = 1;
    actor.b_out[1] = 2;
    gradient.gates[0].input[0][0] = 2;
    gradient.bias[1] = 1e-8f;

    const auto first = optimizer.apply(actor, gradient);
    check(optimizer.updates, 1, 0, "first age");
    check(optimizer.first[0], .2, 1e-7, "first moment");
    check(optimizer.second[0], .004, 1e-7, "second moment");
    check(actor.gates[0].input[0][0], 1.9989, 1e-7, "first bias correction plus input decay");
    check(actor.gates[0].recurrent[0][0], 2.99985, 2e-7, "zero-gradient recurrent decay");
    check(actor.W_out[0][0], -3.9998, 2e-7, "zero-gradient output decay");
    check(actor.b_out[0], 1, 0, "zero-gradient bias excludes decay");
    check(actor.b_out[1], 1.9995, 1e-7, "bias update with epsilon and unit factor");
    check(first.update_l2, std::sqrt(.0011*.0011 + .00015*.00015 + .0002*.0002 + .0005*.0005),
          3e-7, "actual update norm");

    gradient.gates[0].input[0][0] = -1;
    const double previous = actor.gates[0].input[0][0];
    optimizer.apply(actor, gradient);
    check(optimizer.updates, 2, 0, "second age");
    check(optimizer.first[0], .08, 1e-7, "second first moment");
    check(optimizer.second[0], .004996, 1e-7, "second second moment");
    const double change = .001 * (.08 / .19) / (std::sqrt(.004996 / .001999) + 1e-8)
                          + previous * .0001 / 2;
    check(actor.gates[0].input[0][0], previous - change, 1e-7, "second bias correction");

    const auto saved_first = optimizer.first;
    const auto saved_second = optimizer.second;
    const auto saved_actor = actor;
    gradient.gates[0].input[0][0] = std::numeric_limits<float>::infinity();
    bool rejected = false;
    try { optimizer.apply(actor, gradient); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected || optimizer.first != saved_first || optimizer.second != saved_second ||
        optimizer.updates != 2 || !std::isinf(optimizer.last_gradient[0]))
        throw std::runtime_error("non-finite update must preserve last valid state");
    check(actor.gates[0].input[0][0], saved_actor.gates[0].input[0][0], 0, "failed update changed actor");

    Target target{};
    rnn::update_target(target, actor, 0);
    check(target.gates[0].input[0][0], actor.gates[0].input[0][0], 0, "target parameter copy");
    if (optimizer.first != saved_first || optimizer.second != saved_second || optimizer.updates != 2)
        throw std::runtime_error("target update changed optimizer");
    const std::filesystem::path checkpoint = "checkpoint-test/adam-actor.txt";
    if (!rnn::save_actor(target, checkpoint)) throw std::runtime_error("checkpoint save");
    Target loaded{};
    if (!rnn::load_actor(loaded, checkpoint)) throw std::runtime_error("checkpoint load");
    check(loaded.gates[0].input[0][0], target.gates[0].input[0][0], 0, "checkpoint input weight");
    check(loaded.b_out[1], target.b_out[1], 0, "checkpoint bias");
    std::cout << "GRU Adam numerical, decay, target, rollback and checkpoint tests passed\n";
}
