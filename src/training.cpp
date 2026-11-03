#include "training.h"

#include <charconv>
#include <chrono>
#include <string_view>

#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
template <typename CONFIG>
void evaluate(learning_to_fly::TrainingState<CONFIG>& ts, unsigned episodes) {
    using TI = typename CONFIG::TI;
    using T = typename CONFIG::T;
    using Environment = typename CONFIG::ENVIRONMENT;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, Environment::OBSERVATION_DIM>> observation;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, Environment::ACTION_DIM>> action;
    rlt::malloc(ts.device, observation);
    rlt::malloc(ts.device, action);

    auto& env = ts.env_eval;
    auto& rng = ts.rng_eval;
    double total = 0;
    for (unsigned episode = 0; episode < episodes; ++episode) {
        ts.rnn_actor.reset();
        typename Environment::State state, next_state;
        rlt::sample_initial_state(ts.device, env, state, rng);
        double episode_return = 0;
        for (TI t = 0; t < CONFIG::ENVIRONMENT_STEP_LIMIT_EVALUATION; ++t) {
            rlt::observe(ts.device, env, state, observation, rng);
            auto output = ts.rnn_actor.forward(
                learning_to_fly::rnn::extract_rltools_observation(observation)
            );
            for (TI j = 0; j < Environment::ACTION_DIM; ++j)
                rlt::set(action, 0, j, output[j]);
            rlt::step(ts.device, env, state, action, next_state, rng);
            episode_return += rlt::reward(ts.device, env, state, action, next_state, rng);
            if (rlt::terminated(ts.device, env, next_state, rng)) break;
            state = next_state;
        }
        total += episode_return;
        std::cout << "Evaluation episode " << episode + 1 << " return " << episode_return << '\n';
    }
    std::cout << "Evaluation mean return " << total / episodes << '\n';
    rlt::free(ts.device, observation);
    rlt::free(ts.device, action);
}
#endif

int main(int argc, char** argv) {
    using CONFIG = learning_to_fly::config::Config<learning_to_fly::config::DEFAULT_ABLATION_SPEC>;
    using TI = typename CONFIG::TI;
    TI steps = CONFIG::STEP_LIMIT;
    unsigned seed = 0;
#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
    unsigned episodes = 10;
    std::string checkpoint;
    bool evaluation_only = false;
#endif
    auto parse = [](std::string_view value, auto& destination) {
        auto end = value.data() + value.size();
        auto result = std::from_chars(value.data(), end, destination);
        return result.ec == std::errc{} && result.ptr == end;
    };
    for (int i = 1; i < argc; ++i) {
        std::string_view option = argv[i];
        if (i + 1 < argc) {
            std::string_view value = argv[++i];
            if (option == "--seed" && parse(value, seed)) continue;
            if (option == "--steps" && parse(value, steps) && steps > 0 && steps <= CONFIG::STEP_LIMIT)
                continue;
#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
            if ((option == "--evaluate" || option == "--warm-start") && checkpoint.empty()) {
                checkpoint = value;
                evaluation_only = option == "--evaluate";
                continue;
            }
            if (option == "--episodes" && parse(value, episodes) && episodes > 0) continue;
#endif
        }
        std::cerr << "Usage: " << argv[0] << " [--seed N] [--steps N]"
#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
                  << " [--warm-start FILE | --evaluate FILE [--episodes N]]"
#endif
                  << '\n';
        return 2;
    }

    auto start = std::chrono::high_resolution_clock::now();
    learning_to_fly::TrainingState<CONFIG> ts;
#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
    std::cout << "Actor " << decltype(ts.rnn_actor)::name;
#else
    std::cout << "Actor mlp";
#endif
    std::cout << " seed " << CONFIG::BASE_SEED + seed << " steps " << steps << '\n';
    learning_to_fly::init(ts, static_cast<TI>(seed));
#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
    if (!checkpoint.empty()) {
        if (!learning_to_fly::rnn::load_actor(ts.rnn_actor, checkpoint)) {
            std::cerr << "Invalid or incompatible actor checkpoint: " << checkpoint << '\n';
            learning_to_fly::destroy(ts);
            return 1;
        }
        if (evaluation_only) {
            evaluate(ts, episodes);
            learning_to_fly::destroy(ts);
            return 0;
        }
        ts.rnn_actor_target = ts.rnn_actor;
        std::cout << "Warm-started actor from " << checkpoint << '\n';
    }
#endif

    for (TI i = 0; i < steps; ++i)
        learning_to_fly::step(ts, i + 1 == steps);
    learning_to_fly::destroy(ts);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Training took: " << std::chrono::duration<double>(end - start).count() << "s\n";
    return 0;
}
