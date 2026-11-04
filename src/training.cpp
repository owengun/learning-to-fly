#include "training.h"

#include <charconv>
#include <chrono>
#include <string_view>
#include "original_mlp_checkpoint.h"

template <typename CONFIG>
void evaluate(learning_to_fly::TrainingState<CONFIG>& ts, unsigned episodes) {
    using TI = typename CONFIG::TI;
    using T = typename CONFIG::T;
    using Environment = typename CONFIG::ENVIRONMENT;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, Environment::OBSERVATION_DIM>> observation;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, Environment::ACTION_DIM>> action;
    rlt::malloc(ts.device, observation);
    rlt::malloc(ts.device, action);

#ifndef LEARNING_TO_FLY_EXPERIMENTAL_RNN
    typename CONFIG::ACTOR_TYPE::template DoubleBuffer<1> actor_buffer;
    rlt::malloc(ts.device, actor_buffer);
#endif
    auto& env = ts.env_eval;
    auto& rng = ts.rng_eval;
    double total = 0;
    for (unsigned episode = 0; episode < episodes; ++episode) {
#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
        ts.rnn_actor.reset();
#endif
        typename Environment::State state, next_state;
        rlt::sample_initial_state(ts.device, env, state, rng);
        double episode_return = 0;
        for (TI t = 0; t < CONFIG::ENVIRONMENT_STEP_LIMIT_EVALUATION; ++t) {
            rlt::observe(ts.device, env, state, observation, rng);
#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
            auto output = ts.rnn_actor.forward(
                learning_to_fly::rnn::extract_rltools_observation(observation)
            );
            for (TI j = 0; j < Environment::ACTION_DIM; ++j)
                rlt::set(action, 0, j, output[j]);
#else
            rlt::evaluate(ts.device, ts.actor_critic.actor, observation, action, actor_buffer);
#endif
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
#ifndef LEARNING_TO_FLY_EXPERIMENTAL_RNN
    rlt::free(ts.device, actor_buffer);
#endif
}

int main(int argc, char** argv) {
    using CONFIG = learning_to_fly::config::Config<learning_to_fly::config::DEFAULT_ABLATION_SPEC>;
    using TI = typename CONFIG::TI;
    TI steps = CONFIG::STEP_LIMIT;
    unsigned seed = 0;
    unsigned episodes = 10;
    std::string checkpoint;
    bool evaluation_only = false;
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
            if ((option == "--evaluate"
#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
                 || option == "--warm-start"
#endif
                ) && checkpoint.empty()) {
                checkpoint = value;
                evaluation_only = option == "--evaluate";
                continue;
            }
            if (option == "--episodes" && parse(value, episodes) && episodes > 0) continue;
        }
        std::cerr << "Usage: " << argv[0] << " [--seed N] [--steps N]"
#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
                  << " [--warm-start FILE | --evaluate FILE [--episodes N]]"
#else
                  << " [--evaluate FILE] [--episodes N]"
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
    if (!checkpoint.empty()) {
#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
        const bool loaded = learning_to_fly::rnn::load_actor(ts.rnn_actor, checkpoint);
#else
        const bool loaded = learning_to_fly::load_original_mlp(ts.actor_critic.actor, checkpoint);
#endif
        if (!loaded) {
            std::cerr << "Invalid or incompatible actor checkpoint: " << checkpoint << '\n';
            learning_to_fly::destroy(ts);
            return 1;
        }
        if (evaluation_only) {
            evaluate(ts, episodes);
            learning_to_fly::destroy(ts);
            return 0;
        }
#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
        ts.rnn_actor_target = ts.rnn_actor;
        std::cout << "Warm-started actor from " << checkpoint << '\n';
#endif
    }

    for (TI i = 0; i < steps; ++i) {
#ifndef LEARNING_TO_FLY_EXPERIMENTAL_RNN
        if (i % 10000 == 0 || i + 1 == steps) {
            const std::filesystem::path directory = "checkpoints/mlp-original";
            const bool saved_actor = learning_to_fly::save_original_mlp(
                ts.actor_critic.actor, directory / ("actor_step_" + std::to_string(i) + ".txt"));
            const bool saved_target = learning_to_fly::save_original_mlp(
                ts.actor_critic.actor_target, directory / ("target_step_" + std::to_string(i) + ".txt"));
            if (!saved_actor || !saved_target) {
                std::cerr << "Original MLP checkpoint write failed at step " << i << '\n';
                learning_to_fly::destroy(ts);
                return 1;
            }
        }
#endif
        learning_to_fly::step(ts, i + 1 == steps);
    }
    learning_to_fly::destroy(ts);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Training took: " << std::chrono::duration<double>(end - start).count() << "s\n";
    return 0;
}
