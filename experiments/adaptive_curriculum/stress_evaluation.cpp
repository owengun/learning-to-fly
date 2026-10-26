#include "../../src/training.h"

#include <highfive/H5File.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace stress_evaluation {
    namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;

    using CONFIG = learning_to_fly::config::Config<learning_to_fly::config::DEFAULT_ABLATION_SPEC>;
    using DEVICE = typename CONFIG::DEVICE;
    using T = typename CONFIG::T;
    using TI = typename CONFIG::TI;
    using ENVIRONMENT = typename CONFIG::ENVIRONMENT_EVALUATION;
    using EVAL_SPEC = typename CONFIG::ABLATION_SPEC_EVAL_INSTANCE;
    using PARAMETER_SET = parameters::environment<T, TI, EVAL_SPEC>;

    constexpr TI N_EPISODES = 1000;
    constexpr TI STEP_LIMIT = CONFIG::ENVIRONMENT_STEP_LIMIT_EVALUATION;

    struct Model {
        std::string label;
        TI seed;
        std::string checkpoint;
    };

    struct Result {
        T return_mean;
        T return_std;
        T length_mean;
        T length_std;
        T mean_position_error;
        T success_rate;
    };

    Result evaluate_scenario(
        DEVICE& device,
        ENVIRONMENT& env,
        typename CONFIG::ACTOR_TYPE& actor,
        typename CONFIG::ACTOR_TYPE::template DoubleBuffer<1>& actor_buffer,
        T force_factor,
        T mass_factor,
        TI model_seed
    ) {
        using ACTION_SPEC = rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>;
        using OBSERVATION_SPEC = rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>;

        rlt::MatrixDynamic<ACTION_SPEC> action;
        rlt::MatrixDynamic<OBSERVATION_SPEC> observation;
        rlt::MatrixDynamic<OBSERVATION_SPEC> observation_normalized;
        rlt::MatrixDynamic<OBSERVATION_SPEC> observation_mean;
        rlt::MatrixDynamic<OBSERVATION_SPEC> observation_std;
        rlt::malloc(device, action);
        rlt::malloc(device, observation);
        rlt::malloc(device, observation_normalized);
        rlt::malloc(device, observation_mean);
        rlt::malloc(device, observation_std);
        rlt::set_all(device, observation_mean, (T)0);
        rlt::set_all(device, observation_std, (T)1);

        const auto nominal = PARAMETER_SET::parameters;
        env.parameters = nominal;
        env.parameters.disturbances.random_force.std = nominal.disturbances.random_force.std * force_factor;
        env.parameters.dynamics.mass = nominal.dynamics.mass * mass_factor;

        T return_sum = 0;
        T return_squared_sum = 0;
        T length_sum = 0;
        T length_squared_sum = 0;
        T position_error_sum = 0;
        TI position_error_count = 0;
        TI successes = 0;

        typename DEVICE::SPEC::RANDOM random_device;
        bool ui = false;
        for(TI episode = 0; episode < N_EPISODES; episode++) {
            // Re-seeding each episode gives baseline and adaptive checkpoints
            // the same initial-state and disturbance draws for paired tests.
            const TI episode_seed = 1000000 * (model_seed + 1) + episode;
            auto rng = rlt::random::default_engine(random_device, episode_seed);
            typename ENVIRONMENT::State state;
            rlt::sample_initial_state(device, env, state, rng);
            T episode_return = 0;
            TI episode_length = 0;

            for(TI step = 0; step < STEP_LIMIT; step++) {
                rlt::observe(device, env, state, observation, rng);
                rlt::normalize(device, observation_mean, observation_std, observation, observation_normalized);
                rlt::evaluate(device, actor, observation_normalized, action, actor_buffer);
                for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++) {
                    rlt::set(action, 0, action_i, rlt::math::clamp<T>(device.math, rlt::get(action, 0, action_i), -1, 1));
                }

                typename ENVIRONMENT::State next_state;
                rlt::step(device, env, state, action, next_state, rng);
                episode_return += rlt::reward(device, env, state, action, next_state, rng);
                const T squared_position = next_state.position[0] * next_state.position[0]
                                         + next_state.position[1] * next_state.position[1]
                                         + next_state.position[2] * next_state.position[2];
                position_error_sum += rlt::math::sqrt(device.math, squared_position);
                position_error_count++;
                episode_length++;
                state = next_state;
                if(rlt::terminated(device, env, state, rng)) {
                    break;
                }
            }

            if(episode_length == STEP_LIMIT) {
                successes++;
            }
            return_sum += episode_return;
            return_squared_sum += episode_return * episode_return;
            length_sum += episode_length;
            length_squared_sum += episode_length * episode_length;
        }

        const T return_mean = return_sum / N_EPISODES;
        const T length_mean = length_sum / N_EPISODES;
        Result result{
            return_mean,
            rlt::math::sqrt(device.math, return_squared_sum / N_EPISODES - return_mean * return_mean),
            length_mean,
            rlt::math::sqrt(device.math, length_squared_sum / N_EPISODES - length_mean * length_mean),
            position_error_sum / position_error_count,
            (T)successes / N_EPISODES
        };

        rlt::free(device, action);
        rlt::free(device, observation);
        rlt::free(device, observation_normalized);
        rlt::free(device, observation_mean);
        rlt::free(device, observation_std);
        return result;
    }
}

int main(int argc, char** argv) {
    using namespace stress_evaluation;
    if(argc < 5 || (argc - 2) % 3 != 0) {
        std::cerr << "Usage: stress_evaluation OUTPUT.csv LABEL SEED CHECKPOINT [LABEL SEED CHECKPOINT ...]\n";
        return 2;
    }

    std::vector<Model> models;
    for(int arg = 2; arg < argc; arg += 3) {
        models.push_back({argv[arg], static_cast<TI>(std::stoi(argv[arg + 1])), argv[arg + 2]});
    }

    constexpr T FORCE_FACTORS[] = {(T)1.0, (T)1.5, (T)2.0, (T)3.0};
    constexpr T MASS_FACTORS[] = {(T)1.0, (T)1.05, (T)1.10, (T)1.20};

    DEVICE device;
    ENVIRONMENT env;
    bool ui = false;
    typename CONFIG::ACTOR_TYPE actor;
    typename CONFIG::ACTOR_TYPE::template DoubleBuffer<1> actor_buffer;
    rlt::malloc(device, env);
    rlt::init(device, env, ui);
    rlt::malloc(device, actor);
    rlt::malloc(device, actor_buffer);

    std::ofstream output(argv[1]);
    output << "model,seed,force_factor,mass_factor,return_mean,return_std,episode_length_mean,episode_length_std,mean_position_error,success_rate\n";
    output << std::setprecision(10);

    for(const auto& model: models) {
        std::cout << "Loading " << model.label << " seed " << model.seed << " from " << model.checkpoint << std::endl;
        auto data_file = HighFive::File(model.checkpoint, HighFive::File::ReadOnly);
        rlt::load(device, actor, data_file.getGroup("actor"));

        for(const T force_factor: FORCE_FACTORS) {
            for(const T mass_factor: MASS_FACTORS) {
                const auto result = evaluate_scenario(device, env, actor, actor_buffer, force_factor, mass_factor, model.seed);
                output << model.label << ',' << model.seed << ',' << force_factor << ',' << mass_factor << ','
                       << result.return_mean << ',' << result.return_std << ','
                       << result.length_mean << ',' << result.length_std << ','
                       << result.mean_position_error << ',' << result.success_rate << '\n';
                output.flush();
                std::cout << model.label << " seed " << model.seed
                          << " force x" << force_factor << " mass x" << mass_factor
                          << ": length " << result.length_mean
                          << ", success " << result.success_rate
                          << ", position error " << result.mean_position_error << std::endl;
            }
        }
    }

    rlt::free(device, actor_buffer);
    rlt::free(device, actor);
    return 0;
}
