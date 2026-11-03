#ifndef LEARNING_TO_FLY_ACTOR_MLP
#define LEARNING_TO_FLY_EXPERIMENTAL_RNN
#endif
#include <iomanip>
#include <string>
#include <cstddef>
#include <array>
#include <cmath>
#include <limits>
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;

#include <learning_to_fly/simulator/operations_cpu.h>
#include <learning_to_fly/simulator/metrics.h>

#include "config/config.h"

#include <rl_tools/rl/algorithms/td3/loop.h>

#include "training_state.h"

#include "steps/checkpoint.h"
#include "steps/critic_reset.h"
#include "steps/curriculum.h"
#include "steps/log_reward.h"
#include "steps/logger.h"
#include "steps/trajectory_collection.h"
#include "steps/validation.h"

#include "helpers.h"

#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
#include "../experiments/rnn/rnn_td3_actor_update.h"
#include "../experiments/rnn/rnn_td3_critic_update.h"
#include "../experiments/rnn/rnn_environment_step.h"
#include "../experiments/rnn/actor_checkpoint.h"
#endif

#include <filesystem>
#include <fstream>
#include <iostream>

namespace learning_to_fly {

    // ============================================
    // Initialize training
    // ============================================

    template <typename T_CONFIG>
    void init(
        TrainingState<T_CONFIG>& ts,
        typename T_CONFIG::TI seed = 0
    ) {
        using CONFIG = T_CONFIG;
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        using ABLATION_SPEC = typename CONFIG::ABLATION_SPEC;

        auto env_parameters =
            parameters::environment<
                T, TI, ABLATION_SPEC
            >::parameters;

        auto env_parameters_eval =
            parameters::environment<
                T,
                TI,
                config::template ABLATION_SPEC_EVAL<ABLATION_SPEC>
            >::parameters;

        for (auto& env : ts.envs) {
            env.parameters = env_parameters;
        }

        ts.env_eval.parameters = env_parameters_eval;

        TI effective_seed = CONFIG::BASE_SEED + seed;

        ts.run_name =
            helpers::run_name<ABLATION_SPEC, CONFIG>(
                effective_seed
            );

        rlt::construct(
            ts.device,
            ts.device.logger,
            std::string("logs"),
            ts.run_name
        );

        rlt::set_step(ts.device, ts.device.logger, 0);

        rlt::add_scalar(
            ts.device,
            ts.device.logger,
            "loop/seed",
            effective_seed
        );

        // Initialize original TD3 components,
        // including both MLP critics.
        rlt::rl::algorithms::td3::loop::init(
            ts,
            effective_seed
        );

        ts.off_policy_runner.parameters =
            CONFIG::off_policy_runner_parameters;

#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN

        // Initialize persistent RNN Actor.
        ts.rnn_actor.initialize(effective_seed);
        ts.rnn_actor.reset();

        // Target Actor starts with identical parameters.
        ts.rnn_actor_target = ts.rnn_actor;
        ts.rnn_actor_target.reset();

        std::cout
            << "[RNN] Actor initialized"
            << std::endl;

#endif

        for (typename CONFIG::ENVIRONMENT& env :
             ts.validation_envs) {

            env.parameters =
                parameters::environment<
                    T, TI, ABLATION_SPEC
                >::parameters;
        }

        rlt::malloc(
            ts.device,
            ts.validation_actor_buffers
        );

        rlt::init(
            ts.device,
            ts.task,
            ts.validation_envs,
            ts.rng_eval
        );

        std::cout << "Environment Info:\n";

        std::cout
            << "\tObservation dim: "
            << CONFIG::ENVIRONMENT::OBSERVATION_DIM
            << std::endl;

        std::cout
            << "\tObservation dim privileged: "
            << CONFIG::ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED
            << std::endl;

        std::cout
            << "\tAction dim: "
            << CONFIG::ENVIRONMENT::ACTION_DIM
            << std::endl;
    }


    // ============================================
    // Training step
    // ============================================

    template <typename CONFIG>
    void step(TrainingState<CONFIG>& ts, bool save_final_checkpoint = false) {

        using TI = typename CONFIG::TI;

        if (ts.step % 10000 == 0) {
            std::cout
                << "Step: "
                << ts.step
                << std::endl;
        }

#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN

        // Keep the logger's step synchronized.
        rlt::set_step(
            ts.device,
            ts.device.logger,
            ts.step
        );

        // ----------------------------------------
        // 1. Environment interaction using RNN
        // ----------------------------------------

        rnn::rnn_environment_step(
            ts,
            ts.rnn_actor
        );

        // ----------------------------------------
        // Training episode statistics (exploration noise ON).
        // These are NOT noise-free validation returns.
        // ----------------------------------------
        {
            constexpr std::size_t EPISODE_WINDOW = 100;
            constexpr std::size_t CAPACITY =
                decltype(ts.off_policy_runner)::REPLAY_BUFFER_SPEC::CAPACITY;

            // This experiment currently uses a single environment.
            static double episode_return = 0.0;
            static std::size_t episode_length = 0;
            static std::size_t episode_count = 0;
            static std::array<double, EPISODE_WINDOW> recent_returns{};
            static std::array<double, EPISODE_WINDOW> recent_lengths{};
            static std::size_t recent_count = 0;
            static std::size_t recent_next = 0;
            static std::size_t last_report_step = 0;

            const auto& rb = ts.off_policy_runner.replay_buffers[0];
            if (rb.position != 0 || rb.full) {
                const std::size_t latest =
                    (static_cast<std::size_t>(rb.position) + CAPACITY - 1)
                    % CAPACITY;
                const double reward = static_cast<double>(
                    rlt::get(rb.rewards, latest, 0));
                const bool terminated =
                    static_cast<bool>(rlt::get(rb.terminated, latest, 0));
                const bool truncated =
                    static_cast<bool>(rlt::get(rb.truncated, latest, 0));

                episode_return += reward;
                ++episode_length;

                if (terminated || truncated) {
                    ++episode_count;
                    recent_returns[recent_next] = episode_return;
                    recent_lengths[recent_next] =
                        static_cast<double>(episode_length);
                    recent_next = (recent_next + 1) % EPISODE_WINDOW;
                    if (recent_count < EPISODE_WINDOW) ++recent_count;

                    const std::filesystem::path csv_path =
                        std::string("checkpoints/") + decltype(ts.rnn_actor)::name + "/training_episodes.csv";
                    std::error_code dir_error;
                    std::filesystem::create_directories(
                        csv_path.parent_path(), dir_error);
                    if (!dir_error) {
                        const bool write_header =
                            !std::filesystem::exists(csv_path);
                        std::ofstream csv(csv_path, std::ios::app);
                        if (csv) {
                            if (write_header) {
                                csv << "step,episode,return,length,terminated,truncated\n";
                            }
                            csv << ts.step << ',' << episode_count << ','
                                << std::setprecision(10) << episode_return << ','
                                << episode_length << ',' << terminated << ','
                                << truncated << '\n';
                        }
                    }

                    episode_return = 0.0;
                    episode_length = 0;
                }
            }

            // Print the latest completed episode window every 10k steps.
            if (ts.step > 0 && ts.step % 10000 == 0 &&
                last_report_step != ts.step) {
                last_report_step = ts.step;
                if (recent_count == 0) {
                    std::cout << "[RNN Train] No completed episodes yet"
                              << std::endl;
                } else {
                    double sum_returns = 0.0;
                    double sum_lengths = 0.0;
                    for (std::size_t i = 0; i < recent_count; ++i) {
                        sum_returns += recent_returns[i];
                        sum_lengths += recent_lengths[i];
                    }
                    std::cout << "[RNN Train] Step " << ts.step
                              << " | Episodes " << episode_count
                              << " | Recent " << recent_count
                              << " | Mean Return "
                              << sum_returns / recent_count
                              << " | Mean Length "
                              << sum_lengths / recent_count
                              << " (exploration ON)" << std::endl;
                }
            }
        }


        // ----------------------------------------
        // 2. Train the original MLP Critics
        // ----------------------------------------

        if (
            ts.step > CONFIG::N_WARMUP_STEPS_CRITIC &&
            ts.step %
                CONFIG::TD3_PARAMETERS::
                    CRITIC_TRAINING_INTERVAL == 0
        ) {

            constexpr std::size_t sequence_length = 8;
            constexpr auto capacity = CONFIG::OFF_POLICY_RUNNER_SPEC::REPLAY_BUFFER_CAPACITY;
            using RNN = decltype(ts.rnn_actor_target);
            using Sequence = std::array<typename RNN::Input, sequence_length>;
            constexpr auto batch_size = CONFIG::TD3_PARAMETERS::CRITIC_BATCH_SIZE;
            std::array<Sequence, batch_size> current_sequences;
            std::array<Sequence, batch_size> next_sequences;

            for (TI critic_i = 0; critic_i < 2; ++critic_i) {
                if (!rnn::gather_aligned_sequence_batch<sequence_length, capacity>(
                        ts.device, ts.off_policy_runner.replay_buffers[0],
                        ts.critic_batch, current_sequences, next_sequences, ts.rng)) {
                    continue;
                }

                auto& target = ts.rnn_actor_target;
                auto& buffers = ts.critic_training_buffers;
                for (TI row = 0; row < batch_size; ++row) {
                    target.reset();
                    typename RNN::Output action{};
                    for (const auto& observation : next_sequences[row]) {
                        action = target.forward(observation);
                    }
                    for (TI j = 0; j < CONFIG::ENVIRONMENT::ACTION_DIM; ++j) {
                        rlt::set(buffers.next_actions, row, j, action[j]);
                    }
                }

                rlt::target_action_noise(
                    ts.device, ts.actor_critic, buffers.target_next_action_noise, ts.rng
                );
                rnn::train_critic_with_rnn_targets(
                    ts.device, ts.actor_critic,
                    critic_i == 0 ? ts.actor_critic.critic_1 : ts.actor_critic.critic_2,
                    ts.critic_batch, ts.critic_optimizers[critic_i],
                    ts.critic_buffers[critic_i], buffers
                );
            }
        }


        // ----------------------------------------
        // 3. Train the RNN Actor
        // ----------------------------------------

        if (
            ts.step > CONFIG::N_WARMUP_STEPS_ACTOR &&
            ts.step %
                CONFIG::TD3_PARAMETERS::
                    ACTOR_TRAINING_INTERVAL == 0
        ) {

        
            // Actual RNN Actor gradient update.
            bool updated =
                rnn::train_rnn_actor_td3<8>(
                    ts,
                    ts.rnn_actor,
                    0.0001f
                );

            static unsigned long long
                rnn_update_count = 0;

            if (updated) {

                ++rnn_update_count;

                if (
                    rnn_update_count == 1 ||
                    rnn_update_count % 1000 == 0
                ) {

                    std::cout
                        << "[RNN] Update count: "
                        << rnn_update_count
                        << " at step "
                        << ts.step
                        << std::endl;
                }
            }
        }


        // ----------------------------------------
        // 4. Update Critic target networks
        // ----------------------------------------

        if (
            ts.step > CONFIG::N_WARMUP_STEPS_CRITIC &&
            ts.step %
                CONFIG::TD3_PARAMETERS::
                    CRITIC_TARGET_UPDATE_INTERVAL == 0
        ) {

            rlt::update_critic_targets(
                ts.device,
                ts.actor_critic
            );
        }



        // ----------------------------------------
        // 5. Soft-update the TARGET RNN Actor
        // ----------------------------------------

        if (
            ts.step > CONFIG::N_WARMUP_STEPS_ACTOR &&
            ts.step %
                CONFIG::TD3_PARAMETERS::
                    ACTOR_TARGET_UPDATE_INTERVAL == 0
        ) {
            rnn::update_target(ts.rnn_actor_target, ts.rnn_actor,
                               CONFIG::TD3_PARAMETERS::ACTOR_POLYAK);
        }


        // ----------------------------------------
        // Save RNN weights every 10000 steps
        // ----------------------------------------

        if (ts.step % 10000 == 0 ||
            save_final_checkpoint || ts.step + 1 >= CONFIG::STEP_LIMIT) {

            std::string filename =
                std::string("checkpoints/") + decltype(ts.rnn_actor)::name + "/actor_step_" +
                std::to_string(ts.step) +
                ".txt";

            bool saved = rnn::save_actor(
                ts.rnn_actor,
                filename
            );

            if (saved) {
                std::cout
                    << "[RNN] Weights saved at step "
                    << ts.step
                    << std::endl;
            }
        }


        // ----------------------------------------
        // 6. Advance training
        // ----------------------------------------

        // Do not call the original TD3 loop step:
        // it would interact with the environment
        // a second time using the MLP Actor.

        ++ts.step;

        ts.finished =
            ts.step >= CONFIG::STEP_LIMIT;


#else

        // ========================================
        // Original Baseline TD3
        // ========================================

        steps::logger(ts);
        steps::checkpoint(ts);
        steps::validation(ts);
        steps::curriculum(ts);

        rlt::rl::algorithms::td3::loop::step(ts);

        steps::trajectory_collection(ts);

#endif
    }


    // ============================================
    // Destroy training
    // ============================================

    template <typename CONFIG>
    void destroy(TrainingState<CONFIG>& ts) {

        rlt::rl::algorithms::td3::loop::destroy(ts);

        rlt::destroy(
            ts.device,
            ts.task
        );

        rlt::free(
            ts.device,
            ts.validation_actor_buffers
        );
    }

} // namespace learning_to_fly
