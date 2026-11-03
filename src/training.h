
#define LEARNING_TO_FLY_EXPERIMENTAL_RNN

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
#include "../experiments/rnn/rnn_environment_step.h"
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
    void step(TrainingState<CONFIG>& ts) {

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
        // 2. Train the original MLP Critics
        // ----------------------------------------

        if (
            ts.step > CONFIG::N_WARMUP_STEPS_CRITIC &&
            ts.step %
                CONFIG::TD3_PARAMETERS::
                    CRITIC_TRAINING_INTERVAL == 0
        ) {

            for (TI critic_i = 0; critic_i < 2; ++critic_i) {

                rlt::target_action_noise(
                    ts.device,
                    ts.actor_critic,
                    ts.critic_training_buffers
                        .target_next_action_noise,
                    ts.rng
                );

                rlt::gather_batch(
                    ts.device,
                    ts.off_policy_runner,
                    ts.critic_batch,
                    ts.rng
                );

                rlt::train_critic(
                    ts.device,
                    ts.actor_critic,

                    critic_i == 0
                        ? ts.actor_critic.critic_1
                        : ts.actor_critic.critic_2,

                    ts.critic_batch,
                    ts.critic_optimizers[critic_i],
                    ts.actor_buffers[critic_i],
                    ts.critic_buffers[critic_i],
                    ts.critic_training_buffers
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

            // Temporary auxiliary MLP training.
            // Existing train_critic() still uses
            // the MLP target Actor.
            rlt::gather_batch(
                ts.device,
                ts.off_policy_runner,
                ts.actor_batch,
                ts.rng
            );

            rlt::train_actor(
                ts.device,
                ts.actor_critic,
                ts.actor_batch,
                ts.actor_optimizer,
                ts.actor_buffers[0],
                ts.critic_buffers[0],
                ts.actor_training_buffers
            );

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
        // 5. Temporary MLP target Actor update
        // ----------------------------------------

        if (
            ts.step > CONFIG::N_WARMUP_STEPS_ACTOR &&
            ts.step %
                CONFIG::TD3_PARAMETERS::
                    ACTOR_TARGET_UPDATE_INTERVAL == 0
        ) {

            rlt::update_actor_target(
                ts.device,
                ts.actor_critic
            );
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
