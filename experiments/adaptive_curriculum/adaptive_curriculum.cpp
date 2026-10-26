#include "../../src/training.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace adaptive_curriculum_experiment {
    namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;

    struct AdaptiveAblationSpec: learning_to_fly::config::DEFAULT_ABLATION_SPEC{
        // The fixed 100k-step curriculum is disabled. This experiment applies
        // reward and robustness changes only after a performance gate passes.
        static constexpr bool ENABLE_CURRICULUM = false;
    };

    template<typename T_CONFIG>
    struct TrainingState: learning_to_fly::TrainingState<T_CONFIG>{
        using CONFIG = T_CONFIG;
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        using RNG = decltype(rlt::random::default_engine(typename CONFIG::DEVICE::SPEC::RANDOM()));

        struct Transition{
            TI step;
            TI from_stage;
            TI to_stage;
            T episode_length_mean;
            T mean_position_error;
        };

        TI curriculum_stage = 0;
        TI reward_stage = 0;
        TI consecutive_passes = 0;
        TI last_transition_step = 0;
        T nominal_mass = 0;
        T base_force_std = 0;
        RNG rng_curriculum;
        std::vector<Transition> transitions;
    };

    template<typename CONFIG>
    void init(TrainingState<CONFIG>& ts, typename CONFIG::TI seed){
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        using ABLATION_SPEC = typename CONFIG::ABLATION_SPEC;

        auto env_parameters = parameters::environment<T, TI, ABLATION_SPEC>::parameters;
        auto env_parameters_eval = parameters::environment<T, TI, learning_to_fly::config::template ABLATION_SPEC_EVAL<ABLATION_SPEC>>::parameters;
        for(auto& env: ts.envs){
            env.parameters = env_parameters;
        }
        ts.env_eval.parameters = env_parameters_eval;

        const TI effective_seed = CONFIG::BASE_SEED + seed;
        ts.run_name = learning_to_fly::helpers::run_name<ABLATION_SPEC, CONFIG>(effective_seed) + "_adaptive_wind_mass";
        rlt::construct(ts.device, ts.device.logger, std::string("logs"), ts.run_name);
        rlt::set_step(ts.device, ts.device.logger, 0);
        rlt::add_scalar(ts.device, ts.device.logger, "loop/seed", effective_seed);
        rlt::rl::algorithms::td3::loop::init(ts, effective_seed);
        ts.off_policy_runner.parameters = CONFIG::off_policy_runner_parameters;

        for(typename CONFIG::ENVIRONMENT& env: ts.validation_envs){
            // Validation stays fixed at the nominal baseline distribution.
            env.parameters = parameters::environment<T, TI, ABLATION_SPEC>::parameters;
        }
        rlt::malloc(ts.device, ts.validation_actor_buffers);
        rlt::init(ts.device, ts.task, ts.validation_envs, ts.rng_eval);

        ts.curriculum_stage = 0;
        ts.reward_stage = 0;
        ts.consecutive_passes = 0;
        ts.last_transition_step = 0;
        ts.nominal_mass = env_parameters.dynamics.mass;
        ts.base_force_std = env_parameters.disturbances.random_force.std;
        ts.rng_curriculum = rlt::random::default_engine(typename CONFIG::DEVICE::SPEC::RANDOM(), effective_seed + 1000003);

        std::cout << "Adaptive wind/mass curriculum" << std::endl;
        std::cout << "Seed: " << effective_seed << std::endl;
        std::cout << "Nominal mass: " << ts.nominal_mass << " kg" << std::endl;
        std::cout << "Base force std: " << ts.base_force_std << " N" << std::endl;
    }

    template<typename CONFIG>
    typename CONFIG::T mean_position_error(TrainingState<CONFIG>& ts){
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        T sum = 0;
        TI count = 0;
        for(TI episode_i = 0; episode_i < CONFIG::VALIDATION_N_EPISODES; episode_i++){
            auto& episode_buffer = ts.task.episode_buffer[episode_i];
            for(TI step_i = 0; step_i < ts.task.episode_length[episode_i]; step_i++){
                auto& state = rlt::get(episode_buffer.states, step_i, 0);
                const T squared = state.position[0] * state.position[0]
                                + state.position[1] * state.position[1]
                                + state.position[2] * state.position[2];
                sum += rlt::math::sqrt(ts.device.math, squared);
                count++;
            }
        }
        return count > 0 ? sum / count : std::numeric_limits<T>::infinity();
    }

    template<typename CONFIG>
    void apply_stage(TrainingState<CONFIG>& ts){
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        constexpr T WIND_FACTORS[] = {(T)1.0, (T)1.5, (T)2.0, (T)3.0};
        constexpr T MASS_MAX_FACTORS[] = {(T)1.0, (T)1.05, (T)1.10, (T)1.20};

        const TI stage = ts.curriculum_stage;
        for(auto& env: ts.off_policy_runner.envs){
            env.parameters.disturbances.random_force.std = ts.base_force_std * WIND_FACTORS[stage];
        }

        // Match the original reward curriculum for the first two promotions.
        // Stage 3 changes robustness only, so the final reward target is not
        // made harder than the fixed baseline.
        const TI target_reward_stage = stage < 2 ? stage : 2;
        bool reward_changed = false;
        while(ts.reward_stage < target_reward_stage){
            for(auto& env: ts.off_policy_runner.envs){
                env.parameters.mdp.reward.action = rlt::math::min(ts.device.math, env.parameters.mdp.reward.action * (T)1.4, (T)1.0);
                env.parameters.mdp.reward.position = rlt::math::min(ts.device.math, env.parameters.mdp.reward.position * (T)1.2, (T)40.0);
                env.parameters.mdp.reward.linear_velocity = rlt::math::min(ts.device.math, env.parameters.mdp.reward.linear_velocity * (T)1.4, (T)1.0);
            }
            ts.reward_stage++;
            reward_changed = true;
        }
        if(reward_changed && (ts.off_policy_runner.replay_buffers[0].position > 0 || ts.off_policy_runner.replay_buffers[0].full)){
            rlt::recalculate_rewards(ts.device, ts.off_policy_runner.replay_buffers[0], ts.off_policy_runner.envs[0], ts.rng_eval);
        }

        rlt::add_scalar(ts.device, ts.device.logger, "adaptive_curriculum/stage", stage);
        rlt::add_scalar(ts.device, ts.device.logger, "adaptive_curriculum/wind_force_std", ts.base_force_std * WIND_FACTORS[stage]);
        rlt::add_scalar(ts.device, ts.device.logger, "adaptive_curriculum/mass_factor_min", (T)1.0);
        rlt::add_scalar(ts.device, ts.device.logger, "adaptive_curriculum/mass_factor_max", MASS_MAX_FACTORS[stage]);
    }

    template<typename CONFIG>
    void randomize_mass_on_episode_reset(TrainingState<CONFIG>& ts){
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        constexpr T MASS_MAX_FACTORS[] = {(T)1.0, (T)1.05, (T)1.10, (T)1.20};
        typename CONFIG::DEVICE::SPEC::RANDOM random_device;
        for(TI env_i = 0; env_i < CONFIG::N_ENVIRONMENTS; env_i++){
            if(rlt::get(ts.off_policy_runner.truncated, 0, env_i)){
                const T factor = rlt::random::uniform_real_distribution(
                    random_device,
                    (T)1.0,
                    MASS_MAX_FACTORS[ts.curriculum_stage],
                    ts.rng_curriculum
                );
                ts.off_policy_runner.envs[env_i].parameters.dynamics.mass = ts.nominal_mass * factor;
                rlt::add_scalar(ts.device, ts.device.logger, "adaptive_curriculum/episode_mass_factor", factor, 1000);
            }
        }
    }

    template<typename CONFIG>
    void adaptive_validation_and_gate(TrainingState<CONFIG>& ts){
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        constexpr TI VALIDATION_INTERVAL = 25000;
        constexpr TI MIN_STAGE_DWELL = 50000;
        constexpr TI REQUIRED_CONSECUTIVE_PASSES = 2;
        constexpr TI MAX_STAGE = 3;
        constexpr T LENGTH_THRESHOLDS[] = {(T)100, (T)300, (T)450};
        constexpr T POSITION_ERROR_THRESHOLDS[] = {(T)0.45, (T)0.35, (T)0.25};

        if(ts.step % VALIDATION_INTERVAL != 0){
            return;
        }

        rlt::reset(ts.device, ts.task, ts.rng_eval);
        bool completed = false;
        while(!completed){
            completed = rlt::step(ts.device, ts.task, ts.actor_critic.actor, ts.validation_actor_buffers, ts.rng_eval);
        }
        rlt::analyse_log(ts.device, ts.task, typename TrainingState<CONFIG>::SPEC::METRICS{});

        const T episode_length_mean = rlt::evaluate(
            ts.device,
            rlt::rl::utils::validation::metrics::EpisodeLengthMean{},
            ts.task
        );
        const T position_error = mean_position_error(ts);
        rlt::add_scalar(ts.device, ts.device.logger, "adaptive_curriculum/gate_episode_length", episode_length_mean);
        rlt::add_scalar(ts.device, ts.device.logger, "adaptive_curriculum/gate_mean_position_error", position_error);

        if(ts.curriculum_stage >= MAX_STAGE){
            return;
        }

        const bool dwell_satisfied = ts.step - ts.last_transition_step >= MIN_STAGE_DWELL;
        const bool performance_satisfied = episode_length_mean >= LENGTH_THRESHOLDS[ts.curriculum_stage]
                                           && position_error <= POSITION_ERROR_THRESHOLDS[ts.curriculum_stage];
        if(dwell_satisfied && performance_satisfied){
            ts.consecutive_passes++;
        }
        else{
            ts.consecutive_passes = 0;
        }
        rlt::add_scalar(ts.device, ts.device.logger, "adaptive_curriculum/consecutive_passes", ts.consecutive_passes);

        if(ts.consecutive_passes >= REQUIRED_CONSECUTIVE_PASSES){
            const TI old_stage = ts.curriculum_stage;
            ts.curriculum_stage++;
            ts.consecutive_passes = 0;
            ts.last_transition_step = ts.step;
            ts.transitions.push_back({ts.step, old_stage, ts.curriculum_stage, episode_length_mean, position_error});
            apply_stage(ts);
            std::cout << "Curriculum transition " << old_stage << " -> " << ts.curriculum_stage
                      << " at step " << ts.step
                      << " (episode length " << episode_length_mean
                      << ", mean position error " << position_error << ")" << std::endl;
        }
    }

    template<typename CONFIG>
    void step(TrainingState<CONFIG>& ts){
        if(ts.step % 10000 == 0){
            std::cout << "Step: " << ts.step << ", adaptive stage: " << ts.curriculum_stage << std::endl;
        }
        rlt::set_step(ts.device, ts.device.logger, ts.step);
        learning_to_fly::steps::checkpoint<CONFIG>(ts);
        adaptive_validation_and_gate(ts);
        randomize_mass_on_episode_reset(ts);
        rlt::rl::algorithms::td3::loop::step(ts);
        learning_to_fly::steps::trajectory_collection<CONFIG>(ts);
    }

    template<typename CONFIG>
    void save_results(TrainingState<CONFIG>& ts, double elapsed_seconds){
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;

        std::ofstream transitions(std::string("curriculum_transitions_") + ts.run_name + ".csv");
        transitions << "step,from_stage,to_stage,episode_length_mean,mean_position_error\n";
        for(const auto& transition: ts.transitions){
            transitions << transition.step << ',' << transition.from_stage << ',' << transition.to_stage << ','
                        << transition.episode_length_mean << ',' << transition.mean_position_error << '\n';
        }

        std::ofstream summary(std::string("run_summary_") + ts.run_name + ".csv");
        summary << "run_name,elapsed_seconds,final_stage,transition_count\n";
        summary << ts.run_name << ',' << std::setprecision(10) << elapsed_seconds << ','
                << ts.curriculum_stage << ',' << ts.transitions.size() << '\n';

#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
        auto data_file = HighFive::File(std::string("learning_curves_") + ts.run_name + ".h5", HighFive::File::Overwrite);
        std::vector<TI> steps;
        std::vector<T> returns_mean;
        std::vector<T> returns_std;
        std::vector<T> episode_length_mean;
        std::vector<T> episode_length_std;
        for(TI eval_i = 0; eval_i < TrainingState<CONFIG>::N_EVALUATIONS; eval_i++){
            steps.push_back(eval_i * CONFIG::EVALUATION_INTERVAL);
            returns_mean.push_back(ts.evaluation_results[eval_i].returns_mean);
            returns_std.push_back(ts.evaluation_results[eval_i].returns_std);
            episode_length_mean.push_back(ts.evaluation_results[eval_i].episode_length_mean);
            episode_length_std.push_back(ts.evaluation_results[eval_i].episode_length_std);
        }
        data_file.createDataSet("step", steps);
        data_file.createDataSet("returns_mean", returns_mean);
        data_file.createDataSet("returns_std", returns_std);
        data_file.createDataSet("episode_length_mean", episode_length_mean);
        data_file.createDataSet("episode_length_std", episode_length_std);
#endif
    }
}

int main(int argc, char** argv){
    using CONFIG = learning_to_fly::config::Config<adaptive_curriculum_experiment::AdaptiveAblationSpec>;
    using TI = typename CONFIG::TI;
    TI seed = 0;
    if(argc > 1){
        seed = static_cast<TI>(std::stoi(argv[1]));
    }

    adaptive_curriculum_experiment::TrainingState<CONFIG> ts;
    const auto start = std::chrono::high_resolution_clock::now();
    adaptive_curriculum_experiment::init(ts, seed);
    adaptive_curriculum_experiment::apply_stage(ts);
    for(TI step_i = 0; step_i < CONFIG::STEP_LIMIT; step_i++){
        adaptive_curriculum_experiment::step(ts);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    adaptive_curriculum_experiment::save_results(ts, elapsed_seconds);
    learning_to_fly::destroy<CONFIG>(ts);
    std::cout << "Training took: " << elapsed_seconds << "s" << std::endl;
    return 0;
}

