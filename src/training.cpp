#include "training.h"

#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string_view>
#ifdef LEARNING_TO_FLY_ACTOR_GRU_ADAM
#include <thread>
#endif
#include "original_mlp_checkpoint.h"

template <typename CONFIG>
void evaluate(learning_to_fly::TrainingState<CONFIG>& ts, unsigned episodes,
              std::ostream* trajectory) {
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
    if (trajectory) {
        *trajectory << "episode,step,dt_s,reward,terminated,time_limit,"
                       "x_m,y_m,z_m,qw,qx,qy,qz,vx_m_s,vy_m_s,vz_m_s,"
                       "wx_rad_s,wy_rad_s,wz_rad_s,rpm_0,rpm_1,rpm_2,rpm_3,"
                       "action_0,action_1,action_2,action_3";
#if defined(LEARNING_TO_FLY_ACTOR_GRU) || defined(LEARNING_TO_FLY_ACTOR_GRU_HISTORY)
        *trajectory << ",hidden_l2,update_mean,reset_mean,update_saturated,reset_saturated,candidate_saturated";
#endif
        *trajectory << '\n' << std::setprecision(std::numeric_limits<T>::max_digits10);
    }
    double total = 0;
    for (unsigned episode = 0; episode < episodes; ++episode) {
#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
        ts.rnn_actor.reset();
#endif
        typename Environment::State state, next_state;
        rlt::sample_initial_state(ts.device, env, state, rng);
        double episode_return = 0;
#ifdef LEARNING_TO_FLY_ACTOR_GRU_HISTORY
        std::array<float, 4> previous_action{};
#endif
        for (TI t = 0; t < CONFIG::ENVIRONMENT_STEP_LIMIT_EVALUATION; ++t) {
            rlt::observe(ts.device, env, state, observation, rng);
#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
#ifdef LEARNING_TO_FLY_ACTOR_GRU_HISTORY
            auto input = learning_to_fly::rnn::extract_rltools_observation(observation);
            learning_to_fly::rnn::HistoryInput augmented{};
            std::copy(input.begin(), input.end(), augmented.begin());
            std::copy(previous_action.begin(), previous_action.end(), augmented.begin() + 18);
            const auto frame = ts.rnn_actor.advance(augmented, ts.rnn_actor.hidden);
            ts.rnn_actor.hidden = frame.state;
            const auto output = ts.rnn_actor.action(frame.state.h);
            previous_action = output;  // Evaluation disables action exploration and action noise.
#elif defined(LEARNING_TO_FLY_ACTOR_GRU)
            auto input = learning_to_fly::rnn::extract_rltools_observation(observation);
            const auto frame = ts.rnn_actor.advance(input, ts.rnn_actor.hidden);
            ts.rnn_actor.hidden = frame.state;
            const auto output = ts.rnn_actor.action(frame.state.h);
#else
            const auto output = ts.rnn_actor.forward(
                learning_to_fly::rnn::extract_rltools_observation(observation));
#endif
            for (TI j = 0; j < Environment::ACTION_DIM; ++j)
                rlt::set(action, 0, j, output[j]);
#else
            rlt::evaluate(ts.device, ts.actor_critic.actor, observation, action, actor_buffer);
#endif
            const T dt = rlt::step(ts.device, env, state, action, next_state, rng);
            const T reward = rlt::reward(ts.device, env, state, action, next_state, rng);
            episode_return += reward;
            const bool terminated = rlt::terminated(ts.device, env, next_state, rng);
            if (trajectory) {
                *trajectory << episode + 1 << ',' << t + 1 << ',' << dt << ','
                            << reward << ',' << terminated << ','
                            << (!terminated && t + 1 == CONFIG::ENVIRONMENT_STEP_LIMIT_EVALUATION);
                for (T value : next_state.position) *trajectory << ',' << value;
                for (T value : next_state.orientation) *trajectory << ',' << value;
                for (T value : next_state.linear_velocity) *trajectory << ',' << value;
                for (T value : next_state.angular_velocity) *trajectory << ',' << value;
                for (T value : next_state.rpm) *trajectory << ',' << value;
                for (TI j = 0; j < Environment::ACTION_DIM; ++j)
                    *trajectory << ',' << rlt::get(action, 0, j);
#if defined(LEARNING_TO_FLY_ACTOR_GRU) || defined(LEARNING_TO_FLY_ACTOR_GRU_HISTORY)
                double hidden_sq = 0, update = 0, reset = 0;
                std::size_t update_sat = 0, reset_sat = 0, candidate_sat = 0;
                const auto n = frame.state.h.size();
                for (std::size_t i = 0; i < n; ++i) {
                    hidden_sq += frame.state.h[i] * frame.state.h[i];
                    update += frame.gate[0][i];
                    reset += frame.gate[1][i];
                    update_sat += frame.gate[0][i] <= 0.05f || frame.gate[0][i] >= 0.95f;
                    reset_sat += frame.gate[1][i] <= 0.05f || frame.gate[1][i] >= 0.95f;
                    candidate_sat += std::abs(frame.gate[2][i]) >= 0.95f;
                }
                *trajectory << ',' << std::sqrt(hidden_sq) << ',' << update / n << ','
                            << reset / n << ',' << static_cast<double>(update_sat) / n << ','
                            << static_cast<double>(reset_sat) / n << ','
                            << static_cast<double>(candidate_sat) / n;
#endif
                *trajectory << '\n';
            }
            if (terminated) break;
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

#ifdef LEARNING_TO_FLY_ACTOR_GRU_ADAM
bool await_screen_decision(const std::filesystem::path& directory) {
    {
        std::ofstream ready(directory / "ready");
        ready << "100000\n";
        ready.close();
        if (!ready) throw std::runtime_error("Cannot write screening gate marker");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
    while (std::chrono::steady_clock::now() < deadline) {
        std::ifstream decision(directory / "decision");
        std::string command;
        if (decision >> command) {
            if (command == "continue") return true;
            if (command == "stop") return false;
            throw std::runtime_error("Invalid screening decision");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("Timed out awaiting screening decision");
}
#endif

int main(int argc, char** argv) {
    using CONFIG = learning_to_fly::config::Config<learning_to_fly::config::DEFAULT_ABLATION_SPEC>;
    using TI = typename CONFIG::TI;
    TI steps = CONFIG::STEP_LIMIT;
    unsigned seed = 0;
    unsigned episodes = 10;
    std::string checkpoint;
    std::string trajectory_path;
    bool evaluation_only = false;
#ifdef LEARNING_TO_FLY_ACTOR_GRU_ADAM
    std::filesystem::path screen_gate;
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
            if ((option == "--evaluate"
#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
                 || option == "--warm-start"
#endif
                ) && checkpoint.empty()) {
                checkpoint = value;
                evaluation_only = option == "--evaluate";
                continue;
            }
#ifdef LEARNING_TO_FLY_ACTOR_GRU_ADAM
            if (option == "--screen-gate" && screen_gate.empty()) {
                screen_gate = value;
                continue;
            }
#endif
            if (option == "--episodes" && parse(value, episodes) && episodes > 0) continue;
            if (option == "--trajectory-csv" && trajectory_path.empty()) {
                trajectory_path = value;
                continue;
            }
        }
        std::cerr << "Usage: " << argv[0] << " [--seed N] [--steps N]"
#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
                  << " [--warm-start FILE | --evaluate FILE [--episodes N]]"
#else
                  << " [--evaluate FILE] [--episodes N]"
#endif
                  << " [--trajectory-csv FILE]"
#ifdef LEARNING_TO_FLY_ACTOR_GRU_ADAM
                  << " [--screen-gate DIRECTORY]"
#endif
                  << '\n';
        return 2;
    }
    if (!trajectory_path.empty() && !evaluation_only) {
        std::cerr << "--trajectory-csv requires --evaluate\n";
        return 2;
    }

#ifdef LEARNING_TO_FLY_ACTOR_GRU_ADAM
    if (!screen_gate.empty() &&
        (steps <= 100001 || !checkpoint.empty() || !std::filesystem::is_directory(screen_gate) ||
         std::filesystem::exists(screen_gate / "ready") ||
         std::filesystem::exists(screen_gate / "decision"))) {
        std::cerr << "--screen-gate requires a fresh gate directory and a new run beyond 100001 steps\n";
        return 2;
    }
#endif
    auto start = std::chrono::high_resolution_clock::now();
    learning_to_fly::TrainingState<CONFIG> ts;
#ifdef LEARNING_TO_FLY_EXPERIMENTAL_RNN
    std::cout << "Actor " << decltype(ts.rnn_actor)::name;
#else
    std::cout << "Actor mlp";
#endif
    std::cout << " seed " << CONFIG::BASE_SEED + seed << " steps " << steps << '\n';
    learning_to_fly::init(ts, static_cast<TI>(seed));
#ifdef LEARNING_TO_FLY_ACTOR_GRU_ADAM
    std::cout << "GRU Actor optimizer Adam, using the original MLP parameter configuration\n";
#endif
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
            std::ofstream trajectory;
            if (!trajectory_path.empty()) {
                trajectory.open(trajectory_path);
                if (!trajectory) {
                    std::cerr << "Cannot open trajectory CSV: " << trajectory_path << '\n';
                    learning_to_fly::destroy(ts);
                    return 1;
                }
            }
            evaluate(ts, episodes, trajectory.is_open() ? &trajectory : nullptr);
            if (trajectory.is_open()) trajectory.close();
            learning_to_fly::destroy(ts);
            if (!trajectory) std::cerr << "Trajectory CSV write failed\n";
            return trajectory ? 0 : 1;
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
#ifdef LEARNING_TO_FLY_ACTOR_GRU_ADAM
        try {
            learning_to_fly::step(ts, i + 1 == steps);
            if (i == 100000 && !screen_gate.empty() && !await_screen_decision(screen_gate)) break;
        } catch (const std::exception& error) {
            std::cerr << "GRU Adam stopped at step " << i << " - " << error.what() << '\n';
            const auto directory = std::filesystem::path("failure") / ("step_" + std::to_string(i));
            try {
                std::filesystem::create_directories(directory);
                std::ofstream metadata(directory / "reason.txt");
                metadata << error.what() << '\n';
                metadata.flush();
                const bool actor_saved = learning_to_fly::rnn::save_actor(ts.rnn_actor, directory / "actor.txt");
                const bool target_saved = learning_to_fly::rnn::save_actor(ts.rnn_actor_target, directory / "target.txt");
                const bool optimizer_saved = ts.history_adam.save_diagnostic_state(directory / "actor_adam.csv");
                if (!metadata || !actor_saved || !target_saved || !optimizer_saved)
                    std::cerr << "Failure snapshot write incomplete\n";
            } catch (const std::exception& snapshot_error) {
                std::cerr << "Failure snapshot write failed - " << snapshot_error.what() << '\n';
            }
            learning_to_fly::destroy(ts);
            return 1;
        }
#else
        learning_to_fly::step(ts, i + 1 == steps);
#endif
    }
#ifdef LEARNING_TO_FLY_ACTOR_GRU_ADAM
    std::cout << "Training finished at step " << ts.step << '\n';
#endif
    learning_to_fly::destroy(ts);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Training took: " << std::chrono::duration<double>(end - start).count() << "s\n";
    return 0;
}
