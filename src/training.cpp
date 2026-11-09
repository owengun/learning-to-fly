
#include "training.h"

#include <chrono>
#include <string>

using learning_to_fly::config::TrajectoryAblationSpec;

template <typename T_ABLATION_SPEC>
void run(typename learning_to_fly::config::Config<T_ABLATION_SPEC>::TI seed_offset = 0, typename learning_to_fly::config::Config<T_ABLATION_SPEC>::TI max_steps = 0){
    using namespace learning_to_fly::config;

    using CONFIG = learning_to_fly::config::Config<T_ABLATION_SPEC>;
    using TI = typename CONFIG::TI;

#ifdef LEARNING_TO_FLY_IN_SECONDS_BENCHMARK
    constexpr TI NUM_RUNS = 1;
#else
    constexpr TI NUM_RUNS = 1;
#endif

    TI total_steps = (max_steps > 0 && max_steps <= CONFIG::STEP_LIMIT) ? max_steps : CONFIG::STEP_LIMIT;
    if (total_steps > 0 && total_steps % 100000 == 0) {
        total_steps += 1;
    }

    for (TI run_i = 0; run_i < NUM_RUNS; run_i++){
        TI seed = seed_offset + run_i;
        std::cout << "Run " << seed << " (trajectory points: " << T_ABLATION_SPEC::NUM_TRAJECTORY_POINTS;
        if constexpr (T_ABLATION_SPEC::NUM_TRAJECTORY_POINTS > 0) {
            double dt = (double)T_ABLATION_SPEC::TRAJECTORY_INTERVAL_MS / 1000.0;
            std::cout << ", static interval: " << dt << "s, offsets: [";
            for(int p = 0; p < T_ABLATION_SPEC::NUM_TRAJECTORY_POINTS; p++){
                std::cout << (p == 0 ? "" : ", ") << (p + 1) * dt << "s";
            }
            std::cout << "]";
        }
        std::cout << ", steps: " << total_steps << ")\n";

        auto start = std::chrono::high_resolution_clock::now();
        learning_to_fly::TrainingState<CONFIG> ts;
        learning_to_fly::init(ts, seed);

        for(TI step_i=0; step_i < total_steps; step_i++){
            learning_to_fly::step(ts);
        }

        learning_to_fly::destroy(ts);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Training took: " << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << "s" << std::endl;
    }
}

using TI = typename learning_to_fly::config::Config<learning_to_fly::config::DEFAULT_ABLATION_SPEC>::TI;

template<int N, int INTERVAL>
void dispatch_reward(int rew, TI seed, TI max_steps) {
    switch(rew) {
        case 0: run<TrajectoryAblationSpec<N, INTERVAL, 0>>(seed, max_steps); break;
        case 1: run<TrajectoryAblationSpec<N, INTERVAL, 1>>(seed, max_steps); break;
        case 2: run<TrajectoryAblationSpec<N, INTERVAL, 2>>(seed, max_steps); break;
        default: run<TrajectoryAblationSpec<N, INTERVAL, 2>>(seed, max_steps); break;
    }
}

template<int N>
void dispatch_interval(int interval, int rew, TI seed, TI max_steps) {
    switch(interval) {
        case 50: dispatch_reward<N, 50>(rew, seed, max_steps); break;
        case 100: dispatch_reward<N, 100>(rew, seed, max_steps); break;
        case 200: dispatch_reward<N, 200>(rew, seed, max_steps); break;
        case 500: dispatch_reward<N, 500>(rew, seed, max_steps); break;
        default: dispatch_reward<N, 100>(rew, seed, max_steps); break;
    }
}

void dispatch_run(int num_points, int interval_ms, int reward_type, TI seed, TI max_steps){
    if (num_points == 0) {
        dispatch_reward<0, 0>(reward_type, seed, max_steps);
        return;
    }
    switch(num_points){
        case 1: dispatch_interval<1>(interval_ms, reward_type, seed, max_steps); break;
        case 2: dispatch_interval<2>(interval_ms, reward_type, seed, max_steps); break;
        case 3: dispatch_interval<3>(interval_ms, reward_type, seed, max_steps); break;
        default:
            std::cerr << "Unsupported num_points: " << num_points << " (supported: 0, 1, 2, 3)" << std::endl;
            break;
    }
}

int main(int argc, char** argv){
    TI seed = 0;
    TI max_steps = 0;
    int num_points = 0;
    int interval_ms = 100;
    int reward_type = 2;

    if(argc >= 2) num_points = std::stoi(argv[1]);
    if(argc >= 3) interval_ms = std::stoi(argv[2]);
    if(argc >= 4) reward_type = std::stoi(argv[3]);
    if(argc >= 5) seed = std::stoi(argv[4]);
    if(argc >= 6) max_steps = std::stoi(argv[5]);

    dispatch_run(num_points, interval_ms, reward_type, seed, max_steps);
    return 0;
}