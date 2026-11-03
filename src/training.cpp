
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

void dispatch_run(int num_points, int interval_ms, TI seed, TI max_steps){
    if (num_points == 0) {
        run<TrajectoryAblationSpec<0>>(seed, max_steps);
        return;
    }
    auto run_interval = [&](auto spec_tag){
        using SPEC = decltype(spec_tag);
        run<SPEC>(seed, max_steps);
    };
    if (num_points == 3) {
        switch(interval_ms){
            case 100: run_interval(TrajectoryAblationSpec<3, 100>{}); return;
            case 500: run_interval(TrajectoryAblationSpec<3, 500>{}); return;
            case 1000: run_interval(TrajectoryAblationSpec<3, 1000>{}); return;
            case 1500: run_interval(TrajectoryAblationSpec<3, 1500>{}); return;
            default:
                std::cerr << "Unsupported interval: " << interval_ms << "ms (supported: 100, 500, 1000, 1500)" << std::endl;
                return;
        }
    }
    if (num_points == 1) {
        switch(interval_ms){
            case 100: run_interval(TrajectoryAblationSpec<1, 100>{}); return;
            case 500: run_interval(TrajectoryAblationSpec<1, 500>{}); return;
            case 1000: run_interval(TrajectoryAblationSpec<1, 1000>{}); return;
            case 1500: run_interval(TrajectoryAblationSpec<1, 1500>{}); return;
            default: run_interval(TrajectoryAblationSpec<1, 100>{}); return;
        }
    }
    if (num_points == 2) {
        switch(interval_ms){
            case 100: run_interval(TrajectoryAblationSpec<2, 100>{}); return;
            case 500: run_interval(TrajectoryAblationSpec<2, 500>{}); return;
            case 1000: run_interval(TrajectoryAblationSpec<2, 1000>{}); return;
            case 1500: run_interval(TrajectoryAblationSpec<2, 1500>{}); return;
            default: run_interval(TrajectoryAblationSpec<2, 100>{}); return;
        }
    }
    if (num_points == 4) {
        switch(interval_ms){
            case 100: run_interval(TrajectoryAblationSpec<4, 100>{}); return;
            case 500: run_interval(TrajectoryAblationSpec<4, 500>{}); return;
            case 1000: run_interval(TrajectoryAblationSpec<4, 1000>{}); return;
            case 1500: run_interval(TrajectoryAblationSpec<4, 1500>{}); return;
            default: run_interval(TrajectoryAblationSpec<4, 100>{}); return;
        }
    }
    std::cerr << "Unsupported num_points: " << num_points << " (supported: 0, 1, 2, 3, 4)" << std::endl;
}

int main(int argc, char** argv){
    TI seed = 0;
    TI max_steps = 0;
    std::string arg1 = (argc >= 2) ? std::string(argv[1]) : "intervals";

    if(arg1 == "intervals"){
        if(argc >= 3) seed = std::stoi(argv[2]);
        if(argc >= 4) max_steps = std::stoi(argv[3]);
        dispatch_run(3, 100, seed, max_steps);
        dispatch_run(3, 500, seed, max_steps);
        dispatch_run(3, 1000, seed, max_steps);
        dispatch_run(3, 1500, seed, max_steps);
        return 0;
    }
    if(arg1 == "all"){
        if(argc >= 3) seed = std::stoi(argv[2]);
        if(argc >= 4) max_steps = std::stoi(argv[3]);
        dispatch_run(0, 0, seed, max_steps);
        dispatch_run(3, 100, seed, max_steps);
        dispatch_run(3, 500, seed, max_steps);
        dispatch_run(3, 1000, seed, max_steps);
        dispatch_run(3, 1500, seed, max_steps);
        return 0;
    }

    int num_points = std::stoi(arg1);
    int interval_ms = 100;
    int arg_idx = 2;
    if(argc >= 3 && num_points > 0){
        // check if argv[2] is interval
        int val = std::stoi(argv[2]);
        if(val == 100 || val == 500 || val == 1000 || val == 1500){
            interval_ms = val;
            arg_idx = 3;
        }
    }
    if(argc > arg_idx) seed = std::stoi(argv[arg_idx++]);
    if(argc > arg_idx) max_steps = std::stoi(argv[arg_idx++]);

    dispatch_run(num_points, interval_ms, seed, max_steps);
    return 0;
}