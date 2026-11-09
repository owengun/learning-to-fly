#include "training.h"
#include <highfive/H5File.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <filesystem>
#include <algorithm>

using learning_to_fly::config::TrajectoryAblationSpec;

struct EvaluationSummary {
    int num_points = 0;
    int interval_ms = 0;
    int total_episodes = 0;
    int completed_episodes = 0;
    double avg_mean_pos_error_m = 0;
    double avg_rmse_pos_error_m = 0;
    double avg_max_pos_error_m = 0;
    double avg_mean_vel_error_mps = 0;
    double settled_within_10cm_fraction = 0;
    double settled_within_5cm_fraction = 0;
    double settled_within_2cm_fraction = 0;
    double settled_within_1cm_fraction = 0;
    double settled_within_20cm_fraction = 0;
    double avg_episode_length = 0;
};

template <int N_POINTS, int INTERVAL_MS = 100>
EvaluationSummary evaluate_trajectory_model(const std::string& checkpoint_path, const std::string& csv_path) {
    using namespace learning_to_fly::config;
    using ABLATION_SPEC = TrajectoryAblationSpec<N_POINTS, INTERVAL_MS>;
    using CONFIG = Config<ABLATION_SPEC>;
    using T = typename CONFIG::T;
    using TI = typename CONFIG::TI;

    learning_to_fly::TrainingState<CONFIG> ts;
    learning_to_fly::init(ts, 42); // deterministic seed

    std::cout << "\n=======================================================\n";
    std::cout << "Evaluating Model with " << N_POINTS << " points, interval: " << ((double)INTERVAL_MS / 1000.0) << "s\n";
    std::cout << "Loading checkpoint: " << checkpoint_path << std::endl;

    auto data_file = HighFive::File(checkpoint_path, HighFive::File::ReadOnly);
    rlt::load(ts.device, ts.actor_critic.actor, data_file.getGroup("actor"));

    // Reset task and ensure strictly canonical trajectory parameters
    rlt::reset(ts.device, ts.task, ts.rng_eval);
    constexpr TI N_EPISODES = decltype(ts.task)::SPEC::N_EPISODES;
    constexpr TI MAX_EP_LEN = decltype(ts.task)::SPEC::MAX_EPISODE_LENGTH;
    for(TI ep_i = 0; ep_i < N_EPISODES; ep_i++){
        rlt::initial_state(ts.device, ts.validation_envs[ep_i], ts.task.state[ep_i]);
    }

    bool completed = false;
    while(!completed){
        completed = rlt::step(ts.device, ts.task, ts.actor_critic.actor, ts.validation_actor_buffers, ts.rng_eval);
    }

    EvaluationSummary summary;
    summary.num_points = N_POINTS;
    summary.total_episodes = N_EPISODES;

    std::ofstream csv(csv_path);
    csv << "episode,step,t,pos_x,pos_y,pos_z,target_x,target_y,target_z,pos_error,vel_error\n";

    double total_pos_err_sum = 0;
    double total_pos_sq_err_sum = 0;
    double total_max_pos_err_sum = 0;
    double total_vel_err_sum = 0;
    double total_ep_len = 0;
    long long total_steps_count = 0;
    long long within_1cm_steps_count = 0;
    long long within_2cm_steps_count = 0;
    long long within_5cm_steps_count = 0;
    long long within_10cm_steps_count = 0;
    long long within_20cm_steps_count = 0;

    for(TI ep_i = 0; ep_i < N_EPISODES; ep_i++){
        auto& eb = ts.task.episode_buffer[ep_i];
        TI ep_len = ts.task.episode_length[ep_i];
        total_ep_len += ep_len;
        if(ep_len == MAX_EP_LEN){
            summary.completed_episodes++;
        }

        double ep_max_err = 0;
        for(TI step_i = 0; step_i < ep_len; step_i++){
            auto& state = get(eb.states, step_i, 0);
            T target_pos[3], target_vel[3];
            rlt::rl::environments::multirotor::target_trajectory(ts.device, state, (T)0, target_pos, target_vel);

            double dx = state.position[0] - target_pos[0];
            double dy = state.position[1] - target_pos[1];
            double dz = state.position[2] - target_pos[2];
            double pos_err = std::sqrt(dx*dx + dy*dy + dz*dz);

            double dvx = state.linear_velocity[0] - target_vel[0];
            double dvy = state.linear_velocity[1] - target_vel[1];
            double dvz = state.linear_velocity[2] - target_vel[2];
            double vel_err = std::sqrt(dvx*dvx + dvy*dvy + dvz*dvz);

            total_pos_err_sum += pos_err;
            total_pos_sq_err_sum += pos_err * pos_err;
            total_vel_err_sum += vel_err;
            total_steps_count++;

            if(pos_err < 0.01) within_1cm_steps_count++;
            if(pos_err < 0.02) within_2cm_steps_count++;
            if(pos_err < 0.05) within_5cm_steps_count++;
            if(pos_err < 0.10) within_10cm_steps_count++;
            if(pos_err < 0.20) within_20cm_steps_count++;
            if(pos_err > ep_max_err) ep_max_err = pos_err;

            if(ep_i == 0){ // Log full first trajectory to CSV
                csv << ep_i << "," << step_i << "," << (step_i * ts.validation_envs[0].parameters.integration.dt)
                    << "," << state.position[0] << "," << state.position[1] << "," << state.position[2]
                    << "," << target_pos[0] << "," << target_pos[1] << "," << target_pos[2]
                    << "," << pos_err << "," << vel_err << "\n";
            }
        }
        total_max_pos_err_sum += ep_max_err;
    }
    csv.close();

    summary.avg_episode_length = total_ep_len / N_EPISODES;
    if(total_steps_count > 0){
        summary.avg_mean_pos_error_m = total_pos_err_sum / total_steps_count;
        summary.avg_rmse_pos_error_m = std::sqrt(total_pos_sq_err_sum / total_steps_count);
        summary.avg_mean_vel_error_mps = total_vel_err_sum / total_steps_count;
        summary.settled_within_1cm_fraction = (double)within_1cm_steps_count / total_steps_count;
        summary.settled_within_2cm_fraction = (double)within_2cm_steps_count / total_steps_count;
        summary.settled_within_5cm_fraction = (double)within_5cm_steps_count / total_steps_count;
        summary.settled_within_10cm_fraction = (double)within_10cm_steps_count / total_steps_count;
        summary.settled_within_20cm_fraction = (double)within_20cm_steps_count / total_steps_count;
    }
    summary.avg_max_pos_error_m = total_max_pos_err_sum / N_EPISODES;

    std::cout << "Mean Episode Length: " << summary.avg_episode_length << " / " << MAX_EP_LEN << "\n";
    std::cout << "Completed Episodes (No Crash <5cm): " << summary.completed_episodes << " / " << N_EPISODES
              << " (" << (100.0 * summary.completed_episodes / N_EPISODES) << "%)\n";
    std::cout << "Average Position Error: " << std::fixed << std::setprecision(2) << (summary.avg_mean_pos_error_m * 100.0) << " cm\n";
    std::cout << "RMS Position Error: " << std::fixed << std::setprecision(2) << (summary.avg_rmse_pos_error_m * 100.0) << " cm\n";
    std::cout << "Average Max Position Error: " << std::fixed << std::setprecision(2) << (summary.avg_max_pos_error_m * 100.0) << " cm\n";
    std::cout << "Steps within 5cm: " << std::fixed << std::setprecision(1) << (summary.settled_within_5cm_fraction * 100.0) << "%\n";
    std::cout << "Steps within 2cm: " << std::fixed << std::setprecision(1) << (summary.settled_within_2cm_fraction * 100.0) << "%\n";
    std::cout << "Steps within 1cm: " << std::fixed << std::setprecision(1) << (summary.settled_within_1cm_fraction * 100.0) << "%\n";
    std::cout << "Saved trajectory log to: " << csv_path << std::endl;
    std::cout << std::defaultfloat << std::setprecision(6);
    std::cout << "METRICS_JSON: {\"mean_err_m\": " << summary.avg_mean_pos_error_m
              << ", \"rmse_err_m\": " << summary.avg_rmse_pos_error_m
              << ", \"max_err_m\": " << summary.avg_max_pos_error_m
              << ", \"dwell_5cm\": " << summary.settled_within_5cm_fraction
              << ", \"dwell_2cm\": " << summary.settled_within_2cm_fraction
              << ", \"dwell_1cm\": " << summary.settled_within_1cm_fraction
              << ", \"survival\": " << ((double)summary.completed_episodes / summary.total_episodes)
              << ", \"ep_length\": " << summary.avg_episode_length << "}\n";

    summary.num_points = N_POINTS;
    summary.interval_ms = INTERVAL_MS;
    learning_to_fly::destroy(ts);
    return summary;
}

std::string find_latest_checkpoint(int n_points, int interval_ms = 0) {
    std::string tag;
    if(n_points == 0) tag = "_t0_";
    else if(interval_ms > 0) tag = "_t" + std::to_string(n_points) + "_i" + std::to_string(interval_ms) + "_";
    else tag = "_t" + std::to_string(n_points) + "_";

    std::filesystem::path checkpoint_dir = "checkpoints/multirotor_td3";
    std::vector<std::string> matching_runs;
    for (const auto& entry : std::filesystem::directory_iterator(checkpoint_dir)) {
        if (entry.is_directory() && entry.path().filename().string().find(tag) != std::string::npos) {
            matching_runs.push_back(entry.path().string());
        }
    }
    std::sort(matching_runs.begin(), matching_runs.end());
    if (matching_runs.empty()) return "";

    std::vector<std::string> checkpoints;
    for (const auto& entry : std::filesystem::directory_iterator(matching_runs.back())) {
        if (entry.is_regular_file() && entry.path().extension() == ".h5") {
            checkpoints.push_back(entry.path().string());
        }
    }
    std::sort(checkpoints.begin(), checkpoints.end());
    if (checkpoints.empty()) return "";
    return checkpoints.back();
}

int main(int argc, char** argv) {
    std::filesystem::create_directories("results/trajectories");
    if(argc >= 2) {
        std::string cp = argv[1];
        int points = (argc >= 3) ? std::stoi(argv[2]) : 0;
        int interval = (argc >= 4) ? std::stoi(argv[3]) : 100;
        std::cout << "Evaluating custom checkpoint: " << cp << " (points=" << points << ", interval=" << interval << "ms)\n";
        if(points == 0) evaluate_trajectory_model<0>(cp, "results/trajectories/trajectory_custom.csv");
        else if(points == 1) {
            if(interval == 50) evaluate_trajectory_model<1, 50>(cp, "results/trajectories/trajectory_custom.csv");
            else if(interval == 200) evaluate_trajectory_model<1, 200>(cp, "results/trajectories/trajectory_custom.csv");
            else if(interval == 500) evaluate_trajectory_model<1, 500>(cp, "results/trajectories/trajectory_custom.csv");
            else evaluate_trajectory_model<1, 100>(cp, "results/trajectories/trajectory_custom.csv");
        }
        else if(points == 2) {
            if(interval == 50) evaluate_trajectory_model<2, 50>(cp, "results/trajectories/trajectory_custom.csv");
            else if(interval == 200) evaluate_trajectory_model<2, 200>(cp, "results/trajectories/trajectory_custom.csv");
            else if(interval == 500) evaluate_trajectory_model<2, 500>(cp, "results/trajectories/trajectory_custom.csv");
            else evaluate_trajectory_model<2, 100>(cp, "results/trajectories/trajectory_custom.csv");
        }
        else if(points == 3) {
            if(interval == 50) evaluate_trajectory_model<3, 50>(cp, "results/trajectories/trajectory_custom.csv");
            else if(interval == 200) evaluate_trajectory_model<3, 200>(cp, "results/trajectories/trajectory_custom.csv");
            else if(interval == 500) evaluate_trajectory_model<3, 500>(cp, "results/trajectories/trajectory_custom.csv");
            else evaluate_trajectory_model<3, 100>(cp, "results/trajectories/trajectory_custom.csv");
        }
        return 0;
    }

    std::vector<EvaluationSummary> summaries;
    std::string cp0 = find_latest_checkpoint(0);
    std::string cp1 = find_latest_checkpoint(1, 100);
    std::string cp2 = find_latest_checkpoint(2, 100);
    std::string cp3 = find_latest_checkpoint(3, 100);
    std::string cp_i500 = find_latest_checkpoint(3, 500);
    std::string cp_i1000 = find_latest_checkpoint(3, 1000);
    std::string cp_i1500 = find_latest_checkpoint(3, 1500);

    if(!cp0.empty()) summaries.push_back(evaluate_trajectory_model<0>(cp0, "results/trajectories/trajectory_baseline_t0.csv"));
    if(!cp1.empty()) summaries.push_back(evaluate_trajectory_model<1, 100>(cp1, "results/trajectories/trajectory_points_n1.csv"));
    if(!cp2.empty()) summaries.push_back(evaluate_trajectory_model<2, 100>(cp2, "results/trajectories/trajectory_points_n2.csv"));
    if(!cp3.empty()) summaries.push_back(evaluate_trajectory_model<3, 100>(cp3, "results/trajectories/trajectory_interval_0.1s.csv"));

    std::cout << "\n===================================================================================================================================\n";
    std::cout << "                                            LOOKAHEAD HORIZON COMPARISON (N = 0, 1, 2, 3)                                          \n";
    std::cout << "===================================================================================================================================\n";
    std::cout << "Model (Points N) | Lookahead Offsets       | Ep Length | Survival (<5cm) | Mean Pos Error | RMSE Error | <5cm   | <2cm   | <1cm   \n";
    std::cout << "-----------------+-------------------------+-----------+-----------------+----------------+------------+--------+--------+--------\n";
    for(const auto& s : summaries) {
        std::string model_label = "Baseline (N=0)";
        std::string lookahead_str = "None (0s)";
        if(s.num_points > 0){
            double dt = (double)s.interval_ms / 1000.0;
            std::stringstream ss_lbl, ss_off;
            ss_lbl << "Lookahead (N=" << s.num_points << ")";
            model_label = ss_lbl.str();

            ss_off << "[";
            for(int p = 0; p < s.num_points; p++){
                ss_off << (p == 0 ? "" : ", ") << std::fixed << std::setprecision(1) << (p + 1) * dt << "s";
            }
            ss_off << "]";
            lookahead_str = ss_off.str();
        }

        std::cout << " " << std::setw(15) << std::left << model_label << " | "
                  << std::setw(23) << std::left << lookahead_str << " | "
                  << std::setw(7) << std::right << std::fixed << std::setprecision(1) << s.avg_episode_length << "   | "
                  << std::setw(13) << std::fixed << std::setprecision(1) << (100.0 * s.completed_episodes / s.total_episodes) << "%   | "
                  << std::setw(12) << std::fixed << std::setprecision(2) << (s.avg_mean_pos_error_m * 100.0) << "cm | "
                  << std::setw(8) << std::fixed << std::setprecision(2) << (s.avg_rmse_pos_error_m * 100.0) << "cm | "
                  << std::setw(6) << std::fixed << std::setprecision(1) << (s.settled_within_5cm_fraction * 100.0) << "% | "
                  << std::setw(6) << std::fixed << std::setprecision(1) << (s.settled_within_2cm_fraction * 100.0) << "% | "
                  << std::setw(6) << std::fixed << std::setprecision(1) << (s.settled_within_1cm_fraction * 100.0) << "%\n";
    }
    std::cout << "===================================================================================================================================\n";
    return 0;
}
