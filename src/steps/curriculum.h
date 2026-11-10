namespace learning_to_fly {
    namespace steps {
        template <typename CONFIG>
        bool curriculum(TrainingState<CONFIG>& ts){
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            bool updated = false;
            if constexpr(CONFIG::ABLATION_SPEC::ENABLE_CURRICULUM == true) {
                if(ts.step != 0 && ts.step % 100000 == 0 && ts.step != (CONFIG::STEP_LIMIT - 1)){
                    rlt::add_scalar(ts.device, ts.device.logger, "td3/gamma", ts.actor_critic.gamma);
                    rlt::add_scalar(ts.device, ts.device.logger, "td3/target_next_action_noise_std", ts.actor_critic.target_next_action_noise_std);
                    rlt::add_scalar(ts.device, ts.device.logger, "td3/target_next_action_noise_clip", ts.actor_critic.target_next_action_noise_clip);
                    rlt::add_scalar(ts.device, ts.device.logger, "off_policy_runner/exploration_noise", ts.off_policy_runner.parameters.exploration_noise);


                    for(auto& env : ts.off_policy_runner.envs){
                        {
                            T action_weight = env.parameters.mdp.reward.action;
                            action_weight *= 1.4;
                            T action_weight_limit = 1.0;
                            action_weight = action_weight > action_weight_limit ? action_weight_limit : action_weight;
                            env.parameters.mdp.reward.action = action_weight;
                            rlt::add_scalar(ts.device, ts.device.logger, "reward_function/action_weight", action_weight);
                        }
                        {
                            T position_weight = env.parameters.mdp.reward.position;
                            position_weight *= 1.2;
                            T position_weight_limit = 40;
                            position_weight = position_weight > position_weight_limit ? position_weight_limit : position_weight;
                            env.parameters.mdp.reward.position = position_weight;
                            rlt::add_scalar(ts.device, ts.device.logger, "reward_function/position_weight", position_weight);
                        }
                        {
                            T linear_velocity_weight = env.parameters.mdp.reward.linear_velocity;
                            linear_velocity_weight *= 1.4;
                            T linear_velocity_weight_limit = 1;
                            linear_velocity_weight = linear_velocity_weight > linear_velocity_weight_limit ? linear_velocity_weight_limit : linear_velocity_weight;
                            env.parameters.mdp.reward.linear_velocity = linear_velocity_weight;
                            rlt::add_scalar(ts.device, ts.device.logger, "reward_function/linear_velocity_weight", linear_velocity_weight);
                        }
                    }
                    if constexpr(CONFIG::ABLATION_SPEC::RECALCULATE_REWARDS == true){
                        auto& buffer = ts.off_policy_runner.replay_buffers[0];
                        const auto& env = ts.off_policy_runner.envs[0];
                        const TI count = buffer.full ? buffer.CAPACITY : buffer.position;
                        for (TI index = 0; index < count; ++index) {
                            const auto& state = rlt::get(buffer.states, index, 0);
                            const auto& next_state = rlt::get(buffer.next_states, index, 0);
                            const auto action = rlt::row(ts.device, buffer.actions, index);
                            const auto reward = rlt::reward(ts.device, env, state, action, next_state, ts.rng_eval);
                            rlt::set(buffer.rewards, index, 0, reward);
                        }
                        std::cout << "[Curriculum] step " << ts.step
                                  << ", recalculated " << count << " replay rewards" << std::endl;
                    }
                    updated = true;
                }
            }
            if(CONFIG::ABLATION_SPEC::EXPLORATION_NOISE_DECAY == true){
                if(ts.step % 100000 == 0 && ts.step >= 500000){
                    constexpr T noise_decay_base = 0.90;
                    ts.off_policy_runner.parameters.exploration_noise *= noise_decay_base;
                    ts.actor_critic.target_next_action_noise_std *= noise_decay_base;
                    ts.actor_critic.target_next_action_noise_clip *= noise_decay_base;
                }
            }
            return updated;
        }

#ifdef LEARNING_TO_FLY_ACTOR_GRU_SECOND_CURRICULUM_LR
        template <typename CONFIG>
        void curriculum_with_actor_decay(TrainingState<CONFIG>& ts) {
            if (curriculum(ts) && ts.step == 200000) {
                using Parameters = typename CONFIG::OPTIMIZER::PARAMETERS;
                ts.history_adam.alpha = Parameters::ALPHA * static_cast<typename CONFIG::T>(0.1);
                std::cout << "[Actor Adam LR] step " << ts.step << ", alpha "
                          << ts.history_adam.alpha << std::endl;
            }
        }
#endif
    }
}

