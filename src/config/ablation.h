namespace learning_to_fly::config{
    struct DEFAULT_ABLATION_SPEC{
        static constexpr bool DISTURBANCE = true;
        static constexpr bool OBSERVATION_NOISE = true;
        static constexpr bool ASYMMETRIC_ACTOR_CRITIC = true;
        static constexpr bool ROTOR_DELAY = true;
        static constexpr bool ACTION_HISTORY = true;
        static constexpr bool ENABLE_CURRICULUM = true;
        static constexpr bool RECALCULATE_REWARDS = true;
        static constexpr bool USE_INITIAL_REWARD_FUNCTION = true;
        static constexpr bool INIT_NORMAL = true;
        static constexpr bool EXPLORATION_NOISE_DECAY = true;
        static constexpr bool TRAJECTORY_TRACKING = true;
        static constexpr int NUM_TRAJECTORY_POINTS = 0;
        static constexpr int TRAJECTORY_INTERVAL_MS = 100;
        static constexpr int REWARD_TYPE = 2; // 0: Quadratic, 1: Absolute, 2: Exponential
        static constexpr int STEP_LIMIT = 3000001;
    };
    template <typename T_ABLATION_SPEC>
    struct ABLATION_SPEC_EVAL: T_ABLATION_SPEC{
        // override everything but ACTION_HISTORY because that changes the observation space
        static constexpr bool DISTURBANCE = true;
        static constexpr bool OBSERVATION_NOISE = true;
        static constexpr bool ASYMMETRIC_ACTOR_CRITIC = true;
        static constexpr bool ROTOR_DELAY = true;
//            static constexpr bool ACTION_HISTORY = {copied from T_ABLATION_SPEC for structural fit};
        static constexpr bool ENABLE_CURRICULUM = true;
        static constexpr bool RECALCULATE_REWARDS = true;
        static constexpr bool USE_INITIAL_REWARD_FUNCTION = false; // Use target reward function as metric
        static constexpr bool INIT_NORMAL = true;
        static constexpr bool EXPLORATION_NOISE_DECAY = true;
    };
    template <int T_NUM_TRAJECTORY_POINTS, int T_TRAJECTORY_INTERVAL_MS = 100, int T_REWARD_TYPE = 2>
    struct TrajectoryAblationSpec: DEFAULT_ABLATION_SPEC{
        static constexpr bool TRAJECTORY_TRACKING = true;
        static constexpr int NUM_TRAJECTORY_POINTS = T_NUM_TRAJECTORY_POINTS;
        static constexpr int TRAJECTORY_INTERVAL_MS = T_TRAJECTORY_INTERVAL_MS;
        static constexpr int REWARD_TYPE = T_REWARD_TYPE;
        static constexpr int STEP_LIMIT = 3000001;
    };
}
