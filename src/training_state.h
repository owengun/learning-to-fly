#include <queue>
#include <vector>
#include <mutex>
#include "../experiments/rnn/matched_mlp_actor.h"
#include "../experiments/rnn/gated_actor.h"
namespace learning_to_fly{
    template <typename T_CONFIG>
    struct TrainingState: rlt::rl::algorithms::td3::loop::TrainingState<T_CONFIG>{
        using CONFIG = T_CONFIG;
        std::string run_name;
        std::queue<std::vector<typename CONFIG::ENVIRONMENT::State>> trajectories;
        std::mutex trajectories_mutex;
        std::vector<typename CONFIG::ENVIRONMENT::State> episode;
        // validation
        rlt::rl::utils::validation::Task<typename CONFIG::TASK_SPEC> task;
        typename CONFIG::ENVIRONMENT validation_envs[CONFIG::VALIDATION_N_EPISODES];
        typename CONFIG::ACTOR_TYPE::template DoubleBuffer<CONFIG::VALIDATION_N_EPISODES> validation_actor_buffers;
        using RecurrentActor = rnn::GatedActor<false>;
#ifdef LEARNING_TO_FLY_ACTOR_MATCHED_MLP
        using ExperimentActor = rnn::MatchedMLPActor;
#else
        using ExperimentActor = RecurrentActor;
#endif
        ExperimentActor rnn_actor;
        ExperimentActor rnn_actor_target;
    };
}
