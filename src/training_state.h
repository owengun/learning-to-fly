#include <queue>
#include <vector>
#include <mutex>
#include "../experiments/rnn/matched_mlp_actor.h"
#include "../experiments/rnn/gated_actor.h"
#ifdef LEARNING_TO_FLY_ACTOR_SEQUENCE_ADAM
#include "../experiments/rnn/actor_adam.h"
#endif
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
#if defined(LEARNING_TO_FLY_ACTOR_RNN_FULL)
        using RecurrentActor = rnn::GatedActor<false, 146, 64, 4, true>;
        static_assert(RecurrentActor::parameter_count == 13764);
#elif defined(LEARNING_TO_FLY_ACTOR_LSTM_FULL)
        using RecurrentActor = rnn::GatedActor<true, 146, 21>;
        static_assert(RecurrentActor::parameter_count == 14200);
#elif defined(LEARNING_TO_FLY_ACTOR_FULL_OBSERVATION)
        using RecurrentActor = rnn::GatedActor<false, 146, 26>;
        static_assert(RecurrentActor::parameter_count == 13602);
#elif defined(LEARNING_TO_FLY_ACTOR_SEQUENCE)
        using RecurrentActor = rnn::GatedActor<false, 22>;
#elif defined(LEARNING_TO_FLY_ACTOR_LSTM)
        using RecurrentActor = rnn::GatedActor<true>;
#elif defined(LEARNING_TO_FLY_ACTOR_GRU)
        using RecurrentActor = rnn::GatedActor<false>;
#else
        using RecurrentActor = rnn::RNNActor<>;
#endif
#ifdef LEARNING_TO_FLY_ACTOR_MATCHED_MLP
        using ExperimentActor = rnn::MatchedMLPActor;
#else
        using ExperimentActor = RecurrentActor;
#endif
        ExperimentActor rnn_actor;
        ExperimentActor rnn_actor_target;
#ifdef LEARNING_TO_FLY_ACTOR_SEQUENCE_ADAM
        rnn::ActorAdam<ExperimentActor, typename CONFIG::OPTIMIZER::PARAMETERS> history_adam;
#endif
    };
}
