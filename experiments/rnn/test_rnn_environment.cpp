
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;
#include <learning_to_fly/simulator/operations_cpu.h>
#include "../../src/config/parameters.h"
#include "../../src/config/ablation.h"

#include "observation_adapter.h"

#include <cmath>
#include <iostream>




int main() {
    using T = float;
    using DEVICE = rlt::devices::DefaultCPU;
    using TI = typename DEVICE::index_t;

    using ABLATION_SPEC = parameters::DefaultAblationSpec;

    using ENVIRONMENT_BUILDER =
        parameters::environment<T, TI, ABLATION_SPEC>;

    using ENVIRONMENT = typename ENVIRONMENT_BUILDER::ENVIRONMENT;

    static_assert(ENVIRONMENT::OBSERVATION_DIM == 146);
    static_assert(ENVIRONMENT::ACTION_DIM == 4);

    DEVICE device{};
    ENVIRONMENT env{};

    env.parameters = ENVIRONMENT_BUILDER::parameters;


    auto rng = rlt::random::default_engine(
        typename DEVICE::SPEC::RANDOM{}, 42
    );

    ENVIRONMENT::State state{};
    ENVIRONMENT::State next_state{};

    rlt::sample_initial_state(device, env, state, rng);

    using ObservationSpec = rlt::matrix::Specification<
        float, TI, 1, ENVIRONMENT::OBSERVATION_DIM
    >;

    using ActionSpec = rlt::matrix::Specification<
        float, TI, 1, ENVIRONMENT::ACTION_DIM
    >;

    rlt::MatrixDynamic<ObservationSpec> observation{};
    rlt::MatrixDynamic<ActionSpec> action{};

    rlt::malloc(device, observation);
    rlt::malloc(device, action);

    using Actor = learning_to_fly::rnn::RNNActor<18, 32, 4>;

    Actor actor;
    actor.initialize(42);
    actor.reset();

    std::cout << "Observation dimension: "
              << ENVIRONMENT::OBSERVATION_DIM << "\n";

    std::cout << "Action dimension: "
              << ENVIRONMENT::ACTION_DIM << "\n";

    for (TI t = 0; t < 10; ++t) {
        // Observe the current simulated drone state.
        rlt::observe(device, env, state, observation, rng);

        // Convert 146-dimensional observation to 18 inputs.
        auto input =
            learning_to_fly::rnn::extract_rltools_observation(
                observation
            );

        // Recurrent forward pass keeps hidden state.
        auto output = actor.forward(input);

        for (TI i = 0; i < ENVIRONMENT::ACTION_DIM; ++i) {
            rlt::set(action, 0, i, output[i]);
        }

        // Apply RNN action to the simulated drone.
        rlt::step(device, env, state, action, next_state, rng);

        std::cout << "Step " << t
                  << " position x: " << next_state.position[0]
                  << " hidden[0]: " << actor.hidden[0]
                  << " action[0]: " << output[0]
                  << "\n";

        state = next_state;

        if (rlt::terminated(device, env, state, rng)) {
            std::cout << "Episode terminated\n";
            break;
        }
    }

    rlt::free(device, action);
    rlt::free(device, observation);

    return 0;
}
