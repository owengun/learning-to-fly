#include "../../src/training.h"
#include <stdexcept>

namespace rnn = learning_to_fly::rnn;
using Config = learning_to_fly::config::Config<learning_to_fly::config::DEFAULT_ABLATION_SPEC>;
using Training = learning_to_fly::TrainingState<Config>;
using Actor = decltype(Training::rnn_actor);
using TI = Config::TI;
static_assert(Config::ENVIRONMENT::OBSERVATION_DIM == 146);
static_assert(Config::ENVIRONMENT::ACTION_HISTORY_LENGTH == 32);
#ifdef LEARNING_TO_FLY_ACTOR_RNN_FULL
static_assert(Actor::parameter_count == 13764);
#elif defined(LEARNING_TO_FLY_ACTOR_LSTM_FULL)
static_assert(Actor::parameter_count == 14200);
#else
static_assert(Actor::parameter_count == 13602);
#endif
using Mlp = Config::ACTOR_TYPE;
static_assert(Mlp::INPUT_DIM * 64 + 64 + 64 * 64 + 64 + 64 * Mlp::OUTPUT_DIM + Mlp::OUTPUT_DIM == 13828);

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    Training ts;
    learning_to_fly::init(ts, 0);
    auto& actor = ts.rnn_actor;
    std::size_t count = 0;
    Actor::visit_parameters(actor, [&](float) { ++count; });
    require(count == Actor::parameter_count, "visitor must enumerate every parameter");
    auto& env = ts.env_eval;
    auto& rng = ts.rng_eval;
    Config::ENVIRONMENT::State state{}, next{};
    rlt::MatrixStatic<rlt::matrix::Specification<float, TI, 1, 146>> observation;
    rlt::MatrixStatic<rlt::matrix::Specification<float, TI, 1, 4>> action;
    rlt::malloc(ts.device, observation);
    rlt::malloc(ts.device, action);

    auto begin_episode = [&]() {
        auto expected_rng = rng;
        Config::ENVIRONMENT::State expected{};
        rlt::sample_initial_state(ts.device, env, expected, expected_rng);
        rnn::reset_evaluation_episode(ts.device, env, state, rng, actor);
        require(actor.hidden.h == Actor::Hidden{} && actor.hidden.cell == Actor::Hidden{}, "evaluation resets hidden and cell state");
        for (TI t = 0; t < 32; ++t)
            for (TI i = 0; i < 4; ++i)
                require(state.action_history[t][i] == expected.action_history[t][i],
                        "evaluation resets environment action history like stock MLP");
    };
    begin_episode();
    std::array<Actor::Input, 40> inputs{};
    std::array<Actor::Output, 40> online{};
    for (std::size_t t = 0; t < inputs.size(); ++t) {
        rlt::observe(ts.device, env, state, observation, rng);
        inputs[t] = rnn::full_observation(observation, 0);
        for (TI j = 0; j < inputs[t].size(); ++j)
            require(inputs[t][j] == rlt::get(observation, 0, j), "input must match all MLP columns");
        for (TI k = 0; k < 32; ++k)
            for (TI j = 0; j < 4; ++j)
                require(inputs[t][18 + 4 * k + j] == state.action_history[k][j],
                        "oldest-to-newest environment history order");
        online[t] = actor.forward(inputs[t]);
        for (TI j = 0; j < 4; ++j) rlt::set(action, 0, j, online[t][j]);
        rlt::step(ts.device, env, state, action, next, rng);
        for (TI j = 0; j < 4; ++j) {
            require(next.action_history[0][j] == state.action_history[1][j], "shift oldest action");
            require(next.action_history[31][j] == online[t][j], "append actual action once");
        }
        state = next;
    }

    rnn::HistoryWindowFor<Actor::Input> window{};
    window.inputs = inputs;
    window.burn = 32;
    window.learn = 8;
    auto hidden = rnn::burn_in(actor, window);
    for (std::size_t t = window.burn; t < inputs.size(); ++t) {
        hidden = actor.advance(inputs[t], hidden).state;
        require(actor.action(hidden.h) == online[t], "online and replay learn window agree");
    }

    const std::filesystem::path checkpoint = std::filesystem::path("checkpoint-test") / (std::string(Actor::name) + ".txt");
    require(rnn::save_actor(actor, checkpoint), "save full-input checkpoint");
    Actor loaded{};
    require(rnn::load_actor(loaded, checkpoint), "load full-input checkpoint");
    require(loaded.forward(inputs[0]) == online[0], "roundtrip same initial input and output");
    begin_episode();
    rlt::observe(ts.device, env, state, observation, rng);
    const auto input = rnn::full_observation(observation, 0);
    loaded.reset();
    require(loaded.forward(input) == actor.forward(input),
            "new episode starts from reinitialized history and hidden state");
#ifdef LEARNING_TO_FLY_ACTOR_MATCHED_RECURRENT
    for (int episode = 0; episode < 2; ++episode) {
        actor.hidden.h.fill(0.7f);
        actor.hidden.cell.fill(1.3f);
        rlt::set(ts.off_policy_runner.truncated, 0, 0, true);
        rnn::rnn_environment_step(ts, actor);
        const auto& replay = ts.off_policy_runner.replay_buffers[0];
        const auto row = (replay.position + replay.CAPACITY - 1) % replay.CAPACITY;
        const auto observation_input = rnn::full_observation(replay.observations, row);
        const auto expected = actor.advance(observation_input, Actor::State{}).state;
        require(actor.hidden.h == expected.h && actor.hidden.cell == expected.cell,
                "normal online episode entry resets hidden and cell before observing");
    }
#endif
    learning_to_fly::destroy(ts);
    std::cout << "146 input columns, episode resets, history order, replay and checkpoint passed\n";
}
