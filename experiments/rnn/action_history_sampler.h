#pragma once

#include "sequence_sampler.h"
#include <algorithm>

namespace learning_to_fly::rnn {

constexpr std::size_t HISTORY_BURN_IN = 32;
constexpr std::size_t HISTORY_LEARN = 8;
using HistoryInput = std::array<float, 22>;

template <typename OBSERVATIONS, typename ACTIONS>
HistoryInput history_input(const OBSERVATIONS& observations, std::size_t row,
                           const ACTIONS& actions, std::size_t previous_row,
                           bool episode_start) {
    namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;
    HistoryInput input{};
    for (std::size_t i = 0; i < 18; ++i) input[i] = rlt::get(observations, row, i);
    if (!episode_start) {
        for (std::size_t i = 0; i < 4; ++i)
            input[18 + i] = rlt::get(actions, previous_row, i);
    }
    return input;
}

template <typename OBSERVATIONS>
std::array<float, 146> full_observation(const OBSERVATIONS& observations, std::size_t row) {
    namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;
    static_assert(OBSERVATIONS::COLS == 146);
    std::array<float, 146> input{};
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = rlt::get(observations, row, i);
    return input;
}

template <typename Input>
struct HistoryWindowFor {
    std::array<Input, HISTORY_BURN_IN + HISTORY_LEARN> inputs{};
    std::array<std::size_t, HISTORY_LEARN> transitions{};
    std::size_t burn = 0;
    std::size_t learn = 0;
};

using HistoryWindow = HistoryWindowFor<HistoryInput>;

template <typename REPLAY, typename Input>
bool extract_history_window(const REPLAY& rb, std::size_t end, std::size_t learn_limit,
                            HistoryWindowFor<Input>& window) {
    namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;
    const std::size_t count = rb.full ? rb.CAPACITY : rb.position;
    if (end >= count || learn_limit == 0 || learn_limit > HISTORY_LEARN) return false;
    const std::size_t oldest = rb.full ? rb.position : 0;
    auto physical = [&](std::size_t logical) { return (oldest + logical) % rb.CAPACITY; };
    auto episode_end = [&](std::size_t logical) {
        const auto row = physical(logical);
        return rlt::get(rb.terminated, row, 0) || rlt::get(rb.truncated, row, 0);
    };
    std::size_t first = end;
    while (first > 0 && end - first + 1 < HISTORY_BURN_IN + learn_limit &&
           !episode_end(first - 1)) {
        --first;
    }
    // A window that starts at the oldest wrapped row has an unknown episode prefix.
    if (rb.full && first == 0) return false;
    const auto length = end - first + 1;
    window.learn = std::min(learn_limit, length);
    window.burn = length - window.learn;
    for (std::size_t i = 0; i < length; ++i) {
        const auto logical = first + i;
        if constexpr (std::tuple_size_v<Input> == 146) {
            window.inputs[i] = full_observation(rb.observations, physical(logical));
        } else {
            const bool start = logical == 0 || episode_end(logical - 1);
            window.inputs[i] = history_input(rb.observations, physical(logical), rb.actions,
                                             start ? 0 : physical(logical - 1), start);
        }
        if (i >= window.burn) window.transitions[i - window.burn] = physical(logical);
    }
    return true;
}

template <typename DEVICE, typename REPLAY, typename RNG, typename Input>
bool sample_history_window(DEVICE&, const REPLAY& rb, std::size_t learn_limit,
                           RNG& rng, HistoryWindowFor<Input>& window) {
    namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;
    using TI = typename DEVICE::index_t;
    const TI count = rb.full ? rb.CAPACITY : rb.position;
    if (count == 0) return false;
    for (int attempt = 0; attempt < 128; ++attempt) {
        const TI end = rlt::random::uniform_int_distribution(
            typename DEVICE::SPEC::RANDOM{}, TI(0), TI(count - 1), rng);
        if (extract_history_window(rb, end, learn_limit, window)) return true;
    }
    return false;
}

template <typename Actor>
typename Actor::State burn_in(const Actor& actor, const HistoryWindowFor<typename Actor::Input>& window) {
    typename Actor::State state{};
    for (std::size_t t = 0; t < window.burn; ++t)
        state = actor.advance(window.inputs[t], state).state;
    return state;
}

template <typename Actor, typename REPLAY>
typename Actor::Output next_history_action(const Actor& actor, const REPLAY& rb,
                                          std::size_t row, const typename Actor::State& state) {
    // The history ends in the current observation. Extend it once using the
    // stored next observation and the action actually applied to this transition.
    if constexpr (std::tuple_size_v<typename Actor::Input> == 146) {
        const auto next = full_observation(rb.next_observations, row);
        return actor.action(actor.advance(next, state).state.h);
    } else {
        const auto next = history_input(rb.next_observations, row, rb.actions, row, false);
        return actor.action(actor.advance(next, state).state.h);
    }
}

} // namespace learning_to_fly::rnn
