
#include <rl_tools/operations/cpu_mux.h>
#include "sequence_sampler.h"

#include <array>
#include <iostream>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;

int main() {
    using DEVICE = rlt::devices::DefaultCPU;
    DEVICE device;

    constexpr std::size_t CAPACITY = 6;
    constexpr std::size_t SEQ_LEN = 3;

    using OBS_SPEC = rlt::matrix::Specification<
        float, DEVICE::index_t, CAPACITY, 18
    >;

    using FLAG_SPEC = rlt::matrix::Specification<
        bool, DEVICE::index_t, CAPACITY, 1
    >;

    struct FakeReplayBuffer {
        rlt::MatrixDynamic<OBS_SPEC> observations;
        rlt::MatrixDynamic<FLAG_SPEC> terminated;
        rlt::MatrixDynamic<FLAG_SPEC> truncated;
        std::size_t position = 0;
        bool full = false;
    };

    FakeReplayBuffer rb;

    rlt::malloc(device, rb.observations);
    rlt::malloc(device, rb.terminated);
    rlt::malloc(device, rb.truncated);

    for (std::size_t i = 0; i < CAPACITY; ++i) {
        for (std::size_t j = 0; j < 18; ++j) {
            rlt::set(rb.observations, i, j,
                     static_cast<float>(i));
        }
        rlt::set(rb.terminated, i, 0, false);
        rlt::set(rb.truncated, i, 0, false);
    }

    std::array<std::array<float, 18>, SEQ_LEN> sequence{};

    // Test 1: chronological order before buffer is full.
    rb.position = 5;
    rb.full = false;

    bool ok = learning_to_fly::rnn::
        extract_sequence<SEQ_LEN, CAPACITY>(rb, 2, sequence);

    if (!ok || sequence[0][0] != 0 ||
        sequence[1][0] != 1 ||
        sequence[2][0] != 2) {
        std::cerr << "Chronological test FAILED\n";
        return 1;
    }

    // Test 2: reject an episode boundary.
    rlt::set(rb.truncated, 1, 0, true);

    if (learning_to_fly::rnn::
        extract_sequence<SEQ_LEN, CAPACITY>(rb, 2, sequence)) {
        std::cerr << "Episode boundary test FAILED\n";
        return 1;
    }

    rlt::set(rb.truncated, 1, 0, false);

    // Termination is allowed on the sampled transition, but not in its history.
    rlt::set(rb.terminated, 2, 0, true);
    if (!learning_to_fly::rnn::extract_sequence<SEQ_LEN, CAPACITY>(rb, 2, sequence) ||
        learning_to_fly::rnn::extract_sequence<SEQ_LEN, CAPACITY>(rb, 3, sequence)) {
        std::cerr << "Terminal transition alignment FAILED\n";
        return 1;
    }
    rlt::set(rb.terminated, 2, 0, false);

    // Test 3: full circular buffer, position = oldest slot.
    rb.full = true;
    rb.position = 2;

    ok = learning_to_fly::rnn::
        extract_sequence<SEQ_LEN, CAPACITY>(rb, 5, sequence);

    if (!ok || sequence[0][0] != 5 ||
        sequence[1][0] != 0 ||
        sequence[2][0] != 1) {
        std::cerr << "Circular buffer test FAILED\n";
        return 1;
    }

    rlt::free(device, rb.observations);
    rlt::free(device, rb.terminated);
    rlt::free(device, rb.truncated);

    std::cout << "Sequence sampler PASSED\n";
    return 0;
}
