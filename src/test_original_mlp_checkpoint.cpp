#include "training.h"
#include "original_mlp_checkpoint.h"

int main() {
    using CONFIG = learning_to_fly::config::Config<learning_to_fly::config::DEFAULT_ABLATION_SPEC>;
    learning_to_fly::TrainingState<CONFIG> state;
    learning_to_fly::init(state, 0);
    auto& actor = state.actor_critic.actor;
    auto values = [&]() {
        std::vector<float> result;
        auto capture = [&](const auto& matrix, auto row, auto col) {
            result.push_back(rlt::get(matrix, row, col));
        };
        learning_to_fly::visit_dense_weights(actor, capture);
        return result;
    };
    const auto before = values();
    const std::filesystem::path path = "checkpoint-test/original.txt";
    bool ok = learning_to_fly::save_original_mlp(actor, path);
    rlt::set(actor.content.biases.parameters, 0, 0, 123.0f);
    ok = ok && learning_to_fly::load_original_mlp(actor, path) && values() == before;
    {
        std::ofstream truncated("checkpoint-test/truncated.txt");
        truncated << "original-mlp-v1 146 4\n1\n";
    }
    ok = ok && !learning_to_fly::load_original_mlp(actor, "checkpoint-test/truncated.txt")
         && values() == before;
    learning_to_fly::destroy(state);
    std::cout << "Original MLP checkpoint roundtrip and rejected-load atomicity "
              << (ok ? "passed" : "failed") << '\n';
    return ok ? 0 : 1;
}
