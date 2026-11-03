#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace learning_to_fly::rnn {

template <typename Actor>
void update_target(Actor& target, const Actor& source, float polyak) {
    // Blend all parameters, including every gate, without copying live state.
    std::vector<float> source_weights;
    Actor::visit_parameters(source, [&](float value) { source_weights.push_back(value); });
    auto weight = source_weights.begin();
    Actor::visit_parameters(target, [&](float& value) {
        value = polyak * value + (1 - polyak) * *weight++;
    });
}

template <typename Actor>
bool save_actor(const Actor& actor, const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path);
    file.precision(std::numeric_limits<float>::max_digits10);
    file << "recurrent-actor-v1 " << Actor::name << '\n';
    Actor::visit_parameters(actor, [&](float value) { file << value << '\n'; });
    file.flush();
    return file.good();
}

template <typename Actor>
bool load_actor(Actor& actor, const std::filesystem::path& path) {
    std::ifstream file(path);
    std::string format, kind;
    file >> format >> kind;
    if (format == "18" && kind == "32" && std::string(Actor::name) == "vanilla") {
        unsigned output_dim = 0;
        if (!(file >> output_dim) || output_dim != 4) return false;
    } else if (format != "recurrent-actor-v1" || kind != Actor::name) {
        return false;
    }

    Actor loaded;
    bool valid = true;
    Actor::visit_parameters(loaded, [&](float& value) {
        if (!(file >> value) || !std::isfinite(value)) valid = false;
    });
    std::string extra;
    if (!valid || file >> extra) return false;
    loaded.reset();
    actor = loaded;
    return true;
}

}
