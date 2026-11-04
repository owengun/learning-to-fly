#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace learning_to_fly {

template <typename Model, typename Visitor>
void visit_dense_weights(Model& model, Visitor& visitor) {
    auto visit_matrix = [&](auto& matrix) {
        using Matrix = std::remove_cv_t<std::remove_reference_t<decltype(matrix)>>;
        for (typename Matrix::TI row = 0; row < Matrix::ROWS; ++row)
            for (typename Matrix::TI col = 0; col < Matrix::COLS; ++col)
                visitor(matrix, row, col);
    };
    visit_matrix(model.content.weights.parameters);
    visit_matrix(model.content.biases.parameters);
    if constexpr (!std::is_same_v<typename Model::NEXT_MODULE,
                                  rlt::nn_models::sequential::OutputModule>) {
        visit_dense_weights(model.next_module, visitor);
    }
}

template <typename Model>
bool save_original_mlp(const Model& model, const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path);
    stream.precision(std::numeric_limits<typename Model::T>::max_digits10);
    stream << "original-mlp-v1 " << Model::INPUT_DIM << " " << Model::OUTPUT_DIM << "\n";
    auto save = [&](const auto& matrix, auto row, auto col) {
        stream << rlt::get(matrix, row, col) << "\n";
    };
    visit_dense_weights(model, save);
    stream.flush();
    return stream.good();
}

template <typename Model>
bool load_original_mlp(Model& model, const std::filesystem::path& path) {
    std::ifstream stream(path);
    std::string format;
    unsigned input = 0, output = 0;
    if (!(stream >> format >> input >> output) || format != "original-mlp-v1" ||
        input != Model::INPUT_DIM || output != Model::OUTPUT_DIM) return false;
    std::vector<typename Model::T> values;
    bool valid = true;
    auto read = [&](const auto&, auto, auto) {
        typename Model::T value{};
        if (!(stream >> value) || !std::isfinite(value)) valid = false;
        values.push_back(value);
    };
    visit_dense_weights(model, read);
    std::string extra;
    if (!valid || stream >> extra) return false;
    auto position = values.begin();
    auto apply = [&](auto& matrix, auto row, auto col) {
        rlt::set(matrix, row, col, *position++);
    };
    visit_dense_weights(model, apply);
    return true;
}
}
