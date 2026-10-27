#include "../../src/training.h"

#include <chrono>
#include <string>

int main(int argc, char** argv){
    using CONFIG = learning_to_fly::config::Config<learning_to_fly::config::DEFAULT_ABLATION_SPEC>;
    using TI = typename CONFIG::TI;
    TI seed = 0;
    if(argc > 1){
        seed = static_cast<TI>(std::stoi(argv[1]));
    }

    learning_to_fly::TrainingState<CONFIG> ts;
    const auto start = std::chrono::high_resolution_clock::now();
    learning_to_fly::init(ts, seed);
    for(TI step_i = 0; step_i < CONFIG::STEP_LIMIT; step_i++){
        learning_to_fly::step(ts);
    }
    learning_to_fly::destroy(ts);
    const auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Training took: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count()
              << "s" << std::endl;
    return 0;
}
