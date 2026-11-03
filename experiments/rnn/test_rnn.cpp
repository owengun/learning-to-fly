#include "rnn_actor.h"
#include <iostream>

int main() {
    using Actor = learning_to_fly::rnn::RNNActor<18, 32, 4>;

    Actor actor;

    // 设置测试权重，使 RNN 能产生非零记忆
    actor.W_x[0][0] = 1.0f;
    actor.W_h[0][0] = 0.5f;
    actor.W_out[0][0] = 1.0f;

    Actor::Input input{};
    input[0] = 1.0f;

    auto output1 = actor.forward(input);
    float hidden1 = actor.hidden[0];

    // 第二次输入全部为零
    input.fill(0.0f);

    auto output2 = actor.forward(input);
    float hidden2 = actor.hidden[0];

    std::cout << "Step 1 hidden: " << hidden1 << "\n";
    std::cout << "Step 1 output: " << output1[0] << "\n";

    std::cout << "Step 2 hidden: " << hidden2 << "\n";
    std::cout << "Step 2 output: " << output2[0] << "\n";

    actor.reset();

    std::cout << "After reset hidden: "
              << actor.hidden[0] << "\n";

    return 0;
}