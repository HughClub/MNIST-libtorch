#pragma once
#include <torch/torch.h>

struct Net : torch::nn::Module {
  Net(
    int hidden_size = 64,
    int width = 28,
    int height = 28,
    int channels = 1,
    int num_classes = 10
  ) {
    // register_module is required
    this->fc1 = register_module("infeat", torch::nn::Linear(width * height * channels, hidden_size));
    // The following two methods are both not feasible
    // this->fc1 = torch::nn::Linear(width * height * channels, hidden_size);
    // this->fc1 = torch::nn::Linear(width * height * channels, hidden_size).ptr();
    this->fc2 = register_module("hidden", torch::nn::Linear(hidden_size, hidden_size/2));
    this->fc3 = register_module("outfeat", torch::nn::Linear(hidden_size/2, num_classes));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1(x.view({x.size(0), -1})));
    x = torch::dropout(x, 0.5, is_training());
    x = torch::relu(fc2(x));
    x = torch::log_softmax(fc3(x), 1);
    return x;
  }

  bool require_grad_(bool mode=true) {
    for (auto & layer : this->parameters()) {
      layer.requires_grad_(mode);
    }
    return mode;
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};
