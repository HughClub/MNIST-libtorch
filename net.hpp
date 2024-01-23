#pragma once
#include<vector>
#include <torch/torch.h>

struct MLP : torch::nn::Module {
  MLP(
    int hidden_size = 64,
    int width = 28,
    int height = 28,
    int channels = 1,
    int num_classes = 10
  ) {
    // register_module is required
    auto fc1 = register_module("infeat", torch::nn::Linear(width * height * channels, hidden_size));
    // The following two methods are both not feasible
    // this->fc1 = torch::nn::Linear(width * height * channels, hidden_size);
    // this->fc1 = torch::nn::Linear(width * height * channels, hidden_size).ptr();
    auto fc2 = register_module("hidden", torch::nn::Linear(hidden_size, hidden_size>>1));
    auto fc3 = register_module("outfeat", torch::nn::Linear(hidden_size>>1, num_classes));
    this->layers = {fc1, fc2, fc3};

  }

  torch::Tensor forward(torch::Tensor x) {
    x = x.view({x.size(0), -1});
    for (int i = 0; i < layers.size()-1; i++) {
      x = torch::relu(layers[i](x));
      if (i != layers.size()-2) {
        x = torch::dropout(x, 0.5, is_training());
      } else {
        x = torch::log_softmax(x, 1);
      }
    }
    return x;
  }

  bool require_grad_(bool mode=true) {
    for (auto & layer : this->parameters()) {
      layer.requires_grad_(mode);
    }
    return mode;
  }

  // torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
  std::vector<torch::nn::Linear> layers;
};

int dim_clac(int dim, int kernel_size, int pad=0, int stride=1, int pool_size=2, int times=1) {
  for (int i=0; i<times; i++) {
    dim = (dim + 2*pad - kernel_size) / stride + 1;
    dim /= pool_size;
  }
  return dim;
}

struct Conv : torch::nn::Module {
    Conv(
      int hidden_size = 64,
      int width = 28,
      int height = 28,
      int kernel_size = 5,
      int hidden_channels = 5,
      int num_classes = 10
    ) {
      conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, hidden_channels, kernel_size)));
      conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden_channels, hidden_channels<<1, kernel_size)));
      pool = register_module("pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
      int last_width = dim_clac(width, kernel_size, 0, 1, 2, 2);
      int last_height = dim_clac(height, kernel_size, 0, 1, 2, 2);
      act_size = hidden_channels*2*last_width*last_height;
      fc1 = register_module("fc1", torch::nn::Linear(act_size, hidden_size));
      fc2 = register_module("fc2", torch::nn::Linear(hidden_size, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
      x = pool->forward(torch::relu(conv1->forward(x)));
      x = pool->forward(torch::relu(conv2->forward(x)));
      x = x.view({-1, act_size});
      x = torch::relu(fc1->forward(x));
      x = fc2->forward(x);
      return torch::log_softmax(x, 1);
    }

    bool require_grad_(bool mode=true) {
      for (auto & layer : this->parameters()) {
        layer.requires_grad_(mode);
      }
      return mode;
    }


    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::MaxPool2d pool{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    int act_size; // after conv & pool
};