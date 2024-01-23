#pragma once
#include <fstream>
#include <iostream>
#include <sstream>

#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/datasets/mnist.h>

using namespace torch::data;


struct DatasetStats {
  double mean, std;
};

auto MNIST_FILES = {
  "t10k-images-idx3-ubyte",
  "t10k-labels-idx1-ubyte",
  "train-images-idx3-ubyte",
  "train-labels-idx1-ubyte"
};

struct Arguments {
  int batch_size=128, epochs=20;
  // int steps=10000; // 10K steps
  double lr=1e-4;
  int hidden_size = 64;
  int log_interval = -1; // -1 to disable
  int train_worker = 4, test_worker = 2;
};

template <typename Target = torch::Tensor>
struct Compose : public transforms::TensorTransform<Target> {
  Compose(const std::initializer_list<transforms::TensorTransform<Target>>& transforms) {
    this->transforms = transforms;
  }
  torch::Tensor operator()(torch::Tensor input) override {
    for (const auto& transform : transforms) {
      input = transform(input);
    }
    return input;
  }
  const std::initializer_list<transforms::TensorTransform<Target>>& transforms;
};

DataLoaderOptions make_options(int batch_size, bool shuffle=false, bool drop_last=true, int workers=1) {
  DataLoaderOptions options = DataLoaderOptions(batch_size);
  options.enforce_ordering(!shuffle);
  options.drop_last(drop_last);
  options.workers(workers);
  return std::move(options);
}

bool exists(const std::string&filename){
  std::ifstream ifs(filename);
  bool ifs_good = ifs.good();
  ifs.close();
  return ifs_good;
}


bool MNIST_checker(const std::string& data_root) {
  const char* about = "Please run \033[31;1mdownload_mnist.sh\033[0m to download it or get it from http://yann.lecun.com/exdb/mnist/";
  bool any_error = false;
  for (auto& set : MNIST_FILES) {
    std::string file = data_root + "/" + set;
    if (!exists(file)) {
      std::cerr << "File :" << file  << " does not exist." << std::endl;
      any_error = true;
    }
  }
  if (any_error) std::cout << about << std::endl;
  return !any_error;
}


DatasetStats 
calc_mean_std(const datasets::MNIST& dataset) {
  const torch::Tensor & images = dataset.images();
  float mean = images.mean().item<float>();
  float std = images.std().item<float>();
  return DatasetStats{mean, std};
}

DatasetStats 
calc_mean_std(const std::initializer_list<datasets::MNIST>& datasets) {
  /**
   * memory unfriendly implementation, but it's simple and easy
  */
  std::vector<torch::Tensor> imageset = {};
  for (const auto& dataset : datasets) {
    imageset.emplace_back(dataset.images());
  }
  torch::Tensor images = torch::cat(imageset, 0);
  float mean = images.mean().item<float>();
  float std = images.std().item<float>();
  return DatasetStats{mean, std};
}

std::string syth_model_name(double acc, std::string const& prefix, Arguments const& args) {
  std::stringstream ss;
  ss << prefix << "_acc" << acc
    << "_bs" << args.batch_size
    << "_lr" << args.lr
    << "_hs" << args.hidden_size
    << "_eps" << args.epochs << ".pt";
  return ss.str();
}