#include <torch/torch.h>
#include <torch/utils.h>
#include <torch/data/dataloader.h>
#include <torch/data/datasets/mnist.h>

#include "net.hpp"
#include "utils.hpp"

using namespace torch::data::datasets;

int main(int argc, char**argv) {
  Arguments arguments = Arguments();
  #pragma region "poor man's cmd parser"
  if (argc != 1) {
    // batch_size, lr, hidden_size, epochs
    if (argc < 3) {
      arguments.batch_size = strtol(argv[1], nullptr, 10);
    }
    if (argc < 4) {
      arguments.lr = strtod(argv[2], nullptr);
    }
    if (argc < 5) {
      arguments.hidden_size = strtol(argv[3], nullptr, 10);
    }
    if (argc < 6) {
      arguments.epochs = strtol(argv[4], nullptr, 10);
    }
  }
  #pragma endregion
  // 0. get MNIST dataset [train, test]
  const char* data_root = "./data"; 
  if (!MNIST_checker(data_root)) { exit(1); }
  auto train_set = MNIST(data_root, MNIST::Mode::kTrain);
  auto test_set = MNIST(data_root, MNIST::Mode::kTest);
  size_t train_set_size = train_set.size().value_or(0);
  size_t test_set_size = test_set.size().value_or(0);
  // 1. calculate MNIST mean and std
  // auto stats = calc_mean_std(train_set, test_set);
  // std::cout << "mean: " << stats.mean << "\tstd: " << stats.std << std::endl;
  // 2. create dataloader
  // double mean = 0.1309, std = 0.3084; // calculation result
  double mean = 0.1307, std = 0.3081; // calculation result
  auto train_loader = torch::data::make_data_loader(
    train_set.map(torch::data::transforms::Normalize<>(mean, std))
      .map(torch::data::transforms::Stack<>()),
    make_options(arguments.batch_size, true, true, arguments.train_worker)
  );
  // torch::data::transforms::Normalize<>(stats.mean, stats.std);
  auto test_loader = torch::data::make_data_loader(
    test_set.map(torch::data::transforms::Normalize<>(mean, std))
      .map(torch::data::transforms::Stack<>()),
    make_options(arguments.batch_size, false, true, arguments.test_worker)
  );

  // 3. get network and optimizer
  auto model = std::make_shared<Net>(arguments.hidden_size);
  auto optim = torch::optim::SGD(model->parameters(), arguments.lr);

  double last_acc = 0;
  // 4. train and test
  for (size_t epoch = 0; epoch < arguments.epochs; ++epoch) {
    // train model
    model->train();
    size_t batch_idx = 0;
    for (const auto& batch : *train_loader) {
      torch::Tensor logits = model->forward(batch.data);
      torch::Tensor loss = torch::nll_loss(logits, batch.target);
      loss.backward();
      optim.step();
      optim.zero_grad();
      if (batch_idx % arguments.log_interval == 0) {
        std::cout << "Train Epoch: " << epoch << " [" << batch_idx * arguments.batch_size << "/" <<
          train_set_size << " (" << 100. * batch_idx * arguments.batch_size / train_set_size << "%)]\t"
          << "Loss: " << loss.item<float>() << std::endl;
      }
      batch_idx++;
    }
    // test
    double test_loss = 0;
    size_t correct = 0;
    model->eval();
    for (const auto& batch : *test_loader) {
      torch::Tensor logits = model->forward(batch.data);
      test_loss += torch::nll_loss(logits, batch.target).item<double>();
      auto pred = torch::argmax(logits, 1);
      correct += torch::sum(pred == batch.target).item<int64_t>();
    }
    std::cout << "Test set: Average loss: " << test_loss / test_set_size << ", Accuracy: " <<
      correct*100. / test_set_size<< "%" << std::endl;
    last_acc = correct*100. / test_set_size;
  }
  // 5. save the model
  std::string model_saving = "MNIST_MLP_"+std::to_string(last_acc)+".pt";
  torch::save(model, model_saving);
  return 0;
}
