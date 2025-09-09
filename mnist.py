import argparse
import time
import functools

import torch as th
import torch.nn as tn
from torch.functional import F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose


def timer_as_return(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> float:
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        return end - start

    return wrapper


# define the model
class Net(tn.Module):
    def __init__(self, hidden_size=64):
        super(Net, self).__init__()
        input_size = 28 * 28 * 1
        self.layers = th.nn.Sequential(
            th.nn.Linear(input_size, hidden_size),
            th.nn.ReLU(),
            th.nn.Dropout(),
            th.nn.Linear(hidden_size, hidden_size // 2),
            th.nn.ReLU(),
            th.nn.Dropout(),
            th.nn.Linear(hidden_size // 2, 10),
        )

    def forward(self, x):
        x = self.layers(x.view(x.shape[0], -1))
        return F.log_softmax(x, dim=-1)


class ConvNet(tn.Module):
    def __init__(
        self,
    ):
        super(ConvNet, self).__init__()
        self.layers = th.nn.Sequential(
            th.nn.Conv2d(1, 32, 3, padding=1),  # (N, 1, 28, 28) -> (N, 32, 28, 28)
            th.nn.ReLU(),
            th.nn.AvgPool2d(2),  # (N, 4, 28, 28) -> (N, 4, 14, 14)
            th.nn.Conv2d(32, 64, 3, padding=1),  # (N, 32, 14, 14) -> (N, 64, 14, 14)
            th.nn.ReLU(),
            th.nn.AvgPool2d(2),  # (N, 64, 14, 14) -> (N, 64, 7, 7)
            th.nn.Conv2d(64, 128, 4),  # (N, 64, 7, 7) -> (N, 128, 4, 4)
            th.nn.ReLU(),
            th.nn.AvgPool2d(2),  # (N, 128, 4, 4) -> (N, 128, 2, 2)
            th.nn.Flatten(),
            th.nn.Linear(128 * 2 * 2, 100),
            th.nn.ReLU(),
            th.nn.Linear(100, 10),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, th.nn.Conv2d):
                th.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, th.nn.Linear):
                th.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )

    def forward(self, x):
        x = self.layers(x)
        return F.log_softmax(x)


def get_model(lr=0.01):
    model = Net()
    return model, th.optim.SGD(model.parameters(), lr=lr)


def get_loader(train: bool = True, batch_size: int = 64):
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    data_root = "./data"
    dataset = MNIST(data_root, transform=transform, train=train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=True)
    return loader

@timer_as_return
def train_epoch(model, optim, train_loader, epoch, device):
    model.train()
    for i, (data, label) in enumerate(train_loader):
        logits = model(data.to(device))
        label = label.to(device)
        loss = th.nn.functional.nll_loss(logits, label)
        loss.backward()
        optim.step()
        optim.zero_grad()
        # if i % 100 == 0:
        #     acc = th.mean((logits.argmax(dim=1) == label).float()).item()
        #     print(f"[{epoch}/{i}] loss: {loss.item():.4f}, acc: {acc:.4f}")


@timer_as_return
@th.inference_mode()
def test_epoch(model, test_loader, epoch, device):
    model.eval()
    correct, losses = 0, 0
    datasize = len(test_loader.dataset)
    for i, (data, label) in enumerate(test_loader):
        logits = model(data.to(device))
        label = label.to(device)
        loss = th.nn.functional.nll_loss(logits, label, reduction="sum").item()
        losses += loss
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
    epoch_loss = losses / datasize
    epoch_acc = correct / datasize
    print(f"[{epoch}] loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")
    return epoch_acc


def train(bs:int, lr:float, hs:int, epochs: int, use_gpu: bool):
    device = th.device("cuda" if th.cuda.is_available() and use_gpu else "cpu")
    # model, optim = get_model()
    model = Net(hs).to(device)
    optim = th.optim.SGD(model.parameters(), lr=lr)
    train_loader = get_loader(train=True, batch_size=bs)
    test_loader = get_loader(train=False, batch_size=bs)
    train_delta, test_delta = 0, 0
    for e in range(epochs):
        train_delta += train_epoch(model, optim, train_loader, e, device)
        test_delta += test_epoch(model, test_loader, e, device)
    print(f"train time: {train_delta:.4f}s, test time: {test_delta:.4f}s")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hs", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args.bs, args.lr, args.hs, args.epochs, args.cuda)
