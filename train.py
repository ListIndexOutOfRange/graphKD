from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from argparse import ArgumentParser, Namespace
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from resnet import ResNet


# _______________________________________________________________________________________________ #

State = dict[str: dict[str: list]]


@dataclass
class Data:

    train_loader: torch.utils.data.DataLoader
    val_loader:   torch.utils.data.DataLoader


@dataclass
class Model:

    network:    torch.nn.Module
    optimizer:  torch.optim.Optimizer
    scheduler:  torch.optim.lr_scheduler._LRScheduler
    criterion:  torch.nn.modules.loss._Loss
    device:     torch.device


# _______________________________________________________________________________________________ #


def make_data(dataset: str, rootdir: str, batch_size: int) -> Data:
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    interpolation = transforms.InterpolationMode.BILINEAR
    policy = transforms.AutoAugmentPolicy.CIFAR10
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, interpolation=interpolation),
        transforms.RandomHorizontalFlip(0.5),
        transforms.AutoAugment(policy, interpolation=interpolation),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=mean, std=std),
    ])
    Dataset = datasets.CIFAR100 if dataset == "cifar100" else datasets.CIFAR10
    train_set = Dataset(rootdir, train=True, transform=train_transform)
    val_set = Dataset(rootdir, train=False, transform=val_transform)
    common_kwargs = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, **common_kwargs, shuffle=True)
    val_loader = DataLoader(val_set, **common_kwargs, shuffle=False)
    return Data(train_loader, val_loader)


def make_model(dataset: str, name: str) -> Model:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = ResNet.from_name(name, num_classes=100 if dataset == "cifar100" else 10).to(device)
    num_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Device : {device} | {num_params:,} params")
    optimizer = SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, patience=10)
    return Model(network, optimizer, scheduler, criterion, device)


# _______________________________________________________________________________________________ #

def get_last_lr(model: Model) -> float:
    return model.optimizer.param_groups[0]['lr']


def epoch(model: Model, data: Data, pbar: tqdm, training: bool) -> tuple[float]:
    loader = data.train_loader if training else data.val_loader
    torch.set_grad_enabled(training)
    model.network.train() if training else model.network.eval()
    running_loss, running_acc = 0, 0
    pbar.reset()
    for X, y in loader:
        X, y = X.to(model.device), y.to(model.device)
        y_hat = model.network(X)
        loss = model.criterion(y_hat, y)
        acc = y_hat.argmax(dim=1).eq(y).sum()
        if training:
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
        loss, acc = loss.detach().item(), acc.detach().item()
        running_loss += loss
        running_acc += acc
        pbar.set_postfix(dict(loss=loss, acc=acc / len(y), lr=get_last_lr(model)))
        pbar.update()
    return running_loss / len(loader), running_acc / len(loader.dataset)


def fit(model: Model, data: Data, n_epochs: int = 200) -> State:
    state = dict(train=dict(loss=list(), acc=list()), val=dict(loss=list(), acc=list()))
    state["lr"] = list()
    epoch_pbar = tqdm(total=n_epochs, desc="Global progress")
    train_pbar = tqdm(total=len(data.train_loader), desc="Training")
    val_pbar = tqdm(total=len(data.val_loader), desc="Validation")
    for _ in range(n_epochs):
        loss, acc = epoch(model, data, train_pbar, training=True)
        state["train"]["loss"].append(loss)
        state["train"]["acc"].append(acc)
        loss, acc = epoch(model, data, val_pbar, training=False)
        state["val"]["loss"].append(loss)
        state["val"]["acc"].append(acc)
        model.scheduler.step(loss)
        state["lr"].append(get_last_lr(model))
        epoch_pbar.update()
    return state


def save(model: Model, state: State, output_dir: str, name: str) -> None:
    dir = Path(output_dir)
    dir.mkdir(exist_ok=True)
    path = dir / f"{name}.pt"
    total_state = dict(
        network=model.network.state_dict(),
        optimizer=model.optimizer.state_dict(),
        scheduler=model.scheduler.state_dict(),
        state=state,
    )
    torch.save(total_state, path)


# _______________________________________________________________________________________________ #

def parse_args() -> Namespace:
    """ Required: network. """
    parser = ArgumentParser('Setup training environment')
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--rootdir", type=str, default="/data/Datasets/cifar100/")
    parser.add_argument("--output_dir", type=str, default="./logs/")
    parser.add_argument("--batch_size", type=int, default=512)
    return parser.parse_args()


def main(
    *,
    dataset: str, rootdir: str, network: str, batch_size: int,
    output_dir: str, name: str = None
) -> None:
    if name is None:
        name = f"{network}_{dataset}"
    data = make_data(dataset, rootdir, batch_size)
    model = make_model(dataset, network)
    state = fit(model, data)
    save(model, state, output_dir, name)


if __name__ == "__main__":
    main(**vars(parse_args()))
