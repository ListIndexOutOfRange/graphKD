from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from collections import OrderedDict
from argparse import ArgumentParser, Namespace
import torch
from torch.utils.data import default_collate, DataLoader
from torchvision import datasets, transforms
from mix import RandomMixup, RandomCutmix
from efficientnet import EfficientNet, efficientnet_params


# _______________________________________________________________________________________________ #

State = dict[str: dict[str: list]]


@dataclass
class Data:

    train_loader: torch.utils.data.DataLoader
    val_loader:   torch.utils.data.DataLoader


@dataclass
class Model:

    network:   torch.nn.Module
    optimizer:  torch.optim.Optimizer
    scheduler:  torch.optim.lr_scheduler._LRScheduler
    criterion:  torch.nn.modules.loss._Loss
    device:     torch.device


# _______________________________________________________________________________________________ #

def make_data(
    dataset: str, rootdir: str, network_name: str,
    batch_size: int, mixup_alpha: float, cutmix_alpha: float,
) -> Data:
    _, _, crop_size, _ = efficientnet_params(network_name)
    resize_size = round(crop_size * 1.146)
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    interpolation = transforms.InterpolationMode.BILINEAR
    policy = transforms.AutoAugmentPolicy.CIFAR10
    # train_transform = transforms.Compose([
    #     transforms.Resize(256, interpolation=interpolation, antialias=True),
    #     transforms.RandomCrop(224, padding=4),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15),
    #     transforms.PILToTensor(),
    #     transforms.ConvertImageDtype(torch.float),
    #     transforms.Normalize(mean=mean, std=std),
    # ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, interpolation=interpolation),
        transforms.RandomHorizontalFlip(0.5),
        transforms.AutoAugment(policy, interpolation=interpolation),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=mean, std=std),
    ])
    collate_fn = None
    num_classes = 100 if dataset == "cifar100" else 10
    mixup_transforms = []
    if mixup_alpha > 0.0:
        mixup_transforms.append(RandomMixup(num_classes, p=1.0, alpha=mixup_alpha))
    if cutmix_alpha > 0.0:
        mixup_transforms.append(RandomCutmix(num_classes, p=1.0, alpha=cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = transforms.RandomChoice(mixup_transforms)
        def collate_fn(batch):  # noqa: E301
            return mixupcutmix(*default_collate(batch))
    Dataset = datasets.CIFAR100 if dataset == "cifar100" else datasets.CIFAR10
    train_set = Dataset(rootdir, train=True, transform=train_transform)
    val_set = Dataset(rootdir, train=False, transform=val_transform)
    common_kwargs = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, **common_kwargs, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, **common_kwargs, shuffle=False)
    return Data(train_loader, val_loader)


def add_swa(model: Model, epochs: int, swa_lr: float = 0.05) -> Model:
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
        0.1 * averaged_model_parameter + 0.9 * model_parameter
    ema_model = torch.optim.swa_utils.AveragedModel(model.network, avg_fn=ema_avg)
    model.swa_network   = ema_model
    model.swa_scheduler = torch.optim.swa_utils.SWALR(model.optimizer, swa_lr=swa_lr)
    model.swa_start     = epochs // 2
    return model


def make_model(
    data: Data, dataset: str, name: str,
    label_smoothing: float, swa: bool, n_epochs: int, max_lr: float
) -> Model:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 100 if dataset == "cifar100" else 10
    network = EfficientNet.from_name(name, num_classes=num_classes).to(device)
    print(f"Num params : {network.num_params:,} | Device : {device}")
    optimizer = torch.optim.AdamW(network.parameters())
    length_params = dict(steps_per_epoch=len(data.train_loader), epochs=n_epochs)
    cycle_params = dict(pct_start=0.1, max_lr=max_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **cycle_params, **length_params)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    model = Model(network, optimizer, scheduler, criterion, device)
    if swa:
        model = add_swa(model, n_epochs)
    return model

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
        if y.ndim == 2:  # mixup or cutmix
            y = y.max(dim=1).values
        acc = y_hat.argmax(dim=1).eq(y).sum()
        if training:
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            model.scheduler.step()
        loss, acc = loss.detach().item(), acc.detach().item()
        running_loss += loss
        running_acc += acc
        pbar.set_postfix(dict(loss=loss, acc=acc / len(y), lr=get_last_lr(model)))
        pbar.update()
    return running_loss / len(loader), running_acc / len(loader.dataset)


def complete_swa(loader: DataLoader, model: Model, dir: str, name: str) -> None:
    dir = Path(dir)
    dir.mkdir(exist_ok=True)
    path = dir / f"{name}_swa.pt"
    torch.optim.swa_utils.update_bn(loader, model.swa_network, model.device)
    # Make SWA modules names equal to original network ones.
    swa_state = model.swa_network.state_dict()
    #   - delete SWA specific key
    del swa_state['n_averaged']
    #   - remove 'module.' prefix
    modules = [('.'.join(k.split('.')[1:]), v) for k, v in swa_state.items()]
    #   - wrap results
    new_swa_state = OrderedDict(modules)
    # Save
    state = dict(network=new_swa_state)
    torch.save(state, path)


def fit(model: Model, data: Data, n_epochs: int = 100) -> State:
    state = dict(train=dict(loss=list(), acc=list()), val=dict(loss=list(), acc=list()))
    state["lr"] = list()
    epoch_pbar = tqdm(total=n_epochs, desc="Global progress")
    train_pbar = tqdm(total=len(data.train_loader), desc="Training")
    val_pbar = tqdm(total=len(data.val_loader), desc="Validation")
    for i in range(n_epochs):
        loss, acc = epoch(model, data, train_pbar, training=True)
        state["train"]["loss"].append(loss)
        state["train"]["acc"].append(acc)
        loss, acc = epoch(model, data, val_pbar, training=False)
        state["val"]["loss"].append(loss)
        state["val"]["acc"].append(acc)
        state["lr"].append(get_last_lr(model))
        epoch_pbar.update()
        if hasattr(model, "swa_start") and i > model.swa_start:
            model.swa_network.update_parameters(model.network)
            model.swa_scheduler.step()
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
    parser = ArgumentParser('Setup training environment')
    parser.add_argument("--network", type=str, default="efficientnet-b0")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--rootdir", type=str, default="/data/Datasets/cifar100/")
    parser.add_argument("--output_dir", type=str, default="./logs/")
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.)
    parser.add_argument("--mixup_alpha", type=float, default=0.)
    parser.add_argument("--cutmix_alpha", type=float, default=0.)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def main(
    *,
    network: str, dataset: str, rootdir: str, output_dir: str, label_smoothing: float, swa: bool,
    mixup_alpha: float, cutmix_alpha: float, batch_size: int, max_lr: float, n_epochs: int,
    name: str = None
) -> None:
    if name is None:
        name = f"{network}_{dataset}"
    data = make_data(dataset, rootdir, network, batch_size, mixup_alpha, cutmix_alpha)
    model = make_model(data, dataset, network, label_smoothing, swa, n_epochs, max_lr)
    state = fit(model, data, n_epochs)
    save(model, state, output_dir, name)
    if swa:
        complete_swa(data.train_loader, model, output_dir, name)


if __name__ == "__main__":
    main(**vars(parse_args()))
