from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from argparse import ArgumentParser, Namespace
import torch
from torchvision import datasets, transforms
from networks import make_resnet, make_classifier


# _______________________________________________________________________________________________ #

State = dict[str: dict[str: list]]


@dataclass
class Data:

    train_loader: torch.utils.data.DataLoader
    val_loader:   torch.utils.data.DataLoader


@dataclass
class Model:

    backbone:   torch.nn.Module
    classifier: torch.nn.Module
    optimizer:  torch.optim.Optimizer
    scheduler:  torch.optim.lr_scheduler._LRScheduler
    criterion:  torch.nn.modules.loss._Loss
    device:     torch.device


# _______________________________________________________________________________________________ #

def parse_args() -> Namespace:
    """ Required: backbone, dataset. """
    parser = ArgumentParser('Setup training environment')
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument('--first_features_maps', type=int, default=64)
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--rootdir", type=str, default="/data/Datasets/cifar100/")
    parser.add_argument("--output_dir", type=str, default="./logs/")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def epoch(model: Model, data: Data, pbar: tqdm, training: bool) -> tuple[float]:
    loader = data.train_loader if training else data.val_loader
    torch.set_grad_enabled(training)
    model.backbone.train() if training else model.backbone.eval()
    model.classifier.train() if training else model.classifier.eval()
    running_loss, running_acc = 0, 0
    pbar.reset()
    for X, y in loader:
        X, y = X.to(model.device), y.to(model.device)
        y_hat = model.classifier(model.backbone(X))
        loss = model.criterion(y_hat, y)
        acc = y_hat.argmax(dim=1).eq(y).sum()
        if training:
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            model.scheduler.step()
        loss, acc = loss.detach().item(), acc.detach().item()
        running_loss += loss
        running_acc += acc
        pbar.set_postfix(dict(loss=loss, acc=acc / len(y)))
        pbar.update()
    return running_loss / len(loader), running_acc / len(loader.dataset)


def fit(model: Model, data: Data, n_epochs: int = 100) -> State:
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
        state["lr"].append(model.optimizer.param_groups[0]['lr'])
        epoch_pbar.update()
    return state


def save(model: Model, state: State, output_dir: str, name: str) -> None:
    dir = Path(output_dir)
    dir.mkdir(exist_ok=True)
    path = dir / f"{name}.pt"
    total_state = dict(
        backbone=model.backbone.state_dict(),
        classifier=model.classifier.state_dict(),
        optimizer=model.optimizer.state_dict(),
        scheduler=model.scheduler.state_dict(),
        state=state,
    )
    torch.save(total_state, path)


# _______________________________________________________________________________________________ #

def make_data(dataset: str, rootdir: str, batch_size: int) -> Data:
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
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
    common_kwargs = dict(batch_size=batch_size, num_workers=4)
    train_loader = torch.utils.data.DataLoader(train_set, **common_kwargs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, **common_kwargs, shuffle=False)
    return Data(train_loader, val_loader)


def make_model(
    data: Data, dataset: str, name: str, first_feature_maps: int, large: bool, n_epochs: int
) -> Model:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, latent_dim = make_resnet(name, first_feature_maps, large)
    classifier = make_classifier(latent_dim, 100 if dataset == "cifar100" else 10)
    backbone, classifier = backbone.to(device), classifier.to(device)
    print(f"Device : {device}")
    print("Num params :")
    print(f" - Backbone : {sum(p.numel() for p in backbone.parameters() if p.requires_grad):,}")
    print(f" - Classifier : {sum(p.numel() for p in classifier.parameters() if p.requires_grad):,}")
    optimizer = torch.optim.AdamW(list(backbone.parameters()) + list(classifier.parameters()))
    length_params = dict(steps_per_epoch=len(data.train_loader), epochs=n_epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, **length_params)
    criterion = torch.nn.CrossEntropyLoss()
    return Model(backbone, classifier, optimizer, scheduler, criterion, device)


# _______________________________________________________________________________________________ #

def main(
    *,
    dataset: str, backbone: str, first_features_maps: int, large: bool,
    rootdir: str, batch_size: int, n_epochs: int, output_dir: str, name: str = None
) -> None:
    if name is None:
        name = f"{backbone}_{dataset}"
    data = make_data(dataset, rootdir, batch_size)
    model = make_model(data, dataset, backbone, first_features_maps, large, n_epochs)
    state = fit(model, data, n_epochs)
    save(model, state, output_dir, name)


if __name__ == "__main__":
    main(**vars(parse_args()))
