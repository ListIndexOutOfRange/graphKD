# Graph Knowledge Distillation


## Baseline fully supervised

- **student**: efficientnet-b0 (4,135,648 params). Pour reproduire:
`python train.py --network efficientnet-b0 --dataset cifar100 --label_smoothing 0.1 --max_lr 0.001 --n_epochs 100`
- **student**: efficientnet-b3 (10,849,932 params). Pour reproduire:
`python train.py --network efficientnet-b3 --dataset cifar100 --label_smoothing 0.1 --max_lr 0.001 --n_epochs 100`


## train.py args

- `--network`: type=str, default="efficientnet-b0"
- `--dataset`: type=str, default="cifar100"
- `--rootdir`: type=str, default="/data/Datasets/cifar100/"
- `--output_dir`: type=str, default="./logs/"
- `--swa`: action="store_true"
- `--label_smoothing`: type=float, default=0.
- `--mixup_alpha`: type=float, default=0.
- `--cutmix_alpha`: type=float, default=0.
- `--batch_size`: type=int, default=256
- `--max_lr`: type=float, default=0.001
- `--n_epochs`: type=int, default=100
- `--name`: type=str, default=None


## Distance

- pas de problème pour calculer des cosine similarities intra layer
- pas de problème pour utiliser POT mais pas facile d'avoir un truc différentiable; c.f:
    - [Optimizing the Gromov-Wasserstein distance with PyTorch](https://pythonot.github.io/master/auto_examples/backends/plot_optim_gromov_pytorch.html)
    - [Gromov and Fused-Gromov-Wasserstein](https://pythonot.github.io/master/auto_examples/index.html#gromov-and-fused-gromov-wasserstein)